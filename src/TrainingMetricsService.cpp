#include "TrainingMetricsService.hpp"
#include "Logger.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <thread>

// HTTP client for pushing metrics to external API daemon (optional)
#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

namespace fs = std::filesystem;

// ============================================================================
// TrainingMetricsService Implementation
// ============================================================================

TrainingMetricsService::TrainingMetricsService(const MetricsServiceConfig& config)
    : config_(config),
      is_training_(false),
      current_session_id_(0),
      samples_since_last_persist_(0) {
    
    // Ensure metrics directory exists
    if (config_.enable_persistence) {
        fs::path metrics_path(config_.metrics_file);
        if (metrics_path.has_parent_path()) {
            fs::create_directories(metrics_path.parent_path());
        }
    }
    
    adai::Logger::info("TrainingMetricsService initialized");
}

TrainingMetricsService::~TrainingMetricsService() {
    if (is_training_) {
        end_session();
    }
    flush_to_disk();
    adai::Logger::info("TrainingMetricsService shutdown");
}

void TrainingMetricsService::start_session(int session_id, int total_epochs, int total_samples) {
    std::string push_json;
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        current_session_id_ = session_id;
        is_training_ = true;

        current_snapshot_ = TrainingMetricsSnapshot();
        current_snapshot_.session_id = session_id;
        current_snapshot_.is_training = true;
        current_snapshot_.total_epochs = total_epochs;
        current_snapshot_.total_samples = total_samples;
        current_snapshot_.session_start_time = std::chrono::system_clock::now();
        current_snapshot_.last_update_time = current_snapshot_.session_start_time;

        session_start_steady_ = std::chrono::steady_clock::now();
        samples_since_last_persist_ = 0;
        last_persist_time_ = std::chrono::system_clock::now();

        adai::Logger::info("Metrics session {} started (epochs={}, samples={})",
                           session_id, total_epochs, total_samples);
        adai::Logger::info("Metrics push config: enable_push={}, push_url={}",
                           config_.enable_push, config_.push_url);

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"session_id\":" << session_id
                 << ",\"total_epochs\":" << total_epochs
                 << ",\"total_samples\":" << total_samples << "}";
            push_json = json.str();
        }
    }  // mutex released here — HTTP call runs WITHOUT holding the lock
    if (should_push) {
        push_to_api("/api/session/start", push_json);
    }
}

void TrainingMetricsService::end_session() {
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        is_training_ = false;
        current_snapshot_.is_training = false;

        // Calculate total training time
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - session_start_steady_);
        current_snapshot_.total_training_time_seconds = duration.count() / 1000.0;

        // Persist final state
        persist_metrics();
        persist_summary();
        if (config_.enable_prometheus_format) {
            persist_prometheus();
        }

        adai::Logger::info("Metrics session {} ended (total_time={:.2f}s, samples={})",
                           current_session_id_.load(),
                           current_snapshot_.total_training_time_seconds,
                           current_snapshot_.total_samples_trained);

        should_push = config_.enable_push;
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/session/end", "{}");
    }
}

bool TrainingMetricsService::is_session_active() const {
    return is_training_;
}

void TrainingMetricsService::start_epoch(int epoch, int total_samples) {
    std::string push_json;
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        current_snapshot_.current_epoch = epoch;
        current_snapshot_.current_sample = 0;
        if (total_samples > 0) {
            current_snapshot_.total_samples = total_samples;
        }

        epoch_start_steady_ = std::chrono::steady_clock::now();

        adai::Logger::debug("Metrics epoch {} started (samples={})", epoch, total_samples);

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"epoch\":" << epoch
                 << ",\"total_samples\":" << total_samples << "}";
            push_json = json.str();
        }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/epoch/start", push_json);
    }
}

void TrainingMetricsService::end_epoch(int epoch, float loss, float validation_loss,
                                       float learning_rate, float perplexity, float gradient_norm) {
    std::string push_json;
    bool should_push = false;
    float stored_perplexity = 0.0f;
    {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Calculate epoch duration
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - epoch_start_steady_);
    double epoch_time = duration.count() / 1000.0;
    
    // Update epoch history
    current_snapshot_.epoch_losses.push_back(loss);
    current_snapshot_.epoch_validation_losses.push_back(validation_loss);
    current_snapshot_.epoch_learning_rates.push_back(learning_rate);
    current_snapshot_.epoch_perplexities.push_back(perplexity > 0 ? perplexity : std::exp(loss));
    // TD-015: persist per-epoch validation perplexity and accuracy
    current_snapshot_.epoch_validation_perplexities.push_back(
        current_snapshot_.current_validation_perplexity > 0.0f
            ? current_snapshot_.current_validation_perplexity
            : (validation_loss > 0.0f ? std::exp(validation_loss) : 0.0f));
    current_snapshot_.epoch_validation_accuracies.push_back(
        current_snapshot_.current_validation_accuracy);
    // Persist per-epoch BLEU/ROUGE scores (use the values already stored by
    // update_generation_quality_metrics(); default -1 if not computed this epoch)
    current_snapshot_.epoch_bleu4.push_back(current_snapshot_.current_bleu4);
    current_snapshot_.epoch_rouge1.push_back(current_snapshot_.current_rouge1);
    current_snapshot_.epoch_rouge2.push_back(current_snapshot_.current_rouge2);
    current_snapshot_.epoch_rougeL.push_back(current_snapshot_.current_rougeL);
    current_snapshot_.epoch_durations.push_back(epoch_time);
    current_snapshot_.epoch_gradient_norms.push_back(gradient_norm);
    
    // Update current metrics
    current_snapshot_.current_loss = loss;
    current_snapshot_.current_validation_loss = validation_loss;
    current_snapshot_.current_learning_rate = learning_rate;
    current_snapshot_.current_gradient_norm = gradient_norm;
    current_snapshot_.current_perplexity = perplexity > 0 ? perplexity : std::exp(loss);
    current_snapshot_.last_update_time = std::chrono::system_clock::now();
    
    // Add persistent record
    PersistentMetricsRecord record;
    record.timestamp = current_snapshot_.last_update_time;
    record.session_id = current_session_id_;
    record.epoch = epoch;
    record.sample = current_snapshot_.total_samples;
    record.loss = loss;
    record.validation_loss = validation_loss;
    record.learning_rate = learning_rate;
    record.gradient_norm = gradient_norm;
    record.perplexity = current_snapshot_.current_perplexity;
    add_record(record);
    
    // Persist if needed
    auto time_since_persist = std::chrono::duration_cast<std::chrono::seconds>(
        current_snapshot_.last_update_time - last_persist_time_);
    if (time_since_persist.count() >= config_.persist_every_seconds) {
        persist_metrics();
        persist_summary();
        if (config_.enable_prometheus_format) {
            persist_prometheus();
        }
        last_persist_time_ = current_snapshot_.last_update_time;
    }
    
    adai::Logger::debug("Metrics epoch {} completed (loss={:.4f}, val_loss={:.4f}, time={:.2f}s)",
                        epoch, loss, validation_loss, epoch_time);

        stored_perplexity = current_snapshot_.current_perplexity;
        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"epoch\":" << epoch
                 << ",\"loss\":" << loss
                 << ",\"validation_loss\":" << validation_loss
                 << ",\"learning_rate\":" << learning_rate
                 << ",\"perplexity\":" << stored_perplexity
                 << ",\"gradient_norm\":" << gradient_norm
                 << ",\"gradient_variance\":" << current_snapshot_.gradient_variance
                 << ",\"compute_time_ratio\":" << current_snapshot_.compute_time_ratio
                 << ",\"weight_update_ratio\":" << current_snapshot_.weight_update_ratio
                 << "}";
            push_json = json.str();
        }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/epoch/end", push_json);
    }
}

void TrainingMetricsService::update_sample_metrics(int sample, float loss,
                                                   float gradient_norm, float learning_rate) {
    std::string push_json;
    bool should_push = false;
    {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_snapshot_.current_sample = sample;
    current_snapshot_.current_loss = loss;
    current_snapshot_.current_gradient_norm = gradient_norm;
    current_snapshot_.current_learning_rate = learning_rate;
    current_snapshot_.current_perplexity = std::exp(loss);
    current_snapshot_.total_samples_trained++;
    current_snapshot_.last_update_time = std::chrono::system_clock::now();
    
    // Update running average
    if (sample == 1) {
        current_snapshot_.running_loss = loss;
    } else {
        float alpha = 0.1f;  // Exponential moving average factor
        current_snapshot_.running_loss = alpha * loss + (1.0f - alpha) * current_snapshot_.running_loss;
    }
    
    samples_since_last_persist_++;
    
    // Update throughput metrics
    update_throughput_metrics();
    
    // Build push payload while lock is held
        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"sample\":" << sample
                 << ",\"loss\":" << loss
                 << ",\"gradient_norm\":" << gradient_norm
                 << ",\"learning_rate\":" << learning_rate << "}";
            push_json = json.str();
        }

    // Persist if needed
    if (samples_since_last_persist_ >= config_.persist_every_samples) {
        PersistentMetricsRecord record;
        record.timestamp = current_snapshot_.last_update_time;
        record.session_id = current_session_id_;
        record.epoch = current_snapshot_.current_epoch;
        record.sample = sample;
        record.loss = loss;
        record.validation_loss = current_snapshot_.current_validation_loss;
        record.learning_rate = learning_rate;
        record.gradient_norm = gradient_norm;
        record.perplexity = current_snapshot_.current_perplexity;
        add_record(record);
        
        persist_metrics();
        samples_since_last_persist_ = 0;
        last_persist_time_ = current_snapshot_.last_update_time;
    }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/metrics/sample", push_json);
    }
}

void TrainingMetricsService::update_validation_metrics(float validation_loss,
                                                       float validation_accuracy,
                                                       float validation_perplexity) {
    std::string push_json;
    bool should_push = false;
    {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_snapshot_.current_validation_loss = validation_loss;
    current_snapshot_.running_validation_loss = validation_loss;
    // Derive perplexity if not supplied (TD-015)
    current_snapshot_.current_validation_perplexity =
        (validation_perplexity > 0.0f) ? validation_perplexity
        : (validation_loss > 0.0f ? std::exp(validation_loss) : 0.0f);
    current_snapshot_.current_validation_accuracy = validation_accuracy;
    current_snapshot_.last_update_time = std::chrono::system_clock::now();

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"validation_loss\":" << validation_loss
                 << ",\"validation_accuracy\":" << validation_accuracy
                 << ",\"validation_perplexity\":"
                 << current_snapshot_.current_validation_perplexity << "}";
            push_json = json.str();
        }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/metrics/validation", push_json);
    }
}

void TrainingMetricsService::update_best_metrics(float validation_loss, int epoch) {
    std::string push_json;
    bool should_push = false;
    {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (validation_loss < current_snapshot_.best_validation_loss) {
        current_snapshot_.best_validation_loss = validation_loss;
        current_snapshot_.best_epoch = epoch;
        adai::Logger::info("New best validation loss: {:.4f} (epoch {})", validation_loss, epoch);

            should_push = config_.enable_push;
            if (should_push) {
                std::ostringstream json;
                json << "{\"validation_loss\":" << validation_loss
                     << ",\"epoch\":" << epoch << "}";
                push_json = json.str();
            }
    }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/metrics/best", push_json);
    }
}

TrainingMetricsSnapshot TrainingMetricsService::get_current_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create a copy and ensure atomic fields are synchronized
    TrainingMetricsSnapshot snapshot = current_snapshot_;
    snapshot.is_training = is_training_.load();
    snapshot.session_id = current_session_id_.load();
    
    // Update time-dependent metrics for current data
    if (is_training_.load()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - session_start_steady_);
        
        if (elapsed.count() > 0) {
            double elapsed_seconds = elapsed.count() / 1000.0;
            snapshot.total_training_time_seconds = elapsed_seconds;
            snapshot.samples_per_second =
                snapshot.total_samples_trained / elapsed_seconds;
            
            // Estimate time remaining
            if (snapshot.total_samples > 0 && snapshot.samples_per_second > 0) {
                int remaining_samples =
                    snapshot.total_samples * snapshot.total_epochs -
                    snapshot.total_samples_trained;
                snapshot.estimated_time_remaining_seconds =
                    remaining_samples / snapshot.samples_per_second;
            }
        }
        
        // Update last_update_time to current time for accurate timestamp
        snapshot.last_update_time = std::chrono::system_clock::now();
    }
    
    return snapshot;
}

std::string TrainingMetricsService::to_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get current snapshot with synchronized atomics and updated time metrics
    // We need to temporarily release the lock to call get_current_snapshot
    // Actually, we'll just duplicate the logic here to avoid deadlock
    TrainingMetricsSnapshot snapshot = current_snapshot_;
    snapshot.is_training = is_training_.load();
    snapshot.session_id = current_session_id_.load();
    
    // Update time-dependent metrics for current data
    if (is_training_.load()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - session_start_steady_);
        
        if (elapsed.count() > 0) {
            double elapsed_seconds = elapsed.count() / 1000.0;
            snapshot.total_training_time_seconds = elapsed_seconds;
            snapshot.samples_per_second =
                snapshot.total_samples_trained / elapsed_seconds;
            
            // Estimate time remaining
            if (snapshot.total_samples > 0 && snapshot.samples_per_second > 0) {
                int remaining_samples =
                    snapshot.total_samples * snapshot.total_epochs -
                    snapshot.total_samples_trained;
                snapshot.estimated_time_remaining_seconds =
                    remaining_samples / snapshot.samples_per_second;
            }
        }
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    oss << "{\n";
    oss << "  \"session_id\": " << snapshot.session_id << ",\n";
    oss << "  \"is_training\": " << (snapshot.is_training ? "true" : "false") << ",\n";
    oss << "  \"timestamp\": \"" << format_timestamp(snapshot.last_update_time) << "\",\n";
    oss << "  \"current_epoch\": " << snapshot.current_epoch << ",\n";
    oss << "  \"total_epochs\": " << snapshot.total_epochs << ",\n";
    oss << "  \"current_sample\": " << snapshot.current_sample << ",\n";
    oss << "  \"total_samples\": " << snapshot.total_samples << ",\n";
    oss << "  \"current_loss\": " << snapshot.current_loss << ",\n";
    oss << "  \"running_loss\": " << snapshot.running_loss << ",\n";
    oss << "  \"current_validation_loss\": " << snapshot.current_validation_loss << ",\n";
    oss << "  \"current_validation_perplexity\": " << snapshot.current_validation_perplexity << ",\n";
    oss << "  \"current_validation_accuracy\": " << snapshot.current_validation_accuracy << ",\n";
    oss << "  \"current_learning_rate\": " << snapshot.current_learning_rate << ",\n";
    oss << "  \"current_gradient_norm\": " << snapshot.current_gradient_norm << ",\n";
    oss << "  \"current_perplexity\": " << snapshot.current_perplexity << ",\n";
    oss << "  \"best_validation_loss\": " << snapshot.best_validation_loss << ",\n";
    oss << "  \"best_epoch\": " << snapshot.best_epoch << ",\n";
    oss << "  \"total_samples_trained\": " << snapshot.total_samples_trained << ",\n";
    oss << "  \"total_training_time_seconds\": " << snapshot.total_training_time_seconds << ",\n";
    oss << "  \"samples_per_second\": " << snapshot.samples_per_second << ",\n";
    oss << "  \"estimated_time_remaining_seconds\": " << snapshot.estimated_time_remaining_seconds << ",\n";
    oss << "  \"gradient_variance\": " << snapshot.gradient_variance << ",\n";
    oss << "  \"compute_time_ratio\": " << snapshot.compute_time_ratio << ",\n";
    oss << "  \"weight_update_ratio\": " << snapshot.weight_update_ratio << ",\n";
    oss << "  \"activation_saturation_ratio\": " << snapshot.activation_saturation_ratio << ",\n";
    oss << "  \"attention_entropy\": " << snapshot.attention_entropy << ",\n";
    oss << "  \"current_bleu4\": "  << snapshot.current_bleu4  << ",\n";
    oss << "  \"current_rouge1\": " << snapshot.current_rouge1 << ",\n";
    oss << "  \"current_rouge2\": " << snapshot.current_rouge2 << ",\n";
    oss << "  \"current_rougeL\": " << snapshot.current_rougeL << "\n";
    oss << "}";
    
    return oss.str();
}

std::string TrainingMetricsService::to_json_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return to_json_summary_internal();
}

std::string TrainingMetricsService::to_prometheus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return to_prometheus_internal();
}

std::string TrainingMetricsService::to_csv_header() const {
    return "timestamp,session_id,epoch,sample,loss,validation_loss,learning_rate,gradient_norm,perplexity";
}

std::string TrainingMetricsService::to_csv_row() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    oss << format_timestamp(current_snapshot_.last_update_time) << ",";
    oss << current_session_id_.load() << ",";
    oss << current_snapshot_.current_epoch << ",";
    oss << current_snapshot_.current_sample << ",";
    oss << current_snapshot_.current_loss << ",";
    oss << current_snapshot_.current_validation_loss << ",";
    oss << current_snapshot_.current_learning_rate << ",";
    oss << current_snapshot_.current_gradient_norm << ",";
    oss << current_snapshot_.current_perplexity;
    
    return oss.str();
}

std::vector<PersistentMetricsRecord> TrainingMetricsService::get_history(int max_records) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (max_records <= 0 || max_records >= static_cast<int>(history_.size())) {
        return history_;
    }
    
    return std::vector<PersistentMetricsRecord>(
        history_.end() - max_records, history_.end());
}

std::vector<PersistentMetricsRecord> TrainingMetricsService::get_session_history(int session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<PersistentMetricsRecord> result;
    for (const auto& record : history_) {
        if (record.session_id == session_id) {
            result.push_back(record);
        }
    }
    return result;
}

std::vector<float> TrainingMetricsService::get_epoch_losses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_snapshot_.epoch_losses;
}

std::vector<float> TrainingMetricsService::get_epoch_validation_losses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_snapshot_.epoch_validation_losses;
}

void TrainingMetricsService::flush_to_disk() {
    if (!config_.enable_persistence) {
        return;
    }
    
    // Capture data with lock, then release before expensive I/O operations
    std::string summary_json;
    std::string prometheus_text;
    bool enable_prometheus;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        summary_json = to_json_summary_internal();
        enable_prometheus = config_.enable_prometheus_format;
        if (enable_prometheus) {
            prometheus_text = to_prometheus_internal();
        }
    }
    
    // Now write to files without holding the lock (allows concurrent reads)
    persist_summary_with_data(summary_json);
    if (enable_prometheus) {
        persist_prometheus_with_data(prometheus_text);
    }
    
    // persist_metrics needs special handling as it writes history
    {
        std::lock_guard<std::mutex> lock(mutex_);
        persist_metrics();
        persist_abnormal_samples();
    }
    
    adai::Logger::debug("Metrics flushed to disk");
}

void TrainingMetricsService::clear_history() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    history_.clear();
    adai::Logger::info("Metrics history cleared");
}

void TrainingMetricsService::set_config(const MetricsServiceConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

MetricsServiceConfig TrainingMetricsService::get_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void TrainingMetricsService::persist_metrics() {
    if (!config_.enable_persistence || history_.empty()) {
        return;
    }
    
    try {
        std::ofstream file(config_.metrics_file, std::ios::app);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open metrics file: {}", config_.metrics_file);
            return;
        }
        
        // Write records as JSON Lines (one JSON object per line)
        for (const auto& record : history_) {
            file << std::fixed << std::setprecision(6);
            file << "{";
            file << "\"timestamp\":\"" << format_timestamp(record.timestamp) << "\",";
            file << "\"session_id\":" << record.session_id << ",";
            file << "\"epoch\":" << record.epoch << ",";
            file << "\"sample\":" << record.sample << ",";
            file << "\"loss\":" << record.loss << ",";
            file << "\"validation_loss\":" << record.validation_loss << ",";
            file << "\"learning_rate\":" << record.learning_rate << ",";
            file << "\"gradient_norm\":" << record.gradient_norm << ",";
            file << "\"perplexity\":" << record.perplexity;
            file << "}\n";
        }
        
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist metrics: {}", e.what());
    }
}

void TrainingMetricsService::persist_summary() {
    if (!config_.enable_persistence) {
        return;
    }

    try {
        std::ofstream file(config_.summary_file);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open summary file: {}", config_.summary_file);
            return;
        }

        file << to_json_summary_internal();  // caller already holds mutex_
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist summary: {}", e.what());
    }
}

void TrainingMetricsService::persist_prometheus() {
    if (!config_.enable_prometheus_format) {
        return;
    }

    try {
        std::ofstream file(config_.prometheus_file);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open Prometheus file: {}", config_.prometheus_file);
            return;
        }

        file << to_prometheus_internal();  // caller already holds mutex_
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist Prometheus metrics: {}", e.what());
    }
}

void TrainingMetricsService::persist_summary_with_data(const std::string& json_data) {
    try {
        std::ofstream file(config_.summary_file);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open summary file: {}", config_.summary_file);
            return;
        }
        
        file << json_data;
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist summary: {}", e.what());
    }
}

void TrainingMetricsService::persist_prometheus_with_data(const std::string& prometheus_data) {
    try {
        std::ofstream file(config_.prometheus_file);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open Prometheus file: {}", config_.prometheus_file);
            return;
        }
        
        file << prometheus_data;
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist Prometheus metrics: {}", e.what());
    }
}

std::string TrainingMetricsService::to_json_summary_internal() const {
    // Caller must hold mutex_ lock
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    oss << "{\n";
    oss << "  \"session_id\": " << current_session_id_.load() << ",\n";
    oss << "  \"timestamp\": \"" << format_timestamp(current_snapshot_.last_update_time) << "\",\n";
    oss << "  \"total_epochs_completed\": " << current_snapshot_.epoch_losses.size() << ",\n";
    oss << "  \"total_samples_trained\": " << current_snapshot_.total_samples_trained << ",\n";
    oss << "  \"total_training_time_seconds\": " << current_snapshot_.total_training_time_seconds << ",\n";
    oss << "  \"best_validation_loss\": " << current_snapshot_.best_validation_loss << ",\n";
    oss << "  \"best_epoch\": " << current_snapshot_.best_epoch << ",\n";
    
    oss << "  \"epoch_losses\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_losses.size(); i++) {
        if (i > 0) oss << ", ";
        oss << current_snapshot_.epoch_losses[i];
    }
    oss << "],\n";
    
    oss << "  \"epoch_validation_losses\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_validation_losses.size(); i++) {
        if (i > 0) oss << ", ";
        oss << current_snapshot_.epoch_validation_losses[i];
    }
    oss << "],\n";
    
    oss << "  \"epoch_learning_rates\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_learning_rates.size(); i++) {
        if (i > 0) oss << ", ";
        oss << current_snapshot_.epoch_learning_rates[i];
    }
    oss << "],\n";
    
    oss << "  \"epoch_perplexities\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_perplexities.size(); i++) {
        if (i > 0) oss << ", ";
        oss << current_snapshot_.epoch_perplexities[i];
    }
    oss << "],\n";
    
    oss << "  \"epoch_durations\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_durations.size(); i++) {
        if (i > 0) oss << ", ";
        oss << current_snapshot_.epoch_durations[i];
    }
    oss << "]\n";
    
    oss << "}";
    
    return oss.str();
}

std::string TrainingMetricsService::to_prometheus_internal() const {
    // Caller must hold mutex_ lock
    int session_id = current_session_id_.load();
    bool is_training = is_training_.load();
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_snapshot_.last_update_time.time_since_epoch()).count();
    
    oss << "# HELP training_session_id Current training session ID\n";
    oss << "# TYPE training_session_id gauge\n";
    oss << "training_session_id " << session_id << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_is_active Whether training is currently active\n";
    oss << "# TYPE training_is_active gauge\n";
    oss << "training_is_active " << (is_training ? 1 : 0) << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_current_epoch Current training epoch\n";
    oss << "# TYPE training_current_epoch gauge\n";
    oss << "training_current_epoch " << current_snapshot_.current_epoch << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_loss Current training loss\n";
    oss << "# TYPE training_loss gauge\n";
    oss << "training_loss " << current_snapshot_.current_loss << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_validation_loss Current validation loss\n";
    oss << "# TYPE training_validation_loss gauge\n";
    oss << "training_validation_loss " << current_snapshot_.current_validation_loss << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_learning_rate Current learning rate\n";
    oss << "# TYPE training_learning_rate gauge\n";
    oss << "training_learning_rate " << current_snapshot_.current_learning_rate << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_gradient_norm Current gradient norm\n";
    oss << "# TYPE training_gradient_norm gauge\n";
    oss << "training_gradient_norm " << current_snapshot_.current_gradient_norm << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_perplexity Current perplexity\n";
    oss << "# TYPE training_perplexity gauge\n";
    oss << "training_perplexity " << current_snapshot_.current_perplexity << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_samples_total Total samples trained\n";
    oss << "# TYPE training_samples_total counter\n";
    oss << "training_samples_total " << current_snapshot_.total_samples_trained << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_time_seconds_total Total training time in seconds\n";
    oss << "# TYPE training_time_seconds_total counter\n";
    oss << "training_time_seconds_total " << current_snapshot_.total_training_time_seconds << " " << timestamp_ms << "\n\n";
    
    oss << "# HELP training_samples_per_second Training throughput\n";
    oss << "# TYPE training_samples_per_second gauge\n";
    oss << "training_samples_per_second " << current_snapshot_.samples_per_second << " " << timestamp_ms << "\n\n";
    
    return oss.str();
}

void TrainingMetricsService::add_record(const PersistentMetricsRecord& record) {
    history_.push_back(record);
    trim_history();
}

void TrainingMetricsService::trim_history() {
    if (history_.size() > static_cast<size_t>(config_.max_records_in_memory)) {
        // Keep only the most recent records
        history_.erase(
            history_.begin(),
            history_.begin() + (history_.size() - config_.max_records_in_memory));
    }
}

void TrainingMetricsService::update_throughput_metrics() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - session_start_steady_);
    
    if (elapsed.count() > 0) {
        double elapsed_seconds = elapsed.count() / 1000.0;
        current_snapshot_.total_training_time_seconds = elapsed_seconds;
        current_snapshot_.samples_per_second =
            current_snapshot_.total_samples_trained / elapsed_seconds;
        
        // Estimate time remaining
        if (current_snapshot_.total_samples > 0 && current_snapshot_.samples_per_second > 0) {
            int remaining_samples =
                current_snapshot_.total_samples * current_snapshot_.total_epochs -
                current_snapshot_.total_samples_trained;
            current_snapshot_.estimated_time_remaining_seconds =
                remaining_samples / current_snapshot_.samples_per_second;
        }
    }
}

std::string TrainingMetricsService::escape_json(const std::string& s) const {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

std::string TrainingMetricsService::format_timestamp(
    const std::chrono::system_clock::time_point& tp) const {
    
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm = *std::localtime(&time_t);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    // Add milliseconds
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()) % 1000;
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

// ============================================================================
// HTTP Push to External Metrics API Daemon
// ============================================================================

#ifdef BUILD_METRICS_API_SERVER

std::string TrainingMetricsService::build_push_url(const std::string& endpoint) const {
    std::string url = config_.push_url;
    // Remove trailing slash if present
    if (!url.empty() && url[url.length() - 1] == '/') {
        url = url.substr(0, url.length() - 1);
    }
    return url + endpoint;
}

void TrainingMetricsService::push_to_api(const std::string& endpoint, const std::string& json_body) {
    if (!config_.enable_push) {
        return;
    }

    // Fire-and-forget: dispatch HTTP call onto a detached thread so the
    // training loop is never blocked by network latency or a slow/unresponsive
    // metrics server.
    std::string push_url  = config_.push_url;
    int         timeout   = config_.push_timeout_ms;

    std::thread([push_url, timeout, endpoint, json_body]() {
        try {
            // Parse URL to extract host and port
            std::string url  = push_url;
            std::string host = "localhost";
            int         port = 8081;

            size_t proto_pos = url.find("://");
            if (proto_pos != std::string::npos) {
                url = url.substr(proto_pos + 3);
            }

            size_t port_pos = url.find(':');
            if (port_pos != std::string::npos) {
                host = url.substr(0, port_pos);
                size_t path_pos = url.find('/', port_pos);
                std::string port_str;
                if (path_pos != std::string::npos) {
                    port_str = url.substr(port_pos + 1, path_pos - port_pos - 1);
                } else {
                    port_str = url.substr(port_pos + 1);
                }
                port = std::stoi(port_str);
            } else {
                size_t path_pos = url.find('/');
                if (path_pos != std::string::npos) {
                    host = url.substr(0, path_pos);
                } else {
                    host = url;
                }
            }

            httplib::Client client(host.c_str(), port);
            client.set_connection_timeout(0, timeout * 1000);
            client.set_read_timeout(timeout / 1000, (timeout % 1000) * 1000);
            client.set_write_timeout(timeout / 1000, (timeout % 1000) * 1000);

            auto res = client.Post(endpoint.c_str(), json_body, "application/json");

            if (!res) {
                adai::Logger::debug("Metrics push to {}{}: connection failed", push_url, endpoint);
            } else if (res->status != 200) {
                adai::Logger::debug("Metrics push to {}{}: HTTP {}", push_url, endpoint, res->status);
            }
        } catch (const std::exception& e) {
            adai::Logger::debug("Metrics push exception {}{}: {}", push_url, endpoint, e.what());
        }
    }).detach();
}

#else

// Stub implementations when httplib is not available
std::string TrainingMetricsService::build_push_url(const std::string&) const {
    return "";
}

void TrainingMetricsService::push_to_api(const std::string&, const std::string&) {
    // No-op when httplib not available
}

#endif

// ============================================================================
// TD-013: Advanced Metrics & Outlier Detection
// ============================================================================

void TrainingMetricsService::update_advanced_epoch_metrics(float gradient_variance,
                                                           float compute_time_ratio,
                                                           float weight_update_ratio) {
    std::string push_json;
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_snapshot_.gradient_variance   = gradient_variance;
        current_snapshot_.compute_time_ratio  = compute_time_ratio;
        current_snapshot_.weight_update_ratio = weight_update_ratio;
        adai::Logger::debug("Advanced epoch metrics: grad_var={:.4f}, compute_ratio={:.4f}, wu_ratio={:.6f}",
                            gradient_variance, compute_time_ratio, weight_update_ratio);
        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"gradient_variance\":" << gradient_variance
                 << ",\"compute_time_ratio\":" << compute_time_ratio
                 << ",\"weight_update_ratio\":" << weight_update_ratio << "}";
            push_json = json.str();
        }
    }
    if (should_push) {
        push_to_api("/api/metrics/advanced", push_json);
    }
}

void TrainingMetricsService::update_activation_saturation(float ratio) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_snapshot_.activation_saturation_ratio = ratio;
    adai::Logger::debug("Activation saturation ratio: {:.4f}", ratio);
}

void TrainingMetricsService::update_attention_entropy(float entropy) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_snapshot_.attention_entropy = entropy;
    adai::Logger::debug("Attention entropy: {:.4f}", entropy);
}

void TrainingMetricsService::update_generation_quality_metrics(
        float bleu4, float rouge1, float rouge2, float rougeL) {
    std::string push_json;
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_snapshot_.current_bleu4  = bleu4;
        current_snapshot_.current_rouge1 = rouge1;
        current_snapshot_.current_rouge2 = rouge2;
        current_snapshot_.current_rougeL = rougeL;
        current_snapshot_.last_update_time = std::chrono::system_clock::now();

        adai::Logger::info("Generation quality — BLEU-4: {:.4f}  ROUGE-1: {:.4f}  "
                           "ROUGE-2: {:.4f}  ROUGE-L: {:.4f}",
                           bleu4, rouge1, rouge2, rougeL);

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << std::fixed << std::setprecision(6);
            json << "{\"bleu4\":"  << bleu4
                 << ",\"rouge1\":" << rouge1
                 << ",\"rouge2\":" << rouge2
                 << ",\"rougeL\":" << rougeL << "}";
            push_json = json.str();
        }
    }
    if (should_push) {
        push_to_api("/api/metrics/generation-quality", push_json);
    }
}

void TrainingMetricsService::flag_abnormal_sample(const AbnormalSample& sample) {
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (static_cast<int>(abnormal_samples_.size()) >= config_.max_abnormal_samples) {
            // Drop oldest entry to stay within cap
            abnormal_samples_.erase(abnormal_samples_.begin());
        }
        abnormal_samples_.push_back(sample);

        adai::Logger::warn("Abnormal sample flagged — epoch={} sample={} loss={:.4f} grad={:.4f} reason={}",
                           sample.epoch, sample.sample_id, sample.loss,
                           sample.grad_norm, sample.reason);

        persist_abnormal_samples();
    }
}

std::vector<AbnormalSample> TrainingMetricsService::get_abnormal_samples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return abnormal_samples_;
}

void TrainingMetricsService::persist_abnormal_samples() {
    // Caller must hold mutex_
    if (!config_.enable_persistence || abnormal_samples_.empty()) {
        return;
    }

    try {
        // Ensure directory exists
        fs::path p(config_.abnormal_samples_file);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }

        std::ofstream file(config_.abnormal_samples_file);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open abnormal samples file: {}",
                               config_.abnormal_samples_file);
            return;
        }

        file << "[\n";
        for (size_t i = 0; i < abnormal_samples_.size(); ++i) {
            const auto& s = abnormal_samples_[i];
            file << "  {";
            file << "\"epoch\":" << s.epoch << ",";
            file << "\"sample_id\":" << s.sample_id << ",";
            file << "\"loss\":" << std::fixed << std::setprecision(6) << s.loss << ",";
            file << "\"grad_norm\":" << s.grad_norm << ",";
            file << "\"reason\":\"" << escape_json(s.reason) << "\",";
            file << "\"input_text\":\"" << escape_json(s.input_text) << "\",";
            file << "\"target_text\":\"" << escape_json(s.target_text) << "\",";
            file << "\"timestamp\":\"" << format_timestamp(s.timestamp) << "\"";
            file << "}";
            if (i + 1 < abnormal_samples_.size()) file << ",";
            file << "\n";
        }
        file << "]\n";
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist abnormal samples: {}", e.what());
    }
}

// ============================================================================
// GlobalMetricsService Implementation
// ============================================================================

std::unique_ptr<TrainingMetricsService> GlobalMetricsService::instance_ = nullptr;
std::mutex GlobalMetricsService::instance_mutex_;

TrainingMetricsService& GlobalMetricsService::instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_unique<TrainingMetricsService>();
    }
    return *instance_;
}

void GlobalMetricsService::initialize(const MetricsServiceConfig& config) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    instance_ = std::make_unique<TrainingMetricsService>(config);
}

void GlobalMetricsService::shutdown() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    instance_.reset();
}
