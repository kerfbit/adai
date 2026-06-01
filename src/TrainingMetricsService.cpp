#include "TrainingMetricsService.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <utility>
#include "Logger.hpp"
#include "MetricsSessionRegistry.hpp"

// HTTP client for pushing metrics to external API daemon (optional)
#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#include <cmath>
#endif

namespace fs = std::filesystem;

namespace {

void ensure_parent_directory(const std::string& file_path) {
    fs::path path(file_path);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
}

void ensure_metrics_output_directories(const MetricsServiceConfig& config) {
    if (!config.enable_persistence) {
        return;
    }

    ensure_parent_directory(config.metrics_file);
    ensure_parent_directory(config.summary_file);
    ensure_parent_directory(config.abnormal_samples_file);

    if (config.enable_prometheus_format) {
        ensure_parent_directory(config.prometheus_file);
    }
}

}  // namespace

// ============================================================================
// TrainingMetricsService Implementation
// ============================================================================

TrainingMetricsService::TrainingMetricsService(MetricsServiceConfig config)
    : config_(std::move(config)), is_training_(false), current_session_id_(0) {
    // Ensure all configured output directories exist so callers can safely pass
    // per-session file paths in the config copy used to construct the service.
    if (config_.enable_persistence) {
        ensure_metrics_output_directories(config_);
        // Restore snapshot from last persisted summary (survives restarts)
        if (fs::exists(config_.summary_file)) {
            restore_from_summary();
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

void TrainingMetricsService::start_session(int session_id, int total_epochs, int total_samples,
                                           const std::string& label,
                                           const std::string& config_snapshot) {
    std::string push_json;
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        const float previous_best_validation_loss = current_snapshot_.best_validation_loss;
        const int previous_best_epoch = current_snapshot_.best_epoch;

        current_session_id_ = session_id;
        is_training_ = true;
        label_ = label;
        config_snapshot_ = config_snapshot;

        current_snapshot_ = TrainingMetricsSnapshot();
        current_snapshot_.session_id = session_id;
        current_snapshot_.is_training = true;
        current_snapshot_.label = label;
        current_snapshot_.config_snapshot = config_snapshot;
        current_snapshot_.total_epochs = total_epochs;
        current_snapshot_.total_samples = total_samples;
        current_snapshot_.session_start_time = std::chrono::system_clock::now();
        current_snapshot_.last_update_time = current_snapshot_.session_start_time;
        current_snapshot_.best_validation_loss = previous_best_validation_loss;
        current_snapshot_.best_epoch = previous_best_epoch;

        session_start_steady_ = std::chrono::steady_clock::now();
        samples_since_last_persist_ = 0;
        last_persist_time_ = std::chrono::system_clock::now();

        adai::Logger::info("Metrics session {} started (epochs={}, samples={}, label={})",
                           session_id, total_epochs, total_samples,
                           label.empty() ? "(none)" : label);
        adai::Logger::info("Metrics push config: enable_push={}, push_url={}", config_.enable_push,
                           config_.push_url);

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"session_id\":" << session_id << ",\"total_epochs\":" << total_epochs
                 << ",\"total_samples\":" << total_samples;
            if (!label.empty()) {
                json << ",\"label\":\"" << escape_json(label) << "\"";
            }
            if (!config_snapshot.empty()) {
                json << ",\"config\":" << config_snapshot;
            }
            json << "}";
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
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - session_start_steady_);
        current_snapshot_.total_training_time_seconds =
            static_cast<double>(duration.count()) / 1000.0;

        // Persist final state
        persist_metrics();
        persist_summary();
        if (config_.enable_prometheus_format) {
            persist_prometheus();
        }

        adai::Logger::info(
            "Metrics session {} ended (total_time={:.2f}s, samples={})", current_session_id_.load(),
            current_snapshot_.total_training_time_seconds, current_snapshot_.total_samples_trained);

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
            json << "{\"epoch\":" << epoch << ",\"total_samples\":" << total_samples << "}";
            push_json = json.str();
        }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/epoch/start", push_json);
    }
}

void TrainingMetricsService::end_epoch(int epoch, float loss, float validation_loss,
                                       float learning_rate, float perplexity, float gradient_norm,
                                       double epoch_time_seconds) {
    std::string push_json;
    bool should_push = false;
    float stored_perplexity = 0.0f;
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Calculate epoch duration — use caller-provided value if available (avoids
        // double-measurement when the API server receives a push from the trainer).
        double epoch_time = NAN;
        if (epoch_time_seconds > 0.0) {
            epoch_time = epoch_time_seconds;
        } else {
            auto now = std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch_start_steady_);
            epoch_time = static_cast<double>(duration.count()) / 1000.0;
        }

        // Update epoch history
        current_snapshot_.epoch_losses.push_back(loss);
        current_snapshot_.epoch_validation_losses.push_back(validation_loss);
        current_snapshot_.epoch_learning_rates.push_back(learning_rate);
        current_snapshot_.epoch_perplexities.push_back(perplexity > 0 ? perplexity
                                                                      : std::exp(loss));
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
        // Persist per-epoch padding efficiency (-1 if not computed this epoch)
        current_snapshot_.epoch_padding_efficiencies.push_back(
            current_snapshot_.current_padding_efficiency);
        // TD-017: Persist per-epoch adaptive clip threshold (-1 if adaptive clipping not active)
        // Note: update_adaptive_clip_epoch() already pushes into epoch_adaptive_clip_thresholds;
        // nothing to do here — the push from ChatbotTrainer arrives before end_epoch() is called.
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

        adai::Logger::debug(
            "Metrics epoch {} completed (loss={:.4f}, val_loss={:.4f}, time={:.2f}s)", epoch, loss,
            validation_loss, epoch_time);

        stored_perplexity = current_snapshot_.current_perplexity;
        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"epoch\":" << epoch << ",\"loss\":" << loss
                 << ",\"validation_loss\":" << validation_loss
                 << ",\"learning_rate\":" << learning_rate
                 << ",\"perplexity\":" << stored_perplexity
                 << ",\"gradient_norm\":" << gradient_norm << ",\"epoch_time\":" << epoch_time
                 << ",\"gradient_variance\":" << current_snapshot_.gradient_variance
                 << ",\"compute_time_ratio\":" << current_snapshot_.compute_time_ratio
                 << ",\"weight_update_ratio\":" << current_snapshot_.weight_update_ratio
                 << ",\"activation_saturation_ratio\":"
                 << current_snapshot_.activation_saturation_ratio
                 << ",\"attention_entropy\":" << current_snapshot_.attention_entropy
                 << ",\"current_padding_efficiency\":"
                 << current_snapshot_.current_padding_efficiency << "}";
            push_json = json.str();
        }
    }  // mutex released here
    if (should_push) {
        push_to_api("/api/epoch/end", push_json);
    }
}

void TrainingMetricsService::update_sample_metrics(int sample, float loss, float gradient_norm,
                                                   float learning_rate) {
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
            current_snapshot_.running_loss =
                alpha * loss + (1.0f - alpha) * current_snapshot_.running_loss;
        }

        samples_since_last_persist_++;

        // Update throughput metrics
        update_throughput_metrics();

        // Build push payload while lock is held
        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"sample\":" << sample << ",\"loss\":" << loss
                 << ",\"gradient_norm\":" << gradient_norm << ",\"learning_rate\":" << learning_rate
                 << "}";
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
            (validation_perplexity > 0.0f)
                ? validation_perplexity
                : (validation_loss > 0.0f ? std::exp(validation_loss) : 0.0f);
        current_snapshot_.current_validation_accuracy = validation_accuracy;
        current_snapshot_.last_update_time = std::chrono::system_clock::now();

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << "{\"validation_loss\":" << validation_loss
                 << ",\"validation_accuracy\":" << validation_accuracy
                 << ",\"validation_perplexity\":" << current_snapshot_.current_validation_perplexity
                 << "}";
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
            adai::Logger::info("New best validation loss: {:.4f} (epoch {})", validation_loss,
                               epoch);

            should_push = config_.enable_push;
            if (should_push) {
                std::ostringstream json;
                json << "{\"validation_loss\":" << validation_loss << ",\"epoch\":" << epoch << "}";
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
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - session_start_steady_);

        if (elapsed.count() > 0) {
            double elapsed_seconds = static_cast<double>(elapsed.count()) / 1000.0;
            snapshot.total_training_time_seconds = elapsed_seconds;
            snapshot.samples_per_second = static_cast<float>(
                static_cast<float>(snapshot.total_samples_trained) / elapsed_seconds);

            // Estimate time remaining
            if (snapshot.total_samples > 0 && snapshot.samples_per_second > 0) {
                int remaining_samples =
                    snapshot.total_samples * snapshot.total_epochs - snapshot.total_samples_trained;
                snapshot.estimated_time_remaining_seconds =
                    static_cast<float>(remaining_samples) / snapshot.samples_per_second;
            }
        }

        // TD-019: Do NOT overwrite last_update_time — preserve the true ingest timestamp.
    }

    // TD-019: Compute stale-state fields from the preserved ingest timestamp.
    {
        auto now_wall = std::chrono::system_clock::now();
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                        now_wall - snapshot.last_update_time)
                        .count();
        snapshot.seconds_since_last_update = static_cast<double>(secs);
        snapshot.is_stale =
            snapshot.is_training && (secs > config_.staleness_threshold_seconds);
        snapshot.effective_is_training = snapshot.is_training && !snapshot.is_stale;
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
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - session_start_steady_);

        if (elapsed.count() > 0) {
            double elapsed_seconds = static_cast<double>(elapsed.count()) / 1000.0;
            snapshot.total_training_time_seconds = elapsed_seconds;
            snapshot.samples_per_second = static_cast<float>(
                static_cast<float>(snapshot.total_samples_trained) / elapsed_seconds);

            // Estimate time remaining
            if (snapshot.total_samples > 0 && snapshot.samples_per_second > 0) {
                int remaining_samples =
                    snapshot.total_samples * snapshot.total_epochs - snapshot.total_samples_trained;
                snapshot.estimated_time_remaining_seconds =
                    static_cast<float>(remaining_samples) / snapshot.samples_per_second;
            }
        }
    }

    // TD-019: Compute stale-state fields from preserved ingest timestamp.
    {
        auto now_wall = std::chrono::system_clock::now();
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                        now_wall - snapshot.last_update_time)
                        .count();
        snapshot.seconds_since_last_update = static_cast<double>(secs);
        snapshot.is_stale =
            snapshot.is_training && (secs > config_.staleness_threshold_seconds);
        snapshot.effective_is_training = snapshot.is_training && !snapshot.is_stale;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "{\n";
    oss << "  \"session_id\": " << snapshot.session_id << ",\n";
    oss << "  \"is_training\": " << (snapshot.is_training ? "true" : "false") << ",\n";
    oss << R"(  "timestamp": ")" << format_timestamp(snapshot.last_update_time) << "\",\n";
    oss << "  \"current_epoch\": " << snapshot.current_epoch << ",\n";
    oss << "  \"total_epochs\": " << snapshot.total_epochs << ",\n";
    oss << "  \"current_sample\": " << snapshot.current_sample << ",\n";
    oss << "  \"total_samples\": " << snapshot.total_samples << ",\n";
    oss << "  \"current_loss\": " << snapshot.current_loss << ",\n";
    oss << "  \"running_loss\": " << snapshot.running_loss << ",\n";
    oss << "  \"current_validation_loss\": " << snapshot.current_validation_loss << ",\n";
    oss << "  \"current_validation_perplexity\": " << snapshot.current_validation_perplexity
        << ",\n";
    oss << "  \"current_validation_accuracy\": " << snapshot.current_validation_accuracy << ",\n";
    oss << "  \"current_learning_rate\": " << snapshot.current_learning_rate << ",\n";
    oss << "  \"current_gradient_norm\": " << snapshot.current_gradient_norm << ",\n";
    oss << "  \"current_perplexity\": " << snapshot.current_perplexity << ",\n";
    oss << "  \"best_validation_loss\": " << snapshot.best_validation_loss << ",\n";
    oss << "  \"best_epoch\": " << snapshot.best_epoch << ",\n";
    oss << "  \"total_samples_trained\": " << snapshot.total_samples_trained << ",\n";
    oss << "  \"total_training_time_seconds\": " << snapshot.total_training_time_seconds << ",\n";
    oss << "  \"samples_per_second\": " << snapshot.samples_per_second << ",\n";
    oss << "  \"estimated_time_remaining_seconds\": " << snapshot.estimated_time_remaining_seconds
        << ",\n";
    oss << "  \"gradient_variance\": " << snapshot.gradient_variance << ",\n";
    oss << "  \"compute_time_ratio\": " << snapshot.compute_time_ratio << ",\n";
    oss << "  \"weight_update_ratio\": " << snapshot.weight_update_ratio << ",\n";
    oss << "  \"activation_saturation_ratio\": " << snapshot.activation_saturation_ratio << ",\n";
    oss << "  \"attention_entropy\": " << snapshot.attention_entropy << ",\n";
    oss << "  \"current_bleu4\": " << snapshot.current_bleu4 << ",\n";
    oss << "  \"current_rouge1\": " << snapshot.current_rouge1 << ",\n";
    oss << "  \"current_rouge2\": " << snapshot.current_rouge2 << ",\n";
    oss << "  \"current_rougeL\": " << snapshot.current_rougeL << ",\n";
    oss << "  \"current_padding_efficiency\": " << snapshot.current_padding_efficiency << ",\n";
    oss << "  \"current_adaptive_clip_threshold\": " << snapshot.current_adaptive_clip_threshold
        << ",\n";
    oss << "  \"current_adaptive_clip_spikes\": " << snapshot.current_adaptive_clip_spikes << ",\n";
    oss << "  \"is_stale\": " << (snapshot.is_stale ? "true" : "false") << ",\n";
    oss << "  \"seconds_since_last_update\": " << snapshot.seconds_since_last_update << ",\n";
    oss << "  \"effective_is_training\": "
        << (snapshot.effective_is_training ? "true" : "false") << "\n";
    oss << "}";

    return oss.str();
}

std::string TrainingMetricsService::to_json_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return to_json_summary_internal();
}

std::string TrainingMetricsService::to_prometheus(const std::string& session_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return to_prometheus_internal(session_key);
}

std::string TrainingMetricsService::to_csv_header() {
    return "timestamp,session_id,epoch,sample,loss,validation_loss,learning_rate,gradient_norm,"
           "perplexity";
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

    return std::vector<PersistentMetricsRecord>(history_.end() - max_records, history_.end());
}

std::vector<PersistentMetricsRecord> TrainingMetricsService::get_session_history(
    int session_id) const {
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
    bool enable_prometheus = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        summary_json = to_json_summary_internal();
        enable_prometheus = config_.enable_prometheus_format;
        if (enable_prometheus) {
            prometheus_text = to_prometheus_internal("");
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
    ensure_metrics_output_directories(config_);
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
        ensure_parent_directory(config_.metrics_file);
        std::ofstream file(config_.metrics_file, std::ios::app);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open metrics file: {}", config_.metrics_file);
            return;
        }

        // Write records as JSON Lines (one JSON object per line)
        for (const auto& record : history_) {
            file << std::fixed << std::setprecision(6);
            file << "{";
            file << R"("timestamp":")" << format_timestamp(record.timestamp) << "\",";
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
        ensure_parent_directory(config_.summary_file);
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
        ensure_parent_directory(config_.prometheus_file);
        std::ofstream file(config_.prometheus_file);
        if (!file.is_open()) {
            adai::Logger::warn("Failed to open Prometheus file: {}", config_.prometheus_file);
            return;
        }

        file << to_prometheus_internal("");  // caller already holds mutex_
        file.close();
    } catch (const std::exception& e) {
        adai::Logger::error("Failed to persist Prometheus metrics: {}", e.what());
    }
}

void TrainingMetricsService::persist_summary_with_data(const std::string& json_data) {
    try {
        ensure_parent_directory(config_.summary_file);
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
        ensure_parent_directory(config_.prometheus_file);
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

void TrainingMetricsService::restore_from_summary() {
    // Restore current_snapshot_ from the persisted summary JSON.
    // Called at construction before any lock is needed.
    try {
        std::ifstream f(config_.summary_file);
        if (!f.is_open()) {
            return;
        }
        std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();

        // Parse a scalar float value following "key":
        auto parse_float = [&](const std::string& key, float fallback) -> float {
            auto pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) {
                return fallback;
            }
            pos = json.find(':', pos);
            if (pos == std::string::npos) {
                return fallback;
            }
            try {
                return std::stof(json.substr(pos + 1));
            } catch (...) {
                return fallback;
            }
        };
        auto parse_int = [&](const std::string& key, int fallback) -> int {
            auto pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) {
                return fallback;
            }
            pos = json.find(':', pos);
            if (pos == std::string::npos) {
                return fallback;
            }
            try {
                return std::stoi(json.substr(pos + 1));
            } catch (...) {
                return fallback;
            }
        };
        auto parse_double = [&](const std::string& key, double fallback) -> double {
            auto pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) {
                return fallback;
            }
            pos = json.find(':', pos);
            if (pos == std::string::npos) {
                return fallback;
            }
            try {
                return std::stod(json.substr(pos + 1));
            } catch (...) {
                return fallback;
            }
        };
        auto parse_float_array = [&](const std::string& key) -> std::vector<float> {
            std::vector<float> result;
            auto pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) {
                return result;
            }
            auto lb = json.find('[', pos);
            auto rb = json.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) {
                return result;
            }
            std::istringstream ss(json.substr(lb + 1, rb - lb - 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    result.push_back(std::stof(token));
                } catch (...) {
                    continue;  // skip malformed token
                }
            }
            return result;
        };
        auto parse_double_array = [&](const std::string& key) -> std::vector<double> {
            std::vector<double> result;
            auto pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) {
                return result;
            }
            auto lb = json.find('[', pos);
            auto rb = json.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) {
                return result;
            }
            std::istringstream ss(json.substr(lb + 1, rb - lb - 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    result.push_back(std::stod(token));
                } catch (...) {
                    continue;  // skip malformed token
                }
            }
            return result;
        };

        int session_id = parse_int("session_id", 0);
        current_session_id_ = session_id;
        current_snapshot_.session_id = session_id;
        current_snapshot_.total_samples_trained = parse_int("total_samples_trained", 0);
        current_snapshot_.total_training_time_seconds =
            parse_double("total_training_time_seconds", 0.0);
        current_snapshot_.best_validation_loss =
            parse_float("best_validation_loss", std::numeric_limits<float>::max());
        current_snapshot_.best_epoch = parse_int("best_epoch", 0);

        current_snapshot_.epoch_losses = parse_float_array("epoch_losses");
        current_snapshot_.epoch_validation_losses = parse_float_array("epoch_validation_losses");
        current_snapshot_.epoch_learning_rates = parse_float_array("epoch_learning_rates");
        current_snapshot_.epoch_perplexities = parse_float_array("epoch_perplexities");
        current_snapshot_.epoch_durations = parse_double_array("epoch_durations");

        current_snapshot_.current_bleu4 = parse_float("current_bleu4", -1.0f);
        current_snapshot_.current_rouge1 = parse_float("current_rouge1", -1.0f);
        current_snapshot_.current_rouge2 = parse_float("current_rouge2", -1.0f);
        current_snapshot_.current_rougeL = parse_float("current_rougeL", -1.0f);
        current_snapshot_.epoch_bleu4 = parse_float_array("epoch_bleu4");
        current_snapshot_.epoch_rouge1 = parse_float_array("epoch_rouge1");
        current_snapshot_.epoch_rouge2 = parse_float_array("epoch_rouge2");
        current_snapshot_.epoch_rougeL = parse_float_array("epoch_rougeL");

        // Current epoch = number of completed epoch records
        current_snapshot_.current_epoch = static_cast<int>(current_snapshot_.epoch_losses.size());

        adai::Logger::info("Metrics state restored from '{}' (session={}, epochs={})",
                           config_.summary_file, session_id, current_snapshot_.epoch_losses.size());
    } catch (const std::exception& e) {
        adai::Logger::warn("Could not restore metrics snapshot: {}", e.what());
    }
}

std::string TrainingMetricsService::to_json_summary_internal() const {
    // Caller must hold mutex_ lock
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "{\n";
    oss << "  \"session_id\": " << current_session_id_.load() << ",\n";
    oss << R"(  "timestamp": ")" << format_timestamp(current_snapshot_.last_update_time) << "\",\n";
    oss << "  \"total_epochs_completed\": " << current_snapshot_.epoch_losses.size() << ",\n";
    oss << "  \"total_samples_trained\": " << current_snapshot_.total_samples_trained << ",\n";
    oss << "  \"total_training_time_seconds\": " << current_snapshot_.total_training_time_seconds
        << ",\n";
    oss << "  \"best_validation_loss\": " << current_snapshot_.best_validation_loss << ",\n";
    oss << "  \"best_epoch\": " << current_snapshot_.best_epoch << ",\n";

    oss << "  \"epoch_losses\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_losses.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_losses[i];
    }
    oss << "],\n";

    oss << "  \"epoch_validation_losses\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_validation_losses.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_validation_losses[i];
    }
    oss << "],\n";

    oss << "  \"epoch_learning_rates\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_learning_rates.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_learning_rates[i];
    }
    oss << "],\n";

    oss << "  \"epoch_perplexities\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_perplexities.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_perplexities[i];
    }
    oss << "],\n";

    oss << "  \"epoch_durations\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_durations.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_durations[i];
    }
    oss << "],\n";

    // Generation quality scalars (TD-016)
    oss << "  \"current_bleu4\": " << current_snapshot_.current_bleu4 << ",\n";
    oss << "  \"current_rouge1\": " << current_snapshot_.current_rouge1 << ",\n";
    oss << "  \"current_rouge2\": " << current_snapshot_.current_rouge2 << ",\n";
    oss << "  \"current_rougeL\": " << current_snapshot_.current_rougeL << ",\n";

    oss << "  \"epoch_bleu4\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_bleu4.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_bleu4[i];
    }
    oss << "],\n";

    oss << "  \"epoch_rouge1\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_rouge1.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_rouge1[i];
    }
    oss << "],\n";

    oss << "  \"epoch_rouge2\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_rouge2.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_rouge2[i];
    }
    oss << "],\n";

    oss << "  \"epoch_rougeL\": [";
    for (size_t i = 0; i < current_snapshot_.epoch_rougeL.size(); i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << current_snapshot_.epoch_rougeL[i];
    }
    oss << "]\n";

    oss << "}";

    return oss.str();
}

std::string TrainingMetricsService::to_prometheus_internal(const std::string& session_key) const {
    // Caller must hold mutex_ lock
    int session_id = current_session_id_.load();
    bool is_training = is_training_.load();

    // When session_key is non-empty, prefix each metric value with a label set so that
    // concurrent sessions are distinguishable by Prometheus scrapers (TD-021 §4.8).
    const std::string lbl =
        session_key.empty() ? "" : "{session=\"" + session_key + "\"} ";

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            current_snapshot_.last_update_time.time_since_epoch())
                            .count();

    oss << "# HELP training_session_id Current training session ID\n";
    oss << "# TYPE training_session_id gauge\n";
    oss << "training_session_id " << lbl << session_id << " " << timestamp_ms << "\n\n";

    oss << "# HELP training_is_active Whether training is currently active\n";
    oss << "# TYPE training_is_active gauge\n";
    oss << "training_is_active " << lbl << (is_training ? 1 : 0) << " " << timestamp_ms << "\n\n";

    oss << "# HELP training_current_epoch Current training epoch\n";
    oss << "# TYPE training_current_epoch gauge\n";
    oss << "training_current_epoch " << lbl << current_snapshot_.current_epoch << " "
        << timestamp_ms << "\n\n";

    oss << "# HELP training_loss Current training loss\n";
    oss << "# TYPE training_loss gauge\n";
    oss << "training_loss " << lbl << current_snapshot_.current_loss << " " << timestamp_ms
        << "\n\n";

    oss << "# HELP training_validation_loss Current validation loss\n";
    oss << "# TYPE training_validation_loss gauge\n";
    oss << "training_validation_loss " << lbl << current_snapshot_.current_validation_loss << " "
        << timestamp_ms << "\n\n";

    oss << "# HELP training_learning_rate Current learning rate\n";
    oss << "# TYPE training_learning_rate gauge\n";
    oss << "training_learning_rate " << lbl << current_snapshot_.current_learning_rate << " "
        << timestamp_ms << "\n\n";

    oss << "# HELP training_gradient_norm Current gradient norm\n";
    oss << "# TYPE training_gradient_norm gauge\n";
    oss << "training_gradient_norm " << lbl << current_snapshot_.current_gradient_norm << " "
        << timestamp_ms << "\n\n";

    oss << "# HELP training_perplexity Current perplexity\n";
    oss << "# TYPE training_perplexity gauge\n";
    oss << "training_perplexity " << lbl << current_snapshot_.current_perplexity << " "
        << timestamp_ms << "\n\n";

    oss << "# HELP training_samples_total Total samples trained\n";
    oss << "# TYPE training_samples_total counter\n";
    oss << "training_samples_total " << lbl << current_snapshot_.total_samples_trained << " "
        << timestamp_ms << "\n\n";

    oss << "# HELP training_time_seconds_total Total training time in seconds\n";
    oss << "# TYPE training_time_seconds_total counter\n";
    oss << "training_time_seconds_total " << lbl << current_snapshot_.total_training_time_seconds
        << " " << timestamp_ms << "\n\n";

    oss << "# HELP training_samples_per_second Training throughput\n";
    oss << "# TYPE training_samples_per_second gauge\n";
    oss << "training_samples_per_second " << lbl << current_snapshot_.samples_per_second << " "
        << timestamp_ms << "\n\n";

    return oss.str();
}

void TrainingMetricsService::add_record(const PersistentMetricsRecord& record) {
    history_.push_back(record);
    trim_history();
}

void TrainingMetricsService::trim_history() {
    if (history_.size() > static_cast<size_t>(config_.max_records_in_memory)) {
        // Keep only the most recent records
        history_.erase(history_.begin(),
                       history_.begin() + static_cast<std::ptrdiff_t>(
                                              history_.size() -
                                              static_cast<size_t>(config_.max_records_in_memory)));
    }
}

void TrainingMetricsService::update_throughput_metrics() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - session_start_steady_);

    if (elapsed.count() > 0) {
        double elapsed_seconds = static_cast<double>(elapsed.count()) / 1000.0;
        current_snapshot_.total_training_time_seconds = elapsed_seconds;
        current_snapshot_.samples_per_second = static_cast<float>(
            static_cast<float>(current_snapshot_.total_samples_trained) / elapsed_seconds);

        // Estimate time remaining
        if (current_snapshot_.total_samples > 0 && current_snapshot_.samples_per_second > 0) {
            int remaining_samples =
                current_snapshot_.total_samples * current_snapshot_.total_epochs -
                current_snapshot_.total_samples_trained;
            current_snapshot_.estimated_time_remaining_seconds =
                static_cast<float>(remaining_samples) / current_snapshot_.samples_per_second;
        }
    }
}

std::string TrainingMetricsService::escape_json(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

std::string TrainingMetricsService::format_timestamp(
    const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm = *std::localtime(&time_t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");

    // Add milliseconds
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

// ============================================================================
// HTTP Push to External Metrics API Daemon
// ============================================================================

#ifdef BUILD_METRICS_API_SERVER

std::string TrainingMetricsService::build_push_url(const std::string& endpoint) const {
    std::string url = config_.push_url;
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }

    std::string endpoint_path = endpoint;
    if (endpoint_path.empty()) {
        endpoint_path = "/";
    }
    if (endpoint_path.front() != '/') {
        endpoint_path.insert(endpoint_path.begin(), '/');
    }

    // If the caller configured a session-scoped base URL
    // (.../api/sessions/{key}), translate legacy flat endpoints to
    // per-session relative routes.
    if (url.find("/api/sessions/") != std::string::npos) {
        if (endpoint_path.rfind("/api/session/", 0) == 0) {
            endpoint_path = "/" + endpoint_path.substr(13);
        } else if (endpoint_path.rfind("/api/", 0) == 0) {
            endpoint_path = endpoint_path.substr(4);
        }
    }

    return url + endpoint_path;
}

void TrainingMetricsService::push_to_api(const std::string& endpoint,
                                         const std::string& json_body) {
    std::string request_url;
    int timeout = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!config_.enable_push) {
            return;
        }
        timeout = config_.push_timeout_ms;
        request_url = build_push_url(endpoint);
    }

    // Fire-and-forget: dispatch HTTP call onto a detached thread so the
    // training loop is never blocked by network latency or a slow/unresponsive
    // metrics server.
    std::thread([request_url, timeout, json_body]() {
        try {
            // Parse URL to extract host and port
            std::string url = request_url;
            std::string host = "localhost";
            int port = 8081;
            std::string path = "/";

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
                    path = url.substr(path_pos);
                } else {
                    port_str = url.substr(port_pos + 1);
                }
                port = std::stoi(port_str);
            } else {
                size_t path_pos = url.find('/');
                if (path_pos != std::string::npos) {
                    host = url.substr(0, path_pos);
                    path = url.substr(path_pos);
                } else {
                    host = url;
                }
            }

            httplib::Client client(host, port);
            client.set_connection_timeout(0, static_cast<long>(timeout) * 1000);
            client.set_read_timeout(timeout / 1000, static_cast<long>(timeout % 1000) * 1000);
            client.set_write_timeout(timeout / 1000, static_cast<long>(timeout % 1000) * 1000);

            auto res = client.Post(path, json_body, "application/json");

            if (!res) {
                adai::Logger::debug("Metrics push to {}: connection failed", request_url);
            } else if (res->status != 200) {
                adai::Logger::debug("Metrics push to {}: HTTP {}", request_url, res->status);
            }
        } catch (const std::exception& e) {
            adai::Logger::debug("Metrics push exception {}: {}", request_url, e.what());
        }
    }).detach();
}

#else

// Stub implementations when httplib is not available
// NOLINTBEGIN(readability-convert-member-functions-to-static)
std::string TrainingMetricsService::build_push_url(const std::string&) const {
    return "";
}
// NOLINTEND(readability-convert-member-functions-to-static)

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
        current_snapshot_.gradient_variance = gradient_variance;
        current_snapshot_.compute_time_ratio = compute_time_ratio;
        current_snapshot_.weight_update_ratio = weight_update_ratio;
        adai::Logger::debug(
            "Advanced epoch metrics: grad_var={:.4f}, compute_ratio={:.4f}, wu_ratio={:.6f}",
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

void TrainingMetricsService::update_padding_efficiency(float efficiency) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_snapshot_.current_padding_efficiency = efficiency;
    adai::Logger::debug("Batch padding efficiency: {:.4f}", efficiency);
}

// ── TD-017: Adaptive gradient clipping metrics ────────────────────────────────

void TrainingMetricsService::update_adaptive_clip_metrics(float effective_clip_threshold,
                                                          int cumulative_spike_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_snapshot_.current_adaptive_clip_threshold = effective_clip_threshold;
    current_snapshot_.current_adaptive_clip_spikes = cumulative_spike_count;
}

void TrainingMetricsService::update_adaptive_clip_epoch(float avg_clip_threshold,
                                                        int total_spike_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_snapshot_.current_adaptive_clip_threshold = avg_clip_threshold;
    current_snapshot_.current_adaptive_clip_spikes = total_spike_count;
    current_snapshot_.epoch_adaptive_clip_thresholds.push_back(avg_clip_threshold);
    adai::Logger::info("Adaptive clip epoch avg: {:.4f}  spikes: {}", avg_clip_threshold,
                       total_spike_count);
}

// ─────────────────────────────────────────────────────────────────────────────

void TrainingMetricsService::update_generation_quality_metrics(float bleu4, float rouge1,
                                                               float rouge2, float rougeL) {
    std::string push_json;
    bool should_push = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_snapshot_.current_bleu4 = bleu4;
        current_snapshot_.current_rouge1 = rouge1;
        current_snapshot_.current_rouge2 = rouge2;
        current_snapshot_.current_rougeL = rougeL;
        current_snapshot_.last_update_time = std::chrono::system_clock::now();

        adai::Logger::info(
            "Generation quality — BLEU-4: {:.4f}  ROUGE-1: {:.4f}  "
            "ROUGE-2: {:.4f}  ROUGE-L: {:.4f}",
            bleu4, rouge1, rouge2, rougeL);

        should_push = config_.enable_push;
        if (should_push) {
            std::ostringstream json;
            json << std::fixed << std::setprecision(6);
            json << "{\"bleu4\":" << bleu4 << ",\"rouge1\":" << rouge1 << ",\"rouge2\":" << rouge2
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

        adai::Logger::warn(
            "Abnormal sample flagged — epoch={} sample={} loss={:.4f} grad={:.4f} reason={}",
            sample.epoch, sample.sample_id, sample.loss, sample.grad_norm, sample.reason);

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
            file << R"("reason":")" << escape_json(s.reason) << "\",";
            file << R"("input_text":")" << escape_json(s.input_text) << "\",";
            file << R"("target_text":")" << escape_json(s.target_text) << "\",";
            file << R"("timestamp":")" << format_timestamp(s.timestamp) << "\"";
            file << "}";
            if (i + 1 < abnormal_samples_.size()) {
                file << ",";
            }
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

std::unique_ptr<MetricsSessionRegistry> GlobalMetricsService::registry_ = nullptr;
std::mutex GlobalMetricsService::instance_mutex_;

TrainingMetricsService& GlobalMetricsService::instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!registry_) {
        registry_ = std::make_unique<MetricsSessionRegistry>();
    }

    auto service = registry_->create_or_get_session("0-default");
    if (!service) {
        throw std::runtime_error(
            "GlobalMetricsService failed to provision default metrics session");
    }
    return *service;
}

void GlobalMetricsService::initialize(const MetricsServiceConfig& config) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    registry_ = std::make_unique<MetricsSessionRegistry>(config);
}

void GlobalMetricsService::shutdown() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    registry_.reset();
}
