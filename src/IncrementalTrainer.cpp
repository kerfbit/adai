#include "IncrementalTrainer.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#include <cctype>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "Config.hpp"
#include "DataFetcher.hpp"
#include "Logger.hpp"
#include "TrainingMetricsAPI.hpp"
#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#include <cmath>
#endif

// Bring Logger into scope without qualifying every call
using adai::Logger;

// Legacy ANSI codes kept for print_session_history / print_data_registry (intentional TUI output)
#define COLOR_RESET "\033[0m"
#define COLOR_INFO "\033[1;36m"
#define COLOR_SUCCESS "\033[1;32m"
#define COLOR_WARNING "\033[1;33m"
#define COLOR_ERROR "\033[1;31m"
#define COLOR_PROGRESS "\033[1;35m"

namespace fs = std::filesystem;

namespace {

std::string trim_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string sanitize_session_key(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch) || ch == '-' || ch == '_') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('-');
        }
    }
    return out;
}

std::string detect_hostname_fragment() {
    std::string host = "host";
#ifdef _WIN32
    if (const char* env_host = std::getenv("COMPUTERNAME")) {
        host = env_host;
    }
#else
    std::array<char, 256> buffer{};
    if (gethostname(buffer.data(), buffer.size() - 1) == 0) {
        host = buffer.data();
    }
#endif

    host = sanitize_session_key(host);
    if (host.empty()) {
        host = "host";
    }
    if (host.size() > 8) {
        host = host.substr(0, 8);
    }
    return host;
}

int detect_pid_mod_10000() {
#ifdef _WIN32
    return static_cast<int>(_getpid() % 10000);
#else
    return static_cast<int>(getpid() % 10000);
#endif
}

std::string derive_metrics_session_key(int session_id) {
    const std::string host = detect_hostname_fragment();
    return std::to_string(session_id) + "-" + host;
}

std::string build_metrics_session_push_base(const std::string& metrics_server_url,
                                            const std::string& session_key) {
    const std::string base = trim_trailing_slashes(metrics_server_url);
    return base + "/api/sessions/" + session_key;
}

/**
 * @brief Auto-derive a human-readable session label when none is configured.
 *
 * Format: "#{id}: {stem} ({host}, {date})"
 *   id    — numeric session ID
 *   stem  — filename stem of the model file (e.g. "chatbot_model")
 *   host  — sanitised first-8-chars of hostname
 *   date  — ISO-8601 date of the training run (YYYY-MM-DD)
 */
std::string derive_metrics_session_label(int session_id,
                                         const std::string& model_path) {
    // Stem: filename without extension
    std::string stem = fs::path(model_path).stem().string();
    if (stem.empty()) stem = "model";

    // Host (first 8 chars, sanitised)
    const std::string host = detect_hostname_fragment();

    // Date: YYYY-MM-DD
    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char date_buf[16];
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &tm_buf);

    std::ostringstream label;
    label << "#" << session_id << ": " << stem << " (" << host << ", " << date_buf << ")";
    return label.str();
}

/**
 * @brief Build a compact JSON config snapshot from IncrementalConfig.
 *
 * Only key hyper-parameters are included; this is stored on the registry
 * for display in the dashboard and Prometheus labels.
 */
std::string build_config_snapshot(const IncrementalConfig& cfg) {
    const TrainingConfig& bc = cfg.base_config;
    std::ostringstream json;
    json << std::fixed;
    json << "{"
         << "\"d_model\":" << bc.d_model
         << ",\"heads\":" << bc.num_heads
         << ",\"d_ff\":" << bc.d_ff
         << ",\"enc_layers\":" << bc.num_encoder_layers
         << ",\"dec_layers\":" << bc.num_decoder_layers
         << ",\"lr\":" << bc.learning_rate
         << ",\"batch\":" << bc.batch_size
         << ",\"grad_accum\":" << bc.gradient_accumulation_steps
         << "}";
    return json.str();
}

}  // namespace

// ============================================================================
// build_model() — THE single point for EncoderDecoderModel construction.
// Reads vocab_path_ and config.base_config to build the model.  Every code
// path that creates or recreates the model must call this method; there is no
// other place in the codebase that instantiates EncoderDecoderModel.
// ============================================================================
void IncrementalTrainer::build_model() {
    Logger::info(
        "Building model: d_model={} heads={} d_ff={} enc_layers={} dec_layers={} max_seq={}",
        config.base_config.d_model, config.base_config.num_heads, config.base_config.d_ff,
        config.base_config.num_encoder_layers, config.base_config.num_decoder_layers,
        config.base_config.max_seq_length);

    auto tok = std::make_unique<BPETokenizer>();
    tok->load_vocab(vocab_path_);
    Logger::info("Tokenizer loaded (vocab size: {})", tok->get_vocab_size());

    model = std::make_unique<EncoderDecoderModel>(
        tok->get_vocab_size(), config.base_config.d_model, config.base_config.num_encoder_layers,
        config.base_config.num_decoder_layers, config.base_config.num_heads,
        config.base_config.d_ff, config.base_config.max_seq_length);
    model->set_tokenizer(tok.release());
}

// ============================================================================
// make_incremental_config() — translate ServiceConfig → IncrementalConfig.
// ============================================================================
/*static*/
IncrementalConfig IncrementalTrainer::make_incremental_config(const adai::ServiceConfig& svc) {
    IncrementalConfig cfg;
    cfg.base_config.d_model = static_cast<int>(svc.d_model);
    cfg.base_config.num_heads = static_cast<int>(svc.num_heads);
    cfg.base_config.d_ff = static_cast<int>(svc.d_ff);
    cfg.base_config.num_encoder_layers = static_cast<int>(svc.num_encoder_layers);
    cfg.base_config.num_decoder_layers = static_cast<int>(svc.num_decoder_layers);
    cfg.base_config.max_seq_length = static_cast<int>(svc.max_seq_length);
    cfg.base_config.learning_rate = svc.learning_rate;
    cfg.base_config.num_epochs = svc.num_epochs;
    cfg.base_config.weight_decay = svc.weight_decay;
    cfg.base_config.gradient_clip_norm = svc.gradient_clip;
    // TD-017: Map adaptive gradient clipping fields
    cfg.base_config.adaptive_gradient_clip = svc.adaptive_gradient_clip;
    cfg.base_config.gradient_clip_min = svc.gradient_clip_min;
    cfg.base_config.gradient_clip_max = svc.gradient_clip_max;
    cfg.base_config.gradient_clip_ema_decay = svc.gradient_clip_ema_decay;
    cfg.base_config.gradient_clip_headroom = svc.gradient_clip_headroom;
    cfg.base_config.gradient_clip_warmup_steps = svc.gradient_clip_warmup_steps;
    cfg.base_config.gradient_clip_spike_k = svc.gradient_clip_spike_k;
    cfg.base_config.batch_size = svc.batch_size;
    cfg.base_config.enable_early_stopping = true;
    cfg.base_config.patience = 5;
    cfg.base_config.restore_best_weights = true;
    // Suppress verbose per-sample logging from ChatbotTrainer, as we use
    // the TUI dashboard for real-time feedback.
    cfg.base_config.log_level = LogLevel::NORMAL;

    // Metrics service configuration
    cfg.enable_metrics_service = svc.enable_metrics_service;
    cfg.metrics_push_enabled = svc.metrics_push_enabled;
    // TODO: See TECHNICAL_DEBT.md TD-018 - Map metrics_session_key from ServiceConfig; prefix
    //   push_url with /api/sessions/{metrics_session_key} so each trainer targets its own slot.
    cfg.metrics_server_url = svc.metrics_server_url;
    cfg.metrics_config.enable_push = svc.metrics_push_enabled;
    cfg.metrics_config.push_url = svc.metrics_server_url;
    cfg.metrics_config.push_timeout_ms = svc.metrics_push_timeout_ms;
    cfg.metrics_config.enable_persistence = svc.metrics_enable_persistence;
    cfg.metrics_config.metrics_file = svc.metrics_file;
    cfg.metrics_config.summary_file = svc.metrics_summary_file;
    cfg.metrics_config.persist_every_samples = svc.metrics_persist_every_samples;
    cfg.metrics_config.persist_every_seconds = svc.metrics_persist_every_seconds;
    cfg.metrics_config.max_records_in_memory = svc.metrics_max_records_in_memory;
    cfg.metrics_config.max_records_on_disk = svc.metrics_max_records_on_disk;
    cfg.metrics_config.enable_prometheus_format = svc.metrics_enable_prometheus;
    cfg.metrics_config.prometheus_file = svc.metrics_prometheus_file;

    // Generation quality metrics
    cfg.base_config.enable_generation_quality_metrics = svc.enable_generation_quality_metrics;
    cfg.base_config.generation_quality_sample_size = svc.generation_quality_sample_size;
    cfg.base_config.generation_quality_max_tokens = svc.generation_quality_max_tokens;
    cfg.base_config.generation_quality_async_threshold = svc.generation_quality_async_threshold;

    // Session directory
    if (!svc.session_dir.empty()) {
        cfg.session_dir = svc.session_dir;
    }

    return cfg;
}

// ============================================================================
// Primary constructor — config.conf is the required entry point.
// ============================================================================
IncrementalTrainer::IncrementalTrainer(const std::string& config_file_path)
    : current_session_id(0),
      samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path(""),
      dashboard_lines_drawn_(0),
      current_sample_in_epoch_(0),
      total_samples_in_epoch_(0),
      running_sample_loss_(0.0f),
      current_item_loss_(0.0f),
      current_item_grad_norm_(0.0f),
      current_item_lr_(0.0f) {
    Logger::info("Loading configuration from: {}",
                 config_file_path.empty() ? "<system default>" : config_file_path);

    adai::ServiceConfig svc = config_file_path.empty() ? adai::ConfigLoader::load()
                                                       : adai::ConfigLoader::load(config_file_path);

    if (svc.vocab_path.empty()) {
        throw std::runtime_error(
            "VOCAB_PATH must be set in config.conf (or via the VOCAB_PATH environment variable)");
    }

    vocab_path_ = svc.vocab_path;
    model_path_ = svc.model_path.empty() ? "chatbot_model.bin" : svc.model_path;
    config = make_incremental_config(svc);
    config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
    dataset_config_ = DatasetRegistry::make_config(svc);

    // MetricsPushClient is created per training run; use NullMetricsReporter until then.
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    ensure_directories_exist();

    load_session_history();

    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            Logger::info("Best checkpoint: {} (val loss: {})", best_checkpoint_path,
                         best_validation_loss);
            if (fs::exists(best_checkpoint_path)) {
                try {
                    model->load_model(best_checkpoint_path);
                    Logger::info("Resumed from best checkpoint: {}", best_checkpoint_path);
                } catch (...) {
                    Logger::warn("Could not load best checkpoint, keeping current model weights");
                }
            }
        }
    }

    // Check for an in-progress best snapshot from an interrupted session.
    // If session_{current_session_id}_best.bin exists and is newer than the
    // completed-run checkpoint we just loaded, prefer it — it represents the
    // best weights found before the interruption.
    {
        std::ostringstream in_progress_oss;
        in_progress_oss << get_session_dir() << "/session_" << current_session_id << "_best.bin";
        std::string in_progress_best = in_progress_oss.str();

        if (fs::exists(in_progress_best + ".config")) {
            bool prefer_in_progress = best_checkpoint_path.empty();

            if (!prefer_in_progress && fs::exists(best_checkpoint_path + ".config")) {
                auto best_mtime = fs::last_write_time(best_checkpoint_path + ".config");
                auto in_prog_mtime = fs::last_write_time(in_progress_best + ".config");
                prefer_in_progress = (in_prog_mtime > best_mtime);
            }

            if (prefer_in_progress) {
                try {
                    model->load_model(in_progress_best);
                    Logger::info("Resumed from in-progress best snapshot: {}", in_progress_best);
                } catch (...) {
                    Logger::warn(
                        "Could not load in-progress best snapshot, keeping previous weights");
                }
            }
        }
    }

    last_save_time = std::chrono::system_clock::now();
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_ = session_start_time_steady_;
}

// ============================================================================
// Explicit-paths constructor (low-level, uses default IncrementalConfig).
// ============================================================================
IncrementalTrainer::IncrementalTrainer(std::string vocab_path, const std::string& model_path)
    : model_path_(model_path),
      vocab_path_(std::move(vocab_path)),
      current_session_id(0),
      samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path(""),
      dashboard_lines_drawn_(0),
      current_sample_in_epoch_(0),
      total_samples_in_epoch_(0),
      running_sample_loss_(0.0f),
      current_item_loss_(0.0f),
      current_item_grad_norm_(0.0f),
      current_item_lr_(0.0f) {
    Logger::info("Initializing Incremental Training System...");

    // MetricsPushClient is created per training run; use NullMetricsReporter until then.
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    // Build model using default config.
    build_model();

    ensure_directories_exist();
    load_session_history();

    // Initialize best checkpoint from history (TD-005)
    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            Logger::info("Best checkpoint: {} (val loss: {})", best_checkpoint_path,
                         best_validation_loss);
            if (fs::exists(best_checkpoint_path)) {
                try {
                    model->load_model(best_checkpoint_path);
                    Logger::info("Resumed from best checkpoint: {}", best_checkpoint_path);
                } catch (...) {
                    Logger::warn("Could not load best checkpoint, keeping current model weights");
                }
            }
        }
    }

    last_save_time = std::chrono::system_clock::now();
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_ = session_start_time_steady_;
}

// ============================================================================
// Explicit-paths + pre-built config constructor.
// Applies config BEFORE building the model so no defaults are baked in.
// ============================================================================
IncrementalTrainer::IncrementalTrainer(std::string vocab_path, const std::string& model_path,
                                       IncrementalConfig cfg)
    : config(std::move(cfg)),
      model_path_(model_path),
      vocab_path_(std::move(vocab_path)),
      current_session_id(0),
      samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path(""),
      dashboard_lines_drawn_(0),
      current_sample_in_epoch_(0),
      total_samples_in_epoch_(0),
      running_sample_loss_(0.0f),
      current_item_loss_(0.0f),
      current_item_grad_norm_(0.0f),
      current_item_lr_(0.0f) {
    Logger::info("Initializing Incremental Training System...");

    // set config FIRST so build_model() uses the correct architecture

    // MetricsPushClient is created per training run; use NullMetricsReporter until then.
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    build_model();

    ensure_directories_exist();
    load_session_history();

    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            Logger::info("Best checkpoint: {} (val loss: {})", best_checkpoint_path,
                         best_validation_loss);
            if (fs::exists(best_checkpoint_path)) {
                try {
                    model->load_model(best_checkpoint_path);
                    Logger::info("Resumed from best checkpoint: {}", best_checkpoint_path);
                } catch (...) {
                    Logger::warn("Could not load best checkpoint, keeping current model weights");
                }
            }
        }
    }

    last_save_time = std::chrono::system_clock::now();
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_ = session_start_time_steady_;
}

void IncrementalTrainer::set_config(const IncrementalConfig& cfg) {
    config = cfg;
}

IncrementalConfig& IncrementalTrainer::get_config() {
    return config;
}

void IncrementalTrainer::reset_model_for_config() {
    // Delegate entirely to build_model() — the single construction point.
    build_model();
    Logger::info("Model reinitialized with fresh weights (architecture reset)");
}

// TODO(TD-028): Move add_new_data, add_new_data_batch, clear_pending_data,
// get_pending_data_files, get_trained_data_files to DatasetRegistry
bool IncrementalTrainer::add_new_data(const std::string& data_file) {
    if (!fs::exists(data_file)) {
        Logger::error("Data file not found: {}", data_file);
        return false;
    }

    // Check if already trained
    if (is_data_trained(data_file)) {
        Logger::warn("Data file already trained, skipping: {}", data_file);
        return false;
    }

    pending_data_files.push_back(data_file);
    Logger::info("Added new data file: {}", data_file);

    // Save pending files list
    save_pending_data_list();

    return true;
}

bool IncrementalTrainer::add_new_data_batch(const std::vector<std::string>& data_files) {
    int added = 0;
    for (const auto& file : data_files) {
        if (add_new_data(file)) {
            added++;
        }
    }
    Logger::info("Added {}/{} new data files", added, data_files.size());
    return added > 0;
}

void IncrementalTrainer::clear_pending_data() {
    pending_data_files.clear();
}

std::vector<std::string> IncrementalTrainer::get_pending_data_files() const {
    return pending_data_files;
}

std::vector<std::string> IncrementalTrainer::get_trained_data_files() const {
    return std::vector<std::string>(trained_data_files.begin(), trained_data_files.end());
}


bool IncrementalTrainer::train_on_files(const std::vector<std::string>& files, int num_epochs) {
    Logger::info("Starting Incremental Training Session #{}", current_session_id + 1);
    Logger::info("Files to train: {}", files.size());

    initialize_session();

    // Reset dashboard state and record session start time (TD-009)
    dashboard_lines_drawn_ = 0;
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_ = session_start_time_steady_;

    // Load all data — files are independent so load them in parallel
    std::vector<ConversationPair> training_pairs;
    std::vector<ConversationPair> validation_pairs;

    {
        const int n_files = static_cast<int>(files.size());
        std::vector<std::vector<ConversationPair>> per_file(n_files);

#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int fi = 0; fi < n_files; ++fi) {
            load_conversation_pairs(files[fi], per_file[fi]);
        }

        // Sequential: split, collect pairs, update data registry
        for (int fi = 0; fi < n_files; ++fi) {
            const auto& data_file = files[fi];
            auto& pairs = per_file[fi];
            int loaded = static_cast<int>(pairs.size());
            if (loaded > 0) {
                int val_size = loaded / config.base_config.validation_split;
                for (int i = 0; i < loaded; ++i) {
                    if (i < val_size) {
                        validation_pairs.push_back(pairs[i]);
                    } else {
                        training_pairs.push_back(pairs[i]);
                    }
                }

                DataVersion dv;
                dv.data_file = data_file;
                dv.checksum = compute_data_checksum(data_file);
                dv.num_samples = loaded;
                dv.added_time = std::chrono::system_clock::now();
                dv.trained = true;
                data_registry.push_back(dv);
                trained_data_files.insert(data_file);
            }
        }
    }

    Logger::info("Total training samples: {}", training_pairs.size());
    Logger::info("Total validation samples: {}", validation_pairs.size());

    // Store total sample count for dashboard and reset per-sample state
    total_samples_in_epoch_ = static_cast<int>(training_pairs.size());
    current_sample_in_epoch_ = 0;
    running_sample_loss_ = 0.0f;

    // Create trainer and configure
    ChatbotTrainer trainer(config.base_config);
    trainer.set_model(std::move(model));

    // TD-021: Create MetricsPushClient for this session, or keep NullMetricsReporter.
    // Retry up to 3 times with a suffix if the server returns 409 Conflict.
    if (!config.metrics_server_url.empty()) {
        const std::string base_key =
            sanitize_session_key(derive_metrics_session_key(current_session_id + 1));
        std::string session_key = base_key;
        std::unique_ptr<MetricsPushClient> pc;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (attempt > 0) {
                session_key = base_key + "-" + std::to_string(attempt + 1);
                Logger::warn("Metrics key conflict (409), retrying with '{}'", session_key);
                std::this_thread::sleep_for(std::chrono::milliseconds(100 * attempt));
            }
            const std::string push_url =
                build_metrics_session_push_base(config.metrics_server_url, session_key);
            pc = std::make_unique<MetricsPushClient>(push_url, config.metrics_push_timeout_ms);
            const std::string label =
                config.metrics_session_label.empty()
                    ? derive_metrics_session_label(current_session_id + 1, model_path_)
                    : config.metrics_session_label;
            const std::string snapshot = build_config_snapshot(config);
            const int rc = pc->start_session(current_session_id + 1, num_epochs,
                                             static_cast<int>(training_pairs.size()),
                                             label, snapshot);
            if (rc != 409) break;
        }
        active_session_key_ = session_key;
        push_client_ = pc.get();
        metrics_reporter_ = std::move(pc);
    } else {
        push_client_ = nullptr;
        active_session_key_.clear();
        metrics_reporter_ = std::make_unique<NullMetricsReporter>();
    }
    trainer.set_metrics_reporter(metrics_reporter_.get());

    for (const auto& pair : training_pairs) {
        trainer.add_training_pair(pair.input, pair.response);
    }

    // ── TD-009: Register per-epoch callback for timing, metrics, and live dashboard ──
    trainer.set_epoch_callback([this](int epoch, int total, float loss, float val_loss, float lr) {
        // Measure wall-clock time for this epoch
        auto now = std::chrono::steady_clock::now();
        double epoch_secs = std::chrono::duration<double>(now - epoch_start_time_steady_).count();
        epoch_start_time_steady_ = now;  // reset for next epoch

        // Store per-epoch metrics in the current session (session_history.back())
        if (!session_history.empty()) {
            auto& session = session_history.back();
            session.per_epoch_losses.push_back(loss);
            session.per_epoch_validation_losses.push_back(val_loss);
            session.per_epoch_learning_rates.push_back(lr);
            session.training_time_per_epoch.push_back(epoch_secs);
            session.per_epoch_perplexities.push_back(std::exp(loss));
            session.per_epoch_validation_perplexities.push_back(std::exp(val_loss));


            // Reset per-sample counters for the next epoch
            current_sample_in_epoch_ = 0;
            running_sample_loss_ = 0.0f;
            current_item_loss_ = 0.0f;
            current_item_grad_norm_ = 0.0f;
        }
    });

    // ── Per-sample callback: update running state and redraw dashboard ──
    trainer.set_sample_callback([this](int sample, int total_samples, float running_loss,
                                       float step_loss, float grad_norm, float lr) {
        // Reset the epoch timer on the very first sample so that throughput
        // excludes preprocessing (split + tokenization) time.
        if (sample == 1) {
            epoch_start_time_steady_ = std::chrono::steady_clock::now();
        }
    });

    // ── Best-model callback: persist a named snapshot every time validation improves ──
    {
        std::ostringstream best_path_oss;
        best_path_oss << get_session_dir() << "/session_" << current_session_id << "_best.bin";
        std::string best_snapshot_path = best_path_oss.str();

        trainer.set_best_model_callback(
            [this, &trainer, best_snapshot_path](int epoch, float val_loss) {
                try {
                    trainer.save_to(best_snapshot_path);
                    Logger::info("  [best] New best (epoch {}, val_loss {:.4f}) saved to: {}",
                                 epoch, val_loss, best_snapshot_path);
                } catch (const std::exception& e) {
                    Logger::warn("  [warn] Failed to save best snapshot: {}", e.what());
                }
            });
    }

    Logger::info("Training for {} epochs...", num_epochs);
    bool success = trainer.train(num_epochs);

    // End metrics push session and reset to null reporter until the next run.
    if (push_client_) {
        push_client_->end_session();
        push_client_ = nullptr;
    }
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    // Retrieve model after training
    model = trainer.release_model();

    if (success) {
        // Save checkpoint
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);

        // Finalize session (per-epoch vectors already populated by callback)
        float final_loss = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(static_cast<int>(training_pairs.size()), num_epochs, final_loss,
                         final_val_loss);

        save_session_history();
        Logger::info("Incremental training session completed successfully");
        print_training_summary();
    }

    return success;
}


bool IncrementalTrainer::retrain_on_files(const std::vector<std::string>& files, int num_epochs) {
    Logger::info("Starting full retrain on {} file(s)", files.size());
    initialize_session();

    // Reset dashboard state (TD-009)
    dashboard_lines_drawn_ = 0;
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_ = session_start_time_steady_;

    // Load all data — files are independent so load them in parallel
    std::vector<ConversationPair> all_pairs;
    {
        const int n_files = static_cast<int>(files.size());
        std::vector<std::vector<ConversationPair>> per_file(n_files);

#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int fi = 0; fi < n_files; ++fi) {
            load_conversation_pairs(files[fi], per_file[fi]);
        }

        // Merge in order so dataset ordering is deterministic
        for (int fi = 0; fi < n_files; ++fi) {
            all_pairs.insert(all_pairs.end(), per_file[fi].begin(), per_file[fi].end());
        }
    }

    Logger::info("Total samples: {}", all_pairs.size());

    // Store total sample count for dashboard and reset per-sample state
    // (validation split is done inside ChatbotTrainer, approximate here)
    int val_size = static_cast<int>(all_pairs.size()) / config.base_config.validation_split;
    total_samples_in_epoch_ = static_cast<int>(all_pairs.size()) - val_size;
    current_sample_in_epoch_ = 0;
    running_sample_loss_ = 0.0f;

    // Create trainer
    ChatbotTrainer trainer(config.base_config);
    trainer.set_model(std::move(model));

    // TD-021: Create MetricsPushClient for this session, or keep NullMetricsReporter.
    // Retry up to 3 times with a suffix if the server returns 409 Conflict.
    if (!config.metrics_server_url.empty()) {
        const std::string base_key =
            sanitize_session_key(derive_metrics_session_key(current_session_id + 1));
        std::string session_key = base_key;
        std::unique_ptr<MetricsPushClient> pc;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (attempt > 0) {
                session_key = base_key + "-" + std::to_string(attempt + 1);
                Logger::warn("Metrics key conflict (409), retrying with '{}'", session_key);
                std::this_thread::sleep_for(std::chrono::milliseconds(100 * attempt));
            }
            const std::string push_url =
                build_metrics_session_push_base(config.metrics_server_url, session_key);
            pc = std::make_unique<MetricsPushClient>(push_url, config.metrics_push_timeout_ms);
            const std::string label =
                config.metrics_session_label.empty()
                    ? derive_metrics_session_label(current_session_id + 1, model_path_)
                    : config.metrics_session_label;
            const std::string snapshot = build_config_snapshot(config);
            const int rc = pc->start_session(current_session_id + 1, num_epochs,
                                             total_samples_in_epoch_,
                                             label, snapshot);
            if (rc != 409) break;
        }
        active_session_key_ = session_key;
        push_client_ = pc.get();
        metrics_reporter_ = std::move(pc);
    } else {
        push_client_ = nullptr;
        active_session_key_.clear();
        metrics_reporter_ = std::make_unique<NullMetricsReporter>();
    }
    trainer.set_metrics_reporter(metrics_reporter_.get());

    for (const auto& pair : all_pairs) {
        trainer.add_training_pair(pair.input, pair.response);
    }

    // TD-009: Register per-epoch callback
    trainer.set_epoch_callback([this](int epoch, int total, float loss, float val_loss, float lr) {
        auto now = std::chrono::steady_clock::now();
        double epoch_secs = std::chrono::duration<double>(now - epoch_start_time_steady_).count();
        epoch_start_time_steady_ = now;
        if (!session_history.empty()) {
            auto& session = session_history.back();
            session.per_epoch_losses.push_back(loss);
            session.per_epoch_validation_losses.push_back(val_loss);
            session.per_epoch_learning_rates.push_back(lr);
            session.training_time_per_epoch.push_back(epoch_secs);
            session.per_epoch_perplexities.push_back(std::exp(loss));
            session.per_epoch_validation_perplexities.push_back(std::exp(val_loss));

            // Reset per-sample counters for the next epoch
            current_sample_in_epoch_ = 0;
            running_sample_loss_ = 0.0f;
            current_item_loss_ = 0.0f;
            current_item_grad_norm_ = 0.0f;
        }
    });

    // ── Per-sample callback for full retrain ──
    trainer.set_sample_callback([this](int sample, int total_samples, float running_loss,
                                       float step_loss, float grad_norm, float lr) {
        // Reset the epoch timer on the very first sample so that throughput
        // excludes preprocessing (split + tokenization) time.
        if (sample == 1) {
            epoch_start_time_steady_ = std::chrono::steady_clock::now();
        }
        current_sample_in_epoch_ = sample;
        total_samples_in_epoch_ = total_samples;
        running_sample_loss_ = running_loss;
        current_item_loss_ = step_loss;
        current_item_grad_norm_ = grad_norm;
        current_item_lr_ = lr;
    });

    bool success = trainer.train(num_epochs);
    model = trainer.release_model();

    // End metrics push session and reset to null reporter until the next run.
    if (push_client_) {
        push_client_->end_session();
        push_client_ = nullptr;
    }
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    if (success) {
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);

        float final_loss = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(static_cast<int>(all_pairs.size()), num_epochs, final_loss,
                         final_val_loss);

        save_session_history();
    }

    return success;
}

bool IncrementalTrainer::resume_last_session() {
    DatasetRegistry reg(dataset_config_);
    std::string run_id = dataset_config_.run_id;
    if (run_id.empty()) {
        run_id = detect_hostname_fragment() + "_" + std::to_string(detect_pid_mod_10000());
    }
    auto pending = reg.acquire_pending(run_id);
    if (pending.empty()) {
        Logger::warn("No pending data files to resume training on");
        return false;
    }

    std::string resume_checkpoint = best_checkpoint_path;
    int resume_session_id = -1;

    if (!session_history.empty()) {
        resume_session_id = session_history.back().session_id;
    }

    std::ostringstream in_progress_oss;
    in_progress_oss << get_session_dir() << "/session_" << current_session_id << "_best.bin";
    const std::string in_progress_best = in_progress_oss.str();

    if (fs::exists(in_progress_best + ".config")) {
        resume_checkpoint = in_progress_best;
    } else if (resume_checkpoint.empty() && !session_history.empty()) {
        resume_checkpoint = session_history.back().checkpoint_path;
    }

    if (resume_checkpoint.empty()) {
        Logger::warn(
            "No resumable checkpoint metadata found; continuing from currently loaded model "
            "weights");
    } else {
        // Check if checkpoint files exist (check for .config which should always be present)
        if (!fs::exists(resume_checkpoint + ".config")) {
            Logger::error("Checkpoint file not found: {}.config", resume_checkpoint);
            return false;
        }

        if (resume_session_id >= 0) {
            Logger::info("Resuming training from session #{} using checkpoint {}",
                         resume_session_id, resume_checkpoint);
        } else {
            Logger::info("Resuming training using checkpoint {}", resume_checkpoint);
        }

        if (!load_model(resume_checkpoint)) {
            return false;
        }
    }

    bool ok = train_on_files(pending, config.base_config.num_epochs);
    if (ok) {
        std::vector<int> counts;
        counts.reserve(pending.size());
        for (const auto& path : pending) {
            int n = 0;
            for (const auto& dv : data_registry) {
                if (dv.data_file == path) { n = dv.num_samples; break; }
            }
            counts.push_back(n);
        }
        reg.mark_trained(run_id, pending, counts);
    } else {
        reg.release_pending(run_id, pending);
    }
    return ok;
}

bool IncrementalTrainer::load_session_history() {
    std::string history_file = get_session_dir() + "/session_history.txt";

    if (!fs::exists(history_file)) {
        return false;
    }

    std::ifstream file(history_file);
    if (!file.is_open()) {
        return false;
    }

    session_history.clear();
    std::string line;

    auto parse_float_list = [](const std::string& s, std::vector<float>& out) {
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                out.push_back(std::stof(token));
            } catch (...) {
                continue;  // skip malformed token
            }
        }
    };
    auto parse_double_list = [](const std::string& s, std::vector<double>& out) {
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                out.push_back(std::stod(token));
            } catch (...) {
                continue;  // skip malformed token
            }
        }
    };

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        TrainingSession session;

        iss >> session.session_id >> session.samples_trained >> session.epochs_completed >>
            session.final_loss >> session.final_validation_loss >> session.checkpoint_path;

        if (iss.fail() || session.checkpoint_path.empty()) {
            Logger::warn("Skipping malformed session history line: {}", line);
            continue;
        }

        // v2 extended format: checkpoint_path may contain
        // "|losses:...|vallosses:...|lrs:...|times:..."
        size_t pipe = session.checkpoint_path.find('|');
        if (pipe != std::string::npos) {
            std::string extended = session.checkpoint_path.substr(pipe + 1);
            session.checkpoint_path = session.checkpoint_path.substr(0, pipe);

            std::istringstream es(extended);
            std::string segment;
            while (std::getline(es, segment, '|')) {
                auto colon = segment.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                std::string key = segment.substr(0, colon);
                std::string value = segment.substr(colon + 1);
                if (key == "losses") {
                    parse_float_list(value, session.per_epoch_losses);
                } else if (key == "vallosses") {
                    parse_float_list(value, session.per_epoch_validation_losses);
                } else if (key == "lrs") {
                    parse_float_list(value, session.per_epoch_learning_rates);
                } else if (key == "times") {
                    parse_double_list(value, session.training_time_per_epoch);
                }
            }
        }

        session_history.push_back(session);

        if (session.session_id >= current_session_id) {
            current_session_id = session.session_id + 1;
        }
    }

    Logger::info("Loaded {} previous sessions", session_history.size());
    return true;
}

bool IncrementalTrainer::save_session_history() {
    std::string history_file = get_session_dir() + "/session_history.txt";

    std::ofstream file(history_file);
    if (!file.is_open()) {
        Logger::error("Failed to save session history");
        return false;
    }

    file << "# VERSION 2\n";
    file << "# session_id samples_trained epochs final_loss final_val_loss "
            "checkpoint_path[|losses:...|vallosses:...|lrs:...|times:...]\n";

    auto join_floats = [](const std::vector<float>& v) -> std::string {
        std::ostringstream oss;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) {
                oss << ',';
            }
            oss << v[i];
        }
        return oss.str();
    };
    auto join_doubles = [](const std::vector<double>& v) -> std::string {
        std::ostringstream oss;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) {
                oss << ',';
            }
            oss << v[i];
        }
        return oss.str();
    };

    for (const auto& session : session_history) {
        file << session.session_id << " " << session.samples_trained << " "
             << session.epochs_completed << " " << session.final_loss << " "
             << session.final_validation_loss << " " << session.checkpoint_path;
        if (!session.per_epoch_losses.empty()) {
            file << "|losses:" << join_floats(session.per_epoch_losses)
                 << "|vallosses:" << join_floats(session.per_epoch_validation_losses)
                 << "|lrs:" << join_floats(session.per_epoch_learning_rates)
                 << "|times:" << join_doubles(session.training_time_per_epoch);
        }
        file << "\n";
    }

    return true;
}

TrainingSession IncrementalTrainer::get_current_session() {
    if (session_history.empty()) {
        return {};
    }
    return session_history.back();
}

std::vector<TrainingSession> IncrementalTrainer::get_session_history() const {
    return session_history;
}

void IncrementalTrainer::remove_model_files(const std::string& base_path) {
    static const std::vector<std::string> sidecars = {".config", ".vocab", ".encoder", ".decoder",
                                                      ".lm_head"};
    std::error_code ec;
    // Remove bare base file (empty marker written by finalize_model)
    if (fs::exists(base_path, ec)) {
        fs::remove(base_path, ec);
    }
    for (const auto& ext : sidecars) {
        std::string p = base_path + ext;
        if (fs::exists(p, ec)) {
            fs::remove(p, ec);
            if (!ec) {
                Logger::info("Removed: {}", p);
            }
        }
    }
}

void IncrementalTrainer::cleanup_old_sessions() {
    if (session_history.size() <= config.max_sessions_to_keep) {
        return;
    }

    int to_remove = static_cast<int>(session_history.size()) - config.max_sessions_to_keep;

    for (int i = 0; i < to_remove; ++i) {
        const auto& session = session_history[i];

        // TD-005: Check if we're deleting the best checkpoint
        bool deleting_best = (session.checkpoint_path == best_checkpoint_path);

        // Delete checkpoint and all sidecar files
        remove_model_files(session.checkpoint_path);
        Logger::info("Removed old checkpoint: {}", session.checkpoint_path);

        // Also remove any in-progress best snapshot for this session
        // (it is superseded by the checkpoint and no subsequent session needs it)
        {
            // Derive the session id from the checkpoint path (session_N_checkpoint.bin)
            std::string cp = session.checkpoint_path;
            std::string stem = fs::path(cp).stem().string();  // "session_N_checkpoint"
            // Replace "_checkpoint" suffix with "_best"
            const std::string tag = "_checkpoint";
            auto pos = stem.rfind(tag);
            if (pos != std::string::npos) {
                std::string best_stem = stem.substr(0, pos) + "_best";
                std::string best_path =
                    fs::path(cp).parent_path().string() + "/" + best_stem + ".bin";
                remove_model_files(best_path);
            }
        }

        // If we deleted the best checkpoint, find the next best from remaining sessions
        if (deleting_best && config.enable_checkpoint_symlinks) {
            best_validation_loss = std::numeric_limits<float>::max();
            best_checkpoint_path = "";

            // Search through remaining sessions (after the ones we're removing)
            for (size_t j = to_remove; j < session_history.size(); ++j) {
                const auto& remaining_session = session_history[j];
                if (remaining_session.final_validation_loss < best_validation_loss &&
                    fs::exists(remaining_session.checkpoint_path)) {
                    best_validation_loss = remaining_session.final_validation_loss;
                    best_checkpoint_path = remaining_session.checkpoint_path;
                }
            }

            // Update the best_checkpoint symlink to new best (or remove if no sessions remain)
            if (!best_checkpoint_path.empty()) {
                update_best_checkpoint(best_validation_loss, best_checkpoint_path);
            } else {
                // No valid checkpoints remain, remove the symlink
                remove_symlink_if_exists(config.best_symlink_name);
            }
        }
    }

    // Remove from history
    session_history.erase(session_history.begin(), session_history.begin() + to_remove);

    // TD-005: Update latest checkpoint symlink to point to the newest remaining session
    if (config.enable_checkpoint_symlinks && !session_history.empty()) {
        const auto& latest_session = session_history.back();
        if (fs::exists(latest_session.checkpoint_path)) {
            update_checkpoint_symlinks(latest_session.checkpoint_path);
        }
    }
}

// TODO(TD-028): Move to DatasetRegistry::load_registry()
bool IncrementalTrainer::load_data_registry() {
    std::string registry_file = get_session_dir() + "/" + dataset_config_.data_registry_file;

    if (!fs::exists(registry_file)) {
        return false;
    }

    std::ifstream file(registry_file);
    if (!file.is_open()) {
        return false;
    }

    data_registry.clear();
    trained_data_files.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        DataVersion dv;
        int trained_int = 0;

        iss >> dv.data_file >> dv.checksum >> dv.num_samples >> trained_int;
        dv.trained = (trained_int == 1);

        data_registry.push_back(dv);

        if (dv.trained) {
            trained_data_files.insert(dv.data_file);
        }
    }

    Logger::info("Loaded data registry: {} files ({} trained)", data_registry.size(),
                 trained_data_files.size());

    return true;
}

// TODO(TD-028): Move to DatasetRegistry::save_registry()
bool IncrementalTrainer::save_data_registry() {
    std::string registry_file = get_session_dir() + "/" + dataset_config_.data_registry_file;

    std::ofstream file(registry_file);
    if (!file.is_open()) {
        Logger::error("Failed to save data registry");
        return false;
    }

    file << "# Data Registry: data_file checksum num_samples trained\n";

    for (const auto& dv : data_registry) {
        file << dv.data_file << " " << dv.checksum << " " << dv.num_samples << " "
             << (dv.trained ? 1 : 0) << "\n";
    }

    return true;
}

bool IncrementalTrainer::is_data_trained(const std::string& data_file) {
    return trained_data_files.find(data_file) != trained_data_files.end();
}

// TODO(TD-028): Move to DatasetRegistry::compute_checksum()
std::string IncrementalTrainer::compute_data_checksum(const std::string& data_file) {
    // Simple checksum: file size + modification time
    if (!fs::exists(data_file)) {
        return "MISSING";
    }

    auto size = fs::file_size(data_file);
    auto ftime = fs::last_write_time(data_file);

    std::ostringstream oss;
    // file_time_type::duration::rep is __int128 on macOS (libc++) which has no
    // operator<< overload — cast to long long to keep it portable.
    oss << size << "_" << static_cast<long long>(ftime.time_since_epoch().count());

    return oss.str();
}

bool IncrementalTrainer::save_model(const std::string& path) {
    try {
        model->save_model(path);
        Logger::info("Model saved to: {}", path);
        return true;
    } catch (const std::exception& e) {
        Logger::error("Failed to save model: {}", e.what());
        return false;
    }
}

bool IncrementalTrainer::load_model(const std::string& path) {
    try {
        model->load_model(path);
        Logger::info("Model loaded from: {}", path);
        return true;
    } catch (const std::exception& e) {
        Logger::error("Failed to load model: {}", e.what());
        return false;
    }
}

std::string IncrementalTrainer::get_latest_checkpoint() {
    if (session_history.empty()) {
        return "";
    }
    return session_history.back().checkpoint_path;
}

// Returns a sparkline string visualising values (low = good for loss)
static std::string make_sparkline(const std::vector<float>& values, int width = 30) {
    if (values.empty()) {
        return std::string(width, '-');
    }
    static const std::array<const char*, 8> bars = {"\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83",
                                                    "\xe2\x96\x84", "\xe2\x96\x85", "\xe2\x96\x86",
                                                    "\xe2\x96\x87", "\xe2\x96\x88"};
    float mn = *std::min_element(values.begin(), values.end());
    float mx = *std::max_element(values.begin(), values.end());
    float range = mx - mn;
    std::string result;
    // sample at most `width` points
    int step = std::max(1, (int)values.size() / width);
    for (int i = 0; i < (int)values.size(); i += step) {
        int idx = (range < 1e-9f) ? 0 : (int)(7.0f * (values[i] - mn) / range);
        idx = std::clamp(idx, 0, 7);
        result += bars[idx];
    }
    return result;
}

void IncrementalTrainer::print_training_summary() const {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║       Incremental Training Summary           ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n";

    std::cout << "  Sessions       : " << session_history.size() << "\n";
    std::cout << "  Data files used: " << trained_data_files.size() << "\n";
    std::cout << "  Total samples  : " << get_total_samples_trained() << "\n";
    std::cout << "  Total time     : " << std::fixed << std::setprecision(2)
              << get_total_training_time_hours() << " h\n";

    // TD-005: Checkpoint symlink information
    if (config.enable_checkpoint_symlinks) {
        std::cout << "\n  Checkpoint links:\n";
        if (fs::exists(config.latest_symlink_name)) {
            std::cout << "    latest: " << config.latest_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.latest_symlink_name)) {
                std::cout << " -> " << fs::read_symlink(config.latest_symlink_name).string();
            }
            std::cout << "\n";
        }
        if (fs::exists(config.best_symlink_name)) {
            std::cout << "    best  : " << config.best_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.best_symlink_name)) {
                std::cout << " -> " << fs::read_symlink(config.best_symlink_name).string();
            }
            std::cout << "  (val loss: " << std::fixed << std::setprecision(4)
                      << best_validation_loss << ")\n";
        }
    }

    // Per-session details with sparklines
    for (const auto& s : session_history) {
        std::cout << "\n  Session #" << s.session_id << "  samples=" << s.samples_trained
                  << "  epochs=" << s.epochs_completed << "  loss=" << std::fixed
                  << std::setprecision(4) << s.final_loss << "  val=" << s.final_validation_loss
                  << "\n";
        std::cout << "    checkpoint: " << s.checkpoint_path << "\n";

        if (!s.per_epoch_losses.empty()) {
            float best_val = std::numeric_limits<float>::max();
            double total_t = 0.0;
            for (float v : s.per_epoch_validation_losses) {
                best_val = std::min(best_val, v);
            }
            for (double t : s.training_time_per_epoch) {
                total_t += t;
            }

            std::cout << "    loss      : " << make_sparkline(s.per_epoch_losses) << "\n";
            std::cout << "    val loss  : " << make_sparkline(s.per_epoch_validation_losses)
                      << "\n";
            std::cout << "    best val  : " << std::fixed << std::setprecision(4) << best_val
                      << "\n";
            if (!s.training_time_per_epoch.empty()) {
                std::cout << "    epoch time: avg " << std::fixed << std::setprecision(1)
                          << (total_t / static_cast<double>(s.training_time_per_epoch.size()))
                          << "s" << "  total " << format_duration(total_t) << "\n";
            }
        }
    }

    std::cout << "\n";
}

void IncrementalTrainer::print_session_history() {
    std::cout << COLOR_INFO << "\n📜 Session History:" << COLOR_RESET << '\n';
    std::cout << "Session | Samples | Epochs | Loss   | Val Loss | Checkpoint" << '\n';
    std::cout << "--------|---------|--------|--------|----------|------------" << '\n';

    for (const auto& session : session_history) {
        std::cout << std::setw(7) << session.session_id << " | " << std::setw(7)
                  << session.samples_trained << " | " << std::setw(6) << session.epochs_completed
                  << " | " << std::setw(6) << std::fixed << std::setprecision(3)
                  << session.final_loss << " | " << std::setw(8) << session.final_validation_loss
                  << " | " << session.checkpoint_path << '\n';
    }
}

void IncrementalTrainer::print_data_registry() {
    std::cout << COLOR_INFO << "\n📋 Data Registry:" << COLOR_RESET << '\n';
    std::cout << "Trained | Samples | Data File" << '\n';
    std::cout << "--------|---------|----------" << '\n';

    for (const auto& dv : data_registry) {
        std::cout << std::setw(7) << (dv.trained ? "✓" : " ") << " | " << std::setw(7)
                  << dv.num_samples << " | " << dv.data_file << '\n';
    }
}

float IncrementalTrainer::get_total_training_time_hours() const {
    float total_hours = 0.0f;
    for (const auto& session : session_history) {
        auto duration =
            std::chrono::duration_cast<std::chrono::hours>(session.end_time - session.start_time);
        total_hours += static_cast<float>(duration.count());
    }
    return total_hours;
}

int IncrementalTrainer::get_total_samples_trained() const {
    int total = 0;
    for (const auto& dv : data_registry) {
        if (dv.trained) {
            total += dv.num_samples;
        }
    }
    return total;
}

bool IncrementalTrainer::initialize_session() {
    TrainingSession session;
    session.session_id = current_session_id;
    session.start_time = std::chrono::system_clock::now();
    session.samples_trained = 0;
    session.epochs_completed = 0;
    session.final_loss = 0.0f;
    session.final_validation_loss = 0.0f;

    session_history.push_back(session);
    last_save_time = std::chrono::system_clock::now();
    samples_since_last_save = 0;

    return true;
}

// See TD-005 in TECHNICAL_DEBT.md - Checkpoint management and symbolic links
bool IncrementalTrainer::finalize_session(int samples_trained, int epochs_completed,
                                          float final_loss, float final_val_loss) {
    if (session_history.empty()) {
        return false;
    }

    auto& session = session_history.back();
    session.end_time = std::chrono::system_clock::now();
    session.samples_trained = samples_trained;
    session.epochs_completed = epochs_completed;
    session.final_loss = final_loss;
    session.final_validation_loss = final_val_loss;
    session.checkpoint_path = generate_session_checkpoint_path();
    // Per-epoch metrics are accumulated live via the epoch callback in train_on_files()

    // Remove the in-progress best snapshot — it is now superseded by the finalized checkpoint
    {
        std::string cp = session.checkpoint_path;
        std::string stem = fs::path(cp).stem().string();  // "session_N_checkpoint"
        const std::string tag = "_checkpoint";
        auto pos = stem.rfind(tag);
        if (pos != std::string::npos) {
            std::string best_stem = stem.substr(0, pos) + "_best";
            std::string best_path = fs::path(cp).parent_path().string() + "/" + best_stem + ".bin";
            remove_model_files(best_path);
            Logger::info("Removed superseded best snapshot: {}", best_path);
        }
    }

    // TD-005: Checkpoint symlink management
    update_checkpoint_symlinks(session.checkpoint_path);
    update_best_checkpoint(final_val_loss, session.checkpoint_path);

    current_session_id++;

    // Cleanup old sessions
    cleanup_old_sessions();

    return true;
}

bool IncrementalTrainer::should_auto_save() {
    if (!config.auto_save_enabled) {
        return false;
    }

    // Check sample count
    if (config.auto_save_every_samples > 0 &&
        samples_since_last_save >= config.auto_save_every_samples) {
        return true;
    }

    // Check time elapsed
    if (config.auto_save_every_minutes > 0) {
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - last_save_time);
        if (elapsed.count() >= config.auto_save_every_minutes) {
            return true;
        }
    }

    return false;
}

void IncrementalTrainer::perform_auto_save() {
    std::string auto_save_path =
        get_session_dir() + "/auto_save_session_" + std::to_string(current_session_id) + ".bin";

    if (save_model(auto_save_path)) {
        Logger::info("Auto-saved checkpoint to {}", auto_save_path);
        last_save_time = std::chrono::system_clock::now();
        samples_since_last_save = 0;
    }
}

std::string IncrementalTrainer::generate_session_checkpoint_path() {
    std::ostringstream oss;
    oss << get_session_dir() << "/session_" << current_session_id << "_checkpoint.bin";
    return oss.str();
}

bool IncrementalTrainer::reset_all(bool keep_data_registry) {
    Logger::info("=== IncrementalTrainer::reset_all() called ===");

    // ------------------------------------------------------------------
    // 1. Backup the main model file so the old weights can be recovered.
    // ------------------------------------------------------------------
    if (!model_path_.empty() && fs::exists(model_path_)) {
        std::string backup = model_path_ + ".bak";
        try {
            fs::rename(model_path_, backup);
            Logger::info("Old model backed up to: {}", backup);
        } catch (const std::exception& e) {
            Logger::warn("Could not back up model file: {}", e.what());
        }
    }

    // ------------------------------------------------------------------
    // 2. Remove all per-session files from the session directory.
    // ------------------------------------------------------------------
    std::string sdir = get_session_dir();
    if (fs::exists(sdir)) {
        // Session-directory files to always remove
        const std::vector<std::string> sdir_remove = {
            "session_history.txt",
            "pending_files.txt",
        };
        for (const auto& name : sdir_remove) {
            std::string p = sdir;
            p += '/';
            p += name;
            std::error_code ec;
            auto st = fs::symlink_status(p, ec);
            if (!ec && st.type() != fs::file_type::not_found) {
                fs::remove(p, ec);
                if (!ec) {
                    Logger::info("Removed: {}", p);
                }
            }
        }

        // The symlinks live in CWD (same directory as the binary / working dir),
        // not inside the session dir — remove them from their actual location.
        const std::vector<std::string> cwd_symlinks = {
            config.latest_symlink_name,
            config.best_symlink_name,
        };
        for (const auto& name : cwd_symlinks) {
            // Remove from CWD
            remove_symlink_if_exists(name);
            // Also remove from session dir in case an older version placed them there
            std::string session_path = sdir;
            session_path += '/';
            session_path += name;
            remove_symlink_if_exists(session_path);
        }

        // Remove all checkpoint / autosave .bin files
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(sdir, ec)) {
            const auto& path = entry.path();
            std::string ext = path.extension().string();
            std::string stem = path.stem().string();
            if (ext == ".bin" &&
                (stem.rfind("session_", 0) == 0 || stem.rfind("auto_save_", 0) == 0)) {
                fs::remove(path, ec);
                if (!ec) {
                    Logger::info("Removed checkpoint: {}", path.string());
                }
            }
        }

        // Optionally remove the data registry
        std::string registry_file = sdir + "/" + dataset_config_.data_registry_file;
        if (!keep_data_registry) {
            if (fs::exists(registry_file)) {
                std::error_code ec2;
                fs::remove(registry_file, ec2);
                Logger::info("Data registry removed");
            }
        } else {
            // Mark every entry as untrained so the next retrain picks them up
            for (auto& dv : data_registry) {
                dv.trained = false;
            }
            save_data_registry();
            Logger::info("Data registry preserved; all entries marked untrained");
        }
    }

    // ------------------------------------------------------------------
    // 3. Reset all in-memory tracking state.
    // ------------------------------------------------------------------
    session_history.clear();
    current_session_id = 0;
    pending_data_files.clear();
    samples_since_last_save = 0;
    best_validation_loss = std::numeric_limits<float>::max();
    best_checkpoint_path.clear();
    dashboard_lines_drawn_ = 0;

    if (!keep_data_registry) {
        data_registry.clear();
        trained_data_files.clear();
    }

    // ------------------------------------------------------------------
    // 4. Rebuild model from current config (new architecture).
    // ------------------------------------------------------------------
    reset_model_for_config();

    // ------------------------------------------------------------------
    // 5. Recreate the session directory and persist empty state.
    // ------------------------------------------------------------------
    ensure_directories_exist();
    save_session_history();
    save_pending_data_list();

    Logger::info(
        "Reset complete. Model rebuilt with d_model={} heads={} enc_layers={} dec_layers={} "
        "d_ff={} max_seq={}",
        config.base_config.d_model, config.base_config.num_heads,
        config.base_config.num_encoder_layers, config.base_config.num_decoder_layers,
        config.base_config.d_ff, config.base_config.max_seq_length);
    return true;
}

std::string IncrementalTrainer::get_session_dir() const {
    return config.session_dir;
}

void IncrementalTrainer::ensure_directories_exist() {
    if (!fs::exists(config.session_dir)) {
        fs::create_directories(config.session_dir);
    }

    if (dataset_config_.cache_tokenized_data && !fs::exists(dataset_config_.tokenized_cache_dir)) {
        fs::create_directories(dataset_config_.tokenized_cache_dir);
    }
}

// TODO(TD-028): Move to DatasetRegistry::save_pending_list()
bool IncrementalTrainer::save_pending_data_list() {
    std::string pending_file = get_session_dir() + "/pending_files.txt";
    std::ofstream file(pending_file);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& pending_file_path : pending_data_files) {
        file << pending_file_path << '\n';
    }

    return true;
}

// TODO(TD-028): Move to DatasetRegistry::load_pending_list()
bool IncrementalTrainer::load_pending_data_list() {
    std::string pending_file = get_session_dir() + "/pending_files.txt";
    if (!fs::exists(pending_file)) {
        return false;
    }

    std::ifstream file(pending_file);
    if (!file.is_open()) {
        return false;
    }

    pending_data_files.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            pending_data_files.push_back(line);
        }
    }

    if (!pending_data_files.empty()) {
        Logger::info("Loaded {} pending data files", pending_data_files.size());
    }

    return true;
}

// ============================================================================
// Symlink Management (TD-005)
// ============================================================================

bool IncrementalTrainer::is_windows_platform() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool IncrementalTrainer::create_or_update_symlink(const std::string& target,
                                                  const std::string& link_path) {
    if (!config.enable_checkpoint_symlinks) {
        return false;
    }

    // Remove existing symlink/file at link_path
    remove_symlink_if_exists(link_path);

    try {
        if (is_windows_platform()) {
            std::error_code ec;
            fs::copy_file(target, link_path, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                Logger::warn("Failed to copy checkpoint file: {}", ec.message());
                return false;
            }
            Logger::info("Copied checkpoint to: {}", link_path);
        } else {
            std::error_code ec;
            fs::create_symlink(target, link_path, ec);
            if (ec) {
                Logger::warn("Failed to create symlink: {}", ec.message());
                return false;
            }
            Logger::info("Created symlink: {} -> {}", link_path, target);
        }
        return true;
    } catch (const std::exception& e) {
        Logger::warn("Failed to create/update checkpoint link: {}", e.what());
        return false;
    }
}

bool IncrementalTrainer::remove_symlink_if_exists(const std::string& link_path) {
    try {
        // Use symlink_status (lstat) rather than exists() (stat): exists() follows
        // symlinks and returns false for broken symlinks, leaving stale dangling
        // links behind and preventing create_symlink from succeeding afterwards.
        std::error_code sc;
        auto st = fs::symlink_status(link_path, sc);
        if (!sc && st.type() != fs::file_type::not_found) {
            std::error_code ec;
            fs::remove(link_path, ec);
            if (ec) {
                Logger::warn("Failed to remove existing link '{}': {}", link_path, ec.message());
                return false;
            }
            Logger::debug("Removed existing link: {}", link_path);
        }
        return true;
    } catch (const std::exception& e) {
        Logger::warn("Error removing link '{}': {}", link_path, e.what());
        return false;
    }
}

void IncrementalTrainer::update_checkpoint_symlinks(const std::string& checkpoint_path) {
    if (!config.enable_checkpoint_symlinks) {
        return;
    }

    // Create/update "latest_checkpoint.bin" symlink in root directory
    std::string latest_link = config.latest_symlink_name;
    if (!create_or_update_symlink(checkpoint_path, latest_link)) {
        Logger::error("Failed to update latest checkpoint symlink! target={} link={}",
                      checkpoint_path, latest_link);
    }
}

void IncrementalTrainer::update_best_checkpoint(float validation_loss,
                                                const std::string& checkpoint_path) {
    if (!config.enable_checkpoint_symlinks) {
        return;
    }

    // Check if this is the best validation loss so far
    bool is_best = false;

    if (session_history.size() <= 1) {
        // First session - always best
        is_best = true;
    } else {
        // Compare with previous best
        if (validation_loss < best_validation_loss) {
            is_best = true;
        }
    }

    if (is_best) {
        best_validation_loss = validation_loss;
        best_checkpoint_path = checkpoint_path;

        std::string best_link = config.best_symlink_name;
        if (!create_or_update_symlink(checkpoint_path, best_link)) {
            Logger::error("Failed to update best checkpoint symlink! target={} link={}",
                          checkpoint_path, best_link);
        } else {
            Logger::info("New best checkpoint! Validation loss: {:.4f}", validation_loss);
        }
    }
}

std::string IncrementalTrainer::get_best_checkpoint_path() const {
    return best_checkpoint_path;
}

// ============================================================================

// TODO(TD-028): Move to DatasetRegistry::load_conversation_pairs() as a static method
int IncrementalTrainer::load_conversation_pairs(const std::string& filepath,
                                                std::vector<ConversationPair>& pairs) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        Logger::error("Cannot open file: {}", filepath);
        return 0;
    }

    std::string line;
    std::string current_input;
    std::string current_response;
    int pair_count = 0;

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) {
            // End of pair
            if (!current_input.empty() && !current_response.empty()) {
                pairs.emplace_back(current_input, current_response);
                pair_count++;
                current_input.clear();
                current_response.clear();
            }
            continue;
        }

        if (line.substr(0, 6) == "INPUT:") {
            // Commit any already-complete pair before starting a new one
            // (handles files with no blank-line separator between pairs)
            if (!current_input.empty() && !current_response.empty()) {
                pairs.emplace_back(current_input, current_response);
                pair_count++;
                current_input.clear();
                current_response.clear();
            }
            current_input = line.substr(6);
            current_input.erase(0, current_input.find_first_not_of(" \t"));
        } else if (line.substr(0, 9) == "RESPONSE:") {
            current_response = line.substr(9);
            current_response.erase(0, current_response.find_first_not_of(" \t"));
        }
    }

    // Don't forget last pair
    if (!current_input.empty() && !current_response.empty()) {
        pairs.emplace_back(current_input, current_response);
        pair_count++;
    }

    file.close();

    Logger::info("Loaded {} pairs from: {}", pair_count, filepath);

    return pair_count;
}

// Project Gutenberg integration (TD-028: thin wrappers delegating to DataFetcher)

bool IncrementalTrainer::add_gutenberg_book(int book_id, int num_pairs) {
    DataFetcher fetcher;
    std::string path = fetcher.fetch_gutenberg(book_id, num_pairs);
    if (path.empty()) return false;
    return add_new_data(path);
}

bool IncrementalTrainer::add_gutenberg_books(const std::vector<int>& book_ids,
                                             int num_pairs_per_book) {
    int success_count = 0;
    for (int book_id : book_ids) {
        if (add_gutenberg_book(book_id, num_pairs_per_book)) {
            success_count++;
        }
    }
    Logger::info("Added {}/{} books to training queue", success_count, book_ids.size());
    return success_count > 0;
}

// ============================================================================
// Dashboard helpers (TD-009)
// ============================================================================

std::string IncrementalTrainer::format_duration(double seconds) {
    int secs = static_cast<int>(seconds);
    int mins = secs / 60;
    int hours = mins / 60;
    secs %= 60;
    mins %= 60;

    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << "h" << std::setw(2) << std::setfill('0') << mins << "m";
    } else if (mins > 0) {
        oss << mins << "m" << std::setw(2) << std::setfill('0') << secs << "s";
    } else {
        oss << seconds << "s";
    }
    return oss.str();
}

std::string IncrementalTrainer::progress_bar(int current, int total, int bar_width) {
    if (total <= 0) {
        return std::string(bar_width, '-');
    }
    int filled = (int)((double)current / total * bar_width);
    filled = std::clamp(filled, 0, bar_width);
    std::string bar(filled, '=');
    if (filled < bar_width) {
        bar += '>';
    }
    bar += std::string(std::max(0, bar_width - (int)bar.size()), ' ');
    return bar;
}

// HuggingFace Datasets integration (TD-028: thin wrapper delegating to DataFetcher)

bool IncrementalTrainer::add_huggingface_dataset(const std::string& dataset_id, int num_pairs,
                                                 const std::string& split,
                                                 const std::string& input_field,
                                                 const std::string& output_field) {
    DataFetcher fetcher;
    std::string path = fetcher.fetch_huggingface(dataset_id, num_pairs, split, input_field,
                                                 output_field);
    if (path.empty()) return false;
    return add_new_data(path);
}

// ============================================================================
// Metrics API Server Management
// ============================================================================
