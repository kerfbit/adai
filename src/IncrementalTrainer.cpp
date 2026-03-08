#include "IncrementalTrainer.hpp"
#include "Config.hpp"
#include "Logger.hpp"
#include "TrainingMetricsAPI.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <ctime>
#include <regex>
#include <cstdlib>
#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#endif

// Bring Logger into scope without qualifying every call
using adai::Logger;

// Legacy ANSI codes kept for print_session_history / print_data_registry (intentional TUI output)
#define COLOR_RESET    "\033[0m"
#define COLOR_INFO     "\033[1;36m"
#define COLOR_SUCCESS  "\033[1;32m"
#define COLOR_WARNING  "\033[1;33m"
#define COLOR_ERROR    "\033[1;31m"
#define COLOR_PROGRESS "\033[1;35m"

namespace fs = std::filesystem;

// ============================================================================
// build_model() — THE single point for EncoderDecoderModel construction.
// Reads vocab_path_ and config.base_config to build the model.  Every code
// path that creates or recreates the model must call this method; there is no
// other place in the codebase that instantiates EncoderDecoderModel.
// ============================================================================
void IncrementalTrainer::build_model() {
    Logger::info("Building model: d_model={} heads={} d_ff={} enc_layers={} dec_layers={} max_seq={}",
                 config.base_config.d_model,
                 config.base_config.num_heads,
                 config.base_config.d_ff,
                 config.base_config.num_encoder_layers,
                 config.base_config.num_decoder_layers,
                 config.base_config.max_seq_length);

    auto tok = std::make_unique<BPETokenizer>();
    tok->load_vocab(vocab_path_);
    Logger::info("Tokenizer loaded (vocab size: {})", tok->get_vocab_size());

    model = std::make_unique<EncoderDecoderModel>(
        tok->get_vocab_size(),
        config.base_config.d_model,
        config.base_config.num_encoder_layers,
        config.base_config.num_decoder_layers,
        config.base_config.num_heads,
        config.base_config.d_ff,
        config.base_config.max_seq_length
    );
    model->set_tokenizer(tok.release());
}

// ============================================================================
// make_incremental_config() — translate ServiceConfig → IncrementalConfig.
// ============================================================================
/*static*/
IncrementalConfig IncrementalTrainer::make_incremental_config(const adai::ServiceConfig& svc) {
    IncrementalConfig cfg;
    cfg.base_config.d_model            = static_cast<int>(svc.d_model);
    cfg.base_config.num_heads          = static_cast<int>(svc.num_heads);
    cfg.base_config.d_ff               = static_cast<int>(svc.d_ff);
    cfg.base_config.num_encoder_layers = static_cast<int>(svc.num_encoder_layers);
    cfg.base_config.num_decoder_layers = static_cast<int>(svc.num_decoder_layers);
    cfg.base_config.max_seq_length     = static_cast<int>(svc.max_seq_length);
    cfg.base_config.learning_rate      = svc.learning_rate;
    cfg.base_config.num_epochs         = svc.num_epochs;
    cfg.base_config.weight_decay       = svc.weight_decay;
    cfg.base_config.gradient_clip_norm = svc.gradient_clip;
    cfg.base_config.batch_size         = svc.batch_size;
    // Suppress verbose per-sample logging from ChatbotTrainer, as we use
    // the TUI dashboard for real-time feedback.
    cfg.base_config.log_level          = LogLevel::NORMAL;
    
    // Metrics service configuration
    cfg.enable_metrics_service = svc.enable_metrics_service;
    cfg.metrics_push_enabled = svc.metrics_push_enabled;
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
    
    return cfg;
}

// ============================================================================
// Primary constructor — config.conf is the required entry point.
// ============================================================================
IncrementalTrainer::IncrementalTrainer(const std::string& config_file_path)
    : current_session_id(0), samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path(""),
      dashboard_lines_drawn_(0),
      current_sample_in_epoch_(0), total_samples_in_epoch_(0), running_sample_loss_(0.0f),
      current_item_loss_(0.0f), current_item_grad_norm_(0.0f), current_item_lr_(0.0f) {

    Logger::info("Loading configuration from: {}",
                 config_file_path.empty() ? "<system default>" : config_file_path);

    adai::ServiceConfig svc = config_file_path.empty()
        ? adai::ConfigLoader::load()
        : adai::ConfigLoader::load(config_file_path);

    if (svc.vocab_path.empty()) {
        throw std::runtime_error(
            "VOCAB_PATH must be set in config.conf (or via the VOCAB_PATH environment variable)");
    }

    vocab_path_ = svc.vocab_path;
    model_path_ = svc.model_path.empty() ? "chatbot_model.bin" : svc.model_path;
    config      = make_incremental_config(svc);
    config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;

    // Initialize metrics service if enabled
    if (config.enable_metrics_service) {
        config.metrics_config.enable_persistence = true;
        config.metrics_config.metrics_file = "training_sessions/metrics.jsonl";
        config.metrics_config.summary_file = "training_sessions/metrics_summary.json";
        config.metrics_config.persist_every_samples = 100;
        config.metrics_config.persist_every_seconds = 30;
        // Preserve push settings from config
        // (already set by make_incremental_config from ServiceConfig)
        metrics_service_ = std::make_unique<TrainingMetricsService>(config.metrics_config);
        Logger::info("TrainingMetricsService initialized (push enabled: {}, URL: {})", 
                    config.metrics_config.enable_push, config.metrics_config.push_url);
    }

    Logger::info("Initializing Incremental Training System...");
    build_model();

    // Try to load existing model checkpoint
    std::ifstream model_file(model_path_);
    if (model_file.good()) {
        try {
            model->load_model(model_path_);
            Logger::info("Model loaded from: {}", model_path_);
        } catch (...) {
            Logger::warn("Could not load model, using fresh initialization");
        }
    }

    ensure_directories_exist();
    load_session_history();
    load_data_registry();
    load_pending_data_list();

    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            Logger::info("Best checkpoint: {} (val loss: {})", best_checkpoint_path, best_validation_loss);
        }
    }

    last_save_time             = std::chrono::system_clock::now();
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_   = session_start_time_steady_;
}

// ============================================================================
// Explicit-paths constructor (low-level, uses default IncrementalConfig).
// ============================================================================
IncrementalTrainer::IncrementalTrainer(const std::string& vocab_path,
                                       const std::string& model_path)
    : current_session_id(0), samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path(""),
      dashboard_lines_drawn_(0),
      current_sample_in_epoch_(0), total_samples_in_epoch_(0), running_sample_loss_(0.0f),
      current_item_loss_(0.0f), current_item_grad_norm_(0.0f), current_item_lr_(0.0f) {

    Logger::info("Initializing Incremental Training System...");

    vocab_path_ = vocab_path;
    model_path_ = model_path;

    // Initialize metrics service with default config
    if (config.enable_metrics_service) {
        config.metrics_config.enable_persistence = true;
        config.metrics_config.metrics_file = "training_sessions/metrics.jsonl";
        config.metrics_config.summary_file = "training_sessions/metrics_summary.json";
        metrics_service_ = std::make_unique<TrainingMetricsService>(config.metrics_config);
        Logger::info("TrainingMetricsService initialized (push enabled: {}, URL: {})", 
                    config.metrics_config.enable_push, config.metrics_config.push_url);
    }

    // Build model using default config.
    build_model();

    // Try to load existing model checkpoint
    std::ifstream model_file(model_path);
    if (model_file.good()) {
        try {
            model->load_model(model_path);
            Logger::info("Model loaded from: {}", model_path);
        } catch (...) {
            Logger::warn("Could not load model, using fresh initialization");
        }
    }

    ensure_directories_exist();
    load_session_history();
    load_data_registry();
    load_pending_data_list();

    // Initialize best checkpoint from history (TD-005)
    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            Logger::info("Best checkpoint: {} (val loss: {})", best_checkpoint_path, best_validation_loss);
        }
    }

    last_save_time             = std::chrono::system_clock::now();
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_   = session_start_time_steady_;
}

// ============================================================================
// Explicit-paths + pre-built config constructor.
// Applies config BEFORE building the model so no defaults are baked in.
// ============================================================================
IncrementalTrainer::IncrementalTrainer(const std::string& vocab_path,
                                       const std::string& model_path,
                                       const IncrementalConfig& cfg)
    : current_session_id(0), samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path(""),
      dashboard_lines_drawn_(0),
      current_sample_in_epoch_(0), total_samples_in_epoch_(0), running_sample_loss_(0.0f),
      current_item_loss_(0.0f), current_item_grad_norm_(0.0f), current_item_lr_(0.0f) {

    Logger::info("Initializing Incremental Training System...");

    vocab_path_ = vocab_path;
    model_path_ = model_path;
    config      = cfg;  // set config FIRST so build_model() uses the correct architecture

    // Initialize metrics service if enabled
    if (config.enable_metrics_service) {
        config.metrics_config.enable_persistence = true;
        config.metrics_config.metrics_file = "training_sessions/metrics.jsonl";
        config.metrics_config.summary_file = "training_sessions/metrics_summary.json";
        metrics_service_ = std::make_unique<TrainingMetricsService>(config.metrics_config);
        Logger::info("TrainingMetricsService initialized (push enabled: {}, URL: {})", 
                    config.metrics_config.enable_push, config.metrics_config.push_url);
    }

    build_model();

    // Try to load existing model checkpoint
    std::ifstream model_file(model_path);
    if (model_file.good()) {
        try {
            model->load_model(model_path);
            Logger::info("Model loaded from: {}", model_path);
        } catch (...) {
            Logger::warn("Could not load model, using fresh initialization");
        }
    }

    ensure_directories_exist();
    load_session_history();
    load_data_registry();
    load_pending_data_list();

    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (session.final_validation_loss < best_validation_loss) {
                best_validation_loss = session.final_validation_loss;
                best_checkpoint_path = session.checkpoint_path;
            }
        }
        if (!best_checkpoint_path.empty()) {
            Logger::info("Best checkpoint: {} (val loss: {})", best_checkpoint_path, best_validation_loss);
        }
    }

    last_save_time             = std::chrono::system_clock::now();
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_   = session_start_time_steady_;
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

bool IncrementalTrainer::train_incremental(int num_epochs) {
    if (pending_data_files.empty()) {
        Logger::warn("No pending data files to train on");
        return false;
    }

    Logger::info("Starting Incremental Training Session #{}", current_session_id + 1);
    Logger::info("Pending data files: {}", pending_data_files.size());

    initialize_session();

    // Reset dashboard state and record session start time (TD-009)
    dashboard_lines_drawn_     = 0;
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_   = session_start_time_steady_;

    // Load all pending data — files are independent so load them in parallel
    std::vector<ConversationPair> training_pairs;
    std::vector<ConversationPair> validation_pairs;

    {
        const int n_files = static_cast<int>(pending_data_files.size());
        std::vector<std::vector<ConversationPair>> per_file(n_files);

#ifdef ADAI_ENABLE_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int fi = 0; fi < n_files; ++fi) {
            load_conversation_pairs(pending_data_files[fi], per_file[fi]);
        }

        // Sequential: split, collect pairs, update data registry
        for (int fi = 0; fi < n_files; ++fi) {
            const auto& data_file = pending_data_files[fi];
            auto& pairs = per_file[fi];
            int loaded = static_cast<int>(pairs.size());
            if (loaded > 0) {
                int val_size = loaded / config.base_config.validation_split;
                for (int i = 0; i < loaded; ++i) {
                    if (i < val_size)
                        validation_pairs.push_back(pairs[i]);
                    else
                        training_pairs.push_back(pairs[i]);
                }

                DataVersion dv;
                dv.data_file   = data_file;
                dv.checksum    = compute_data_checksum(data_file);
                dv.num_samples = loaded;
                dv.added_time  = std::chrono::system_clock::now();
                dv.trained     = true;
                data_registry.push_back(dv);
                trained_data_files.insert(data_file);
            }
        }
    }

    Logger::info("Total training samples: {}",   training_pairs.size());
    Logger::info("Total validation samples: {}", validation_pairs.size());

    // Store total sample count for dashboard and reset per-sample state
    total_samples_in_epoch_ = static_cast<int>(training_pairs.size());
    current_sample_in_epoch_ = 0;
    running_sample_loss_ = 0.0f;

    // Create trainer and configure
    ChatbotTrainer trainer(config.base_config);
    trainer.set_model(std::move(model));

    // Set metrics service on trainer if enabled
    if (metrics_service_) {
        trainer.set_metrics_service(metrics_service_.get());
        // Start metrics session
        metrics_service_->start_session(current_session_id + 1, num_epochs, 
                                       static_cast<int>(training_pairs.size()));
    }

    for (const auto& pair : training_pairs) {
        trainer.add_training_pair(pair.input, pair.response);
    }

    // Initial dashboard draw before the first epoch begins
    if (!session_history.empty()) {
        display_dashboard(session_history.back(), 0, num_epochs, false);
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

            // Redraw the live dashboard (epoch is 0-based, display 1-based)
            bool is_last = (epoch + 1 == total);
            display_dashboard(session, epoch + 1, total, is_last);

            // Reset per-sample counters for the next epoch
            current_sample_in_epoch_ = 0;
            running_sample_loss_     = 0.0f;
            current_item_loss_       = 0.0f;
            current_item_grad_norm_  = 0.0f;
        }
        // NOTE: Do NOT log to stdout/stderr here. Any extra line written between
        // display_dashboard() calls offsets the cursor-up count stored in
        // dashboard_lines_drawn_ and causes the dashboard to drift downward
        // instead of redrawing in place.
    });

    // ── Per-sample callback: update running state and redraw dashboard ──
    trainer.set_sample_callback([this](int sample, int total_samples, float running_loss, float step_loss, float grad_norm, float lr) {
        // Reset the epoch timer on the very first sample so that throughput
        // excludes preprocessing (split + tokenization) time.
        if (sample == 1) {
            epoch_start_time_steady_ = std::chrono::steady_clock::now();
        }
        current_sample_in_epoch_ = sample;
        total_samples_in_epoch_  = total_samples;
        running_sample_loss_     = running_loss;
        current_item_loss_       = step_loss;
        current_item_grad_norm_  = grad_norm;
        current_item_lr_         = lr;
        if (!session_history.empty()) {
            // Use the current per-epoch data — back() reflects epochs already done;
            // the running loss from the sample callback covers the in-progress epoch.
            display_dashboard(session_history.back(),
                              static_cast<int>(session_history.back().per_epoch_losses.size()),
                              // epoch arg = completed epochs (in-progress epoch not yet in back())
                              config.base_config.num_epochs,
                              false);
        }
    });

    Logger::info("Training for {} epochs...", num_epochs);
    bool success = trainer.train(num_epochs);

    // End metrics session if enabled
    if (metrics_service_) {
        metrics_service_->end_session();
    }

    // Retrieve model after training
    model = trainer.release_model();

    if (success) {
        // Save checkpoint
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);

        // Finalize session (per-epoch vectors already populated by callback)
        float final_loss     = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(static_cast<int>(training_pairs.size()), num_epochs,
                         final_loss, final_val_loss);

        // Clear pending data
        pending_data_files.clear();
        save_pending_data_list();

        // Persist
        save_data_registry();
        save_session_history();

        Logger::info("Incremental training session completed successfully");
        print_training_summary();
    }

    return success;
}

bool IncrementalTrainer::train_on_new_data_only(int num_epochs) {
    return train_incremental(num_epochs);
}

bool IncrementalTrainer::train_full_retrain(int num_epochs) {
    Logger::info("Starting full retrain on all data");

    // Collect all trained data files
    std::vector<std::string> all_data_files;
    for (const auto& dv : data_registry) {
        if (dv.trained) {
            all_data_files.push_back(dv.data_file);
        }
    }

    // Add pending files
    all_data_files.insert(all_data_files.end(), pending_data_files.begin(), pending_data_files.end());

    if (all_data_files.empty()) {
        Logger::warn("No data files to train on");
        return false;
    }

    Logger::info("Retraining on {} data file(s)", all_data_files.size());
    initialize_session();

    // Reset dashboard state (TD-009)
    dashboard_lines_drawn_     = 0;
    session_start_time_steady_ = std::chrono::steady_clock::now();
    epoch_start_time_steady_   = session_start_time_steady_;

    // Load all data — files are independent so load them in parallel
    std::vector<ConversationPair> all_pairs;
    {
        const int n_files = static_cast<int>(all_data_files.size());
        std::vector<std::vector<ConversationPair>> per_file(n_files);

#ifdef ADAI_ENABLE_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int fi = 0; fi < n_files; ++fi) {
            load_conversation_pairs(all_data_files[fi], per_file[fi]);
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

    // Set metrics service on trainer if enabled
    if (metrics_service_) {
        trainer.set_metrics_service(metrics_service_.get());
        // Start metrics session for full retrain
        metrics_service_->start_session(current_session_id + 1, num_epochs,
                                       total_samples_in_epoch_);
    }

    for (const auto& pair : all_pairs) {
        trainer.add_training_pair(pair.input, pair.response);
    }

    // Initial dashboard draw before the first epoch begins
    if (!session_history.empty()) {
        display_dashboard(session_history.back(), 0, num_epochs, false);
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
            display_dashboard(session, epoch + 1, total, (epoch + 1 == total));

            // Reset per-sample counters for the next epoch
            current_sample_in_epoch_ = 0;
            running_sample_loss_     = 0.0f;
            current_item_loss_       = 0.0f;
            current_item_grad_norm_  = 0.0f;
        }
    });

    // ── Per-sample callback for full retrain ──
    trainer.set_sample_callback([this](int sample, int total_samples, float running_loss, float step_loss, float grad_norm, float lr) {
        // Reset the epoch timer on the very first sample so that throughput
        // excludes preprocessing (split + tokenization) time.
        if (sample == 1) {
            epoch_start_time_steady_ = std::chrono::steady_clock::now();
        }
        current_sample_in_epoch_ = sample;
        total_samples_in_epoch_  = total_samples;
        running_sample_loss_     = running_loss;
        current_item_loss_       = step_loss;
        current_item_grad_norm_  = grad_norm;
        current_item_lr_         = lr;
        if (!session_history.empty()) {
            display_dashboard(session_history.back(),
                              static_cast<int>(session_history.back().per_epoch_losses.size()),
                              config.base_config.num_epochs,
                              false);
        }
    });

    bool success = trainer.train(num_epochs);
    model = trainer.release_model();

    // End metrics session if enabled
    if (metrics_service_) {
        metrics_service_->end_session();
    }

    if (success) {
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);

        float final_loss     = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(static_cast<int>(all_pairs.size()), num_epochs, final_loss, final_val_loss);

        pending_data_files.clear();
        save_data_registry();
        save_session_history();
    }

    return success;
}

bool IncrementalTrainer::resume_last_session() {
    if (session_history.empty()) {
        Logger::warn("No previous sessions to resume");
        return false;
    }

    const auto& last_session = session_history.back();

    // Validate checkpoint path
    if (last_session.checkpoint_path.empty()) {
        Logger::error("Invalid session: checkpoint path is empty");
        return false;
    }

    // Check if checkpoint files exist (check for .config which should always be present)
    if (!fs::exists(last_session.checkpoint_path + ".config")) {
        Logger::error("Checkpoint file not found: {}.config", last_session.checkpoint_path);
        return false;
    }

    Logger::info("Resuming from session #{}", last_session.session_id);
    return load_model(last_session.checkpoint_path);
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
            try { out.push_back(std::stof(token)); } catch (...) {}
        }
    };
    auto parse_double_list = [](const std::string& s, std::vector<double>& out) {
        std::istringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try { out.push_back(std::stod(token)); } catch (...) {}
        }
    };

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        TrainingSession session;

        iss >> session.session_id >> session.samples_trained >> session.epochs_completed
            >> session.final_loss >> session.final_validation_loss >> session.checkpoint_path;

        if (iss.fail() || session.checkpoint_path.empty()) {
            Logger::warn("Skipping malformed session history line: {}", line);
            continue;
        }

        // v2 extended format: checkpoint_path may contain "|losses:...|vallosses:...|lrs:...|times:..."
        size_t pipe = session.checkpoint_path.find('|');
        if (pipe != std::string::npos) {
            std::string extended = session.checkpoint_path.substr(pipe + 1);
            session.checkpoint_path = session.checkpoint_path.substr(0, pipe);

            std::istringstream es(extended);
            std::string segment;
            while (std::getline(es, segment, '|')) {
                auto colon = segment.find(':');
                if (colon == std::string::npos) continue;
                std::string key   = segment.substr(0, colon);
                std::string value = segment.substr(colon + 1);
                if      (key == "losses")    parse_float_list(value,  session.per_epoch_losses);
                else if (key == "vallosses") parse_float_list(value,  session.per_epoch_validation_losses);
                else if (key == "lrs")       parse_float_list(value,  session.per_epoch_learning_rates);
                else if (key == "times")     parse_double_list(value, session.training_time_per_epoch);
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
            if (i) oss << ',';
            oss << v[i];
        }
        return oss.str();
    };
    auto join_doubles = [](const std::vector<double>& v) -> std::string {
        std::ostringstream oss;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) oss << ',';
            oss << v[i];
        }
        return oss.str();
    };

    for (const auto& session : session_history) {
        file << session.session_id << " "
             << session.samples_trained << " "
             << session.epochs_completed << " "
             << session.final_loss << " "
             << session.final_validation_loss << " "
             << session.checkpoint_path;
        if (!session.per_epoch_losses.empty()) {
            file << "|losses:"    << join_floats(session.per_epoch_losses)
                 << "|vallosses:" << join_floats(session.per_epoch_validation_losses)
                 << "|lrs:"       << join_floats(session.per_epoch_learning_rates)
                 << "|times:"     << join_doubles(session.training_time_per_epoch);
        }
        file << "\n";
    }

    return true;
}

TrainingSession IncrementalTrainer::get_current_session() const {
    if (session_history.empty()) {
        return TrainingSession();
    }
    return session_history.back();
}

std::vector<TrainingSession> IncrementalTrainer::get_session_history() const {
    return session_history;
}

void IncrementalTrainer::cleanup_old_sessions() {
    if (session_history.size() <= config.max_sessions_to_keep) {
        return;
    }
    
    int to_remove = session_history.size() - config.max_sessions_to_keep;
    
    for (int i = 0; i < to_remove; ++i) {
        const auto& session = session_history[i];
        
        // TD-005: Check if we're deleting the best checkpoint
        bool deleting_best = (session.checkpoint_path == best_checkpoint_path);
        
        // Delete checkpoint file
        if (fs::exists(session.checkpoint_path)) {
            fs::remove(session.checkpoint_path);
            Logger::info("Removed old checkpoint: {}", session.checkpoint_path);
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

bool IncrementalTrainer::load_data_registry() {
    std::string registry_file = get_session_dir() + "/" + config.data_registry_file;
    
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
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        DataVersion dv;
        int trained_int;
        
        iss >> dv.data_file >> dv.checksum >> dv.num_samples >> trained_int;
        dv.trained = (trained_int == 1);
        
        data_registry.push_back(dv);
        
        if (dv.trained) {
            trained_data_files.insert(dv.data_file);
        }
    }
    
    Logger::info("Loaded data registry: {} files ({} trained)", data_registry.size(), trained_data_files.size());
    
    return true;
}

bool IncrementalTrainer::save_data_registry() {
    std::string registry_file = get_session_dir() + "/" + config.data_registry_file;
    
    std::ofstream file(registry_file);
    if (!file.is_open()) {
        Logger::error("Failed to save data registry");
        return false;
    }
    
    file << "# Data Registry: data_file checksum num_samples trained\n";
    
    for (const auto& dv : data_registry) {
        file << dv.data_file << " "
             << dv.checksum << " "
             << dv.num_samples << " "
             << (dv.trained ? 1 : 0) << "\n";
    }
    
    return true;
}

bool IncrementalTrainer::is_data_trained(const std::string& data_file) {
    return trained_data_files.find(data_file) != trained_data_files.end();
}

std::string IncrementalTrainer::compute_data_checksum(const std::string& data_file) {
    // Simple checksum: file size + modification time
    if (!fs::exists(data_file)) {
        return "MISSING";
    }
    
    auto size = fs::file_size(data_file);
    auto ftime = fs::last_write_time(data_file);
    
    std::ostringstream oss;
    oss << size << "_" << ftime.time_since_epoch().count();
    
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

std::string IncrementalTrainer::get_latest_checkpoint() const {
    if (session_history.empty()) {
        return "";
    }
    return session_history.back().checkpoint_path;
}

// Returns a sparkline string visualising values (low = good for loss)
static std::string make_sparkline(const std::vector<float>& values, int width = 30) {
    if (values.empty()) return std::string(width, '-');
    const char* bars[] = {"\xe2\x96\x81","\xe2\x96\x82","\xe2\x96\x83","\xe2\x96\x84",
                          "\xe2\x96\x85","\xe2\x96\x86","\xe2\x96\x87","\xe2\x96\x88"};
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
            if (!is_windows_platform() && fs::is_symlink(config.latest_symlink_name))
                std::cout << " -> " << fs::read_symlink(config.latest_symlink_name).string();
            std::cout << "\n";
        }
        if (fs::exists(config.best_symlink_name)) {
            std::cout << "    best  : " << config.best_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.best_symlink_name))
                std::cout << " -> " << fs::read_symlink(config.best_symlink_name).string();
            std::cout << "  (val loss: " << std::fixed << std::setprecision(4) << best_validation_loss << ")\n";
        }
    }

    // Per-session details with sparklines
    for (const auto& s : session_history) {
        std::cout << "\n  Session #" << s.session_id
                  << "  samples=" << s.samples_trained
                  << "  epochs=" << s.epochs_completed
                  << "  loss=" << std::fixed << std::setprecision(4) << s.final_loss
                  << "  val=" << s.final_validation_loss << "\n";
        std::cout << "    checkpoint: " << s.checkpoint_path << "\n";

        if (!s.per_epoch_losses.empty()) {
            float best_val = std::numeric_limits<float>::max();
            double total_t = 0.0;
            for (float v : s.per_epoch_validation_losses)
                best_val = std::min(best_val, v);
            for (double t : s.training_time_per_epoch)
                total_t += t;

            std::cout << "    loss      : " << make_sparkline(s.per_epoch_losses) << "\n";
            std::cout << "    val loss  : " << make_sparkline(s.per_epoch_validation_losses) << "\n";
            std::cout << "    best val  : " << std::fixed << std::setprecision(4) << best_val << "\n";
            if (!s.training_time_per_epoch.empty())
                std::cout << "    epoch time: avg " << std::fixed << std::setprecision(1)
                          << (total_t / s.training_time_per_epoch.size()) << "s"
                          << "  total " << format_duration(total_t) << "\n";
        }
    }

    std::cout << "\n";
}

void IncrementalTrainer::print_session_history() const {
    std::cout << COLOR_INFO << "\n📜 Session History:" << COLOR_RESET << std::endl;
    std::cout << "Session | Samples | Epochs | Loss   | Val Loss | Checkpoint" << std::endl;
    std::cout << "--------|---------|--------|--------|----------|------------" << std::endl;
    
    for (const auto& session : session_history) {
        std::cout << std::setw(7) << session.session_id << " | "
                  << std::setw(7) << session.samples_trained << " | "
                  << std::setw(6) << session.epochs_completed << " | "
                  << std::setw(6) << std::fixed << std::setprecision(3) << session.final_loss << " | "
                  << std::setw(8) << session.final_validation_loss << " | "
                  << session.checkpoint_path << std::endl;
    }
}

void IncrementalTrainer::print_data_registry() const {
    std::cout << COLOR_INFO << "\n📋 Data Registry:" << COLOR_RESET << std::endl;
    std::cout << "Trained | Samples | Data File" << std::endl;
    std::cout << "--------|---------|----------" << std::endl;
    
    for (const auto& dv : data_registry) {
        std::cout << std::setw(7) << (dv.trained ? "✓" : " ") << " | "
                  << std::setw(7) << dv.num_samples << " | "
                  << dv.data_file << std::endl;
    }
}

float IncrementalTrainer::get_total_training_time_hours() const {
    float total_hours = 0.0f;
    for (const auto& session : session_history) {
        auto duration = std::chrono::duration_cast<std::chrono::hours>(
            session.end_time - session.start_time);
        total_hours += duration.count();
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
bool IncrementalTrainer::finalize_session(int samples_trained, int epochs_completed, float final_loss, float final_val_loss) {
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
    // Per-epoch metrics are accumulated live via the epoch callback in train_incremental()

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
    std::string auto_save_path = get_session_dir() + "/auto_save_session_" + 
                                 std::to_string(current_session_id) + ".bin";
    
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
            std::string p = sdir + "/" + name;
            std::error_code ec;
            auto st = fs::symlink_status(p, ec);
            if (!ec && st.type() != fs::file_type::not_found) {
                fs::remove(p, ec);
                if (!ec) Logger::info("Removed: {}", p);
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
            remove_symlink_if_exists(sdir + "/" + name);
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
                if (!ec) Logger::info("Removed checkpoint: {}", path.string());
            }
        }

        // Optionally remove the data registry
        std::string registry_file = sdir + "/" + config.data_registry_file;
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

    Logger::info("Reset complete. Model rebuilt with d_model={} heads={} enc_layers={} dec_layers={} d_ff={} max_seq={}",
        config.base_config.d_model,
        config.base_config.num_heads,
        config.base_config.num_encoder_layers,
        config.base_config.num_decoder_layers,
        config.base_config.d_ff,
        config.base_config.max_seq_length);
    return true;
}

std::string IncrementalTrainer::get_session_dir() const {
    return config.session_dir;
}

void IncrementalTrainer::ensure_directories_exist() {
    if (!fs::exists(config.session_dir)) {
        fs::create_directories(config.session_dir);
    }
    
    if (config.cache_tokenized_data && !fs::exists(config.tokenized_cache_dir)) {
        fs::create_directories(config.tokenized_cache_dir);
    }
}

bool IncrementalTrainer::save_pending_data_list() {
    std::string pending_file = get_session_dir() + "/pending_files.txt";
    std::ofstream file(pending_file);
    if (!file.is_open()) {
        return false;
    }
    
    for (const auto& data_file : pending_data_files) {
        file << data_file << "\n";
    }
    
    return true;
}

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

bool IncrementalTrainer::is_windows_platform() const {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

bool IncrementalTrainer::create_or_update_symlink(const std::string& target, const std::string& link_path) {
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

void IncrementalTrainer::update_best_checkpoint(float validation_loss, const std::string& checkpoint_path) {
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

// Project Gutenberg integration

std::string IncrementalTrainer::get_gutenberg_url(int book_id) const {
    // Project Gutenberg uses a tiered directory structure
    // Book 12345 is at: https://www.gutenberg.org/files/12345/12345-0.txt
    // Try UTF-8 version first (-0.txt), fallback to plain ASCII (.txt)
    std::ostringstream oss;
    oss << "https://www.gutenberg.org/files/" << book_id << "/" << book_id << "-0.txt";
    return oss.str();
}

bool IncrementalTrainer::download_file(const std::string& url, const std::string& output_path) {
    Logger::info("Downloading: {}", url);

    std::ostringstream cmd;
    cmd << "curl -L -f -s -o \"" << output_path << "\" \"" << url << "\"";

    int result = std::system(cmd.str().c_str());

    if (result == 0 && fs::exists(output_path) && fs::file_size(output_path) > 0) {
        Logger::info("Downloaded to: {}", output_path);
        return true;
    }

    if (url.find("-0.txt") != std::string::npos) {
        std::string fallback_url = url;
        size_t pos = fallback_url.find("-0.txt");
        fallback_url.replace(pos, 6, ".txt");

        Logger::info("Trying fallback URL: {}", fallback_url);
        std::ostringstream fallback_cmd;
        fallback_cmd << "curl -L -f -s -o \"" << output_path << "\" \"" << fallback_url << "\"";

        result = std::system(fallback_cmd.str().c_str());

        if (result == 0 && fs::exists(output_path) && fs::file_size(output_path) > 0) {
            Logger::info("Downloaded to: {}", output_path);
            return true;
        }
    }

    Logger::error("Failed to download: {}", url);
    return false;
}

bool IncrementalTrainer::download_gutenberg_book(int book_id, const std::string& output_dir) {
    // Create output directory if needed
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    
    std::string url = get_gutenberg_url(book_id);
    std::ostringstream output_path;
    output_path << output_dir << "/gutenberg_" << book_id << ".txt";
    
    return download_file(url, output_path.str());
}

bool IncrementalTrainer::download_gutenberg_books(const std::vector<int>& book_ids, 
                                                  const std::string& output_dir) {
    int success_count = 0;
    
    for (int book_id : book_ids) {
        if (download_gutenberg_book(book_id, output_dir)) {
            success_count++;
        }
    }
    
    Logger::info("Downloaded {}/{} books", success_count, book_ids.size());
    return success_count > 0;
}

std::string IncrementalTrainer::clean_gutenberg_text(const std::string& raw_text) {
    std::string cleaned = raw_text;
    
    // Remove Project Gutenberg header (before "*** START OF")
    size_t start_pos = cleaned.find("*** START OF");
    if (start_pos != std::string::npos) {
        start_pos = cleaned.find("\n", start_pos);
        if (start_pos != std::string::npos) {
            cleaned = cleaned.substr(start_pos + 1);
        }
    }
    
    // Remove Project Gutenberg footer (after "*** END OF")
    size_t end_pos = cleaned.find("*** END OF");
    if (end_pos != std::string::npos) {
        cleaned = cleaned.substr(0, end_pos);
    }
    
    // Remove excessive whitespace
    cleaned = std::regex_replace(cleaned, std::regex("[ \\t]+"), " ");
    cleaned = std::regex_replace(cleaned, std::regex("\\n{3,}"), "\n\n");
    
    return cleaned;
}

std::vector<std::string> IncrementalTrainer::extract_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    
    // Simple sentence splitting on . ! ?
    std::regex sentence_regex("[^.!?]+[.!?]+");
    
    auto sentences_begin = std::sregex_iterator(text.begin(), text.end(), sentence_regex);
    auto sentences_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = sentences_begin; i != sentences_end; ++i) {
        std::string sentence = i->str();
        
        // Trim whitespace
        sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
        sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);
        
        // Skip very short or very long sentences
        if (sentence.length() > 20 && sentence.length() < 500) {
            sentences.push_back(sentence);
        }
    }
    
    return sentences;
}

std::string IncrementalTrainer::generate_question_from_sentence(const std::string& sentence) {
    // Simple question generation patterns
    std::vector<std::string> question_templates = {
        "What does this mean: ",
        "Can you explain: ",
        "Tell me about: ",
        "What is this about: ",
        "Explain this: ",
        "What does this say: "
    };
    
    // Select a random template
    int idx = rand() % question_templates.size();
    return question_templates[idx] + sentence;
}

std::vector<std::pair<std::string, std::string>> 
IncrementalTrainer::create_qa_pairs_from_text(const std::vector<std::string>& sentences, int max_pairs) {
    std::vector<std::pair<std::string, std::string>> pairs;
    
    // Create pairs from consecutive sentences
    for (size_t i = 0; i < sentences.size() - 1 && pairs.size() < max_pairs; i += 2) {
        std::string question = generate_question_from_sentence(sentences[i]);
        std::string answer = sentences[i + 1];
        
        // Also create reverse pairs (context + question)
        if (pairs.size() < max_pairs) {
            pairs.emplace_back(question, answer);
        }
        
        // Create "summarize" style pairs
        if (i + 2 < sentences.size() && pairs.size() < max_pairs) {
            std::string context = sentences[i] + " " + sentences[i + 1];
            std::string summary = sentences[i + 2];
            pairs.emplace_back("Summarize: " + context, summary);
        }
    }
    
    return pairs;
}

bool IncrementalTrainer::convert_gutenberg_to_training_data(const std::string& text_file,
                                                            const std::string& output_file,
                                                            int max_pairs) {
    Logger::info("Converting Gutenberg text to training pairs: {}", text_file);

    std::ifstream file(text_file);
    if (!file.is_open()) {
        Logger::error("Cannot open: {}", text_file);
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string raw_text = buffer.str();
    file.close();
    
    // Clean the text
    std::string cleaned_text = clean_gutenberg_text(raw_text);
    
    // Extract sentences
    std::vector<std::string> sentences = extract_sentences(cleaned_text);
    Logger::info("Extracted {} sentences", sentences.size());

    if (sentences.empty()) {
        Logger::error("No valid sentences found in {}", text_file);
        return false;
    }

    auto pairs = create_qa_pairs_from_text(sentences, max_pairs);
    Logger::info("Created {} conversation pairs", pairs.size());

    std::ofstream out(output_file);
    if (!out.is_open()) {
        Logger::error("Cannot create: {}", output_file);
        return false;
    }
    
    for (const auto& pair : pairs) {
        out << "INPUT: " << pair.first << "\n";
        out << "RESPONSE: " << pair.second << "\n";
        out << "\n";
    }
    
    out.close();
    
    Logger::info("Training data saved to: {}", output_file);
    return true;
}

bool IncrementalTrainer::add_gutenberg_book(int book_id, int num_pairs) {
    std::string output_dir = "gutenberg_data";
    
    // Download the book
    if (!download_gutenberg_book(book_id, output_dir)) {
        return false;
    }
    
    // Convert to training data
    std::ostringstream text_file, training_file;
    text_file << output_dir << "/gutenberg_" << book_id << ".txt";
    training_file << output_dir << "/gutenberg_" << book_id << "_training.txt";
    
    if (!convert_gutenberg_to_training_data(text_file.str(), training_file.str(), num_pairs)) {
        return false;
    }
    
    // Add to pending data
    return add_new_data(training_file.str());
}

bool IncrementalTrainer::add_gutenberg_books(const std::vector<int>& book_ids, int num_pairs_per_book) {
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
    int secs  = static_cast<int>(seconds);
    int mins  = secs / 60;
    int hours = mins / 60;
    secs  %= 60;
    mins  %= 60;

    std::ostringstream oss;
    if (hours > 0)
        oss << hours << "h" << std::setw(2) << std::setfill('0') << mins << "m";
    else if (mins > 0)
        oss << mins  << "m" << std::setw(2) << std::setfill('0') << secs << "s";
    else
        oss << seconds << "s";
    return oss.str();
}

std::string IncrementalTrainer::progress_bar(int current, int total, int bar_width) {
    if (total <= 0) return std::string(bar_width, '-');
    int filled = (int)((double)current / total * bar_width);
    filled = std::clamp(filled, 0, bar_width);
    std::string bar(filled, '=');
    if (filled < bar_width) bar += '>';
    bar += std::string(std::max(0, bar_width - (int)bar.size()), ' ');
    return bar;
}

void IncrementalTrainer::display_dashboard(const TrainingSession& session,
                                           int current_epoch, int total_epochs,
                                           bool is_final) const {
    using namespace std::chrono;

    // ── ANSI color codes ──────────────────────────────────────────────────
    static const char* C_GREEN = "\033[1;32m";
    static const char* C_AMBER = "\033[1;33m";
    static const char* C_RED   = "\033[1;31m";
    static const char* C_CYAN  = "\033[1;36m";
    static const char* C_WHITE = "\033[1;37m";
    static const char* C_RESET = "\033[0m";

    // ── Timing ────────────────────────────────────────────────────────────
    auto now      = steady_clock::now();
    double elapsed = duration<double>(now - session_start_time_steady_).count();

    double avg_epoch_time = 0.0;
    if (!session.training_time_per_epoch.empty()) {
        for (double t : session.training_time_per_epoch) avg_epoch_time += t;
        avg_epoch_time /= session.training_time_per_epoch.size();
    }
    double eta_secs = (total_epochs - current_epoch) * avg_epoch_time;

    float cur_loss = session.per_epoch_losses.empty()            ? 0.0f : session.per_epoch_losses.back();
    float cur_val  = session.per_epoch_validation_losses.empty() ? 0.0f : session.per_epoch_validation_losses.back();
    // Prefer the live per-sample LR (updated every step); fall back to the
    // last epoch-end value once the epoch callback has run at least once.
    float cur_lr   = (current_item_lr_ > 0.0f)
                   ? current_item_lr_
                   : (session.per_epoch_learning_rates.empty() ? 0.0f : session.per_epoch_learning_rates.back());
    double last_t  = session.training_time_per_epoch.empty()     ? 0.0  : session.training_time_per_epoch.back();

    float prev_loss = (session.per_epoch_losses.size() >= 2)
                    ? session.per_epoch_losses[session.per_epoch_losses.size() - 2] : cur_loss;
    float prev_val  = (session.per_epoch_validation_losses.size() >= 2)
                    ? session.per_epoch_validation_losses[session.per_epoch_validation_losses.size() - 2] : cur_val;

    // Per-sample trend: track previous values across draws
    static float prev_item_loss_draw_    = 0.0f;
    static float prev_running_loss_draw_ = 0.0f;
    static float prev_item_ppl_draw_     = 0.0f;
    static float prev_running_ppl_draw_  = 0.0f;
    static float prev_gnorm_draw_        = 0.0f;

    // Best val across whole session
    float best_val = std::numeric_limits<float>::max();
    for (float v : session.per_epoch_validation_losses) best_val = std::min(best_val, v);

    // ── Clear + cursor home ───────────────────────────────────────────────
    std::cout << "\033[2J\033[H";

    // ── Box helpers ───────────────────────────────────────────────────────
    const int W = 80;
    auto hline = [&](const char* l, const char* r) {
        std::cout << l;
        for (int i = 0; i < W - 2; ++i) std::cout << "\xe2\x94\x80";
        std::cout << r << "\n";
    };
    // row() skips ANSI escape sequences when computing visible width so that
    // colored strings are padded correctly to fill box columns.
    auto row = [&](const std::string& content) {
        int vis = 0;
        bool in_ansi = false;
        for (unsigned char c : content) {
            if (c == '\033') { in_ansi = true;  continue; }
            if (in_ansi)     { if (std::isalpha(c)) in_ansi = false; continue; }
            if ((c & 0xC0) != 0x80) ++vis;   // non-continuation UTF-8 byte
        }
        int pad = std::max(0, W - 2 - vis);
        std::cout << "\xe2\x94\x82" << content << std::string(pad, ' ') << "\xe2\x94\x82\n";
    };

    // ── Formatting helpers ────────────────────────────────────────────────
    auto fmt_f = [](float v, int prec = 4) -> std::string {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    };
    // Wrap text in an ANSI color, always ending with reset
    auto colorize = [&](const std::string& s, const char* code) -> std::string {
        return std::string(code) + s + C_RESET;
    };
    // Pad a colored string to `width` visible chars (padding appended after reset)
    auto cpad = [](const std::string& colored, int vis_len, int width) -> std::string {
        return colored + std::string(std::max(0, width - vis_len), ' ');
    };

    // Color thresholds ---------------------------------------------------
    // Loss:       green < 1.0,  amber 1.0–4.0,  red ≥ 4.0
    auto loss_color = [&](float v) -> const char* {
        return (v < 1.0f) ? C_GREEN : (v < 4.0f) ? C_AMBER : C_RED;
    };
    // Perplexity: green < 3,    amber 3–55,      red ≥ 55
    auto ppl_color = [&](float v) -> const char* {
        return (v < 3.0f) ? C_GREEN : (v < 55.0f) ? C_AMBER : C_RED;
    };
    // Grad norm:  green < 1,    amber 1–5,       red ≥ 5
    auto gnorm_color = [&](float v) -> const char* {
        return (v < 1.0f) ? C_GREEN : (v < 5.0f) ? C_AMBER : C_RED;
    };

    // Trend icon: ▼ (down) / ▲ (up) / ─ (flat).
    // lower_is_better=true  → ▼ is green (improving), ▲ is red (worsening).
    // lower_is_better=false → ▲ is green, ▼ is red.
    auto trend = [&](float cur, float prev, bool lower_is_better = true) -> std::string {
        if (std::abs(cur - prev) < 1e-7f)
            return std::string(C_WHITE) + "\xe2\x94\x80" + C_RESET;  // ─
        bool improving = lower_is_better ? (cur < prev) : (cur > prev);
        const char* col  = improving ? C_GREEN : C_RED;
        // ▼ = \xe2\x96\xbc   ▲ = \xe2\x96\xb2
        const char* icon = (cur < prev) ? "\xe2\x96\xbc" : "\xe2\x96\xb2";
        return std::string(col) + icon + C_RESET;
    };

    // ── Title / epoch progress bar ────────────────────────────────────────
    std::string bar = "[" + progress_bar(current_epoch, total_epochs, 42) + "]";
    std::string pct;
    {
        std::ostringstream ss;
        ss << std::setw(3) << (total_epochs > 0 ? current_epoch * 100 / total_epochs : 0) << "%";
        pct = ss.str();
    }

    bool mid_epoch    = (current_sample_in_epoch_ > 0 &&
                         current_sample_in_epoch_ < total_samples_in_epoch_);
    int display_epoch = mid_epoch ? current_epoch + 1 : current_epoch;

    std::ostringstream title_ss;
    title_ss << " ADAI Training Dashboard  |  Session #" << session.session_id
             << "  |  Epoch " << display_epoch << "/" << total_epochs;
    if (mid_epoch)               title_ss << " \xe2\x96\xb6";   // ▶
    else if (is_final)           title_ss << " \xe2\x9c\x85";   // ✅
    else if (current_epoch == 0) title_ss << " \xe2\x8f\xb3";   // ⏳
    std::string title = title_ss.str();
    int title_pad = std::max(0, W - 4 - (int)title.size());

    hline("\xe2\x94\x8c", "\xe2\x94\x90");
    row(" " + title + std::string(title_pad, ' ') + " ");
    hline("\xe2\x94\x9c", "\xe2\x94\xa4");

    row(" Progress: " + bar + " " + pct);
    row(" Elapsed : " + format_duration(elapsed) + "   ETA total: " +
        (avg_epoch_time > 0 ? format_duration(eta_secs) : "--:--"));

    // ── Training Item section ─────────────────────────────────────────────
    hline("\xe2\x94\x9c", "\xe2\x94\xa4");
    row(" Training Item");
    hline("\xe2\x94\x9c", "\xe2\x94\xa4");

    if (total_samples_in_epoch_ > 0) {
        // Sample progress bar
        {
            std::ostringstream ss;
            if (current_sample_in_epoch_ > 0) {
                std::string sbar = "[" + progress_bar(current_sample_in_epoch_, total_samples_in_epoch_, 34) + "]";
                int spct = current_sample_in_epoch_ * 100 / total_samples_in_epoch_;
                ss << " " << sbar << " " << std::setw(3) << spct << "%"
                   << "  " << current_sample_in_epoch_ << "/" << total_samples_in_epoch_;
            } else {
                ss << " [" << std::string(34, '.') << "]   0%   0/" << total_samples_in_epoch_ << "  waiting...";
            }
            row(ss.str());
        }
        // Item loss + running avg (colored, trend icon)
        {
            std::ostringstream ss;
            if (current_sample_in_epoch_ > 0) {
                std::string il = fmt_f(current_item_loss_);
                std::string rl = fmt_f(running_sample_loss_);
                ss << " Item loss: "
                   << cpad(colorize(il, loss_color(current_item_loss_)),   (int)il.size(), 8)
                   << " " << trend(current_item_loss_, prev_item_loss_draw_)
                   << "  Running avg: "
                   << cpad(colorize(rl, loss_color(running_sample_loss_)), (int)rl.size(), 8)
                   << " " << trend(running_sample_loss_, prev_running_loss_draw_);
            } else {
                ss << " Item loss: --          Running avg: --";
            }
            row(ss.str());
        }
        // Item PPL + running PPL (colored, trend icon)
        {
            std::ostringstream ss;
            if (current_sample_in_epoch_ > 0) {
                float ippl = std::exp(current_item_loss_);
                float rppl = std::exp(running_sample_loss_);
                std::string ip = fmt_f(ippl, 2);
                std::string rp = fmt_f(rppl, 2);
                ss << " Item PPL : "
                   << cpad(colorize(ip, ppl_color(ippl)), (int)ip.size(), 8)
                   << " " << trend(ippl, prev_item_ppl_draw_)
                   << "  Running PPL: "
                   << cpad(colorize(rp, ppl_color(rppl)), (int)rp.size(), 8)
                   << " " << trend(rppl, prev_running_ppl_draw_);
            } else {
                ss << " Item PPL : --          Running PPL: --";
            }
            row(ss.str());
        }
        // Grad norm (colored, trend icon) + throughput + ETA
        {
            std::ostringstream ss;
            if (current_sample_in_epoch_ > 0) {
                // Use (sample - 1) as the completed count since the epoch timer
                // is reset on the sample-1 callback entry, so elapsed is ~0 then.
                int completed = current_sample_in_epoch_ - 1;
                double epoch_elapsed = duration<double>(now - epoch_start_time_steady_).count();
                double sps = (completed > 0 && epoch_elapsed > 0.0)
                             ? static_cast<double>(completed) / epoch_elapsed : 0.0;
                double epoch_eta = (sps > 0.0)
                    ? (total_samples_in_epoch_ - current_sample_in_epoch_) / sps : 0.0;
                std::string gn = fmt_f(current_item_grad_norm_);
                std::string sps_str;
                std::string eta_str;
                if (sps > 0.0) {
                    std::ostringstream sps_ss;
                    if (sps >= 1.0) {
                        sps_ss << std::fixed << std::setprecision(1) << sps << " samp/s";
                    } else {
                        // Too slow for per-second display — show per-minute instead
                        sps_ss << std::fixed << std::setprecision(1) << (sps * 60.0) << " samp/m";
                    }
                    sps_str = sps_ss.str();
                    eta_str = format_duration(epoch_eta);
                } else {
                    sps_str = "-- samp/s";
                    eta_str = "--:--";
                }
                ss << " Grad norm: "
                   << cpad(colorize(gn, gnorm_color(current_item_grad_norm_)), (int)gn.size(), 8)
                   << " " << trend(current_item_grad_norm_, prev_gnorm_draw_)
                   << "  " << sps_str
                   << "   ETA epoch: " << eta_str;
            } else {
                ss << " Grad norm: --          -- samp/s   ETA epoch: --:--";
            }
            row(ss.str());
        }
    } else {
        row(" No training data loaded yet.");
    }

    // Update previous sample values for trend arrows on next draw
    prev_item_loss_draw_    = current_item_loss_;
    prev_running_loss_draw_ = running_sample_loss_;
    prev_item_ppl_draw_     = std::exp(current_item_loss_);
    prev_running_ppl_draw_  = std::exp(running_sample_loss_);
    prev_gnorm_draw_        = current_item_grad_norm_;

    // ── Per-epoch metrics ─────────────────────────────────────────────────
    hline("\xe2\x94\x9c", "\xe2\x94\xa4");

    // Loss row
    {
        bool have = !session.per_epoch_losses.empty();
        std::string cl = have ? fmt_f(cur_loss) : "------";
        std::string cv = have ? fmt_f(cur_val)  : "------";
        std::ostringstream ss;
        ss << " Loss     : "
           << cpad(have ? colorize(cl, loss_color(cur_loss)) : cl, (int)cl.size(), 8)
           << " " << (have ? trend(cur_loss, prev_loss) : std::string(" "))
           << "    Val Loss : "
           << cpad(have ? colorize(cv, loss_color(cur_val))  : cv, (int)cv.size(), 8)
           << " " << (have ? trend(cur_val,  prev_val)  : std::string(" "));
        row(ss.str());
    }
    // PPL row
    {
        float cur_ppl  = session.per_epoch_perplexities.empty()            ? 0.0f : session.per_epoch_perplexities.back();
        float cur_vppl = session.per_epoch_validation_perplexities.empty() ? 0.0f : session.per_epoch_validation_perplexities.back();
        float prev_ppl  = (session.per_epoch_perplexities.size() >= 2)
                        ? session.per_epoch_perplexities[session.per_epoch_perplexities.size() - 2] : cur_ppl;
        float prev_vppl = (session.per_epoch_validation_perplexities.size() >= 2)
                        ? session.per_epoch_validation_perplexities[session.per_epoch_validation_perplexities.size() - 2] : cur_vppl;
        std::ostringstream ss;
        if (session.per_epoch_perplexities.empty()) {
            ss << " PPL      : --           Val PPL  : --";
        } else {
            std::string pp  = fmt_f(cur_ppl,  2);
            std::string pvp = fmt_f(cur_vppl, 2);
            ss << " PPL      : "
               << cpad(colorize(pp,  ppl_color(cur_ppl)),  (int)pp.size(),  8)
               << " " << trend(cur_ppl,  prev_ppl)
               << "    Val PPL  : "
               << cpad(colorize(pvp, ppl_color(cur_vppl)), (int)pvp.size(), 8)
               << " " << trend(cur_vppl, prev_vppl);
        }
        row(ss.str());
    }
    // LR row (cyan — informational, no threshold)
    {
        std::ostringstream lr_ss;
        lr_ss << std::scientific << std::setprecision(3) << cur_lr;
        std::ostringstream ss;
        ss << " LR       : " << colorize(lr_ss.str(), C_CYAN)
           << "    Epoch dur: " << std::fixed << std::setprecision(1) << last_t << "s";
        row(ss.str());
    }
    // Best val row (always green — it's the best we've done)
    {
        float best_ppl = std::numeric_limits<float>::max();
        for (float v : session.per_epoch_validation_perplexities) best_ppl = std::min(best_ppl, v);
        bool have_best = (best_val < std::numeric_limits<float>::max());
        std::string bv = have_best ? fmt_f(best_val) : "------";
        std::ostringstream ss;
        ss << " Best val : " << (have_best ? colorize(bv, C_GREEN) : bv)
           << "  Best PPL : "
           << (session.per_epoch_validation_perplexities.empty()
                   ? std::string("--")
                   : colorize(fmt_f(best_ppl, 2), C_GREEN));
        row(ss.str());
    }
    {
        std::ostringstream ss;
        ss << " Avg epoch: " << format_duration(avg_epoch_time);
        row(ss.str());
    }

    hline("\xe2\x94\x94", "\xe2\x94\x98");

    std::cout << std::flush;
}

// ============================================================================
// Metrics API Server Management
// ============================================================================

