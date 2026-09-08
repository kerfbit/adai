// @adai-status: beta        (large, actively evolving core trainer)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07

#include "IncrementalTrainer.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "Config.hpp"
#include "Logger.hpp"
#include "ModelNameClient.hpp"  // always included for complete type (unique_ptr destructor)
#include "TrainingMetricsAPI.hpp"

IncrementalTrainer::~IncrementalTrainer() = default;
#ifdef ADAI_ENABLE_OPENMP
#include <omp.h>
#include <cmath>
#endif

// Bring Logger into scope without qualifying every call
using adai::Logger;

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

// Cache key for ChatbotTrainer's on-disk tokenized-data cache (see
// ChatbotTrainer::preprocess_data()). Built from everything that affects
// tokenization output: each input file's content fingerprint (so a changed
// dataset invalidates the cache), the vocab file's fingerprint (so a
// retrained/rebuilt vocab invalidates it), and the tokenizer mode + max
// sequence length (both affect encoding/truncation directly). Files are
// sorted first so the same file set in a different acquire order still
// produces the same key. Hashed (not used verbatim) to keep the resulting
// cache filename short and filesystem-safe regardless of how many files or
// how long their paths are.
std::string compute_tokenized_cache_key(std::vector<std::string> files,
                                        const std::string& vocab_path,
                                        TokenizerMode tokenizer_mode, int max_seq_length) {
    std::sort(files.begin(), files.end());
    std::ostringstream oss;
    for (const auto& f : files) {
        oss << f << ':' << DatasetRegistry::compute_checksum(f) << '|';
    }
    oss << "vocab:" << DatasetRegistry::compute_checksum(vocab_path) << '|'
        << "mode:" << static_cast<int>(tokenizer_mode) << '|' << "maxlen:" << max_seq_length;

    std::ostringstream hex;
    hex << std::hex << std::hash<std::string>{}(oss.str());
    return hex.str();
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
std::string derive_metrics_session_label(int session_id, const std::string& model_path) {
    // Stem: filename without extension
    std::string stem = fs::path(model_path).stem().string();
    if (stem.empty())
        stem = "model";

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
 * for Prometheus labels and session history.
 */
std::string build_config_snapshot(const IncrementalConfig& cfg) {
    const TrainingConfig& bc = cfg.base_config;
    std::ostringstream json;
    json << std::fixed;
    json << "{" << "\"d_model\":" << bc.d_model << ",\"heads\":" << bc.num_heads
         << ",\"d_ff\":" << bc.d_ff << ",\"enc_layers\":" << bc.num_encoder_layers
         << ",\"dec_layers\":" << bc.num_decoder_layers << ",\"lr\":" << bc.learning_rate
         << ",\"batch\":" << bc.batch_size << ",\"grad_accum\":" << bc.gradient_accumulation_steps
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

    auto tok = std::make_unique<BPETokenizer>(config.base_config.tokenizer_mode);
    tok->load_vocab(vocab_path_);
    Logger::info("Tokenizer loaded (vocab size: {}, mode: {})", tok->get_vocab_size(),
                 tok->is_unicode_mode() ? "unicode" : "ascii");

    model = std::make_unique<EncoderDecoderModel>(
        tok->get_vocab_size(), config.base_config.d_model, config.base_config.num_encoder_layers,
        config.base_config.num_decoder_layers, config.base_config.num_heads,
        config.base_config.d_ff, config.base_config.max_seq_length);
    model->set_tokenizer(tok.release());
}

// ============================================================================
// bootstrap_vocab() — build vocabulary from training data on first run.
// ============================================================================
bool IncrementalTrainer::bootstrap_vocab(const std::vector<ConversationPair>& pairs) {
    Logger::info("Building initial vocabulary from {} conversation pairs → {}", pairs.size(),
                 vocab_path_);
    std::vector<std::string> texts;
    texts.reserve(pairs.size() * 2);
    for (const auto& p : pairs) {
        if (!p.input.empty())
            texts.push_back(p.input);
        if (!p.response.empty())
            texts.push_back(p.response);
    }
    if (texts.empty()) {
        Logger::error("Cannot build vocabulary: training data contains no text");
        return false;
    }

    // Determine target size: explicit override or data-driven recommendation.
    int target_size = vocab_build_size_;
    if (target_size <= 0) {
        target_size = BPETokenizer::recommend_vocab_size(
            texts, config.base_config.d_model, config.base_config.num_encoder_layers,
            config.base_config.num_decoder_layers, config.base_config.max_seq_length,
            config.base_config.tokenizer_mode);
        Logger::info("Auto-sized vocabulary target: {} tokens", target_size);
        Logger::info("  (corpus: {} texts | d_model: {} | layers: {}enc+{}dec | seq_len: {})",
                     texts.size(), config.base_config.d_model,
                     config.base_config.num_encoder_layers, config.base_config.num_decoder_layers,
                     config.base_config.max_seq_length);
    } else {
        Logger::info("Using explicit vocabulary target: {} tokens", target_size);
    }

    BPETokenizer tok(config.base_config.tokenizer_mode);
    tok.build_vocab(texts, target_size);
    tok.save_vocab(vocab_path_);

    // Measure and report fertility so users can judge quality.
    float fertility = tok.measure_fertility(texts);
    Logger::info("Vocabulary saved to '{}' ({} tokens, fertility {:.2f} tokens/word)", vocab_path_,
                 tok.get_vocab_size(), fertility);
    if (fertility > 2.5f) {
        Logger::warn(
            "High fertility ({:.2f}): vocabulary may be too small for this corpus; "
            "consider setting VOCAB_BUILD_SIZE to a larger value",
            fertility);
    } else if (fertility < 1.1f && fertility > 0.0f) {
        Logger::warn("Low fertility ({:.2f}): vocabulary may be oversized for this corpus",
                     fertility);
    }

    pending_vocab_build_ = false;
    return true;
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
    cfg.base_config.gradient_accumulation_steps = svc.gradient_accumulation_steps;
    cfg.base_config.enable_early_stopping = true;
    cfg.base_config.patience = 5;
    cfg.base_config.restore_best_weights = true;
    cfg.base_config.log_level = LogLevel::NORMAL;

    // Metrics push configuration
    cfg.metrics_server_url = svc.metrics_server_url;
    cfg.metrics_push_timeout_ms = svc.metrics_push_timeout_ms;
    cfg.metrics_heartbeat_interval_ms = svc.metrics_heartbeat_interval_ms;
    cfg.metrics_session_label = svc.metrics_session_label;

    // Model Name Service configuration
    cfg.mns_server_url = svc.name_service_url;
    cfg.mns_model_name = svc.model_name;

    // Generation quality metrics
    cfg.base_config.enable_generation_quality_metrics = svc.enable_generation_quality_metrics;
    cfg.base_config.generation_quality_sample_size = svc.generation_quality_sample_size;
    cfg.base_config.generation_quality_max_tokens = svc.generation_quality_max_tokens;
    cfg.base_config.generation_quality_async_threshold = svc.generation_quality_async_threshold;

    // Session directory
    if (!svc.session_dir.empty()) {
        cfg.session_dir = svc.session_dir;
    }

    // Auto-save / checkpoint retention — previously always used the
    // hardcoded IncrementalConfig defaults regardless of config.conf.
    cfg.auto_save_enabled = svc.auto_save_enabled;
    cfg.auto_save_every_samples = svc.auto_save_every_samples;
    cfg.auto_save_every_minutes = svc.auto_save_every_minutes;
    cfg.max_sessions_to_keep = svc.max_sessions_to_keep;

    // Tokenizer mode
    cfg.base_config.tokenizer_mode =
        svc.unicode_tokenizer ? TokenizerMode::UNICODE : TokenizerMode::ASCII;

    // Distributed dataset registry configuration — reuse the existing mapping so
    // dataset_config_ (below) is correctly populated regardless of which
    // IncrementalTrainer constructor a caller uses (fixes resume/reset silently
    // ignoring REGISTRY_SERVER_URL; see CLAUDE.md "Distributed Dataset Registry").
    cfg.dataset = DatasetRegistry::make_config(svc);

    return cfg;
}

// ============================================================================
// Primary constructor — config.conf is the required entry point.
// ============================================================================
IncrementalTrainer::IncrementalTrainer(const std::string& config_file_path)
    : current_session_id(0),
      samples_since_last_save(0),
      best_validation_loss(std::numeric_limits<float>::max()),
      best_checkpoint_path("") {
    Logger::info("Loading configuration from: {}",
                 config_file_path.empty() ? "<system default>" : config_file_path);

    adai::ServiceConfig svc = config_file_path.empty() ? adai::ConfigLoader::load()
                                                       : adai::ConfigLoader::load(config_file_path);

    model_path_ = svc.model_path.empty() ? "chatbot_model.bin" : svc.model_path;

    // Derive vocab_path from model_name (canonical name) or model_path stem when not configured.
    if (!svc.vocab_path.empty()) {
        vocab_path_ = svc.vocab_path;
    } else if (!svc.model_name.empty()) {
        vocab_path_ = svc.model_name + ".vocab";
        Logger::info("VOCAB_PATH not configured; derived from model name: {}", vocab_path_);
    } else {
        vocab_path_ = fs::path(model_path_).stem().string() + ".vocab";
        Logger::info("VOCAB_PATH not configured; derived from model path: {}", vocab_path_);
    }

    config = make_incremental_config(svc);
    config.base_config.lr_schedule = LRSchedule::WARMUP_COSINE;
    vocab_build_size_ = svc.vocab_build_size;
    dataset_config_ = config.dataset;
    // Threaded through to ChatbotTrainer via config.base_config (its constructor
    // takes only TrainingConfig) — tokenized_cache_key itself is set fresh per
    // train_on_files()/retrain_on_files() call, see there.
    config.base_config.cache_tokenized_data = dataset_config_.cache_tokenized_data;
    config.base_config.tokenized_cache_dir = dataset_config_.tokenized_cache_dir;

    // MetricsPushClient is created per training run; use NullMetricsReporter until then.
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    ensure_directories_exist();

    if (!fs::exists(vocab_path_)) {
        Logger::info("Vocabulary '{}' not found — will be built from first training run",
                     vocab_path_);
        pending_vocab_build_ = true;
        last_save_time = std::chrono::system_clock::now();
        return;
    }

    build_model();
    load_session_history();

    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (is_sane_checkpoint_candidate(session) &&
                session.final_validation_loss < best_validation_loss) {
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

    // Now that this run's resume decision is locked in, sweep away anything
    // left behind by a crashed prior attempt.
    cleanup_dead_sessions();

    last_save_time = std::chrono::system_clock::now();
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
      best_checkpoint_path("") {
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
            if (is_sane_checkpoint_candidate(session) &&
                session.final_validation_loss < best_validation_loss) {
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

    cleanup_dead_sessions();

    last_save_time = std::chrono::system_clock::now();
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
      best_checkpoint_path("") {
    Logger::info("Initializing Incremental Training System...");

    // set config FIRST so build_model() uses the correct architecture
    dataset_config_ = config.dataset;
    // Threaded through to ChatbotTrainer via config.base_config (its constructor
    // takes only TrainingConfig) — tokenized_cache_key itself is set fresh per
    // train_on_files()/retrain_on_files() call, see there.
    config.base_config.cache_tokenized_data = dataset_config_.cache_tokenized_data;
    config.base_config.tokenized_cache_dir = dataset_config_.tokenized_cache_dir;

    // MetricsPushClient is created per training run; use NullMetricsReporter until then.
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    build_model();

    ensure_directories_exist();
    load_session_history();

    if (!session_history.empty()) {
        for (const auto& session : session_history) {
            if (is_sane_checkpoint_candidate(session) &&
                session.final_validation_loss < best_validation_loss) {
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

    cleanup_dead_sessions();

    last_save_time = std::chrono::system_clock::now();
}

void IncrementalTrainer::set_config(const IncrementalConfig& cfg) {
    config = cfg;
}

IncrementalConfig& IncrementalTrainer::get_config() {
    return config;
}

void IncrementalTrainer::reset_model_for_config() {
    // A freshly-built model has no history to compare against — clear tracking
    // state too, otherwise the next session inherits a stale best_validation_loss
    // (and best_checkpoint_path pointing at a now-irrelevant checkpoint) from
    // whatever was previously in session_history.txt.
    session_history.clear();
    current_session_id = 0;
    samples_since_last_save = 0;
    best_validation_loss = std::numeric_limits<float>::max();
    best_checkpoint_path.clear();

    build_model();
    save_session_history();
    Logger::info("Model reinitialized with fresh weights (architecture reset)");
}

std::string IncrementalTrainer::begin_run(bool is_retrain) {
    current_run_id_.clear();
    current_mns_session_id_.clear();

#ifdef BUILD_MNS_SERVER
    if (!config.mns_server_url.empty() && !config.mns_model_name.empty()) {
        if (!mns_client_)
            mns_client_ = std::make_unique<adai::ModelNameClient>(config.mns_server_url, 5000);
        try {
            // metrics_session_key isn't known yet at this point (the metrics
            // push session starts later, inside run_training()) — MNS's
            // TrainingHistoryEntry.metrics_session_key is best-effort
            // correlation, not required for run/session numbering.
            current_run_id_ = mns_client_->set_training(config.mns_model_name, is_retrain);
            Logger::info("MNS allocated run_id '{}' for model '{}' ({})", current_run_id_,
                         config.mns_model_name, is_retrain ? "retrain" : "continuing");
        } catch (const std::exception& e) {
            Logger::warn("MNS set_training failed: {}", e.what());
        }
    }

    if (!current_run_id_.empty()) {
        try {
            DatasetRegistry reg(dataset_config_);
            current_mns_session_id_ = reg.next_session(config.mns_model_name, current_run_id_);
            Logger::info("Registry allocated session_id '{}' for run '{}'",
                         current_mns_session_id_, current_run_id_);
        } catch (const std::exception& e) {
            Logger::warn("DatasetRegistry::next_session failed: {}", e.what());
        }
    }
#else
    (void)is_retrain;
#endif

    if (control_) {
        control_->set_run_id(current_run_id_);
        control_->set_session_id(current_mns_session_id_);
        control_->set_model_name(config.mns_model_name);
    }

    return current_run_id_;
}

bool IncrementalTrainer::run_training(ChatbotTrainer& trainer, int num_epochs,
                                      int metrics_sample_count, int finalize_sample_count,
                                      bool enable_best_model_snapshot, bool reset_best_tracking) {
    // Start metrics session now — tokenization is already done so the server
    // timeout window covers only the actual training wait.
    // Retry up to 3 times with a suffix if the server returns 409 Conflict.
    if (!config.metrics_server_url.empty()) {
        const std::string base_key =
            sanitize_session_key(derive_metrics_session_key(current_session_id + 1));
        std::string session_key = base_key;
        std::unique_ptr<MetricsPushClient> pc;
        int rc = 0;
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (attempt > 0) {
                session_key = base_key + "-" + std::to_string(attempt + 1);
                Logger::warn("Metrics key conflict (409), retrying with '{}'", session_key);
                std::this_thread::sleep_for(std::chrono::milliseconds(100 * attempt));
            }
            const std::string push_url =
                build_metrics_session_push_base(config.metrics_server_url, session_key);
            pc = std::make_unique<MetricsPushClient>(push_url, config.metrics_push_timeout_ms, 1024,
                                                     config.metrics_heartbeat_interval_ms);
            const std::string label =
                config.metrics_session_label.empty()
                    ? derive_metrics_session_label(current_session_id + 1, model_path_)
                    : config.metrics_session_label;
            const std::string snapshot = build_config_snapshot(config);
            rc = pc->start_session(current_session_id + 1, num_epochs, metrics_sample_count, label,
                                   snapshot, reset_best_tracking);
            if (rc != 409)
                break;
        }

        if (rc >= 200 && rc < 300) {
            active_session_key_ = session_key;
            push_client_ = pc.get();
            metrics_reporter_ = std::move(pc);
        } else {
            // The server never created a session for this key (registry full, 409 exhausted
            // after 3 attempts, connection failure, etc.) — pushing epoch/sample metrics to a
            // session that doesn't exist would fail just as silently as this did, so training
            // would run to completion with no visible record on the metrics server. Fail loud
            // here instead of wiring up a push client that quietly drops everything.
            Logger::error(
                "Metrics session/start failed for key '{}' (HTTP {}) — training will proceed "
                "WITHOUT metrics reporting for this run",
                session_key, rc);
            push_client_ = nullptr;
            active_session_key_.clear();
            metrics_reporter_ = std::make_unique<NullMetricsReporter>();
        }
    } else {
        push_client_ = nullptr;
        active_session_key_.clear();
        metrics_reporter_ = std::make_unique<NullMetricsReporter>();
    }
    trainer.set_metrics_reporter(metrics_reporter_.get());

    // `serve`'s admin API: /admin/pause sets control_->paused, which doubles
    // as ChatbotTrainer's cooperative abort flag — checked only at
    // optimizer-step boundaries, so a pause never leaves a half-applied
    // gradient update. No-op (nullptr) outside of `serve`.
    if (control_) {
        trainer.set_abort_flag(&control_->paused);
        control_->phase = adai::TrainerPhase::Tokenizing;
        control_->total_epochs = num_epochs;
        control_->current_epoch = 0;
        control_->log(adai::TrainerLogLevel::Info,
                      "Starting training pass: tokenizing (" + std::to_string(num_epochs) +
                          " epoch(s) requested)");
    }

    // MNS training-lock/run_id (and, transitively, the registry session_id) are
    // obtained by begin_run() — called by the caller (train/retrain/resume
    // command handlers) before any data is acquired, so the dataset-ownership
    // run_id and MNS's run_id are the same canonical identifier. current_run_id_
    // is empty here when MNS isn't configured; every MNS call below is a no-op
    // in that case.

    // Per-epoch callback: record timing and per-epoch metrics into session
    // history, and push progress to MNS so a killed/crashed trainer still
    // leaves an accurate last-known state (see CLAUDE.md "Configuration").
    auto epoch_start = std::chrono::steady_clock::now();
    int epochs_fully_completed = 0;
    float session_best_val_loss = std::numeric_limits<float>::max();
    trainer.set_epoch_callback([this, &epoch_start, &epochs_fully_completed,
                                &session_best_val_loss](int epoch, int total, float loss,
                                                        float val_loss, float lr) {
        auto now = std::chrono::steady_clock::now();
        double epoch_secs = std::chrono::duration<double>(now - epoch_start).count();
        epoch_start = now;
        epochs_fully_completed = epoch;
        if (!session_history.empty()) {
            auto& session = session_history.back();
            session.per_epoch_losses.push_back(loss);
            session.per_epoch_validation_losses.push_back(val_loss);
            session.per_epoch_learning_rates.push_back(lr);
            session.training_time_per_epoch.push_back(epoch_secs);
            session.per_epoch_perplexities.push_back(std::exp(loss));
            session.per_epoch_validation_perplexities.push_back(std::exp(val_loss));
        }
        if (std::isfinite(val_loss) && val_loss < session_best_val_loss) {
            session_best_val_loss = val_loss;
        }
        if (control_) {
            control_->current_epoch = epoch;
            if (std::isfinite(val_loss)) {
                control_->best_loss = static_cast<double>(session_best_val_loss);
            }
        }
#ifdef BUILD_MNS_SERVER
        if (mns_client_ && !current_run_id_.empty()) {
            try {
                mns_client_->push_progress(config.mns_model_name, current_run_id_,
                                           current_mns_session_id_, epoch,
                                           static_cast<double>(loss),
                                           static_cast<double>(session_best_val_loss));
            } catch (const std::exception& e) {
                Logger::warn("MNS push_progress failed: {}", e.what());
            }
        }
#endif
    });

    // Reset epoch timer on the first sample to exclude data-loading time, and
    // drive TD-005 auto-save so a crash mid-run doesn't lose all progress —
    // previously should_auto_save()/perform_auto_save() were never called from
    // anywhere, so samples_since_last_save never advanced and nothing was ever
    // persisted before finalize_session() at the very end of the whole run.
    //
    // last_save_time is also (re)armed here, not just at construction/session-init
    // (see initialize_session()) — both of those run before dataset download and
    // tokenization, which for a large dataset can take hours. Left as-is,
    // should_auto_save()'s time-based check (auto_save_every_minutes, default 30)
    // was already satisfied by the time the very first sample finished, so
    // perform_auto_save() — and the checkpoint write inside it — fired on sample 1
    // of every run, before a full batch had ever been trained. Resetting it here
    // means the auto-save clock reflects actual training time elapsed, not
    // setup/download/tokenize time.
    int cumulative_samples_this_session = 0;
    trainer.set_sample_callback([this, &epoch_start, &epochs_fully_completed,
                                 &cumulative_samples_this_session](int sample, int, float running_loss,
                                                                    float, float, float) {
        if (sample == 1) {
            epoch_start = std::chrono::steady_clock::now();
            last_save_time = std::chrono::system_clock::now();
            if (control_) {
                // sample==1 fires at the start of every epoch (the loop index
                // resets each epoch), so only log once, on the true first
                // optimizer step of the whole pass — epochs_fully_completed
                // only advances once epoch_callback_ has fired.
                if (epochs_fully_completed == 0) {
                    control_->log(adai::TrainerLogLevel::Info, "Training started");
                }
                control_->phase = adai::TrainerPhase::Training;
            }
        }

        ++cumulative_samples_this_session;
        ++samples_since_last_save;
        if (control_) {
            control_->samples_trained_this_pass = cumulative_samples_this_session;
            control_->last_loss = static_cast<double>(running_loss);
            // Forced checkpoint (POST /admin/checkpoint) — independent of the
            // auto-save cadence, consumed once via exchange().
            if (control_->checkpoint_requested.exchange(false)) {
                control_->phase = adai::TrainerPhase::Checkpointing;
                perform_auto_save(epochs_fully_completed + 1, cumulative_samples_this_session);
                control_->phase = adai::TrainerPhase::Training;
            }
        }
        if (should_auto_save()) {
            perform_auto_save(epochs_fully_completed + 1, cumulative_samples_this_session);
        }
    });

    if (enable_best_model_snapshot) {
        std::ostringstream oss;
        oss << get_session_dir() << "/session_" << current_session_id << "_best.bin";
        std::string best_path = oss.str();
        trainer.set_best_model_callback([this, &trainer, best_path](int epoch, float val_loss) {
            try {
                trainer.save_to(best_path);
                Logger::info("  [best] New best (epoch {}, val_loss {:.4f}) saved to: {}", epoch,
                             val_loss, best_path);
            } catch (const std::exception& e) {
                Logger::warn("  [warn] Failed to save best snapshot: {}", e.what());
            }
        });
    }

    Logger::info("Training for {} epochs...", num_epochs);
    bool success = trainer.train(num_epochs);
    model = trainer.release_model();

    // A pass drained via /admin/pause (control_->paused) ends with train()
    // returning true (no exception; it just did fewer epochs than asked) —
    // was_aborted() is the only signal that this wasn't a normal completion.
    // Treat it like the existing failure path: save a checkpoint of whatever
    // progress was made, but do NOT finalize_session()/mark_trained() — the
    // caller (train_on_files/retrain_on_files/resume_last_session) releases
    // the claimed files back to pending on a false return, exactly as it
    // already does for a genuine failure, so the next pass (this process
    // after /admin/resume, or a future one) picks the same files back up.
    const bool aborted = trainer.was_aborted();
    if (aborted) {
        const std::string message = "Training pass paused via admin API (epoch " +
            std::to_string(epochs_fully_completed) + ", " +
            std::to_string(cumulative_samples_this_session) +
            " sample(s) this pass) — checkpointing and releasing claimed files back to pending";
        if (control_) {
            control_->log(adai::TrainerLogLevel::Warn, message);
            control_->phase = adai::TrainerPhase::Checkpointing;
        } else {
            Logger::info("{}", message);
        }
        perform_auto_save(epochs_fully_completed + 1, cumulative_samples_this_session);
        if (control_) {
            control_->phase = adai::TrainerPhase::Idle;
        }
    }

    if (push_client_) {
        push_client_->end_session();
        push_client_ = nullptr;
    }
    metrics_reporter_ = std::make_unique<NullMetricsReporter>();

    if (success && !aborted) {
        std::string checkpoint_path = generate_session_checkpoint_path();
        save_model(checkpoint_path);

        float final_loss = trainer.get_final_training_loss();
        float final_val_loss = trainer.get_final_validation_loss();
        finalize_session(finalize_sample_count, num_epochs, final_loss, final_val_loss);
        save_session_history();

#ifdef BUILD_MNS_SERVER
        // MNS: notify candidate with checkpoint artifact
        if (mns_client_ && !current_run_id_.empty()) {
            adai::ArtifactLocation artifact;
            artifact.path = checkpoint_path;
            artifact.format = "adai-native";
            const std::map<std::string, std::string> summary = {
                {"final_loss", std::to_string(static_cast<double>(final_loss))},
                {"final_val_loss", std::to_string(static_cast<double>(final_val_loss))},
                {"epochs", std::to_string(num_epochs)}};
            try {
                mns_client_->set_candidate(config.mns_model_name, current_run_id_, artifact,
                                          summary);
            } catch (const std::exception& e) {
                Logger::warn("MNS set_candidate failed: {}", e.what());
            }
        }
#endif
        if (control_) {
            control_->log(adai::TrainerLogLevel::Info, "Training pass complete");
        }
    } else if (!aborted && control_) {
        // success == false here — a genuine failure (exception inside
        // ChatbotTrainer::train(), not an admin-requested pause).
        control_->log(adai::TrainerLogLevel::Error, "Training pass failed");
    }

    // The aborted branch above already returned control_ to Idle; cover the
    // normal-completion and genuine-failure paths here so `serve`'s
    // /admin/status always reads Idle once run_training() has returned,
    // regardless of which path was taken.
    if (control_ && !aborted) {
        control_->phase = adai::TrainerPhase::Idle;
    }

    return success && !aborted;
}

bool IncrementalTrainer::train_on_files(const std::vector<std::string>& files, int num_epochs) {
    const std::string start_message = "Starting incremental training session #" +
        std::to_string(current_session_id + 1) + " (" + std::to_string(files.size()) + " file(s))";
    if (control_) {
        control_->log(adai::TrainerLogLevel::Info, start_message);
        control_->phase = adai::TrainerPhase::LoadingData;
    } else {
        Logger::info("{}", start_message);
    }
    initialize_session();

    // Load files in parallel, then merge in submission order. Do NOT pre-split here:
    // ChatbotTrainer::train() performs its own validation split internally
    // (split_data(), called once per train() invocation whenever validation_data is
    // empty). Pre-splitting here as well used to carve out a second, redundant
    // validation set from what was left after this split — silently discarding the
    // first split's validation_pairs (never fed to the trainer) and leaving only
    // ~81% of the loaded data actually used for training.
    std::vector<ConversationPair> all_pairs;
    {
        const int n_files = static_cast<int>(files.size());
        std::vector<std::vector<ConversationPair>> per_file(n_files);
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int fi = 0; fi < n_files; ++fi)
            load_conversation_pairs(files[fi], per_file[fi]);

        for (int fi = 0; fi < n_files; ++fi)
            all_pairs.insert(all_pairs.end(), per_file[fi].begin(), per_file[fi].end());
    }

    // Mirrors ChatbotTrainer::split_data()'s arithmetic so the metrics session is told
    // the real post-split training size up front, without duplicating the actual split.
    const int total_loaded = static_cast<int>(all_pairs.size());
    const int val_split = config.base_config.validation_split;
    const int expected_val_size = (val_split > 0) ? (total_loaded / val_split) : 0;
    const int expected_train_size = total_loaded - expected_val_size;
    Logger::info("Total samples loaded: {}", total_loaded);
    Logger::info("Expected split — training: {}, validation: {}", expected_train_size,
                 expected_val_size);

    // First-run vocabulary bootstrap: build vocab from all available text then construct model.
    if (pending_vocab_build_) {
        if (!bootstrap_vocab(all_pairs)) {
            Logger::error("Vocabulary bootstrap failed; aborting training");
            return false;
        }
        build_model();
        load_session_history();
    }

    config.base_config.tokenized_cache_key = compute_tokenized_cache_key(
        files, vocab_path_, config.base_config.tokenizer_mode, config.base_config.max_seq_length);
    ChatbotTrainer trainer(config.base_config);
    trainer.set_model(std::move(model));
    for (const auto& pair : all_pairs)
        trainer.add_training_pair(pair.input, pair.response);

    // Each call gets its own metrics session key (derived from current_session_id),
    // so there is never a genuine same-lineage session to preserve best-loss
    // continuity with — reset_best_tracking=true avoids inheriting a stale
    // best_validation_loss from an unrelated prior session (TD-018: the metrics
    // server keeps a single global snapshot, not one scoped per session key).
    bool success = run_training(trainer, num_epochs, expected_train_size, expected_train_size,
                                /*enable_best_model_snapshot=*/true,
                                /*reset_best_tracking=*/true);
    if (success) {
        Logger::info("Incremental training session completed successfully");
        print_training_summary();
    }
    return success;
}

bool IncrementalTrainer::retrain_on_files(const std::vector<std::string>& files, int num_epochs) {
    const std::string start_message = "Starting full retrain on " + std::to_string(files.size()) + " file(s)";
    if (control_) {
        control_->log(adai::TrainerLogLevel::Info, start_message);
        control_->phase = adai::TrainerPhase::LoadingData;
    } else {
        Logger::info("{}", start_message);
    }
    initialize_session();

    // Load files in parallel, then merge in order for deterministic dataset ordering.
    std::vector<ConversationPair> all_pairs;
    {
        const int n_files = static_cast<int>(files.size());
        std::vector<std::vector<ConversationPair>> per_file(n_files);
#ifdef ADAI_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int fi = 0; fi < n_files; ++fi)
            load_conversation_pairs(files[fi], per_file[fi]);

        for (int fi = 0; fi < n_files; ++fi)
            all_pairs.insert(all_pairs.end(), per_file[fi].begin(), per_file[fi].end());
    }
    Logger::info("Total samples: {}", all_pairs.size());

    // First-run vocabulary bootstrap.
    if (pending_vocab_build_) {
        if (!bootstrap_vocab(all_pairs)) {
            Logger::error("Vocabulary bootstrap failed; aborting training");
            return false;
        }
        build_model();
        load_session_history();
    }

    // Validation split is handled inside ChatbotTrainer; compute the training
    // count here only to report an accurate sample count to the metrics server.
    const int total = static_cast<int>(all_pairs.size());
    const int val_size = total / config.base_config.validation_split;
    const int training_sample_count = total - val_size;

    config.base_config.tokenized_cache_key = compute_tokenized_cache_key(
        files, vocab_path_, config.base_config.tokenizer_mode, config.base_config.max_seq_length);
    ChatbotTrainer trainer(config.base_config);
    trainer.set_model(std::move(model));
    for (const auto& pair : all_pairs)
        trainer.add_training_pair(pair.input, pair.response);

    return run_training(trainer, num_epochs, training_sample_count, total,
                        /*enable_best_model_snapshot=*/true, /*reset_best_tracking=*/true);
}

bool IncrementalTrainer::resume_last_session() {
    DatasetRegistry reg(dataset_config_);
    // Prefer the MNS-allocated run_id (set by begin_run(), called by the
    // caller before resume_last_session()) so dataset-registry file ownership
    // uses the same canonical run_id as MNS; fall back to the local
    // RUN_ID/hostname+pid-derived value when MNS isn't configured.
    std::string run_id = !current_run_id_.empty() ? current_run_id_ : dataset_config_.run_id;
    if (run_id.empty()) {
        run_id = detect_hostname_fragment() + "_" + std::to_string(detect_pid_mod_10000());
    }
    auto resp = reg.acquire_pending(run_id);
    if (resp.files.empty()) {
        Logger::warn("No pending data files to resume training on");
        return false;
    }
    // resume_last_session runs on the same machine as the registry (localhost path);
    // ftp_server_host is always empty here.  Use registry paths directly.
    std::vector<std::string> pending = resp.registry_paths();

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
        std::vector<int> counts(pending.size(), 0);
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

bool IncrementalTrainer::is_sane_checkpoint_candidate(const TrainingSession& session) {
    // Zero-sample sessions never ran validation, so final_validation_loss is
    // left at its zero-initialized default rather than a real measurement.
    if (session.samples_trained <= 0) {
        return false;
    }
    // NaN/Inf means the run diverged (or crashed mid-computation) before
    // producing a usable loss; an exact 0.0 is not an achievable cross-entropy
    // loss and indicates the value was never actually written.
    if (!std::isfinite(session.final_validation_loss) || session.final_validation_loss <= 0.0f) {
        return false;
    }
    // A checkpoint whose required sidecar files have been deleted (or never
    // fully written) out from under its history entry can't be resumed from.
    // EncoderDecoderModel::save_model() never writes a bare file at
    // session.checkpoint_path itself (no extension) — only the five sidecars
    // below — so checking for that bare path here would reject every
    // legitimately-saved checkpoint unconditionally. load_model() requires
    // all five to succeed (see EncoderDecoderModel.cpp), so check all five,
    // not just .config.
    if (session.checkpoint_path.empty()) {
        return false;
    }
    static const std::array<const char*, 5> required_sidecars = {".config", ".vocab", ".encoder",
                                                                  ".decoder", ".lm_head"};
    for (const char* ext : required_sidecars) {
        if (!fs::exists(session.checkpoint_path + ext)) {
            return false;
        }
    }
    return true;
}

void IncrementalTrainer::cleanup_dead_sessions() {
    const std::string sdir = get_session_dir();
    std::error_code ec;
    if (!fs::exists(sdir, ec)) {
        return;
    }

    // Checkpoints referenced by a (still-to-be-vetted) history entry are the
    // only "session_N_checkpoint.bin" files considered live at this point.
    std::set<std::string> known_checkpoints;
    for (const auto& session : session_history) {
        known_checkpoints.insert(session.checkpoint_path);
    }

    for (const auto& dir_entry : fs::directory_iterator(sdir, ec)) {
        if (ec) {
            break;
        }
        const auto& path = dir_entry.path();
        if (path.extension().string() != ".bin") {
            continue;  // only inspect base checkpoint files; sidecars follow their base
        }
        const std::string stem = path.stem().string();

        int session_id = -1;
        bool is_best = false;
        bool is_autosave = false;

        if (stem.rfind("session_", 0) == 0) {
            const std::string rest = stem.substr(8);
            const size_t sep = rest.find('_');
            if (sep == std::string::npos) {
                continue;
            }
            try {
                session_id = std::stoi(rest.substr(0, sep));
            } catch (...) {
                continue;
            }
            const std::string kind = rest.substr(sep + 1);
            if (kind == "best") {
                is_best = true;
            } else if (kind != "checkpoint") {
                continue;  // unrecognized "session_*" file — leave it alone
            }
        } else if (stem.rfind("auto_save_session_", 0) == 0) {
            static const std::string autosave_prefix = "auto_save_session_";
            try {
                session_id = std::stoi(stem.substr(autosave_prefix.size()));
            } catch (...) {
                continue;
            }
            is_autosave = true;
        } else {
            continue;
        }

        const std::string full = path.string();
        std::string reason;

        if (is_best || is_autosave) {
            // A completed session's own best-snapshot / autosave is deleted by
            // finalize_session()/cleanup_old_sessions(); any that still exist
            // under an id other than the current in-progress session belong to
            // a run that crashed before reaching that cleanup.
            if (session_id != current_session_id) {
                reason = "orphaned snapshot from a crashed/superseded session";
            }
        } else if (known_checkpoints.find(full) == known_checkpoints.end()) {
            // A bare checkpoint with no matching history line: the run
            // crashed between writing the file and appending its record.
            reason = "checkpoint has no matching session_history entry";
        }

        if (!reason.empty()) {
            Logger::info("Cleaning up dead session artifact: {} ({})", full, reason);
            remove_model_files(full);
        }
    }

    // Drop history entries whose recorded results can no longer be trusted
    // (zero samples trained, non-finite/non-positive loss, or a checkpoint
    // that's gone missing) so they stop being reconsidered on every restart.
    bool history_changed = false;
    for (auto it = session_history.begin(); it != session_history.end();) {
        if (!is_sane_checkpoint_candidate(*it)) {
            Logger::info(
                "Removing broken session {} from history (samples_trained={}, val_loss={})",
                it->session_id, it->samples_trained, it->final_validation_loss);
            remove_model_files(it->checkpoint_path);
            it = session_history.erase(it);
            history_changed = true;
        } else {
            ++it;
        }
    }
    if (history_changed) {
        save_session_history();
    }
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
    const int max_keep = control_ ? control_->max_sessions_to_keep.load() : config.max_sessions_to_keep;
    if (static_cast<int>(session_history.size()) <= max_keep) {
        return;
    }

    int to_remove = static_cast<int>(session_history.size()) - max_keep;

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
                if (is_sane_checkpoint_candidate(remaining_session) &&
                    remaining_session.final_validation_loss < best_validation_loss) {
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
    Logger::info("╔══════════════════════════════════════════════╗");
    Logger::info("║       Incremental Training Summary           ║");
    Logger::info("╚══════════════════════════════════════════════╝");
    Logger::info("  Sessions       : {}", session_history.size());
    Logger::info("  Total samples  : {}", get_total_samples_trained());
    Logger::info("  Total time     : {:.2f} h", get_total_training_time_hours());

    if (config.enable_checkpoint_symlinks) {
        Logger::info("  Checkpoint links:");
        if (fs::exists(config.latest_symlink_name)) {
            std::string line = "    latest: " + config.latest_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.latest_symlink_name))
                line += " -> " + fs::read_symlink(config.latest_symlink_name).string();
            Logger::info("{}", line);
        }
        if (fs::exists(config.best_symlink_name)) {
            std::string line = "    best  : " + config.best_symlink_name;
            if (!is_windows_platform() && fs::is_symlink(config.best_symlink_name))
                line += " -> " + fs::read_symlink(config.best_symlink_name).string();
            Logger::info("{}  (val loss: {:.4f})", line, best_validation_loss);
        }
    }

    for (const auto& s : session_history) {
        Logger::info("  Session #{}  samples={}  epochs={}  loss={:.4f}  val={:.4f}", s.session_id,
                     s.samples_trained, s.epochs_completed, s.final_loss, s.final_validation_loss);
        Logger::info("    checkpoint: {}", s.checkpoint_path);

        if (!s.per_epoch_losses.empty()) {
            float best_val = std::numeric_limits<float>::max();
            double total_t = 0.0;
            for (float v : s.per_epoch_validation_losses)
                best_val = std::min(best_val, v);
            for (double t : s.training_time_per_epoch)
                total_t += t;

            Logger::info("    loss      : {}", make_sparkline(s.per_epoch_losses));
            Logger::info("    val loss  : {}", make_sparkline(s.per_epoch_validation_losses));
            Logger::info("    best val  : {:.4f}", best_val);
            if (!s.training_time_per_epoch.empty()) {
                Logger::info("    epoch time: avg {:.1f}s  total {}",
                             total_t / static_cast<double>(s.training_time_per_epoch.size()),
                             format_duration(total_t));
            }
        }
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
    for (const auto& s : session_history) {
        total += s.samples_trained;
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
    // When running under `serve`, control_'s tunables are live (PUT
    // /admin/config) and take precedence over the config-file values baked
    // in at construction time; otherwise fall back to config as before.
    const bool enabled = control_ ? control_->auto_save_enabled.load() : config.auto_save_enabled;
    if (!enabled) {
        return false;
    }
    const int every_samples =
        control_ ? control_->auto_save_every_samples.load() : config.auto_save_every_samples;
    const int every_minutes =
        control_ ? control_->auto_save_every_minutes.load() : config.auto_save_every_minutes;

    // Check sample count
    if (every_samples > 0 && samples_since_last_save >= every_samples) {
        return true;
    }

    // Check time elapsed
    if (every_minutes > 0) {
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - last_save_time);
        if (elapsed.count() >= every_minutes) {
            return true;
        }
    }

    return false;
}

void IncrementalTrainer::perform_auto_save(int current_epoch, int cumulative_samples_trained) {
    std::string auto_save_path =
        get_session_dir() + "/auto_save_session_" + std::to_string(current_session_id) + ".bin";

    if (save_model(auto_save_path)) {
        // Routed through control_->log() (which itself calls Logger::info) rather
        // than calling Logger::info directly here too — avoids double-logging the
        // same line when running under `serve` (control_ non-null).
        if (control_) {
            control_->log(adai::TrainerLogLevel::Info, "Checkpoint saved: " + auto_save_path);
        } else {
            Logger::info("Auto-saved checkpoint to {}", auto_save_path);
        }
        last_save_time = std::chrono::system_clock::now();
        samples_since_last_save = 0;

        if (control_) {
            control_->checkpoints_written.fetch_add(1);
            control_->last_checkpoint_time_unix.store(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());
            control_->set_last_checkpoint_path(auto_save_path);
        }

        // Persist the in-progress session too, so a crash mid-run doesn't wipe
        // session_history back to empty. final_validation_loss is deliberately
        // left at its 0.0 default here (not known until the epoch finishes),
        // which keeps is_sane_checkpoint_candidate() correctly excluding this
        // still-in-progress entry from "best checkpoint" selection on restart.
        if (!session_history.empty()) {
            auto& session = session_history.back();
            session.samples_trained = cumulative_samples_trained;
            session.epochs_completed = current_epoch;
            session.checkpoint_path = auto_save_path;
            save_session_history();
            Logger::info("Auto-saved session history ({} samples, epoch {})",
                         cumulative_samples_trained, current_epoch);
        }
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
            // Re-queue all previously trained files as pending so the next run
            // picks them all up (equivalent to "mark all untrained").
            DatasetRegistry reg(dataset_config_);
            reg.load_registry();
            auto previously_trained = reg.trained_files();
            std::error_code ec2;
            fs::remove(registry_file, ec2);  // clear trained-status on disk
            if (!previously_trained.empty()) {
                DatasetRegistry fresh(dataset_config_);
                fresh.add_files(previously_trained);
                Logger::info("Data registry preserved; {} entries re-queued as pending",
                             previously_trained.size());
            } else {
                Logger::info("Data registry preserved; no trained entries to re-queue");
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. Reset all in-memory tracking state.
    // ------------------------------------------------------------------
    session_history.clear();
    current_session_id = 0;
    samples_since_last_save = 0;
    best_validation_loss = std::numeric_limits<float>::max();
    best_checkpoint_path.clear();

    // ------------------------------------------------------------------
    // 4. Rebuild model from current config (new architecture).
    // ------------------------------------------------------------------
    reset_model_for_config();

    // ------------------------------------------------------------------
    // 5. Recreate the session directory and persist empty state.
    // ------------------------------------------------------------------
    ensure_directories_exist();
    save_session_history();

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

    // Detect format from first non-empty line
    std::string first_line;
    while (std::getline(file, first_line)) {
        first_line.erase(0, first_line.find_first_not_of(" \t\r\n"));
        if (!first_line.empty())
            break;
    }
    file.seekg(0);

    int pair_count = 0;

    if (!first_line.empty() && first_line.front() == '{') {
        // JSONL training format
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line.front() != '{')
                continue;
            std::string in, resp;
            SampleMeta meta;
            if (parse_jsonl_sample(line, in, resp, meta)) {
                pairs.emplace_back(std::move(in), std::move(resp), std::move(meta));
                ++pair_count;
            }
        }
    } else {
        // Legacy INPUT:/RESPONSE: format
        std::string line, current_input, current_response;
        while (std::getline(file, line)) {
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);

            if (line.empty()) {
                if (!current_input.empty() && !current_response.empty()) {
                    pairs.emplace_back(current_input, current_response);
                    ++pair_count;
                    current_input.clear();
                    current_response.clear();
                }
                continue;
            }

            if (line.substr(0, 6) == "INPUT:") {
                if (!current_input.empty() && !current_response.empty()) {
                    pairs.emplace_back(current_input, current_response);
                    ++pair_count;
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
        if (!current_input.empty() && !current_response.empty()) {
            pairs.emplace_back(current_input, current_response);
            ++pair_count;
        }
    }

    file.close();
    Logger::info("Loaded {} pairs from: {}", pair_count, filepath);
    return pair_count;
}

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

// ============================================================================
// Metrics API Server Management
// ============================================================================
