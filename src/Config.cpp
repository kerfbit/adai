// @adai-status: stable
// @adai-version: 1.30.0
// @adai-reviewed: 2026-11-09

#include "Config.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include "GenerationSettings.hpp"
#include "Logger.hpp"

// Hot-reloading implemented - see DAEMON_IMPLEMENTATION_COMPLETE.md
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Configuration and Service Management #1) - JSON Configuration Format Support
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Configuration and Service Management #2) - Configuration Profiles (dev, staging, prod)

namespace adai {

GPUStrategy gpu_strategy_from_string(const std::string& s) {
    if (s == "full") return GPUStrategy::FULL;
    if (s == "background") return GPUStrategy::BACKGROUND;
    Logger::deferred(Logger::Level::WARN, "Unknown GPU_STRATEGY '" + s + "', using 'background'");
    return GPUStrategy::BACKGROUND;
}

// ============================================================
// Helper Functions
// ============================================================

std::optional<std::string> ConfigLoader::get_env(const std::string& var_name) {
    const char* value = std::getenv(var_name.c_str());
    if (value != nullptr && value[0] != '\0') {
        return std::string(value);
    }
    return std::nullopt;
}

namespace {
// TD-287: numbers must use the whole value. std::stoi/stoull/stof stop at the first bad character
// (PORT=8080x read as 8080) and stoull wraps a negative (D_MODEL=-1 read as 18446744073709551615).
// Each throws std::invalid_argument / std::out_of_range, which the callers turn into a warning
// that keeps the setting's previous value.
int parse_int(const std::string& value) {
    size_t used = 0;
    const int n = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument("trailing characters after the number");
    return n;
}

size_t parse_size(const std::string& value) {
    if (value.find('-') != std::string::npos) throw std::invalid_argument("must not be negative");
    size_t used = 0;
    const unsigned long long n = std::stoull(value, &used);
    if (used != value.size()) throw std::invalid_argument("trailing characters after the number");
    return static_cast<size_t>(n);
}

float parse_float(const std::string& value) {
    size_t used = 0;
    const float f = std::stof(value, &used);
    if (used != value.size()) throw std::invalid_argument("trailing characters after the number");
    if (!std::isfinite(f)) throw std::out_of_range("not a finite number");
    return f;
}
}  // namespace

// Public, non-throwing forms of the parsers above, for command-line flags (TD-240).
std::optional<int> parse_strict_int(const std::string& value) {
    try {
        return parse_int(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<size_t> parse_strict_size(const std::string& value) {
    try {
        return parse_size(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<float> parse_strict_float(const std::string& value) {
    try {
        return parse_float(value);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<int> ConfigLoader::get_env_int(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        try {
            return parse_int(*value);
        } catch (...) {
            Logger::deferred(Logger::Level::WARN, "Invalid integer value for " + var_name + ": " + *value);
        }
    }
    return std::nullopt;
}

std::optional<size_t> ConfigLoader::get_env_size_t(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        try {
            return parse_size(*value);
        } catch (...) {
            Logger::deferred(Logger::Level::WARN, "Invalid size_t value for " + var_name + ": " + *value);
        }
    }
    return std::nullopt;
}

std::optional<float> ConfigLoader::get_env_float(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        try {
            return parse_float(*value);
        } catch (...) {
            Logger::deferred(Logger::Level::WARN, "Invalid float value for " + var_name + ": " + *value);
        }
    }
    return std::nullopt;
}

namespace {
// TD-286: the one boolean syntax for config files and environment variables: true/1/yes/on/enabled
// and false/0/no/off/disabled, any case; anything else is no value (a warning, and the default stays). Seven file keys used to accept only lowercase
// true/1/yes and read every other spelling (True, ON...) as false without a word.
std::optional<bool> parse_bool(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "true" || lower == "1" || lower == "yes" || lower == "on" || lower == "enabled") return true;
    if (lower == "false" || lower == "0" || lower == "no" || lower == "off" || lower == "disabled") return false;
    return std::nullopt;
}

// A boolean config-file value; throws for anything parse_bool() rejects, so load_from_file() warns
// and leaves the setting at its previous value.
bool parse_file_bool(const std::string& key, const std::string& value) {
    if (auto b = parse_bool(value)) return *b;
    throw std::invalid_argument("not a boolean (use true/false, yes/no, on/off, enabled/disabled or 1/0) for " + key);
}
}  // namespace

std::optional<bool> parse_strict_bool(const std::string& value) { return parse_bool(value); }

std::optional<bool> ConfigLoader::get_env_bool(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        if (auto b = parse_bool(*value)) return b;
        Logger::deferred(Logger::Level::WARN, "Invalid boolean value for " + var_name + ": " + *value);
    }
    return std::nullopt;
}

// ============================================================
// Configuration File Parsing
// ============================================================

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

namespace {
// TD-289: a value ends at an unquoted '#' that follows whitespace ("PATH=/models/chat # prod" is
// "/models/chat"; "SECRET=a#b" keeps its '#'). A value in matching quotes is taken as written, and
// only a comment may follow the closing quote.
std::string strip_quotes_and_comment(const std::string& value) {
    if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'')) {
        const size_t close = value.find(value.front(), 1);
        if (close != std::string::npos) {
            const std::string rest = trim(value.substr(close + 1));
            if (rest.empty() || rest.front() == '#') return value.substr(1, close - 1);
        }
    }
    for (size_t i = 1; i < value.size(); ++i) {
        if (value[i] == '#' && (value[i - 1] == ' ' || value[i - 1] == '\t')) return trim(value.substr(0, i));
    }
    return value;
}

// TD-289: the canonical spelling of an enum value, matched case-insensitively, or nullopt.
std::optional<std::string> match_choice(const std::string& value, const std::vector<std::string>& choices) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& choice : choices) {
        std::string c = choice;
        std::transform(c.begin(), c.end(), c.begin(), ::tolower);
        if (c == lower) return choice;
    }
    return std::nullopt;
}

const std::vector<std::string>& kStrategies = adai::generation_strategy_names();  // TD-254: shared
const std::vector<std::string> kLogLevels = {"DEBUG", "INFO", "WARN", "ERROR"};
const std::vector<std::string>& kStorageBackends = metrics_storage_backend_names();  // TD-439: shared
const std::vector<std::string> kDatasetKinds = {"", "encoder", "decoder", "world_model", "chatbot"};
const std::vector<std::string> kTokenizerModes = {"unicode", "ascii"};
const std::vector<std::string> kGpuStrategies = {"background", "full"};
const std::vector<std::string> kLrSchedules = {"linear_warmup", "constant"};  // TD-410

// TD-290: keys the shipped config files carry for features that don't read them yet.
// HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT waits for an auxiliary coverage loss in the training loop.
const std::set<std::string> kReservedKeys = {"HIPPOCAMPAL_COVERAGE_LOSS_WEIGHT"};

// An enum config value in its canonical spelling; LOG_LEVEL also takes WARNING (as Logger does).
// Throws for anything else, so load_from_file() warns and keeps the previous value.
std::string parse_file_choice(const std::string& key, const std::string& value,
                              const std::vector<std::string>& choices) {
    if (key == "LOG_LEVEL" && match_choice(value, {"WARNING"})) return "WARN";
    if (auto c = match_choice(value, choices)) return *c;
    std::string list;
    for (const auto& c : choices) list += (list.empty() ? "" : ", ") + (c.empty() ? std::string("(empty)") : c);
    throw std::invalid_argument("must be one of " + list);
}

// The same for an environment variable: warns and returns nullopt instead of throwing.
std::optional<std::string> env_choice(const std::string& key, const std::optional<std::string>& value,
                                      const std::vector<std::string>& choices) {
    if (!value) return std::nullopt;
    try {
        return parse_file_choice(key, *value, choices);
    } catch (const std::exception& e) {
        Logger::deferred(Logger::Level::WARN, "Invalid value for " + key + ": " + *value + " (" + e.what() + ")");
        return std::nullopt;
    }
}
}  // namespace

void ConfigLoader::load_from_file(ServiceConfig& config, const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        // Config file is optional, so just log a warning
        Logger::deferred(Logger::Level::INFO, "Configuration file not found: " + file_path);
        return;
    }

    Logger::deferred(Logger::Level::INFO, "Loading configuration from: " + file_path);

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        line_number++;
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse key=value
        size_t pos = line.find('=');
        if (pos == std::string::npos) {
            Logger::deferred(Logger::Level::WARN,
                             "Invalid line " + std::to_string(line_number) + " in config file: " + line);
            continue;
        }

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        value = strip_quotes_and_comment(value);  // TD-289

        config.file_values[key] = value;  // TD-285: what reload() diffs

        // Map configuration keys to struct fields
        try {
            if (key == "MODEL_PATH") {
                config.model_path = value;
            } else if (key == "VOCAB_PATH") {
                config.vocab_path = value;
            } else if (key == "DRAFT_MODEL_PATH") {
                config.draft_model_path = value;
            } else if (key == "SPECULATIVE_NUM_CANDIDATES") {
                config.speculative_num_candidates = parse_int(value);
            } else if (key == "PORT") {
                config.port = parse_int(value);
            } else if (key == "SESSION_TIMEOUT") {
                config.session_timeout = parse_int(value);
            } else if (key == "MAX_SESSIONS") {  // TD-232
                config.max_sessions = parse_int(value);
            } else if (key == "MAX_BATCH_SIZE") {  // TD-235
                config.max_batch_size = parse_int(value);
            } else if (key == "LOG_LEVEL") {
                config.log_level = parse_file_choice(key, value, kLogLevels);  // TD-289
            } else if (key == "LOG_FILE_PATH") {
                config.log_file_path = value;
            } else if (key == "SESSION_DIR") {
                config.session_dir = value;
            } else if (key == "LOG_MAX_SIZE_MB") {
                config.log_max_size_mb = parse_size(value);
            } else if (key == "LOG_MAX_FILES") {
                config.log_max_files = parse_size(value);
            } else if (key == "LOG_COMPRESS") {
                config.log_compress = parse_file_bool(key, value);  // TD-286
            } else if (key == "D_MODEL") {
                config.d_model = parse_size(value);
            } else if (key == "NUM_HEADS") {
                config.num_heads = parse_size(value);
            } else if (key == "D_FF") {
                config.d_ff = parse_size(value);
            } else if (key == "NUM_ENCODER_LAYERS") {
                config.num_encoder_layers = parse_size(value);
            } else if (key == "NUM_DECODER_LAYERS") {
                config.num_decoder_layers = parse_size(value);
            } else if (key == "MAX_SEQ_LENGTH") {
                config.max_seq_length = parse_size(value);
                // Training hyperparameters
            } else if (key == "LEARNING_RATE") {
                config.learning_rate = parse_float(value);
            } else if (key == "NUM_EPOCHS") {
                config.num_epochs = parse_int(value);
            } else if (key == "WEIGHT_DECAY") {
                config.weight_decay = parse_float(value);
            } else if (key == "SEED") {
                config.seed = parse_int(value);  // TD-368
            } else if (key == "DROPOUT") {
                config.dropout = parse_float(value);  // TD-344
            } else if (key == "EARLY_STOPPING_ENABLED") {
                config.early_stopping_enabled = parse_file_bool(key, value);  // TD-410
            } else if (key == "EARLY_STOPPING_PATIENCE") {
                config.early_stopping_patience = parse_int(value);
            } else if (key == "RESTORE_BEST_WEIGHTS") {
                config.restore_best_weights = parse_file_bool(key, value);
            } else if (key == "LR_SCHEDULE") {
                config.lr_schedule = parse_file_choice(key, value, kLrSchedules);
            } else if (key == "GRADIENT_CLIP") {
                config.gradient_clip = parse_float(value);
            } else if (key == "GRADIENT_CLIP_ADAPTIVE") {
                config.adaptive_gradient_clip = parse_file_bool(key, value);  // TD-286
            } else if (key == "GRADIENT_CLIP_MIN") {
                config.gradient_clip_min = parse_float(value);
            } else if (key == "GRADIENT_CLIP_MAX") {
                config.gradient_clip_max = parse_float(value);
            } else if (key == "GRADIENT_CLIP_EMA_DECAY") {
                config.gradient_clip_ema_decay = parse_float(value);
            } else if (key == "GRADIENT_CLIP_HEADROOM") {
                config.gradient_clip_headroom = parse_float(value);
            } else if (key == "GRADIENT_CLIP_WARMUP_STEPS") {
                config.gradient_clip_warmup_steps = parse_int(value);
            } else if (key == "GRADIENT_CLIP_SPIKE_K") {
                config.gradient_clip_spike_k = parse_float(value);
            } else if (key == "BATCH_SIZE") {
                config.batch_size = parse_int(value);
            } else if (key == "GRADIENT_ACCUMULATION_STEPS") {
                config.gradient_accumulation_steps = parse_int(value);
            } else if (key == "MAX_LENGTH" || key == "MAX_GEN_LENGTH") {
                config.max_gen_length = parse_size(value);
            } else if (key == "TEMPERATURE") {
                config.temperature = parse_float(value);
            } else if (key == "TOP_P") {
                config.top_p = parse_float(value);
            } else if (key == "TOP_K") {
                config.top_k = parse_int(value);
            } else if (key == "BEAM_WIDTH") {
                config.beam_width = parse_int(value);
            } else if (key == "STRATEGY") {
                config.strategy = parse_file_choice(key, value, kStrategies);  // TD-289
                // Training Metrics Service configuration
            } else if (key == "ENABLE_METRICS_SERVICE") {
                config.enable_metrics_service = parse_file_bool(key, value);  // TD-286
            } else if (key == "METRICS_SERVER_URL") {
                config.metrics_server_url = value;
            } else if (key == "METRICS_PUSH_TIMEOUT_MS") {
                config.metrics_push_timeout_ms = parse_int(value);
            } else if (key == "METRICS_SERVER_HEADERS") {
                config.metrics_server_headers = value;
            } else if (key == "METRICS_SERVER_CA_FILE") {
                config.metrics_server_ca_file = value;
            } else if (key == "METRICS_HEARTBEAT_INTERVAL_MS") {
                config.metrics_heartbeat_interval_ms = parse_int(value);
            } else if (key == "METRICS_ENABLE_PERSISTENCE") {
                config.metrics_enable_persistence = parse_file_bool(key, value);  // TD-286
            } else if (key == "METRICS_FILE") {
                config.metrics_file = value;
            } else if (key == "METRICS_SUMMARY_FILE") {
                config.metrics_summary_file = value;
            } else if (key == "METRICS_PERSIST_EVERY_SAMPLES") {
                config.metrics_persist_every_samples = parse_int(value);
            } else if (key == "METRICS_PERSIST_EVERY_SECONDS") {
                config.metrics_persist_every_seconds = parse_int(value);
            } else if (key == "METRICS_MAX_RECORDS_IN_MEMORY") {
                config.metrics_max_records_in_memory = parse_int(value);
            } else if (key == "METRICS_MAX_RECORDS_ON_DISK") {
                config.metrics_max_records_on_disk = parse_int(value);
            } else if (key == "METRICS_RETENTION_DAYS") {
                config.metrics_retention_days = parse_int(value);
            } else if (key == "METRICS_FILE_MAX_MB") {
                config.metrics_file_max_mb = parse_int(value);
            } else if (key == "METRICS_FILE_BACKUPS") {
                config.metrics_file_backups = parse_int(value);
            } else if (key == "METRICS_ENABLE_PROMETHEUS") {
                config.metrics_enable_prometheus = parse_file_bool(key, value);  // TD-286
            } else if (key == "METRICS_PROMETHEUS_FILE") {
                config.metrics_prometheus_file = value;
            } else if (key == "METRICS_SESSION_KEY") {
                config.metrics_session_key = value;
            } else if (key == "METRICS_SESSION_LABEL") {
                config.metrics_session_label = value;
            } else if (key == "METRICS_MAX_LIVE_SESSIONS") {
                config.metrics_max_live_sessions = parse_size(value);
            } else if (key == "METRICS_COMPLETED_TTL_SECONDS") {
                config.metrics_completed_ttl_seconds = parse_int(value);
            } else if (key == "METRICS_SWEEP_INTERVAL_SECONDS") {
                config.metrics_sweep_interval_seconds = parse_int(value);
            } else if (key == "METRICS_STALENESS_THRESHOLD_SECONDS") {
                config.metrics_staleness_threshold_seconds = parse_int(value);
            } else if (key == "METRICS_API_PORT") {
                config.metrics_api_port = parse_int(value);
            } else if (key == "METRICS_API_ALLOW_CONTROL") {
                config.metrics_api_allow_control = parse_file_bool(key, value);  // TD-286
            } else if (key == "METRICS_API_CORS_ALLOWED_ORIGINS") {
                config.metrics_api_cors_allowed_origins = value;
                // Metrics database persistence (TD-020)
            } else if (key == "METRICS_STORAGE_BACKEND") {
                config.metrics_storage_backend = parse_file_choice(key, value, kStorageBackends);  // TD-289
            } else if (key == "METRICS_DB_PATH") {
                config.metrics_db_path = value;
            } else if (key == "METRICS_DB_URL") {
                config.metrics_db_url = value;
            } else if (key == "METRICS_DB_POOL_SIZE") {
                config.metrics_db_pool_size = parse_int(value);
                // Generation quality metrics configuration
            } else if (key == "ENABLE_GENERATION_QUALITY_METRICS") {
                config.enable_generation_quality_metrics = parse_file_bool(key, value);  // TD-286
            } else if (key == "GENERATION_QUALITY_SAMPLE_SIZE") {
                config.generation_quality_sample_size = parse_int(value);
            } else if (key == "GENERATION_QUALITY_MAX_TOKENS") {
                config.generation_quality_max_tokens = parse_int(value);
            } else if (key == "GENERATION_QUALITY_ASYNC_THRESHOLD") {
                config.generation_quality_async_threshold = parse_int(value);
            } else if (key == "LOSS_QUALITY_BACKFILL") {  // TD-273
                config.loss_quality_backfill = parse_file_bool(key, value);
            } else if (key == "GENERATION_QUALITY_BACKFILL") {  // TD-273
                config.generation_quality_backfill = parse_file_bool(key, value);
                // RAG configuration
            } else if (key == "ALLOW_UNTRAINED_MODEL") {                     // TD-246
                config.allow_untrained_model = parse_file_bool(key, value);  // TD-286
            } else if (key == "RAG_ENABLED") {
                config.rag_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "RAG_DOCS_PATH") {
                config.rag_docs_path = value;
            } else if (key == "RAG_NUM_DOCS") {
                config.rag_num_docs = parse_int(value);
            } else if (key == "RAG_THRESHOLD") {
                config.rag_threshold = parse_float(value);
            } else if (key == "RAG_MAX_CONTEXT_LENGTH") {
                config.rag_max_context_length = parse_int(value);
                // Distributed registry configuration (TD-028 Phase 9)
            } else if (key == "REGISTRY_SERVER_URL") {
                config.registry_server_url = value;
            } else if (key == "RUN_GROUP") {
                config.run_group = value;
            } else if (key == "RUN_ID") {
                config.run_id = value;
            } else if (key == "DATASET_KIND") {
                config.dataset_kind = parse_file_choice(key, value, kDatasetKinds);  // TD-289
            } else if (key == "CACHE_TOKENIZED_DATA") {
                config.cache_tokenized_data = parse_file_bool(key, value);  // TD-286
            } else if (key == "TOKENIZED_CACHE_DIR") {
                config.tokenized_cache_dir = value;
            } else if (key == "REGISTRY_TIMEOUT_MS") {
                config.registry_timeout_ms = parse_int(value);
            } else if (key == "REGISTRY_LISTEN_PORT") {
                config.registry_listen_port = parse_int(value);
            } else if (key == "REGISTRY_DATA_DIR") {
                config.registry_data_dir = value;
            } else if (key == "REGISTRY_ADMIN_ENABLED") {  // TD-489
                config.registry_admin_enabled = parse_file_bool(key, value);
            } else if (key == "REGISTRY_CLAIM_LEASE_MINUTES") {
                config.registry_claim_lease_minutes = parse_int(value);
            } else if (key == "REGISTRY_MAX_UPLOAD_MB") {
                config.registry_max_upload_mb = parse_int(value);
            } else if (key == "REGISTRY_MAX_CONCURRENT_FETCHES") {
                config.registry_max_concurrent_fetches = parse_int(value);
                // FTP Dataset Transport configuration (Phase 10)
            } else if (key == "FTP_SERVER_ENABLED") {                     // TD-284
                config.ftp_server_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "FTP_ADVERTISE_IP") {                       // TD-284
                config.ftp_advertise_ip = value;
            } else if (key == "FTP_SERVER_PORT") {
                config.ftp_server_port = parse_int(value);
            } else if (key == "FTP_PASV_PORT_MIN") {
                config.ftp_pasv_port_min = parse_int(value);
            } else if (key == "FTP_PASV_PORT_MAX") {
                config.ftp_pasv_port_max = parse_int(value);
            } else if (key == "FTP_TOKEN_TTL_MINUTES") {
                config.ftp_token_ttl_minutes = parse_int(value);
            } else if (key == "FTP_DATA_SERVER_SECRET") {
                config.ftp_data_server_secret = value;
            } else if (key == "DOWNLOAD_DIR") {
                config.download_dir = value;
            } else if (key == "MAX_PARALLEL_DOWNLOADS") {
                config.max_parallel_downloads = parse_int(value);
            } else if (key == "LARGE_FILE_WARN_THRESHOLD_MB") {
                config.large_file_warn_threshold_mb = parse_int(value);
            } else if (key == "FTPS_CA_FILE") {
                config.ftps_ca_file = value;
            } else if (key == "FTPS_PINNED_PUBLIC_KEY") {
                config.ftps_pinned_public_key = value;
                // Phase 3: security hardening
            } else if (key == "FTP_MAX_SESSIONS_PER_RUN") {
                config.ftp_max_sessions_per_run = parse_int(value);
            } else if (key == "FTPS_ENABLED") {
                config.ftps_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "FTP_CERT_FILE") {
                config.ftp_cert_file = value;
            } else if (key == "FTP_KEY_FILE") {
                config.ftp_key_file = value;
                // Model Name Service configuration
            } else if (key == "NAME_SERVICE_URL") {
                config.name_service_url = value;
            } else if (key == "NAME_SERVICE_HEADERS") {
                config.name_service_headers = value;
            } else if (key == "NAME_SERVICE_CA_FILE") {
                config.name_service_ca_file = value;
            } else if (key == "NAME_SERVICE_PORT") {
                config.name_service_port = parse_int(value);
            } else if (key == "NAME_SERVICE_DIR") {
                config.name_service_dir = value;
            } else if (key == "NAME_SERVICE_TIMEOUT_MS") {
                config.name_service_timeout_ms = parse_int(value);
            } else if (key == "MNS_ADMIN_ENABLED") {  // TD-456
                config.name_service_admin_enabled = parse_file_bool(key, value);
            } else if (key == "MODEL_NAME") {
                config.model_name = value;
            } else if (key == "MODEL_ROLE") {
                config.model_role = value;
                // Incremental trainer admin API configuration
            } else if (key == "TRAINER_ADMIN_ENABLED") {
                config.trainer_admin_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "TRAINER_ADMIN_PORT") {
                config.trainer_admin_port = parse_int(value);
            } else if (key == "TRAINER_ADMIN_HOST") {
                config.trainer_admin_host = value;
            } else if (key == "TRAINER_ADMIN_DIR") {
                config.trainer_admin_dir = value;
            } else if (key == "TRAINER_CHILD_ADMIN_PORT") {
                config.trainer_child_admin_port = parse_int(value);
                // World-model (LeJEPA) pretraining configuration (TD-183)
            } else if (key == "WORLD_MODEL_ENABLED") {
                config.world_model_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "WORLD_MODEL_D_MODEL") {
                config.world_model_d_model = parse_size(value);
            } else if (key == "WORLD_MODEL_NUM_LAYERS") {
                config.world_model_num_layers = parse_size(value);
            } else if (key == "WORLD_MODEL_NUM_HEADS") {
                config.world_model_num_heads = parse_size(value);
            } else if (key == "WORLD_MODEL_D_FF") {
                config.world_model_d_ff = parse_size(value);
            } else if (key == "WORLD_MODEL_SIGREG_LAMBDA") {
                config.world_model_sigreg_lambda = parse_float(value);
            } else if (key == "WORLD_MODEL_SIGREG_NUM_SKETCHES") {
                config.world_model_sigreg_num_sketches = parse_size(value);
            } else if (key == "WORLD_MODEL_SIGREG_BATCH_SIZE") {
                config.world_model_sigreg_batch_size = parse_size(value);
            } else if (key == "WORLD_MODEL_INJECT_EVERY_N_LAYERS") {
                config.world_model_inject_every_n_layers = parse_size(value);
            } else if (key == "WORLD_MODEL_ARTIFACT_PATH") {
                config.world_model_artifact_path = value;
                // Hippocampal memory configuration (TD-185 filed; TD-186 wires)
            } else if (key == "HIPPOCAMPAL_MEMORY_ENABLED") {
                config.hippocampal_memory_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "HIPPOCAMPAL_MEMORY_CAPACITY") {
                config.hippocampal_memory_capacity = parse_size(value);
            } else if (key == "HIPPOCAMPAL_STATE_PATH") {
                config.hippocampal_state_path = value;
            } else if (key == "HIPPOCAMPAL_SAVE_INTERVAL_SECONDS") {
                config.hippocampal_save_interval_seconds = parse_int(value);
            } else if (key == "HIPPOCAMPAL_REPETITION_ALPHA") {
                config.hippocampal_repetition_alpha = parse_float(value);
            } else if (key == "HIPPOCAMPAL_REPETITION_DECAY") {
                config.hippocampal_repetition_decay = parse_float(value);
            } else if (key == "HIPPOCAMPAL_CROSS_REFERENCE_ALPHA") {
                config.hippocampal_cross_reference_alpha = parse_float(value);
            } else if (key == "HIPPOCAMPAL_ASSOCIATION_DECAY") {
                config.hippocampal_association_decay = parse_float(value);
                // Auto-save / checkpoint retention configuration
            } else if (key == "AUTO_SAVE_ENABLED") {
                config.auto_save_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "RESUME_FROM_BEST_CHECKPOINT") {
                config.resume_from_best_checkpoint = parse_file_bool(key, value);  // TD-286
            } else if (key == "AUTO_SAVE_EVERY_SAMPLES") {
                config.auto_save_every_samples = parse_int(value);
            } else if (key == "AUTO_SAVE_EVERY_MINUTES") {
                config.auto_save_every_minutes = parse_int(value);
            } else if (key == "MAX_SESSIONS_TO_KEEP") {
                config.max_sessions_to_keep = parse_int(value);
                // GPU configuration
            } else if (key == "GPU_ENABLED") {
                config.gpu_enabled = parse_file_bool(key, value);  // TD-286
            } else if (key == "GPU_DEVICE_ID") {
                config.gpu_device_id = parse_int(value);
            } else if (key == "GPU_MEMORY_FRACTION") {
                config.gpu_memory_fraction = parse_float(value);
            } else if (key == "GPU_STRATEGY") {
                config.gpu_strategy = gpu_strategy_from_string(parse_file_choice(key, value, kGpuStrategies));
            } else if (key == "TOKENIZER_MODE") {
                config.unicode_tokenizer = parse_file_choice(key, value, kTokenizerModes) == "unicode";  // TD-289
            } else if (key == "VOCAB_BUILD_SIZE") {
                config.vocab_build_size = parse_int(value);
            } else if (kReservedKeys.count(key) != 0) {
                // TD-290: shipped in the config files for a feature that doesn't read it yet; no warning.
            } else {
                Logger::deferred(Logger::Level::WARN, "Unknown configuration key: " + key);
            }
        } catch (const std::exception& e) {
            Logger::deferred(Logger::Level::WARN, "Invalid value for " + key + ": " + value + " (" + e.what() + ")");
        }
    }
}

// ============================================================
// Environment Variable Loading
// ============================================================

void ConfigLoader::load_from_env(ServiceConfig& config) {
    // Server configuration
    if (auto val = get_env("MODEL_PATH")) {
        config.model_path = *val;
    }
    if (auto val = get_env("VOCAB_PATH")) {
        config.vocab_path = *val;
    }
    if (auto val = get_env("DRAFT_MODEL_PATH")) {
        config.draft_model_path = *val;
    }
    if (auto val = get_env_int("SPECULATIVE_NUM_CANDIDATES")) {
        config.speculative_num_candidates = *val;
    }
    if (auto val = get_env_int("PORT")) {
        config.port = *val;
    }
    if (auto val = get_env_int("SESSION_TIMEOUT")) {
        config.session_timeout = *val;
    }
    if (auto val = get_env_int("MAX_SESSIONS")) {  // TD-232
        config.max_sessions = *val;
    }
    if (auto val = get_env_int("MAX_BATCH_SIZE")) {  // TD-235
        config.max_batch_size = *val;
    }
    if (auto val = env_choice("LOG_LEVEL", get_env("LOG_LEVEL"), kLogLevels)) {
        config.log_level = *val;
    }
    if (auto val = get_env("LOG_FILE_PATH")) {
        config.log_file_path = *val;
    }
    if (auto val = get_env("SESSION_DIR")) {
        config.session_dir = *val;
    }
    if (auto val = get_env_size_t("LOG_MAX_SIZE_MB")) {
        config.log_max_size_mb = *val;
    }
    if (auto val = get_env_size_t("LOG_MAX_FILES")) {
        config.log_max_files = *val;
    }
    if (auto val = get_env_bool("LOG_COMPRESS")) {
        config.log_compress = *val;
    }

    // Model architecture
    if (auto val = get_env_size_t("D_MODEL")) {
        config.d_model = *val;
    }
    if (auto val = get_env_size_t("NUM_HEADS")) {
        config.num_heads = *val;
    }
    if (auto val = get_env_size_t("D_FF")) {
        config.d_ff = *val;
    }
    if (auto val = get_env_size_t("NUM_ENCODER_LAYERS")) {
        config.num_encoder_layers = *val;
    }
    if (auto val = get_env_size_t("NUM_DECODER_LAYERS")) {
        config.num_decoder_layers = *val;
    }
    if (auto val = get_env_size_t("MAX_SEQ_LENGTH")) {
        config.max_seq_length = *val;
    }

    // Training hyperparameters
    if (auto val = get_env_float("LEARNING_RATE")) {
        config.learning_rate = *val;
    }
    if (auto val = get_env_int("NUM_EPOCHS")) {
        config.num_epochs = *val;
    }
    if (auto val = get_env_float("WEIGHT_DECAY")) {
        config.weight_decay = *val;
    }
    if (auto val = get_env_bool("EARLY_STOPPING_ENABLED")) {  // TD-410
        config.early_stopping_enabled = *val;
    }
    if (auto val = get_env_int("EARLY_STOPPING_PATIENCE")) {
        config.early_stopping_patience = *val;
    }
    if (auto val = get_env_bool("RESTORE_BEST_WEIGHTS")) {
        config.restore_best_weights = *val;
    }
    if (auto val = env_choice("LR_SCHEDULE", get_env("LR_SCHEDULE"), kLrSchedules)) {
        config.lr_schedule = *val;
    }
    if (auto val = get_env_int("SEED")) {
        config.seed = *val;  // TD-368
    }
    if (auto val = get_env_float("DROPOUT")) {
        config.dropout = *val;
    }
    if (auto val = get_env_float("GRADIENT_CLIP")) {
        config.gradient_clip = *val;
    }
    if (auto val = get_env_bool("GRADIENT_CLIP_ADAPTIVE")) {
        config.adaptive_gradient_clip = *val;
    }
    if (auto val = get_env_float("GRADIENT_CLIP_MIN")) {
        config.gradient_clip_min = *val;
    }
    if (auto val = get_env_float("GRADIENT_CLIP_MAX")) {
        config.gradient_clip_max = *val;
    }
    if (auto val = get_env_float("GRADIENT_CLIP_EMA_DECAY")) {
        config.gradient_clip_ema_decay = *val;
    }
    if (auto val = get_env_float("GRADIENT_CLIP_HEADROOM")) {
        config.gradient_clip_headroom = *val;
    }
    if (auto val = get_env_int("GRADIENT_CLIP_WARMUP_STEPS")) {
        config.gradient_clip_warmup_steps = *val;
    }
    if (auto val = get_env_float("GRADIENT_CLIP_SPIKE_K")) {
        config.gradient_clip_spike_k = *val;
    }
    if (auto val = get_env_int("BATCH_SIZE")) {
        config.batch_size = *val;
    }
    if (auto val = get_env_int("GRADIENT_ACCUMULATION_STEPS")) {
        config.gradient_accumulation_steps = *val;
    }

    // Generation parameters
    // Support both MAX_LENGTH and MAX_GEN_LENGTH for compatibility
    if (auto val = get_env_size_t("MAX_GEN_LENGTH")) {
        config.max_gen_length = *val;
    }
    if (auto val = get_env_size_t("MAX_LENGTH")) {
        config.max_gen_length = *val;
    }

    if (auto val = get_env_float("TEMPERATURE")) {
        config.temperature = *val;
    }
    if (auto val = get_env_float("TOP_P")) {
        config.top_p = *val;
    }
    if (auto val = get_env_int("TOP_K")) {
        config.top_k = *val;
    }
    if (auto val = get_env_int("BEAM_WIDTH")) {
        config.beam_width = *val;
    }
    if (auto val = env_choice("STRATEGY", get_env("STRATEGY"), kStrategies)) {
        config.strategy = *val;
    }

    // Training Metrics Service configuration
    // TD-091 (fixed): these 19 keys (this block + the Generation quality
    // metrics block below) were settable via the config file but never had a
    // load_from_env() counterpart, silently breaking the "env vars override
    // config file" precedence CLAUDE.md documents as a uniform rule for every
    // key — an env var set for e.g. METRICS_SERVER_URL or
    // METRICS_HEARTBEAT_INTERVAL_MS had no effect if a config file also set a
    // value, since load_from_env() never checked for it at all.
    if (auto val = get_env_bool("ENABLE_METRICS_SERVICE")) {
        config.enable_metrics_service = *val;
    }
    if (auto val = get_env("METRICS_SERVER_URL")) {
        config.metrics_server_url = *val;
    }
    if (auto val = get_env_int("METRICS_PUSH_TIMEOUT_MS")) {
        config.metrics_push_timeout_ms = *val;
    }
    if (auto val = get_env("METRICS_SERVER_HEADERS")) {
        config.metrics_server_headers = *val;
    }
    if (auto val = get_env("METRICS_SERVER_CA_FILE")) {
        config.metrics_server_ca_file = *val;
    }
    if (auto val = get_env_int("METRICS_HEARTBEAT_INTERVAL_MS")) {
        config.metrics_heartbeat_interval_ms = *val;
    }
    if (auto val = get_env_bool("METRICS_ENABLE_PERSISTENCE")) {
        config.metrics_enable_persistence = *val;
    }
    if (auto val = get_env("METRICS_FILE")) {
        config.metrics_file = *val;
    }
    if (auto val = get_env("METRICS_SUMMARY_FILE")) {
        config.metrics_summary_file = *val;
    }
    if (auto val = get_env_int("METRICS_PERSIST_EVERY_SAMPLES")) {
        config.metrics_persist_every_samples = *val;
    }
    if (auto val = get_env_int("METRICS_PERSIST_EVERY_SECONDS")) {
        config.metrics_persist_every_seconds = *val;
    }
    if (auto val = get_env_int("METRICS_MAX_RECORDS_IN_MEMORY")) {
        config.metrics_max_records_in_memory = *val;
    }
    if (auto val = get_env_int("METRICS_MAX_RECORDS_ON_DISK")) {
        config.metrics_max_records_on_disk = *val;
    }
    if (auto val = get_env_int("METRICS_RETENTION_DAYS")) {
        config.metrics_retention_days = *val;
    }
    if (auto val = get_env_int("METRICS_FILE_MAX_MB")) {
        config.metrics_file_max_mb = *val;
    }
    if (auto val = get_env_int("METRICS_FILE_BACKUPS")) {
        config.metrics_file_backups = *val;
    }
    if (auto val = get_env_bool("METRICS_ENABLE_PROMETHEUS")) {
        config.metrics_enable_prometheus = *val;
    }
    if (auto val = get_env("METRICS_PROMETHEUS_FILE")) {
        config.metrics_prometheus_file = *val;
    }
    if (auto val = get_env_int("METRICS_API_PORT")) {
        config.metrics_api_port = *val;
    }
    if (auto val = get_env_bool("METRICS_API_ALLOW_CONTROL")) {
        config.metrics_api_allow_control = *val;
    }
    if (auto val = get_env("METRICS_API_CORS_ALLOWED_ORIGINS")) {
        config.metrics_api_cors_allowed_origins = *val;
    }

    // Generation quality metrics configuration
    if (auto val = get_env_bool("ENABLE_GENERATION_QUALITY_METRICS")) {
        config.enable_generation_quality_metrics = *val;
    }
    if (auto val = get_env_int("GENERATION_QUALITY_SAMPLE_SIZE")) {
        config.generation_quality_sample_size = *val;
    }
    if (auto val = get_env_int("GENERATION_QUALITY_MAX_TOKENS")) {
        config.generation_quality_max_tokens = *val;
    }
    if (auto val = get_env_int("GENERATION_QUALITY_ASYNC_THRESHOLD")) {
        config.generation_quality_async_threshold = *val;
    }
    if (auto val = get_env_bool("LOSS_QUALITY_BACKFILL")) {  // TD-273
        config.loss_quality_backfill = *val;
    }
    if (auto val = get_env_bool("GENERATION_QUALITY_BACKFILL")) {  // TD-273
        config.generation_quality_backfill = *val;
    }

    // Multi-instance metrics configuration
    if (auto val = get_env("METRICS_SESSION_KEY")) {
        config.metrics_session_key = *val;
    }
    if (auto val = get_env("METRICS_SESSION_LABEL")) {
        config.metrics_session_label = *val;
    }
    if (auto val = get_env_size_t("METRICS_MAX_LIVE_SESSIONS")) {
        config.metrics_max_live_sessions = *val;
    }
    if (auto val = get_env_int("METRICS_COMPLETED_TTL_SECONDS")) {
        config.metrics_completed_ttl_seconds = *val;
    }
    if (auto val = get_env_int("METRICS_SWEEP_INTERVAL_SECONDS")) {
        config.metrics_sweep_interval_seconds = *val;
    }
    if (auto val = get_env_int("METRICS_STALENESS_THRESHOLD_SECONDS")) {
        config.metrics_staleness_threshold_seconds = *val;
    }

    // Metrics database persistence (TD-020)
    if (auto val = env_choice("METRICS_STORAGE_BACKEND", get_env("METRICS_STORAGE_BACKEND"), kStorageBackends)) {
        config.metrics_storage_backend = *val;
    }
    if (auto val = get_env("METRICS_DB_PATH")) {
        config.metrics_db_path = *val;
    }
    if (auto val = get_env("METRICS_DB_URL")) {
        config.metrics_db_url = *val;
    }
    if (auto val = get_env_int("METRICS_DB_POOL_SIZE")) {
        config.metrics_db_pool_size = *val;
    }

    if (auto val = get_env_bool("ALLOW_UNTRAINED_MODEL")) {  // TD-246
        config.allow_untrained_model = *val;
    }

    // RAG configuration
    if (auto val = get_env_bool("RAG_ENABLED")) {
        config.rag_enabled = *val;
    }
    if (auto val = get_env("RAG_DOCS_PATH")) {
        config.rag_docs_path = *val;
    }
    if (auto val = get_env_int("RAG_NUM_DOCS")) {
        config.rag_num_docs = *val;
    }
    if (auto val = get_env_float("RAG_THRESHOLD")) {
        config.rag_threshold = *val;
    }
    if (auto val = get_env_int("RAG_MAX_CONTEXT_LENGTH")) {
        config.rag_max_context_length = *val;
    }

    // GPU configuration
    if (auto val = get_env_bool("GPU_ENABLED")) {
        config.gpu_enabled = *val;
    }
    if (auto val = get_env_int("GPU_DEVICE_ID")) {
        config.gpu_device_id = *val;
    }
    if (auto val = get_env_float("GPU_MEMORY_FRACTION")) {
        config.gpu_memory_fraction = *val;
    }
    if (auto val = env_choice("GPU_STRATEGY", get_env("GPU_STRATEGY"), kGpuStrategies)) {
        config.gpu_strategy = gpu_strategy_from_string(*val);
    }
    if (auto val = env_choice("TOKENIZER_MODE", get_env("TOKENIZER_MODE"), kTokenizerModes)) {
        config.unicode_tokenizer = (*val == "unicode");
    }
    if (auto val = get_env_int("VOCAB_BUILD_SIZE")) {
        config.vocab_build_size = *val;
    }

    // Distributed registry (TD-028 Phase 9)
    if (auto val = get_env("REGISTRY_SERVER_URL")) {
        config.registry_server_url = *val;
    }
    if (auto val = get_env("RUN_GROUP")) {
        config.run_group = *val;
    }
    if (auto val = get_env("RUN_ID")) {
        config.run_id = *val;
    }
    if (auto val = env_choice("DATASET_KIND", get_env("DATASET_KIND"), kDatasetKinds)) {
        config.dataset_kind = *val;
    }
    if (auto val = get_env_bool("CACHE_TOKENIZED_DATA")) {
        config.cache_tokenized_data = *val;
    }
    if (auto val = get_env("TOKENIZED_CACHE_DIR")) {
        config.tokenized_cache_dir = *val;
    }
    if (auto val = get_env_int("REGISTRY_TIMEOUT_MS")) {
        config.registry_timeout_ms = *val;
    }
    if (auto val = get_env_int("REGISTRY_LISTEN_PORT")) {
        config.registry_listen_port = *val;
    }
    if (auto val = get_env("REGISTRY_DATA_DIR")) {
        config.registry_data_dir = *val;
    }
    if (auto val = get_env_bool("REGISTRY_ADMIN_ENABLED")) {  // TD-489
        config.registry_admin_enabled = *val;
    }
    if (auto val = get_env_int("REGISTRY_CLAIM_LEASE_MINUTES")) {
        config.registry_claim_lease_minutes = *val;
    }
    if (auto val = get_env_int("REGISTRY_MAX_UPLOAD_MB")) {
        config.registry_max_upload_mb = *val;
    }
    if (auto val = get_env_int("REGISTRY_MAX_CONCURRENT_FETCHES")) {
        config.registry_max_concurrent_fetches = *val;
    }

    // FTP Dataset Transport (Phase 10)
    if (auto val = get_env_bool("FTP_SERVER_ENABLED")) {  // TD-284
        config.ftp_server_enabled = *val;
    }
    if (auto val = get_env("FTP_ADVERTISE_IP")) {  // TD-284
        config.ftp_advertise_ip = *val;
    }
    if (auto val = get_env_int("FTP_SERVER_PORT")) {
        config.ftp_server_port = *val;
    }
    if (auto val = get_env_int("FTP_PASV_PORT_MIN")) {
        config.ftp_pasv_port_min = *val;
    }
    if (auto val = get_env_int("FTP_PASV_PORT_MAX")) {
        config.ftp_pasv_port_max = *val;
    }
    if (auto val = get_env_int("FTP_TOKEN_TTL_MINUTES")) {
        config.ftp_token_ttl_minutes = *val;
    }
    if (auto val = get_env("FTP_DATA_SERVER_SECRET")) {
        config.ftp_data_server_secret = *val;
    }
    if (auto val = get_env("DOWNLOAD_DIR")) {
        config.download_dir = *val;
    }
    if (auto val = get_env_int("MAX_PARALLEL_DOWNLOADS")) {
        config.max_parallel_downloads = *val;
    }
    if (auto val = get_env_int("LARGE_FILE_WARN_THRESHOLD_MB")) {
        config.large_file_warn_threshold_mb = *val;
    }
    if (auto val = get_env("FTPS_CA_FILE")) {
        config.ftps_ca_file = *val;
    }
    if (auto val = get_env("FTPS_PINNED_PUBLIC_KEY")) {
        config.ftps_pinned_public_key = *val;
    }
    // Phase 3: security hardening
    if (auto val = get_env_int("FTP_MAX_SESSIONS_PER_RUN")) {
        config.ftp_max_sessions_per_run = *val;
    }
    if (auto val = get_env_bool("FTPS_ENABLED")) {  // TD-286: used to bypass get_env_bool()
        config.ftps_enabled = *val;
    }
    if (auto val = get_env("FTP_CERT_FILE")) {
        config.ftp_cert_file = *val;
    }
    if (auto val = get_env("FTP_KEY_FILE")) {
        config.ftp_key_file = *val;
    }

    // Model Name Service
    if (auto val = get_env("NAME_SERVICE_URL")) {
        config.name_service_url = *val;
    }
    if (auto val = get_env("NAME_SERVICE_HEADERS")) {
        config.name_service_headers = *val;
    }
    if (auto val = get_env("NAME_SERVICE_CA_FILE")) {
        config.name_service_ca_file = *val;
    }
    if (auto val = get_env_int("NAME_SERVICE_PORT")) {
        config.name_service_port = *val;
    }
    if (auto val = get_env("NAME_SERVICE_DIR")) {
        config.name_service_dir = *val;
    }
    if (auto val = get_env_int("NAME_SERVICE_TIMEOUT_MS")) {
        config.name_service_timeout_ms = *val;
    }
    if (auto val = get_env_bool("MNS_ADMIN_ENABLED")) {  // TD-456
        config.name_service_admin_enabled = *val;
    }
    if (auto val = get_env("MODEL_NAME")) {
        config.model_name = *val;
    }
    if (auto val = get_env("MODEL_ROLE")) {
        config.model_role = *val;
    }

    // Incremental trainer admin API
    if (auto val = get_env_bool("TRAINER_ADMIN_ENABLED")) {
        config.trainer_admin_enabled = *val;
    }
    if (auto val = get_env_int("TRAINER_ADMIN_PORT")) {
        config.trainer_admin_port = *val;
    }
    if (auto val = get_env("TRAINER_ADMIN_HOST")) {
        config.trainer_admin_host = *val;
    }
    if (auto val = get_env("TRAINER_ADMIN_DIR")) {
        config.trainer_admin_dir = *val;
    }
    if (auto val = get_env_int("TRAINER_CHILD_ADMIN_PORT")) {
        config.trainer_child_admin_port = *val;
    }

    // World-model (LeJEPA) pretraining configuration (TD-183)
    if (auto val = get_env_bool("WORLD_MODEL_ENABLED")) {
        config.world_model_enabled = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_D_MODEL")) {
        config.world_model_d_model = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_NUM_LAYERS")) {
        config.world_model_num_layers = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_NUM_HEADS")) {
        config.world_model_num_heads = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_D_FF")) {
        config.world_model_d_ff = *val;
    }
    if (auto val = get_env_float("WORLD_MODEL_SIGREG_LAMBDA")) {
        config.world_model_sigreg_lambda = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_SIGREG_NUM_SKETCHES")) {
        config.world_model_sigreg_num_sketches = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_SIGREG_BATCH_SIZE")) {
        config.world_model_sigreg_batch_size = *val;
    }
    if (auto val = get_env_size_t("WORLD_MODEL_INJECT_EVERY_N_LAYERS")) {
        config.world_model_inject_every_n_layers = *val;
    }
    if (auto val = get_env("WORLD_MODEL_ARTIFACT_PATH")) {
        config.world_model_artifact_path = *val;
    }

    // Hippocampal memory configuration (TD-185 filed; TD-186 wires)
    if (auto val = get_env_bool("HIPPOCAMPAL_MEMORY_ENABLED")) {
        config.hippocampal_memory_enabled = *val;
    }
    if (auto val = get_env_size_t("HIPPOCAMPAL_MEMORY_CAPACITY")) {
        config.hippocampal_memory_capacity = *val;
    }
    if (auto val = get_env("HIPPOCAMPAL_STATE_PATH")) {
        config.hippocampal_state_path = *val;
    }
    if (auto val = get_env_int("HIPPOCAMPAL_SAVE_INTERVAL_SECONDS")) {
        config.hippocampal_save_interval_seconds = *val;
    }
    if (auto val = get_env_float("HIPPOCAMPAL_REPETITION_ALPHA")) {
        config.hippocampal_repetition_alpha = *val;
    }
    if (auto val = get_env_float("HIPPOCAMPAL_REPETITION_DECAY")) {
        config.hippocampal_repetition_decay = *val;
    }
    if (auto val = get_env_float("HIPPOCAMPAL_CROSS_REFERENCE_ALPHA")) {
        config.hippocampal_cross_reference_alpha = *val;
    }
    if (auto val = get_env_float("HIPPOCAMPAL_ASSOCIATION_DECAY")) {
        config.hippocampal_association_decay = *val;
    }

    // Auto-save / checkpoint retention
    if (auto val = get_env_bool("AUTO_SAVE_ENABLED")) {
        config.auto_save_enabled = *val;
    }
    if (auto val = get_env_bool("RESUME_FROM_BEST_CHECKPOINT")) {
        config.resume_from_best_checkpoint = *val;
    }
    if (auto val = get_env_int("AUTO_SAVE_EVERY_SAMPLES")) {
        config.auto_save_every_samples = *val;
    }
    if (auto val = get_env_int("AUTO_SAVE_EVERY_MINUTES")) {
        config.auto_save_every_minutes = *val;
    }
    if (auto val = get_env_int("MAX_SESSIONS_TO_KEEP")) {
        config.max_sessions_to_keep = *val;
    }
}

// ============================================================
// Public API
// ============================================================

std::string ConfigLoader::discover_config_path(const std::string& explicit_path,
                                                const std::string& service_filename) {
    if (!explicit_path.empty()) {
        return explicit_path;
    }

    const auto exists = [](const std::string& path) {
        std::ifstream f(path);
        return f.good();
    };

    if (exists(service_filename)) {
        return service_filename;
    }
    const std::string etc_service_path = "/etc/adai/" + service_filename;
    if (exists(etc_service_path)) {
        return etc_service_path;
    }
    // Legacy fallback: pre-split monolithic config.conf.
    if (exists("config.conf")) {
        return "config.conf";
    }
    return "/etc/adai/config.conf";
}

ServiceConfig ConfigLoader::load() {
    ServiceConfig config;

    // Try to load from default config file location
    const char* default_config = "/etc/adai/config.conf";
    load_from_file(config, default_config);

    // Environment variables override file configuration
    load_from_env(config);

    return config;
}

ServiceConfig ConfigLoader::load(const std::string& config_file_path) {
    ServiceConfig config;

    // First load from specified config file
    load_from_file(config, config_file_path);

    // Then override with environment variables
    load_from_env(config);

    return config;
}

// TD-290: through the logger (it used to be std::cout), and every section of the config.
void ConfigLoader::print(const ServiceConfig& config) {
    std::istringstream lines(to_string(config));
    for (std::string line; std::getline(lines, line);) Logger::info("{}", line);
}

// ============================================================
// Configuration Hot-Reloading
// ============================================================

bool ConfigLoader::reload(ServiceConfig& config, const std::string& config_file_path,
                          std::mutex& mutex, const std::set<std::string>* live_keys,
                          const std::function<void(ServiceConfig&)>& cli_overrides,
                          const std::set<std::string>* cli_keys) {
    // Get timestamp for logging
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

    Logger::info("==================================================");
    Logger::info("Configuration Reload Triggered at {}", timestamp.str());
    Logger::info("==================================================");

    // Create a temporary config to load new values
    ServiceConfig new_config;

    // Load from file (don't apply yet)
    Logger::info("Loading configuration from: {}", config_file_path);
    load_from_file(new_config, config_file_path);

    // Override with environment variables (maintain priority)
    load_from_env(new_config);

    // TD-242: and with the command-line flags, which a reload used to drop.
    if (cli_overrides) cli_overrides(new_config);

    // Validate the new configuration
    std::vector<std::string> errors;
    if (!validate(new_config, errors)) {
        Logger::error("Configuration validation failed:");
        for (const auto& error : errors) {
            Logger::error("  - {}", error);
        }
        Logger::error("Configuration reload aborted - keeping current configuration");
        Logger::info("==================================================");
        return false;
    }

    Logger::info("Configuration validation passed");

    // TD-285: the valid new config is always applied. It used to be applied only when one of the
    // ~25 fields detect_changes() compared had changed, so a reload that changed only other keys
    // (METRICS_SERVER_URL, AUTO_SAVE_*, RAG, GPU, MNS...) reported success and kept the old values.
    std::vector<std::string> changes;
    std::vector<std::string> restart_keys;
    std::vector<std::string> held_keys;  // TD-242
    {
        std::lock_guard<std::mutex> lock(mutex);
        changes = detect_changes(config, new_config);
        for (const auto& key : changed_file_keys(config, new_config)) {
            if (cli_keys && cli_keys->count(key) != 0) {
                held_keys.push_back(key);
            } else if (live_keys && live_keys->count(key) == 0) {
                restart_keys.push_back(key);
            }
        }
        config = new_config;
    }

    if (changes.empty()) {
        Logger::info("No configuration changes detected");
    } else {
        Logger::info("Configuration updated; changes:");
        for (const auto& change : changes) {
            Logger::info("  {}", change);
        }
    }
    // TD-285/TD-251: changes this binary doesn't apply until it restarts.
    for (const auto& key : restart_keys) {
        Logger::warn("  {} changed but takes effect only after a restart", key);
    }
    // TD-242: file changes that a command-line flag overrides, so the flag's value stays.
    for (const auto& key : held_keys) {
        Logger::warn("  {} changed in the file but is held by a command-line flag", key);
    }

    Logger::info("==================================================");
    return true;
}

// Run by reload() and, since TD-288, at startup by every daemon and incremental_trainer
// (validate_or_exit()); chatbot_api_server's startup call is TD-241.
bool ConfigLoader::validate(const ServiceConfig& config, std::vector<std::string>& errors) {
    bool valid = true;

    // Validate port range
    if (config.port < 1 || config.port > 65535) {
        errors.push_back("Invalid port: " + std::to_string(config.port) + " (must be 1-65535)");
        valid = false;
    }

    // Validate session timeout
    if (config.session_timeout < 1) {
        errors.push_back("Invalid session_timeout: " + std::to_string(config.session_timeout) +
                         " (must be >= 1)");
        valid = false;
    }

    if (config.rag_enabled && config.rag_num_docs < 1) {  // TD-487
        errors.push_back("Invalid rag_num_docs: " + std::to_string(config.rag_num_docs) + " (must be >= 1)");
        valid = false;
    }
    if (config.rag_enabled && config.rag_max_context_length < 1) {  // TD-487
        errors.push_back("Invalid rag_max_context_length: " + std::to_string(config.rag_max_context_length) +
                         " (must be >= 1)");
        valid = false;
    }
    if (config.max_batch_size < 1) {  // TD-235
        errors.push_back("Invalid max_batch_size: " + std::to_string(config.max_batch_size) +
                         " (must be >= 1)");
        valid = false;
    }
    if (config.max_sessions < 1) {  // TD-232
        errors.push_back("Invalid max_sessions: " + std::to_string(config.max_sessions) +
                         " (must be >= 1)");
        valid = false;
    }

    // Validate log level
    std::vector<std::string> valid_levels = {"DEBUG", "INFO", "WARN", "ERROR"};
    std::string upper_log_level = config.log_level;
    std::transform(upper_log_level.begin(), upper_log_level.end(), upper_log_level.begin(),
                   ::toupper);
    if (std::find(valid_levels.begin(), valid_levels.end(), upper_log_level) ==
        valid_levels.end()) {
        errors.push_back("Invalid log_level: " + config.log_level +
                         " (must be DEBUG, INFO, WARN, or ERROR)");
        valid = false;
    }

    // Validate log file configuration
    if (config.log_max_size_mb < 1 || config.log_max_size_mb > 1024) {
        errors.push_back("Invalid log_max_size_mb: " + std::to_string(config.log_max_size_mb) +
                         " (must be 1-1024)");
        valid = false;
    }

    if (config.log_max_files < 1 || config.log_max_files > 100) {
        errors.push_back("Invalid log_max_files: " + std::to_string(config.log_max_files) +
                         " (must be 1-100)");
        valid = false;
    }

    // Validate model architecture parameters
    if (config.d_model < 64 || config.d_model > 8192) {
        errors.push_back("Invalid d_model: " + std::to_string(config.d_model) +
                         " (must be 64-8192)");
        valid = false;
    }

    if (config.num_heads < 1 || config.num_heads > 64) {
        errors.push_back("Invalid num_heads: " + std::to_string(config.num_heads) +
                         " (must be 1-64)");
        valid = false;
    }

    // d_model must be divisible by num_heads (only check if num_heads is valid to avoid division by
    // zero)
    if (config.num_heads > 0 && config.d_model % config.num_heads != 0) {
        errors.push_back("d_model (" + std::to_string(config.d_model) +
                         ") must be divisible by num_heads (" + std::to_string(config.num_heads) +
                         ")");
        valid = false;
    }

    if (config.d_ff < 64 || config.d_ff > 32768) {
        errors.push_back("Invalid d_ff: " + std::to_string(config.d_ff) + " (must be 64-32768)");
        valid = false;
    }

    if (config.num_encoder_layers < 1 || config.num_encoder_layers > 48) {
        errors.push_back("Invalid num_encoder_layers: " +
                         std::to_string(config.num_encoder_layers) + " (must be 1-48)");
        valid = false;
    }

    if (config.num_decoder_layers < 1 || config.num_decoder_layers > 48) {
        errors.push_back("Invalid num_decoder_layers: " +
                         std::to_string(config.num_decoder_layers) + " (must be 1-48)");
        valid = false;
    }

    if (config.max_seq_length < 16 || config.max_seq_length > 32768) {
        errors.push_back("Invalid max_seq_length: " + std::to_string(config.max_seq_length) +
                         " (must be 16-32768)");
        valid = false;
    }

    // Validate world-model (LeJEPA) architecture parameters (TD-183) — same bounds as the
    // chatbot architecture block above, since these size an equivalent transformer-encoder
    // stack. Checked unconditionally (not just when world_model_enabled) so a config that
    // enables the feature later, without a fresh edit to these fields, is already valid.
    if (config.world_model_d_model < 64 || config.world_model_d_model > 8192) {
        errors.push_back("Invalid world_model_d_model: " +
                         std::to_string(config.world_model_d_model) + " (must be 64-8192)");
        valid = false;
    }
    if (config.world_model_num_heads < 1 || config.world_model_num_heads > 64) {
        errors.push_back("Invalid world_model_num_heads: " +
                         std::to_string(config.world_model_num_heads) + " (must be 1-64)");
        valid = false;
    }
    if (config.world_model_num_heads > 0 &&
        config.world_model_d_model % config.world_model_num_heads != 0) {
        errors.push_back("world_model_d_model (" + std::to_string(config.world_model_d_model) +
                         ") must be divisible by world_model_num_heads (" +
                         std::to_string(config.world_model_num_heads) + ")");
        valid = false;
    }
    if (config.world_model_d_ff < 64 || config.world_model_d_ff > 32768) {
        errors.push_back("Invalid world_model_d_ff: " + std::to_string(config.world_model_d_ff) +
                         " (must be 64-32768)");
        valid = false;
    }
    if (config.world_model_num_layers < 1 || config.world_model_num_layers > 48) {
        errors.push_back("Invalid world_model_num_layers: " +
                         std::to_string(config.world_model_num_layers) + " (must be 1-48)");
        valid = false;
    }
    if (config.world_model_sigreg_lambda < 0.0f) {
        errors.push_back("Invalid world_model_sigreg_lambda: " +
                         std::to_string(config.world_model_sigreg_lambda) + " (must be >= 0.0)");
        valid = false;
    }
    if (config.world_model_sigreg_num_sketches < 1 ||
        config.world_model_sigreg_num_sketches > 4096) {
        errors.push_back("Invalid world_model_sigreg_num_sketches: " +
                         std::to_string(config.world_model_sigreg_num_sketches) +
                         " (must be 1-4096)");
        valid = false;
    }
    if (config.world_model_sigreg_batch_size < 2 || config.world_model_sigreg_batch_size > 4096) {
        errors.push_back("Invalid world_model_sigreg_batch_size: " +
                         std::to_string(config.world_model_sigreg_batch_size) +
                         " (must be 2-4096; SIGReg needs at least 2 pooled embeddings)");
        valid = false;
    }
    // TD-186: DecoderBlock's own gated world-model cross-attention is constructed with the
    // chatbot's own d_model on both the query and kv-projection sides (CrossAttention has no
    // separate kv-dimension parameter) — a mismatch here would crash deep inside a matrix
    // multiply the first time a gated layer actually runs, rather than at config-load time with
    // a clear message. Only checked once injection is actually requested; a world model
    // pretrained at a different size with injection left at 0 (the default) is unaffected.
    if (config.world_model_inject_every_n_layers > 0 &&
        config.world_model_d_model != config.d_model) {
        errors.push_back(
            "world_model_d_model (" + std::to_string(config.world_model_d_model) +
            ") must equal d_model (" + std::to_string(config.d_model) +
            ") whenever world_model_inject_every_n_layers > 0 (currently " +
            std::to_string(config.world_model_inject_every_n_layers) +
            ") — DecoderBlock's gated world-model cross-attention has no separate kv-dimension");
        valid = false;
    }

    // Validate hippocampal memory parameters (TD-185 filed; TD-186 wires)
    if (config.hippocampal_memory_capacity < 1 || config.hippocampal_memory_capacity > 65536) {
        errors.push_back("Invalid hippocampal_memory_capacity: " +
                         std::to_string(config.hippocampal_memory_capacity) +
                         " (must be 1-65536)");
        valid = false;
    }
    if (config.hippocampal_save_interval_seconds < 0) {  // TD-250
        errors.push_back("Invalid hippocampal_save_interval_seconds: " +
                         std::to_string(config.hippocampal_save_interval_seconds) + " (must be >= 0; 0 = only at shutdown)");
        valid = false;
    }
    if (config.hippocampal_repetition_alpha < 0.0f) {
        errors.push_back("Invalid hippocampal_repetition_alpha: " +
                         std::to_string(config.hippocampal_repetition_alpha) +
                         " (must be >= 0.0)");
        valid = false;
    }
    if (config.hippocampal_repetition_decay <= 0.0f || config.hippocampal_repetition_decay > 1.0f) {
        errors.push_back("Invalid hippocampal_repetition_decay: " +
                         std::to_string(config.hippocampal_repetition_decay) +
                         " (must be in (0.0, 1.0])");
        valid = false;
    }

    // Validate generation parameters
    // TD-254: the ranges are shared with ChatbotAPI's per-request settings and the chatbot CLI.
    using Limits = adai::GenerationLimits;
    if (config.max_gen_length < Limits::kMinMaxLength || config.max_gen_length > Limits::kMaxMaxLength) {
        errors.push_back("Invalid max_gen_length: " + std::to_string(config.max_gen_length) +
                         " (must be 1-4096)");
        valid = false;
    }

    if (config.temperature < Limits::kMinTemperature || config.temperature > Limits::kMaxTemperature) {
        errors.push_back("Invalid temperature: " + std::to_string(config.temperature) +
                         " (must be 0.0-2.0)");
        valid = false;
    }

    if (config.top_p < Limits::kMinTopP || config.top_p > Limits::kMaxTopP) {
        errors.push_back("Invalid top_p: " + std::to_string(config.top_p) + " (must be 0.0-1.0)");
        valid = false;
    }

    if (config.top_k < Limits::kMinTopK || config.top_k > Limits::kMaxTopK) {
        errors.push_back("Invalid top_k: " + std::to_string(config.top_k) + " (must be 1-1000)");
        valid = false;
    }

    if (config.beam_width < Limits::kMinBeamWidth || config.beam_width > Limits::kMaxBeamWidth) {
        errors.push_back("Invalid beam_width: " + std::to_string(config.beam_width) +
                         " (must be 1-16)");
        valid = false;
    }

    // Validate strategy
    const auto& valid_strategies = adai::generation_strategy_names();
    if (std::find(valid_strategies.begin(), valid_strategies.end(), config.strategy) ==
        valid_strategies.end()) {
        errors.push_back("Invalid strategy: " + config.strategy +
                         " (must be greedy, beam, temperature, top_k, or nucleus)");
        valid = false;
    }

    // TD-288: every other numeric field that divides, allocates, binds a port or is a probability.
    auto require = [&](bool ok, const std::string& name, const std::string& value, const char* rule) {
        if (ok) return;
        errors.push_back("Invalid " + name + ": " + value + " (" + rule + ")");
        valid = false;
    };
    auto at_least = [&](long long v, long long min, const std::string& name) {
        require(v >= min, name, std::to_string(v), min == 0 ? "must be >= 0" : "must be >= 1");
    };
    auto is_port = [&](int v, const std::string& name) {
        require(v >= 1 && v <= 65535, name, std::to_string(v), "must be 1-65535");
    };

    // Training. gradient_accumulation_steps is a divisor and a modulus in ChatbotTrainer.
    require(config.learning_rate > 0.0f, "learning_rate", std::to_string(config.learning_rate), "must be > 0");
    at_least(config.num_epochs, 1, "num_epochs");
    at_least(config.batch_size, 1, "batch_size");
    at_least(config.gradient_accumulation_steps, 1, "gradient_accumulation_steps");
    require(config.weight_decay >= 0.0f, "weight_decay", std::to_string(config.weight_decay), "must be >= 0");
    require(config.dropout >= 0.0f && config.dropout < 1.0f, "dropout",
            std::to_string(config.dropout),
            "must be in [0, 1)");  // TD-344
    at_least(config.early_stopping_patience, 1, "early_stopping_patience");  // TD-410
    require(config.gradient_clip >= 0.0f, "gradient_clip", std::to_string(config.gradient_clip), "must be >= 0");
    // Adaptive clipping, checked even while it's off (like the world-model block above).
    require(config.gradient_clip_min > 0.0f, "gradient_clip_min", std::to_string(config.gradient_clip_min),
            "must be > 0");
    require(config.gradient_clip_max >= config.gradient_clip_min, "gradient_clip_max",
            std::to_string(config.gradient_clip_max), "must be >= gradient_clip_min");
    require(config.gradient_clip_ema_decay > 0.0f && config.gradient_clip_ema_decay <= 1.0f,
            "gradient_clip_ema_decay", std::to_string(config.gradient_clip_ema_decay), "must be in (0.0, 1.0]");
    require(config.gradient_clip_headroom > 0.0f, "gradient_clip_headroom",
            std::to_string(config.gradient_clip_headroom), "must be > 0");
    at_least(config.gradient_clip_warmup_steps, 0, "gradient_clip_warmup_steps");
    require(config.gradient_clip_spike_k > 0.0f, "gradient_clip_spike_k", std::to_string(config.gradient_clip_spike_k),
            "must be > 0");
    at_least(config.vocab_build_size, 0, "vocab_build_size");
    at_least(config.speculative_num_candidates, 1, "speculative_num_candidates");

    require(config.hippocampal_association_decay > 0.0f && config.hippocampal_association_decay <= 1.0f,
            "hippocampal_association_decay", std::to_string(config.hippocampal_association_decay),
            "must be in (0.0, 1.0]");
    require(config.hippocampal_cross_reference_alpha >= 0.0f, "hippocampal_cross_reference_alpha",
            std::to_string(config.hippocampal_cross_reference_alpha), "must be >= 0");

    require(config.gpu_memory_fraction > 0.0f && config.gpu_memory_fraction <= 1.0f, "gpu_memory_fraction",
            std::to_string(config.gpu_memory_fraction), "must be in (0.0, 1.0]");
    at_least(config.gpu_device_id, 0, "gpu_device_id");

    // Ports.
    is_port(config.metrics_api_port, "metrics_api_port");
    is_port(config.registry_listen_port, "registry_listen_port");
    is_port(config.ftp_server_port, "ftp_server_port");
    is_port(config.ftp_pasv_port_min, "ftp_pasv_port_min");
    is_port(config.ftp_pasv_port_max, "ftp_pasv_port_max");
    require(config.ftp_pasv_port_min <= config.ftp_pasv_port_max, "ftp_pasv_port_max",
            std::to_string(config.ftp_pasv_port_max), "must be >= ftp_pasv_port_min");
    is_port(config.name_service_port, "name_service_port");
    is_port(config.trainer_admin_port, "trainer_admin_port");
    is_port(config.trainer_child_admin_port, "trainer_child_admin_port");
    require(config.trainer_child_admin_port != config.trainer_admin_port, "trainer_child_admin_port",
            std::to_string(config.trainer_child_admin_port), "must differ from trainer_admin_port");

    // Metrics client and daemon. 0 means "off" or "no limit" where a field allows it.
    at_least(config.metrics_push_timeout_ms, 1, "metrics_push_timeout_ms");
    at_least(config.metrics_heartbeat_interval_ms, 1, "metrics_heartbeat_interval_ms");
    at_least(config.metrics_persist_every_samples, 0, "metrics_persist_every_samples");
    at_least(config.metrics_persist_every_seconds, 0, "metrics_persist_every_seconds");
    at_least(config.metrics_max_records_in_memory, 1, "metrics_max_records_in_memory");
    at_least(config.metrics_max_records_on_disk, 0, "metrics_max_records_on_disk");
    at_least(config.metrics_retention_days, 0, "metrics_retention_days");
    at_least(config.metrics_file_max_mb, 0, "metrics_file_max_mb");
    at_least(config.metrics_file_backups, 0, "metrics_file_backups");
    at_least(static_cast<long long>(config.metrics_max_live_sessions), 1, "metrics_max_live_sessions");
    at_least(config.metrics_completed_ttl_seconds, 0, "metrics_completed_ttl_seconds");
    at_least(config.metrics_staleness_threshold_seconds, 1, "metrics_staleness_threshold_seconds");
    at_least(config.metrics_db_pool_size, 1, "metrics_db_pool_size");
    at_least(config.generation_quality_sample_size, 0, "generation_quality_sample_size");
    at_least(config.generation_quality_max_tokens, 1, "generation_quality_max_tokens");
    at_least(config.generation_quality_async_threshold, 0, "generation_quality_async_threshold");

    // Registry, FTP and MNS.
    at_least(config.registry_timeout_ms, 1, "registry_timeout_ms");
    at_least(config.registry_claim_lease_minutes, 0, "registry_claim_lease_minutes");
    at_least(config.registry_max_upload_mb, 1, "registry_max_upload_mb");
    at_least(config.registry_max_concurrent_fetches, 1, "registry_max_concurrent_fetches");
    at_least(config.ftp_token_ttl_minutes, 1, "ftp_token_ttl_minutes");
    at_least(config.max_parallel_downloads, 1, "max_parallel_downloads");
    at_least(config.large_file_warn_threshold_mb, 0, "large_file_warn_threshold_mb");
    at_least(config.ftp_max_sessions_per_run, 0, "ftp_max_sessions_per_run");
    at_least(config.name_service_timeout_ms, 1, "name_service_timeout_ms");

    // Checkpointing.
    at_least(config.auto_save_every_samples, 0, "auto_save_every_samples");
    at_least(config.auto_save_every_minutes, 0, "auto_save_every_minutes");
    at_least(config.max_sessions_to_keep, 0, "max_sessions_to_keep");
    at_least(config.seed, 0, "seed");

    return valid;
}

void ConfigLoader::validate_or_exit(const ServiceConfig& config, const std::string& binary_name) {
    std::vector<std::string> errors;
    if (validate(config, errors)) return;
    Logger::error("{}: the configuration is invalid:", binary_name);
    for (const auto& error : errors) {
        Logger::error("  {}", error);
    }
    std::exit(1);
}

namespace {
// TD-285: keys whose values shouldn't appear in a log.
bool is_secret_key(const std::string& key) {
    for (const char* marker : {"SECRET", "PASSWORD", "TOKEN", "HEADERS", "CREDENTIAL"}) {
        if (key.find(marker) != std::string::npos) return true;
    }
    return false;
}
}  // namespace

std::string ConfigLoader::to_string(const ServiceConfig& c) {
    std::ostringstream out;
    auto section = [&](const char* name) { out << name << ":\n"; };
    auto item = [&](const std::string& key, const std::string& value) {
        std::string shown = value.empty() ? "<not set>" : value;
        if (!value.empty() && (is_secret_key(key) || key == "METRICS_DB_URL")) shown = "<hidden>";
        out << "  " << std::left << std::setw(36) << key << shown << '\n';
    };
    auto num = [](auto v) {
        std::ostringstream s;
        s << v;
        return s.str();
    };
    auto flag = [](bool b) { return std::string(b ? "true" : "false"); };

    out << "==================================================\n";
    out << "               ADAI Configuration\n";
    out << "==================================================\n";
    section("Server");
    item("MODEL_PATH", c.model_path);
    item("VOCAB_PATH", c.vocab_path);
    item("DRAFT_MODEL_PATH", c.draft_model_path);
    item("SPECULATIVE_NUM_CANDIDATES", num(c.speculative_num_candidates));
    item("PORT", num(c.port));
    item("SESSION_TIMEOUT", num(c.session_timeout) + " minutes");
    item("MAX_SESSIONS", num(c.max_sessions));
    item("MAX_BATCH_SIZE", num(c.max_batch_size));
    item("ALLOW_UNTRAINED_MODEL", flag(c.allow_untrained_model));
    item("SESSION_DIR", c.session_dir);
    section("Logging");
    item("LOG_LEVEL", c.log_level);
    item("LOG_FILE_PATH", c.log_file_path.empty() ? "<console only>" : c.log_file_path);
    item("LOG_MAX_SIZE_MB", num(c.log_max_size_mb));
    item("LOG_MAX_FILES", num(c.log_max_files));
    item("LOG_COMPRESS", flag(c.log_compress));
    section("Model architecture");
    item("D_MODEL", num(c.d_model));
    item("NUM_HEADS", num(c.num_heads));
    item("D_FF", num(c.d_ff));
    item("NUM_ENCODER_LAYERS", num(c.num_encoder_layers));
    item("NUM_DECODER_LAYERS", num(c.num_decoder_layers));
    item("MAX_SEQ_LENGTH", num(c.max_seq_length));
    item("TOKENIZER_MODE", c.unicode_tokenizer ? "unicode" : "ascii");
    item("VOCAB_BUILD_SIZE", num(c.vocab_build_size));
    section("Training");
    item("LEARNING_RATE", num(c.learning_rate));
    item("NUM_EPOCHS", num(c.num_epochs));
    item("BATCH_SIZE", num(c.batch_size));
    item("GRADIENT_ACCUMULATION_STEPS", num(c.gradient_accumulation_steps));
    item("WEIGHT_DECAY", num(c.weight_decay));
    item("DROPOUT", num(c.dropout));
    item("SEED", num(c.seed));
    item("EARLY_STOPPING_ENABLED", flag(c.early_stopping_enabled));
    item("EARLY_STOPPING_PATIENCE", num(c.early_stopping_patience));
    item("RESTORE_BEST_WEIGHTS", flag(c.restore_best_weights));
    item("LR_SCHEDULE", c.lr_schedule);
    item("GRADIENT_CLIP", num(c.gradient_clip));
    item("GRADIENT_CLIP_ADAPTIVE", flag(c.adaptive_gradient_clip));
    item("GRADIENT_CLIP_MIN", num(c.gradient_clip_min));
    item("GRADIENT_CLIP_MAX", num(c.gradient_clip_max));
    item("GRADIENT_CLIP_EMA_DECAY", num(c.gradient_clip_ema_decay));
    item("GRADIENT_CLIP_HEADROOM", num(c.gradient_clip_headroom));
    item("GRADIENT_CLIP_WARMUP_STEPS", num(c.gradient_clip_warmup_steps));
    item("GRADIENT_CLIP_SPIKE_K", num(c.gradient_clip_spike_k));
    item("CACHE_TOKENIZED_DATA", flag(c.cache_tokenized_data));
    item("TOKENIZED_CACHE_DIR", c.tokenized_cache_dir);
    item("ENABLE_GENERATION_QUALITY_METRICS", flag(c.enable_generation_quality_metrics));
    item("GENERATION_QUALITY_SAMPLE_SIZE", num(c.generation_quality_sample_size));
    item("GENERATION_QUALITY_MAX_TOKENS", num(c.generation_quality_max_tokens));
    item("GENERATION_QUALITY_ASYNC_THRESHOLD", num(c.generation_quality_async_threshold));
    item("LOSS_QUALITY_BACKFILL", flag(c.loss_quality_backfill));
    item("GENERATION_QUALITY_BACKFILL", flag(c.generation_quality_backfill));
    section("Checkpointing");
    item("AUTO_SAVE_ENABLED", flag(c.auto_save_enabled));
    item("AUTO_SAVE_EVERY_SAMPLES", num(c.auto_save_every_samples));
    item("AUTO_SAVE_EVERY_MINUTES", num(c.auto_save_every_minutes));
    item("MAX_SESSIONS_TO_KEEP", num(c.max_sessions_to_keep));
    item("RESUME_FROM_BEST_CHECKPOINT", flag(c.resume_from_best_checkpoint));
    section("Generation");
    item("MAX_GEN_LENGTH", num(c.max_gen_length));
    item("TEMPERATURE", num(c.temperature));
    item("TOP_P", num(c.top_p));
    item("TOP_K", num(c.top_k));
    item("BEAM_WIDTH", num(c.beam_width));
    item("STRATEGY", c.strategy);
    section("RAG");
    item("RAG_ENABLED", flag(c.rag_enabled));
    item("RAG_DOCS_PATH", c.rag_docs_path);
    item("RAG_NUM_DOCS", num(c.rag_num_docs));
    item("RAG_THRESHOLD", num(c.rag_threshold));
    item("RAG_MAX_CONTEXT_LENGTH", num(c.rag_max_context_length));
    section("GPU");
    item("GPU_ENABLED", flag(c.gpu_enabled));
    item("GPU_DEVICE_ID", num(c.gpu_device_id));
    item("GPU_MEMORY_FRACTION", num(c.gpu_memory_fraction));
    item("GPU_STRATEGY", c.gpu_strategy == GPUStrategy::FULL ? "full" : "background");
    section("Metrics client");
    item("ENABLE_METRICS_SERVICE", flag(c.enable_metrics_service));
    item("METRICS_SERVER_URL", c.metrics_server_url);
    item("METRICS_SERVER_HEADERS", c.metrics_server_headers);
    item("METRICS_SERVER_CA_FILE", c.metrics_server_ca_file);
    item("METRICS_PUSH_TIMEOUT_MS", num(c.metrics_push_timeout_ms));
    item("METRICS_HEARTBEAT_INTERVAL_MS", num(c.metrics_heartbeat_interval_ms));
    item("METRICS_SESSION_KEY", c.metrics_session_key);
    item("METRICS_SESSION_LABEL", c.metrics_session_label);
    section("Metrics daemon");
    item("METRICS_API_PORT", num(c.metrics_api_port));
    item("METRICS_API_ALLOW_CONTROL", flag(c.metrics_api_allow_control));
    item("METRICS_API_CORS_ALLOWED_ORIGINS", c.metrics_api_cors_allowed_origins);
    item("METRICS_STORAGE_BACKEND", c.metrics_storage_backend);
    item("METRICS_DB_PATH", c.metrics_db_path);
    item("METRICS_DB_URL", c.metrics_db_url);
    item("METRICS_DB_POOL_SIZE", num(c.metrics_db_pool_size));
    item("METRICS_ENABLE_PERSISTENCE", flag(c.metrics_enable_persistence));
    item("METRICS_FILE", c.metrics_file);
    item("METRICS_SUMMARY_FILE", c.metrics_summary_file);
    item("METRICS_PERSIST_EVERY_SAMPLES", num(c.metrics_persist_every_samples));
    item("METRICS_PERSIST_EVERY_SECONDS", num(c.metrics_persist_every_seconds));
    item("METRICS_MAX_RECORDS_IN_MEMORY", num(c.metrics_max_records_in_memory));
    item("METRICS_MAX_RECORDS_ON_DISK", num(c.metrics_max_records_on_disk));
    item("METRICS_RETENTION_DAYS", num(c.metrics_retention_days));
    item("METRICS_FILE_MAX_MB", num(c.metrics_file_max_mb));
    item("METRICS_FILE_BACKUPS", num(c.metrics_file_backups));
    item("METRICS_ENABLE_PROMETHEUS", flag(c.metrics_enable_prometheus));
    item("METRICS_PROMETHEUS_FILE", c.metrics_prometheus_file);
    item("METRICS_MAX_LIVE_SESSIONS", num(c.metrics_max_live_sessions));
    item("METRICS_COMPLETED_TTL_SECONDS", num(c.metrics_completed_ttl_seconds));
    item("METRICS_SWEEP_INTERVAL_SECONDS", num(c.metrics_sweep_interval_seconds));
    item("METRICS_STALENESS_THRESHOLD_SECONDS", num(c.metrics_staleness_threshold_seconds));
    section("Dataset registry");
    item("REGISTRY_SERVER_URL", c.registry_server_url);
    item("RUN_GROUP", c.run_group);
    item("RUN_ID", c.run_id);
    item("DATASET_KIND", c.dataset_kind);
    item("REGISTRY_TIMEOUT_MS", num(c.registry_timeout_ms));
    item("REGISTRY_LISTEN_PORT", num(c.registry_listen_port));
    item("REGISTRY_DATA_DIR", c.registry_data_dir);
    item("REGISTRY_ADMIN_ENABLED", flag(c.registry_admin_enabled));
    item("REGISTRY_CLAIM_LEASE_MINUTES", num(c.registry_claim_lease_minutes));
    item("REGISTRY_MAX_UPLOAD_MB", num(c.registry_max_upload_mb));
    item("REGISTRY_MAX_CONCURRENT_FETCHES", num(c.registry_max_concurrent_fetches));
    item("DOWNLOAD_DIR", c.download_dir);
    item("MAX_PARALLEL_DOWNLOADS", num(c.max_parallel_downloads));
    item("LARGE_FILE_WARN_THRESHOLD_MB", num(c.large_file_warn_threshold_mb));
    section("FTP");
    item("FTP_SERVER_ENABLED", flag(c.ftp_server_enabled));
    item("FTP_ADVERTISE_IP", c.ftp_advertise_ip);
    item("FTP_SERVER_PORT", num(c.ftp_server_port));
    item("FTP_PASV_PORT_MIN", num(c.ftp_pasv_port_min));
    item("FTP_PASV_PORT_MAX", num(c.ftp_pasv_port_max));
    item("FTP_TOKEN_TTL_MINUTES", num(c.ftp_token_ttl_minutes));
    item("FTP_MAX_SESSIONS_PER_RUN", num(c.ftp_max_sessions_per_run));
    item("FTP_DATA_SERVER_SECRET", c.ftp_data_server_secret);
    item("FTPS_ENABLED", flag(c.ftps_enabled));
    item("FTP_CERT_FILE", c.ftp_cert_file);
    item("FTP_KEY_FILE", c.ftp_key_file);
    item("FTPS_CA_FILE", c.ftps_ca_file);
    item("FTPS_PINNED_PUBLIC_KEY", c.ftps_pinned_public_key);
    section("Model Name Service");
    item("NAME_SERVICE_URL", c.name_service_url);
    item("NAME_SERVICE_HEADERS", c.name_service_headers);
    item("NAME_SERVICE_CA_FILE", c.name_service_ca_file);
    item("NAME_SERVICE_PORT", num(c.name_service_port));
    item("NAME_SERVICE_DIR", c.name_service_dir);
    item("NAME_SERVICE_TIMEOUT_MS", num(c.name_service_timeout_ms));
    item("MNS_ADMIN_ENABLED", flag(c.name_service_admin_enabled));
    item("MODEL_NAME", c.model_name);
    item("MODEL_ROLE", c.model_role);
    section("Trainer admin API");
    item("TRAINER_ADMIN_ENABLED", flag(c.trainer_admin_enabled));
    item("TRAINER_ADMIN_HOST", c.trainer_admin_host);
    item("TRAINER_ADMIN_PORT", num(c.trainer_admin_port));
    item("TRAINER_ADMIN_DIR", c.trainer_admin_dir);
    item("TRAINER_CHILD_ADMIN_PORT", num(c.trainer_child_admin_port));
    section("World model");
    item("WORLD_MODEL_ENABLED", flag(c.world_model_enabled));
    item("WORLD_MODEL_ARTIFACT_PATH", c.world_model_artifact_path);
    item("WORLD_MODEL_D_MODEL", num(c.world_model_d_model));
    item("WORLD_MODEL_NUM_LAYERS", num(c.world_model_num_layers));
    item("WORLD_MODEL_NUM_HEADS", num(c.world_model_num_heads));
    item("WORLD_MODEL_D_FF", num(c.world_model_d_ff));
    item("WORLD_MODEL_SIGREG_LAMBDA", num(c.world_model_sigreg_lambda));
    item("WORLD_MODEL_SIGREG_NUM_SKETCHES", num(c.world_model_sigreg_num_sketches));
    item("WORLD_MODEL_SIGREG_BATCH_SIZE", num(c.world_model_sigreg_batch_size));
    item("WORLD_MODEL_INJECT_EVERY_N_LAYERS", num(c.world_model_inject_every_n_layers));
    section("Hippocampal memory");
    item("HIPPOCAMPAL_MEMORY_ENABLED", flag(c.hippocampal_memory_enabled));
    item("HIPPOCAMPAL_MEMORY_CAPACITY", num(c.hippocampal_memory_capacity));
    item("HIPPOCAMPAL_STATE_PATH", c.hippocampal_state_path);
    item("HIPPOCAMPAL_SAVE_INTERVAL_SECONDS", num(c.hippocampal_save_interval_seconds));
    item("HIPPOCAMPAL_REPETITION_ALPHA", num(c.hippocampal_repetition_alpha));
    item("HIPPOCAMPAL_REPETITION_DECAY", num(c.hippocampal_repetition_decay));
    item("HIPPOCAMPAL_CROSS_REFERENCE_ALPHA", num(c.hippocampal_cross_reference_alpha));
    item("HIPPOCAMPAL_ASSOCIATION_DECAY", num(c.hippocampal_association_decay));
    out << "==================================================\n";
    return out.str();
}

std::vector<std::string> ConfigLoader::changed_file_keys(const ServiceConfig& old_config,
                                                         const ServiceConfig& new_config) {
    std::vector<std::string> keys;
    for (const auto& [key, value] : new_config.file_values) {
        auto it = old_config.file_values.find(key);
        if (it == old_config.file_values.end() || it->second != value) keys.push_back(key);
    }
    for (const auto& [key, value] : old_config.file_values) {
        if (new_config.file_values.count(key) == 0) keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::vector<std::string> ConfigLoader::detect_changes(const ServiceConfig& old_config,
                                                      const ServiceConfig& new_config) {
    std::vector<std::string> changes;

    // TD-285: configs read from files are compared key by key, so every change is reported.
    if (!old_config.file_values.empty() || !new_config.file_values.empty()) {
        for (const auto& key : changed_file_keys(old_config, new_config)) {
            auto before = old_config.file_values.find(key);
            auto after = new_config.file_values.find(key);
            const bool secret = is_secret_key(key);
            auto show = [secret](const std::string& v) { return secret ? std::string("<hidden>") : "'" + v + "'"; };
            if (before == old_config.file_values.end()) {
                changes.push_back(key + ": added (" + show(after->second) + ")");
            } else if (after == new_config.file_values.end()) {
                changes.push_back(key + ": removed (was " + show(before->second) + ")");
            } else {
                changes.push_back(key + ": " + show(before->second) + " -> " + show(after->second));
            }
        }
        return changes;
    }

    // Configs not loaded from a file: compare the fields below.

    // Server configuration changes
    if (old_config.model_path != new_config.model_path) {
        changes.push_back("model_path: '" + old_config.model_path + "' -> '" +
                          new_config.model_path + "'");
    }
    if (old_config.vocab_path != new_config.vocab_path) {
        changes.push_back("vocab_path: '" + old_config.vocab_path + "' -> '" +
                          new_config.vocab_path + "'");
    }
    if (old_config.draft_model_path != new_config.draft_model_path) {
        changes.push_back("draft_model_path: '" + old_config.draft_model_path + "' -> '" +
                          new_config.draft_model_path + "'");
    }
    if (old_config.port != new_config.port) {
        changes.push_back("port: " + std::to_string(old_config.port) + " -> " +
                          std::to_string(new_config.port));
    }
    if (old_config.session_timeout != new_config.session_timeout) {
        changes.push_back("session_timeout: " + std::to_string(old_config.session_timeout) +
                          " -> " + std::to_string(new_config.session_timeout));
    }
    if (old_config.log_level != new_config.log_level) {
        changes.push_back("log_level: " + old_config.log_level + " -> " + new_config.log_level);
    }
    if (old_config.log_file_path != new_config.log_file_path) {
        changes.push_back("log_file_path: '" + old_config.log_file_path + "' -> '" +
                          new_config.log_file_path + "'");
    }
    if (old_config.session_dir != new_config.session_dir) {
        changes.push_back("session_dir: '" + old_config.session_dir + "' -> '" +
                          new_config.session_dir + "'");
    }
    if (old_config.log_max_size_mb != new_config.log_max_size_mb) {
        changes.push_back("log_max_size_mb: " + std::to_string(old_config.log_max_size_mb) +
                          " -> " + std::to_string(new_config.log_max_size_mb));
    }
    if (old_config.log_max_files != new_config.log_max_files) {
        changes.push_back("log_max_files: " + std::to_string(old_config.log_max_files) + " -> " +
                          std::to_string(new_config.log_max_files));
    }
    if (old_config.log_compress != new_config.log_compress) {
        changes.push_back(
            "log_compress: " + std::string(old_config.log_compress ? "true" : "false") + " -> " +
            std::string(new_config.log_compress ? "true" : "false"));
    }

    // Model architecture changes
    if (old_config.d_model != new_config.d_model) {
        changes.push_back("d_model: " + std::to_string(old_config.d_model) + " -> " +
                          std::to_string(new_config.d_model));
    }
    if (old_config.num_heads != new_config.num_heads) {
        changes.push_back("num_heads: " + std::to_string(old_config.num_heads) + " -> " +
                          std::to_string(new_config.num_heads));
    }
    if (old_config.d_ff != new_config.d_ff) {
        changes.push_back("d_ff: " + std::to_string(old_config.d_ff) + " -> " +
                          std::to_string(new_config.d_ff));
    }
    if (old_config.num_encoder_layers != new_config.num_encoder_layers) {
        changes.push_back("num_encoder_layers: " + std::to_string(old_config.num_encoder_layers) +
                          " -> " + std::to_string(new_config.num_encoder_layers));
    }
    if (old_config.num_decoder_layers != new_config.num_decoder_layers) {
        changes.push_back("num_decoder_layers: " + std::to_string(old_config.num_decoder_layers) +
                          " -> " + std::to_string(new_config.num_decoder_layers));
    }
    if (old_config.max_seq_length != new_config.max_seq_length) {
        changes.push_back("max_seq_length: " + std::to_string(old_config.max_seq_length) + " -> " +
                          std::to_string(new_config.max_seq_length));
    }

    // Generation parameters changes
    if (old_config.max_gen_length != new_config.max_gen_length) {
        changes.push_back("max_gen_length: " + std::to_string(old_config.max_gen_length) + " -> " +
                          std::to_string(new_config.max_gen_length));
    }
    if (old_config.temperature != new_config.temperature) {
        changes.push_back("temperature: " + std::to_string(old_config.temperature) + " -> " +
                          std::to_string(new_config.temperature));
    }
    if (old_config.top_p != new_config.top_p) {
        changes.push_back("top_p: " + std::to_string(old_config.top_p) + " -> " +
                          std::to_string(new_config.top_p));
    }
    if (old_config.top_k != new_config.top_k) {
        changes.push_back("top_k: " + std::to_string(old_config.top_k) + " -> " +
                          std::to_string(new_config.top_k));
    }
    if (old_config.beam_width != new_config.beam_width) {
        changes.push_back("beam_width: " + std::to_string(old_config.beam_width) + " -> " +
                          std::to_string(new_config.beam_width));
    }
    if (old_config.strategy != new_config.strategy) {
        changes.push_back("strategy: " + old_config.strategy + " -> " + new_config.strategy);
    }

    return changes;
}

}  // namespace adai
