#include "Config.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "Logger.hpp"

// Hot-reloading implemented - see DAEMON_IMPLEMENTATION_COMPLETE.md
// TODO: See TECHNICAL_DEBT.md Future Enhancement #4 - JSON Configuration Format Support
// TODO: See TECHNICAL_DEBT.md Future Enhancement #5 - Configuration Profiles (dev, staging, prod)

namespace adai {

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

std::optional<int> ConfigLoader::get_env_int(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        try {
            return std::stoi(*value);
        } catch (...) {
            std::cerr << "Warning: Invalid integer value for " << var_name << ": " << *value
                      << '\n';
        }
    }
    return std::nullopt;
}

std::optional<size_t> ConfigLoader::get_env_size_t(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        try {
            return static_cast<size_t>(std::stoull(*value));
        } catch (...) {
            std::cerr << "Warning: Invalid size_t value for " << var_name << ": " << *value << '\n';
        }
    }
    return std::nullopt;
}

std::optional<float> ConfigLoader::get_env_float(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        try {
            return std::stof(*value);
        } catch (...) {
            std::cerr << "Warning: Invalid float value for " << var_name << ": " << *value << '\n';
        }
    }
    return std::nullopt;
}

std::optional<bool> ConfigLoader::get_env_bool(const std::string& var_name) {
    auto value = get_env(var_name);
    if (value) {
        std::string lower = *value;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "true" || lower == "1" || lower == "yes" || lower == "on") {
            return true;
        }
        if (lower == "false" || lower == "0" || lower == "no" || lower == "off") {
            return false;
        }
        std::cerr << "Warning: Invalid boolean value for " << var_name << ": " << *value << '\n';
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

void ConfigLoader::load_from_file(ServiceConfig& config, const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        // Config file is optional, so just log a warning
        std::cerr << "Note: Configuration file not found: " << file_path << '\n';
        return;
    }

    std::cout << "Loading configuration from: " << file_path << '\n';

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
            std::cerr << "Warning: Invalid line " << line_number << " in config file: " << line
                      << '\n';
            continue;
        }

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        // Remove quotes from value if present
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }

        // Map configuration keys to struct fields
        try {
            if (key == "MODEL_PATH") {
                config.model_path = value;
            } else if (key == "VOCAB_PATH") {
                config.vocab_path = value;
            } else if (key == "PORT") {
                config.port = std::stoi(value);
            } else if (key == "SESSION_TIMEOUT") {
                config.session_timeout = std::stoi(value);
            } else if (key == "LOG_LEVEL") {
                config.log_level = value;
            } else if (key == "LOG_FILE_PATH") {
                config.log_file_path = value;
            } else if (key == "SESSION_DIR") {
                config.session_dir = value;
            } else if (key == "LOG_MAX_SIZE_MB") {
                config.log_max_size_mb = static_cast<size_t>(std::stoull(value));
            } else if (key == "LOG_MAX_FILES") {
                config.log_max_files = static_cast<size_t>(std::stoull(value));
            } else if (key == "LOG_COMPRESS") {
                std::string lower = value;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                config.log_compress =
                    (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
            } else if (key == "D_MODEL") {
                config.d_model = static_cast<size_t>(std::stoull(value));
            } else if (key == "NUM_HEADS") {
                config.num_heads = static_cast<size_t>(std::stoull(value));
            } else if (key == "D_FF") {
                config.d_ff = static_cast<size_t>(std::stoull(value));
            } else if (key == "NUM_ENCODER_LAYERS") {
                config.num_encoder_layers = static_cast<size_t>(std::stoull(value));
            } else if (key == "NUM_DECODER_LAYERS") {
                config.num_decoder_layers = static_cast<size_t>(std::stoull(value));
            } else if (key == "MAX_SEQ_LENGTH") {
                config.max_seq_length = static_cast<size_t>(std::stoull(value));
                // Training hyperparameters
            } else if (key == "LEARNING_RATE") {
                config.learning_rate = std::stof(value);
            } else if (key == "NUM_EPOCHS") {
                config.num_epochs = std::stoi(value);
            } else if (key == "WEIGHT_DECAY") {
                config.weight_decay = std::stof(value);
            } else if (key == "GRADIENT_CLIP") {
                config.gradient_clip = std::stof(value);
            } else if (key == "GRADIENT_CLIP_ADAPTIVE") {
                config.adaptive_gradient_clip = (value == "true" || value == "1" || value == "yes");
            } else if (key == "GRADIENT_CLIP_MIN") {
                config.gradient_clip_min = std::stof(value);
            } else if (key == "GRADIENT_CLIP_MAX") {
                config.gradient_clip_max = std::stof(value);
            } else if (key == "GRADIENT_CLIP_EMA_DECAY") {
                config.gradient_clip_ema_decay = std::stof(value);
            } else if (key == "GRADIENT_CLIP_HEADROOM") {
                config.gradient_clip_headroom = std::stof(value);
            } else if (key == "GRADIENT_CLIP_WARMUP_STEPS") {
                config.gradient_clip_warmup_steps = std::stoi(value);
            } else if (key == "GRADIENT_CLIP_SPIKE_K") {
                config.gradient_clip_spike_k = std::stof(value);
            } else if (key == "BATCH_SIZE") {
                config.batch_size = std::stoi(value);
            } else if (key == "MAX_LENGTH" || key == "MAX_GEN_LENGTH") {
                config.max_gen_length = static_cast<size_t>(std::stoull(value));
            } else if (key == "TEMPERATURE") {
                config.temperature = std::stof(value);
            } else if (key == "TOP_P") {
                config.top_p = std::stof(value);
            } else if (key == "TOP_K") {
                config.top_k = std::stoi(value);
            } else if (key == "BEAM_WIDTH") {
                config.beam_width = std::stoi(value);
            } else if (key == "STRATEGY") {
                config.strategy = value;
                // Training Metrics Service configuration
            } else if (key == "ENABLE_METRICS_SERVICE") {
                config.enable_metrics_service = (value == "true" || value == "1" || value == "yes");
            } else if (key == "METRICS_PUSH_ENABLED") {
                config.metrics_push_enabled = (value == "true" || value == "1" || value == "yes");
            } else if (key == "METRICS_SERVER_URL") {
                config.metrics_server_url = value;
            } else if (key == "METRICS_PUSH_TIMEOUT_MS") {
                config.metrics_push_timeout_ms = std::stoi(value);
            } else if (key == "METRICS_ENABLE_PERSISTENCE") {
                config.metrics_enable_persistence =
                    (value == "true" || value == "1" || value == "yes");
            } else if (key == "METRICS_FILE") {
                config.metrics_file = value;
            } else if (key == "METRICS_SUMMARY_FILE") {
                config.metrics_summary_file = value;
            } else if (key == "METRICS_PERSIST_EVERY_SAMPLES") {
                config.metrics_persist_every_samples = std::stoi(value);
            } else if (key == "METRICS_PERSIST_EVERY_SECONDS") {
                config.metrics_persist_every_seconds = std::stoi(value);
            } else if (key == "METRICS_MAX_RECORDS_IN_MEMORY") {
                config.metrics_max_records_in_memory = std::stoi(value);
            } else if (key == "METRICS_MAX_RECORDS_ON_DISK") {
                config.metrics_max_records_on_disk = std::stoi(value);
            } else if (key == "METRICS_ENABLE_PROMETHEUS") {
                config.metrics_enable_prometheus =
                    (value == "true" || value == "1" || value == "yes");
            } else if (key == "METRICS_PROMETHEUS_FILE") {
                config.metrics_prometheus_file = value;
            } else if (key == "METRICS_SESSION_KEY") {
                config.metrics_session_key = value;
            } else if (key == "METRICS_MAX_LIVE_SESSIONS") {
                config.metrics_max_live_sessions = static_cast<size_t>(std::stoull(value));
            } else if (key == "METRICS_COMPLETED_TTL_SECONDS") {
                config.metrics_completed_ttl_seconds = std::stoi(value);
            } else if (key == "METRICS_SWEEP_INTERVAL_SECONDS") {
                config.metrics_sweep_interval_seconds = std::stoi(value);
            } else if (key == "METRICS_API_PORT") {
                config.metrics_api_port = std::stoi(value);
            } else if (key == "METRICS_API_ALLOW_CONTROL") {
                config.metrics_api_allow_control =
                    (value == "true" || value == "1" || value == "yes");
                // Generation quality metrics configuration
            } else if (key == "ENABLE_GENERATION_QUALITY_METRICS") {
                config.enable_generation_quality_metrics =
                    (value == "true" || value == "1" || value == "yes");
            } else if (key == "GENERATION_QUALITY_SAMPLE_SIZE") {
                config.generation_quality_sample_size = std::stoi(value);
            } else if (key == "GENERATION_QUALITY_MAX_TOKENS") {
                config.generation_quality_max_tokens = std::stoi(value);
                // RAG configuration
            } else if (key == "RAG_ENABLED") {
                std::string lower = value;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                config.rag_enabled =
                    (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
            } else if (key == "RAG_DOCS_PATH") {
                config.rag_docs_path = value;
            } else if (key == "RAG_NUM_DOCS") {
                config.rag_num_docs = std::stoi(value);
            } else if (key == "RAG_THRESHOLD") {
                config.rag_threshold = std::stof(value);
            } else if (key == "RAG_MAX_CONTEXT_LENGTH") {
                config.rag_max_context_length = std::stoi(value);
                // GPU configuration
            } else if (key == "GPU_ENABLED") {
                std::string lower = value;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                config.gpu_enabled =
                    (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
            } else if (key == "GPU_DEVICE_ID") {
                config.gpu_device_id = std::stoi(value);
            } else if (key == "GPU_MEMORY_FRACTION") {
                config.gpu_memory_fraction = std::stof(value);
            } else {
                std::cerr << "Warning: Unknown configuration key: " << key << '\n';
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid value for " << key << ": " << value << " (" << e.what()
                      << ")" << '\n';
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
    if (auto val = get_env_int("PORT")) {
        config.port = *val;
    }
    if (auto val = get_env_int("SESSION_TIMEOUT")) {
        config.session_timeout = *val;
    }
    if (auto val = get_env("LOG_LEVEL")) {
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
    if (auto val = get_env("STRATEGY")) {
        config.strategy = *val;
    }

    // Multi-instance metrics configuration
    if (auto val = get_env("METRICS_SESSION_KEY")) {
        config.metrics_session_key = *val;
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
}

// ============================================================
// Public API
// ============================================================

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

void ConfigLoader::print(const ServiceConfig& config) {
    std::cout << "==================================================" << '\n';
    std::cout << "         ADAI Chatbot Service Configuration" << '\n';
    std::cout << "==================================================" << '\n';
    std::cout << "Server Settings:" << '\n';
    std::cout << "  Model path:       "
              << (config.model_path.empty() ? "<new model>" : config.model_path) << '\n';
    std::cout << "  Vocabulary:       "
              << (config.vocab_path.empty() ? "<not set>" : config.vocab_path) << '\n';
    std::cout << "  Port:             " << config.port << '\n';
    std::cout << "  Session timeout:  " << config.session_timeout << " minutes" << '\n';
    std::cout << "  Log level:        " << config.log_level << '\n';
    std::cout << "  Log file:         "
              << (config.log_file_path.empty() ? "<console only>" : config.log_file_path) << '\n';
    std::cout << "  Session dir:      " << config.session_dir << '\n';
    if (!config.log_file_path.empty()) {
        std::cout << "  Log max size:     " << config.log_max_size_mb << " MB" << '\n';
        std::cout << "  Log max files:    " << config.log_max_files << '\n';
        std::cout << "  Log compression:  " << (config.log_compress ? "enabled" : "disabled")
                  << '\n';
    }
    std::cout << '\n';
    std::cout << "Model Architecture:" << '\n';
    std::cout << "  d_model:          " << config.d_model << '\n';
    std::cout << "  num_heads:        " << config.num_heads << '\n';
    std::cout << "  d_ff:             " << config.d_ff << '\n';
    std::cout << "  encoder_layers:   " << config.num_encoder_layers << '\n';
    std::cout << "  decoder_layers:   " << config.num_decoder_layers << '\n';
    std::cout << "  max_seq_length:   " << config.max_seq_length << '\n';
    std::cout << '\n';
    std::cout << "Generation Parameters:" << '\n';
    std::cout << "  max_length:       " << config.max_gen_length << '\n';
    std::cout << "  temperature:      " << config.temperature << '\n';
    std::cout << "  top_p:            " << config.top_p << '\n';
    std::cout << "  top_k:            " << config.top_k << '\n';
    std::cout << "  beam_width:       " << config.beam_width << '\n';
    std::cout << "  strategy:         " << config.strategy << '\n';
    std::cout << '\n';
    std::cout << "RAG Configuration:" << '\n';
    std::cout << "  rag_enabled:      " << (config.rag_enabled ? "true" : "false") << '\n';
    if (config.rag_enabled) {
        std::cout << "  rag_docs_path:    "
                  << (config.rag_docs_path.empty() ? "<not set>" : config.rag_docs_path) << '\n';
        std::cout << "  rag_num_docs:     " << config.rag_num_docs << '\n';
        std::cout << "  rag_threshold:    " << config.rag_threshold << '\n';
        std::cout << "  rag_max_context:  " << config.rag_max_context_length << " tokens" << '\n';
    }
    std::cout << "==================================================" << '\n';
}

// ============================================================
// Configuration Hot-Reloading
// ============================================================

bool ConfigLoader::reload(ServiceConfig& config, const std::string& config_file_path,
                          std::mutex& mutex) {
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

    // Detect changes before applying new config
    std::vector<std::string> changes = detect_changes(config, new_config);

    if (changes.empty()) {
        Logger::info("No configuration changes detected");
        Logger::info("==================================================");
        return true;
    }

    // Apply the new configuration with thread safety
    {
        std::lock_guard<std::mutex> lock(mutex);

        // Copy the old config for comparison
        ServiceConfig old_config = config;

        // Apply new configuration
        config = new_config;

        Logger::info("Configuration updated successfully");
        Logger::info("Changes applied:");
        for (const auto& change : changes) {
            Logger::info("  {}", change);
        }
    }

    Logger::info("==================================================");
    return true;
}

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

    // Validate generation parameters
    if (config.max_gen_length < 1 || config.max_gen_length > 4096) {
        errors.push_back("Invalid max_gen_length: " + std::to_string(config.max_gen_length) +
                         " (must be 1-4096)");
        valid = false;
    }

    if (config.temperature < 0.0f || config.temperature > 2.0f) {
        errors.push_back("Invalid temperature: " + std::to_string(config.temperature) +
                         " (must be 0.0-2.0)");
        valid = false;
    }

    if (config.top_p < 0.0f || config.top_p > 1.0f) {
        errors.push_back("Invalid top_p: " + std::to_string(config.top_p) + " (must be 0.0-1.0)");
        valid = false;
    }

    if (config.top_k < 1 || config.top_k > 1000) {
        errors.push_back("Invalid top_k: " + std::to_string(config.top_k) + " (must be 1-1000)");
        valid = false;
    }

    if (config.beam_width < 1 || config.beam_width > 16) {
        errors.push_back("Invalid beam_width: " + std::to_string(config.beam_width) +
                         " (must be 1-16)");
        valid = false;
    }

    // Validate strategy
    std::vector<std::string> valid_strategies = {"greedy", "beam", "temperature", "top_k",
                                                 "nucleus"};
    if (std::find(valid_strategies.begin(), valid_strategies.end(), config.strategy) ==
        valid_strategies.end()) {
        errors.push_back("Invalid strategy: " + config.strategy +
                         " (must be greedy, beam, temperature, top_k, or nucleus)");
        valid = false;
    }

    return valid;
}

std::vector<std::string> ConfigLoader::detect_changes(const ServiceConfig& old_config,
                                                      const ServiceConfig& new_config) {
    std::vector<std::string> changes;

    // Server configuration changes
    if (old_config.model_path != new_config.model_path) {
        changes.push_back("model_path: '" + old_config.model_path + "' -> '" +
                          new_config.model_path + "'");
    }
    if (old_config.vocab_path != new_config.vocab_path) {
        changes.push_back("vocab_path: '" + old_config.vocab_path + "' -> '" +
                          new_config.vocab_path + "'");
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
