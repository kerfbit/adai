#include "Config.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

// TODO: See TECHNICAL_DEBT.md Future Enhancement #3 - Configuration Hot-Reloading
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
            std::cerr << "Warning: Invalid integer value for " << var_name 
                      << ": " << *value << std::endl;
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
            std::cerr << "Warning: Invalid size_t value for " << var_name 
                      << ": " << *value << std::endl;
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
            std::cerr << "Warning: Invalid float value for " << var_name 
                      << ": " << *value << std::endl;
        }
    }
    return std::nullopt;
}

// ============================================================
// Configuration File Parsing
// ============================================================

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

void ConfigLoader::load_from_file(ServiceConfig& config, const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        // Config file is optional, so just log a warning
        std::cerr << "Note: Configuration file not found: " << file_path << std::endl;
        return;
    }
    
    std::cout << "Loading configuration from: " << file_path << std::endl;
    
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
            std::cerr << "Warning: Invalid line " << line_number 
                      << " in config file: " << line << std::endl;
            continue;
        }
        
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        
        // Remove quotes from value if present
        if (value.size() >= 2 && 
            ((value.front() == '"' && value.back() == '"') ||
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
            } else {
                std::cerr << "Warning: Unknown configuration key: " << key << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid value for " << key << ": " << value 
                      << " (" << e.what() << ")" << std::endl;
        }
    }
}

// ============================================================
// Environment Variable Loading
// ============================================================

void ConfigLoader::load_from_env(ServiceConfig& config) {
    // Server configuration
    if (auto val = get_env("MODEL_PATH")) config.model_path = *val;
    if (auto val = get_env("VOCAB_PATH")) config.vocab_path = *val;
    if (auto val = get_env_int("PORT")) config.port = *val;
    if (auto val = get_env_int("SESSION_TIMEOUT")) config.session_timeout = *val;
    if (auto val = get_env("LOG_LEVEL")) config.log_level = *val;
    
    // Model architecture
    if (auto val = get_env_size_t("D_MODEL")) config.d_model = *val;
    if (auto val = get_env_size_t("NUM_HEADS")) config.num_heads = *val;
    if (auto val = get_env_size_t("D_FF")) config.d_ff = *val;
    if (auto val = get_env_size_t("NUM_ENCODER_LAYERS")) config.num_encoder_layers = *val;
    if (auto val = get_env_size_t("NUM_DECODER_LAYERS")) config.num_decoder_layers = *val;
    if (auto val = get_env_size_t("MAX_SEQ_LENGTH")) config.max_seq_length = *val;
    
    // Generation parameters
    // Support both MAX_LENGTH and MAX_GEN_LENGTH for compatibility
    if (auto val = get_env_size_t("MAX_GEN_LENGTH")) config.max_gen_length = *val;
    if (auto val = get_env_size_t("MAX_LENGTH")) config.max_gen_length = *val;
    
    if (auto val = get_env_float("TEMPERATURE")) config.temperature = *val;
    if (auto val = get_env_float("TOP_P")) config.top_p = *val;
    if (auto val = get_env_int("TOP_K")) config.top_k = *val;
    if (auto val = get_env_int("BEAM_WIDTH")) config.beam_width = *val;
    if (auto val = get_env("STRATEGY")) config.strategy = *val;
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
    std::cout << "==================================================" << std::endl;
    std::cout << "         ADAI Chatbot Service Configuration" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Server Settings:" << std::endl;
    std::cout << "  Model path:       " << (config.model_path.empty() ? "<new model>" : config.model_path) << std::endl;
    std::cout << "  Vocabulary:       " << (config.vocab_path.empty() ? "<not set>" : config.vocab_path) << std::endl;
    std::cout << "  Port:             " << config.port << std::endl;
    std::cout << "  Session timeout:  " << config.session_timeout << " minutes" << std::endl;
    std::cout << "  Log level:        " << config.log_level << std::endl;
    std::cout << std::endl;
    std::cout << "Model Architecture:" << std::endl;
    std::cout << "  d_model:          " << config.d_model << std::endl;
    std::cout << "  num_heads:        " << config.num_heads << std::endl;
    std::cout << "  d_ff:             " << config.d_ff << std::endl;
    std::cout << "  encoder_layers:   " << config.num_encoder_layers << std::endl;
    std::cout << "  decoder_layers:   " << config.num_decoder_layers << std::endl;
    std::cout << "  max_seq_length:   " << config.max_seq_length << std::endl;
    std::cout << std::endl;
    std::cout << "Generation Parameters:" << std::endl;
    std::cout << "  max_length:       " << config.max_gen_length << std::endl;
    std::cout << "  temperature:      " << config.temperature << std::endl;
    std::cout << "  top_p:            " << config.top_p << std::endl;
    std::cout << "  top_k:            " << config.top_k << std::endl;
    std::cout << "  beam_width:       " << config.beam_width << std::endl;
    std::cout << "  strategy:         " << config.strategy << std::endl;
    std::cout << "==================================================" << std::endl;
}

} // namespace adai
