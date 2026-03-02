#pragma once

#include <string>
#include <map>
#include <optional>
#include <mutex>
#include <vector>

// Hot-reloading implemented - see DAEMON_IMPLEMENTATION_COMPLETE.md
// TODO: See TECHNICAL_DEBT.md Future Enhancement #4 - JSON Configuration Format Support
//       Add support for JSON format in addition to key=value format
// TODO: See TECHNICAL_DEBT.md Future Enhancement #5 - Configuration Profiles
//       Support named profiles (dev, staging, prod) with inheritance

namespace adai {

/**
 * @brief Configuration for the ADAI chatbot service.
 * 
 * Configuration values are loaded with the following priority (highest to lowest):
 * 1. Environment variables
 * 2. Configuration file
 * 3. Default values
 */
struct ServiceConfig {
    // ============================================================
    // Server Configuration
    // ============================================================
    
    /// Path to the model file
    std::string model_path;
    
    /// Path to the vocabulary file
    std::string vocab_path;
    
    /// Server port (default: 8080)
    int port = 8080;
    
    /// Session timeout in minutes (default: 30)
    int session_timeout = 30;
    
    /// Logging level: DEBUG, INFO, WARN, ERROR (default: INFO)
    std::string log_level = "INFO";
    
    /// Log file path (empty = console only, set to enable file logging)
    std::string log_file_path;
    
    /// Maximum log file size in MB before rotation (default: 10)
    size_t log_max_size_mb = 10;
    
    /// Maximum number of rotated log files to keep (default: 5)
    size_t log_max_files = 5;
    
    /// Enable log file compression (default: false)
    bool log_compress = false;
    
    // ============================================================
    // Model Architecture Parameters
    // ============================================================
    
    /// Model dimension (default: 512)
    size_t d_model = 512;
    
    /// Number of attention heads (default: 8)
    size_t num_heads = 8;
    
    /// Feed-forward dimension (default: 2048)
    size_t d_ff = 2048;
    
    /// Number of encoder layers (default: 6)
    size_t num_encoder_layers = 6;
    
    /// Number of decoder layers (default: 6)
    size_t num_decoder_layers = 6;
    
    /// Maximum sequence length (default: 1024)
    size_t max_seq_length = 1024;
    
    // ============================================================
    // Generation Parameters
    // ============================================================
    
    /// Maximum generation length (default: 100)
    size_t max_gen_length = 100;
    
    /// Generation temperature (default: 1.0)
    float temperature = 1.0f;
    
    /// Nucleus sampling threshold (default: 0.9)
    float top_p = 0.9f;
    
    /// Top-k sampling parameter (default: 50)
    int top_k = 50;
    
    /// Beam search width (default: 4)
    int beam_width = 4;
    
    /// Generation strategy: greedy, beam, temperature, top_k, nucleus (default: nucleus)
    std::string strategy = "nucleus";
};

/**
 * @brief Configuration loader for the ADAI service.
 * 
 * Loads configuration from environment variables and/or a configuration file.
 */
class ConfigLoader {
public:
    /**
     * @brief Load configuration with defaults.
     */
    static ServiceConfig load();
    
    /**
     * @brief Load configuration from a file and environment variables.
     * 
     * @param config_file_path Path to the configuration file
     * @return ServiceConfig The loaded configuration
     */
    static ServiceConfig load(const std::string& config_file_path);
    
    /**
     * @brief Print the current configuration to stdout.
     * 
     * @param config The configuration to print
     */
    static void print(const ServiceConfig& config);
    
    /**
     * @brief Reload configuration from file with validation.
     * 
     * This method is thread-safe and validates the new configuration
     * before applying it. If validation fails, the current config remains unchanged.
     * 
     * @param config Current configuration to update
     * @param config_file_path Path to the configuration file
     * @param mutex Mutex to protect the config during update
     * @return true if reload succeeded, false if validation failed
     */
    static bool reload(ServiceConfig& config, const std::string& config_file_path, std::mutex& mutex);
    
    /**
     * @brief Validate configuration parameters.
     * 
     * @param config Configuration to validate
     * @param errors Vector to store validation error messages
     * @return true if configuration is valid, false otherwise
     */
    static bool validate(const ServiceConfig& config, std::vector<std::string>& errors);
    
    /**
     * @brief Detect and log configuration changes.
     * 
     * @param old_config Previous configuration
     * @param new_config New configuration
     * @return Vector of change descriptions
     */
    static std::vector<std::string> detect_changes(const ServiceConfig& old_config, const ServiceConfig& new_config);

private:
    /**
     * @brief Load configuration from environment variables.
     * 
     * @param config Configuration struct to update
     */
    static void load_from_env(ServiceConfig& config);
    
    /**
     * @brief Load configuration from a file.
     * 
     * Supports simple key=value format.
     * 
     * @param config Configuration struct to update
     * @param file_path Path to the configuration file
     */
    static void load_from_file(ServiceConfig& config, const std::string& file_path);
    
    /**
     * @brief Get environment variable as string.
     * 
     * @param var_name Environment variable name
     * @return std::optional<std::string> The value if present
     */
    static std::optional<std::string> get_env(const std::string& var_name);
    
    /**
     * @brief Get environment variable as integer.
     * 
     * @param var_name Environment variable name
     * @return std::optional<int> The value if present and valid
     */
    static std::optional<int> get_env_int(const std::string& var_name);
    
    /**
     * @brief Get environment variable as size_t.
     * 
     * @param var_name Environment variable name
     * @return std::optional<size_t> The value if present and valid
     */
    static std::optional<size_t> get_env_size_t(const std::string& var_name);
    
    /**
     * @brief Get environment variable as float.
     * 
     * @param var_name Environment variable name
     * @return std::optional<float> The value if present and valid
     */
    static std::optional<float> get_env_float(const std::string& var_name);
    
    /**
     * @brief Get environment variable as boolean.
     * 
     * @param var_name Environment variable name
     * @return std::optional<bool> The value if present and valid
     */
    static std::optional<bool> get_env_bool(const std::string& var_name);
};

} // namespace adai
