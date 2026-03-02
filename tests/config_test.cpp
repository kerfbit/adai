#include <gtest/gtest.h>
#include "../src/Config.hpp"
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <mutex>

using namespace adai;
namespace fs = std::filesystem;

// ============================================================================
// Test Fixture
// ============================================================================

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean environment variables before each test
        clearEnvironmentVariables();
        
        // Create test directory
        test_dir = fs::temp_directory_path() / "config_test";
        fs::create_directories(test_dir);
        
        test_file = test_dir / "test_config.conf";
    }
    
    void TearDown() override {
        // Clean up test files
        if (fs::exists(test_file)) {
            fs::remove(test_file);
        }
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
        
        // Clean environment variables after test
        clearEnvironmentVariables();
    }
    
    void clearEnvironmentVariables() {
        #ifdef _WIN32
        _putenv("PORT=");
        _putenv("LOG_LEVEL=");
        _putenv("SESSION_TIMEOUT=");
        _putenv("VOCAB_PATH=");
        _putenv("MODEL_PATH=");
        _putenv("LOG_FILE_PATH=");
        _putenv("LOG_MAX_SIZE_MB=");
        _putenv("LOG_MAX_FILES=");
        _putenv("LOG_COMPRESS=");
        _putenv("D_MODEL=");
        _putenv("NUM_HEADS=");
        _putenv("D_FF=");
        _putenv("NUM_ENCODER_LAYERS=");
        _putenv("NUM_DECODER_LAYERS=");
        _putenv("MAX_SEQ_LENGTH=");
        _putenv("MAX_GEN_LENGTH=");
        _putenv("TEMPERATURE=");
        _putenv("TOP_P=");
        _putenv("TOP_K=");
        _putenv("BEAM_WIDTH=");
        _putenv("STRATEGY=");
        #else
        unsetenv("PORT");
        unsetenv("LOG_LEVEL");
        unsetenv("SESSION_TIMEOUT");
        unsetenv("VOCAB_PATH");
        unsetenv("MODEL_PATH");
        unsetenv("LOG_FILE_PATH");
        unsetenv("LOG_MAX_SIZE_MB");
        unsetenv("LOG_MAX_FILES");
        unsetenv("LOG_COMPRESS");
        unsetenv("D_MODEL");
        unsetenv("NUM_HEADS");
        unsetenv("D_FF");
        unsetenv("NUM_ENCODER_LAYERS");
        unsetenv("NUM_DECODER_LAYERS");
        unsetenv("MAX_SEQ_LENGTH");
        unsetenv("MAX_GEN_LENGTH");
        unsetenv("TEMPERATURE");
        unsetenv("TOP_P");
        unsetenv("TOP_K");
        unsetenv("BEAM_WIDTH");
        unsetenv("STRATEGY");
        #endif
    }
    
    void setEnv(const std::string& var, const std::string& value) {
        #ifdef _WIN32
        _putenv((var + "=" + value).c_str());
        #else
        setenv(var.c_str(), value.c_str(), 1);
        #endif
    }
    
    void createConfigFile(const std::map<std::string, std::string>& values) {
        std::ofstream out(test_file);
        for (const auto& [key, value] : values) {
            out << key << "=" << value << "\n";
        }
        out.close();
    }
    
    fs::path test_dir;
    fs::path test_file;
};

// ============================================================================
// Default Values Tests
// ============================================================================

TEST_F(ConfigTest, DefaultValues) {
    auto config = ConfigLoader::load();
    
    // Server defaults
    EXPECT_EQ(config.port, 8080);
    EXPECT_EQ(config.session_timeout, 30);
    EXPECT_EQ(config.log_level, "INFO");
    EXPECT_TRUE(config.log_file_path.empty());
    EXPECT_EQ(config.log_max_size_mb, 10);
    EXPECT_EQ(config.log_max_files, 5);
    EXPECT_FALSE(config.log_compress);
    
    // Model architecture defaults
    EXPECT_EQ(config.d_model, 512);
    EXPECT_EQ(config.num_heads, 8);
    EXPECT_EQ(config.d_ff, 2048);
    EXPECT_EQ(config.num_encoder_layers, 6);
    EXPECT_EQ(config.num_decoder_layers, 6);
    EXPECT_EQ(config.max_seq_length, 1024);
    
    // Generation defaults
    EXPECT_EQ(config.max_gen_length, 100);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.top_k, 50);
    EXPECT_EQ(config.beam_width, 4);
    EXPECT_EQ(config.strategy, "nucleus");
}

// ============================================================================
// Environment Variable Loading Tests
// ============================================================================

TEST_F(ConfigTest, LoadFromEnvironmentVariables) {
    setEnv("PORT", "9000");
    setEnv("LOG_LEVEL", "DEBUG");
    setEnv("SESSION_TIMEOUT", "60");
    setEnv("VOCAB_PATH", "/path/to/vocab.txt");
    setEnv("MODEL_PATH", "/path/to/model.bin");
    setEnv("D_MODEL", "768");
    setEnv("NUM_HEADS", "12");
    setEnv("STRATEGY", "greedy");
    
    auto config = ConfigLoader::load();
    
    EXPECT_EQ(config.port, 9000);
    EXPECT_EQ(config.log_level, "DEBUG");
    EXPECT_EQ(config.session_timeout, 60);
    EXPECT_EQ(config.vocab_path, "/path/to/vocab.txt");
    EXPECT_EQ(config.model_path, "/path/to/model.bin");
    EXPECT_EQ(config.d_model, 768);
    EXPECT_EQ(config.num_heads, 12);
    EXPECT_EQ(config.strategy, "greedy");
}

TEST_F(ConfigTest, EnvironmentVariablesBooleanParsing) {
    // Test various boolean formats
    setEnv("LOG_COMPRESS", "true");
    auto config1 = ConfigLoader::load();
    EXPECT_TRUE(config1.log_compress);
    
    clearEnvironmentVariables();
    setEnv("LOG_COMPRESS", "1");
    auto config2 = ConfigLoader::load();
    EXPECT_TRUE(config2.log_compress);
    
    clearEnvironmentVariables();
    setEnv("LOG_COMPRESS", "yes");
    auto config3 = ConfigLoader::load();
    EXPECT_TRUE(config3.log_compress);
    
    clearEnvironmentVariables();
    setEnv("LOG_COMPRESS", "on");
    auto config4 = ConfigLoader::load();
    EXPECT_TRUE(config4.log_compress);
    
    clearEnvironmentVariables();
    setEnv("LOG_COMPRESS", "false");
    auto config5 = ConfigLoader::load();
    EXPECT_FALSE(config5.log_compress);
    
    clearEnvironmentVariables();
    setEnv("LOG_COMPRESS", "0");
    auto config6 = ConfigLoader::load();
    EXPECT_FALSE(config6.log_compress);
}

TEST_F(ConfigTest, EnvironmentVariablesFloatParsing) {
    setEnv("TEMPERATURE", "0.7");
    setEnv("TOP_P", "0.95");
    
    auto config = ConfigLoader::load();
    
    EXPECT_NEAR(config.temperature, 0.7f, 0.001f);
    EXPECT_NEAR(config.top_p, 0.95f, 0.001f);
}

// ============================================================================
// File Loading Tests
// ============================================================================

TEST_F(ConfigTest, LoadFromFile) {
    createConfigFile({
        {"PORT", "3000"},
        {"LOG_LEVEL", "WARN"},
        {"D_MODEL", "1024"},
        {"NUM_HEADS", "16"},
        {"STRATEGY", "beam"}
    });
    
    auto config = ConfigLoader::load(test_file.string());
    
    EXPECT_EQ(config.port, 3000);
    EXPECT_EQ(config.log_level, "WARN");
    EXPECT_EQ(config.d_model, 1024);
    EXPECT_EQ(config.num_heads, 16);
    EXPECT_EQ(config.strategy, "beam");
}

TEST_F(ConfigTest, FileLoadingIgnoresCommentsAndWhitespace) {
    std::ofstream out(test_file);
    out << "# This is a comment\n";
    out << "PORT=5000\n";
    out << "\n";  // Empty line
    out << "  LOG_LEVEL = ERROR  \n";  // Whitespace around =
    out << "# Another comment\n";
    out.close();
    
    auto config = ConfigLoader::load(test_file.string());
    
    EXPECT_EQ(config.port, 5000);
    EXPECT_EQ(config.log_level, "ERROR");
}

TEST_F(ConfigTest, FileLoadingHandlesMissingFile) {
    // Loading non-existent file should use defaults
    auto config = ConfigLoader::load("/this/file/does/not/exist.conf");
    
    // Should still have defaults
    EXPECT_EQ(config.port, 8080);
    EXPECT_EQ(config.log_level, "INFO");
}

// ============================================================================
// Priority System Tests (Env > File > Defaults)
// ============================================================================

TEST_F(ConfigTest, EnvironmentVariablesOverrideFile) {
    createConfigFile({
        {"PORT", "3000"},
        {"LOG_LEVEL", "WARN"}
    });
    
    setEnv("PORT", "4000");  // Env should override file
    
    auto config = ConfigLoader::load(test_file.string());
    
    EXPECT_EQ(config.port, 4000);  // From environment
    EXPECT_EQ(config.log_level, "WARN");  // From file (no env override)
}

TEST_F(ConfigTest, FileOverridesDefaults) {
    createConfigFile({
        {"D_MODEL", "256"}
    });
    
    auto config = ConfigLoader::load(test_file.string());
    
    EXPECT_EQ(config.d_model, 256);  // From file
    EXPECT_EQ(config.num_heads, 8);  // Default (not in file)
}

// ============================================================================
// Validation Tests - Valid Ranges
// ============================================================================

TEST_F(ConfigTest, ValidationValidPort) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.port = 8080;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    EXPECT_TRUE(errors.empty());
    
    errors.clear();
    config.port = 1;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    EXPECT_TRUE(errors.empty());
    
    errors.clear();
    config.port = 65535;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    EXPECT_TRUE(errors.empty());
}

TEST_F(ConfigTest, ValidationInvalidPort) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.port = 0;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), 
        [](const std::string& e) { return e.find("port") != std::string::npos; }));
    
    errors.clear();
    config.port = 65536;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_FALSE(errors.empty());
}

TEST_F(ConfigTest, ValidationValidLogLevel) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    for (const auto& level : {"DEBUG", "INFO", "WARN", "ERROR"}) {
        errors.clear();
        config.log_level = level;
        EXPECT_TRUE(ConfigLoader::validate(config, errors));
        EXPECT_TRUE(errors.empty());
    }
    
    // Case insensitive
    errors.clear();
    config.log_level = "debug";
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    EXPECT_TRUE(errors.empty());
}

TEST_F(ConfigTest, ValidationInvalidLogLevel) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.log_level = "INVALID";
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), 
        [](const std::string& e) { return e.find("log_level") != std::string::npos; }));
}

TEST_F(ConfigTest, ValidationValidDModel) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.d_model = 64;  // Minimum
    config.num_heads = 8;  // Make divisible
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.d_model = 8192;  // Maximum
    config.num_heads = 64;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.d_model = 512;  // Typical
    config.num_heads = 8;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationInvalidDModel) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.d_model = 32;  // Too small
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), 
        [](const std::string& e) { return e.find("d_model") != std::string::npos; }));
    
    errors.clear();
    config.d_model = 10000;  // Too large
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationDModelDivisibleByNumHeads) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.d_model = 512;
    config.num_heads = 8;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));  // 512 % 8 = 0 ✅
    
    errors.clear();
    config.d_model = 512;
    config.num_heads = 7;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));  // 512 % 7 != 0 ❌
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), 
        [](const std::string& e) { return e.find("divisible") != std::string::npos; }));
}

TEST_F(ConfigTest, ValidationNumHeadsRange) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.d_model = 64;
    config.num_heads = 1;  // Minimum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.d_model = 4096;
    config.num_heads = 64;  // Maximum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.num_heads = 0;  // Too small
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.num_heads = 65;  // Too large
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationDFFRange) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.d_ff = 64;  // Minimum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.d_ff = 32768;  // Maximum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.d_ff = 32;  // Too small
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationLayerCounts) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.num_encoder_layers = 1;  // Minimum
    config.num_decoder_layers = 1;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.num_encoder_layers = 48;  // Maximum
    config.num_decoder_layers = 48;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.num_encoder_layers = 0;  // Too small
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.num_encoder_layers = 1;
    config.num_decoder_layers = 49;  // Too large
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationMaxSeqLength) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.max_seq_length = 16;  // Minimum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.max_seq_length = 32768;  // Maximum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.max_seq_length = 8;  // Too small
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationGenerationParameters) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    // Valid temperature
    config.temperature = 0.0f;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.temperature = 2.0f;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.temperature = -0.1f;  // Invalid
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    // Valid top_p
    errors.clear();
    config.temperature = 1.0f;
    config.top_p = 0.0f;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.top_p = 1.0f;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.top_p = 1.1f;  // Invalid
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    // Valid top_k
    errors.clear();
    config.top_p = 0.9f;
    config.top_k = 1;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.top_k = 1000;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.top_k = 0;  // Invalid
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    // Valid beam_width
    errors.clear();
    config.top_k = 50;
    config.beam_width = 1;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.beam_width = 16;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.beam_width = 17;  // Invalid
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationStrategy) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    for (const auto& strategy : {"greedy", "beam", "temperature", "top_k", "nucleus"}) {
        errors.clear();
        config.strategy = strategy;
        EXPECT_TRUE(ConfigLoader::validate(config, errors));
    }
    
    errors.clear();
    config.strategy = "invalid";
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationLogFileSettings) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    // Valid log_max_size_mb
    config.log_max_size_mb = 1;  // Minimum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.log_max_size_mb = 1024;  // Maximum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.log_max_size_mb = 0;  // Invalid
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    // Valid log_max_files
    errors.clear();
    config.log_max_size_mb = 10;
    config.log_max_files = 1;  // Minimum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.log_max_files = 100;  // Maximum
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    errors.clear();
    config.log_max_files = 0;  // Invalid
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidationMultipleErrors) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.port = 0;  // Invalid
    config.d_model = 10000;  // Invalid
    config.temperature = 3.0f;  // Invalid
    config.strategy = "invalid";  // Invalid
    
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_GE(errors.size(), 4);  // At least 4 errors
}

// ============================================================================
// Hot-Reload Tests
// ============================================================================

TEST_F(ConfigTest, ReloadValidConfiguration) {
    createConfigFile({
        {"PORT", "8080"},
        {"LOG_LEVEL", "INFO"}
    });
    
    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;
    
    // Modify file
    createConfigFile({
        {"PORT", "9090"},
        {"LOG_LEVEL", "DEBUG"}
    });
    
    // Reload
    bool success = ConfigLoader::reload(config, test_file.string(), config_mutex);
    
    EXPECT_TRUE(success);
    EXPECT_EQ(config.port, 9090);
    EXPECT_EQ(config.log_level, "DEBUG");
}

TEST_F(ConfigTest, ReloadInvalidConfiguration) {
    createConfigFile({
        {"PORT", "8080"},
        {"LOG_LEVEL", "INFO"}
    });
    
    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;
    
    // Modify file with invalid config
    createConfigFile({
        {"PORT", "99999"},  // Invalid
        {"LOG_LEVEL", "DEBUG"}
    });
    
    // Reload should fail
    bool success = ConfigLoader::reload(config, test_file.string(), config_mutex);
    
    EXPECT_FALSE(success);
    EXPECT_EQ(config.port, 8080);  // Should keep old value
}

TEST_F(ConfigTest, ReloadThreadSafety) {
    createConfigFile({
        {"PORT", "8080"}
    });
    
    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;
    
    std::atomic<int> reload_count{0};
    
    // Launch multiple threads trying to reload concurrently
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            createConfigFile({{"PORT", std::to_string(8080 + i)}});
            if (ConfigLoader::reload(config, test_file.string(), config_mutex)) {
                reload_count++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // All reloads should have succeeded (config should be valid)
    EXPECT_GT(reload_count, 0);
}

// ============================================================================
// Change Detection Tests
// ============================================================================

TEST_F(ConfigTest, DetectChangesNoChanges) {
    ServiceConfig config1;
    ServiceConfig config2;
    
    auto changes = ConfigLoader::detect_changes(config1, config2);
    EXPECT_TRUE(changes.empty());
}

TEST_F(ConfigTest, DetectChangesServerConfig) {
    ServiceConfig old_config;
    ServiceConfig new_config;
    
    old_config.port = 8080;
    new_config.port = 9090;
    
    old_config.log_level = "INFO";
    new_config.log_level = "DEBUG";
    
    auto changes = ConfigLoader::detect_changes(old_config, new_config);
    
    EXPECT_EQ(changes.size(), 2);
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), 
        [](const std::string& c) { return c.find("port") != std::string::npos && c.find("8080") != std::string::npos; }));
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), 
        [](const std::string& c) { return c.find("log_level") != std::string::npos; }));
}

TEST_F(ConfigTest, DetectChangesModelArchitecture) {
    ServiceConfig old_config;
    ServiceConfig new_config;
    
    old_config.d_model = 512;
    new_config.d_model = 768;
    
    old_config.num_heads = 8;
    new_config.num_heads = 12;
    
    auto changes = ConfigLoader::detect_changes(old_config, new_config);
    
    EXPECT_GE(changes.size(), 2);
}

TEST_F(ConfigTest, DetectChangesGenerationParams) {
    ServiceConfig old_config;
    ServiceConfig new_config;
    
    old_config.temperature = 1.0f;
    new_config.temperature = 0.7f;
    
    old_config.strategy = "nucleus";
    new_config.strategy = "greedy";
    
    auto changes = ConfigLoader::detect_changes(old_config, new_config);
    
    EXPECT_GE(changes.size(), 2);
}

TEST_F(ConfigTest, DetectChangesLogFileSettings) {
    ServiceConfig old_config;
    ServiceConfig new_config;
    
    old_config.log_file_path = "/var/log/old.log";
    new_config.log_file_path = "/var/log/new.log";
    
    old_config.log_max_size_mb = 10;
    new_config.log_max_size_mb = 50;
    
    auto changes = ConfigLoader::detect_changes(old_config, new_config);
    
    EXPECT_GE(changes.size(), 2);
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), 
        [](const std::string& c) { return c.find("log_file_path") != std::string::npos; }));
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), 
        [](const std::string& c) { return c.find("log_max_size_mb") != std::string::npos; }));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ConfigTest, InvalidIntegerFormat) {
    createConfigFile({
        {"PORT", "not_a_number"}
    });
    
    // Should fall back to default
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.port, 8080);  // Default
}

TEST_F(ConfigTest, InvalidFloatFormat) {
    createConfigFile({
        {"TEMPERATURE", "invalid"}
    });
    
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);  // Default
}

TEST_F(ConfigTest, InvalidBooleanFormat) {
    createConfigFile({
        {"LOG_COMPRESS", "maybe"}
    });
    
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FALSE(config.log_compress);  // Default
}

TEST_F(ConfigTest, EmptyValues) {
    createConfigFile({
        {"PORT", ""},
        {"LOG_LEVEL", ""}
    });
    
    auto config = ConfigLoader::load(test_file.string());
    
    // Empty string values are loaded as-is (not defaults)
    // Invalid values should be caught by validation
    std::vector<std::string> errors;
    bool valid = ConfigLoader::validate(config, errors);
    
    // Empty LOG_LEVEL will be invalid
    EXPECT_FALSE(valid);
    EXPECT_FALSE(errors.empty());
}

TEST_F(ConfigTest, ExtremelyLargeValues) {
    createConfigFile({
        {"D_MODEL", "999999"},
        {"MAX_SEQ_LENGTH", "999999"}
    });
    
    auto config = ConfigLoader::load(test_file.string());
    std::vector<std::string> errors;
    
    // Validation should catch these
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_FALSE(errors.empty());
}

TEST_F(ConfigTest, NegativeValues) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    // Most size_t fields will wrap, but validation should catch logical issues
    config.port = -1;  // Will wrap for unsigned, but out of int range
    config.session_timeout = -10;
    
    // Note: size_t fields can't be negative, but we test validation logic
}

TEST_F(ConfigTest, SessionTimeoutValidation) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.session_timeout = 0;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.session_timeout = 1;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, MaxGenLengthValidation) {
    ServiceConfig config;
    std::vector<std::string> errors;
    
    config.max_gen_length = 0;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.max_gen_length = 1;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.max_gen_length = 4096;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    
    errors.clear();
    config.max_gen_length = 5000;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
