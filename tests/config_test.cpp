#include "../src/Config.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <sstream>
#include <spdlog/sinks/ostream_sink.h>
#include "../src/Logger.hpp"
#include <thread>

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
        _putenv("DRAFT_MODEL_PATH=");
        _putenv("SPECULATIVE_NUM_CANDIDATES=");
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
        _putenv("METRICS_SESSION_KEY=");
        _putenv("METRICS_MAX_LIVE_SESSIONS=");
        _putenv("METRICS_COMPLETED_TTL_SECONDS=");
        _putenv("METRICS_SWEEP_INTERVAL_SECONDS=");
        _putenv("ENABLE_METRICS_SERVICE=");
        _putenv("METRICS_SERVER_URL=");
        _putenv("METRICS_HEARTBEAT_INTERVAL_MS=");
        _putenv("WORLD_MODEL_ENABLED=");
        _putenv("WORLD_MODEL_D_MODEL=");
        _putenv("WORLD_MODEL_NUM_LAYERS=");
        _putenv("WORLD_MODEL_NUM_HEADS=");
        _putenv("WORLD_MODEL_D_FF=");
        _putenv("WORLD_MODEL_SIGREG_LAMBDA=");
        _putenv("WORLD_MODEL_SIGREG_NUM_SKETCHES=");
        _putenv("WORLD_MODEL_SIGREG_BATCH_SIZE=");
        _putenv("WORLD_MODEL_INJECT_EVERY_N_LAYERS=");
        _putenv("TRAINER_PASS_STALL_MINUTES=");  // TD-569
#else
        unsetenv("PORT");
        unsetenv("LOG_LEVEL");
        unsetenv("SESSION_TIMEOUT");
        unsetenv("VOCAB_PATH");
        unsetenv("MODEL_PATH");
        unsetenv("DRAFT_MODEL_PATH");
        unsetenv("SPECULATIVE_NUM_CANDIDATES");
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
        unsetenv("METRICS_SESSION_KEY");
        unsetenv("METRICS_MAX_LIVE_SESSIONS");
        unsetenv("METRICS_COMPLETED_TTL_SECONDS");
        unsetenv("METRICS_SWEEP_INTERVAL_SECONDS");
        unsetenv("GPU_ENABLED");
        unsetenv("GPU_DEVICE_ID");
        unsetenv("GPU_MEMORY_FRACTION");
        unsetenv("GPU_STRATEGY");
        // FTP Dataset Transport (Phase 10)
        unsetenv("FTP_SERVER_PORT");
        unsetenv("FTP_PASV_PORT_MIN");
        unsetenv("FTP_PASV_PORT_MAX");
        unsetenv("FTP_TOKEN_TTL_MINUTES");
        unsetenv("FTP_DATA_SERVER_SECRET");
        unsetenv("DOWNLOAD_DIR");
        unsetenv("MAX_PARALLEL_DOWNLOADS");
        unsetenv("LARGE_FILE_WARN_THRESHOLD_MB");
        unsetenv("FTPS_CA_FILE");
        unsetenv("FTPS_PINNED_PUBLIC_KEY");
        unsetenv("TOKENIZER_MODE");
        unsetenv("ENABLE_METRICS_SERVICE");
        unsetenv("METRICS_SERVER_URL");
        unsetenv("METRICS_HEARTBEAT_INTERVAL_MS");
        unsetenv("WORLD_MODEL_ENABLED");
        unsetenv("WORLD_MODEL_D_MODEL");
        unsetenv("WORLD_MODEL_NUM_LAYERS");
        unsetenv("WORLD_MODEL_NUM_HEADS");
        unsetenv("WORLD_MODEL_D_FF");
        unsetenv("WORLD_MODEL_SIGREG_LAMBDA");
        unsetenv("WORLD_MODEL_SIGREG_NUM_SKETCHES");
        unsetenv("WORLD_MODEL_SIGREG_BATCH_SIZE");
        unsetenv("WORLD_MODEL_INJECT_EVERY_N_LAYERS");
        unsetenv("TRAINER_PASS_STALL_MINUTES");  // TD-569
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

    // Multi-instance metrics defaults (TD-018 phase 5)
    EXPECT_TRUE(config.metrics_session_key.empty());
    EXPECT_EQ(config.metrics_max_live_sessions, 16);
    EXPECT_EQ(config.metrics_completed_ttl_seconds, 3600);
    EXPECT_EQ(config.metrics_sweep_interval_seconds, 60);
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

TEST_F(ConfigTest, LoadDraftModelSettingsFromEnvironmentVariables) {
    setEnv("DRAFT_MODEL_PATH", "/env/draft.bin");
    setEnv("SPECULATIVE_NUM_CANDIDATES", "8");

    auto config = ConfigLoader::load();

    EXPECT_EQ(config.draft_model_path, "/env/draft.bin");
    EXPECT_EQ(config.speculative_num_candidates, 8);
}

TEST_F(ConfigTest, LoadMultiInstanceMetricsFromEnvironmentVariables) {
    setEnv("METRICS_SESSION_KEY", "42-gpu0");
    setEnv("METRICS_MAX_LIVE_SESSIONS", "32");
    setEnv("METRICS_COMPLETED_TTL_SECONDS", "900");
    setEnv("METRICS_SWEEP_INTERVAL_SECONDS", "15");

    auto config = ConfigLoader::load();

    EXPECT_EQ(config.metrics_session_key, "42-gpu0");
    EXPECT_EQ(config.metrics_max_live_sessions, 32);
    EXPECT_EQ(config.metrics_completed_ttl_seconds, 900);
    EXPECT_EQ(config.metrics_sweep_interval_seconds, 15);
}

// TD-091: ENABLE_METRICS_SERVICE, METRICS_SERVER_URL, and
// METRICS_HEARTBEAT_INTERVAL_MS (plus 16 sibling metrics/generation-quality
// keys) were settable via the config file but had no load_from_env()
// counterpart at all, silently breaking "env vars override config file" for
// exactly these keys.
TEST_F(ConfigTest, LoadTrainingMetricsServiceKeysFromEnvironmentVariables) {
    // Deliberately set every value to differ from ServiceConfig's own default
    // (enable_metrics_service defaults true, metrics_server_url defaults to
    // "http://localhost:8081") — a value that happened to match the default
    // would pass this assertion whether or not the env var was ever read,
    // silently failing to catch a regression here.
    setEnv("ENABLE_METRICS_SERVICE", "false");
    setEnv("METRICS_SERVER_URL", "http://metrics-host:9999");
    setEnv("METRICS_HEARTBEAT_INTERVAL_MS", "5000");

    auto config = ConfigLoader::load();

    EXPECT_FALSE(config.enable_metrics_service);
    EXPECT_EQ(config.metrics_server_url, "http://metrics-host:9999");
    EXPECT_EQ(config.metrics_heartbeat_interval_ms, 5000);
}

TEST_F(ConfigTest, TrainingMetricsServiceEnvironmentVariablesOverrideFile) {
    createConfigFile({{"METRICS_SERVER_URL", "http://file-configured:8081"}});
    setEnv("METRICS_SERVER_URL", "http://env-configured:8081");

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.metrics_server_url, "http://env-configured:8081");
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
    createConfigFile({{"PORT", "3000"},
                      {"LOG_LEVEL", "WARN"},
                      {"D_MODEL", "1024"},
                      {"NUM_HEADS", "16"},
                      {"STRATEGY", "beam"}});

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.port, 3000);
    EXPECT_EQ(config.log_level, "WARN");
    EXPECT_EQ(config.d_model, 1024);
    EXPECT_EQ(config.num_heads, 16);
    EXPECT_EQ(config.strategy, "beam");
}

TEST_F(ConfigTest, LoadDraftModelSettingsFromFile) {
    // TD-038: speculative decoding config.
    createConfigFile(
        {{"DRAFT_MODEL_PATH", "/path/to/draft.bin"}, {"SPECULATIVE_NUM_CANDIDATES", "6"}});

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.draft_model_path, "/path/to/draft.bin");
    EXPECT_EQ(config.speculative_num_candidates, 6);
}

TEST_F(ConfigTest, DraftModelPathDefaultsEmptyAndDisabled) {
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_TRUE(config.draft_model_path.empty());
    EXPECT_EQ(config.speculative_num_candidates, 4);
}

TEST_F(ConfigTest, LoadMultiInstanceMetricsFromFile) {
    createConfigFile({{"METRICS_SESSION_KEY", "1-default"},
                      {"METRICS_MAX_LIVE_SESSIONS", "20"},
                      {"METRICS_COMPLETED_TTL_SECONDS", "1800"},
                      {"METRICS_SWEEP_INTERVAL_SECONDS", "45"}});

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.metrics_session_key, "1-default");
    EXPECT_EQ(config.metrics_max_live_sessions, 20);
    EXPECT_EQ(config.metrics_completed_ttl_seconds, 1800);
    EXPECT_EQ(config.metrics_sweep_interval_seconds, 45);
}

TEST_F(ConfigTest, FileLoadingIgnoresCommentsAndWhitespace) {
    std::ofstream out(test_file);
    out << "# This is a comment\n";
    out << "PORT=5000\n";
    out << "\n";                       // Empty line
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
    createConfigFile({{"PORT", "3000"}, {"LOG_LEVEL", "WARN"}});

    setEnv("PORT", "4000");  // Env should override file

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.port, 4000);         // From environment
    EXPECT_EQ(config.log_level, "WARN");  // From file (no env override)
}

TEST_F(ConfigTest, FileOverridesDefaults) {
    createConfigFile({{"D_MODEL", "256"}});

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
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
        return e.find("port") != std::string::npos;
    }));

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
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
        return e.find("log_level") != std::string::npos;
    }));
}

TEST_F(ConfigTest, ValidationValidDModel) {
    ServiceConfig config;
    std::vector<std::string> errors;

    config.d_model = 64;   // Minimum
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
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
        return e.find("d_model") != std::string::npos;
    }));

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
    EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
        return e.find("divisible") != std::string::npos;
    }));
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

    config.port = 0;              // Invalid
    config.d_model = 10000;       // Invalid
    config.temperature = 3.0f;    // Invalid
    config.strategy = "invalid";  // Invalid

    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_GE(errors.size(), 4);  // At least 4 errors
}

// ============================================================================
// Hot-Reload Tests
// ============================================================================

TEST_F(ConfigTest, ReloadValidConfiguration) {
    createConfigFile({{"PORT", "8080"}, {"LOG_LEVEL", "INFO"}});

    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;

    // Modify file
    createConfigFile({{"PORT", "9090"}, {"LOG_LEVEL", "DEBUG"}});

    // Reload
    bool success = ConfigLoader::reload(config, test_file.string(), config_mutex);

    EXPECT_TRUE(success);
    EXPECT_EQ(config.port, 9090);
    EXPECT_EQ(config.log_level, "DEBUG");
}

TEST_F(ConfigTest, ReloadInvalidConfiguration) {
    createConfigFile({{"PORT", "8080"}, {"LOG_LEVEL", "INFO"}});

    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;

    // Modify file with invalid config
    createConfigFile({{"PORT", "99999"},  // Invalid
                      {"LOG_LEVEL", "DEBUG"}});

    // Reload should fail
    bool success = ConfigLoader::reload(config, test_file.string(), config_mutex);

    EXPECT_FALSE(success);
    EXPECT_EQ(config.port, 8080);  // Should keep old value
}

// TD-285: reload() applied the new config only when one of ~25 hand-picked fields changed, so a
// reload changing only other keys reported success and kept the old values.
namespace {
struct LogCapture {
    std::ostringstream out;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(out);
    LogCapture() { adai::Logger::get()->sinks().push_back(sink); }
    ~LogCapture() {
        auto& sinks = adai::Logger::get()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
    }
};
}  // namespace

TEST_F(ConfigTest, ReloadAppliesAndLogsAChangeToAnyKey) {
    createConfigFile({{"METRICS_SERVER_URL", "http://a:8081"}, {"AUTO_SAVE_EVERY_MINUTES", "10"}});
    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;
    createConfigFile({{"METRICS_SERVER_URL", "http://b:8081"}, {"AUTO_SAVE_EVERY_MINUTES", "25"}});

    LogCapture log;
    EXPECT_TRUE(ConfigLoader::reload(config, test_file.string(), config_mutex));
    EXPECT_EQ(config.metrics_server_url, "http://b:8081");
    EXPECT_EQ(config.auto_save_every_minutes, 25);
    EXPECT_NE(log.out.str().find("METRICS_SERVER_URL: 'http://a:8081' -> 'http://b:8081'"), std::string::npos)
        << log.out.str();
    EXPECT_NE(log.out.str().find("AUTO_SAVE_EVERY_MINUTES: '10' -> '25'"), std::string::npos);
}

TEST_F(ConfigTest, ReloadReportsAddedRemovedAndHidesSecrets) {
    createConfigFile({{"PORT", "8080"}, {"FTP_DATA_SERVER_SECRET", "old-secret"}, {"RAG_NUM_DOCS", "3"}});
    const ServiceConfig before = ConfigLoader::load(test_file.string());
    createConfigFile({{"PORT", "8080"}, {"FTP_DATA_SERVER_SECRET", "new-secret"}, {"TOP_P", "0.8"}});
    const ServiceConfig after = ConfigLoader::load(test_file.string());

    const auto changes = ConfigLoader::detect_changes(before, after);
    std::string all;
    for (const auto& c : changes) all += c + "\n";
    EXPECT_NE(all.find("FTP_DATA_SERVER_SECRET: <hidden> -> <hidden>"), std::string::npos) << all;
    EXPECT_EQ(all.find("secret'"), std::string::npos) << "a secret value was logged: " << all;
    EXPECT_NE(all.find("TOP_P: added ('0.8')"), std::string::npos) << all;
    EXPECT_NE(all.find("RAG_NUM_DOCS: removed (was '3')"), std::string::npos) << all;
    EXPECT_EQ(all.find("PORT"), std::string::npos) << "an unchanged key was reported";
}

// TD-571: METRICS_DB_URL can carry a database password; to_string() hid it, the reload diff didn't.
TEST_F(ConfigTest, ReloadHidesTheMetricsDatabaseUrl) {
    createConfigFile({{"METRICS_DB_URL", "postgresql://u:old-pass@h/db"}});
    const ServiceConfig before = ConfigLoader::load(test_file.string());
    createConfigFile({{"METRICS_DB_URL", "postgresql://u:new-pass@h/db"}});
    const ServiceConfig after = ConfigLoader::load(test_file.string());
    std::string all;
    for (const auto& c : ConfigLoader::detect_changes(before, after))
        all += c + "\n";
    EXPECT_NE(all.find("METRICS_DB_URL: <hidden> -> <hidden>"), std::string::npos) << all;
    EXPECT_EQ(all.find("pass@"), std::string::npos) << "the password was logged: " << all;
    EXPECT_EQ(ConfigLoader::to_string(after).find("pass@"), std::string::npos);
}

TEST_F(ConfigTest, ReloadNamesChangesTheBinaryAppliesOnlyAfterARestart) {
    createConfigFile({{"PORT", "8080"}, {"TEMPERATURE", "0.7"}});
    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;
    createConfigFile({{"PORT", "9090"}, {"TEMPERATURE", "0.9"}});

    const std::set<std::string> live = {"TEMPERATURE"};
    LogCapture log;
    EXPECT_TRUE(ConfigLoader::reload(config, test_file.string(), config_mutex, &live));
    EXPECT_NE(log.out.str().find("PORT changed but takes effect only after a restart"), std::string::npos)
        << log.out.str();
    EXPECT_EQ(log.out.str().find("TEMPERATURE changed but"), std::string::npos);
}

TEST_F(ConfigTest, ReloadThreadSafety) {
    createConfigFile({{"PORT", "8080"}});

    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;

    std::atomic<int> reload_count{0};

    // Launch multiple threads trying to reload concurrently
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&, i]() {
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
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), [](const std::string& c) {
        return c.find("port") != std::string::npos && c.find("8080") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), [](const std::string& c) {
        return c.find("log_level") != std::string::npos;
    }));
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
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), [](const std::string& c) {
        return c.find("log_file_path") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(changes.begin(), changes.end(), [](const std::string& c) {
        return c.find("log_max_size_mb") != std::string::npos;
    }));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ConfigTest, InvalidIntegerFormat) {
    createConfigFile({{"PORT", "not_a_number"}});

    // Should fall back to default
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.port, 8080);  // Default
}

TEST_F(ConfigTest, InvalidFloatFormat) {
    createConfigFile({{"TEMPERATURE", "invalid"}});

    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);  // Default
}

TEST_F(ConfigTest, InvalidBooleanFormat) {
    createConfigFile({{"LOG_COMPRESS", "maybe"}});

    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FALSE(config.log_compress);  // Default
}

TEST_F(ConfigTest, EmptyValues) {
    createConfigFile({{"PORT", ""}, {"LOG_LEVEL", ""}});

    auto config = ConfigLoader::load(test_file.string());

    // Both are refused at load (an empty number, and an empty LOG_LEVEL since TD-289), so the
    // defaults stay and the config is valid.
    const ServiceConfig defaults;
    EXPECT_EQ(config.port, defaults.port);
    EXPECT_EQ(config.log_level, defaults.log_level);
    std::vector<std::string> errors;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ExtremelyLargeValues) {
    createConfigFile({{"D_MODEL", "999999"}, {"MAX_SEQ_LENGTH", "999999"}});

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
// GPU Strategy Config Tests (TD-030)
// ============================================================================

TEST_F(ConfigTest, GpuStrategyFileBackground) {
    createConfigFile({{"GPU_STRATEGY", "background"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.gpu_strategy, adai::GPUStrategy::BACKGROUND);
}

TEST_F(ConfigTest, GpuStrategyFileFull) {
    createConfigFile({{"GPU_STRATEGY", "full"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.gpu_strategy, adai::GPUStrategy::FULL);
}

TEST_F(ConfigTest, GpuStrategyUnknownDefaultsToBackground) {
    createConfigFile({{"GPU_STRATEGY", "turbo"}});
    testing::internal::CaptureStderr();
    auto config = ConfigLoader::load(test_file.string());
    std::string err = testing::internal::GetCapturedStderr();
    EXPECT_EQ(config.gpu_strategy, adai::GPUStrategy::BACKGROUND);
    EXPECT_NE(err.find("turbo"), std::string::npos);
}

TEST_F(ConfigTest, GpuStrategyEnvVarOverridesFile) {
    createConfigFile({{"GPU_STRATEGY", "background"}});
    setEnv("GPU_STRATEGY", "full");
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.gpu_strategy, adai::GPUStrategy::FULL);
}

// ============================================================================
// FTP Dataset Transport Config Tests (Phase 10)
// ============================================================================

TEST_F(ConfigTest, FtpDefaultValues) {
    auto config = ConfigLoader::load();

    EXPECT_EQ(config.ftp_server_port, 2121);
    EXPECT_EQ(config.ftp_pasv_port_min, 50000);
    EXPECT_EQ(config.ftp_pasv_port_max, 50099);
    EXPECT_EQ(config.ftp_token_ttl_minutes, 30);
    EXPECT_EQ(config.ftp_data_server_secret, "");  // TD-284: empty = a random password per token
    EXPECT_FALSE(config.ftp_server_enabled);
    EXPECT_TRUE(config.ftp_advertise_ip.empty());
    EXPECT_TRUE(config.download_dir.empty());
    EXPECT_EQ(config.max_parallel_downloads, 4);
    EXPECT_EQ(config.large_file_warn_threshold_mb, 500);
}

TEST_F(ConfigTest, LoadFtpFieldsFromFile) {
    createConfigFile({
        {"FTP_SERVER_PORT", "3121"},
        {"FTP_PASV_PORT_MIN", "60000"},
        {"FTP_PASV_PORT_MAX", "60099"},
        {"FTP_TOKEN_TTL_MINUTES", "60"},
        {"FTP_DATA_SERVER_SECRET", "s3cr3t-key"},
        {"DOWNLOAD_DIR", "/mnt/datasets"},
        {"MAX_PARALLEL_DOWNLOADS", "8"},
        {"LARGE_FILE_WARN_THRESHOLD_MB", "1024"},
    });

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.ftp_server_port, 3121);
    EXPECT_EQ(config.ftp_pasv_port_min, 60000);
    EXPECT_EQ(config.ftp_pasv_port_max, 60099);
    EXPECT_EQ(config.ftp_token_ttl_minutes, 60);
    EXPECT_EQ(config.ftp_data_server_secret, "s3cr3t-key");
    EXPECT_EQ(config.download_dir, "/mnt/datasets");
    EXPECT_EQ(config.max_parallel_downloads, 8);
    EXPECT_EQ(config.large_file_warn_threshold_mb, 1024);
}

TEST_F(ConfigTest, LoadFtpFieldsFromEnvVars) {
    setEnv("FTP_SERVER_PORT", "4121");
    setEnv("FTP_PASV_PORT_MIN", "55000");
    setEnv("FTP_PASV_PORT_MAX", "55049");
    setEnv("FTP_TOKEN_TTL_MINUTES", "15");
    setEnv("FTP_DATA_SERVER_SECRET", "env-secret");
    setEnv("DOWNLOAD_DIR", "/tmp/adai_downloads");
    setEnv("MAX_PARALLEL_DOWNLOADS", "2");
    setEnv("LARGE_FILE_WARN_THRESHOLD_MB", "250");

    auto config = ConfigLoader::load();

    EXPECT_EQ(config.ftp_server_port, 4121);
    EXPECT_EQ(config.ftp_pasv_port_min, 55000);
    EXPECT_EQ(config.ftp_pasv_port_max, 55049);
    EXPECT_EQ(config.ftp_token_ttl_minutes, 15);
    EXPECT_EQ(config.ftp_data_server_secret, "env-secret");
    EXPECT_EQ(config.download_dir, "/tmp/adai_downloads");
    EXPECT_EQ(config.max_parallel_downloads, 2);
    EXPECT_EQ(config.large_file_warn_threshold_mb, 250);
}

TEST_F(ConfigTest, FtpEnvVarOverridesFile) {
    createConfigFile({
        {"FTP_SERVER_PORT", "3121"},
        {"DOWNLOAD_DIR", "/mnt/datasets"},
    });
    setEnv("FTP_SERVER_PORT", "5121");  // env overrides file

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.ftp_server_port, 5121);          // from env
    EXPECT_EQ(config.download_dir, "/mnt/datasets");  // from file
}

// TD-339: trainer-side FTPS server authentication keys, from the file and the environment.
TEST_F(ConfigTest, FtpsVerificationKeysLoad) {
    auto defaults = ConfigLoader::load();
    EXPECT_TRUE(defaults.ftps_ca_file.empty());
    EXPECT_TRUE(defaults.ftps_pinned_public_key.empty());

    createConfigFile({
        {"FTPS_CA_FILE", "/etc/adai/registry-ca.pem"},
        {"FTPS_PINNED_PUBLIC_KEY", "sha256//file-pin="},
    });
    setEnv("FTPS_PINNED_PUBLIC_KEY", "sha256//env-pin=");  // env overrides file
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.ftps_ca_file, "/etc/adai/registry-ca.pem");
    EXPECT_EQ(config.ftps_pinned_public_key, "sha256//env-pin=");

}

// ============================================================================
// World-Model (LeJEPA) Pretraining Config Tests (TD-183)
// ============================================================================

TEST_F(ConfigTest, WorldModelDefaultValues) {
    auto config = ConfigLoader::load();

    EXPECT_FALSE(config.world_model_enabled);
    EXPECT_EQ(config.world_model_d_model, 512u);
    EXPECT_EQ(config.world_model_num_layers, 6u);
    EXPECT_EQ(config.world_model_num_heads, 8u);
    EXPECT_EQ(config.world_model_d_ff, 2048u);
    EXPECT_FLOAT_EQ(config.world_model_sigreg_lambda, 1.0f);
    EXPECT_EQ(config.world_model_sigreg_num_sketches, 64u);
    EXPECT_EQ(config.world_model_sigreg_batch_size, 32u);
    // TD-186: 0 (never inject), not TD-183's original 1 — this field now has real effect
    // (IncrementalTrainer::build_model() actually constructs gated layers from it), so the
    // config-level default reverted to matching every other gate's own "off by default"
    // guarantee. See this field's own doc comment in Config.hpp for the full reasoning.
    EXPECT_EQ(config.world_model_inject_every_n_layers, 0u);
}

TEST_F(ConfigTest, HippocampalMemoryDefaultValues) {
    auto config = ConfigLoader::load();

    EXPECT_FALSE(config.hippocampal_memory_enabled);
    EXPECT_EQ(config.hippocampal_memory_capacity, 512u);
    EXPECT_FLOAT_EQ(config.hippocampal_repetition_alpha, 0.0f);
    EXPECT_FLOAT_EQ(config.hippocampal_repetition_decay, 0.95f);
    EXPECT_EQ(config.hippocampal_state_path, "");  // TD-250: derived from MODEL_PATH
    EXPECT_EQ(config.hippocampal_save_interval_seconds, 300);
}

// TD-250: the hippocampal memory's own path and save interval.
TEST_F(ConfigTest, HippocampalStatePathAndSaveIntervalLoadFromFile) {
    createConfigFile({{"HIPPOCAMPAL_STATE_PATH", "/var/lib/adai/memory.bin"},
                      {"HIPPOCAMPAL_SAVE_INTERVAL_SECONDS", "60"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.hippocampal_state_path, "/var/lib/adai/memory.bin");
    EXPECT_EQ(config.hippocampal_save_interval_seconds, 60);
    std::vector<std::string> errors;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
    config.hippocampal_save_interval_seconds = 0;  // only at shutdown
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, LoadWorldModelFieldsFromFile) {
    createConfigFile({
        {"WORLD_MODEL_ENABLED", "true"},
        {"WORLD_MODEL_D_MODEL", "128"},
        {"WORLD_MODEL_NUM_LAYERS", "4"},
        {"WORLD_MODEL_NUM_HEADS", "2"},
        {"WORLD_MODEL_D_FF", "256"},
        {"WORLD_MODEL_SIGREG_LAMBDA", "2.5"},
        {"WORLD_MODEL_SIGREG_NUM_SKETCHES", "32"},
        {"WORLD_MODEL_SIGREG_BATCH_SIZE", "48"},
        {"WORLD_MODEL_INJECT_EVERY_N_LAYERS", "3"},
    });

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_TRUE(config.world_model_enabled);
    EXPECT_EQ(config.world_model_d_model, 128u);
    EXPECT_EQ(config.world_model_num_layers, 4u);
    EXPECT_EQ(config.world_model_num_heads, 2u);
    EXPECT_EQ(config.world_model_d_ff, 256u);
    EXPECT_FLOAT_EQ(config.world_model_sigreg_lambda, 2.5f);
    EXPECT_EQ(config.world_model_sigreg_num_sketches, 32u);
    EXPECT_EQ(config.world_model_sigreg_batch_size, 48u);
    EXPECT_EQ(config.world_model_inject_every_n_layers, 3u);
}

TEST_F(ConfigTest, LoadWorldModelFieldsFromEnvVars) {
    setEnv("WORLD_MODEL_ENABLED", "1");
    setEnv("WORLD_MODEL_D_MODEL", "256");
    setEnv("WORLD_MODEL_NUM_LAYERS", "3");
    setEnv("WORLD_MODEL_NUM_HEADS", "4");
    setEnv("WORLD_MODEL_D_FF", "512");
    setEnv("WORLD_MODEL_SIGREG_LAMBDA", "0.5");
    setEnv("WORLD_MODEL_SIGREG_NUM_SKETCHES", "16");
    setEnv("WORLD_MODEL_SIGREG_BATCH_SIZE", "8");
    setEnv("WORLD_MODEL_INJECT_EVERY_N_LAYERS", "2");

    auto config = ConfigLoader::load();

    EXPECT_TRUE(config.world_model_enabled);
    EXPECT_EQ(config.world_model_d_model, 256u);
    EXPECT_EQ(config.world_model_num_layers, 3u);
    EXPECT_EQ(config.world_model_num_heads, 4u);
    EXPECT_EQ(config.world_model_d_ff, 512u);
    EXPECT_FLOAT_EQ(config.world_model_sigreg_lambda, 0.5f);
    EXPECT_EQ(config.world_model_sigreg_num_sketches, 16u);
    EXPECT_EQ(config.world_model_sigreg_batch_size, 8u);
    EXPECT_EQ(config.world_model_inject_every_n_layers, 2u);
}

TEST_F(ConfigTest, WorldModelEnvVarOverridesFile) {
    createConfigFile({
        {"WORLD_MODEL_ENABLED", "true"},
        {"WORLD_MODEL_D_MODEL", "128"},
    });
    setEnv("WORLD_MODEL_D_MODEL", "64");  // env overrides file

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_TRUE(config.world_model_enabled);      // from file
    EXPECT_EQ(config.world_model_d_model, 64u);   // from env
}

TEST_F(ConfigTest, WorldModelDisabledDoesNotAffectChatbotArchitecture) {
    // TD-183 Action Item 3: enabling/configuring the world model must have zero effect on the
    // chatbot's own D_MODEL/etc. fallback block.
    createConfigFile({
        {"D_MODEL", "512"},
        {"WORLD_MODEL_ENABLED", "true"},
        {"WORLD_MODEL_D_MODEL", "128"},
    });

    auto config = ConfigLoader::load(test_file.string());

    EXPECT_EQ(config.d_model, 512u);
    EXPECT_EQ(config.world_model_d_model, 128u);
}

TEST_F(ConfigTest, ValidationWorldModelArchitecture) {
    createConfigFile({
        {"WORLD_MODEL_D_MODEL", "100"},   // not divisible by num_heads below
        {"WORLD_MODEL_NUM_HEADS", "7"},
        {"WORLD_MODEL_D_FF", "10"},       // below minimum
        {"WORLD_MODEL_NUM_LAYERS", "0"},  // below minimum
        {"WORLD_MODEL_SIGREG_LAMBDA", "-1.0"},
        {"WORLD_MODEL_SIGREG_NUM_SKETCHES", "0"},
    });

    auto config = ConfigLoader::load(test_file.string());
    std::vector<std::string> errors;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    EXPECT_GE(errors.size(), 5u);
}

// SIGReg runs across a batch's pooled embeddings, so a batch must hold at least 2.
TEST_F(ConfigTest, ValidationRejectsASigregBatchBelowTwo) {
    createConfigFile({{"WORLD_MODEL_SIGREG_BATCH_SIZE", "1"}});
    auto config = ConfigLoader::load(test_file.string());
    std::vector<std::string> errors;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
    bool mentioned = false;
    for (const auto& e : errors)
        mentioned = mentioned || e.find("world_model_sigreg_batch_size") != std::string::npos;
    EXPECT_TRUE(mentioned);
}

TEST_F(ConfigTest, ValidationWorldModelDefaultsPassValidation) {
    // The defaults (feature disabled) must already be internally consistent, since validation
    // runs unconditionally rather than only when world_model_enabled is true.
    auto config = ConfigLoader::load();
    std::vector<std::string> errors;
    EXPECT_TRUE(ConfigLoader::validate(config, errors));
}

// ============================================================================
// Tokenizer Mode Config Tests
// ============================================================================

// TD-564: Unicode (one unit per UTF-8 code point) is the default; it was ASCII bytes.
TEST_F(ConfigTest, TokenizerModeDefaultsToUnicode) {
    auto config = ConfigLoader::load();
    EXPECT_TRUE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeUnicodeFromFile) {
    createConfigFile({{"TOKENIZER_MODE", "unicode"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_TRUE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeAsciiFromFile) {
    createConfigFile({{"TOKENIZER_MODE", "ascii"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FALSE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeUnicodeCaseInsensitiveFromFile) {
    createConfigFile({{"TOKENIZER_MODE", "UNICODE"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_TRUE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeMixedCaseFromFile) {
    createConfigFile({{"TOKENIZER_MODE", "Unicode"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_TRUE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeUnknownValueKeepsTheDefault) {
    createConfigFile({{"TOKENIZER_MODE", "utf8"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_TRUE(config.unicode_tokenizer);  // TD-564: the default, unicode
}

TEST_F(ConfigTest, TokenizerModeFromEnvVar) {
    setEnv("TOKENIZER_MODE", "unicode");
    auto config = ConfigLoader::load();
    EXPECT_TRUE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeAsciiFromEnvVar) {
    setEnv("TOKENIZER_MODE", "ascii");
    auto config = ConfigLoader::load();
    EXPECT_FALSE(config.unicode_tokenizer);
}

TEST_F(ConfigTest, TokenizerModeEnvVarOverridesFile) {
    createConfigFile({{"TOKENIZER_MODE", "ascii"}});
    setEnv("TOKENIZER_MODE", "unicode");
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_TRUE(config.unicode_tokenizer);  // env wins
}

TEST_F(ConfigTest, TokenizerModeFileOverridesDefault) {
    createConfigFile({{"TOKENIZER_MODE", "ascii"}});
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FALSE(config.unicode_tokenizer);  // file overrides the default, unicode (TD-564)
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// TD-286: one boolean syntax for every key, file and environment
// ============================================================================

namespace {
// Every boolean ServiceConfig key, with the field it sets.
const std::vector<std::pair<std::string, bool ServiceConfig::*>> kBoolKeys = {
    {"LOG_COMPRESS", &ServiceConfig::log_compress},
    {"GRADIENT_CLIP_ADAPTIVE", &ServiceConfig::adaptive_gradient_clip},
    {"ENABLE_METRICS_SERVICE", &ServiceConfig::enable_metrics_service},
    {"METRICS_ENABLE_PERSISTENCE", &ServiceConfig::metrics_enable_persistence},
    {"METRICS_ENABLE_PROMETHEUS", &ServiceConfig::metrics_enable_prometheus},
    {"METRICS_API_ALLOW_CONTROL", &ServiceConfig::metrics_api_allow_control},
    {"ENABLE_GENERATION_QUALITY_METRICS", &ServiceConfig::enable_generation_quality_metrics},
    {"ALLOW_UNTRAINED_MODEL", &ServiceConfig::allow_untrained_model},
    {"RAG_ENABLED", &ServiceConfig::rag_enabled},
    {"GPU_ENABLED", &ServiceConfig::gpu_enabled},
    {"FTPS_ENABLED", &ServiceConfig::ftps_enabled},
    {"FTP_SERVER_ENABLED", &ServiceConfig::ftp_server_enabled},
    {"MNS_ADMIN_ENABLED", &ServiceConfig::name_service_admin_enabled},             // TD-456
    {"REGISTRY_ADMIN_ENABLED", &ServiceConfig::registry_admin_enabled},            // TD-489
    {"LOSS_QUALITY_BACKFILL", &ServiceConfig::loss_quality_backfill},              // TD-273
    {"GENERATION_QUALITY_BACKFILL", &ServiceConfig::generation_quality_backfill},  // TD-273
};
}  // namespace

TEST_F(ConfigTest, EveryBooleanKeyAcceptsAnyCaseOfTheSameWords) {
    // Seven keys used to read "True", "ON" and "Yes" as false.
    for (const auto& [key, field] : kBoolKeys) {
        for (const char* yes : {"True", "ON", "Yes", "1", "Enabled"}) {
            createConfigFile({{key, yes}});
            EXPECT_TRUE(ConfigLoader::load(test_file.string()).*field) << key << "=" << yes;
        }
        for (const char* no : {"False", "OFF", "No", "0", "disabled"}) {
            createConfigFile({{key, no}});
            EXPECT_FALSE(ConfigLoader::load(test_file.string()).*field) << key << "=" << no;
        }
    }
}

TEST_F(ConfigTest, ABooleanThatIsntOneLeavesTheDefault) {
    // A non-boolean used to be read as false; now it's refused with a warning and the default stays.
    createConfigFile({{"METRICS_API_ALLOW_CONTROL", "maybe"}, {"GPU_ENABLED", "perhaps"}});
    const ServiceConfig defaults;
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.metrics_api_allow_control, defaults.metrics_api_allow_control);
    EXPECT_EQ(config.gpu_enabled, defaults.gpu_enabled);
}

// TD-273: the loss score is on and the per-pair generation one off unless configured; both can come
// from the environment.
TEST_F(ConfigTest, QualityBackfillDefaultsAndEnvironment) {
    const ServiceConfig defaults;
    EXPECT_TRUE(defaults.loss_quality_backfill);
    EXPECT_FALSE(defaults.generation_quality_backfill);
    ::setenv("LOSS_QUALITY_BACKFILL", "off", 1);
    ::setenv("GENERATION_QUALITY_BACKFILL", "Yes", 1);
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    ::unsetenv("LOSS_QUALITY_BACKFILL");
    ::unsetenv("GENERATION_QUALITY_BACKFILL");
    EXPECT_FALSE(config.loss_quality_backfill);
    EXPECT_TRUE(config.generation_quality_backfill);
}

TEST_F(ConfigTest, FtpsEnabledFromTheEnvironmentAcceptsAnyCase) {
    ::setenv("FTPS_ENABLED", "TRUE", 1);
    const bool on = ConfigLoader::load(test_file.string()).ftps_enabled;
    ::unsetenv("FTPS_ENABLED");
    EXPECT_TRUE(on);  // the env path used to bypass get_env_bool() and accept only lowercase
}

// ============================================================================
// TD-287: a number must be the whole value, and a size can't be negative
// ============================================================================

TEST_F(ConfigTest, ANumberWithTrailingCharactersLeavesTheDefault) {
    // stoi/stof stopped at the first bad character: PORT=8080x was read as 8080.
    createConfigFile({{"PORT", "8080x"},
                      {"D_MODEL", "256abc"},
                      {"TEMPERATURE", "0.5f"},
                      {"WORLD_MODEL_D_MODEL", "128.5"}});
    const ServiceConfig defaults;
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.port, defaults.port);
    EXPECT_EQ(config.d_model, defaults.d_model);
    EXPECT_FLOAT_EQ(config.temperature, defaults.temperature);
    EXPECT_EQ(config.world_model_d_model, defaults.world_model_d_model);
}

TEST_F(ConfigTest, ANegativeSizeLeavesTheDefault) {
    // stoull wrapped a negative: D_MODEL=-1 was read as 18446744073709551615.
    createConfigFile({{"D_MODEL", "-1"},
                      {"MAX_SEQ_LENGTH", "-512"},
                      {"LOG_MAX_FILES", "-3"},
                      {"HIPPOCAMPAL_MEMORY_CAPACITY", "-8"}});
    const ServiceConfig defaults;
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.d_model, defaults.d_model);
    EXPECT_EQ(config.max_seq_length, defaults.max_seq_length);
    EXPECT_EQ(config.log_max_files, defaults.log_max_files);
    EXPECT_EQ(config.hippocampal_memory_capacity, defaults.hippocampal_memory_capacity);
}

TEST_F(ConfigTest, ANonFiniteFloatLeavesTheDefault) {
    createConfigFile({{"LEARNING_RATE", "nan"}, {"TEMPERATURE", "inf"}});
    const ServiceConfig defaults;
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_FLOAT_EQ(config.learning_rate, defaults.learning_rate);
    EXPECT_FLOAT_EQ(config.temperature, defaults.temperature);
}

TEST_F(ConfigTest, WellFormedNumbersStillLoad) {
    createConfigFile({{"PORT", "9090"},
                      {"D_MODEL", "256"},
                      {"TEMPERATURE", "0.25"},
                      {"LEARNING_RATE", "1e-4"},
                      {"GPU_DEVICE_ID", "-1"}});  // a negative int is still a valid int
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.port, 9090);
    EXPECT_EQ(config.d_model, 256u);
    EXPECT_FLOAT_EQ(config.temperature, 0.25f);
    EXPECT_FLOAT_EQ(config.learning_rate, 1e-4f);
    EXPECT_EQ(config.gpu_device_id, -1);
}

TEST_F(ConfigTest, EnvironmentNumbersAreCheckedTheSameWay) {
    ::setenv("PORT", "8080x", 1);
    ::setenv("D_MODEL", "-1", 1);
    ::setenv("TEMPERATURE", "0.5abc", 1);
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    ::unsetenv("PORT");
    ::unsetenv("D_MODEL");
    ::unsetenv("TEMPERATURE");
    const ServiceConfig defaults;
    EXPECT_EQ(config.port, defaults.port);
    EXPECT_EQ(config.d_model, defaults.d_model);
    EXPECT_FLOAT_EQ(config.temperature, defaults.temperature);
}

// ============================================================================
// TD-288: validate() covers every field that divides, allocates, binds or is a probability
// ============================================================================

namespace {
// validate() on the defaults with one change; returns the errors.
std::vector<std::string> errors_after(const std::function<void(ServiceConfig&)>& change) {
    ServiceConfig config;
    change(config);
    std::vector<std::string> errors;
    ConfigLoader::validate(config, errors);
    return errors;
}

bool mentions(const std::vector<std::string>& errors, const std::string& field) {
    for (const auto& e : errors) {
        if (e.find(field) != std::string::npos) return true;
    }
    return false;
}
}  // namespace

TEST_F(ConfigTest, TheDefaultsPassEveryRule) {
    const auto errors = errors_after([](ServiceConfig&) {});
    EXPECT_TRUE(errors.empty()) << errors.front();
}

TEST_F(ConfigTest, ValidateRejectsEachBadValue) {
    // The three the TD found by running the old validate(), then one per new rule.
    const std::vector<std::pair<std::string, std::function<void(ServiceConfig&)>>> cases = {
        {"gradient_accumulation_steps", [](ServiceConfig& c) { c.gradient_accumulation_steps = 0; }},
        {"learning_rate", [](ServiceConfig& c) { c.learning_rate = -1.0f; }},
        {"gpu_memory_fraction", [](ServiceConfig& c) { c.gpu_memory_fraction = 7.5f; }},
        {"gpu_memory_fraction", [](ServiceConfig& c) { c.gpu_memory_fraction = 0.0f; }},
        {"learning_rate", [](ServiceConfig& c) { c.learning_rate = 0.0f; }},
        {"num_epochs", [](ServiceConfig& c) { c.num_epochs = 0; }},
        {"batch_size", [](ServiceConfig& c) { c.batch_size = 0; }},
        {"weight_decay", [](ServiceConfig& c) { c.weight_decay = -0.1f; }},
        {"gradient_clip", [](ServiceConfig& c) { c.gradient_clip = -1.0f; }},
        {"gradient_clip_min", [](ServiceConfig& c) { c.gradient_clip_min = 0.0f; }},
        {"gradient_clip_max", [](ServiceConfig& c) { c.gradient_clip_max = 0.05f; }},
        {"gradient_clip_ema_decay", [](ServiceConfig& c) { c.gradient_clip_ema_decay = 0.0f; }},
        {"gradient_clip_ema_decay", [](ServiceConfig& c) { c.gradient_clip_ema_decay = 1.5f; }},
        {"gradient_clip_headroom", [](ServiceConfig& c) { c.gradient_clip_headroom = 0.0f; }},
        {"gradient_clip_warmup_steps", [](ServiceConfig& c) { c.gradient_clip_warmup_steps = -1; }},
        {"gradient_clip_spike_k", [](ServiceConfig& c) { c.gradient_clip_spike_k = 0.0f; }},
        {"vocab_build_size", [](ServiceConfig& c) { c.vocab_build_size = -1; }},
        {"speculative_num_candidates", [](ServiceConfig& c) { c.speculative_num_candidates = 0; }},
        {"hippocampal_association_decay", [](ServiceConfig& c) { c.hippocampal_association_decay = 0.0f; }},
        {"hippocampal_association_decay", [](ServiceConfig& c) { c.hippocampal_association_decay = 1.1f; }},
        {"hippocampal_cross_reference_alpha", [](ServiceConfig& c) { c.hippocampal_cross_reference_alpha = -1.0f; }},
        {"hippocampal_save_interval_seconds", [](ServiceConfig& c) { c.hippocampal_save_interval_seconds = -1; }},
        {"gpu_device_id", [](ServiceConfig& c) { c.gpu_device_id = -1; }},
        {"metrics_api_port", [](ServiceConfig& c) { c.metrics_api_port = 0; }},
        {"registry_listen_port", [](ServiceConfig& c) { c.registry_listen_port = 70000; }},
        {"ftp_server_port", [](ServiceConfig& c) { c.ftp_server_port = -1; }},
        {"ftp_pasv_port_min", [](ServiceConfig& c) { c.ftp_pasv_port_min = 0; }},
        {"ftp_pasv_port_max", [](ServiceConfig& c) { c.ftp_pasv_port_max = 65536; }},
        {"ftp_pasv_port_max", [](ServiceConfig& c) { c.ftp_pasv_port_min = 50100; c.ftp_pasv_port_max = 50000; }},
        {"name_service_port", [](ServiceConfig& c) { c.name_service_port = 0; }},
        {"trainer_admin_port", [](ServiceConfig& c) { c.trainer_admin_port = 0; }},
        {"trainer_child_admin_port", [](ServiceConfig& c) { c.trainer_child_admin_port = 0; }},
        {"trainer_child_admin_port", [](ServiceConfig& c) { c.trainer_child_admin_port = c.trainer_admin_port; }},
        {"metrics_push_timeout_ms", [](ServiceConfig& c) { c.metrics_push_timeout_ms = 0; }},
        {"metrics_heartbeat_interval_ms", [](ServiceConfig& c) { c.metrics_heartbeat_interval_ms = 0; }},
        {"metrics_persist_every_samples", [](ServiceConfig& c) { c.metrics_persist_every_samples = -1; }},
        {"metrics_persist_every_seconds", [](ServiceConfig& c) { c.metrics_persist_every_seconds = -1; }},
        {"metrics_max_records_in_memory", [](ServiceConfig& c) { c.metrics_max_records_in_memory = 0; }},
        {"metrics_max_records_on_disk", [](ServiceConfig& c) { c.metrics_max_records_on_disk = -1; }},
        {"metrics_retention_days", [](ServiceConfig& c) { c.metrics_retention_days = -1; }},
        {"metrics_file_max_mb", [](ServiceConfig& c) { c.metrics_file_max_mb = -1; }},
        {"metrics_file_backups", [](ServiceConfig& c) { c.metrics_file_backups = -1; }},
        {"metrics_max_live_sessions", [](ServiceConfig& c) { c.metrics_max_live_sessions = 0; }},
        {"metrics_completed_ttl_seconds", [](ServiceConfig& c) { c.metrics_completed_ttl_seconds = -1; }},
        {"metrics_staleness_threshold_seconds", [](ServiceConfig& c) { c.metrics_staleness_threshold_seconds = 0; }},
        {"metrics_db_pool_size", [](ServiceConfig& c) { c.metrics_db_pool_size = 0; }},
        {"generation_quality_sample_size", [](ServiceConfig& c) { c.generation_quality_sample_size = -1; }},
        {"generation_quality_max_tokens", [](ServiceConfig& c) { c.generation_quality_max_tokens = 0; }},
        {"generation_quality_async_threshold", [](ServiceConfig& c) { c.generation_quality_async_threshold = -1; }},
        {"registry_timeout_ms", [](ServiceConfig& c) { c.registry_timeout_ms = 0; }},
        {"registry_claim_lease_minutes", [](ServiceConfig& c) { c.registry_claim_lease_minutes = -1; }},
        {"registry_max_upload_mb", [](ServiceConfig& c) { c.registry_max_upload_mb = 0; }},
        {"registry_max_concurrent_fetches", [](ServiceConfig& c) { c.registry_max_concurrent_fetches = 0; }},
        {"ftp_token_ttl_minutes", [](ServiceConfig& c) { c.ftp_token_ttl_minutes = 0; }},
        {"max_parallel_downloads", [](ServiceConfig& c) { c.max_parallel_downloads = 0; }},
        {"large_file_warn_threshold_mb", [](ServiceConfig& c) { c.large_file_warn_threshold_mb = -1; }},
        {"ftp_max_sessions_per_run", [](ServiceConfig& c) { c.ftp_max_sessions_per_run = -1; }},
        {"name_service_timeout_ms", [](ServiceConfig& c) { c.name_service_timeout_ms = 0; }},
        {"auto_save_every_samples", [](ServiceConfig& c) { c.auto_save_every_samples = -1; }},
        {"auto_save_every_minutes", [](ServiceConfig& c) { c.auto_save_every_minutes = -1; }},
        {"max_sessions_to_keep", [](ServiceConfig& c) { c.max_sessions_to_keep = -1; }},
    };
    for (const auto& [field, change] : cases) {
        EXPECT_TRUE(mentions(errors_after(change), field)) << field << " wasn't rejected";
    }
}

// TD-410: early stopping and the schedule are settings, with the values that used to be fixed.
TEST_F(ConfigTest, TrainingPolicyKeys) {
    const adai::ServiceConfig defaults;
    EXPECT_TRUE(defaults.early_stopping_enabled);
    EXPECT_EQ(defaults.early_stopping_patience, 5);
    EXPECT_TRUE(defaults.restore_best_weights);
    EXPECT_EQ(defaults.lr_schedule, "linear_warmup");

    {
        std::ofstream file(test_file);
        file << "EARLY_STOPPING_ENABLED=false\nEARLY_STOPPING_PATIENCE=9\n"
                "RESTORE_BEST_WEIGHTS=no\nLR_SCHEDULE=CONSTANT\n";
    }
    auto config = ConfigLoader::load(test_file.string());
    EXPECT_FALSE(config.early_stopping_enabled);
    EXPECT_EQ(config.early_stopping_patience, 9);
    EXPECT_FALSE(config.restore_best_weights);
    EXPECT_EQ(config.lr_schedule, "constant");

    {
        std::ofstream file(test_file);
        file << "LR_SCHEDULE=warmup_cosine\n";
    }
    EXPECT_EQ(ConfigLoader::load(test_file.string()).lr_schedule, "linear_warmup")
        << "a decaying schedule isn't offered (TD-269)";
}

TEST_F(ConfigTest, ValidateRejectsPatienceBelowOne) {
    adai::ServiceConfig config;
    config.early_stopping_patience = 0;
    std::vector<std::string> errors;
    EXPECT_FALSE(ConfigLoader::validate(config, errors));
}

TEST_F(ConfigTest, ValidateAcceptsTheDocumentedZeroes) {
    // 0 means "off" or "no limit" for these, so it must stay valid.
    const auto errors = errors_after([](ServiceConfig& c) {
        c.metrics_max_records_on_disk = 0;
        c.metrics_retention_days = 0;
        c.metrics_file_max_mb = 0;
        c.registry_claim_lease_minutes = 0;
        c.large_file_warn_threshold_mb = 0;
        c.ftp_max_sessions_per_run = 0;
        c.auto_save_every_samples = 0;
        c.auto_save_every_minutes = 0;
        c.max_sessions_to_keep = 0;
        c.gradient_clip = 0.0f;
        c.weight_decay = 0.0f;
        c.gradient_clip_warmup_steps = 0;
        c.generation_quality_sample_size = 0;
        c.ftp_pasv_port_min = c.ftp_pasv_port_max;  // a one-port range
        c.gpu_memory_fraction = 1.0f;
    });
    EXPECT_TRUE(errors.empty()) << errors.front();
}

TEST_F(ConfigTest, EveryShippedConfigFilePassesValidation) {
    // Every binary now refuses to start on a validate() error, so the shipped files must pass.
    const std::filesystem::path root = std::filesystem::path(__FILE__).parent_path().parent_path();
    for (const char* name : {"config.chatbot.conf", "config.trainer.conf", "config.metrics.conf",
                             "config.mns.conf", "config.registry.conf", "config.conf"}) {
        const auto path = root / name;
        ASSERT_TRUE(std::filesystem::exists(path)) << path;
        std::vector<std::string> errors;
        EXPECT_TRUE(ConfigLoader::validate(ConfigLoader::load(path.string()), errors))
            << name << ": " << (errors.empty() ? "" : errors.front());
    }
}

// ============================================================================
// TD-289: inline comments end a value; enum values are checked at load
// ============================================================================

TEST_F(ConfigTest, AnInlineCommentEndsTheValue) {
    std::ofstream(test_file) << "MODEL_PATH=/models/chat # production model\n"
                                "PORT=9090\t# the API port\n"
                                "VOCAB_PATH=\"/data/my # vocab.txt\"  # quoted: the '#' is part of it\n"
                                "FTP_DATA_SERVER_SECRET=abc#def\n"
                                "RUN_GROUP='group # one'\n";
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.model_path, "/models/chat");
    EXPECT_EQ(config.port, 9090);
    EXPECT_EQ(config.vocab_path, "/data/my # vocab.txt");
    EXPECT_EQ(config.ftp_data_server_secret, "abc#def");  // no space before '#': not a comment
    EXPECT_EQ(config.run_group, "group # one");
}

TEST_F(ConfigTest, AnUnknownEnumValueLeavesTheDefault) {
    // Each used to be stored as written; TOKENIZER_MODE=utf8 silently meant ASCII.
    createConfigFile({{"TOKENIZER_MODE", "utf8"},
                      {"STRATEGY", "nucleous"},
                      {"LOG_LEVEL", "verbose"},
                      {"METRICS_STORAGE_BACKEND", "postgress"},
                      {"DATASET_KIND", "world-model"}});
    const ServiceConfig defaults;
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_EQ(config.unicode_tokenizer, defaults.unicode_tokenizer);
    EXPECT_EQ(config.strategy, defaults.strategy);
    EXPECT_EQ(config.log_level, defaults.log_level);
    EXPECT_EQ(config.metrics_storage_backend, defaults.metrics_storage_backend);
    EXPECT_EQ(config.dataset_kind, defaults.dataset_kind);
}

TEST_F(ConfigTest, EnumValuesAreMatchedInAnyCase) {
    createConfigFile({{"TOKENIZER_MODE", "ASCII"},
                      {"STRATEGY", "Beam"},
                      {"LOG_LEVEL", "warning"},
                      {"METRICS_STORAGE_BACKEND", "SQLite"},
                      {"DATASET_KIND", "World_Model"}});
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    EXPECT_FALSE(config.unicode_tokenizer);
    EXPECT_EQ(config.strategy, "beam");
    EXPECT_EQ(config.log_level, "WARN");
    EXPECT_EQ(config.metrics_storage_backend, "sqlite");
    EXPECT_EQ(config.dataset_kind, "world_model");
    std::vector<std::string> errors;
    EXPECT_TRUE(ConfigLoader::validate(config, errors)) << errors.front();
}

TEST_F(ConfigTest, AnEmptyDatasetKindIsTheLegacyPool) {
    createConfigFile({{"DATASET_KIND", ""}});
    EXPECT_EQ(ConfigLoader::load(test_file.string()).dataset_kind, "");
}

TEST_F(ConfigTest, EnvironmentEnumValuesAreCheckedTheSameWay) {
    ::setenv("STRATEGY", "nucleous", 1);
    ::setenv("TOKENIZER_MODE", "ASCII", 1);
    ::setenv("DATASET_KIND", "Chatbot", 1);
    const ServiceConfig config = ConfigLoader::load(test_file.string());
    ::unsetenv("STRATEGY");
    ::unsetenv("TOKENIZER_MODE");
    ::unsetenv("DATASET_KIND");
    EXPECT_EQ(config.strategy, ServiceConfig{}.strategy);
    EXPECT_FALSE(config.unicode_tokenizer);
    EXPECT_EQ(config.dataset_kind, "chatbot");
}

TEST_F(ConfigTest, ATokenizerModeTypoDoesntSwitchToAscii) {
    // The file says unicode; a typo'd env value used to override it to ASCII without a word.
    createConfigFile({{"TOKENIZER_MODE", "unicode"}});
    ::setenv("TOKENIZER_MODE", "utf8", 1);
    const bool unicode = ConfigLoader::load(test_file.string()).unicode_tokenizer;
    ::unsetenv("TOKENIZER_MODE");
    EXPECT_TRUE(unicode);
}

// ============================================================================
// TD-290: print() logs every section through the logger, secrets hidden
// ============================================================================

TEST_F(ConfigTest, ToStringCoversEverySectionAndHidesSecrets) {
    ServiceConfig config;
    config.ftp_data_server_secret = "s3cret-value";
    config.metrics_server_headers = "CF-Access-Client-Secret: abc123";
    config.metrics_db_url = "postgresql://user:pw@db/metrics";
    config.registry_server_url = "http://registry:8082";
    config.world_model_d_model = 768;
    const std::string text = ConfigLoader::to_string(config);
    for (const char* heading : {"Server:", "Logging:", "Model architecture:", "Training:", "Checkpointing:",
                                "Generation:", "RAG:", "GPU:", "Metrics client:", "Metrics daemon:",
                                "Dataset registry:", "FTP:", "Model Name Service:", "Trainer admin API:",
                                "World model:", "Hippocampal memory:"}) {
        EXPECT_NE(text.find(heading), std::string::npos) << heading;
    }
    EXPECT_NE(text.find("http://registry:8082"), std::string::npos);
    EXPECT_NE(text.find("768"), std::string::npos);
    for (const char* secret : {"s3cret-value", "abc123", "user:pw"}) {
        EXPECT_EQ(text.find(secret), std::string::npos) << secret << " leaked";
    }
    EXPECT_NE(text.find("<hidden>"), std::string::npos);
}

TEST_F(ConfigTest, PrintGoesThroughTheLoggerNotStdout) {
    std::ostringstream captured_stdout;
    std::streambuf* old = std::cout.rdbuf(captured_stdout.rdbuf());
    LogCapture log;
    ConfigLoader::print(ServiceConfig{});
    std::cout.rdbuf(old);
    EXPECT_TRUE(captured_stdout.str().empty()) << captured_stdout.str();
    EXPECT_NE(log.out.str().find("TRAINER_ADMIN_PORT"), std::string::npos);
}

TEST_F(ConfigTest, GpuStrategyIsMatchedInAnyCase) {
    createConfigFile({{"GPU_STRATEGY", "FULL"}});
    EXPECT_EQ(ConfigLoader::load(test_file.string()).gpu_strategy, GPUStrategy::FULL);
}

// ============================================================================
// TD-242: reload() re-applies the command-line flags
// ============================================================================

TEST_F(ConfigTest, ReloadReappliesCommandLineOverridesAndSaysWhichAreHeld) {
    createConfigFile({{"TEMPERATURE", "0.7"}, {"TOP_P", "0.9"}});
    ServiceConfig config = ConfigLoader::load(test_file.string());
    config.temperature = 0.3f;  // as --temperature 0.3 did at startup
    const auto cli = [](ServiceConfig& c) { c.temperature = 0.3f; };
    const std::set<std::string> cli_keys = {"TEMPERATURE"};
    const std::set<std::string> live = {"TEMPERATURE", "TOP_P"};
    std::mutex config_mutex;
    createConfigFile({{"TEMPERATURE", "0.9"}, {"TOP_P", "0.5"}});

    LogCapture log;
    EXPECT_TRUE(ConfigLoader::reload(config, test_file.string(), config_mutex, &live, cli, &cli_keys));
    EXPECT_FLOAT_EQ(config.temperature, 0.3f);  // used to become the file's 0.9
    EXPECT_FLOAT_EQ(config.top_p, 0.5f);
    EXPECT_NE(log.out.str().find("TEMPERATURE changed in the file but is held by a command-line flag"),
              std::string::npos)
        << log.out.str();
    EXPECT_EQ(log.out.str().find("TOP_P changed in the file but is held"), std::string::npos);
}

TEST_F(ConfigTest, ReloadValidatesWithTheCommandLineOverridesApplied) {
    // A flag that makes the reloaded config invalid is caught, as at startup.
    createConfigFile({{"PORT", "8080"}});
    ServiceConfig config = ConfigLoader::load(test_file.string());
    std::mutex config_mutex;
    const auto cli = [](ServiceConfig& c) { c.session_timeout = -5; };
    EXPECT_FALSE(ConfigLoader::reload(config, test_file.string(), config_mutex, nullptr, cli));
    EXPECT_EQ(config.session_timeout, ServiceConfig{}.session_timeout);
}

// TD-569: trainer_service's stall limit for a pass, from the file and the environment, 0 to 10080.
TEST_F(ConfigTest, TrainerPassStallMinutes) {
    EXPECT_EQ(ServiceConfig{}.trainer_pass_stall_minutes, 30);

    createConfigFile({{"TRAINER_PASS_STALL_MINUTES", "45"}});
    EXPECT_EQ(ConfigLoader::load(test_file.string()).trainer_pass_stall_minutes, 45);
    setEnv("TRAINER_PASS_STALL_MINUTES", "0");
    EXPECT_EQ(ConfigLoader::load(test_file.string()).trainer_pass_stall_minutes, 0);

    ServiceConfig config;
    std::vector<std::string> errors;
    for (const int ok : {0, 1, 10080}) {
        config.trainer_pass_stall_minutes = ok;
        errors.clear();
        EXPECT_TRUE(ConfigLoader::validate(config, errors)) << ok;
    }
    for (const int bad : {-1, 10081}) {
        config.trainer_pass_stall_minutes = bad;
        errors.clear();
        EXPECT_FALSE(ConfigLoader::validate(config, errors)) << bad;
        EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const std::string& e) {
            return e.find("trainer_pass_stall_minutes") != std::string::npos;
        }));
    }
}
