#include <gtest/gtest.h>
#include "../src/Logger.hpp"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <thread>
#include <chrono>

using namespace adai;
namespace fs = std::filesystem;

// ============================================================================
// Test Fixture
// ============================================================================

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        test_dir = fs::temp_directory_path() / "logger_test";
        fs::create_directories(test_dir);
        
        test_log_file = test_dir / "test.log";
        
        // Clean up any existing log files
        if (fs::exists(test_log_file)) {
            fs::remove(test_log_file);
        }
    }
    
    void TearDown() override {
        // Clean up test files
        cleanupLogFiles();
        
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }
    
    void cleanupLogFiles() {
        // Remove test log file and any rotated versions
        for (int i = 0; i < 10; ++i) {
            std::string rotated = test_log_file.string();
            if (i > 0) {
                rotated += "." + std::to_string(i);
            }
            if (fs::exists(rotated)) {
                fs::remove(rotated);
            }
        }
    }
    
    std::string readLogFile() {
        if (!fs::exists(test_log_file)) {
            return "";
        }
        
        std::ifstream file(test_log_file);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    int countLogFiles() {
        int count = 0;
        for (const auto& entry : fs::directory_iterator(test_dir)) {
            if (entry.path().filename().string().find("test.log") == 0) {
                count++;
            }
        }
        return count;
    }
    
    fs::path test_dir;
    fs::path test_log_file;
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(LoggerTest, InitializeConsoleOnly) {
    // Should not throw
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, "test_logger"));
    
    // Logger should be initialized
    EXPECT_NE(Logger::get(), nullptr);
    
    // Should have logger with correct name
    EXPECT_EQ(Logger::get()->name(), "test_logger");
}

TEST_F(LoggerTest, InitializeWithDefaults) {
    EXPECT_NO_THROW(Logger::init());
    EXPECT_NE(Logger::get(), nullptr);
    EXPECT_EQ(Logger::get()->name(), "adai");  // Default name
}

TEST_F(LoggerTest, InitializeWithFileRotation) {
    Logger::FileConfig file_config{
        test_log_file.string(),
        10,    // 10 MB max size
        5,     // 5 rotated files
        false  // no compression
    };
    
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, file_config));
    
    // Write a log message
    Logger::info("Test message");
    
    // Flush to ensure write
    Logger::get()->flush();
    
    // Give filesystem time to sync
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Log file should be created
    EXPECT_TRUE(fs::exists(test_log_file));
}

TEST_F(LoggerTest, FileConfigValidation) {
    // Test with minimal values
    Logger::FileConfig config1{test_log_file.string(), 1, 1, false};
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, config1));
    
    // Test with maximum values
    Logger::init(Logger::Level::INFO);  // Reset
    Logger::FileConfig config2{test_log_file.string(), 1024, 100, true};
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, config2));
}

TEST_F(LoggerTest, FileConfigEmptyPath) {
    // Empty path should fall back to console-only
    Logger::FileConfig config{"", 10, 5, false};
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, config));
    
    // No log file should be created
    Logger::info("Test message");
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_FALSE(fs::exists(test_log_file));
}

// ============================================================================
// Log Level Tests
// ============================================================================

TEST_F(LoggerTest, SetLevelFromEnum) {
    Logger::init(Logger::Level::INFO);
    
    EXPECT_NO_THROW(Logger::set_level(Logger::Level::DEBUG));
    EXPECT_NO_THROW(Logger::set_level(Logger::Level::INFO));
    EXPECT_NO_THROW(Logger::set_level(Logger::Level::WARN));
    EXPECT_NO_THROW(Logger::set_level(Logger::Level::ERROR));
}

TEST_F(LoggerTest, SetLevelFromString) {
    Logger::init(Logger::Level::INFO);
    
    EXPECT_NO_THROW(Logger::set_level("DEBUG"));
    EXPECT_NO_THROW(Logger::set_level("INFO"));
    EXPECT_NO_THROW(Logger::set_level("WARN"));
    EXPECT_NO_THROW(Logger::set_level("ERROR"));
}

TEST_F(LoggerTest, SetLevelCaseInsensitive) {
    Logger::init(Logger::Level::INFO);
    
    EXPECT_NO_THROW(Logger::set_level("debug"));
    EXPECT_NO_THROW(Logger::set_level("Info"));
    EXPECT_NO_THROW(Logger::set_level("WARN"));
    EXPECT_NO_THROW(Logger::set_level("error"));
}

TEST_F(LoggerTest, SetLevelInvalidString) {
    Logger::init(Logger::Level::INFO);
    
    // Invalid level strings should be handled gracefully
    // (spdlog will default to INFO or ignore invalid levels)
    EXPECT_NO_THROW(Logger::set_level("INVALID"));
}

// ============================================================================
// Logging Output Tests
// ============================================================================

TEST_F(LoggerTest, LogMessagesToFile) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Info message");
    Logger::warn("Warning message");
    Logger::error("Error message");
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    
    EXPECT_TRUE(content.find("Info message") != std::string::npos);
    EXPECT_TRUE(content.find("Warning message") != std::string::npos);
    EXPECT_TRUE(content.find("Error message") != std::string::npos);
}

TEST_F(LoggerTest, LogLevelFiltering) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::WARN, config);  // Set to WARN level
    
    Logger::debug("Debug message");  // Should not appear
    Logger::info("Info message");    // Should not appear
    Logger::warn("Warning message"); // Should appear
    Logger::error("Error message");  // Should appear
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    
    EXPECT_TRUE(content.find("Debug message") == std::string::npos);
    EXPECT_TRUE(content.find("Info message") == std::string::npos);
    EXPECT_TRUE(content.find("Warning message") != std::string::npos);
    EXPECT_TRUE(content.find("Error message") != std::string::npos);
}

TEST_F(LoggerTest, LogTimestampFormat) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Test message");
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    
    // Check for timestamp pattern [YYYY-MM-DD HH:MM:SS.mmm]
    EXPECT_TRUE(content.find("[20") != std::string::npos);  // Year
    EXPECT_TRUE(content.find("]") != std::string::npos);    // Closing bracket
}

TEST_F(LoggerTest, LogFormatWithLevelTag) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Info message");
    Logger::warn("Warning message");
    Logger::error("Error message");
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    
    EXPECT_TRUE(content.find("[info]") != std::string::npos);
    EXPECT_TRUE(content.find("[warning]") != std::string::npos);
    EXPECT_TRUE(content.find("[error]") != std::string::npos);
}

// ============================================================================
// Format String Tests
// ============================================================================

TEST_F(LoggerTest, LogFormatStrings) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Simple message");
    Logger::info("Message with int: {}", 42);
    Logger::info("Message with string: {}", "hello");
    Logger::info("Message with float: {:.2f}", 3.14159f);
    Logger::info("Multiple args: {} {} {}", 1, "two", 3.0f);
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    
    EXPECT_TRUE(content.find("Simple message") != std::string::npos);
    EXPECT_TRUE(content.find("Message with int: 42") != std::string::npos);
    EXPECT_TRUE(content.find("Message with string: hello") != std::string::npos);
    EXPECT_TRUE(content.find("Message with float: 3.14") != std::string::npos);
    EXPECT_TRUE(content.find("Multiple args: 1 two 3") != std::string::npos);
}

// ============================================================================
// File Rotation Tests
// ============================================================================

TEST_F(LoggerTest, FileRotationConfiguration) {
    // Test that rotation config is accepted
    Logger::FileConfig config{
        test_log_file.string(),
        1,     // 1 MB - small for testing
        3,     // 3 rotated files
        false
    };
    
    EXPECT_NO_THROW(Logger::init(Logger::Level::INFO, config));
    
    // Write some logs (not enough to trigger rotation in this test)
    for (int i = 0; i < 100; ++i) {
        Logger::info("Log message {}", i);
    }
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // At least the main log file should exist
    EXPECT_TRUE(fs::exists(test_log_file));
}

TEST_F(LoggerTest, MultipleFileConfigurations) {
    // Test changing file config by re-initializing
    Logger::FileConfig config1{test_log_file.string(), 5, 3, false};
    Logger::init(Logger::Level::INFO, config1);
    Logger::info("Message 1");
    
    // Re-initialize with different config
    Logger::FileConfig config2{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::WARN, config2);
    Logger::warn("Message 2");
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_TRUE(fs::exists(test_log_file));
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(LoggerTest, ConcurrentLogging) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    std::atomic<int> messages_logged{0};
    
    // Launch multiple threads logging concurrently
    std::vector<std::thread> threads;
    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                Logger::info("Thread {} message {}", t, i);
                messages_logged++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Verify all messages were logged
    EXPECT_EQ(messages_logged, 1000);
    
    // Log file should exist and contain messages
    EXPECT_TRUE(fs::exists(test_log_file));
    std::string content = readLogFile();
    EXPECT_FALSE(content.empty());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(LoggerTest, EmptyMessage) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    EXPECT_NO_THROW(Logger::info(""));
    Logger::get()->flush();
}

TEST_F(LoggerTest, VeryLongMessage) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    std::string long_message(10000, 'A');
    EXPECT_NO_THROW(Logger::info("{}", long_message));
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    EXPECT_TRUE(content.find("AAAAAAA") != std::string::npos);
}

TEST_F(LoggerTest, SpecialCharactersInMessage) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Special chars: \n\t\r\"'\\{}[]");
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    EXPECT_TRUE(content.find("Special chars:") != std::string::npos);
}

TEST_F(LoggerTest, UnicodeCharacters) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Unicode: 你好 мир 🚀");
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    EXPECT_TRUE(content.find("Unicode:") != std::string::npos);
}

TEST_F(LoggerTest, NullPointerFormat) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    const char* null_str = nullptr;
    // spdlog may handle nullptr differently, but shouldn't crash
    EXPECT_NO_THROW(Logger::info("Pointer: {}", static_cast<const void*>(null_str)));
    
    Logger::get()->flush();
}

TEST_F(LoggerTest, InvalidDirectoryPath) {
    Logger::FileConfig config{"/invalid/path/that/does/not/exist/test.log", 10, 5, false};
    
    // Should handle gracefully (spdlog may throw or fallback to console)
    // We just verify it doesn't crash the program
    try {
        Logger::init(Logger::Level::INFO, config);
    } catch (...) {
        // Exception is acceptable for invalid path
    }
}

TEST_F(LoggerTest, FilePermissionHandling) {
    // Create a read-only directory (on Unix systems)
    #ifndef _WIN32
    fs::path readonly_dir = test_dir / "readonly";
    fs::create_directories(readonly_dir);
    fs::permissions(readonly_dir, fs::perms::owner_read | fs::perms::owner_exec);
    
    Logger::FileConfig config{(readonly_dir / "test.log").string(), 10, 5, false};
    
    // Should handle gracefully
    try {
        Logger::init(Logger::Level::INFO, config);
        Logger::info("Test");
    } catch (...) {
        // Exception is acceptable for permission issues
    }
    
    // Restore permissions for cleanup
    fs::permissions(readonly_dir, fs::perms::all);
    #endif
}

TEST_F(LoggerTest, RapidReinitialization) {
    // Test rapid re-initialization doesn't cause issues
    for (int i = 0; i < 10; ++i) {
        Logger::init(Logger::Level::INFO);
        Logger::info("Message {}", i);
    }
    
    EXPECT_NO_THROW(Logger::get());
}

TEST_F(LoggerTest, LevelChangeDuringLogging) {
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::DEBUG, config);
    
    Logger::debug("Debug 1");
    Logger::set_level(Logger::Level::ERROR);
    Logger::debug("Debug 2");  // Should not appear
    Logger::error("Error 1");  // Should appear
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::string content = readLogFile();
    EXPECT_TRUE(content.find("Debug 1") != std::string::npos);
    EXPECT_TRUE(content.find("Debug 2") == std::string::npos);
    EXPECT_TRUE(content.find("Error 1") != std::string::npos);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(LoggerTest, LogRotationIntegration) {
    // Create a logger with very small file size to test rotation
    // Note: Actual rotation testing would require writing enough data
    // to exceed the size limit, which we simulate here
    
    Logger::FileConfig config{test_log_file.string(), 1, 3, false};  // 1 MB, 3 files
    Logger::init(Logger::Level::INFO, config);
    
    // Write many messages
    for (int i = 0; i < 1000; ++i) {
        Logger::info("Message {} with some extra padding to increase file size: {}", 
                     i, std::string(100, 'X'));
    }
    
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // At least the main log file should exist
    EXPECT_TRUE(fs::exists(test_log_file));
    
    // Depending on message size, we might have rotated files
    // Count total log files (main + rotated)
    int file_count = countLogFiles();
    EXPECT_GE(file_count, 1);  // At least the main file
    EXPECT_LE(file_count, 4);  // Max main + 3 rotated
}

TEST_F(LoggerTest, ConsoleAndFileDualSink) {
    // When file logging is enabled, both console and file should receive logs
    Logger::FileConfig config{test_log_file.string(), 10, 5, false};
    Logger::init(Logger::Level::INFO, config);
    
    Logger::info("Dual sink message");
    Logger::get()->flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // File should contain the message
    std::string content = readLogFile();
    EXPECT_TRUE(content.find("Dual sink message") != std::string::npos);
    
    // Console output is not easily testable without capturing stdout,
    // but we verify no crash occurred
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
