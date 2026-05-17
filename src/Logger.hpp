#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <string>

namespace adai {

/**
 * @brief Centralized logging utility for ADAI service
 *
 * Provides structured logging with configurable levels and formatting.
 * Uses spdlog internally for high-performance logging.
 *
 * TODO: See TECHNICAL_DEBT.md Future Enhancement #8 - JSON log output for machine parsing
 * TODO: See TECHNICAL_DEBT.md Future Enhancement #10 - Per-module logging levels
 * TODO: See TECHNICAL_DEBT.md Future Enhancement #11 - Custom sinks (systemd journal, syslog)
 */
class Logger {
   public:
    /**
     * @brief Logging levels
     */
    enum class Level { DEBUG, INFO, WARN, ERROR };

    /**
     * @brief Log file configuration
     */
    struct FileConfig {
        std::string path;         // Log file path (empty = disabled)
        size_t max_size_mb = 10;  // Max file size before rotation (MB)
        size_t max_files = 5;     // Max number of rotated files
        bool compress = false;    // Enable compression (future enhancement)
    };

    /**
     * @brief Initialize the logger
     *
     * @param level Logging level
     * @param name Logger name (default: "adai")
     */
    static void init(Level level = Level::INFO, const std::string& name = "adai");

    /**
     * @brief Initialize the logger with file rotation support
     *
     * @param level Logging level
     * @param file_config File logging configuration
     * @param name Logger name (default: "adai")
     */
    static void init(Level level, const FileConfig& file_config, const std::string& name = "adai");

    /**
     * @brief Set logging level from string
     *
     * @param level_str Level as string: "DEBUG", "INFO", "WARN", "ERROR"
     */
    static void set_level(const std::string& level_str);

    /**
     * @brief Set logging level
     *
     * @param level Logging level
     */
    static void set_level(Level level);

    /**
     * @brief Get the spdlog logger instance
     *
     * @return Shared pointer to spdlog logger
     */
    static std::shared_ptr<spdlog::logger> get();

    /**
     * @brief Log debug message
     *
     * @tparam Args Variadic template for format arguments
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    static void debug(const char* fmt, Args&&... args) {
        get()->debug(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Log info message
     *
     * @tparam Args Variadic template for format arguments
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    static void info(const char* fmt, Args&&... args) {
        get()->info(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Log warning message
     *
     * @tparam Args Variadic template for format arguments
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    static void warn(const char* fmt, Args&&... args) {
        get()->warn(fmt, std::forward<Args>(args)...);
    }

    /**
     * @brief Log error message
     *
     * @tparam Args Variadic template for format arguments
     * @param fmt Format string
     * @param args Format arguments
     */
    template <typename... Args>
    static void error(const char* fmt, Args&&... args) {
        get()->error(fmt, std::forward<Args>(args)...);
    }

   private:
    static std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace adai
