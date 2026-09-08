// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "Logger.hpp"
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>

// File rotation implemented - see LOG_FILE_ROTATION_COMPLETE.md
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #1) - JSON Log Output Format
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #2) - Per-Module Log Levels
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #3) - Custom Log Sinks (syslog, network)

namespace adai {

// Static member initialization
std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;

void Logger::init(Level level, const std::string& name) {
    // Drop existing logger if re-initializing
    if (logger_) {
        spdlog::drop(name);
        logger_ = nullptr;
    }

    // Use stderr for the console sink so structured log output does not share
    // the same file descriptor as std::cout (used by the TUI dashboard).
    // stdout = program/TUI output; stderr = diagnostics/logs (POSIX convention).
    logger_ = spdlog::stderr_color_mt(name);

    // TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #1) - Support JSON format
    // Add LOG_FORMAT configuration option (text/json)
    // Implement JSON formatter with structured fields

    // Set pattern: [timestamp] [level] message
    // Example: [2026-03-01 10:30:45.123] [INFO] Server starting...
    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // Set default level
    set_level(level);

    // Flush on every message (important for containerized environments)
    logger_->flush_on(spdlog::level::info);
}

void Logger::init(Level level, const FileConfig& file_config, const std::string& name) {
    // Drop existing logger if re-initializing
    if (logger_) {
        spdlog::drop(name);
        logger_ = nullptr;
    }

    std::vector<spdlog::sink_ptr> sinks;

    // Use stderr so log output does not interfere with std::cout / TUI dashboard.
    auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    sinks.push_back(console_sink);

    // Add rotating file sink if path is provided
    if (!file_config.path.empty()) {
        try {
            // Convert MB to bytes for spdlog
            size_t max_size_bytes = file_config.max_size_mb * 1024 * 1024;

            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                file_config.path, max_size_bytes, file_config.max_files);

            // Use same pattern for file output
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
            sinks.push_back(file_sink);

            // Note: Compression support would require post-rotation hooks
            // spdlog doesn't natively support compression, but files can be
            // compressed externally via logrotate or custom scripts
            if (file_config.compress) {
                // Log a message that compression is noted but requires external tool
                // We'll log this after logger is created
            }
        } catch (const spdlog::spdlog_ex& ex) {
            // If file sink creation fails, fall back to console only
            std::cerr << "Failed to create rotating file sink: " << ex.what() << '\n';
            std::cerr << "Falling back to console-only logging" << '\n';
        }
    }

    // Create logger with all configured sinks
    logger_ = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    spdlog::register_logger(logger_);

    // Set default level
    set_level(level);

    // Flush on every info message
    logger_->flush_on(spdlog::level::info);

    // Log file configuration info
    if (!file_config.path.empty()) {
        logger_->info("File logging enabled:");
        logger_->info("  Path: {}", file_config.path);
        logger_->info("  Max size: {} MB", file_config.max_size_mb);
        logger_->info("  Max files: {}", file_config.max_files);

        if (file_config.compress) {
            logger_->info("  Compression: enabled (requires external tool like logrotate)");
            logger_->info("  Note: spdlog rotates files but doesn't compress them.");
            logger_->info("  Use logrotate or similar tool to compress old log files.");
        }
    }
}

void Logger::set_level(const std::string& level_str) {
    // Convert to uppercase for case-insensitive comparison
    std::string upper_level = level_str;
    std::transform(upper_level.begin(), upper_level.end(), upper_level.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (upper_level == "DEBUG") {
        set_level(Level::DEBUG);
    } else if (upper_level == "INFO") {
        set_level(Level::INFO);
    } else if (upper_level == "WARN" || upper_level == "WARNING") {
        set_level(Level::WARN);
    } else if (upper_level == "ERROR") {
        set_level(Level::ERROR);
    } else {
        // Default to INFO if unknown level
        if (logger_) {
            logger_->warn("Unknown log level '{}', defaulting to INFO", level_str);
        }
        set_level(Level::INFO);
    }
}

void Logger::set_level(Level level) {
    if (!logger_) {
        init(level);
        return;
    }

    switch (level) {
        case Level::DEBUG:
            logger_->set_level(spdlog::level::debug);
            break;
        case Level::INFO:
            logger_->set_level(spdlog::level::info);
            break;
        case Level::WARN:
            logger_->set_level(spdlog::level::warn);
            break;
        case Level::ERROR:
            logger_->set_level(spdlog::level::err);
            break;
    }
}

std::shared_ptr<spdlog::logger> Logger::get() {
    if (!logger_) {
        init(Level::INFO);
    }
    return logger_;
}

}  // namespace adai
