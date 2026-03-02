#include "Logger.hpp"
#include <spdlog/pattern_formatter.h>
#include <algorithm>
#include <cctype>

// TODO: See TECHNICAL_DEBT.md Future Enhancement #8 - JSON Log Output Format
// TODO: See TECHNICAL_DEBT.md Future Enhancement #9 - File Rotation and Management
// TODO: See TECHNICAL_DEBT.md Future Enhancement #10 - Per-Module Log Levels
// TODO: See TECHNICAL_DEBT.md Future Enhancement #11 - Custom Log Sinks (syslog, network)

namespace adai {

// Static member initialization
std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;

void Logger::init(Level level, const std::string& name) {
    if (!logger_) {
        // Create console logger with color support
        logger_ = spdlog::stdout_color_mt(name);
        
        // TODO: See TECHNICAL_DEBT.md Future Enhancement #8 - Support JSON format
        // Add LOG_FORMAT configuration option (text/json)
        // Implement JSON formatter with structured fields
        
        // Set pattern: [timestamp] [level] message
        // Example: [2026-03-01 10:30:45.123] [INFO] Server starting...
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        
        // TODO: See TECHNICAL_DEBT.md Future Enhancement #9 - Add rotating file sink
        // Add support for file-based logging with automatic rotation
        // Support compression of old logs
        
        // Set default level
        set_level(level);
        
        // Flush on every message (important for containerized environments)
        logger_->flush_on(spdlog::level::info);
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

} // namespace adai
