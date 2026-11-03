// @adai-status: stable
// @adai-version: 1.2.0
// @adai-reviewed: 2026-11-02

#include "Logger.hpp"
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <mutex>
#include <vector>

// File rotation implemented - see LOG_FILE_ROTATION_COMPLETE.md
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #1) - JSON Log Output Format
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #2) - Per-Module Log Levels
// TODO: See TECHNICAL_DEBT.md Future Enhancement (Logging and Observability #3) - Custom Log Sinks (syslog, network)

namespace adai {

// Static member initialization
std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;
bool Logger::configured_ = false;

namespace {
// TD-290: messages logged through deferred() before the binary's own init().
struct DeferredMessage {
    spdlog::log_clock::time_point time;
    spdlog::level::level_enum level;
    std::string text;
};
std::mutex deferred_mutex;
std::vector<DeferredMessage> deferred_messages;
constexpr size_t kMaxDeferredMessages = 256;

spdlog::level::level_enum to_spdlog(Logger::Level level) {
    switch (level) {
        case Logger::Level::DEBUG: return spdlog::level::debug;
        case Logger::Level::INFO: return spdlog::level::info;
        case Logger::Level::WARN: return spdlog::level::warn;
        case Logger::Level::ERROR: return spdlog::level::err;
    }
    return spdlog::level::info;
}
}  // namespace

void Logger::deferred(Level level, const std::string& message) {
    const bool configured = configured_;
    get()->log(to_spdlog(level), "{}", message);
    if (configured) return;
    std::lock_guard<std::mutex> lock(deferred_mutex);
    if (deferred_messages.size() < kMaxDeferredMessages) {
        deferred_messages.push_back({spdlog::log_clock::now(), to_spdlog(level), message});
    }
}

void Logger::replay_deferred_to_file(const spdlog::sink_ptr& file_sink, Level min_level) {
    std::lock_guard<std::mutex> lock(deferred_mutex);
    if (file_sink) {
        for (const auto& m : deferred_messages) {
            if (m.level < to_spdlog(min_level)) continue;  // TD-431: LOG_LEVEL applies to the replay too
            spdlog::details::log_msg msg(spdlog::source_loc{}, "adai", m.level, m.text);
            msg.time = m.time;
            file_sink->log(msg);
        }
        file_sink->flush();
    }
    deferred_messages.clear();
}

void Logger::init(Level level, const std::string& name) {
    // TODO: See TD-430 in TECHNICAL_DEBT.md - drops only the NEW name; re-init while other threads log is unsafe
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

    configured_ = true;
    replay_deferred_to_file(nullptr);  // TD-290: already on the console, and there's no file
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

    // TD-290: start-up messages logged before this init() reach the file (they were on the console).
    replay_deferred_to_file(sinks.size() > 1 ? sinks.back() : nullptr, level);

    // Create logger with all configured sinks
    logger_ = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
    configured_ = true;
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

Logger::Level Logger::parse_level(const std::string& level_str) {
    std::string upper = level_str;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
    if (upper == "DEBUG") return Level::DEBUG;
    if (upper == "WARN" || upper == "WARNING") return Level::WARN;
    if (upper == "ERROR") return Level::ERROR;
    return Level::INFO;
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

// TODO: See TD-430 in TECHNICAL_DEBT.md - unsynchronized lazy init: concurrent first use makes a log call throw
// spdlog_ex("logger with name 'adai' already exists") (verified); logger_ itself is a racy static shared_ptr
std::shared_ptr<spdlog::logger> Logger::get() {
    if (!logger_) {
        init(Level::INFO);
        configured_ = false;  // a lazy console logger; the binary hasn't set its own up yet
    }
    return logger_;
}

}  // namespace adai
