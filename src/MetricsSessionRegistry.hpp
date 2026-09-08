#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "Logger.hpp"
#include "MetricsDatabase.hpp"
#include "TrainingMetricsService.hpp"

struct MetricsSessionSummary {
    std::string key;
    int session_id = 0;
    std::string label;  ///< human-readable label; empty until TrainingMetricsService populates
                        ///< (TD-021 step 8)
    std::string config_snapshot;  ///< compact training-config JSON; empty until step 8
    bool is_training = false;
    bool effective_is_training =
        false;  ///< is_training && !is_stale — matches health-check liveness
    int current_epoch = 0;
    int total_epochs = 0;
    float current_loss = 0.0f;
    float current_validation_loss = 0.0f;
    float best_validation_loss = std::numeric_limits<float>::max();
    std::chrono::system_clock::time_point session_start_time;
    std::chrono::system_clock::time_point last_update_time;
    std::string metrics_url;
};

/// Result of MetricsSessionRegistry::start_session_or_conflict(). `conflict == true` means a
/// genuinely live (is_training && !is_stale) session already owns the requested key and no
/// replacement occurred; `service` is only meaningful when `conflict == false`.
struct SessionStartResult {
    std::shared_ptr<TrainingMetricsService> service;
    bool conflict = false;
};

class MetricsSessionRegistry {
   public:
    explicit MetricsSessionRegistry(MetricsServiceConfig base_config = MetricsServiceConfig(),
                                    size_t max_live_sessions = 16, int completed_ttl_seconds = 3600,
                                    int sweep_interval_seconds = 60,
                                    const std::string& storage_backend = "",
                                    const std::string& db_path = "", const std::string& db_url = "",
                                    int db_pool_size = 4)
        : base_config_(std::move(base_config)),
          max_live_sessions_(max_live_sessions),
          completed_ttl_seconds_(completed_ttl_seconds),
          sweep_interval_seconds_(sweep_interval_seconds) {
        // Create DB backend if configured (TD-020)
        if (!storage_backend.empty() && storage_backend != "file") {
            try {
                db_ =
                    MetricsDatabaseFactory::create(storage_backend, db_path, db_url, db_pool_size);
                if (db_) {
                    adai::Logger::info("[MetricsSessionRegistry] Database backend initialized: {}",
                                       storage_backend);
                }
            } catch (const std::exception& e) {
                adai::Logger::error("[MetricsSessionRegistry] Failed to create DB backend: {}",
                                    e.what());
            }
        }
        if (sweep_interval_seconds_ > 0) {
            sweep_thread_ = std::thread(&MetricsSessionRegistry::sweep_loop, this);
        }
    }

    ~MetricsSessionRegistry() {
        if (sweep_thread_.joinable()) {
            stop_sweep_ = true;
            sweep_cv_.notify_all();
            sweep_thread_.join();
        }
    }

    // Non-copyable, non-movable (owns a thread and a mutex)
    MetricsSessionRegistry(const MetricsSessionRegistry&) = delete;
    MetricsSessionRegistry& operator=(const MetricsSessionRegistry&) = delete;
    std::shared_ptr<TrainingMetricsService> create_or_get_session(const std::string& key) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);

        evict_completed_sessions_locked(completed_ttl_seconds_);

        auto existing = sessions_.find(key);
        if (existing != sessions_.end()) {
            if (should_replace_completed_session(existing->second.service)) {
                archive_and_remove_locked(existing);
            } else {
                return existing->second.service;
            }
        }

        if (sessions_.size() >= max_live_sessions_) {
            return nullptr;
        }

        auto service = std::make_shared<TrainingMetricsService>(config_for_session(key));
        if (db_) {
            service->set_database(db_.get(), key);
        }
        sessions_.emplace(key, SessionEntry{service, std::chrono::system_clock::now()});
        return service;
    }

    // Atomically checks whether `key` is already held by a genuinely live session and, only if
    // not, reclaims (archiving first) or creates a session for it — all under one lock
    // acquisition. This must be used instead of a separate get_session() + create_or_get_session()
    // pair when the caller needs to refuse a duplicate /session/start: doing the conflict check
    // and the replace-if-stale decision as two separate registry calls leaves a window where a
    // session that is still genuinely training (e.g. mid-validation, briefly past the staleness
    // threshold) gets silently archived and replaced by the second caller before the check can
    // see its real state.
    SessionStartResult start_session_or_conflict(const std::string& key) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);

        evict_completed_sessions_locked(completed_ttl_seconds_);

        auto existing = sessions_.find(key);
        if (existing != sessions_.end()) {
            if (should_replace_completed_session(existing->second.service)) {
                adai::Logger::warn(
                    "[MetricsSessionRegistry] key='{}' is stale/completed — archiving and "
                    "allowing restart",
                    key);
                archive_and_remove_locked(existing);
            } else {
                const auto snapshot = existing->second.service->get_current_snapshot();
                if (has_started(snapshot)) {
                    // Genuinely live — refuse without touching the existing session.
                    return SessionStartResult{nullptr, true};
                }
                // Never started (freshly created placeholder) — safe to hand back as-is.
                return SessionStartResult{existing->second.service, false};
            }
        }

        if (sessions_.size() >= max_live_sessions_) {
            return SessionStartResult{nullptr, false};
        }

        auto service = std::make_shared<TrainingMetricsService>(config_for_session(key));
        if (db_) {
            service->set_database(db_.get(), key);
        }
        sessions_.emplace(key, SessionEntry{service, std::chrono::system_clock::now()});
        return SessionStartResult{service, false};
    }

    std::optional<std::shared_ptr<TrainingMetricsService>> get_session(
        const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        auto existing = sessions_.find(key);
        if (existing == sessions_.end()) {
            adai::Logger::warn("[get_session] key='{}' (len={}) not found; map has {} entries:",
                               key, key.size(), sessions_.size());
            for (const auto& [k, _] : sessions_) {
                adai::Logger::warn("  -> '{}' (len={})", k, k.size());
            }
            return std::nullopt;
        }
        return existing->second.service;
    }

    std::vector<MetricsSessionSummary> list_sessions() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);

        std::vector<MetricsSessionSummary> summaries;
        summaries.reserve(sessions_.size());
        for (const auto& [key, entry] : sessions_) {
            auto snapshot = entry.service->get_current_snapshot();
            MetricsSessionSummary summary;
            summary.key = key;
            summary.session_id = snapshot.session_id;
            summary.label = snapshot.label;
            summary.config_snapshot = snapshot.config_snapshot;
            summary.is_training = snapshot.is_training;
            summary.effective_is_training = snapshot.effective_is_training;
            summary.current_epoch = snapshot.current_epoch;
            summary.total_epochs = snapshot.total_epochs;
            summary.current_loss = snapshot.current_loss;
            summary.current_validation_loss = snapshot.current_validation_loss;
            summary.best_validation_loss = snapshot.best_validation_loss;
            summary.session_start_time = snapshot.session_start_time;
            summary.last_update_time = snapshot.last_update_time;
            summary.metrics_url = "/api/sessions/" + key + "/metrics/current";
            summaries.push_back(std::move(summary));
        }

        // Supplement with completed sessions from DB that have been evicted (TD-020)
        if (db_) {
            try {
                auto db_sessions = db_->list_sessions(std::optional<bool>(false));
                for (const auto& rec : db_sessions) {
                    if (sessions_.count(rec.key))
                        continue;
                    // Archived rows are permanent history kept only so their metrics can
                    // still be queried by exact key — they must never clutter the live
                    // dashboard/session-picker listing.
                    if (is_archived_key(rec.key))
                        continue;
                    MetricsSessionSummary summary;
                    summary.key = rec.key;
                    summary.session_id = rec.session_id;
                    summary.label = rec.label;
                    summary.config_snapshot = rec.config_json;
                    summary.is_training = rec.is_training;
                    summary.effective_is_training = false;
                    summary.current_epoch = rec.total_epochs;
                    summary.total_epochs = rec.total_epochs;
                    summary.best_validation_loss = rec.best_validation_loss;
                    // A completed/archived session has no live TrainingMetricsSnapshot, so
                    // there's no "current" loss anymore — final_loss/final_validation_loss are
                    // the last values recorded before it stopped being live, which is the
                    // closest equivalent and prevents this from silently reading as 0.
                    summary.current_loss = rec.final_loss;
                    summary.current_validation_loss = rec.final_validation_loss;
                    summary.session_start_time = rec.created_at;
                    summary.last_update_time = rec.last_update_at;
                    summary.metrics_url = "/api/sessions/" + rec.key + "/metrics/current";
                    summaries.push_back(std::move(summary));
                }
            } catch (const std::exception& e) {
                adai::Logger::warn("[MetricsSessionRegistry] DB list_sessions failed: {}",
                                   e.what());
            }
        }

        return summaries;
    }

    IMetricsDatabase* get_database() const {
        return db_.get();
    }

    size_t evict_completed_sessions(int max_age_seconds) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        return evict_completed_sessions_locked(max_age_seconds);
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        return sessions_.size();
    }

    size_t max_live_sessions() const {
        return max_live_sessions_;
    }

    // ── Admin-mutable settings (see TrainingMetricsAPI's PUT /admin/config) ───

    void set_max_live_sessions(size_t n) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        max_live_sessions_ = n;
    }

    int completed_ttl_seconds() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        return completed_ttl_seconds_;
    }

    void set_completed_ttl_seconds(int seconds) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        completed_ttl_seconds_ = seconds;
    }

    int sweep_interval_seconds() const {
        std::lock_guard<std::mutex> lock(sweep_mutex_);
        return sweep_interval_seconds_;
    }

    // Safe to call while the sweep thread is running: notify_all() wakes its
    // wait_for(), which re-reads sweep_interval_seconds_ fresh on the next loop
    // iteration (see sweep_loop()) rather than being stuck on the duration it
    // was constructed with.
    void set_sweep_interval_seconds(int seconds) {
        {
            std::lock_guard<std::mutex> lock(sweep_mutex_);
            sweep_interval_seconds_ = seconds;
        }
        sweep_cv_.notify_all();
    }

    MetricsServiceConfig base_metrics_config() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        return base_config_;
    }

    // Applies `mutator` to base_config_ (so future sessions pick up the change)
    // and to every currently-live session's config (so the change is immediate).
    // Each session's config is read-mutate-written individually rather than
    // overwritten wholesale, so per-session derived file paths (see
    // config_for_session) are preserved.
    void update_metrics_config(const std::function<void(MetricsServiceConfig&)>& mutator) {
        std::unique_lock<std::shared_mutex> lock(registry_mutex_);
        mutator(base_config_);
        for (auto& [key, entry] : sessions_) {
            auto cfg = entry.service->get_config();
            mutator(cfg);
            entry.service->set_config(cfg);
        }
    }

   private:
    struct SessionEntry {
        std::shared_ptr<TrainingMetricsService> service;
        std::chrono::system_clock::time_point created_at;
        std::string label;            ///< populated by start_session() (TD-021 step 8)
        std::string config_snapshot;  ///< populated by start_session() (TD-021 step 8)
    };

    MetricsServiceConfig config_for_session(const std::string& key) const {
        if (key == "0-default") {
            return base_config_;
        }

        MetricsServiceConfig config = base_config_;
        config.metrics_file = derive_session_file_path(base_config_.metrics_file, key, "_metrics");
        config.summary_file =
            derive_session_file_path(base_config_.summary_file, key, "_metrics_summary");
        config.prometheus_file =
            derive_session_file_path(base_config_.prometheus_file, key, "_metrics");
        config.abnormal_samples_file =
            derive_session_file_path(base_config_.abnormal_samples_file, key, "_abnormal_samples");
        return config;
    }

    static std::string derive_session_file_path(const std::string& base_file,
                                                const std::string& key, const std::string& suffix) {
        namespace fs = std::filesystem;

        fs::path path(base_file);
        const auto extension = path.extension().string();
        const auto parent = path.parent_path();
        return (parent / (key + suffix + extension)).string();
    }

    static bool has_started(const TrainingMetricsSnapshot& snapshot) {
        return snapshot.session_id != 0 || snapshot.total_epochs != 0 ||
               snapshot.total_samples != 0 || snapshot.total_samples_trained != 0;
    }

    static bool should_replace_completed_session(
        const std::shared_ptr<TrainingMetricsService>& service) {
        const auto snapshot = service->get_current_snapshot();
        if (!has_started(snapshot))
            return false;
        if (!snapshot.is_training)
            return true;           // completed normally
        return snapshot.is_stale;  // trainer died without posting /end
    }

    static bool is_completed_and_stale(const SessionEntry& entry, int max_age_seconds) {
        const auto snapshot = entry.service->get_current_snapshot();
        if (!has_started(snapshot))
            return false;
        // Actively training and receiving updates — do not evict.
        if (snapshot.is_training && !snapshot.is_stale)
            return false;

        const auto last_activity = snapshot.last_update_time.time_since_epoch().count() > 0
                                       ? snapshot.last_update_time
                                       : entry.created_at;
        const auto max_age = std::chrono::seconds(std::max(0, max_age_seconds));
        return (std::chrono::system_clock::now() - last_activity) >= max_age;
    }

    // Marker embedded in keys produced by make_archived_key(); used to recognize and filter
    // out archived rows in list_sessions()'s DB supplement, since they are permanent
    // bookkeeping residue, not sessions a dashboard should ever show or let a user pick.
    static constexpr const char* kArchivedKeyMarker = "_archived_";

    static bool is_archived_key(const std::string& key) {
        return key.find(kArchivedKeyMarker) != std::string::npos;
    }

    // Builds a unique key for a session that is going stale/completed and is about to be
    // dropped from the live map, so a subsequent session that reuses the same name never
    // shares database rows or on-disk files with the old (dead) run.
    std::string make_archived_key(const std::string& key) {
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
        return key + kArchivedKeyMarker + std::to_string(now_ms) + "_" +
               std::to_string(archive_counter_.fetch_add(1, std::memory_order_relaxed));
    }

    // Renames the on-disk metrics files derived from `key` to instead be derived from
    // `archived_key`. The "0-default" key intentionally maps to fixed, shared legacy paths
    // (see config_for_session) rather than per-session paths, so there is nothing to rename.
    void archive_session_files_locked(const std::string& key, const std::string& archived_key,
                                      const std::shared_ptr<TrainingMetricsService>& service) {
        if (key == "0-default")
            return;

        namespace fs = std::filesystem;
        const auto config = service->get_config();
        const std::string* paths[] = {&config.metrics_file, &config.summary_file,
                                      &config.prometheus_file, &config.abnormal_samples_file};
        for (const auto* path_ptr : paths) {
            const std::string& path = *path_ptr;
            if (path.empty())
                continue;

            fs::path p(path);
            std::error_code exists_ec;
            if (!fs::exists(p, exists_ec))
                continue;

            std::string filename = p.filename().string();
            const auto pos = filename.find(key);
            if (pos == std::string::npos)
                continue;
            filename.replace(pos, key.size(), archived_key);

            std::error_code rename_ec;
            fs::rename(p, p.parent_path() / filename, rename_ec);
            if (rename_ec) {
                adai::Logger::warn("[MetricsSessionRegistry] Failed to archive file '{}': {}",
                                   p.string(), rename_ec.message());
            }
        }
    }

    // Archives (renames in the DB and on disk) a session that is leaving the live map,
    // then erases it. Must be called with registry_mutex_ held exclusively.
    void archive_and_remove_locked(std::unordered_map<std::string, SessionEntry>::iterator it) {
        const std::string key = it->first;
        const auto service = it->second.service;
        const std::string archived_key = make_archived_key(key);

        if (db_) {
            try {
                db_->archive_session(key, archived_key);
            } catch (const std::exception& e) {
                adai::Logger::error(
                    "[MetricsSessionRegistry] archive_session failed for key='{}': {}", key,
                    e.what());
            }
        }
        archive_session_files_locked(key, archived_key, service);

        sessions_.erase(it);
    }

    size_t evict_completed_sessions_locked(int max_age_seconds) {
        std::vector<std::string> expired_keys;
        expired_keys.reserve(sessions_.size());
        for (const auto& [key, entry] : sessions_) {
            if (is_completed_and_stale(entry, max_age_seconds)) {
                expired_keys.push_back(key);
            }
        }

        for (const auto& key : expired_keys) {
            auto it = sessions_.find(key);
            if (it != sessions_.end()) {
                archive_and_remove_locked(it);
            }
        }
        return expired_keys.size();
    }

    void sweep_loop() {
        std::unique_lock<std::mutex> lock(sweep_mutex_);
        while (!stop_sweep_) {
            sweep_cv_.wait_for(lock, std::chrono::seconds(sweep_interval_seconds_),
                               [this] { return stop_sweep_.load(); });
            if (stop_sweep_)
                break;
            lock.unlock();
            evict_completed_sessions(completed_ttl_seconds_);
            lock.lock();
        }
    }

    MetricsServiceConfig base_config_;
    std::unique_ptr<IMetricsDatabase> db_;
    mutable std::shared_mutex registry_mutex_;
    std::unordered_map<std::string, SessionEntry> sessions_;
    size_t max_live_sessions_;
    int completed_ttl_seconds_;
    int sweep_interval_seconds_;
    std::atomic<uint64_t> archive_counter_{0};

    std::atomic<bool> stop_sweep_{false};
    mutable std::mutex sweep_mutex_;  ///< guards sweep_cv_ wait only; mutable for the const getter
    std::condition_variable_any sweep_cv_;
    std::thread sweep_thread_;  ///< last member — joins after other members are valid
};