#pragma once

// @adai-status: stable
// @adai-version: 1.3.0
// @adai-reviewed: 2026-10-25


#include <algorithm>
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
#include <unordered_set>
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
        // TD-446: the thread always runs; an interval <= 0 means "disabled" and it idles until the
        // interval changes. It used to start only for a positive interval, so a later change from 0
        // to a positive value did nothing.
        sweep_thread_ = std::thread(&MetricsSessionRegistry::sweep_loop, this);
    }

    ~MetricsSessionRegistry() {
        if (sweep_thread_.joinable()) {
            {
                // Under the sweep mutex, so the stop can't slip in between the thread checking its
                // predicate and blocking (a disabled sweep waits with no timeout).
                std::lock_guard<std::mutex> lock(sweep_mutex_);
                stop_sweep_ = true;
            }
            sweep_cv_.notify_all();
            sweep_thread_.join();
        }
    }

    // Non-copyable, non-movable (owns a thread and a mutex)
    MetricsSessionRegistry(const MetricsSessionRegistry&) = delete;
    MetricsSessionRegistry& operator=(const MetricsSessionRegistry&) = delete;
    std::shared_ptr<TrainingMetricsService> create_or_get_session(const std::string& key) {
        // TD-446: archiving I/O runs after the registry lock is released (see archive_detached());
        // when this key's own previous session had to be archived, retry once that is done.
        for (;;) {
            std::vector<PendingArchive> pending;
            std::shared_ptr<TrainingMetricsService> result;
            bool retry = false;
            {
                std::unique_lock<std::shared_mutex> lock(registry_mutex_);
                wait_until_not_archiving(lock, key);
                evict_completed_sessions_locked(completed_ttl_seconds_, pending);

                auto existing = sessions_.find(key);
                if (existing != sessions_.end() &&
                    !should_replace_completed_session(existing->second.service)) {
                    result = existing->second.service;
                } else if (existing != sessions_.end()) {
                    pending.push_back(detach_locked(existing));
                    retry = true;
                } else if (make_room_locked(pending)) {
                    result = create_locked(key);
                }
            }
            archive_detached(pending);
            if (!retry)
                return result;
        }
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
        // The decision (conflict, reclaim, create) is made under one lock acquisition. Reclaiming
        // detaches the old session from the map under that lock; its archiving I/O runs after the
        // lock is released (TD-446), and the new session is created only once it's done:
        // concurrent callers for this key wait in wait_until_not_archiving() meanwhile.
        for (;;) {
            std::vector<PendingArchive> pending;
            SessionStartResult result;
            bool retry = false;
            {
                std::unique_lock<std::shared_mutex> lock(registry_mutex_);
                wait_until_not_archiving(lock, key);
                evict_completed_sessions_locked(completed_ttl_seconds_, pending);

                auto existing = sessions_.find(key);
                if (existing != sessions_.end()) {
                    if (should_replace_completed_session(existing->second.service)) {
                        adai::Logger::warn(
                            "[MetricsSessionRegistry] key='{}' is stale/completed — archiving and "
                            "allowing restart",
                            key);
                        pending.push_back(detach_locked(existing));
                        retry = true;
                    } else {
                        const auto snapshot = existing->second.service->get_current_snapshot();
                        // Genuinely live: refuse without touching it. Never started (a freshly
                        // created placeholder): safe to hand back as-is.
                        result = has_started(snapshot)
                                     ? SessionStartResult{nullptr, true}
                                     : SessionStartResult{existing->second.service, false};
                    }
                } else if (make_room_locked(pending)) {
                    result = SessionStartResult{create_locked(key), false};
                }
            }
            archive_detached(pending);
            if (!retry)
                return result;
        }
    }

    std::optional<std::shared_ptr<TrainingMetricsService>> get_session(
        const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);
        auto existing = sessions_.find(key);
        if (existing == sessions_.end()) {
            // TD-446: a miss is normal (completed sessions live only in the DB; the Prometheus
            // aggregate looks every listed session up). It used to log a warning plus one line per
            // live key.
            adai::Logger::debug("[get_session] key='{}' not live ({} live sessions)", key,
                                sessions_.size());
            return std::nullopt;
        }
        return existing->second.service;
    }

    std::vector<MetricsSessionSummary> list_sessions() const {
        std::shared_lock<std::shared_mutex> lock(registry_mutex_);

        std::vector<MetricsSessionSummary> summaries;
        std::unordered_set<std::string> live_keys;
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
            live_keys.insert(key);
        }
        lock.unlock();  // TD-446: the DB query below doesn't need the registry lock

        // Supplement with completed sessions from DB that have been evicted (TD-020)
        if (db_) {
            try {
                auto db_sessions = db_->list_sessions(std::optional<bool>(false));
                for (const auto& rec : db_sessions) {
                    if (live_keys.count(rec.key))
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

    // TD-440: age retention. Deletes every DB session not updated for retention_days (with all of
    // its rows) and every archived session file (any file in the metrics directories whose name
    // carries the archive marker, rotated backups included) last written before the same cutoff.
    // retention_days <= 0 keeps everything. `now` is a parameter so tests can move time forward.
    // Returns the number of DB sessions deleted. The sweep runs this at most once an hour.
    long long apply_retention(std::chrono::system_clock::time_point now) {
        namespace fs = std::filesystem;
        MetricsServiceConfig cfg = base_metrics_config();
        if (cfg.retention_days <= 0)
            return 0;
        const auto max_age = std::chrono::hours(24) * cfg.retention_days;
        const auto cutoff = now - max_age;

        long long deleted = 0;
        if (db_) {
            try {
                deleted = db_->delete_sessions_updated_before(cutoff);
            } catch (const std::exception& e) {
                adai::Logger::error("[MetricsSessionRegistry] retention delete failed: {}",
                                    e.what());
            }
        }

        // File times are on their own clock: compare each file's age with the same cutoff.
        const auto file_cutoff = fs::file_time_type::clock::now() -
                                 std::chrono::duration_cast<fs::file_time_type::duration>(
                                     std::chrono::system_clock::now() - cutoff);
        std::vector<fs::path> dirs;
        for (const std::string* f : {&cfg.metrics_file, &cfg.summary_file, &cfg.prometheus_file,
                                     &cfg.abnormal_samples_file}) {
            if (f->empty())
                continue;
            fs::path dir = fs::path(*f).parent_path();
            if (dir.empty())
                dir = ".";
            if (std::find(dirs.begin(), dirs.end(), dir) == dirs.end())
                dirs.push_back(dir);
        }
        size_t files_removed = 0;
        for (const auto& dir : dirs) {
            std::error_code ec;
            for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
                std::error_code fec;
                if (!it->is_regular_file(fec) ||
                    it->path().filename().string().find(kArchivedKeyMarker) == std::string::npos)
                    continue;
                const auto mtime = fs::last_write_time(it->path(), fec);
                if (!fec && mtime < file_cutoff && fs::remove(it->path(), fec))
                    ++files_removed;
            }
        }
        if (deleted > 0 || files_removed > 0) {
            adai::Logger::info(
                "[MetricsSessionRegistry] retention ({} days): deleted {} DB sessions and {} "
                "archived files",
                cfg.retention_days, deleted, files_removed);
        }
        return deleted;
    }

    size_t evict_completed_sessions(int max_age_seconds) {
        std::vector<PendingArchive> pending;
        {
            std::unique_lock<std::shared_mutex> lock(registry_mutex_);
            evict_completed_sessions_locked(max_age_seconds, pending);
        }
        archive_detached(pending);
        return pending.size();
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

    // Takes effect at once: the sweep thread wakes and starts a fresh wait with the new interval.
    // <= 0 disables sweeping (the thread idles). TD-446: <= 0 used to make the thread spin (0.51
    // CPU-seconds in 2 s, taking the exclusive lock each pass), and 0 -> positive did nothing.
    void set_sweep_interval_seconds(int seconds) {
        {
            std::lock_guard<std::mutex> lock(sweep_mutex_);
            sweep_interval_seconds_ = seconds;
            ++sweep_config_generation_;
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
        // Hygiene: label/config_snapshot used to live here too ("populated by
        // start_session() (TD-021 step 8)") but were never actually written
        // or read anywhere in this file — list_sessions() reads those fields
        // from the service's own snapshot instead. Removed as dead fields on
        // a private struct with no external accessors.
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
    void archive_session_files(const std::string& key, const std::string& archived_key,
                               const std::shared_ptr<TrainingMetricsService>& service) {
        if (key == "0-default")
            return;

        namespace fs = std::filesystem;
        const auto config = service->get_config();
        std::vector<std::string> paths = {config.metrics_file, config.summary_file,
                                          config.prometheus_file, config.abnormal_samples_file};
        // TD-440: the JSONL file's rotated backups belong to the session too.
        if (!config.metrics_file.empty()) {
            for (int i = 1; i <= config.metrics_file_backups; ++i)
                paths.push_back(config.metrics_file + "." + std::to_string(i));
        }
        for (const auto& path : paths) {
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

    // A session leaving the live map, waiting for its archiving I/O.
    struct PendingArchive {
        std::string key;
        std::string archived_key;
        std::shared_ptr<TrainingMetricsService> service;
    };

    // Removes a session from the live map and marks its key as being archived, so no new session
    // takes the key until archive_detached() has renamed the old one's rows and files. Must be
    // called with registry_mutex_ held exclusively.
    PendingArchive detach_locked(std::unordered_map<std::string, SessionEntry>::iterator it) {
        PendingArchive p{it->first, make_archived_key(it->first), it->second.service};
        archiving_keys_.insert(p.key);
        sessions_.erase(it);
        return p;
    }

    // Archives detached sessions: retires each service (final flush, then no more writes, so a
    // request still holding it can't recreate its files or rows), renames its DB rows and files to
    // the archived key, then releases the key. TD-446: this I/O used to run under the exclusive
    // registry lock on every start and sweep. Called WITHOUT registry_mutex_.
    void archive_detached(const std::vector<PendingArchive>& pending) {
        if (pending.empty())
            return;
        for (const auto& p : pending) {
            try {
                p.service->retire();
            } catch (const std::exception& e) {
                adai::Logger::warn("[MetricsSessionRegistry] flush before archiving '{}' failed: {}",
                                   p.key, e.what());
            }
            if (db_) {
                try {
                    db_->archive_session(p.key, p.archived_key);
                } catch (const std::exception& e) {
                    adai::Logger::error(
                        "[MetricsSessionRegistry] archive_session failed for key='{}': {}", p.key,
                        e.what());
                }
            }
            archive_session_files(p.key, p.archived_key, p.service);
        }
        {
            std::unique_lock<std::shared_mutex> lock(registry_mutex_);
            for (const auto& p : pending)
                archiving_keys_.erase(p.key);
        }
        archive_cv_.notify_all();
    }

    void wait_until_not_archiving(std::unique_lock<std::shared_mutex>& lock,
                                  const std::string& key) {
        archive_cv_.wait(lock, [&] { return archiving_keys_.count(key) == 0; });
    }

    // Must be called with registry_mutex_ held exclusively.
    std::shared_ptr<TrainingMetricsService> create_locked(const std::string& key) {
        auto service = std::make_shared<TrainingMetricsService>(config_for_session(key));
        if (db_) {
            service->set_database(db_.get(), key);
        }
        sessions_.emplace(key, SessionEntry{service, std::chrono::system_clock::now()});
        return service;
    }

    // TD-445: whether there's room for one more session, making room if needed. A finished or stale
    // session used to hold a max_live_sessions slot until its completed TTL (an hour by default), so
    // more than max_live_sessions trainer passes an hour got 503 metrics_server_full and trained
    // without metrics. Now, when full, finished/stale sessions are archived (to the DB, as on key
    // reuse) least recently updated first. Live sessions and never-started placeholders (e.g. the
    // startup "0-default" alias) are never evicted, so in effect only live sessions count.
    // Must be called with registry_mutex_ held exclusively.
    bool make_room_locked(std::vector<PendingArchive>& pending) {
        while (sessions_.size() >= max_live_sessions_) {
            auto victim = sessions_.end();
            std::chrono::system_clock::time_point victim_activity{};
            for (auto it = sessions_.begin(); it != sessions_.end(); ++it) {
                if (!should_replace_completed_session(it->second.service))
                    continue;  // live, or never started
                const auto snapshot = it->second.service->get_current_snapshot();
                const auto activity = snapshot.last_update_time.time_since_epoch().count() > 0
                                          ? snapshot.last_update_time
                                          : it->second.created_at;
                if (victim == sessions_.end() || activity < victim_activity) {
                    victim = it;
                    victim_activity = activity;
                }
            }
            if (victim == sessions_.end())
                return false;  // every slot is a live session
            adai::Logger::info(
                "[MetricsSessionRegistry] registry full ({} sessions); archiving finished/stale "
                "session '{}' to make room",
                sessions_.size(), victim->first);
            pending.push_back(detach_locked(victim));
        }
        return true;
    }

    size_t evict_completed_sessions_locked(int max_age_seconds,
                                           std::vector<PendingArchive>& pending) {
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
                pending.push_back(detach_locked(it));
            }
        }
        return expired_keys.size();
    }

    void sweep_loop() {
        std::unique_lock<std::mutex> lock(sweep_mutex_);
        while (!stop_sweep_) {
            // Wait for the interval, or for any interval change (TD-446): a change restarts the
            // wait with the new value instead of sweeping. <= 0 waits until a change or stop.
            const auto generation = sweep_config_generation_;
            auto woken = [&] {
                return stop_sweep_.load() || sweep_config_generation_ != generation;
            };
            if (sweep_interval_seconds_ <= 0) {
                sweep_cv_.wait(lock, woken);
                continue;
            }
            if (sweep_cv_.wait_for(lock, std::chrono::seconds(sweep_interval_seconds_), woken))
                continue;  // stopped, or the interval changed
            lock.unlock();
            // TD-134 (fixed): completed_ttl_seconds_ is guarded by registry_mutex_; read it through
            // the locked getter, not directly (a concurrent PUT /admin/config raced with this).
            evict_completed_sessions(completed_ttl_seconds());
            const auto now = std::chrono::system_clock::now();
            if (now - last_retention_run_ >= std::chrono::hours(1)) {
                last_retention_run_ = now;
                apply_retention(now);
            }
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
    std::uint64_t sweep_config_generation_ = 0;  ///< guarded by sweep_mutex_ (TD-446)
    std::unordered_set<std::string> archiving_keys_;  ///< guarded by registry_mutex_ (TD-446)
    std::condition_variable_any archive_cv_;          ///< signalled when a key finishes archiving
    std::chrono::system_clock::time_point last_retention_run_{};  ///< sweep thread only (TD-440)
    mutable std::mutex sweep_mutex_;  ///< guards sweep_cv_ wait only; mutable for the const getter
    std::condition_variable_any sweep_cv_;
    std::thread sweep_thread_;  ///< last member — joins after other members are valid
};