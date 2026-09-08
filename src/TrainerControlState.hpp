#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>
#include "Logger.hpp"

namespace adai {

/**
 * @brief Lifecycle phase of the `incremental_trainer serve` supervisory loop,
 *        surfaced verbatim by GET /admin/status.
 *
 * No `Stopped`/`Finished` value — the supervisor itself doesn't finish during
 * normal operation, it just cycles between Idle and the active-pass phases
 * for as long as the process runs.
 */
enum class TrainerPhase {
    Idle,          ///< No pass active; polling for pending work.
    LoadingData,   ///< Reading/parsing the acquired dataset files from disk.
    Tokenizing,    ///< ChatbotTrainer::train()'s internal preprocess_data() step.
    Training,      ///< At least one training sample has been processed this pass.
    Checkpointing, ///< A checkpoint write is in flight (best-effort — brief).
    Pausing,       ///< Draining after /admin/pause; returns to Idle once the
                   ///< current optimizer-step boundary is reached and a
                   ///< checkpoint has been written.
};

inline const char* to_string(TrainerPhase p) {
    switch (p) {
        case TrainerPhase::Idle:
            return "idle";
        case TrainerPhase::LoadingData:
            return "loading_data";
        case TrainerPhase::Tokenizing:
            return "tokenizing";
        case TrainerPhase::Training:
            return "training";
        case TrainerPhase::Checkpointing:
            return "checkpointing";
        case TrainerPhase::Pausing:
            return "pausing";
    }
    return "unknown";
}

/**
 * @brief Severity for [TrainerLogEntry] — deliberately the same four levels as
 *        adai::Logger::Level (not a bespoke UI taxonomy), since every entry either
 *        comes from or is mirrored into the real logger — see
 *        TrainerControlState::log().
 */
enum class TrainerLogLevel { Debug, Info, Warn, Error };

inline const char* to_string(TrainerLogLevel level) {
    switch (level) {
        case TrainerLogLevel::Debug:
            return "debug";
        case TrainerLogLevel::Info:
            return "info";
        case TrainerLogLevel::Warn:
            return "warn";
        case TrainerLogLevel::Error:
            return "error";
    }
    return "info";
}

/// One entry in TrainerControlState's in-memory log ring buffer, served by
/// GET /admin/logs. `id` is monotonically increasing per-process (never reused,
/// even after older entries are evicted), so a client can tell entries apart
/// without relying on timestamp uniqueness.
struct TrainerLogEntry {
    std::uint64_t id = 0;
    std::int64_t timestamp_unix_ms = 0;
    TrainerLogLevel level = TrainerLogLevel::Info;
    std::string message;
};

/**
 * @brief In-process shared state between `incremental_trainer serve`'s
 *        supervisory loop (and whichever training pass it's currently
 *        running) and TrainerAdminAPI's HTTP handlers, which run on a
 *        separate thread. No IPC — both live in the same process for the
 *        whole lifetime of `serve`; this is what makes the admin API
 *        genuinely always-on rather than only reachable while a single
 *        `resume` process instance happens to be alive between systemd
 *        restart cycles. See CLAUDE.md / the "admin control daemon" plan.
 *
 * Ownership / threading contract (each field is written from exactly one
 * side and read from the other — no field is read-modify-written from both):
 *  - Tunables (auto_save_*, max_sessions_to_keep): admin thread writes via
 *    PUT /admin/config; IncrementalTrainer reads them (should_auto_save(),
 *    cleanup_old_sessions()) in place of the config-file defaults whenever
 *    a control state is attached.
 *  - Progress fields (phase, current_epoch, total_epochs,
 *    samples_trained_this_pass, last_loss, best_loss): the active training
 *    pass writes; admin thread reads for GET /admin/status.
 *  - `paused`: admin thread sets via POST /admin/pause and clears via
 *    POST /admin/resume; the supervisory loop reads it to skip starting a
 *    new pass, and it doubles as ChatbotTrainer's cooperative abort flag
 *    (passed directly to ChatbotTrainer::set_abort_flag()) so a pass in
 *    flight drains at its next optimizer-step boundary.
 *  - `checkpoint_requested`: admin thread sets via POST /admin/checkpoint;
 *    the active pass consumes it with exchange(false) and forces an
 *    immediate checkpoint write, independent of the auto-save cadence.
 *  - `checkpoints_written` / `last_checkpoint_time_unix`: active pass
 *    increments/stamps after any checkpoint write (auto-save, forced, or
 *    abort-drain); admin thread reads for GET /admin/status.
 *  - Identity strings (run_id/session_id/model_name/last_checkpoint_path):
 *    mutex-guarded, copied under the lock only for the duration of the
 *    string copy — never held across I/O or training work.
 *  - Log ring buffer: any thread may call [log], which both appends to the
 *    buffer and emits through adai::Logger — see [log]'s doc comment.
 */
class TrainerControlState {
   public:
    // ---- Phase / progress (training-pass writes; admin reads) ----
    std::atomic<TrainerPhase> phase{TrainerPhase::Idle};
    std::atomic<int> current_epoch{0};
    std::atomic<int> total_epochs{0};
    std::atomic<long long> samples_trained_this_pass{0};
    std::atomic<double> last_loss{0.0};
    std::atomic<double> best_loss{0.0};

    // ---- Action flags (admin writes; training-pass/supervisor consume) ----
    std::atomic<bool> paused{false};
    std::atomic<bool> checkpoint_requested{false};

    // ---- Checkpoint bookkeeping (training-pass writes; admin reads) ----
    std::atomic<long long> checkpoints_written{0};
    std::atomic<long long> last_checkpoint_time_unix{0};

    // ---- Tunables (admin writes via PUT /admin/config; training-pass reads) ----
    std::atomic<bool> auto_save_enabled{true};
    std::atomic<int> auto_save_every_samples{1000};
    std::atomic<int> auto_save_every_minutes{30};
    std::atomic<int> max_sessions_to_keep{50};

    /// Short-circuits the supervisory loop's idle poll sleep — called by
    /// POST /admin/resume so a newly-queued dataset (or a pause being
    /// cleared) doesn't have to wait out the full poll interval. Spurious
    /// early wakeups are harmless: the loop just re-checks for pending work
    /// and, finding none, goes back to sleep.
    void wake() {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        wake_cv_.notify_all();
    }

    /// Sleeps up to `seconds`, returning early if wake() is called.
    void interruptible_sleep(int seconds) {
        std::unique_lock<std::mutex> lock(wake_mutex_);
        wake_cv_.wait_for(lock, std::chrono::seconds(seconds));
    }

    // ---- Identity strings (mutex-guarded) ----
    std::string get_run_id() const {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        return run_id_;
    }
    void set_run_id(std::string v) {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        run_id_ = std::move(v);
    }
    std::string get_session_id() const {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        return session_id_;
    }
    void set_session_id(std::string v) {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        session_id_ = std::move(v);
    }
    std::string get_model_name() const {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        return model_name_;
    }
    void set_model_name(std::string v) {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        model_name_ = std::move(v);
    }
    std::string get_last_checkpoint_path() const {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        return last_checkpoint_path_;
    }
    void set_last_checkpoint_path(std::string v) {
        std::lock_guard<std::mutex> lock(identity_mutex_);
        last_checkpoint_path_ = std::move(v);
    }

    // ---- Log ring buffer (GET /admin/logs) ----

    /**
     * @brief Records `message` into the in-memory ring buffer GET /admin/logs
     *        serves, and also emits it through the real adai::Logger at the
     *        matching level — this is the single call site every serve-loop/
     *        admin-action log line should go through instead of calling
     *        adai::Logger directly, so a message that should be visible in the
     *        Android ops dashboard's activity log never needs to remember to
     *        also log normally, and vice versa. Safe to call from any thread.
     */
    void log(TrainerLogLevel level, const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(log_mutex_);
            log_entries_.push_back(TrainerLogEntry{
                next_log_id_.fetch_add(1),
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count(),
                level,
                message,
            });
            while (log_entries_.size() > kMaxLogEntries) {
                log_entries_.pop_front();
            }
        }
        switch (level) {
            case TrainerLogLevel::Debug:
                Logger::debug("{}", message);
                break;
            case TrainerLogLevel::Info:
                Logger::info("{}", message);
                break;
            case TrainerLogLevel::Warn:
                Logger::warn("{}", message);
                break;
            case TrainerLogLevel::Error:
                Logger::error("{}", message);
                break;
        }
    }

    /// Returns up to `limit` of the most recent entries (oldest first),
    /// capped at kMaxLogEntries regardless of what `limit` requests.
    std::vector<TrainerLogEntry> recent_logs(std::size_t limit = kMaxLogEntries) const {
        std::lock_guard<std::mutex> lock(log_mutex_);
        const std::size_t n = std::min(limit, log_entries_.size());
        return std::vector<TrainerLogEntry>(log_entries_.end() - static_cast<long>(n), log_entries_.end());
    }

   private:
    mutable std::mutex identity_mutex_;
    std::string run_id_;
    std::string session_id_;
    std::string model_name_;
    std::string last_checkpoint_path_;

    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;

    static constexpr std::size_t kMaxLogEntries = 200;
    mutable std::mutex log_mutex_;
    std::deque<TrainerLogEntry> log_entries_;
    std::atomic<std::uint64_t> next_log_id_{1};
};

}  // namespace adai
