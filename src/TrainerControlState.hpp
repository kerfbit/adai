#pragma once

// @adai-status: stable
// @adai-version: 1.4.0
// @adai-reviewed: 2026-11-03


#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include "Logger.hpp"
#include "TrainerAdminConfig.hpp"

namespace adai {

/**
 * @brief Phase of one `incremental_trainer` training pass, reported by GET /admin/status.
 *
 * TD-522: since TD-172 a TrainerControlState lives for one pass (the `resume --admin-port` child
 * trainer_service launches); trainer_service answers `idle` itself between passes. A pass starts
 * in Idle and goes back to Idle when it ends, however it ends.
 */
enum class TrainerPhase {
    Idle,          ///< No pass active (before it starts, after it ends).
    LoadingData,   ///< Reading/parsing the acquired dataset files from disk.
    Tokenizing,    ///< ChatbotTrainer::train()'s internal preprocess_data() step.
    Training,      ///< At least one training sample has been processed this pass.
    Checkpointing, ///< A checkpoint write is in flight (best-effort — brief).
    Pausing,       ///< TD-522: asked to pause (/admin/pause, SIGTERM/SIGINT) while loading,
                   ///< tokenizing or training; the pass stops at the next optimizer-step
                   ///< boundary, reports Checkpointing while it saves, then Idle.
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
 * @brief State shared between one training pass and its TrainerAdminAPI's HTTP handlers, which run
 *        on another thread of the same process.
 *
 * TD-522: since TD-172 this lives for one pass only: `incremental_trainer --admin-port N resume`, a
 * child of trainer_service, creates it. trainer_service keeps its own process-lifetime state
 * (TrainerServiceControlState: the service-level pause, its idle-poll sleep and wake) and proxies
 * /admin/* to this pass while it runs.
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
 *  - `paused`: admin thread sets it with request_pause() (POST /admin/pause) and
 *    clears it with clear_pause() (POST /admin/resume); incremental_trainer's
 *    signal handler stores it directly (async-signal-safe). It doubles as
 *    ChatbotTrainer's cooperative abort flag (ChatbotTrainer::set_abort_flag()),
 *    so the pass drains at its next optimizer-step boundary.
 *  - `phase`: anyone reads it; it's written only through set_phase(),
 *    request_pause() and clear_pause(), which keep it Pausing while a pause
 *    is pending (TD-522).
 *  - `checkpoint_requested`: admin thread sets via POST /admin/checkpoint
 *    (request_checkpoint()); the active pass consumes it with
 *    take_checkpoint_request() and forces an immediate checkpoint write,
 *    independent of the auto-save cadence, then reports it with
 *    mark_checkpoint_served() (TD-521: an auto-save doesn't count).
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
    // TD-522: read freely; write through set_phase() (the pass) or request_pause()/clear_pause() (admin).
    std::atomic<TrainerPhase> phase{TrainerPhase::Idle};
    std::atomic<int> current_epoch{0};
    std::atomic<int> total_epochs{0};
    std::atomic<long long> samples_trained_this_pass{0};
    // TD-520: NaN until the pass reports a loss (GET /admin/status sends null), so an unset best_loss
    // no longer reads as a perfect 0. A non-finite last_loss after a sample means the pass diverged.
    std::atomic<double> last_loss{std::numeric_limits<double>::quiet_NaN()};
    std::atomic<double> best_loss{std::numeric_limits<double>::quiet_NaN()};

    // ---- Action flags (admin writes; training-pass consumes) ----
    std::atomic<bool> paused{false};
    std::atomic<bool> checkpoint_requested{false};
    // TD-416: the signal (SIGTERM/SIGINT) that asked this pass to stop, or 0. incremental_trainer's
    // signal handler sets it together with `paused`, so the pass drains exactly as for /admin/pause
    // (checkpoint, release claimed files, exit 1); it only changes the log message.
    std::atomic<int> stop_signal{0};

    // ---- Checkpoint bookkeeping (training-pass writes; admin reads) ----
    std::atomic<long long> checkpoints_written{0};
    std::atomic<long long> last_checkpoint_time_unix{0};

    // TD-521: requested checkpoints are numbered, so POST /admin/checkpoint?wait_ms= waits for one
    // written for its own request rather than any checkpoint (an auto-save used to count).
    std::atomic<long long> checkpoint_requests_made{0};    // admin side: the last ticket handed out
    std::atomic<long long> checkpoint_requests_served{0};  // pass side: tickets up to this one are written

    /// Admin side: asks the pass for a checkpoint; returns this request's ticket.
    long long request_checkpoint() {
        const long long ticket = checkpoint_requests_made.fetch_add(1) + 1;
        checkpoint_requested = true;
        return ticket;
    }

    /// Pass side: 0 when no checkpoint was asked for, else the newest ticket the checkpoint about to
    /// be written answers (every request made before it is written counts as served by it).
    long long take_checkpoint_request() {
        if (!checkpoint_requested.exchange(false)) return 0;
        return checkpoint_requests_made.load();
    }

    /// Pass side: the checkpoint for take_checkpoint_request()'s ticket was written.
    void mark_checkpoint_served(long long ticket) {
        long long current = checkpoint_requests_served.load();
        while (current < ticket && !checkpoint_requests_served.compare_exchange_weak(current, ticket)) {
        }
    }

    // ---- Tunables (admin writes via PUT /admin/config; training-pass reads) ----
    std::atomic<bool> auto_save_enabled{true};
    std::atomic<int> auto_save_every_samples{1000};
    std::atomic<int> auto_save_every_minutes{30};
    std::atomic<int> max_sessions_to_keep{50};

    // ---- Phase transitions (TD-522) ----

    /// Pass side: enters `p`. While a pause is pending a working phase (loading, tokenizing,
    /// training) is reported as Pausing instead; Checkpointing and Idle are reported as they are.
    void set_phase(TrainerPhase p) {
        std::lock_guard<std::mutex> lock(phase_mutex_);
        if (paused.load() && is_working(p)) {
            phase_before_pause_ = p;
            p = TrainerPhase::Pausing;
        }
        phase.store(p);
    }

    /// Admin side (POST /admin/pause): asks the pass to drain, and reports Pausing at once if it's
    /// working. Not async-signal-safe (it locks); a signal handler stores `paused` directly, and the
    /// pass's next set_phase() turns that into Pausing.
    void request_pause() {
        std::lock_guard<std::mutex> lock(phase_mutex_);
        paused.store(true);
        const TrainerPhase current = phase.load();
        if (is_working(current)) {
            phase_before_pause_ = current;
            phase.store(TrainerPhase::Pausing);
        }
    }

    /// Admin side (POST /admin/resume): cancels a pending pause. A pass that hasn't reached its
    /// next optimizer-step boundary yet carries on, so it reports the phase it had again.
    void clear_pause() {
        std::lock_guard<std::mutex> lock(phase_mutex_);
        paused.store(false);
        if (phase.load() == TrainerPhase::Pausing) phase.store(phase_before_pause_);
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

    /// The most entries the ring keeps (and GET /admin/logs?limit= accepts).
    static constexpr std::size_t kMaxLogEntries = kMaxAdminLogEntries;
    /// TD-522: the longest message the ring keeps, in bytes; the log file gets the whole message.
    static constexpr std::size_t kMaxLogMessageBytes = 4096;

    /**
     * @brief Emits `message` through adai::Logger at the matching level and records it in the
     *        ring buffer GET /admin/logs serves, so the ops dashboard's activity log shows what the
     *        pass's log file shows. Safe to call from any thread.
     *
     * TD-522: a message below the logger's level goes to neither (the ring used to keep Debug
     * entries the log file never got), and the ring's copy is cut to kMaxLogMessageBytes at a
     * UTF-8 character boundary, with " [truncated]" added.
     */
    void log(TrainerLogLevel level, const std::string& message) {
        if (!Logger::get()->should_log(to_spdlog_level(level))) return;
        {
            std::lock_guard<std::mutex> lock(log_mutex_);
            log_entries_.push_back(TrainerLogEntry{
                next_log_id_.fetch_add(1),
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count(),
                level,
                capped_log_message(message),
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
    static bool is_working(TrainerPhase p) {
        return p == TrainerPhase::LoadingData || p == TrainerPhase::Tokenizing ||
               p == TrainerPhase::Training;
    }

    static spdlog::level::level_enum to_spdlog_level(TrainerLogLevel level) {
        switch (level) {
            case TrainerLogLevel::Debug:
                return spdlog::level::debug;
            case TrainerLogLevel::Info:
                return spdlog::level::info;
            case TrainerLogLevel::Warn:
                return spdlog::level::warn;
            case TrainerLogLevel::Error:
                return spdlog::level::err;
        }
        return spdlog::level::info;
    }

    static std::string capped_log_message(const std::string& message) {
        if (message.size() <= kMaxLogMessageBytes) return message;
        std::size_t cut = kMaxLogMessageBytes;
        // Back up over UTF-8 continuation bytes (10xxxxxx) so a character isn't split.
        while (cut > 0 && (static_cast<unsigned char>(message[cut]) & 0xC0) == 0x80) --cut;
        return message.substr(0, cut) + " [truncated]";
    }

    std::mutex phase_mutex_;
    TrainerPhase phase_before_pause_ = TrainerPhase::Idle;  ///< guarded by phase_mutex_

    mutable std::mutex identity_mutex_;
    std::string run_id_;
    std::string session_id_;
    std::string model_name_;
    std::string last_checkpoint_path_;

    mutable std::mutex log_mutex_;
    std::deque<TrainerLogEntry> log_entries_;
    std::atomic<std::uint64_t> next_log_id_{1};
};

/**
 * @brief TD-522: sets a pass's phase back to Idle when the scope is left by an exception (only the
 *        normal and pause paths used to), so GET /admin/status doesn't report a dead pass as working.
 */
class TrainerPhaseResetOnThrow {
   public:
    explicit TrainerPhaseResetOnThrow(TrainerControlState* control)
        : control_(control), exceptions_(std::uncaught_exceptions()) {}
    ~TrainerPhaseResetOnThrow() {
        if (control_ != nullptr && std::uncaught_exceptions() > exceptions_) {
            control_->set_phase(TrainerPhase::Idle);
        }
    }
    TrainerPhaseResetOnThrow(const TrainerPhaseResetOnThrow&) = delete;
    TrainerPhaseResetOnThrow& operator=(const TrainerPhaseResetOnThrow&) = delete;

   private:
    TrainerControlState* control_;
    int exceptions_;
};

}  // namespace adai
