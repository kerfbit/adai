#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-14

// TD-173: trainer_service's own process-lifetime control state — the piece missing from TD-172's
// initial implementation. TrainerControlState (TrainerControlState.hpp) is deliberately
// reconstructed fresh for every incremental_trainer `resume --admin-port` child, scoped to one
// pass — correct for what it tracks (that pass's live progress). Nothing took over the role
// `serve`'s own single, process-lifetime TrainerControlState used to play for "should the
// *service* keep launching passes at all" — this class is that missing piece, owned by
// trainer_service's own main() for its entire lifetime, independent of how many children come and
// go. Mirrors TrainerControlState's own atomic-fields-plus-condvar-wake conventions deliberately
// (same shape, proven pattern) rather than sharing code with it — the two classes have unrelated
// lifetimes and no reason to be coupled.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace adai {

/**
 * @brief In-process state for trainer_service's own supervisory loop, read/written by
 *        TrainerServiceProxy's HTTP handlers on a separate thread.
 *
 * `paused` is deliberately NOT persisted to daemon_config.db (unlike the auto_save_* /
 * max_sessions_to_keep tunables TrainerControlState mirrors from its own daemon_config.db
 * overlay) — it's a live operational instruction ("stop launching new passes until told
 * otherwise"), not configuration. A trainer_service restart (crash or manual) should come back
 * unpaused, the same way a freshly-started `serve` process used to.
 *
 * Ownership / threading contract:
 *  - `paused`: admin thread sets via POST /admin/pause, clears via POST /admin/resume; the
 *    supervisory loop reads it before every `ChildProcess::start()` call.
 *  - `stop_requested`: the SIGTERM/SIGINT handler sets this via a plain atomic store ONLY —
 *    see the async-signal-safety note on `wake()` below; the supervisory loop reads it to exit.
 *  - Observability fields (`last_launch_unix`, `last_exit_unix`, `last_exit_code`,
 *    `current_child_pid`, `total_passes_*`): the supervisory loop writes after every
 *    `ChildProcess::start()`/exit; the admin thread reads them for GET /admin/status.
 */
class TrainerServiceControlState {
   public:
    // ---- Control flags (admin thread writes; supervisory loop reads) ----
    std::atomic<bool> paused{false};
    std::atomic<bool> stop_requested{false};

    // ---- Observability (supervisory loop writes; admin thread reads) ----
    std::atomic<std::int64_t> service_started_unix{0};
    std::atomic<std::int64_t> last_launch_unix{0};
    std::atomic<std::int64_t> last_exit_unix{0};
    std::atomic<int> last_exit_code{0};
    std::atomic<long long> current_child_pid{0};  // 0 = no child currently running
    std::atomic<long long> total_passes_launched{0};
    std::atomic<long long> total_passes_did_work{0};
    std::atomic<long long> total_passes_crashed{0};  // exit code outside {0,1}: launch failure or
                                                      // killed by signal (see ChildProcess::poll_exit())

    /**
     * @brief Wakes a sleeping `interruptible_sleep()` call immediately — called by POST
     *        /admin/resume so clearing a pause (or a newly-queued dataset) doesn't have to wait
     *        out the remaining poll interval.
     *
     * IMPORTANT — async-signal-safety: this locks a mutex and notifies a condition variable,
     * neither of which is on the POSIX async-signal-safe function list. Safe to call from a real
     * thread (the admin HTTP handler thread calling this from POST /admin/resume); NEVER call
     * this from a SIGTERM/SIGINT handler — those must only flip `stop_requested` via a plain
     * atomic store and let `interruptible_sleep()`'s own once-per-second poll notice it (same
     * ~1s latency the rest of this codebase's signal handling already accepts).
     *
     * wake_requested_ (guarded by wake_mutex_, consumed by interruptible_sleep) exists so a
     * wake() landing *before* interruptible_sleep() starts waiting is not silently lost — same
     * reasoning as TrainerControlState::wake()'s own identical mechanism.
     */
    void wake() {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        wake_requested_ = true;
        wake_cv_.notify_all();
    }

    /// Sleeps up to `seconds`, returning early if wake() is called or stop_requested becomes
    /// true — either while this call is waiting, or already pending from before it started.
    void interruptible_sleep(int seconds) {
        std::unique_lock<std::mutex> lock(wake_mutex_);
        for (int waited = 0; waited < seconds; ++waited) {
            if (wake_cv_.wait_for(lock, std::chrono::seconds(1),
                                  [this] { return wake_requested_ || stop_requested.load(); })) {
                break;
            }
        }
        wake_requested_ = false;
    }

   private:
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
    bool wake_requested_ = false;  // guarded by wake_mutex_
};

}  // namespace adai
