#pragma once

// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-14

// TD-172: small cross-platform "launch and monitor one child process" helper, extracted from
// trainer_service's own main() so the launch/poll/terminate logic is unit-testable against a real
// short-lived child (e.g. a trivial helper binary or `sleep`) without needing a real
// incremental_trainer invocation. Mirrors IncrementalTrainingTool.cpp's existing
// launch_background()'s POSIX-fork/Windows-CreateProcess split, but for a process this class
// itself starts, tracks, and can ask to stop — launch_background() only ever detaches a copy of
// the *current* process into the background and never monitors it again afterward.

#include <string>
#include <vector>

namespace adai {

/**
 * @brief Launches and monitors a single external process at a time.
 *
 * One `ChildProcess` instance manages at most one live child. Calling `start()` while a child is
 * already running (per the last `poll_exit()`/`is_running()` check) is a no-op that returns
 * false — callers are expected to confirm the previous child has exited (via `poll_exit()`)
 * before starting the next one, which is exactly the supervisory-loop shape
 * `IncrementalTrainingTool.cpp`'s old `serve` command already used for its in-process passes.
 *
 * Not thread-safe: intended to be driven from a single supervisory loop thread; `request_stop()`
 * is the one exception (safe to call from a signal handler's own thread — POSIX `kill()` and
 * Windows console-control/`TerminateProcess()` are both safe to call concurrently with
 * `poll_exit()`).
 */
class ChildProcess {
   public:
    ChildProcess() = default;
    ~ChildProcess();

    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

    /**
     * @brief Launches `argv[0]` with `argv[1..]` as its own arguments.
     * @return false if a child is already running, or if the launch itself failed
     *   (fork()/execvp() or CreateProcess() error — logged via adai::Logger, not returned here).
     */
    bool start(const std::vector<std::string>& argv);

    /**
     * @brief Non-blocking check for whether the current child has exited.
     * @param exit_code Out-param, set only when this returns true. The child's real exit code on
     *   a clean exit; 128 (POSIX convention) if it was killed by a signal instead.
     * @return true exactly once per child, the first poll_exit() call after it actually exits
     *   (or immediately, if it already exited before this call). false while still running, or if
     *   no child has ever been started, or after that one true has already been consumed.
     */
    bool poll_exit(int* exit_code);

    /**
     * @brief Best-effort graceful stop request of the current child, if any: SIGTERM on POSIX,
     *   an attempt at `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT)` on Windows (falling back to
     *   immediate `TerminateProcess()` only if that send itself fails — Windows has no direct
     *   SIGTERM analog, but a console-control event reaching the child's own signal-equivalent
     *   handler is closer to graceful than an unconditional force-kill). Does NOT wait for the
     *   child to actually exit and does NOT escalate — fire-and-forget. No-op if no child is
     *   running. Prefer `stop_and_wait()` when the caller needs a guaranteed-terminated outcome
     *   within a bounded time; this lower-level method exists for callers (e.g. a signal handler
     *   re-armed on repeated signals) that just want to nudge a child without blocking at all.
     */
    void request_stop();

    /**
     * @brief Stops the current child and blocks until it's confirmed gone, escalating to a force
     *   kill if it doesn't exit gracefully in time.
     *
     * Calls `request_stop()`, polls `poll_exit()` up to `timeout_ms`, and if the child is still
     * running after that, sends SIGKILL (POSIX) / `TerminateProcess()` (Windows) and does one
     * final bounded wait to reap it. This is what makes shutdown safe against a genuinely wedged
     * child (e.g. this host's own documented GPU-driver *hang*, not just crash — a hang may never
     * respond to the graceful request at all) without relying on an external supervisor (systemd)
     * to eventually SIGKILL the whole cgroup.
     *
     * @param timeout_ms How long to wait for a graceful exit before escalating to a force kill.
     * @param exit_code Out-param, set the same way `poll_exit()`'s is; -2 if the process had to
     *   be force-killed (distinct from -1's "no longer trackable" and 128's "killed by an
     *   ordinary signal" so callers can tell a forced kill apart from either).
     * @return true once the process is confirmed gone (always true if a child was running when
     *   called); false if no child was running at all.
     */
    bool stop_and_wait(int timeout_ms, int* exit_code);

    /// True if a child was started and hasn't been observed to exit yet via poll_exit().
    bool is_running() const {
        return running_;
    }

    /// The current child's PID, or 0 if none is running. For observability (e.g. GET
    /// /admin/status) — not meant for direct signaling; use request_stop()/stop_and_wait().
    long long pid() const;

   private:
    bool running_ = false;
#ifdef _WIN32
    void* process_handle_ = nullptr;  // HANDLE, kept void* to avoid pulling <windows.h> into this header
#else
    long long pid_ = -1;  // pid_t, widened to avoid pulling <sys/types.h> into this header
#endif
};

}  // namespace adai
