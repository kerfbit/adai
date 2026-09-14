#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
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
 * Windows `TerminateProcess()` are both safe to call concurrently with `poll_exit()`).
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
     * @brief Best-effort graceful stop of the current child, if any: SIGTERM on POSIX,
     *   TerminateProcess() on Windows (Windows has no direct SIGTERM analog reachable across
     *   process boundaries without the child cooperating via a console-control-handler dance
     *   this class doesn't set up — TerminateProcess() is abrupt there, matching the same
     *   limitation launch_background()'s own Windows branch already accepts elsewhere in this
     *   codebase). No-op if no child is running.
     */
    void request_stop();

    /// True if a child was started and hasn't been observed to exit yet via poll_exit().
    bool is_running() const {
        return running_;
    }

   private:
    bool running_ = false;
#ifdef _WIN32
    void* process_handle_ = nullptr;  // HANDLE, kept void* to avoid pulling <windows.h> into this header
#else
    long long pid_ = -1;  // pid_t, widened to avoid pulling <sys/types.h> into this header
#endif
};

}  // namespace adai
