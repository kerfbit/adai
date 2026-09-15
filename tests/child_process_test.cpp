/**
 * @file child_process_test.cpp
 * @brief Tests for ChildProcess (TD-172) — trainer_service's cross-platform "launch and monitor
 *        one child process" helper. POSIX-only for now: this sandbox has no way to build/run the
 *        Windows branch (matches this project's existing convention of build-verifying, not
 *        run-verifying, Windows-only code paths — see scripts/build_windows.sh).
 *
 * Exercises real child processes (/bin/sh -c "...") rather than a mocked process layer, same
 * philosophy as trainer_admin_api_test.cpp/trainer_service_proxy_test.cpp using a real
 * httplib::Server instead of a mocked HTTP layer.
 */

#include "ChildProcess.hpp"

#ifndef _WIN32

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using adai::ChildProcess;

namespace {

bool wait_for_exit(ChildProcess& proc, int* exit_code, int timeout_ms = 3000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (proc.poll_exit(exit_code)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

}  // namespace

TEST(ChildProcessTest, StartLaunchesAProcessThatIsInitiallyRunning) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 0.3"}));
    EXPECT_TRUE(proc.is_running());

    int exit_code = -1;
    ASSERT_TRUE(wait_for_exit(proc, &exit_code));
    EXPECT_EQ(exit_code, 0);
    EXPECT_FALSE(proc.is_running());
}

TEST(ChildProcessTest, PollExitReturnsFalseWhileStillRunning) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 1"}));

    int exit_code = -1;
    EXPECT_FALSE(proc.poll_exit(&exit_code));
    EXPECT_TRUE(proc.is_running());

    proc.request_stop();
    ASSERT_TRUE(wait_for_exit(proc, &exit_code, 2000));
}

TEST(ChildProcessTest, CapturesTheChildsRealExitCode) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "exit 7"}));

    int exit_code = -1;
    ASSERT_TRUE(wait_for_exit(proc, &exit_code));
    EXPECT_EQ(exit_code, 7);
}

TEST(ChildProcessTest, RequestStopTerminatesALongRunningChild) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 30"}));
    ASSERT_TRUE(proc.is_running());

    proc.request_stop();

    int exit_code = -1;
    ASSERT_TRUE(wait_for_exit(proc, &exit_code, 2000))
        << "child did not exit within 2s of request_stop()";
    EXPECT_FALSE(proc.is_running());
}

TEST(ChildProcessTest, StartWhileAlreadyRunningIsRejected) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 1"}));
    EXPECT_FALSE(proc.start({"/bin/sh", "-c", "sleep 1"}));

    proc.request_stop();
    int exit_code = -1;
    wait_for_exit(proc, &exit_code, 2000);
}

TEST(ChildProcessTest, StartWithEmptyArgvIsRejected) {
    ChildProcess proc;
    EXPECT_FALSE(proc.start({}));
    EXPECT_FALSE(proc.is_running());
}

TEST(ChildProcessTest, DestructorReapsAStillRunningChildWithoutHanging) {
    // Regression against a leaked zombie / hung destructor: construct in a nested scope so the
    // destructor runs while the child is still alive, mirroring what happens if
    // trainer_service exits (or a test fixture is torn down) without an explicit stop+wait.
    {
        ChildProcess proc;
        ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 30"}));
        ASSERT_TRUE(proc.is_running());
    }  // ~ChildProcess() must terminate + reap here, not block forever on the 30s sleep.
    SUCCEED();
}

TEST(ChildProcessTest, SequentialStartsAfterExitAreIndependent) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "exit 1"}));
    int exit_code = -1;
    ASSERT_TRUE(wait_for_exit(proc, &exit_code));
    EXPECT_EQ(exit_code, 1);

    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "exit 0"}));
    ASSERT_TRUE(wait_for_exit(proc, &exit_code));
    EXPECT_EQ(exit_code, 0);
}

// ============================================================================
// TD-173: pid() and stop_and_wait()
// ============================================================================

TEST(ChildProcessTest, PidReturnsZeroWhenNoChildRunning) {
    ChildProcess proc;
    EXPECT_EQ(proc.pid(), 0);
}

TEST(ChildProcessTest, PidReturnsAPositiveValueWhileRunningAndZeroAfterExit) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 0.3"}));
    EXPECT_GT(proc.pid(), 0);

    int exit_code = -1;
    ASSERT_TRUE(wait_for_exit(proc, &exit_code));
    EXPECT_EQ(proc.pid(), 0);
}

TEST(ChildProcessTest, StopAndWaitReturnsPromptlyForAChildThatExitsGracefully) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "sleep 30"}));

    int exit_code = -1;
    const auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE(proc.stop_and_wait(5000, &exit_code));
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(proc.is_running());
    // A plain `sh -c "sleep 30"` has no SIGTERM trap, so the default disposition (terminate)
    // applies immediately — this should return in well under the 5s grace period, not need the
    // force-kill escalation at all.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 2000);
}

TEST(ChildProcessTest, StopAndWaitReturnsFalseWhenNoChildIsRunning) {
    ChildProcess proc;
    int exit_code = -1;
    EXPECT_FALSE(proc.stop_and_wait(1000, &exit_code));
}

// The regression this item exists to fix: a child that installs `trap "" TERM` explicitly ignores
// SIGTERM (this is exactly the shape of "wedged" process this host's own documented GPU-driver
// *hang* produces — a process that is alive but will never respond to a graceful stop request on
// its own). Before TD-173, request_stop()'s single SIGTERM plus an unconditional blocking
// waitpid() in ~ChildProcess() (or an unbounded re-send loop in TrainerServiceMain.cpp) would hang
// forever here. stop_and_wait() must escalate to SIGKILL after the timeout and return within a
// bounded time regardless.
TEST(ChildProcessTest, StopAndWaitEscalatesToForceKillWhenChildIgnoresSigterm) {
    ChildProcess proc;
    ASSERT_TRUE(proc.start({"/bin/sh", "-c", "trap '' TERM; sleep 30"}));
    // Give the trap a moment to actually install before we test it.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    int exit_code = -1;
    const auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE(proc.stop_and_wait(500, &exit_code));  // short grace period to keep the test fast
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(proc.is_running());
    EXPECT_EQ(exit_code, -2) << "expected the force-kill sentinel, not a normal exit code";
    // Bounded: the 500ms grace period plus a small margin for the SIGKILL to actually land —
    // nowhere near the 30s the child would otherwise have slept for.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 3000);
}

TEST(ChildProcessTest, DestructorEscalatesToForceKillWhenChildIgnoresSigtermWithoutHanging) {
    // Same regression as above, but through the destructor's own implicit cleanup path rather
    // than an explicit stop_and_wait() call — this is what actually protects trainer_service's
    // own shutdown from a wedged child if the main loop's bounded stop_and_wait() call were ever
    // bypassed (e.g. a future code path that just lets a ChildProcess go out of scope directly).
    const auto start = std::chrono::steady_clock::now();
    {
        ChildProcess proc;
        ASSERT_TRUE(proc.start({"/bin/sh", "-c", "trap '' TERM; sleep 30"}));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }  // ~ChildProcess() must escalate to SIGKILL here, not block for the full 30s sleep.
    const auto elapsed = std::chrono::steady_clock::now() - start;

    // The destructor uses a 5s grace period before escalating — bound generously above that
    // (10s) rather than tightly, to avoid a flaky test on a loaded CI machine.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 10);
}

#endif  // !_WIN32
