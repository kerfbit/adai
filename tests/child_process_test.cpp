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

#endif  // !_WIN32
