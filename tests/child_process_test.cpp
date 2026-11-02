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
#include <atomic>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <unistd.h>

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

#ifndef _WIN32
// TD-278: a missing or non-executable binary used to make start() return true; the failure only showed up
// later as exit code 127, which trainer_service counted as a crashed pass and retried indefinitely.
TEST(ChildProcessTest, AFailedExecIsReportedByStart) {
    adai::ChildProcess missing;
    EXPECT_FALSE(missing.start({"/nonexistent/incremental_trainer"}));
    EXPECT_FALSE(missing.is_running());
    EXPECT_EQ(missing.pid(), 0);

    const std::string not_executable =
        (std::filesystem::temp_directory_path() / ("adai_td278_" + std::to_string(::getpid()))).string();
    std::ofstream(not_executable) << "#!/bin/sh\nexit 0\n";
    adai::ChildProcess denied;
    EXPECT_FALSE(denied.start({not_executable}));
    std::filesystem::remove(not_executable);

    // A real launch still works on the same kind of object afterwards.
    EXPECT_TRUE(missing.start({"/bin/true"}));
    int code = -1;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!missing.poll_exit(&code) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(code, 0);
}
#endif

#ifndef _WIN32
// TD-277: every signal death came back as 128, so a SIGSEGV (ai-machine's driver crash) and a SIGKILL (an OOM
// kill) were indistinguishable in trainer_service's last_exit_code. It's now 128 + the signal number.
TEST(ChildProcessTest, ASignalDeathIs128PlusTheSignalNumber) {
    for (const auto& [sig, expected] : std::vector<std::pair<std::string, int>>{{"SEGV", 139}, {"KILL", 137}}) {
        ChildProcess child;
        ASSERT_TRUE(child.start({"/bin/sh", "-c", "kill -" + sig + " $$"}));
        int code = -1;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!child.poll_exit(&code) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        EXPECT_EQ(code, expected) << "SIG" << sig;
    }
}
#endif

#ifndef _WIN32
// TD-279: request_stop() read a plain running_/pid_ pair, and the give-up path cleared pid_ before running_, so
// a concurrent call could send kill(-1, SIGTERM) to every process the user can signal. The members are atomics,
// running_ is cleared first and only a pid > 0 is signalled.
//
// Deliberately no concurrent stress test: on a regression it would send SIGTERM to every process the user
// owns, which ends the desktop session (it did, twice, against the old code). These checks only exercise the
// stopped states, where a missing guard can't reach kill() with -1.
TEST(ChildProcessTest, RequestStopIsANoOpWithoutALiveChild) {
    ChildProcess idle;
    idle.request_stop();  // never started
    EXPECT_FALSE(idle.stop_and_wait(10, nullptr));
    EXPECT_EQ(idle.pid(), 0);

    ChildProcess finished;
    ASSERT_TRUE(finished.start({"/bin/true"}));
    int code = -1;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!finished.poll_exit(&code) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(code, 0);
    EXPECT_EQ(finished.pid(), 0);
    finished.request_stop();  // after the child is reaped
    EXPECT_FALSE(finished.stop_and_wait(10, nullptr));
}
#endif

#ifdef __linux__
// TD-280: a pass outlived a hard-killed supervisor. The child now asks for SIGTERM when its parent dies
// (PR_SET_PDEATHSIG). A disposable "supervisor" process forked by this test starts the child and is then
// SIGKILLed; the child must go too. Signals go only to that supervisor and, as cleanup, to the child, each
// a checked pid above 0 (see TD-279 for why nothing here may ever signal anything else).
TEST(ChildProcessTest, TheChildDiesWithAHardKilledSupervisor) {
    int fds[2];
    ASSERT_EQ(::pipe(fds), 0);
    const pid_t supervisor = ::fork();
    ASSERT_GE(supervisor, 0);
    if (supervisor == 0) {
        ::close(fds[0]);
        ChildProcess child;
        long long pid = child.start({"/bin/sleep", "30"}) ? child.pid() : 0;
        ssize_t ignored = ::write(fds[1], &pid, sizeof(pid));
        (void)ignored;
        for (;;) ::pause();  // until SIGKILLed
    }
    ::close(fds[1]);
    long long child_pid = 0;
    ASSERT_EQ(::read(fds[0], &child_pid, sizeof(child_pid)), static_cast<ssize_t>(sizeof(child_pid)));
    ::close(fds[0]);
    ASSERT_GT(child_pid, 0);
    ASSERT_GT(supervisor, 0);

    const auto alive = [](long long pid) {  // a zombie counts as gone
        std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
        std::string line;
        if (!std::getline(stat, line)) return false;
        const auto close_paren = line.rfind(')');
        return close_paren != std::string::npos && close_paren + 2 < line.size() && line[close_paren + 2] != 'Z';
    };
    ASSERT_TRUE(alive(child_pid));

    ::kill(supervisor, SIGKILL);
    ::waitpid(supervisor, nullptr, 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (alive(child_pid) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const bool survived = alive(child_pid);
    if (survived) ::kill(static_cast<pid_t>(child_pid), SIGKILL);  // don't leave it behind
    EXPECT_FALSE(survived) << "the pass outlived its hard-killed supervisor";
}
#endif
