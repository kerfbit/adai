/**
 * @file pass_heartbeat_test.cpp
 * @brief TD-569: the training pass's progress heartbeat (PassHeartbeat) and trainer_service's wait
 * for a pass with its stall check (PassSupervision::wait_for_pass()).
 *
 * The wait tests run real child processes (/bin/sh -c "..."), like child_process_test.cpp. Each
 * child writes the heartbeat file a real pass would, with its own pid. Children are stopped only
 * through ChildProcess::stop_and_wait(), which signals that child's own pid and nothing else.
 */

#include "PassHeartbeat.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>

#ifndef _WIN32
#include <unistd.h>

#include "ChildProcess.hpp"
#include "PassSupervision.hpp"
#include "TrainerServiceControlState.hpp"
#endif

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using adai::PassHeartbeat;
using adai::PassStallWatch;

namespace {

class TempDir {
   public:
    TempDir() {
        std::random_device rd;
        path_ = fs::temp_directory_path() /
                ("adai_pass_heartbeat_" + std::to_string(rd()) + std::to_string(rd()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    std::string file(const std::string& name) const {
        return (path_ / name).string();
    }

   private:
    fs::path path_;
};

void write_text(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::trunc) << text;
}

long long own_pid() {
#ifdef _WIN32
    return 0;
#else
    return static_cast<long long>(getpid());
#endif
}

}  // namespace

TEST(PassHeartbeatTest, NoteCountsEveryCall) {
    const auto before = adai::pass_progress_count();
    adai::note_pass_progress();
    adai::note_pass_progress();
    EXPECT_EQ(adai::pass_progress_count(), before + 2);
}

TEST(PassHeartbeatTest, WriteThenReadRoundTrips) {
    TempDir dir;
    const std::string path = dir.file("hb");
    ASSERT_TRUE(adai::write_pass_heartbeat(path, {1234, 98765432109ULL}));
    const auto read = adai::read_pass_heartbeat(path);
    ASSERT_TRUE(read.has_value());
    EXPECT_EQ(read->pid, 1234);
    EXPECT_EQ(read->count, 98765432109ULL);
    EXPECT_FALSE(fs::exists(path + ".tmp"));
}

TEST(PassHeartbeatTest, AMissingOrMalformedFileReadsAsNothing) {
    TempDir dir;
    EXPECT_FALSE(adai::read_pass_heartbeat(dir.file("missing")).has_value());
    for (const std::string text :
         {"", "1234", "abc 5\n", "0 5\n", "-3 5\n", "12 5 extra\n", "12 x\n"}) {
        write_text(dir.file("hb"), text);
        EXPECT_FALSE(adai::read_pass_heartbeat(dir.file("hb")).has_value()) << "'" << text << "'";
    }
}

TEST(PassHeartbeatTest, TheWriterReportsThisProcessAndFollowsTheCount) {
    TempDir dir;
    const std::string path = dir.file("sub/hb");  // the writer creates the directory
    {
        adai::PassHeartbeatWriter writer(path, 20ms);
        std::optional<PassHeartbeat> first;
        for (int i = 0; i < 100 && !first; ++i) {
            std::this_thread::sleep_for(10ms);
            first = adai::read_pass_heartbeat(path);
        }
        ASSERT_TRUE(first.has_value());
        EXPECT_EQ(first->pid, own_pid());

        adai::note_pass_progress();
        const auto expected = adai::pass_progress_count();
        bool followed = false;
        for (int i = 0; i < 100 && !followed; ++i) {
            std::this_thread::sleep_for(10ms);
            const auto now = adai::read_pass_heartbeat(path);
            followed = now && now->count == expected;
        }
        EXPECT_TRUE(followed);
    }
    EXPECT_FALSE(fs::exists(path)) << "the writer removes its file when it stops";
}

TEST(PassStallWatchTest, StalledOnlyAfterTheLimitWithoutAChange) {
    PassStallWatch watch(10s);
    const auto t0 = PassStallWatch::Clock::now();
    watch.start(42, t0);
    EXPECT_FALSE(watch.observe(PassHeartbeat{42, 1}, t0 + 1s));
    EXPECT_FALSE(
        watch.observe(PassHeartbeat{42, 1}, t0 + 11s));  // 10 s since the change at t0 + 1 s
    EXPECT_TRUE(watch.observe(PassHeartbeat{42, 1}, t0 + 12s));
    EXPECT_EQ(watch.since_progress(t0 + 12s), 11s);
}

TEST(PassStallWatchTest, AChangedCountRestartsTheClock) {
    PassStallWatch watch(10s);
    const auto t0 = PassStallWatch::Clock::now();
    watch.start(42, t0);
    EXPECT_FALSE(watch.observe(PassHeartbeat{42, 1}, t0 + 8s));
    EXPECT_FALSE(watch.observe(PassHeartbeat{42, 2}, t0 + 16s));
    EXPECT_FALSE(watch.observe(PassHeartbeat{42, 3}, t0 + 24s));
    EXPECT_FALSE(watch.observe(PassHeartbeat{42, 3}, t0 + 34s));
    EXPECT_TRUE(watch.observe(PassHeartbeat{42, 3}, t0 + 35s));
}

TEST(PassStallWatchTest, AHeartbeatFromAnotherProcessIsNotProgress) {
    // An earlier pass's file, still there when the next pass starts, mustn't keep a hung pass
    // alive.
    PassStallWatch watch(10s);
    const auto t0 = PassStallWatch::Clock::now();
    watch.start(42, t0);
    EXPECT_FALSE(watch.observe(PassHeartbeat{41, 1}, t0 + 5s));
    EXPECT_FALSE(watch.observe(PassHeartbeat{41, 2}, t0 + 10s));
    EXPECT_TRUE(watch.observe(PassHeartbeat{41, 3}, t0 + 11s));
}

TEST(PassStallWatchTest, NoHeartbeatAtAllStallsFromTheStart) {
    PassStallWatch watch(10s);
    const auto t0 = PassStallWatch::Clock::now();
    watch.start(42, t0);
    EXPECT_FALSE(watch.observe(std::nullopt, t0 + 10s));
    EXPECT_TRUE(watch.observe(std::nullopt, t0 + 11s));
}

TEST(PassStallWatchTest, AZeroLimitNeverStalls) {
    PassStallWatch watch(0s);
    const auto t0 = PassStallWatch::Clock::now();
    watch.start(42, t0);
    EXPECT_FALSE(watch.observe(std::nullopt, t0 + 1000h));
}

#ifndef _WIN32

namespace {

// A child that writes one heartbeat, as a pass that hangs after starting would, then waits.
std::string beat_once_then_hang(const std::string& hb) {
    return "printf '%s 1\\n' $$ > '" + hb + ".tmp' && mv '" + hb + ".tmp' '" + hb +
           "'; exec sleep 30";
}

// A child that makes progress every 0.1 s for @p beats beats, then exits 0.
std::string beat_then_exit(const std::string& hb, int beats) {
    return "i=0; while [ $i -lt " + std::to_string(beats) +
           " ]; do i=$((i+1)); printf '%s %s\\n' $$ $i > '" + hb + ".tmp' && mv '" + hb +
           ".tmp' '" + hb + "'; sleep 0.1; done; exit 0";
}

adai::PassWaitOptions fast_options(const std::string& hb) {
    adai::PassWaitOptions options;
    options.heartbeat_path = hb;
    options.stall_limit = 1s;
    options.graceful_stop_timeout_ms = 2000;
    options.poll_interval = 20ms;
    options.heartbeat_interval = 100ms;
    return options;
}

}  // namespace

TEST(PassSupervisionTest, APassThatStopsMakingProgressIsStopped) {
    TempDir dir;
    const std::string hb = dir.file("hb");
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    ASSERT_TRUE(child.start({"/bin/sh", "-c", beat_once_then_hang(hb)}));

    const auto started = std::chrono::steady_clock::now();
    const adai::PassWaitResult result = adai::wait_for_pass(child, control, fast_options(hb));
    const auto took = std::chrono::steady_clock::now() - started;

    EXPECT_TRUE(result.stalled);
    EXPECT_FALSE(result.stopped_for_shutdown);
    EXPECT_GE(result.seconds_without_progress, 1);
    EXPECT_EQ(result.exit_code, 128 + 15) << "stopped with SIGTERM";
    EXPECT_GE(took, 1s);
    EXPECT_LT(took, 5s);
    EXPECT_FALSE(child.is_running());
    EXPECT_EQ(control.seconds_since_pass_progress.load(), -1);
}

TEST(PassSupervisionTest, APassThatKeepsMakingProgressRunsPastTheLimit) {
    TempDir dir;
    const std::string hb = dir.file("hb");
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    ASSERT_TRUE(
        child.start({"/bin/sh", "-c", beat_then_exit(hb, 25)}));  // about 2.5 s, past the 1 s limit

    const adai::PassWaitResult result = adai::wait_for_pass(child, control, fast_options(hb));

    EXPECT_FALSE(result.stalled);
    EXPECT_FALSE(result.stopped_for_shutdown);
    EXPECT_EQ(result.exit_code, 0);
}

TEST(PassSupervisionTest, AHungPassThatIgnoresSigtermIsKilled) {
    TempDir dir;
    const std::string hb = dir.file("hb");
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    // A GPU call stuck in the driver doesn't get to its SIGTERM handling either.
    ASSERT_TRUE(child.start(
        {"/bin/sh", "-c",
         "trap '' TERM; printf '%s 1\\n' $$ > '" + hb + "'; while :; do sleep 0.1; done"}));

    adai::PassWaitOptions options = fast_options(hb);
    options.graceful_stop_timeout_ms = 500;
    const adai::PassWaitResult result = adai::wait_for_pass(child, control, options);

    EXPECT_TRUE(result.stalled);
    EXPECT_EQ(result.exit_code, -2) << "force-killed by stop_and_wait()";
    EXPECT_FALSE(child.is_running());
}

TEST(PassSupervisionTest, AnEarlierPassesHeartbeatDoesNotKeepAHungPassAlive) {
    TempDir dir;
    const std::string hb = dir.file("hb");
    ASSERT_TRUE(adai::write_pass_heartbeat(hb, {1, 7}));  // pid 1: never this child
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    ASSERT_TRUE(child.start({"/bin/sh", "-c", "exec sleep 30"}));  // writes no heartbeat at all

    const adai::PassWaitResult result = adai::wait_for_pass(child, control, fast_options(hb));

    EXPECT_TRUE(result.stalled);
}

TEST(PassSupervisionTest, WithoutAHeartbeatFileThereIsNoStallCheck) {
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    ASSERT_TRUE(child.start({"/bin/sh", "-c", "sleep 1.5; exit 4"}));

    adai::PassWaitOptions options = fast_options("");
    const adai::PassWaitResult result = adai::wait_for_pass(child, control, options);

    EXPECT_FALSE(result.stalled);
    EXPECT_EQ(result.exit_code, 4);
}

TEST(PassSupervisionTest, AStopRequestStopsThePass) {
    TempDir dir;
    const std::string hb = dir.file("hb");
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    ASSERT_TRUE(child.start({"/bin/sh", "-c", beat_then_exit(hb, 300)}));

    std::thread stopper([&control] {
        std::this_thread::sleep_for(300ms);
        control.stop_requested = true;
    });
    const adai::PassWaitResult result = adai::wait_for_pass(child, control, fast_options(hb));
    stopper.join();

    EXPECT_TRUE(result.stopped_for_shutdown);
    EXPECT_FALSE(result.stalled);
    EXPECT_FALSE(child.is_running());
}

TEST(PassSupervisionTest, ReportsTheTimeSinceProgressWhileWaiting) {
    TempDir dir;
    const std::string hb = dir.file("hb");
    adai::ChildProcess child;
    adai::TrainerServiceControlState control;
    ASSERT_TRUE(child.start({"/bin/sh", "-c", beat_once_then_hang(hb)}));

    adai::PassWaitOptions options = fast_options(hb);
    options.stall_limit = 3s;
    long long seen_max = -1;
    std::atomic<bool> done{false};
    std::thread observer([&] {
        while (!done) {
            seen_max = std::max(seen_max, control.seconds_since_pass_progress.load());
            std::this_thread::sleep_for(50ms);
        }
    });
    const adai::PassWaitResult result = adai::wait_for_pass(child, control, options);
    done = true;
    observer.join();

    EXPECT_TRUE(result.stalled);
    EXPECT_GE(seen_max, 2);
    EXPECT_EQ(control.seconds_since_pass_progress.load(), -1);
}

#endif  // !_WIN32
