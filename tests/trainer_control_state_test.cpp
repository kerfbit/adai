/**
 * @file trainer_control_state_test.cpp
 * @brief Unit tests for TrainerControlState — the in-process shared state
 *        between `incremental_trainer serve`'s supervisory loop / active
 *        training pass and TrainerAdminAPI's HTTP handlers.
 *
 * No httplib/HTTP involved here at all — these tests exercise only the
 * struct itself (defaults, atomics, identity accessors, the wake()/
 * interruptible_sleep() primitive the supervisory loop's idle poll is built
 * on) so they build and run unconditionally, unlike trainer_admin_api_test.cpp
 * which is gated on HTTPLIB_INCLUDE_DIR.
 */

#include "TrainerControlState.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using adai::TrainerControlState;
using adai::TrainerLogLevel;
using adai::TrainerPhase;

TEST(TrainerControlStateTest, DefaultsAreSaneAndIdle) {
    TrainerControlState state;
    EXPECT_EQ(state.phase.load(), TrainerPhase::Idle);
    EXPECT_EQ(state.current_epoch.load(), 0);
    EXPECT_EQ(state.total_epochs.load(), 0);
    EXPECT_EQ(state.samples_trained_this_pass.load(), 0);
    EXPECT_DOUBLE_EQ(state.last_loss.load(), 0.0);
    EXPECT_DOUBLE_EQ(state.best_loss.load(), 0.0);
    EXPECT_FALSE(state.paused.load());
    EXPECT_FALSE(state.checkpoint_requested.load());
    EXPECT_EQ(state.checkpoints_written.load(), 0);
    EXPECT_EQ(state.last_checkpoint_time_unix.load(), 0);
    EXPECT_TRUE(state.auto_save_enabled.load());
    EXPECT_EQ(state.auto_save_every_samples.load(), 1000);
    EXPECT_EQ(state.auto_save_every_minutes.load(), 30);
    EXPECT_EQ(state.max_sessions_to_keep.load(), 50);
    EXPECT_EQ(state.get_run_id(), "");
    EXPECT_EQ(state.get_session_id(), "");
    EXPECT_EQ(state.get_model_name(), "");
    EXPECT_EQ(state.get_last_checkpoint_path(), "");
}

TEST(TrainerControlStateTest, ToStringCoversEveryPhase) {
    EXPECT_STREQ(adai::to_string(TrainerPhase::Idle), "idle");
    EXPECT_STREQ(adai::to_string(TrainerPhase::LoadingData), "loading_data");
    EXPECT_STREQ(adai::to_string(TrainerPhase::Tokenizing), "tokenizing");
    EXPECT_STREQ(adai::to_string(TrainerPhase::Training), "training");
    EXPECT_STREQ(adai::to_string(TrainerPhase::Checkpointing), "checkpointing");
    EXPECT_STREQ(adai::to_string(TrainerPhase::Pausing), "pausing");
}

TEST(TrainerControlStateTest, IdentitySettersAndGettersRoundTrip) {
    TrainerControlState state;
    state.set_run_id("run-03");
    state.set_session_id("session-07");
    state.set_model_name("ambitious-aardvark");
    state.set_last_checkpoint_path("/opt/adai/training_sessions/auto_save_session_3.bin");

    EXPECT_EQ(state.get_run_id(), "run-03");
    EXPECT_EQ(state.get_session_id(), "session-07");
    EXPECT_EQ(state.get_model_name(), "ambitious-aardvark");
    EXPECT_EQ(state.get_last_checkpoint_path(),
             "/opt/adai/training_sessions/auto_save_session_3.bin");

    // Overwriting replaces, not appends.
    state.set_run_id("run-04");
    EXPECT_EQ(state.get_run_id(), "run-04");
}

TEST(TrainerControlStateTest, PauseIsASimpleFlagUsableAsAnAbortPointer) {
    // The exact usage pattern IncrementalTrainer::run_training() relies on:
    // &state.paused is handed directly to ChatbotTrainer::set_abort_flag(),
    // so it must be a plain std::atomic<bool> the trainer can load() through
    // a const pointer without any TrainerControlState-specific API.
    TrainerControlState state;
    const std::atomic<bool>* abort_flag = &state.paused;
    EXPECT_FALSE(abort_flag->load());
    state.paused = true;
    EXPECT_TRUE(abort_flag->load());
}

TEST(TrainerControlStateTest, CheckpointRequestedIsConsumedOnceViaExchange) {
    TrainerControlState state;
    state.checkpoint_requested = true;
    EXPECT_TRUE(state.checkpoint_requested.exchange(false));
    // Second exchange sees it already cleared — proves it's consume-once,
    // not a level that would fire a checkpoint on every subsequent sample.
    EXPECT_FALSE(state.checkpoint_requested.exchange(false));
    EXPECT_FALSE(state.checkpoint_requested.load());
}

TEST(TrainerControlStateTest, InterruptibleSleepReturnsEarlyOnWake) {
    TrainerControlState state;
    std::atomic<bool> awoke{false};

    std::thread sleeper([&] {
        state.interruptible_sleep(30);  // would otherwise block for 30s
        awoke = true;
    });

    // Give the sleeper a moment to actually enter the wait before waking it,
    // so this isn't just racing a wake() that landed before wait_for() began
    // (wait_for still catches that case via its own internal check, but this
    // makes the test's intent — "wake interrupts an in-progress sleep" — the
    // thing actually being exercised).
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(awoke.load());

    const auto t0 = std::chrono::steady_clock::now();
    state.wake();
    sleeper.join();
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    EXPECT_TRUE(awoke.load());
    // Should return in well under the 30s sleep duration — a couple of
    // seconds of slack covers CI/build-machine scheduling jitter.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5);
}

TEST(TrainerControlStateTest, InterruptibleSleepWithoutWakeRunsTheFullDuration) {
    TrainerControlState state;
    const auto t0 = std::chrono::steady_clock::now();
    state.interruptible_sleep(1);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    EXPECT_GE(elapsed_ms, 900);  // small slack under 1000ms for scheduling jitter
}

TEST(TrainerControlStateTest, ToStringCoversEveryLogLevel) {
    EXPECT_STREQ(adai::to_string(TrainerLogLevel::Debug), "debug");
    EXPECT_STREQ(adai::to_string(TrainerLogLevel::Info), "info");
    EXPECT_STREQ(adai::to_string(TrainerLogLevel::Warn), "warn");
    EXPECT_STREQ(adai::to_string(TrainerLogLevel::Error), "error");
}

TEST(TrainerControlStateTest, LogAppendsToRecentLogsInOrderWithIncreasingIds) {
    TrainerControlState state;
    EXPECT_TRUE(state.recent_logs().empty());

    state.log(TrainerLogLevel::Info, "first message");
    state.log(TrainerLogLevel::Warn, "second message");
    state.log(TrainerLogLevel::Error, "third message");

    const auto entries = state.recent_logs();
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].message, "first message");
    EXPECT_EQ(entries[0].level, TrainerLogLevel::Info);
    EXPECT_EQ(entries[1].message, "second message");
    EXPECT_EQ(entries[1].level, TrainerLogLevel::Warn);
    EXPECT_EQ(entries[2].message, "third message");
    EXPECT_EQ(entries[2].level, TrainerLogLevel::Error);
    // Monotonically increasing, never reused — see TrainerLogEntry's doc comment.
    EXPECT_LT(entries[0].id, entries[1].id);
    EXPECT_LT(entries[1].id, entries[2].id);
    // A real wall-clock timestamp was stamped, not left at the struct default.
    EXPECT_GT(entries[0].timestamp_unix_ms, 0);
}

TEST(TrainerControlStateTest, LogEvictsOldestEntriesPastCapacity) {
    TrainerControlState state;
    // Capacity is 200 (kMaxLogEntries) — push well past it and confirm the
    // buffer stays capped and keeps only the most recent entries.
    for (int i = 0; i < 250; ++i) {
        state.log(TrainerLogLevel::Info, "message " + std::to_string(i));
    }

    const auto entries = state.recent_logs();
    EXPECT_EQ(entries.size(), 200u);
    EXPECT_EQ(entries.front().message, "message 50");   // oldest 50 evicted
    EXPECT_EQ(entries.back().message, "message 249");
}

TEST(TrainerControlStateTest, RecentLogsHonorsAnExplicitSmallerLimit) {
    TrainerControlState state;
    state.log(TrainerLogLevel::Info, "a");
    state.log(TrainerLogLevel::Info, "b");
    state.log(TrainerLogLevel::Info, "c");

    const auto entries = state.recent_logs(2);
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_EQ(entries[0].message, "b");
    EXPECT_EQ(entries[1].message, "c");
}
