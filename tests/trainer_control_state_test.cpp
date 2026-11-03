/**
 * @file trainer_control_state_test.cpp
 * @brief Unit tests for TrainerControlState — the in-process shared state
 *        between one training pass and its TrainerAdminAPI's HTTP handlers.
 *
 * No httplib/HTTP involved here at all — these tests exercise only the
 * struct itself (defaults, atomics, identity accessors, phase transitions,
 * the log ring) so they build and run unconditionally, unlike trainer_admin_api_test.cpp
 * which is gated on HTTPLIB_INCLUDE_DIR.
 */

#include "TrainerControlState.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "Logger.hpp"

using adai::TrainerControlState;
using adai::TrainerLogLevel;
using adai::TrainerPhase;

TEST(TrainerControlStateTest, DefaultsAreSaneAndIdle) {
    TrainerControlState state;
    EXPECT_EQ(state.phase.load(), TrainerPhase::Idle);
    EXPECT_EQ(state.current_epoch.load(), 0);
    EXPECT_EQ(state.total_epochs.load(), 0);
    EXPECT_EQ(state.samples_trained_this_pass.load(), 0);
    // TD-520: no loss yet is NaN (null in /admin/status), not a perfect 0.
    EXPECT_TRUE(std::isnan(state.last_loss.load()));
    EXPECT_TRUE(std::isnan(state.best_loss.load()));
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

// TD-521: requests are numbered; a checkpoint the pass writes serves every request made before it.
TEST(TrainerControlStateTest, CheckpointRequestsAreServedByTicket) {
    TrainerControlState state;
    EXPECT_EQ(state.take_checkpoint_request(), 0);
    const long long first = state.request_checkpoint();
    const long long second = state.request_checkpoint();
    EXPECT_EQ(second, first + 1);
    const long long serving = state.take_checkpoint_request();
    EXPECT_EQ(serving, second);
    EXPECT_EQ(state.take_checkpoint_request(), 0);  // consumed once
    EXPECT_LT(state.checkpoint_requests_served.load(), first);
    state.mark_checkpoint_served(serving);
    EXPECT_GE(state.checkpoint_requests_served.load(), second);
    state.mark_checkpoint_served(first);  // never moves backwards
    EXPECT_EQ(state.checkpoint_requests_served.load(), second);
}

// TD-522: the per-pass state had an unused wake()/interruptible_sleep() copy (nothing sleeps in a
// pass); it's gone, and trainer_service's own TrainerServiceControlState keeps the tested one. A
// pending pause is now reported as `pausing` (it used to be listed but never set).
TEST(TrainerControlStateTest, RequestPauseReportsPausingWhileThePassIsWorking) {
    for (const TrainerPhase working :
         {TrainerPhase::LoadingData, TrainerPhase::Tokenizing, TrainerPhase::Training}) {
        TrainerControlState state;
        state.set_phase(working);
        state.request_pause();
        EXPECT_TRUE(state.paused.load());
        EXPECT_EQ(state.phase.load(), TrainerPhase::Pausing);
        // The pass keeps reporting progress until its next optimizer-step boundary.
        state.set_phase(TrainerPhase::Training);
        EXPECT_EQ(state.phase.load(), TrainerPhase::Pausing);
    }
}

TEST(TrainerControlStateTest, PausingPassReportsCheckpointingThenIdle) {
    TrainerControlState state;
    state.set_phase(TrainerPhase::Training);
    state.request_pause();
    state.set_phase(TrainerPhase::Checkpointing);  // the drain checkpoint
    EXPECT_EQ(state.phase.load(), TrainerPhase::Checkpointing);
    state.set_phase(TrainerPhase::Idle);
    EXPECT_EQ(state.phase.load(), TrainerPhase::Idle);
}

TEST(TrainerControlStateTest, RequestPauseWhileIdleOrCheckpointingKeepsThePhase) {
    TrainerControlState idle;
    idle.request_pause();
    EXPECT_EQ(idle.phase.load(), TrainerPhase::Idle);

    TrainerControlState saving;
    saving.set_phase(TrainerPhase::Checkpointing);
    saving.request_pause();
    EXPECT_EQ(saving.phase.load(), TrainerPhase::Checkpointing);
    saving.set_phase(TrainerPhase::Training);  // back to work after the save: now draining
    EXPECT_EQ(saving.phase.load(), TrainerPhase::Pausing);
}

TEST(TrainerControlStateTest, ClearPauseRestoresTheWorkingPhase) {
    TrainerControlState state;
    state.set_phase(TrainerPhase::Tokenizing);
    state.request_pause();
    state.clear_pause();
    EXPECT_FALSE(state.paused.load());
    EXPECT_EQ(state.phase.load(), TrainerPhase::Tokenizing);
    state.set_phase(TrainerPhase::Training);
    EXPECT_EQ(state.phase.load(), TrainerPhase::Training);
}

// incremental_trainer's SIGTERM/SIGINT handler can only store `paused` (no locking in a signal
// handler); the pass's next phase update reports the drain.
TEST(TrainerControlStateTest, PausedStoredDirectlyIsReportedAtTheNextPhaseUpdate) {
    TrainerControlState state;
    state.set_phase(TrainerPhase::Training);
    state.paused.store(true);
    EXPECT_EQ(state.phase.load(), TrainerPhase::Training);
    state.set_phase(TrainerPhase::Training);
    EXPECT_EQ(state.phase.load(), TrainerPhase::Pausing);
}

// TD-522: a pass that throws used to stay `training`/`loading_data` in /admin/status.
TEST(TrainerControlStateTest, PhaseResetOnThrowSetsIdleOnlyWhenAnExceptionLeavesTheScope) {
    TrainerControlState state;
    try {
        const adai::TrainerPhaseResetOnThrow guard(&state);
        state.set_phase(TrainerPhase::Training);
        throw std::runtime_error("pass failed");
    } catch (const std::runtime_error&) {
    }
    EXPECT_EQ(state.phase.load(), TrainerPhase::Idle);

    {
        const adai::TrainerPhaseResetOnThrow guard(&state);
        state.set_phase(TrainerPhase::Checkpointing);
    }
    EXPECT_EQ(state.phase.load(), TrainerPhase::Checkpointing);  // normal exit: untouched

    // Created inside a catch handler (an exception already in flight): only a new one counts.
    try {
        throw std::runtime_error("outer");
    } catch (const std::runtime_error&) {
        const adai::TrainerPhaseResetOnThrow guard(&state);
        state.set_phase(TrainerPhase::Training);
    }
    EXPECT_EQ(state.phase.load(), TrainerPhase::Training);

    const adai::TrainerPhaseResetOnThrow no_state(nullptr);  // no admin API: a no-op
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

// TD-522: the ring used to keep Debug entries the log file (at LOG_LEVEL) never got.
TEST(TrainerControlStateTest, LogSkipsMessagesBelowTheLoggerLevel) {
    const auto previous = adai::Logger::get()->level();
    adai::Logger::set_level(adai::Logger::Level::INFO);
    TrainerControlState state;
    state.log(TrainerLogLevel::Debug, "hidden");
    state.log(TrainerLogLevel::Info, "shown");
    adai::Logger::set_level(adai::Logger::Level::DEBUG);
    state.log(TrainerLogLevel::Debug, "debug shown");
    adai::Logger::set_level(adai::Logger::Level::ERR);
    state.log(TrainerLogLevel::Warn, "warn hidden");
    state.log(TrainerLogLevel::Error, "error shown");
    adai::Logger::get()->set_level(previous);

    const auto entries = state.recent_logs();
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].message, "shown");
    EXPECT_EQ(entries[1].message, "debug shown");
    EXPECT_EQ(entries[2].message, "error shown");
}

// TD-522: messages were unbounded (a whole exception text or file list per entry, 200 entries).
TEST(TrainerControlStateTest, LogCapsLongMessagesAtACharacterBoundary) {
    TrainerControlState state;
    const std::size_t cap = TrainerControlState::kMaxLogMessageBytes;
    state.log(TrainerLogLevel::Warn, std::string(cap, 'x'));        // exactly the cap: kept whole
    state.log(TrainerLogLevel::Warn, std::string(cap * 3, 'y'));    // cut
    // A two-byte character (é = C3 A9) straddling the cap is dropped whole, not split.
    state.log(TrainerLogLevel::Warn, std::string(cap - 1, 'z') + "\xC3\xA9" + "tail");

    const auto entries = state.recent_logs();
    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].message, std::string(cap, 'x'));
    EXPECT_EQ(entries[1].message, std::string(cap, 'y') + " [truncated]");
    EXPECT_EQ(entries[2].message, std::string(cap - 1, 'z') + " [truncated]");
}
