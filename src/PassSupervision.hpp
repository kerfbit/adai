#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-11

// TD-569: trainer_service's wait for one training pass, moved out of TrainerServiceMain.cpp's loop
// so it can be tested against real child processes. It ends when the pass exits, when a stop is
// requested (SIGTERM/SIGINT to trainer_service), or when the pass has made no progress for the
// stall limit (PassHeartbeat.hpp). The last two stop the child with ChildProcess::stop_and_wait():
// SIGTERM, then a kill after the graceful timeout, signalling only that child's own pid.

#include <chrono>
#include <string>

#include "ChildProcess.hpp"
#include "TrainerServiceControlState.hpp"

namespace adai {

struct PassWaitOptions {
    std::string heartbeat_path;                          // empty: no stall check
    std::chrono::seconds stall_limit{0};                 // 0: no stall check
    int graceful_stop_timeout_ms = 25000;                // before stop_and_wait() kills the child
    std::chrono::milliseconds poll_interval{200};        // how often the child's exit is checked
    std::chrono::milliseconds heartbeat_interval{2000};  // how often the heartbeat file is read
};

struct PassWaitResult {
    int exit_code = 1;
    bool stopped_for_shutdown = false;  // a stop was requested; the child was stopped
    bool stalled = false;               // no progress for the stall limit; the child was stopped
    long long seconds_without_progress = 0;  // when stalled
};

/// Waits for the pass @p child is running (already started) to end, as described above. Keeps
/// @p control's `seconds_since_pass_progress` current while it waits (-1 once the pass has ended).
PassWaitResult wait_for_pass(ChildProcess& child, TrainerServiceControlState& control,
                             const PassWaitOptions& options);

}  // namespace adai
