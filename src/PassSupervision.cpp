// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-11

#include "PassSupervision.hpp"

#include <thread>

#include "Logger.hpp"
#include "PassHeartbeat.hpp"

namespace adai {

PassWaitResult wait_for_pass(ChildProcess& child, TrainerServiceControlState& control,
                             const PassWaitOptions& options) {
    PassWaitResult result;
    const bool watching = !options.heartbeat_path.empty() && options.stall_limit.count() > 0;
    PassStallWatch watch(watching ? options.stall_limit : std::chrono::seconds(0));
    auto now = PassStallWatch::Clock::now();
    watch.start(child.pid(), now);
    auto next_heartbeat_read = now;
    control.seconds_since_pass_progress = 0;

    while (true) {
        if (control.stop_requested.load()) {
            child.stop_and_wait(options.graceful_stop_timeout_ms, &result.exit_code);
            result.stopped_for_shutdown = true;
            break;
        }
        if (watching && (now = PassStallWatch::Clock::now()) >= next_heartbeat_read) {
            next_heartbeat_read = now + options.heartbeat_interval;
            const bool stalled = watch.observe(read_pass_heartbeat(options.heartbeat_path), now);
            control.seconds_since_pass_progress = watch.since_progress(now).count();
            if (stalled) {
                result.stalled = true;
                result.seconds_without_progress = watch.since_progress(now).count();
                Logger::error(
                    "trainer_service: the training pass (pid {}) has made no progress for {}s "
                    "(limit {}s, TRAINER_PASS_STALL_MINUTES) — stopping it; a GPU call that "
                    "never returned after a driver engine reset looks like this (TD-569)",
                    child.pid(), result.seconds_without_progress, options.stall_limit.count());
                child.stop_and_wait(options.graceful_stop_timeout_ms, &result.exit_code);
                break;
            }
        }
        std::this_thread::sleep_for(options.poll_interval);
        if (child.poll_exit(&result.exit_code))
            break;
    }
    control.seconds_since_pass_progress = -1;
    return result;
}

}  // namespace adai
