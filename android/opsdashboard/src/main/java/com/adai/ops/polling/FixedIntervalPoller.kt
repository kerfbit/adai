package com.adai.ops.polling

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import kotlinx.coroutines.delay

/**
 * Simple fixed-interval poller for Models and Registry list/detail screens — those
 * change on the order of minutes, not seconds, so the adaptive backoff state machine
 * in [AdaptivePoller] would be over-engineering here. [poll] is expected to catch its
 * own errors and update its own UI-facing error/staleness state.
 */
class FixedIntervalPoller(private val intervalMs: Long) {
    suspend fun run(poll: suspend () -> Unit) {
        while (true) {
            poll()
            delay(intervalMs)
        }
    }
}
