package com.adai.ops.polling

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import kotlin.math.abs
import kotlin.math.roundToLong
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Result of one poll tick, reported by the caller after hitting the metrics endpoints. */
sealed interface PollOutcome {
    /** [samplesPerSecond] drives adaptive re-tuning; null/non-positive leaves the interval alone. */
    data class Success(val samplesPerSecond: Double? = null) : PollOutcome
    /** The session was evicted server-side (404) — the caller should return to the picker. */
    data object NotFound : PollOutcome
    data class Failure(val message: String) : PollOutcome
}

enum class PollerPhase { LIVE, RECONNECTING, OFFLINE, EVICTED }

data class PollerState(
    val intervalMs: Long,
    val retryCount: Int = 0,
    val phase: PollerPhase = PollerPhase.LIVE,
)

/**
 * Ports the Tizen dashboard's poll()/adaptPollInterval() state machine
 * (tizen-metrics-app/js/app.js) for the Metrics session-detail screen:
 *  - retunes the interval toward 1000/samples_per_second, clamped [200ms, 5000ms],
 *    only when the change exceeds a 10% hysteresis band (avoids interval thrash).
 *  - after MAX_RETRY_BEFORE_SLOWPOLL consecutive failures, forces a fixed 5s
 *    "offline" poll interval rather than continuing to retry at the tuned rate.
 *  - a successful poll following any failures resets the interval to the
 *    configured base interval before re-tuning, rather than resuming mid-backoff.
 *  - a 404 (session evicted) is reported via state so the caller can navigate away;
 *    it does not clear any previously observed data.
 *
 * [onOutcome] is a pure state transition (no coroutines) so the state machine itself
 * is trivial to unit test; [run] is a thin loop wiring it to a real poll function.
 */
class AdaptivePoller(private val baseIntervalMs: Long) {

    private val _state = MutableStateFlow(PollerState(intervalMs = baseIntervalMs))
    val state: StateFlow<PollerState> = _state.asStateFlow()

    fun onOutcome(outcome: PollOutcome): PollerState {
        val current = _state.value
        val newState = when (outcome) {
            is PollOutcome.Success -> {
                val recovering = current.retryCount > 0
                val baseline = if (recovering) baseIntervalMs else current.intervalMs
                val sps = outcome.samplesPerSecond
                val tuned = if (sps != null && sps > 0.0) {
                    val computed = (1000.0 / sps).roundToLong().coerceIn(MIN_INTERVAL_MS, MAX_INTERVAL_MS)
                    val diffRatio = abs(computed - baseline).toDouble() / baseline
                    if (diffRatio > HYSTERESIS_RATIO) computed else baseline
                } else {
                    baseline
                }
                current.copy(intervalMs = tuned, retryCount = 0, phase = PollerPhase.LIVE)
            }
            is PollOutcome.NotFound -> current.copy(phase = PollerPhase.EVICTED)
            is PollOutcome.Failure -> {
                val retryCount = current.retryCount + 1
                if (retryCount < MAX_RETRY_BEFORE_SLOWPOLL) {
                    current.copy(retryCount = retryCount, phase = PollerPhase.RECONNECTING)
                } else {
                    current.copy(
                        intervalMs = SLOW_POLL_INTERVAL_MS,
                        retryCount = retryCount,
                        phase = PollerPhase.OFFLINE,
                    )
                }
            }
        }
        _state.value = newState
        return newState
    }

    /** Loops until [poll] reports [PollOutcome.NotFound] or the calling coroutine is cancelled. */
    suspend fun run(poll: suspend () -> PollOutcome) {
        while (true) {
            val outcome = poll()
            val newState = onOutcome(outcome)
            if (newState.phase == PollerPhase.EVICTED) return
            delay(newState.intervalMs)
        }
    }

    companion object {
        const val MIN_INTERVAL_MS = 200L
        const val MAX_INTERVAL_MS = 5000L
        const val SLOW_POLL_INTERVAL_MS = 5000L
        const val MAX_RETRY_BEFORE_SLOWPOLL = 10
        const val HYSTERESIS_RATIO = 0.10
    }
}
