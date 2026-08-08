package com.adai.ops.polling

import org.junit.Assert.assertEquals
import org.junit.Test

class AdaptivePollerTest {

    @Test
    fun `initial state uses the configured base interval`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        assertEquals(2000L, poller.state.value.intervalMs)
        assertEquals(0, poller.state.value.retryCount)
        assertEquals(PollerPhase.LIVE, poller.state.value.phase)
    }

    @Test
    fun `success without a samples-per-second signal leaves the interval unchanged`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        val state = poller.onOutcome(PollOutcome.Success(samplesPerSecond = null))
        assertEquals(2000L, state.intervalMs)
        assertEquals(PollerPhase.LIVE, state.phase)
    }

    @Test
    fun `a small interval change under the hysteresis band is ignored`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        // 1000/0.51 ~= 1961ms, within 10% of 2000ms -> should NOT retune
        val state = poller.onOutcome(PollOutcome.Success(samplesPerSecond = 0.51))
        assertEquals(2000L, state.intervalMs)
    }

    @Test
    fun `a large interval change beyond the hysteresis band retunes the interval`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        // 1000/10 = 100ms, clamped to the 200ms floor -> far more than 10% away from 2000ms
        val state = poller.onOutcome(PollOutcome.Success(samplesPerSecond = 10.0))
        assertEquals(AdaptivePoller.MIN_INTERVAL_MS, state.intervalMs)
    }

    @Test
    fun `computed interval is clamped to the max when samples per second is very low`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        val state = poller.onOutcome(PollOutcome.Success(samplesPerSecond = 0.01))
        assertEquals(AdaptivePoller.MAX_INTERVAL_MS, state.intervalMs)
    }

    @Test
    fun `failures below the retry threshold keep the interval and mark reconnecting`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        val state = poller.onOutcome(PollOutcome.Failure("boom"))
        assertEquals(2000L, state.intervalMs)
        assertEquals(1, state.retryCount)
        assertEquals(PollerPhase.RECONNECTING, state.phase)
    }

    @Test
    fun `reaching the retry threshold forces the slow-poll interval and offline phase`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        lateinit var last: PollerState
        repeat(AdaptivePoller.MAX_RETRY_BEFORE_SLOWPOLL) {
            last = poller.onOutcome(PollOutcome.Failure("boom"))
        }
        assertEquals(AdaptivePoller.SLOW_POLL_INTERVAL_MS, last.intervalMs)
        assertEquals(PollerPhase.OFFLINE, last.phase)
        assertEquals(AdaptivePoller.MAX_RETRY_BEFORE_SLOWPOLL, last.retryCount)
    }

    @Test
    fun `a success after slow-poll failures resets to the base interval, not the failed interval`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        repeat(AdaptivePoller.MAX_RETRY_BEFORE_SLOWPOLL) {
            poller.onOutcome(PollOutcome.Failure("boom"))
        }
        val recovered = poller.onOutcome(PollOutcome.Success(samplesPerSecond = null))
        assertEquals(2000L, recovered.intervalMs)
        assertEquals(0, recovered.retryCount)
        assertEquals(PollerPhase.LIVE, recovered.phase)
    }

    @Test
    fun `a 404 marks the poller evicted without altering the interval`() {
        val poller = AdaptivePoller(baseIntervalMs = 2000L)
        val state = poller.onOutcome(PollOutcome.NotFound)
        assertEquals(PollerPhase.EVICTED, state.phase)
        assertEquals(2000L, state.intervalMs)
    }
}
