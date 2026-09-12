package com.adai.ops.ui.metrics

import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.CurrentMetricsDto
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.settings.OpsSettings
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.httpException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * TD-048: SessionDetailViewModel had zero test coverage. AdaptivePoller's own state machine
 * (retry/backoff/interval tuning) is already covered by AdaptivePollerTest.kt — these tests
 * focus on what's unique to this ViewModel: the tickOnce() -> PollOutcome mapping, the
 * eviction event, and the endSession()/dismissActionMessage() actions.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class SessionDetailViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private fun repository(service: FakeMetricsApiService) =
        MetricsRepository(FakeApiClientProvider(service), FakeSettingsRepository(OpsSettings(basePollIntervalMs = 2000L)))

    @Test
    fun `a normal tick populates status and current metrics`() = runTest {
        val service = FakeMetricsApiService(
            sessionStatusResponse = { SessionStatusDto(is_training = true, current_epoch = 3) },
            currentMetricsResponse = { CurrentMetricsDto(current_loss = 0.5) },
        )
        val viewModel = SessionDetailViewModel("session-1", repository(service), FakeSettingsRepository())

        val pollJob = backgroundScope.launch { viewModel.poll() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals(3, state.status?.current_epoch)
        assertEquals(0.5, state.current?.current_loss)
        assertFalse(state.isLoading)
        assertNull(state.error)
        pollJob.cancel()
    }

    @Test
    fun `a 404 on sessionStatus evicts the session and emits the eviction event`() = runTest {
        val service = FakeMetricsApiService(
            sessionStatusResponse = { throw httpException(404) },
        )
        val viewModel = SessionDetailViewModel("session-1", repository(service), FakeSettingsRepository())
        val events = mutableListOf<SessionDetailEvent>()
        backgroundScope.launch { viewModel.events.collect { events += it } }

        val pollJob = backgroundScope.launch { viewModel.poll() }
        testScheduler.runCurrent()

        assertEquals(listOf(SessionDetailEvent.SessionEvicted), events)
        pollJob.cancel()
    }

    @Test
    fun `a network failure on a tick surfaces the error without evicting`() = runTest {
        val service = FakeMetricsApiService(
            sessionStatusResponse = { throw httpException(500, "{\"error\":\"internal error\"}") },
        )
        val viewModel = SessionDetailViewModel("session-1", repository(service), FakeSettingsRepository())
        val events = mutableListOf<SessionDetailEvent>()
        backgroundScope.launch { viewModel.events.collect { events += it } }

        val pollJob = backgroundScope.launch { viewModel.poll() }
        testScheduler.runCurrent()

        assertTrue(viewModel.uiState.value.error!!.contains("internal error"))
        assertTrue("a non-404 failure must not evict the session", events.isEmpty())
        pollJob.cancel()
    }

    @Test
    fun `endSession success reports a confirmation message`() = runTest {
        val service = FakeMetricsApiService(
            sessionStatusResponse = { throw httpException(404) }, // evict immediately, no need to poll further
        )
        val viewModel = SessionDetailViewModel("session-1", repository(service), FakeSettingsRepository())

        viewModel.endSession()
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals("Session ended.", state.actionMessage)
        assertFalse(state.actionInProgress)
        assertEquals(listOf("session-1"), service.endSessionCalls)
    }

    @Test
    fun `endSession failure reports the server's error`() = runTest {
        val service = FakeMetricsApiService(
            endSessionResponse = { throw httpException(409, "{\"error\":\"session already ended\"}") },
        )
        val viewModel = SessionDetailViewModel("session-1", repository(service), FakeSettingsRepository())

        viewModel.endSession()
        testScheduler.runCurrent()

        assertTrue(viewModel.uiState.value.actionMessage!!.contains("session already ended"))
    }

    @Test
    fun `dismissActionMessage clears a previously set message`() = runTest {
        val service = FakeMetricsApiService()
        val viewModel = SessionDetailViewModel("session-1", repository(service), FakeSettingsRepository())
        viewModel.endSession()
        testScheduler.runCurrent()
        assertTrue(viewModel.uiState.value.actionMessage != null)

        viewModel.dismissActionMessage()

        assertNull(viewModel.uiState.value.actionMessage)
    }
}
