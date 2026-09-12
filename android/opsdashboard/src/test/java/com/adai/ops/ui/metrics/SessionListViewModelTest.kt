package com.adai.ops.ui.metrics

import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.SessionSummaryDto
import com.adai.ops.network.dto.SessionsResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** TD-048: SessionListViewModel had zero test coverage. */
@OptIn(ExperimentalCoroutinesApi::class)
class SessionListViewModelTest {

    @Test
    fun `a successful refresh populates sessions and clears any error`() = runTest {
        val fakeService = FakeMetricsApiService(
            listSessionsResponse = {
                SessionsResponseDto(sessions = listOf(SessionSummaryDto(key = "session-1", is_training = true)))
            },
        )
        val viewModel = SessionListViewModel(
            MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        val pollJob = backgroundScope.launch { viewModel.pollSessions() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals(listOf(SessionSummaryDto(key = "session-1", is_training = true)), state.sessions)
        assertFalse(state.isLoading)
        assertNull(state.error)
        pollJob.cancel()
    }

    // Deliberately stale-while-error: a transient failure must surface as an error without
    // discarding whatever sessions the previous successful refresh already loaded — otherwise
    // the whole dashboard would flash empty on every single dropped poll.
    @Test
    fun `a failed refresh keeps the previous sessions and surfaces the error`() = runTest {
        var shouldFail = false
        val fakeService = FakeMetricsApiService(
            listSessionsResponse = {
                if (shouldFail) throw IOException("connection refused")
                SessionsResponseDto(sessions = listOf(SessionSummaryDto(key = "session-1")))
            },
        )
        val viewModel = SessionListViewModel(
            MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        val pollJob = backgroundScope.launch { viewModel.pollSessions() }
        testScheduler.runCurrent()
        assertEquals(1, viewModel.uiState.value.sessions.size)

        shouldFail = true
        testScheduler.advanceTimeBy(6000L) // past LIST_POLL_INTERVAL_MS (5000ms) to trigger another poll
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals("connection refused", state.error)
        assertEquals(
            "a failed poll must not wipe out data from the previous successful one",
            listOf(SessionSummaryDto(key = "session-1")),
            state.sessions,
        )
        pollJob.cancel()
    }

    @Test
    fun `toggleShowIdle flips the flag on each call`() = runTest {
        val viewModel = SessionListViewModel(
            MetricsRepository(FakeApiClientProvider(FakeMetricsApiService()), FakeSettingsRepository()),
        )

        assertFalse(viewModel.uiState.value.showIdle)
        viewModel.toggleShowIdle()
        assertTrue(viewModel.uiState.value.showIdle)
        viewModel.toggleShowIdle()
        assertFalse(viewModel.uiState.value.showIdle)
    }
}
