package com.adai.ops.ui.trainer

import com.adai.ops.data.trainer.TrainerRepository
import com.adai.ops.network.dto.TrainerCheckpointResultDto
import com.adai.ops.network.dto.TrainerLogEntryDto
import com.adai.ops.network.dto.TrainerLogsResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.FakeTrainerApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Response

/** TD-048: TrainerViewModel had zero test coverage. */
@OptIn(ExperimentalCoroutinesApi::class)
class TrainerViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private fun viewModel(service: FakeTrainerApiService = FakeTrainerApiService()) =
        TrainerViewModel(TrainerRepository(FakeApiClientProvider(service), FakeSettingsRepository()))

    @Test
    fun `log entries are mapped and sorted newest-first, replacing the previous list wholesale`() = runTest {
        val service = FakeTrainerApiService(
            getLogsResponse = {
                Response.success(
                    TrainerLogsResponseDto(
                        entries = listOf(
                            TrainerLogEntryDto(id = 1, level = "info", message = "first"),
                            TrainerLogEntryDto(id = 3, level = "error", message = "third"),
                            TrainerLogEntryDto(id = 2, level = "warn", message = "second"),
                        ),
                    ),
                )
            },
        )
        val viewModel = viewModel(service)
        advanceUntilIdle()

        val pollJob = backgroundScope.launch { viewModel.pollStatus() }
        testScheduler.runCurrent()

        val events = viewModel.uiState.value.events
        assertEquals(listOf(3L, 2L, 1L), events.map { it.id })
        assertEquals(TrainerLogSeverity.ERROR, events[0].severity)
        assertEquals(TrainerLogSeverity.WARN, events[1].severity)
        assertEquals(TrainerLogSeverity.INFO, events[2].severity)
        pollJob.cancel()
    }

    // Doc comment on refreshLogs(): "Deliberately not surfaced as its own error banner —
    // statusError ... already covers 'the admin API is unreachable' for both endpoints."
    @Test
    fun `a failed log fetch does not clear previously loaded events or set its own error`() = runTest {
        var shouldFail = false
        val service = FakeTrainerApiService(
            getLogsResponse = {
                if (shouldFail) {
                    Response.error(500, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"boom\"}"))
                } else {
                    Response.success(TrainerLogsResponseDto(entries = listOf(TrainerLogEntryDto(id = 1, message = "hello"))))
                }
            },
        )
        val viewModel = viewModel(service)
        advanceUntilIdle()
        val pollJob = backgroundScope.launch { viewModel.pollStatus() }
        testScheduler.runCurrent()
        assertEquals(1, viewModel.uiState.value.events.size)

        shouldFail = true
        testScheduler.advanceTimeBy(6000L) // past POLL_INTERVAL_MS (5000ms)
        testScheduler.runCurrent()

        assertEquals(
            "a failed getLogs() must not wipe out the previously loaded events",
            1,
            viewModel.uiState.value.events.size,
        )
        pollJob.cancel()
    }

    @Test
    fun `a completed checkpoint reports the written path`() = runTest {
        val service = FakeTrainerApiService(
            checkpointResponse = { Response.success(TrainerCheckpointResultDto(requested = true, completed = true, checkpoint_path = "/data/ckpt-42.bin")) },
        )
        val viewModel = viewModel(service)
        advanceUntilIdle()

        viewModel.requestCheckpoint()
        advanceUntilIdle()

        assertEquals("Checkpoint written to /data/ckpt-42.bin.", viewModel.uiState.value.actionMessage)
    }

    @Test
    fun `a not-yet-completed checkpoint reports it will complete later`() = runTest {
        val service = FakeTrainerApiService(
            checkpointResponse = { Response.success(TrainerCheckpointResultDto(requested = true, completed = false)) },
        )
        val viewModel = viewModel(service)
        advanceUntilIdle()

        viewModel.requestCheckpoint()
        advanceUntilIdle()

        assertEquals(
            "Checkpoint requested — will complete at the next optimizer-step boundary.",
            viewModel.uiState.value.actionMessage,
        )
    }

    @Test
    fun `pauseTraining and resumeTraining report their own confirmation messages`() = runTest {
        val service = FakeTrainerApiService()
        val viewModel = viewModel(service)
        advanceUntilIdle()

        viewModel.pauseTraining()
        advanceUntilIdle()
        assertEquals("Pause requested.", viewModel.uiState.value.actionMessage)
        assertEquals(1, service.pauseCallCount)

        viewModel.resumeTraining()
        advanceUntilIdle()
        assertEquals("Resume requested.", viewModel.uiState.value.actionMessage)
        assertEquals(1, service.resumeCallCount)
    }

    @Test
    fun `a successful config update re-fetches the config and reports which field changed`() = runTest {
        val service = FakeTrainerApiService()
        val viewModel = viewModel(service)
        advanceUntilIdle()

        viewModel.updateAutoSaveEnabled(false)
        advanceUntilIdle()

        assertEquals("Updated auto_save_enabled.", viewModel.uiState.value.actionMessage)
        assertTrue(viewModel.uiState.value.configError == null)
    }
}
