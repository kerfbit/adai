package com.adai.ops.ui.models

import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * TD-048: ModelListViewModel had zero test coverage. Same stale-while-error shape as
 * SessionListViewModel's own tests: a failed poll must not wipe out the last good list.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ModelListViewModelTest {

    @Test
    fun `a successful refresh populates models and clears any error`() = runTest {
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, _ ->
                ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "model-1", state = "training")))
            },
        )
        val viewModel = ModelListViewModel(
            ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        val pollJob = backgroundScope.launch { viewModel.pollModels() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals(listOf(ModelRecordDto(model_id = "model-1", state = "training")), state.models)
        assertFalse(state.isLoading)
        assertNull(state.error)
        pollJob.cancel()
    }

    @Test
    fun `a failed refresh keeps the previous models and surfaces the error`() = runTest {
        var shouldFail = false
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, _ ->
                if (shouldFail) throw IOException("connection refused")
                ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "model-1")))
            },
        )
        val viewModel = ModelListViewModel(
            ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        val pollJob = backgroundScope.launch { viewModel.pollModels() }
        testScheduler.runCurrent()
        assertEquals(1, viewModel.uiState.value.models.size)

        shouldFail = true
        testScheduler.advanceTimeBy(9000L) // past LIST_POLL_INTERVAL_MS (8000ms)
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals("connection refused", state.error)
        assertEquals(
            "a failed poll must not wipe out data from the previous successful one",
            listOf(ModelRecordDto(model_id = "model-1")),
            state.models,
        )
        pollJob.cancel()
    }
}
