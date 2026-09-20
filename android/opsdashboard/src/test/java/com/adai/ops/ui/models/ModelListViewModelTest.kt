package com.adai.ops.ui.models

import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ArchDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.network.dto.RegisterModelResultDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: ModelListViewModel had zero test coverage. Same stale-while-error shape as
 * SessionListViewModel's own tests: a failed poll must not wipe out the last good list.
 *
 * TD-196: setKindFilter/registerModel use viewModelScope.launch directly (like
 * ModelDetailViewModel's own action methods), which needs a real Main dispatcher — hence the
 * StandardTestDispatcher setup below, matching ModelDetailViewModelTest's own precedent.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ModelListViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `a successful refresh populates models and clears any error`() = runTest {
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, _, _ ->
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
            listModelsResponse = { _, _, _, _ ->
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

    // ─────────────────────────────────────────────────────────────────────
    // TD-196: kind filter + registerModel
    // ─────────────────────────────────────────────────────────────────────

    @Test
    fun `setKindFilter re-fetches with the new kind and stores the selection`() = runTest {
        var capturedKind: String? = "unset"
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, kind, _ -> capturedKind = kind; ModelsResponseDto() },
        )
        val viewModel = ModelListViewModel(
            ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        viewModel.setKindFilter("encoder")
        testScheduler.runCurrent()

        assertEquals("encoder", capturedKind)
        assertEquals("encoder", viewModel.uiState.value.selectedKind)
    }

    @Test
    fun `registerModel success shows a confirmation message and refreshes the list`() = runTest {
        var getModelCalls = 0
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, _, _ -> getModelCalls++; ModelsResponseDto() },
            registerModelResponse = { Response.success(RegisterModelResultDto(model_id = "uuid-1", state = "initializing")) },
        )
        val viewModel = ModelListViewModel(
            ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        viewModel.registerModel(RegisterModelRequestDto(model_name = "my-encoder", kind = "encoder", arch = ArchDto(d_model = 128)))
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertTrue(state.registerMessage!!.contains("my-encoder"))
        assertFalse(state.registerInProgress)
        assertTrue("registerModel must trigger a refresh() on success", getModelCalls >= 1)
    }

    // TD-199 (review fix): loadChatbotLinkCandidates() must issue its own dedicated
    // encoder/decoder fetches, independent of whatever kind filter chip is currently selected on
    // the list screen — RegisterChatbotDialog used to be handed state.models directly, which is
    // always restricted to the active filter, silently emptying the Decoder (or both) pickers
    // whenever a filter other than "All" was selected.
    @Test
    fun `loadChatbotLinkCandidates fetches encoders and decoders independent of the active list filter`() = runTest {
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, kind, _ ->
                when (kind) {
                    "encoder" -> ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "e1", model_name = "my-enc", kind = "encoder")))
                    "decoder" -> ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "d1", model_name = "my-dec", kind = "decoder")))
                    // Simulates the list screen's own "Encoder" filter chip already being active
                    // (this is exactly the case the bug reproduced under) — must have no bearing
                    // on what loadChatbotLinkCandidates() itself fetches.
                    "chatbot" -> ModelsResponseDto()
                    else -> ModelsResponseDto()
                }
            },
        )
        val viewModel = ModelListViewModel(
            ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        viewModel.setKindFilter("encoder")
        testScheduler.runCurrent()

        viewModel.loadChatbotLinkCandidates()
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals("my-enc", state.encoderCandidates.single().model_name)
        assertEquals("my-dec", state.decoderCandidates.single().model_name)
    }

    @Test
    fun `registerModel failure surfaces the server's error in the register message`() = runTest {
        val fakeService = FakeMnsApiService(
            registerModelResponse = {
                Response.error(409, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"incompatible d_model\"}"))
            },
        )
        val viewModel = ModelListViewModel(
            ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )

        viewModel.registerModel(RegisterModelRequestDto(model_name = "bot", kind = "chatbot"))
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertTrue(state.registerMessage!!.contains("Failed"))
        assertTrue(state.registerMessage!!.contains("incompatible d_model"))
    }
}
