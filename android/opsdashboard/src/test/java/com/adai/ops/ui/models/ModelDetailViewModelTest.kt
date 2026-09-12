package com.adai.ops.ui.models

import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.PromoteResultDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
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
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: ModelDetailViewModel had zero test coverage — including for TD-147's own fix
 * (clearStaleTrainingLock() threading the polled model's run_id through to the server), which
 * had never had a regression test at the ViewModel level despite being verified end-to-end
 * against a real server at the time. The first test below is exactly that regression test.
 *
 * Unlike the list ViewModels' tests, the action methods here (clearStaleTrainingLock(),
 * retireCandidate(), promoteToProduction()) use viewModelScope.launch directly (fire-and-forget
 * button actions, not something the caller awaits), which needs a real Main dispatcher — hence
 * the StandardTestDispatcher setup below.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ModelDetailViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `clearStaleTrainingLock sends the most recently polled model's run_id, not an empty one`() = runTest {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(ModelRecordDto(model_id = "my-model", state = "training", run_id = "run-01")) },
        )
        val viewModel = ModelDetailViewModel(
            modelName = "my-model",
            modelRepository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollModel() }
        testScheduler.runCurrent()
        assertEquals("run-01", viewModel.uiState.value.model?.run_id)

        viewModel.clearStaleTrainingLock()
        testScheduler.runCurrent()

        val (name, body) = fakeService.setStateCalls.single()
        assertEquals("my-model", name)
        assertEquals("candidate", body.state)
        assertEquals("run-01", body.run_id)
        pollJob.cancel()
    }

    @Test
    fun `clearStaleTrainingLock success updates the model and sets a confirmation message`() = runTest {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(ModelRecordDto(model_id = "my-model", state = "training", run_id = "run-01")) },
            setStateResponse = { _, _ -> Response.success(ModelRecordDto(model_id = "my-model", state = "candidate")) },
        )
        val viewModel = ModelDetailViewModel(
            modelName = "my-model",
            modelRepository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollModel() }
        testScheduler.runCurrent()

        viewModel.clearStaleTrainingLock()
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals("candidate", state.model?.state)
        assertTrue(state.actionMessage!!.contains("candidate"))
        assertEquals(false, state.actionInProgress)
        pollJob.cancel()
    }

    @Test
    fun `a rejected action surfaces the server's error in the action message`() = runTest {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(ModelRecordDto(model_id = "my-model", state = "training", run_id = "run-01")) },
            setStateResponse = { _, _ ->
                Response.error(409, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"run_id mismatch\"}"))
            },
        )
        val viewModel = ModelDetailViewModel(
            modelName = "my-model",
            modelRepository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollModel() }
        testScheduler.runCurrent()

        viewModel.clearStaleTrainingLock()
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertTrue(state.actionMessage!!.contains("Failed"))
        assertTrue(state.actionMessage!!.contains("run_id mismatch"))
        pollJob.cancel()
    }

    @Test
    fun `promoteToProduction success re-fetches the model and reports what was retired`() = runTest {
        var getModelCalls = 0
        val fakeService = FakeMnsApiService(
            getModelResponse = {
                getModelCalls++
                Response.success(ModelRecordDto(model_id = "my-model", state = if (getModelCalls == 1) "candidate" else "production"))
            },
            promoteResponse = { _, _ ->
                Response.success(PromoteResultDto(promoted = "my-model", retired = "old-model", role = "chat-primary"))
            },
        )
        val viewModel = ModelDetailViewModel(
            modelName = "my-model",
            modelRepository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollModel() }
        testScheduler.runCurrent()

        viewModel.promoteToProduction("chat-primary")
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertTrue(state.actionMessage!!.contains("old-model"))
        // promoteToProduction() calls refresh() on success, so the model reflects the second
        // (post-promotion) getModel() response, not the stale pre-promotion one.
        assertEquals("production", state.model?.state)
        pollJob.cancel()
    }

    @Test
    fun `dismissActionMessage clears a previously set message`() = runTest {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(ModelRecordDto(model_id = "my-model", state = "training", run_id = "run-01")) },
        )
        val viewModel = ModelDetailViewModel(
            modelName = "my-model",
            modelRepository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollModel() }
        testScheduler.runCurrent()
        viewModel.clearStaleTrainingLock()
        testScheduler.runCurrent()
        assertTrue(viewModel.uiState.value.actionMessage != null)

        viewModel.dismissActionMessage()

        assertNull(viewModel.uiState.value.actionMessage)
        pollJob.cancel()
    }
}
