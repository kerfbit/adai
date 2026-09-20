package com.adai.ops.ui.registry

import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.PendingAddResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.UnassignResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Response

/**
 * TD-205: setKindFilter/poolHealth and the new mutating actions (unassignModel, deleteEntry,
 * manualAdd, upload, migrateToKind, createSegments) added no ViewModel-level coverage of their
 * own when they were built — GroupDetailScreenConfirmActionTest exercises them end-to-end through
 * the full Compose screen, but this file's own action-method tests (below) mirror
 * ModelDetailViewModelTest's tighter, dispatcher-only coverage of the ViewModel logic itself.
 * Unlike the pre-existing two tests above (which only exercise pollGroup()'s own refresh loop,
 * not viewModelScope.launch), the new action methods need the same StandardTestDispatcher setup
 * ModelDetailViewModelTest uses, since they launch via viewModelScope directly.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class GroupDetailViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    // TD-128: registry()'s error used to be silently dropped — refresh()'s `error`
    // computation only ever checked queue()/runs(), even though registry() (the
    // Phase 15 trained-files fetch) and listModels() were fetched right alongside
    // them. This proves a registry() failure now actually surfaces in ui state,
    // with queue()/runs() still succeeding around it.
    @Test
    fun `a registry fetch failure surfaces in ui state even though queue and runs succeed`() = runTest {
        val fakeRegistryService = FakeRegistryApiService(
            registryResponse = { throw IOException("connection refused") },
        )
        val viewModel = GroupDetailViewModel(
            group = "my-group",
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeRegistryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )

        val pollJob = backgroundScope.launch { viewModel.pollGroup() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertNotNull("registry() failing should surface an error", state.error)
        assertEquals("connection refused", state.error)
        // The other three fetches on the same tick should be unaffected.
        assertEquals(emptyList<Any>(), state.queueEntries)
        assertEquals(emptyMap<String, List<String>>(), state.runs)
        pollJob.cancel()
    }

    @Test
    fun `no error when every fetch succeeds`() = runTest {
        val viewModel = GroupDetailViewModel(
            group = "my-group",
            registryRepository = RegistryRepository(FakeApiClientProvider(FakeRegistryApiService()), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )

        val pollJob = backgroundScope.launch { viewModel.pollGroup() }
        testScheduler.runCurrent()

        assertNull(viewModel.uiState.value.error)
        pollJob.cancel()
    }

    @Test
    fun `setKindFilter re-queries that kind's own queue and updates selectedKind`() = runTest {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = { groupPath ->
                if (groupPath == "my-group/chatbot") {
                    QueueResponseDto(entries = listOf(QueueEntryDto(path = "chatbot-file.jsonl")))
                } else {
                    QueueResponseDto()
                }
            },
        )
        val viewModel = GroupDetailViewModel(
            group = "my-group",
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeRegistryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollGroup() }
        testScheduler.runCurrent()
        assertEquals(emptyList<QueueEntryDto>(), viewModel.uiState.value.queueEntries)

        viewModel.setKindFilter("chatbot")
        testScheduler.runCurrent()

        assertEquals("chatbot", viewModel.uiState.value.selectedKind)
        assertEquals(listOf(QueueEntryDto(path = "chatbot-file.jsonl")), viewModel.uiState.value.queueEntries)
        pollJob.cancel()
    }

    @Test
    fun `poolHealth covers every kind independent of the selected filter`() = runTest {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = { groupPath ->
                when (groupPath) {
                    "my-group" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "legacy.jsonl")))
                    "my-group/chatbot" -> QueueResponseDto(
                        entries = listOf(QueueEntryDto(path = "c1.jsonl"), QueueEntryDto(path = "c2.jsonl")),
                    )
                    else -> QueueResponseDto()
                }
            },
            registryResponse = { groupPath ->
                if (groupPath == "my-group/world_model") {
                    RegistryResponseDto(
                        entries = listOf(RegistryEntryDto(data_file = "wm.jsonl", num_samples = 40, trained = true)),
                    )
                } else {
                    RegistryResponseDto()
                }
            },
        )
        val viewModel = GroupDetailViewModel(
            group = "my-group",
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeRegistryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollGroup() }
        testScheduler.runCurrent()

        val poolHealth = viewModel.uiState.value.poolHealth
        assertEquals(listOf(null, "encoder", "decoder", "world_model", "chatbot"), poolHealth.map { it.kind })
        assertEquals(1, poolHealth.single { it.kind == null }.pendingCount)
        assertEquals(2, poolHealth.single { it.kind == "chatbot" }.pendingCount)
        assertEquals(0, poolHealth.single { it.kind == "encoder" }.pendingCount)
        assertEquals(1, poolHealth.single { it.kind == "world_model" }.trainedCount)
        assertEquals(40, poolHealth.single { it.kind == "world_model" }.totalSamples)
        pollJob.cancel()
    }

    @Test
    fun `unassignModel success sets a confirmation message and refreshes`() = runTest {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "a.jsonl", model_name = "model-a")))
            },
            unassignResponse = { _, _ -> Response.success(UnassignResponseDto(unassigned = 1)) },
        )
        val viewModel = GroupDetailViewModel(
            group = "my-group",
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeRegistryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollGroup() }
        testScheduler.runCurrent()
        val entry = viewModel.uiState.value.queueEntries.single()

        viewModel.unassignModel(entry)
        testScheduler.runCurrent()

        assertEquals("Unassigned 'a.jsonl' (was 'model-a').", viewModel.uiState.value.actionMessage)
        assertEquals(false, viewModel.uiState.value.actionInProgress)
        val (_, body) = fakeRegistryService.unassignCalls.single()
        assertEquals(listOf("a.jsonl"), body.paths)
        pollJob.cancel()
    }

    @Test
    fun `migrateToKind failure surfaces the error without crashing`() = runTest {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "a.jsonl")))
            },
            pendingAddResponse = { _, _ -> Response.success(PendingAddResponseDto(added = false, reason = "already pending")) },
        )
        val viewModel = GroupDetailViewModel(
            group = "my-group",
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeRegistryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )
        val pollJob = backgroundScope.launch { viewModel.pollGroup() }
        testScheduler.runCurrent()
        val entry = viewModel.uiState.value.queueEntries.single()

        viewModel.migrateToKind(entry, "chatbot")
        testScheduler.runCurrent()

        assertTrue(viewModel.uiState.value.actionMessage.orEmpty().startsWith("Failed:"))
        assertEquals(false, viewModel.uiState.value.actionInProgress)
        pollJob.cancel()
    }
}
