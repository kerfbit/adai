package com.adai.ops.ui.registry

import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.settings.OpsSettings
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * TD-048: GroupListViewModel had zero test coverage. Mirrors GroupDetailViewModelTest's own
 * "one fetch failing must not affect the others" pattern (TD-128) — here across multiple
 * groups instead of multiple endpoints for one group. TD-580: groups also come from MNS, each
 * non-retired model's run_group.
 */
class GroupListViewModelTest {

    private fun models(service: FakeMnsApiService = FakeMnsApiService()) =
        ModelRepository(FakeApiClientProvider(service), FakeSettingsRepository())

    private fun mnsWith(vararg records: ModelRecordDto) =
        FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto(models = records.toList()) })

    private fun viewModel(
        registry: FakeRegistryApiService,
        configured: List<String>,
        mns: FakeMnsApiService = FakeMnsApiService(),
    ) = GroupListViewModel(
        registryRepository = RegistryRepository(FakeApiClientProvider(registry), FakeSettingsRepository()),
        settingsRepository = FakeSettingsRepository(
            OpsSettings(useSharedHost = true, sharedHost = "localhost", registryGroups = configured),
        ),
        modelRepository = models(mns),
    )

    @Test
    fun `each configured group gets its own pending count`() = runTest {
        val fakeService = FakeRegistryApiService(
            queueResponse = { group ->
                when (group) {
                    "group-a" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "a1"), QueueEntryDto(path = "a2")))
                    "group-b" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "b1")))
                    else -> QueueResponseDto()
                }
            },
        )
        val viewModel = GroupListViewModel(
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
            settingsRepository = FakeSettingsRepository(
                OpsSettings(useSharedHost = true, sharedHost = "localhost", registryGroups = listOf("group-a", "group-b")),
            ),
            modelRepository = models(),
        )

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals(false, state.isLoading)
        assertEquals(
            listOf(GroupSummary("group-a", pendingCount = 2), GroupSummary("group-b", pendingCount = 1)),
            state.groups,
        )
        pollJob.cancel()
    }

    // TD-128-style isolation, applied to the list screen: one group's queue() failing must not
    // drop or corrupt the others' results in the same refresh.
    @Test
    fun `one group failing does not affect the others in the same refresh`() = runTest {
        val fakeService = FakeRegistryApiService(
            // TD-205: refresh() now sums the legacy pool + all 4 kind sub-pools per group
            // (see GroupListViewModel), so this must only return entries for the bare
            // (legacy) group path — an unmatched kind-scoped path (e.g. "healthy-group/chatbot")
            // falling through to the same 1-entry response would inflate the expected count.
            queueResponse = { group ->
                when (group) {
                    "broken-group" -> throw IOException("connection refused")
                    "healthy-group" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "ok1")))
                    else -> QueueResponseDto()
                }
            },
        )
        val viewModel = GroupListViewModel(
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
            settingsRepository = FakeSettingsRepository(
                OpsSettings(
                    useSharedHost = true,
                    sharedHost = "localhost",
                    registryGroups = listOf("broken-group", "healthy-group"),
                ),
            ),
            modelRepository = models(),
        )

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        val broken = state.groups.single { it.name == "broken-group" }
        val healthy = state.groups.single { it.name == "healthy-group" }
        assertEquals("connection refused", broken.error)
        assertNull("a failed group must not also report a stale pendingCount", broken.pendingCount)
        assertEquals(1, healthy.pendingCount)
        assertNull("the healthy group must not be affected by the broken one", healthy.error)
        pollJob.cancel()
    }

    @Test
    fun `no configured groups produces an empty, non-loading list`() = runTest {
        val viewModel = GroupListViewModel(
            registryRepository = RegistryRepository(FakeApiClientProvider(FakeRegistryApiService()), FakeSettingsRepository()),
            settingsRepository = FakeSettingsRepository(
                OpsSettings(useSharedHost = true, sharedHost = "localhost", registryGroups = emptyList()),
            ),
            modelRepository = models(),
        )

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertTrue(state.groups.isEmpty())
        assertEquals(false, state.isLoading)
        pollJob.cancel()
    }

    @Test
    fun `groups named by non-retired MNS models are listed after Settings' groups`() = runTest {
        val registry = FakeRegistryApiService(
            queueResponse = { group ->
                when (group) {
                    "bb01" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "b1")))
                    "bb01/world_model" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "w1"), QueueEntryDto(path = "w2")))
                    else -> QueueResponseDto()
                }
            },
        )
        val mns = mnsWith(
            ModelRecordDto(model_name = "bodacious-badger", run_group = "bb01", state = "initializing"),
            ModelRecordDto(model_name = "world-model-v1", run_group = "bb01", state = "training"),
            ModelRecordDto(model_name = "chatbot-v1", run_group = "chatbot-v1", state = "retired"),
            ModelRecordDto(model_name = "no-group", run_group = ""),
        )
        val viewModel = viewModel(registry, configured = listOf("manual"), mns = mns)

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals(
            listOf(
                GroupSummary("manual", pendingCount = 0),
                GroupSummary("bb01", pendingCount = 3, models = listOf("bodacious-badger", "world-model-v1")),
            ),
            state.groups,
        )
        assertNull(state.discoveryError)
        pollJob.cancel()
    }

    @Test
    fun `a group both in Settings and in MNS is listed once, where Settings puts it`() = runTest {
        val mns = mnsWith(
            ModelRecordDto(model_name = "model-z", run_group = "zz"),
            ModelRecordDto(model_name = "model-a", run_group = "group-a"),
        )
        val viewModel = viewModel(FakeRegistryApiService(), configured = listOf("group-a"), mns = mns)

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val groups = viewModel.uiState.value.groups
        assertEquals(listOf("group-a", "zz"), groups.map { it.name })
        assertEquals(listOf("model-a"), groups.first().models)
        pollJob.cancel()
    }

    @Test
    fun `an unreachable MNS still shows Settings' groups and reports why`() = runTest {
        val mns = FakeMnsApiService(listModelsResponse = { _, _, _, _ -> throw IOException("mns down") })
        val viewModel = viewModel(FakeRegistryApiService(), configured = listOf("group-a"), mns = mns)

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertEquals(listOf(GroupSummary("group-a", pendingCount = 0)), state.groups)
        assertEquals("mns down", state.discoveryError)
        pollJob.cancel()
    }

    @Test
    fun `MNS is asked for more than its default 50 models`() = runTest {
        var askedLimit: Int? = null
        val mns = FakeMnsApiService(listModelsResponse = { _, _, _, limit -> askedLimit = limit; ModelsResponseDto() })
        val viewModel = viewModel(FakeRegistryApiService(), configured = emptyList(), mns = mns)

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        assertTrue("limit was $askedLimit", (askedLimit ?: 0) > 50)
        pollJob.cancel()
    }
}
