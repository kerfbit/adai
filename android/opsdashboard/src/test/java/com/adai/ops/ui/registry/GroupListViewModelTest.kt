package com.adai.ops.ui.registry

import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.settings.OpsSettings
import com.adai.ops.testutil.FakeApiClientProvider
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
 * groups instead of multiple endpoints for one group.
 */
class GroupListViewModelTest {

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
            queueResponse = { group ->
                if (group == "broken-group") throw IOException("connection refused")
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "ok1")))
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
        )

        val pollJob = backgroundScope.launch { viewModel.pollGroups() }
        testScheduler.runCurrent()

        val state = viewModel.uiState.value
        assertTrue(state.groups.isEmpty())
        assertEquals(false, state.isLoading)
        pollJob.cancel()
    }
}
