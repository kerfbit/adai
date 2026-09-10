package com.adai.ops.ui.registry

import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class GroupDetailViewModelTest {

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
}
