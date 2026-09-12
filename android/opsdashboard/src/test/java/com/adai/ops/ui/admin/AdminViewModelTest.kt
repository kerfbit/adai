package com.adai.ops.ui.admin

import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.MetricsAdminConfigDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: AdminViewModel had zero test coverage — including for AdminUiState's own documented
 * invariant ("one daemon's *_error is set independently of the others"), which had never
 * actually been exercised. The second test below is exactly that.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class AdminViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private fun viewModel(
        mnsService: FakeMnsApiService = FakeMnsApiService(),
        registryService: FakeRegistryApiService = FakeRegistryApiService(),
        metricsService: FakeMetricsApiService = FakeMetricsApiService(),
    ) = AdminViewModel(
        modelRepository = ModelRepository(FakeApiClientProvider(mnsService), FakeSettingsRepository()),
        registryRepository = RegistryRepository(FakeApiClientProvider(registryService), FakeSettingsRepository()),
        metricsRepository = MetricsRepository(FakeApiClientProvider(metricsService), FakeSettingsRepository()),
    )

    @Test
    fun `all three configs load successfully on init`() = runTest {
        val viewModel = viewModel(
            mnsService = FakeMnsApiService(getAdminConfigResponse = { Response.success(MnsAdminConfigDto(registry_url = "http://mns")) }),
        )
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals("http://mns", state.mnsConfig?.registry_url)
        assertNotNull(state.registryConfig)
        assertNotNull(state.metricsConfig)
        assertNull(state.mnsError)
        assertNull(state.registryError)
        assertNull(state.metricsError)
        assertEquals(false, state.isLoading)
    }

    // AdminUiState's own doc comment: "One daemon's *_error is set independently of the
    // others — a 403 on mns_server (admin disabled) ... must not block the other two sections
    // from rendering their live config."
    @Test
    fun `one daemon's admin-config failure does not block the other two`() = runTest {
        val viewModel = viewModel(
            mnsService = FakeMnsApiService(
                getAdminConfigResponse = {
                    Response.error(403, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"admin disabled\"}"))
                },
            ),
            registryService = FakeRegistryApiService(
                getAdminConfigResponse = { Response.success(RegistryAdminConfigDto()) },
            ),
            metricsService = FakeMetricsApiService(
                getAdminConfigResponse = { Response.success(MetricsAdminConfigDto()) },
            ),
        )
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertNotNull("mns failing must still surface an error", state.mnsError)
        assertNotNull("registry must still load despite mns failing", state.registryConfig)
        assertNull("registry must not also report an error", state.registryError)
        assertNotNull("metrics must still load despite mns failing", state.metricsConfig)
        assertNull("metrics must not also report an error", state.metricsError)
    }

    @Test
    fun `a successful update re-fetches all configs and reports which field changed`() = runTest {
        val mnsService = FakeMnsApiService(
            getAdminConfigResponse = { Response.success(MnsAdminConfigDto(registry_url = "http://updated")) },
            putAdminConfigResponse = { Response.success(MnsAdminConfigDto(registry_url = "http://updated")) },
        )
        val viewModel = viewModel(mnsService = mnsService)
        advanceUntilIdle()

        viewModel.updateRegistryUrl("http://updated")
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals("http://updated", state.mnsConfig?.registry_url)
        assertEquals("Updated registry_url.", state.actionMessage)
        assertEquals(false, state.actionInProgress)
    }

    @Test
    fun `a failed update reports failure and leaves the existing config alone`() = runTest {
        val mnsService = FakeMnsApiService(
            getAdminConfigResponse = { Response.success(MnsAdminConfigDto(registry_url = "http://original")) },
            putAdminConfigResponse = {
                Response.error(403, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"admin disabled\"}"))
            },
        )
        val viewModel = viewModel(mnsService = mnsService)
        advanceUntilIdle()

        viewModel.updateRegistryUrl("http://attempted")
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals("Failed to update registry_url.", state.actionMessage)
        // loadAll() re-ran as part of the failed runUpdate too, but the server's own config is
        // unchanged, so the original value should still be what's shown.
        assertEquals("http://original", state.mnsConfig?.registry_url)
    }

    @Test
    fun `dismissActionMessage clears a previously set message`() = runTest {
        val mnsService = FakeMnsApiService(
            putAdminConfigResponse = { Response.success(MnsAdminConfigDto()) },
        )
        val viewModel = viewModel(mnsService = mnsService)
        advanceUntilIdle()
        viewModel.updateRegistryUrl("http://new")
        advanceUntilIdle()
        assertTrue(viewModel.uiState.value.actionMessage != null)

        viewModel.dismissActionMessage()

        assertNull(viewModel.uiState.value.actionMessage)
    }
}
