package com.adai.ops.data.metrics

import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.MetricsAdminConfigDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import retrofit2.Response

class MetricsRepositoryTest {

    @Test
    fun `getAdminConfig surfaces a 404 (admin routes not registered) as NotFound`() = runTest {
        // metrics_api_server doesn't register /admin/config at all when allow_control=false
        // at startup, unlike mns_server/registry_server's runtime 403 — see
        // MetricsAdminConfigDto's doc comment.
        val fakeService = FakeMetricsApiService(
            getAdminConfigResponse = {
                Response.error(404, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"not found\"}"))
            },
        )
        val repository = MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.getAdminConfig()

        assertTrue(result is ApiResult.NotFound)
    }

    @Test
    fun `updateMaxLiveSessions sends a single-key int body`() = runTest {
        val fakeService = FakeMetricsApiService(
            putAdminConfigResponse = { Response.success(MetricsAdminConfigDto(max_live_sessions = 50)) },
        )
        val repository = MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.updateMaxLiveSessions(50)

        assertTrue(result is ApiResult.Success)
        assertEquals(50, (result as ApiResult.Success).data.max_live_sessions)
        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(50, body.getValue("max_live_sessions").jsonPrimitive.int)
    }

    @Test
    fun `updateEnablePrometheus sends a single-key boolean body`() = runTest {
        val fakeService = FakeMetricsApiService()
        val repository = MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.updateEnablePrometheus(true)

        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertTrue(body.getValue("enable_prometheus").jsonPrimitive.boolean)
    }
}
