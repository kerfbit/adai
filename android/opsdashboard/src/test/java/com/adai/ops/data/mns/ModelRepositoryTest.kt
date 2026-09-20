package com.adai.ops.data.mns

import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.ArchDto
import com.adai.ops.network.dto.LinkWorldModelRequestDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.PromoteResultDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.network.dto.RegisterModelResultDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import retrofit2.Response

class ModelRepositoryTest {

    @Test
    fun `clearStaleTrainingLock sends state=candidate with the given run_id`() = runTest {
        // TD-147 (fixed): the server's ownership check on this transition requires the model's
        // own current run_id — this test previously asserted the pre-fix (broken) behavior of
        // sending no run_id at all, and had silently stopped compiling against the real
        // ModelRepository once that fix shipped (caught wiring up Android CI for TD-047).
        val fakeService = FakeMnsApiService()
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.clearStaleTrainingLock("my-model", "run-01")

        assertTrue(result is ApiResult.Success)
        val (name, body) = fakeService.setStateCalls.single()
        assertEquals("my-model", name)
        assertEquals("candidate", body.state)
        assertEquals("run-01", body.run_id)
    }

    @Test
    fun `retireCandidate sends state=retired with no run_id`() = runTest {
        val fakeService = FakeMnsApiService()
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.retireCandidate("my-model")

        assertTrue(result is ApiResult.Success)
        val (name, body) = fakeService.setStateCalls.single()
        assertEquals("my-model", name)
        assertEquals("retired", body.state)
        assertNull(body.run_id)
    }

    @Test
    fun `clearStaleTrainingLock surfaces a 409 as Conflict, not a generic ApiError`() = runTest {
        val fakeService = FakeMnsApiService(
            setStateResponse = { _, _ ->
                Response.error(409, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"invalid transition\"}"))
            },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.clearStaleTrainingLock("my-model", "run-01")

        assertTrue(result is ApiResult.Conflict)
        assertEquals("invalid transition", (result as ApiResult.Conflict).message)
    }

    @Test
    fun `promoteToProduction sends the model name for the given role`() = runTest {
        val fakeService = FakeMnsApiService(
            promoteResponse = { _, _ ->
                Response.success(PromoteResultDto(promoted = "my-model", retired = "old-model", role = "chat-primary"))
            },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.promoteToProduction("chat-primary", "my-model")

        assertTrue(result is ApiResult.Success)
        val (role, body) = fakeService.promoteCalls.single()
        assertEquals("chat-primary", role)
        assertEquals("my-model", body.model_name)
        assertEquals("old-model", (result as ApiResult.Success).data.retired)
    }

    @Test
    fun `promoteToProduction surfaces a 409 when the model is not a candidate`() = runTest {
        val fakeService = FakeMnsApiService(
            promoteResponse = { _, _ ->
                Response.error(
                    409,
                    ResponseBody.create(
                        "application/json".toMediaType(),
                        "{\"error\":\"model must be in candidate state to promote to production\"}",
                    ),
                )
            },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.promoteToProduction("chat-primary", "my-model")

        assertTrue(result is ApiResult.Conflict)
    }

    @Test
    fun `updateRegistryUrl sends a single-key body, never the full round-tripped object`() = runTest {
        val fakeService = FakeMnsApiService(
            putAdminConfigResponse = { Response.success(MnsAdminConfigDto(registry_url = "http://new-host:8082")) },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.updateRegistryUrl("http://new-host:8082")

        assertTrue(result is ApiResult.Success)
        assertEquals("http://new-host:8082", (result as ApiResult.Success).data.registry_url)
        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals("http://new-host:8082", body.getValue("registry_url").jsonPrimitive.content)
    }

    @Test
    fun `updateRegistryGroup sends only registry_group, not registry_url`() = runTest {
        val fakeService = FakeMnsApiService()
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.updateRegistryGroup("new-group")

        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals("new-group", body.getValue("registry_group").jsonPrimitive.content)
    }

    @Test
    fun `getAdminConfig surfaces a 403 (admin disabled) as ApiError, not an exception`() = runTest {
        val fakeService = FakeMnsApiService(
            getAdminConfigResponse = {
                Response.error(
                    403,
                    ResponseBody.create(
                        "application/json".toMediaType(),
                        "{\"error\":\"admin config mutation disabled (--admin-enabled=false)\"}",
                    ),
                )
            },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.getAdminConfig()

        assertTrue(result is ApiResult.ApiError)
        assertFalse((result as ApiResult.ApiError).message.isEmpty())
    }

    // ─────────────────────────────────────────────────────────────────────
    // TD-196: kind filter, registerModel, linkWorldModel
    // ─────────────────────────────────────────────────────────────────────

    @Test
    fun `listModels passes the kind query param through`() = runTest {
        var capturedKind: String? = "unset"
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, kind, _ -> capturedKind = kind; ModelsResponseDto() },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.listModels(kind = "encoder")

        assertEquals("encoder", capturedKind)
    }

    @Test
    fun `registerModel sends the full request and returns the model_id on success`() = runTest {
        val fakeService = FakeMnsApiService(
            registerModelResponse = { Response.success(RegisterModelResultDto(model_id = "uuid-1", state = "initializing")) },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())
        val request = RegisterModelRequestDto(
            model_name = "my-encoder", kind = "encoder",
            arch = ArchDto(d_model = 128, num_heads = 4, d_ff = 512, num_encoder_layers = 3, max_seq_length = 256),
        )

        val result = repository.registerModel(request)

        assertTrue(result is ApiResult.Success)
        assertEquals("uuid-1", (result as ApiResult.Success).data.model_id)
        assertEquals(request, fakeService.registerModelCalls.single())
    }

    @Test
    fun `registerModel surfaces a 409 dimension mismatch as Conflict`() = runTest {
        val fakeService = FakeMnsApiService(
            registerModelResponse = {
                Response.error(409, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"incompatible d_model\"}"))
            },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.registerModel(RegisterModelRequestDto(model_name = "bot", kind = "chatbot"))

        assertTrue(result is ApiResult.Conflict)
        assertEquals("incompatible d_model", (result as ApiResult.Conflict).message)
    }

    @Test
    fun `linkWorldModel sends the request to the chatbot's own path`() = runTest {
        val fakeService = FakeMnsApiService()
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())
        val request = LinkWorldModelRequestDto(world_model_name = "my-wm", world_model_inject_every_n_layers = 2)

        val result = repository.linkWorldModel("my-chatbot", request)

        assertTrue(result is ApiResult.Success)
        assertEquals("my-wm", (result as ApiResult.Success).data.world_model_name)
        val (name, body) = fakeService.linkWorldModelCalls.single()
        assertEquals("my-chatbot", name)
        assertEquals(request, body)
    }

    @Test
    fun `linkWorldModel surfaces a 409 d_model mismatch as Conflict`() = runTest {
        val fakeService = FakeMnsApiService(
            linkWorldModelResponse = { _, _ ->
                Response.error(409, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"d_model mismatch\"}"))
            },
        )
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.linkWorldModel("my-chatbot", LinkWorldModelRequestDto(world_model_name = "my-wm"))

        assertTrue(result is ApiResult.Conflict)
        assertEquals("d_model mismatch", (result as ApiResult.Conflict).message)
    }

    @Test
    fun `linkWorldModel with an empty world_model_name detaches`() = runTest {
        val fakeService = FakeMnsApiService()
        val repository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.linkWorldModel("my-chatbot", LinkWorldModelRequestDto(world_model_name = ""))

        assertTrue(result is ApiResult.Success)
        assertEquals("", (result as ApiResult.Success).data.world_model_name)
    }
}
