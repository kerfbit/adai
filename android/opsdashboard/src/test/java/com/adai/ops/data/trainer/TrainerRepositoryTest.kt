package com.adai.ops.data.trainer

import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerCheckpointResultDto
import com.adai.ops.network.dto.TrainerLogEntryDto
import com.adai.ops.network.dto.TrainerLogsResponseDto
import com.adai.ops.network.dto.TrainerPauseResultDto
import com.adai.ops.network.dto.TrainerStatusDto
import com.adai.ops.settings.OpsSettings
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.FakeTrainerApiService
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

class TrainerRepositoryTest {

    @Test
    fun `getStatus maps every field through unchanged`() = runTest {
        val fakeService = FakeTrainerApiService(
            getStatusResponse = {
                Response.success(
                    TrainerStatusDto(
                        phase = "training",
                        paused = false,
                        run_id = "run-03",
                        session_id = "session-07",
                        model_name = "ambitious-aardvark",
                        current_epoch = 2,
                        total_epochs = 5,
                        samples_trained_this_pass = 1234,
                        last_loss = 0.53,
                        best_loss = 0.49,
                        checkpoints_written = 3,
                        last_checkpoint_path = "/opt/adai/training_sessions/auto_save_session_3.bin",
                        last_checkpoint_time_unix = 1_700_000_000,
                    ),
                )
            },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.getStatus()

        assertTrue(result is ApiResult.Success)
        val status = (result as ApiResult.Success).data
        assertEquals("training", status.phase)
        assertEquals("run-03", status.run_id)
        assertEquals("session-07", status.session_id)
        assertEquals(2, status.current_epoch)
        assertEquals(5, status.total_epochs)
        assertEquals(1234L, status.samples_trained_this_pass)
        assertEquals(3L, status.checkpoints_written)
    }

    @Test
    fun `getLogs maps entry fields through unchanged`() = runTest {
        val fakeService = FakeTrainerApiService(
            getLogsResponse = {
                Response.success(
                    TrainerLogsResponseDto(
                        entries = listOf(
                            TrainerLogEntryDto(
                                id = 17,
                                timestamp_unix_ms = 1_755_031_234_567,
                                level = "warn",
                                message = "Pause requested via admin API",
                            ),
                        ),
                    ),
                )
            },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.getLogs()

        assertTrue(result is ApiResult.Success)
        val entry = (result as ApiResult.Success).data.entries.single()
        assertEquals(17L, entry.id)
        assertEquals(1_755_031_234_567L, entry.timestamp_unix_ms)
        assertEquals("warn", entry.level)
        assertEquals("Pause requested via admin API", entry.message)
        assertEquals(1, fakeService.getLogsCallCount)
    }

    @Test
    fun `updateAutoSaveEnabled sends a single-key body, never the full round-tripped object`() = runTest {
        val fakeService = FakeTrainerApiService(
            putAdminConfigResponse = { Response.success(TrainerAdminConfigDto(auto_save_enabled = false)) },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.updateAutoSaveEnabled(false)

        assertTrue(result is ApiResult.Success)
        assertEquals(false, (result as ApiResult.Success).data.auto_save_enabled)
        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(false, body.getValue("auto_save_enabled").jsonPrimitive.boolean)
    }

    @Test
    fun `updateAutoSaveEverySamples sends only auto_save_every_samples`() = runTest {
        val fakeService = FakeTrainerApiService()
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.updateAutoSaveEverySamples(500)

        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(500, body.getValue("auto_save_every_samples").jsonPrimitive.int)
    }

    @Test
    fun `updateAutoSaveEveryMinutes sends only auto_save_every_minutes`() = runTest {
        val fakeService = FakeTrainerApiService()
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.updateAutoSaveEveryMinutes(5)

        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(5, body.getValue("auto_save_every_minutes").jsonPrimitive.int)
    }

    @Test
    fun `updateMaxSessionsToKeep sends only max_sessions_to_keep`() = runTest {
        val fakeService = FakeTrainerApiService()
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.updateMaxSessionsToKeep(10)

        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(10, body.getValue("max_sessions_to_keep").jsonPrimitive.int)
    }

    @Test
    fun `requestCheckpoint forwards wait_ms as a query parameter`() = runTest {
        val fakeService = FakeTrainerApiService(
            checkpointResponse = { _ ->
                Response.success(
                    TrainerCheckpointResultDto(requested = true, completed = true, checkpoints_written = 4),
                )
            },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.requestCheckpoint(waitMs = 1000)

        assertTrue(result is ApiResult.Success)
        assertEquals(4L, (result as ApiResult.Success).data.checkpoints_written)
        assertEquals(listOf(1000), fakeService.checkpointCalls)
    }

    @Test
    fun `requestCheckpoint surfaces a 409 as Conflict when the trainer is idle`() = runTest {
        val fakeService = FakeTrainerApiService(
            checkpointResponse = { _ ->
                Response.error(
                    409,
                    ResponseBody.create(
                        "application/json".toMediaType(),
                        "{\"error\":\"no active training pass; nothing to checkpoint\"}",
                    ),
                )
            },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.requestCheckpoint()

        assertTrue(result is ApiResult.Conflict)
        assertEquals("no active training pass; nothing to checkpoint", (result as ApiResult.Conflict).message)
    }

    @Test
    fun `pause calls POST admin-pause with no body`() = runTest {
        val fakeService = FakeTrainerApiService(
            pauseResponse = { Response.success(TrainerPauseResultDto(paused = true)) },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.pause()

        assertTrue(result is ApiResult.Success)
        assertEquals(true, (result as ApiResult.Success).data.paused)
        assertEquals(1, fakeService.pauseCallCount)
    }

    @Test
    fun `resume calls POST admin-resume with no body`() = runTest {
        val fakeService = FakeTrainerApiService(
            resumeResponse = { Response.success(TrainerPauseResultDto(paused = false)) },
        )
        val repository = TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.resume()

        assertTrue(result is ApiResult.Success)
        assertEquals(false, (result as ApiResult.Success).data.paused)
        assertEquals(1, fakeService.resumeCallCount)
    }

    @Test
    fun `service resolution uses the dedicated trainer Access credentials, never the shared ones`() = runTest {
        val fakeService = FakeTrainerApiService()
        val fakeApiClientProvider = FakeApiClientProvider(fakeService)
        val settings = FakeSettingsRepository(
            OpsSettings(
                useSharedHost = true,
                sharedHost = "trainer.kerfbit.dev",
                useHttpsRelay = true,
                accessClientId = "shared-id",
                accessClientSecret = "shared-secret",
                trainerAccessClientId = "trainer-id",
                trainerAccessClientSecret = "trainer-secret",
            ),
        )
        val repository = TrainerRepository(fakeApiClientProvider, settings)

        repository.getStatus()

        val call = fakeApiClientProvider.calls.single()
        assertEquals("trainer.kerfbit.dev", call.host)
        assertTrue(call.useHttps)
        assertEquals("trainer-id", call.accessClientId)
        assertEquals("trainer-secret", call.accessClientSecret)
    }
}
