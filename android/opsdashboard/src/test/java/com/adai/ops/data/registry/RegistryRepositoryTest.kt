package com.adai.ops.data.registry

import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.FetchResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import retrofit2.Response

class RegistryRepositoryTest {

    @Test
    fun `forceRelease always sends an empty run_id, bypassing the owner check`() = runTest {
        val fakeService = FakeRegistryApiService(
            releaseResponse = { _, _ -> ReleaseResponseDto(released = 3) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.forceRelease("my-group", listOf("a.jsonl", "b.jsonl", "c.jsonl"))

        assertTrue(result is ApiResult.Success)
        assertEquals(3, (result as ApiResult.Success).data.released)
        val (group, body) = fakeService.releaseCalls.single()
        assertEquals("my-group", group)
        assertEquals("", body.run_id)
        assertEquals(listOf("a.jsonl", "b.jsonl", "c.jsonl"), body.files)
    }

    @Test
    fun `assignModel sends model_name and paths and maps the assigned count`() = runTest {
        val fakeService = FakeRegistryApiService(
            assignResponse = { _, _ -> AssignResponseDto(assigned = 1) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.assignModel("my-group", "model-a", listOf("a.jsonl"))

        assertTrue(result is ApiResult.Success)
        assertEquals(1, (result as ApiResult.Success).data.assigned)
        val (group, body) = fakeService.assignCalls.single()
        assertEquals("my-group", group)
        assertEquals("model-a", body.model_name)
        assertEquals(listOf("a.jsonl"), body.paths)
    }

    @Test
    fun `fetchGutenberg sends book_id, num_pairs, and model_name`() = runTest {
        val fakeService = FakeRegistryApiService(
            fetchGutenbergResponse = { _, _ ->
                Response.success(FetchResponseDto(added = true, path = "book.jsonl", pairs_written = 5))
            },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.fetchGutenberg("my-group", bookId = 11, numPairs = 5, modelName = "model-a")

        assertTrue(result is ApiResult.Success)
        assertEquals(5, (result as ApiResult.Success).data.pairs_written)
        val (group, body) = fakeService.fetchGutenbergCalls.single()
        assertEquals("my-group", group)
        assertEquals(11, body.book_id)
        assertEquals(5, body.num_pairs)
        assertEquals("model-a", body.model_name)
    }

    @Test
    fun `fetchHuggingface sends all fields including split and field overrides`() = runTest {
        val fakeService = FakeRegistryApiService(
            fetchHuggingfaceResponse = { _, _ ->
                Response.success(FetchResponseDto(added = true, path = "hf.jsonl", pairs_written = 10))
            },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.fetchHuggingface(
            "my-group",
            datasetId = "tatsu-lab/alpaca",
            numPairs = 10,
            split = "train",
            inputField = "instruction",
            outputField = "output",
            modelName = "model-b",
        )

        assertTrue(result is ApiResult.Success)
        assertEquals(10, (result as ApiResult.Success).data.pairs_written)
        val (group, body) = fakeService.fetchHuggingfaceCalls.single()
        assertEquals("my-group", group)
        assertEquals("tatsu-lab/alpaca", body.dataset_id)
        assertEquals("train", body.split)
        assertEquals("instruction", body.input_field)
        assertEquals("output", body.output_field)
        assertEquals("model-b", body.model_name)
    }

    @Test
    fun `queue maps Phase 15 dataset metadata fields through unchanged`() = runTest {
        val fakeService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(
                    entries = listOf(
                        QueueEntryDto(
                            path = "a.jsonl",
                            source = "gutenberg",
                            added_utc = "2026-08-02T14:30:00Z",
                            size_bytes = 2048,
                            num_entries = 5,
                            checksum = "2048_12345",
                        ),
                    ),
                )
            },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.queue("my-group")

        assertTrue(result is ApiResult.Success)
        val entry = (result as ApiResult.Success).data.entries.single()
        assertEquals("gutenberg", entry.source)
        assertEquals("2026-08-02T14:30:00Z", entry.added_utc)
        assertEquals(2048L, entry.size_bytes)
        assertEquals(5, entry.num_entries)
        assertEquals("2048_12345", entry.checksum)
    }

    @Test
    fun `registry maps Phase 15 source and added_utc fields through unchanged`() = runTest {
        val fakeService = FakeRegistryApiService(
            registryResponse = {
                RegistryResponseDto(
                    entries = listOf(
                        RegistryEntryDto(
                            data_file = "a.jsonl",
                            num_samples = 42,
                            trained = true,
                            added_utc = "2026-08-01T09:00:00Z",
                            source = "upload",
                        ),
                    ),
                )
            },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.registry("my-group")

        assertTrue(result is ApiResult.Success)
        val entry = (result as ApiResult.Success).data.entries.single()
        assertEquals("2026-08-01T09:00:00Z", entry.added_utc)
        assertEquals("upload", entry.source)
    }

    @Test
    fun `updateFtpTokenTtlMinutes sends a single-key body, never the full round-tripped object`() = runTest {
        val fakeService = FakeRegistryApiService(
            putAdminConfigResponse = { Response.success(RegistryAdminConfigDto(ftp_token_ttl_minutes = 30)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.updateFtpTokenTtlMinutes(30)

        assertTrue(result is ApiResult.Success)
        assertEquals(30, (result as ApiResult.Success).data.ftp_token_ttl_minutes)
        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(30, body.getValue("ftp_token_ttl_minutes").jsonPrimitive.int)
    }

    @Test
    fun `updateFtpMaxSessionsPerRun sends only ftp_max_sessions_per_run`() = runTest {
        val fakeService = FakeRegistryApiService()
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.updateFtpMaxSessionsPerRun(5)

        val body = fakeService.putAdminConfigCalls.single()
        assertEquals(1, body.size)
        assertEquals(5, body.getValue("ftp_max_sessions_per_run").jsonPrimitive.int)
    }
}
