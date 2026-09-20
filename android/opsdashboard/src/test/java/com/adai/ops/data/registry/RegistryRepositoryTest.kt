package com.adai.ops.data.registry

import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.DeleteResponseDto
import com.adai.ops.network.dto.FetchResponseDto
import com.adai.ops.network.dto.PendingAddResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.network.dto.SegmentTargetDto
import com.adai.ops.network.dto.UnassignResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
    fun `queue folds a non-null kind into the URL path via groupPath`() = runTest {
        val fakeService = FakeRegistryApiService(
            queueResponse = { groupPath ->
                if (groupPath == "my-group/chatbot") {
                    QueueResponseDto(entries = listOf(QueueEntryDto(path = "a.jsonl")))
                } else {
                    QueueResponseDto()
                }
            },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.queue("my-group", kind = "chatbot")

        assertTrue(result is ApiResult.Success)
        assertEquals(1, (result as ApiResult.Success).data.entries.size)
    }

    @Test
    fun `unassignModel sends paths, force, and an empty segments list when no segment is given`() = runTest {
        val fakeService = FakeRegistryApiService(
            unassignResponse = { _, _ -> Response.success(UnassignResponseDto(unassigned = 1)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.unassignModel("my-group", paths = listOf("a.jsonl"), force = true)

        assertTrue(result is ApiResult.Success)
        assertEquals(1, (result as ApiResult.Success).data.unassigned)
        val (group, body) = fakeService.unassignCalls.single()
        assertEquals("my-group", group)
        assertEquals(listOf("a.jsonl"), body.paths)
        assertTrue(body.force)
        assertTrue(body.segments.isEmpty())
    }

    @Test
    fun `unassignModel targets a segment instead of a whole-file path when given one`() = runTest {
        val fakeService = FakeRegistryApiService()
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        repository.unassignModel("my-group", segment = SegmentTargetDto("a.jsonl", 0, 50))

        val (_, body) = fakeService.unassignCalls.single()
        assertTrue(body.paths.isEmpty())
        assertEquals(listOf(SegmentTargetDto("a.jsonl", 0, 50)), body.segments)
    }

    @Test
    fun `deleteEntries sends force and delete_files through unchanged`() = runTest {
        val fakeService = FakeRegistryApiService(
            deleteResponse = { _, _ -> Response.success(DeleteResponseDto(deleted = 1)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.deleteEntries("my-group", paths = listOf("a.jsonl"), force = true, deleteFiles = true)

        assertTrue(result is ApiResult.Success)
        assertEquals(1, (result as ApiResult.Success).data.deleted)
        val (_, body) = fakeService.deleteCalls.single()
        assertEquals(listOf("a.jsonl"), body.paths)
        assertTrue(body.force)
        assertTrue(body.delete_files)
    }

    @Test
    fun `pendingAdd sends the path and segment range through unchanged`() = runTest {
        val fakeService = FakeRegistryApiService(
            pendingAddResponse = { _, _ -> Response.success(PendingAddResponseDto(added = true)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val result = repository.pendingAdd("my-group", path = "a.jsonl", segmentStart = 0, segmentCount = 10)

        assertTrue(result is ApiResult.Success)
        assertTrue((result as ApiResult.Success).data.added)
        val (_, body) = fakeService.pendingAddCalls.single()
        assertEquals("a.jsonl", body.path)
        assertEquals(0, body.segment_start)
        assertEquals(10, body.segment_count)
    }

    @Test
    fun `upload sends the filename and the exact bytes given`() = runTest {
        val fakeService = FakeRegistryApiService()
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())
        val bytes = byteArrayOf(1, 2, 3, 4)

        val result = repository.upload("my-group", filename = "new.jsonl", bytes = bytes)

        assertTrue(result is ApiResult.Success)
        val (group, filename, sentBytes) = fakeService.uploadCalls.single()
        assertEquals("my-group", group)
        assertEquals("new.jsonl", filename)
        assertTrue(bytes.contentEquals(sentBytes))
    }

    @Test
    fun `migrateToKind queues into destKind, restores the model assignment, then deletes from the legacy pool`() = runTest {
        val fakeService = FakeRegistryApiService(
            pendingAddResponse = { _, _ -> Response.success(PendingAddResponseDto(added = true)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())
        val entry = QueueEntryDto(path = "a.jsonl", model_name = "model-a")

        val result = repository.migrateToKind("my-group", entry, destKind = "chatbot")

        assertTrue(result is ApiResult.Success)
        assertEquals("a.jsonl", (result as ApiResult.Success).data)
        val (addGroup, addBody) = fakeService.pendingAddCalls.single()
        assertEquals("my-group/chatbot", addGroup)
        assertEquals("a.jsonl", addBody.path)
        val (assignGroup, assignBody) = fakeService.assignCalls.single()
        assertEquals("my-group/chatbot", assignGroup)
        assertEquals(listOf(SegmentTargetDto("a.jsonl", -1, -1)), assignBody.segments)
        val (deleteGroup, deleteBody) = fakeService.deleteCalls.single()
        assertEquals("my-group", deleteGroup)
        assertEquals(listOf(SegmentTargetDto("a.jsonl", -1, -1)), deleteBody.segments)
    }

    @Test
    fun `migrateToKind aborts before touching the source entry when pending-add fails`() = runTest {
        val fakeService = FakeRegistryApiService(
            pendingAddResponse = { _, _ -> Response.success(PendingAddResponseDto(added = false)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())
        val entry = QueueEntryDto(path = "a.jsonl", model_name = "model-a")

        val result = repository.migrateToKind("my-group", entry, destKind = "chatbot")

        assertTrue(result is ApiResult.ApiError)
        assertTrue(fakeService.assignCalls.isEmpty())
        assertTrue(fakeService.deleteCalls.isEmpty())
    }

    @Test
    fun `migrateToKind does not restore an assignment when the source entry was unassigned`() = runTest {
        val fakeService = FakeRegistryApiService(
            pendingAddResponse = { _, _ -> Response.success(PendingAddResponseDto(added = true)) },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())
        val entry = QueueEntryDto(path = "a.jsonl", model_name = "")

        repository.migrateToKind("my-group", entry, destKind = "chatbot")

        assertTrue(fakeService.assignCalls.isEmpty())
        assertFalse(fakeService.deleteCalls.isEmpty())
    }

    @Test
    fun `createSegments queues one pendingAdd per range and counts only the successful ones`() = runTest {
        var callCount = 0
        val fakeService = FakeRegistryApiService(
            pendingAddResponse = { _, _ ->
                callCount++
                Response.success(PendingAddResponseDto(added = callCount != 2))
            },
        )
        val repository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository())

        val created = repository.createSegments("my-group", "big.jsonl", ranges = listOf(0 to 5, 5 to 5, 10 to 5))

        assertEquals(2, created)
        assertEquals(3, fakeService.pendingAddCalls.size)
        val ranges = fakeService.pendingAddCalls.map { it.second.segment_start to it.second.segment_count }
        assertEquals(listOf(0 to 5, 5 to 5, 10 to 5), ranges)
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
