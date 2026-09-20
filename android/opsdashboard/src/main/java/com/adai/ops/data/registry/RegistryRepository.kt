package com.adai.ops.data.registry

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-20


import com.adai.ops.network.ApiClientProvider
import com.adai.ops.network.ApiResult
import com.adai.ops.network.RegistryApiService
import com.adai.ops.network.dto.AssignRequestDto
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.DeleteRequestDto
import com.adai.ops.network.dto.DeleteResponseDto
import com.adai.ops.network.dto.FetchGutenbergRequestDto
import com.adai.ops.network.dto.FetchHuggingfaceRequestDto
import com.adai.ops.network.dto.FetchResponseDto
import com.adai.ops.network.dto.HistoryResponseDto
import com.adai.ops.network.dto.PendingAddRequestDto
import com.adai.ops.network.dto.PendingAddResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseRequestDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import com.adai.ops.network.dto.SegmentTargetDto
import com.adai.ops.network.dto.UnassignRequestDto
import com.adai.ops.network.dto.UnassignResponseDto
import com.adai.ops.network.safeApiCall
import com.adai.ops.network.safeResponseCall
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * Client for registry_server. [forceRelease] is the original admin action exposed
 * here: an empty run_id in the release request bypasses the owner check server-side
 * (verified in RegistryTransport.cpp), returning files to the unassigned pool even
 * if a live trainer still holds them. [assignModel], [fetchGutenberg], and
 * [fetchHuggingface] are the Phase 14 additions backing the ops dashboard's
 * download-and-assign control panel.
 *
 * TD-202/TD-205: every method takes an optional [kind] — "encoder"/"decoder"/"world_model"/
 * "chatbot", empty/null = the legacy shared pool — which [groupPath] folds into the URL
 * segment the server's route regex expects. [unassignModel], [deleteEntries], [pendingAdd],
 * and [upload] are new; [migrateToKind] and [createSegments] are client-side composites of
 * existing calls (no new server endpoint needed), mirroring `dataset_manager migrate`'s and
 * `dataset_manager segment`'s own exact sequencing (DatasetManagerTool.cpp).
 */
class RegistryRepository(
    private val apiClientProvider: ApiClientProvider,
    private val settingsRepository: OpsSettingsRepository,
) {
    private suspend fun service(): RegistryApiService {
        val s = settingsRepository.settings.first()
        return apiClientProvider.serviceFor(
            s.effectiveRegistryHost,
            s.registryPort,
            RegistryApiService::class.java,
            s.useHttpsRelay,
            s.accessClientId,
            s.accessClientSecret,
        )
    }

    /** "group" or "group/kind" — the single place this repository builds a kind-scoped path. */
    private fun groupPath(group: String, kind: String? = null): String =
        if (kind.isNullOrEmpty()) group else "$group/$kind"

    suspend fun queue(group: String, kind: String? = null): ApiResult<QueueResponseDto> =
        safeApiCall { service().queue(groupPath(group, kind)) }

    suspend fun registry(group: String, kind: String? = null): ApiResult<RegistryResponseDto> =
        safeApiCall { service().registry(groupPath(group, kind)) }

    suspend fun runs(group: String, kind: String? = null): ApiResult<RunsResponseDto> =
        safeApiCall { service().runs(groupPath(group, kind)) }

    suspend fun history(
        group: String,
        modelId: String? = null,
        kind: String? = null,
    ): ApiResult<HistoryResponseDto> = safeApiCall { service().history(groupPath(group, kind), modelId) }

    /** Admin action: force-release files by sending an empty run_id, bypassing the owner check. */
    suspend fun forceRelease(
        group: String,
        files: List<String>,
        kind: String? = null,
    ): ApiResult<ReleaseResponseDto> =
        safeApiCall { service().release(groupPath(group, kind), ReleaseRequestDto(run_id = "", files = files)) }

    /** TD-205: release exactly one segment-targeted entry (see [QueueEntryDto.isSegment]). */
    suspend fun forceReleaseSegment(
        group: String,
        segment: SegmentTargetDto,
        kind: String? = null,
    ): ApiResult<ReleaseResponseDto> = safeApiCall {
        service().release(
            groupPath(group, kind),
            ReleaseRequestDto(run_id = "", files = emptyList(), segments = listOf(segment)),
        )
    }

    /** Assign [modelName] to [paths] (empty = every pending entry in the group). */
    suspend fun assignModel(
        group: String,
        modelName: String,
        paths: List<String>,
        kind: String? = null,
    ): ApiResult<AssignResponseDto> = safeApiCall {
        service().assign(groupPath(group, kind), AssignRequestDto(model_name = modelName, paths = paths))
    }

    /** TD-205: assign [modelName] to one specific segment-targeted entry. */
    suspend fun assignModelToSegment(
        group: String,
        modelName: String,
        segment: SegmentTargetDto,
        kind: String? = null,
    ): ApiResult<AssignResponseDto> = safeApiCall {
        service().assign(
            groupPath(group, kind),
            AssignRequestDto(model_name = modelName, segments = listOf(segment)),
        )
    }

    /**
     * TD-205: clear a pending entry's model assignment — the natural counterpart to
     * [assignModel]/[assignModelToSegment]. Pass [segment] to target one specific segment
     * instead of a whole-file [paths] entry.
     */
    suspend fun unassignModel(
        group: String,
        modelName: String = "",
        paths: List<String> = emptyList(),
        force: Boolean = false,
        segment: SegmentTargetDto? = null,
        kind: String? = null,
    ): ApiResult<UnassignResponseDto> = safeResponseCall {
        service().unassign(
            groupPath(group, kind),
            UnassignRequestDto(
                model_name = modelName,
                paths = paths,
                force = force,
                segments = segment?.let { listOf(it) } ?: emptyList(),
            ),
        )
    }

    /**
     * TD-205: permanently purge entries matching [paths]/[segment] from both the pending queue
     * and the trained registry — needed for real dataset cleanup, not just organizing.
     */
    suspend fun deleteEntries(
        group: String,
        paths: List<String> = emptyList(),
        force: Boolean = false,
        deleteFiles: Boolean = false,
        segment: SegmentTargetDto? = null,
        kind: String? = null,
    ): ApiResult<DeleteResponseDto> = safeResponseCall {
        service().delete(
            groupPath(group, kind),
            DeleteRequestDto(
                paths = paths,
                force = force,
                delete_files = deleteFiles,
                segments = segment?.let { listOf(it) } ?: emptyList(),
            ),
        )
    }

    /**
     * TD-205: queue an already-existing (server-readable) path — the manual-add half of
     * "Manual add + Upload". [segmentStart]/[segmentCount] queue a specific row-range segment
     * instead of the whole file (see DatasetRegistry::count_pairs() server-side reasoning —
     * the caller is responsible for knowing the file's real pair count first).
     */
    suspend fun pendingAdd(
        group: String,
        path: String,
        segmentStart: Int = -1,
        segmentCount: Int = -1,
        kind: String? = null,
    ): ApiResult<PendingAddResponseDto> = safeResponseCall {
        service().pendingAdd(
            groupPath(group, kind),
            PendingAddRequestDto(path = path, segment_start = segmentStart, segment_count = segmentCount),
        )
    }

    /**
     * TD-205: upload a local file's raw bytes directly from the device — the upload half of
     * "Manual add + Upload". [filename] is a bare basename (no path separators, matching the
     * server's own `is_safe_upload_filename` check).
     */
    suspend fun upload(
        group: String,
        filename: String,
        bytes: ByteArray,
        kind: String? = null,
    ): ApiResult<PendingAddResponseDto> = safeResponseCall {
        val body = bytes.toRequestBody("application/octet-stream".toMediaType())
        service().upload(groupPath(group, kind), filename, body)
    }

    /**
     * TD-205: move a legacy (unkinded) pending [entry] into [destKind]'s own sub-pool —
     * replicates `dataset_manager migrate`'s exact sequence (DatasetManagerTool.cpp:534-623):
     * pending/add into the destination kind, restore the model assignment (if any) via a
     * follow-up assign (the pending-add wire format has no model_name field of its own), then
     * delete the original from the legacy pool. Source is always the legacy pool by design —
     * same as the CLI command, this never migrates between two non-legacy kinds directly.
     *
     * Returns [ApiResult.Success] with the created entry's path on success. A failure at the
     * pending/add step aborts before touching the source entry (so nothing is lost); a failure
     * at the assign or delete step is reported but the entry is still queued at the
     * destination — callers should surface [ApiResult.ApiError]'s message either way.
     */
    suspend fun migrateToKind(
        group: String,
        entry: QueueEntryDto,
        destKind: String,
    ): ApiResult<String> {
        val addResult = pendingAdd(
            group = group,
            path = entry.path,
            segmentStart = entry.segment_start,
            segmentCount = entry.segment_count,
            kind = destKind,
        )
        if (addResult !is ApiResult.Success || !addResult.data.added) {
            return ApiResult.ApiError("Failed to queue '${entry.path}' into '$destKind'")
        }
        val target = SegmentTargetDto(entry.path, entry.segment_start, entry.segment_count)
        if (entry.model_name.isNotEmpty()) {
            assignModelToSegment(group, entry.model_name, target, kind = destKind)
        }
        // Source is always the legacy pool (kind = null), matching dataset_manager migrate.
        deleteEntries(group, force = true, deleteFiles = false, segment = target, kind = null)
        return ApiResult.Success(entry.path)
    }

    /**
     * TD-205: split [path]'s JSONL pairs into [ranges] (each a 0-based [start, start+count)
     * pair-index range) and queue each as its own independently-assignable/deletable pending
     * entry in [kind]'s pool — the Android counterpart to `dataset_manager segment`
     * (DatasetManagerTool.cpp). The caller is responsible for knowing [path]'s total pair
     * count first (e.g. from a prior successful add/fetch's own num_entries) — this method
     * does not itself count pairs, since that requires local file access the app doesn't have
     * for a path that lives only on the registry_server's own storage.
     *
     * Returns the number of segments successfully queued.
     */
    suspend fun createSegments(
        group: String,
        path: String,
        ranges: List<Pair<Int, Int>>,
        kind: String? = null,
    ): Int {
        var created = 0
        for ((start, count) in ranges) {
            val result = pendingAdd(group, path, segmentStart = start, segmentCount = count, kind = kind)
            if (result is ApiResult.Success && result.data.added) {
                created++
            }
        }
        return created
    }

    /** Trigger a server-side Gutenberg fetch; registry_server caches the book and rotates slices per model. */
    suspend fun fetchGutenberg(
        group: String,
        bookId: Int,
        numPairs: Int,
        modelName: String,
        kind: String? = null,
    ): ApiResult<FetchResponseDto> = safeResponseCall {
        service().fetchGutenberg(groupPath(group, kind), FetchGutenbergRequestDto(bookId, numPairs, modelName))
    }

    /** Trigger a server-side HuggingFace fetch; registry_server caches the dataset and rotates slices per model. */
    suspend fun fetchHuggingface(
        group: String,
        datasetId: String,
        numPairs: Int,
        split: String,
        inputField: String,
        outputField: String,
        modelName: String,
        kind: String? = null,
    ): ApiResult<FetchResponseDto> = safeResponseCall {
        service().fetchHuggingface(
            groupPath(group, kind),
            FetchHuggingfaceRequestDto(datasetId, numPairs, split, inputField, outputField, modelName),
        )
    }

    /**
     * GET /admin/config. Response is inspected (not thrown), so a 403 (admin disabled on the
     * server) surfaces as [ApiResult.ApiError] rather than an exception.
     */
    suspend fun getAdminConfig(): ApiResult<RegistryAdminConfigDto> =
        safeResponseCall { service().getAdminConfig() }

    /**
     * Admin action: PUT /admin/config with only {"ftp_token_ttl_minutes": ...} — a single-key
     * body, never the full round-tripped object (see RegistryAdminConfigDto's doc comment).
     */
    suspend fun updateFtpTokenTtlMinutes(minutes: Int): ApiResult<RegistryAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("ftp_token_ttl_minutes", minutes) }) }

    /** Admin action: PUT /admin/config with only {"ftp_max_sessions_per_run": ...}. */
    suspend fun updateFtpMaxSessionsPerRun(count: Int): ApiResult<RegistryAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("ftp_max_sessions_per_run", count) }) }
}
