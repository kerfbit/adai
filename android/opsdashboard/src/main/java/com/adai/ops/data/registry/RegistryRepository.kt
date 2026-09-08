package com.adai.ops.data.registry

// @adai-status: beta
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.ApiClientProvider
import com.adai.ops.network.ApiResult
import com.adai.ops.network.RegistryApiService
import com.adai.ops.network.dto.AssignRequestDto
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.FetchGutenbergRequestDto
import com.adai.ops.network.dto.FetchHuggingfaceRequestDto
import com.adai.ops.network.dto.FetchResponseDto
import com.adai.ops.network.dto.HistoryResponseDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseRequestDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import com.adai.ops.network.safeApiCall
import com.adai.ops.network.safeResponseCall
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Client for registry_server. [forceRelease] is the original admin action exposed
 * here: an empty run_id in the release request bypasses the owner check server-side
 * (verified in RegistryTransport.cpp), returning files to the unassigned pool even
 * if a live trainer still holds them. [assignModel], [fetchGutenberg], and
 * [fetchHuggingface] are the Phase 14 additions backing the ops dashboard's
 * download-and-assign control panel.
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

    suspend fun queue(group: String): ApiResult<QueueResponseDto> = safeApiCall { service().queue(group) }

    suspend fun registry(group: String): ApiResult<RegistryResponseDto> = safeApiCall { service().registry(group) }

    suspend fun runs(group: String): ApiResult<RunsResponseDto> = safeApiCall { service().runs(group) }

    suspend fun history(group: String, modelId: String? = null): ApiResult<HistoryResponseDto> =
        safeApiCall { service().history(group, modelId) }

    /** Admin action: force-release files by sending an empty run_id, bypassing the owner check. */
    suspend fun forceRelease(group: String, files: List<String>): ApiResult<ReleaseResponseDto> =
        safeApiCall { service().release(group, ReleaseRequestDto(run_id = "", files = files)) }

    /** Assign [modelName] to [paths] (empty = every pending entry in the group). */
    suspend fun assignModel(
        group: String,
        modelName: String,
        paths: List<String>,
    ): ApiResult<AssignResponseDto> =
        safeApiCall { service().assign(group, AssignRequestDto(model_name = modelName, paths = paths)) }

    /** Trigger a server-side Gutenberg fetch; registry_server caches the book and rotates slices per model. */
    suspend fun fetchGutenberg(
        group: String,
        bookId: Int,
        numPairs: Int,
        modelName: String,
    ): ApiResult<FetchResponseDto> = safeResponseCall {
        service().fetchGutenberg(group, FetchGutenbergRequestDto(bookId, numPairs, modelName))
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
    ): ApiResult<FetchResponseDto> = safeResponseCall {
        service().fetchHuggingface(
            group,
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
