package com.adai.ops.data.trainer

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.ApiClientProvider
import com.adai.ops.network.ApiResult
import com.adai.ops.network.TrainerApiService
import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerCheckpointResultDto
import com.adai.ops.network.dto.TrainerHealthDto
import com.adai.ops.network.dto.TrainerLogsResponseDto
import com.adai.ops.network.dto.TrainerPauseResultDto
import com.adai.ops.network.dto.TrainerStatusDto
import com.adai.ops.network.safeApiCall
import com.adai.ops.network.safeResponseCall
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Client for `incremental_trainer serve`'s admin API. [requestCheckpoint], [pause], and
 * [resume] are the three control actions this daemon adds beyond the GET/PUT
 * /admin/config pattern the other three daemons already have (see ModelRepository/
 * RegistryRepository/MetricsRepository for that half) — see CLAUDE.md "Incremental
 * trainer admin API".
 *
 * Uses its own Cloudflare Access credentials ([OpsSettings.trainerAccessClientId]/
 * [OpsSettings.trainerAccessClientSecret]), not the shared [OpsSettings.accessClientId]/
 * [OpsSettings.accessClientSecret] the other three repositories use — see
 * docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md's "Configure Cloudflare
 * Access": trainer.kerfbit.dev is gated by its own Access application/service token
 * so it can be revoked independently, since this admin API (unlike the other three)
 * can pause/resume/checkpoint a live training run.
 */
class TrainerRepository(
    private val apiClientProvider: ApiClientProvider,
    private val settingsRepository: OpsSettingsRepository,
) {
    private suspend fun service(): TrainerApiService {
        val s = settingsRepository.settings.first()
        return apiClientProvider.serviceFor(
            s.effectiveTrainerHost,
            s.trainerPort,
            TrainerApiService::class.java,
            s.useHttpsRelay,
            s.trainerAccessClientId,
            s.trainerAccessClientSecret,
        )
    }

    suspend fun health(): ApiResult<TrainerHealthDto> = safeApiCall { service().health() }

    /**
     * GET /admin/config. Response is inspected (not thrown), so a connection failure
     * (the admin port is disabled/not listening — see TrainerAdminConfigDto's doc
     * comment) surfaces as [ApiResult.NetworkError] the same way it would for any other
     * unreachable host.
     */
    suspend fun getAdminConfig(): ApiResult<TrainerAdminConfigDto> =
        safeResponseCall { service().getAdminConfig() }

    suspend fun getStatus(): ApiResult<TrainerStatusDto> = safeResponseCall { service().getStatus() }

    /**
     * GET /admin/logs — real entries from the daemon's own log ring buffer (see
     * TrainerLogEntryDto's doc comment), not a client-side summary.
     */
    suspend fun getLogs(): ApiResult<TrainerLogsResponseDto> = safeResponseCall { service().getLogs() }

    /** Admin action: PUT /admin/config with only {"auto_save_enabled": ...}. */
    suspend fun updateAutoSaveEnabled(enabled: Boolean): ApiResult<TrainerAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("auto_save_enabled", enabled) }) }

    /** Admin action: PUT /admin/config with only {"auto_save_every_samples": ...}. */
    suspend fun updateAutoSaveEverySamples(samples: Int): ApiResult<TrainerAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("auto_save_every_samples", samples) }) }

    /** Admin action: PUT /admin/config with only {"auto_save_every_minutes": ...}. */
    suspend fun updateAutoSaveEveryMinutes(minutes: Int): ApiResult<TrainerAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("auto_save_every_minutes", minutes) }) }

    /** Admin action: PUT /admin/config with only {"max_sessions_to_keep": ...}. */
    suspend fun updateMaxSessionsToKeep(count: Int): ApiResult<TrainerAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("max_sessions_to_keep", count) }) }

    /**
     * Admin action: POST /admin/checkpoint. [waitMs], if given, has the server poll for
     * up to that many milliseconds before responding, so [TrainerCheckpointResultDto.completed]
     * can reflect whether the checkpoint actually finished rather than merely being requested.
     */
    suspend fun requestCheckpoint(waitMs: Int? = null): ApiResult<TrainerCheckpointResultDto> =
        safeResponseCall { service().checkpoint(waitMs) }

    /** Admin action: POST /admin/pause — drains the current pass (if any) and keeps serving. */
    suspend fun pause(): ApiResult<TrainerPauseResultDto> = safeResponseCall { service().pause() }

    /** Admin action: POST /admin/resume — clears the pause flag and wakes the idle-poll sleep. */
    suspend fun resume(): ApiResult<TrainerPauseResultDto> = safeResponseCall { service().resume() }
}
