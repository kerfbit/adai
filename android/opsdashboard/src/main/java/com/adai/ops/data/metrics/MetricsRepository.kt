package com.adai.ops.data.metrics

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.ApiClientProvider
import com.adai.ops.network.ApiResult
import com.adai.ops.network.MetricsApiService
import com.adai.ops.network.dto.AggregateMetricsDto
import com.adai.ops.network.dto.CurrentMetricsDto
import com.adai.ops.network.dto.DbHistoryDto
import com.adai.ops.network.dto.EpochHistoryDto
import com.adai.ops.network.dto.GenerationQualityDto
import com.adai.ops.network.dto.MetricsAdminConfigDto
import com.adai.ops.network.dto.MetricsHealthDto
import com.adai.ops.network.dto.PaddingEfficiencyDto
import com.adai.ops.network.dto.SampleHistoryDto
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.network.dto.SessionsResponseDto
import com.adai.ops.network.dto.SimpleStatusDto
import com.adai.ops.network.safeApiCall
import com.adai.ops.network.safeResponseCall
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Client for metrics_api_server. Mostly read-only; [endSession] is the one admin
 * action — a manual override for a session whose trainer crashed and never called
 * its own /end, driven from an explicit ConfirmActionDialog confirmation in the UI.
 */
class MetricsRepository(
    private val apiClientProvider: ApiClientProvider,
    private val settingsRepository: OpsSettingsRepository,
) {
    private suspend fun service(): MetricsApiService {
        val s = settingsRepository.settings.first()
        return apiClientProvider.serviceFor(
            s.effectiveMetricsHost,
            s.metricsPort,
            MetricsApiService::class.java,
            s.useHttpsRelay,
            s.accessClientId,
            s.accessClientSecret,
        )
    }

    suspend fun listSessions(): ApiResult<SessionsResponseDto> = safeApiCall { service().listSessions() }

    suspend fun currentMetrics(key: String): ApiResult<CurrentMetricsDto> =
        safeApiCall { service().currentMetrics(key) }

    suspend fun sessionStatus(key: String): ApiResult<SessionStatusDto> =
        safeApiCall { service().sessionStatus(key) }

    suspend fun epochHistory(key: String): ApiResult<EpochHistoryDto> =
        safeApiCall { service().epochHistory(key) }

    suspend fun sampleHistory(key: String, maxRecords: Int = 200): ApiResult<SampleHistoryDto> =
        safeApiCall { service().sampleHistory(key, maxRecords) }

    suspend fun dbHistory(key: String, limit: Int = 0): ApiResult<DbHistoryDto> =
        safeApiCall { service().dbHistory(key = key, limit = limit) }

    suspend fun generationQuality(key: String): ApiResult<GenerationQualityDto> =
        safeApiCall { service().generationQuality(key) }

    suspend fun paddingEfficiency(key: String): ApiResult<PaddingEfficiencyDto> =
        safeApiCall { service().paddingEfficiency(key) }

    suspend fun aggregate(): ApiResult<AggregateMetricsDto> = safeApiCall { service().aggregate() }

    suspend fun health(): ApiResult<MetricsHealthDto> = safeApiCall { service().health() }

    suspend fun endSession(key: String): ApiResult<SimpleStatusDto> = safeApiCall { service().endSession(key) }

    /**
     * GET /admin/config. Unlike mns_server/registry_server, this route isn't registered at
     * all when the server was started with allow_control=false, so a disabled server surfaces
     * as [ApiResult.NotFound] here rather than an ApiError — see MetricsAdminConfigDto's doc
     * comment.
     */
    suspend fun getAdminConfig(): ApiResult<MetricsAdminConfigDto> =
        safeResponseCall { service().getAdminConfig() }

    private suspend fun putAdminConfigField(key: String, value: Int): ApiResult<MetricsAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put(key, value) }) }

    private suspend fun putAdminConfigField(key: String, value: Boolean): ApiResult<MetricsAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put(key, value) }) }

    /** Admin action: PUT /admin/config with only {"max_live_sessions": ...}. */
    suspend fun updateMaxLiveSessions(n: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("max_live_sessions", n)

    /** Admin action: PUT /admin/config with only {"completed_ttl_seconds": ...}. */
    suspend fun updateCompletedTtlSeconds(seconds: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("completed_ttl_seconds", seconds)

    /** Admin action: PUT /admin/config with only {"sweep_interval_seconds": ...}. */
    suspend fun updateSweepIntervalSeconds(seconds: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("sweep_interval_seconds", seconds)

    /** Admin action: PUT /admin/config with only {"persist_every_samples": ...}. */
    suspend fun updatePersistEverySamples(n: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("persist_every_samples", n)

    /** Admin action: PUT /admin/config with only {"persist_every_seconds": ...}. */
    suspend fun updatePersistEverySeconds(seconds: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("persist_every_seconds", seconds)

    /** Admin action: PUT /admin/config with only {"max_records_in_memory": ...}. */
    suspend fun updateMaxRecordsInMemory(n: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("max_records_in_memory", n)

    /** Admin action: PUT /admin/config with only {"max_records_on_disk": ...}. */
    suspend fun updateMaxRecordsOnDisk(n: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("max_records_on_disk", n)

    /** Admin action: PUT /admin/config with only {"enable_prometheus": ...}. */
    suspend fun updateEnablePrometheus(enabled: Boolean): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("enable_prometheus", enabled)

    /** Admin action: PUT /admin/config with only {"staleness_threshold_seconds": ...}. */
    suspend fun updateStalenessThresholdSeconds(seconds: Int): ApiResult<MetricsAdminConfigDto> =
        putAdminConfigField("staleness_threshold_seconds", seconds)
}
