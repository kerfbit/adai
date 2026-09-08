package com.adai.ops.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


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
import kotlinx.serialization.json.JsonObject
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query

/**
 * Client for metrics_api_server (default port 8081). Mostly read-only; [endSession] is
 * the one admin action, for manually finalizing a session whose trainer crashed and
 * never called its own /end.
 */
interface MetricsApiService {

    @GET("api/sessions")
    suspend fun listSessions(): SessionsResponseDto

    @GET("api/sessions/{key}/metrics/current")
    suspend fun currentMetrics(@Path("key") key: String): CurrentMetricsDto

    @GET("api/sessions/{key}/status")
    suspend fun sessionStatus(@Path("key") key: String): SessionStatusDto

    @GET("api/sessions/{key}/epochs")
    suspend fun epochHistory(@Path("key") key: String): EpochHistoryDto

    @GET("api/sessions/{key}/metrics/history")
    suspend fun sampleHistory(
        @Path("key") key: String,
        @Query("max_records") maxRecords: Int = 200,
    ): SampleHistoryDto

    /**
     * SQL-backed, not capped by the 10,000-record in-memory ring buffer that
     * [sampleHistory] draws from — used by the watch-face relay to get the whole
     * session's shape rather than just a recent window. Errors (e.g. HTTP 4xx/5xx body)
     * if the server is running with `METRICS_STORAGE_BACKEND=file` (no DB configured);
     * callers should fall back to [sampleHistory] with a high `maxRecords` in that case.
     */
    @GET("api/sessions/{key}/metrics/db-history")
    suspend fun dbHistory(
        @Path("key") key: String,
        @Query("from") from: String? = null,
        @Query("to") to: String? = null,
        @Query("limit") limit: Int = 0,
    ): DbHistoryDto

    @GET("api/sessions/{key}/metrics/generation-quality")
    suspend fun generationQuality(@Path("key") key: String): GenerationQualityDto

    @GET("api/sessions/{key}/metrics/padding-efficiency")
    suspend fun paddingEfficiency(@Path("key") key: String): PaddingEfficiencyDto

    @GET("api/metrics/aggregate")
    suspend fun aggregate(): AggregateMetricsDto

    @GET("health")
    suspend fun health(): MetricsHealthDto

    /** Admin action: finalizes a session — the manual override for a crashed/hung trainer. */
    @POST("api/sessions/{key}/end")
    suspend fun endSession(@Path("key") key: String): SimpleStatusDto

    /**
     * Unlike mns_server/registry_server, this route isn't registered at all when the server
     * was started with allow_control=false, so both this and [putAdminConfig] 404 rather
     * than 403 in that case — Response is still inspected, not thrown, either way.
     */
    @GET("admin/config")
    suspend fun getAdminConfig(): Response<MetricsAdminConfigDto>

    /** [body] must contain only the field(s) actually changed — see MetricsAdminConfigDto's doc comment. */
    @PUT("admin/config")
    suspend fun putAdminConfig(@Body body: JsonObject): Response<MetricsAdminConfigDto>
}
