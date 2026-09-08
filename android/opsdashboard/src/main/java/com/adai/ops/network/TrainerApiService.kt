package com.adai.ops.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerCheckpointResultDto
import com.adai.ops.network.dto.TrainerHealthDto
import com.adai.ops.network.dto.TrainerLogsResponseDto
import com.adai.ops.network.dto.TrainerPauseResultDto
import com.adai.ops.network.dto.TrainerStatusDto
import kotlinx.serialization.json.JsonObject
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Query

/**
 * Client for `incremental_trainer serve`'s admin API (default port 8084, opt-in — see
 * TRAINER_ADMIN_ENABLED in config.trainer.conf). Unlike metrics_api_server/mns_server/
 * registry_server, this port either isn't listening at all (disabled) or has no
 * admin-mutation gate once it is — there's no --admin-enabled equivalent — so every
 * endpoint here is always a live admin endpoint when reachable.
 */
interface TrainerApiService {

    @GET("health")
    suspend fun health(): TrainerHealthDto

    @GET("admin/config")
    suspend fun getAdminConfig(): Response<TrainerAdminConfigDto>

    /** [body] must contain only the field(s) actually changed — see TrainerAdminConfigDto's doc comment. */
    @PUT("admin/config")
    suspend fun putAdminConfig(@Body body: JsonObject): Response<TrainerAdminConfigDto>

    @GET("admin/status")
    suspend fun getStatus(): Response<TrainerStatusDto>

    /** Real daemon log entries — see TrainerLogEntryDto's doc comment. */
    @GET("admin/logs")
    suspend fun getLogs(): Response<TrainerLogsResponseDto>

    /** 409 (Response inspected, not thrown) when idle — nothing to checkpoint. */
    @POST("admin/checkpoint")
    suspend fun checkpoint(@Query("wait_ms") waitMs: Int? = null): Response<TrainerCheckpointResultDto>

    @POST("admin/pause")
    suspend fun pause(): Response<TrainerPauseResultDto>

    @POST("admin/resume")
    suspend fun resume(): Response<TrainerPauseResultDto>
}
