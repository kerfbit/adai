package com.adai.ops.testutil

import com.adai.ops.network.TrainerApiService
import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerCheckpointResultDto
import com.adai.ops.network.dto.TrainerHealthDto
import com.adai.ops.network.dto.TrainerLogsResponseDto
import com.adai.ops.network.dto.TrainerPauseResultDto
import com.adai.ops.network.dto.TrainerStatusDto
import kotlinx.serialization.json.JsonObject
import retrofit2.Response

class FakeTrainerApiService(
    private val healthResponse: () -> TrainerHealthDto = { TrainerHealthDto(status = "ok") },
    private val getAdminConfigResponse: () -> Response<TrainerAdminConfigDto> =
        { Response.success(TrainerAdminConfigDto()) },
    private val putAdminConfigResponse: (JsonObject) -> Response<TrainerAdminConfigDto> =
        { Response.success(TrainerAdminConfigDto()) },
    private val getStatusResponse: () -> Response<TrainerStatusDto> = { Response.success(TrainerStatusDto()) },
    private val getLogsResponse: () -> Response<TrainerLogsResponseDto> = { Response.success(TrainerLogsResponseDto()) },
    private val checkpointResponse: (Int?) -> Response<TrainerCheckpointResultDto> =
        { Response.success(TrainerCheckpointResultDto()) },
    private val pauseResponse: () -> Response<TrainerPauseResultDto> = { Response.success(TrainerPauseResultDto(paused = true)) },
    private val resumeResponse: () -> Response<TrainerPauseResultDto> = { Response.success(TrainerPauseResultDto(paused = false)) },
) : TrainerApiService {

    val putAdminConfigCalls = mutableListOf<JsonObject>()
    val checkpointCalls = mutableListOf<Int?>()
    var pauseCallCount = 0
    var resumeCallCount = 0
    var getLogsCallCount = 0

    override suspend fun health(): TrainerHealthDto = healthResponse()

    override suspend fun getAdminConfig(): Response<TrainerAdminConfigDto> = getAdminConfigResponse()

    override suspend fun putAdminConfig(body: JsonObject): Response<TrainerAdminConfigDto> {
        putAdminConfigCalls += body
        return putAdminConfigResponse(body)
    }

    override suspend fun getStatus(): Response<TrainerStatusDto> = getStatusResponse()

    override suspend fun getLogs(): Response<TrainerLogsResponseDto> {
        getLogsCallCount++
        return getLogsResponse()
    }

    override suspend fun checkpoint(waitMs: Int?): Response<TrainerCheckpointResultDto> {
        checkpointCalls += waitMs
        return checkpointResponse(waitMs)
    }

    override suspend fun pause(): Response<TrainerPauseResultDto> {
        pauseCallCount++
        return pauseResponse()
    }

    override suspend fun resume(): Response<TrainerPauseResultDto> {
        resumeCallCount++
        return resumeResponse()
    }
}
