package com.adai.ops.testutil

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
import kotlinx.serialization.json.JsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import retrofit2.HttpException
import retrofit2.Response

/** Constructs a real HttpException the way Retrofit would for a non-2xx plain-DTO call —
 * sessionStatus()/currentMetrics()/etc. return the DTO directly (not Response<T>), so
 * safeApiCall's 404/409 mapping only ever sees this, never a Response it can inspect directly. */
fun httpException(code: Int, errorJson: String = "{\"error\":\"not found\"}"): HttpException =
    HttpException(Response.error<Any>(code, errorJson.toResponseBody()))

private fun String.toResponseBody() = ResponseBody.create("application/json".toMediaType(), this)

class FakeMetricsApiService(
    private val listSessionsResponse: () -> SessionsResponseDto = { SessionsResponseDto() },
    private val sessionStatusResponse: (String) -> SessionStatusDto = { SessionStatusDto() },
    private val currentMetricsResponse: (String) -> CurrentMetricsDto = { CurrentMetricsDto() },
    private val epochHistoryResponse: (String) -> EpochHistoryDto = { EpochHistoryDto() },
    private val sampleHistoryResponse: (String, Int) -> SampleHistoryDto = { _, _ -> SampleHistoryDto() },
    private val endSessionResponse: (String) -> SimpleStatusDto = { SimpleStatusDto() },
    private val getAdminConfigResponse: () -> Response<MetricsAdminConfigDto> =
        { Response.success(MetricsAdminConfigDto()) },
    private val putAdminConfigResponse: (JsonObject) -> Response<MetricsAdminConfigDto> =
        { Response.success(MetricsAdminConfigDto()) },
) : MetricsApiService {

    val putAdminConfigCalls = mutableListOf<JsonObject>()
    val endSessionCalls = mutableListOf<String>()

    override suspend fun listSessions(): SessionsResponseDto = listSessionsResponse()

    override suspend fun currentMetrics(key: String): CurrentMetricsDto = currentMetricsResponse(key)

    override suspend fun sessionStatus(key: String): SessionStatusDto = sessionStatusResponse(key)

    override suspend fun epochHistory(key: String): EpochHistoryDto = epochHistoryResponse(key)

    override suspend fun sampleHistory(key: String, maxRecords: Int): SampleHistoryDto =
        sampleHistoryResponse(key, maxRecords)

    override suspend fun dbHistory(key: String, from: String?, to: String?, limit: Int): DbHistoryDto = DbHistoryDto()

    override suspend fun generationQuality(key: String): GenerationQualityDto = GenerationQualityDto()

    override suspend fun paddingEfficiency(key: String): PaddingEfficiencyDto = PaddingEfficiencyDto()

    override suspend fun aggregate(): AggregateMetricsDto = AggregateMetricsDto()

    override suspend fun health(): MetricsHealthDto = MetricsHealthDto()

    override suspend fun endSession(key: String): SimpleStatusDto {
        endSessionCalls += key
        return endSessionResponse(key)
    }

    override suspend fun getAdminConfig(): Response<MetricsAdminConfigDto> = getAdminConfigResponse()

    override suspend fun putAdminConfig(body: JsonObject): Response<MetricsAdminConfigDto> {
        putAdminConfigCalls += body
        return putAdminConfigResponse(body)
    }
}
