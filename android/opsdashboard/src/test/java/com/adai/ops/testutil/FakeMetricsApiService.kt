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
import retrofit2.Response

class FakeMetricsApiService(
    private val getAdminConfigResponse: () -> Response<MetricsAdminConfigDto> =
        { Response.success(MetricsAdminConfigDto()) },
    private val putAdminConfigResponse: (JsonObject) -> Response<MetricsAdminConfigDto> =
        { Response.success(MetricsAdminConfigDto()) },
) : MetricsApiService {

    val putAdminConfigCalls = mutableListOf<JsonObject>()

    override suspend fun listSessions(): SessionsResponseDto = SessionsResponseDto()

    override suspend fun currentMetrics(key: String): CurrentMetricsDto = CurrentMetricsDto()

    override suspend fun sessionStatus(key: String): SessionStatusDto = SessionStatusDto()

    override suspend fun epochHistory(key: String): EpochHistoryDto = EpochHistoryDto()

    override suspend fun sampleHistory(key: String, maxRecords: Int): SampleHistoryDto = SampleHistoryDto()

    override suspend fun dbHistory(key: String, from: String?, to: String?, limit: Int): DbHistoryDto = DbHistoryDto()

    override suspend fun generationQuality(key: String): GenerationQualityDto = GenerationQualityDto()

    override suspend fun paddingEfficiency(key: String): PaddingEfficiencyDto = PaddingEfficiencyDto()

    override suspend fun aggregate(): AggregateMetricsDto = AggregateMetricsDto()

    override suspend fun health(): MetricsHealthDto = MetricsHealthDto()

    override suspend fun endSession(key: String): SimpleStatusDto = SimpleStatusDto()

    override suspend fun getAdminConfig(): Response<MetricsAdminConfigDto> = getAdminConfigResponse()

    override suspend fun putAdminConfig(body: JsonObject): Response<MetricsAdminConfigDto> {
        putAdminConfigCalls += body
        return putAdminConfigResponse(body)
    }
}
