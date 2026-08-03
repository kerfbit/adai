package com.adai.ops.testutil

import com.adai.ops.network.RegistryApiService
import com.adai.ops.network.dto.AssignRequestDto
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.FetchGutenbergRequestDto
import com.adai.ops.network.dto.FetchHuggingfaceRequestDto
import com.adai.ops.network.dto.FetchResponseDto
import com.adai.ops.network.dto.HistoryResponseDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryHealthDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseRequestDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import retrofit2.Response

class FakeRegistryApiService(
    private val queueResponse: (String) -> QueueResponseDto = { QueueResponseDto() },
    private val registryResponse: (String) -> RegistryResponseDto = { RegistryResponseDto() },
    private val releaseResponse: (String, ReleaseRequestDto) -> ReleaseResponseDto = { _, _ -> ReleaseResponseDto() },
    private val assignResponse: (String, AssignRequestDto) -> AssignResponseDto = { _, _ -> AssignResponseDto() },
    private val fetchGutenbergResponse: (String, FetchGutenbergRequestDto) -> Response<FetchResponseDto> =
        { _, _ -> Response.success(FetchResponseDto()) },
    private val fetchHuggingfaceResponse: (String, FetchHuggingfaceRequestDto) -> Response<FetchResponseDto> =
        { _, _ -> Response.success(FetchResponseDto()) },
) : RegistryApiService {

    val releaseCalls = mutableListOf<Pair<String, ReleaseRequestDto>>()
    val assignCalls = mutableListOf<Pair<String, AssignRequestDto>>()
    val fetchGutenbergCalls = mutableListOf<Pair<String, FetchGutenbergRequestDto>>()
    val fetchHuggingfaceCalls = mutableListOf<Pair<String, FetchHuggingfaceRequestDto>>()

    override suspend fun queue(group: String): QueueResponseDto = queueResponse(group)

    override suspend fun registry(group: String): RegistryResponseDto = registryResponse(group)

    override suspend fun runs(group: String): RunsResponseDto = RunsResponseDto()

    override suspend fun history(group: String, modelId: String?): HistoryResponseDto = HistoryResponseDto()

    override suspend fun release(group: String, body: ReleaseRequestDto): ReleaseResponseDto {
        releaseCalls += group to body
        return releaseResponse(group, body)
    }

    override suspend fun assign(group: String, body: AssignRequestDto): AssignResponseDto {
        assignCalls += group to body
        return assignResponse(group, body)
    }

    override suspend fun fetchGutenberg(group: String, body: FetchGutenbergRequestDto): Response<FetchResponseDto> {
        fetchGutenbergCalls += group to body
        return fetchGutenbergResponse(group, body)
    }

    override suspend fun fetchHuggingface(
        group: String,
        body: FetchHuggingfaceRequestDto,
    ): Response<FetchResponseDto> {
        fetchHuggingfaceCalls += group to body
        return fetchHuggingfaceResponse(group, body)
    }

    override suspend fun health(): RegistryHealthDto = RegistryHealthDto()
}
