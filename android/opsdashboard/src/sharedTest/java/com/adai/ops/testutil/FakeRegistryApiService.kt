package com.adai.ops.testutil

// @adai-status: beta        (in-memory fake backing both plain-JVM and instrumented tests, TD-048)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-09-20


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
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.network.dto.RegistryHealthDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseRequestDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import com.adai.ops.network.dto.UnassignRequestDto
import com.adai.ops.network.dto.UnassignResponseDto
import kotlinx.serialization.json.JsonObject
import okhttp3.RequestBody
import retrofit2.Response

class FakeRegistryApiService(
    private val queueResponse: (String) -> QueueResponseDto = { QueueResponseDto() },
    private val registryResponse: (String) -> RegistryResponseDto = { RegistryResponseDto() },
    private val runsResponse: (String) -> RunsResponseDto = { RunsResponseDto() },
    private val releaseResponse: (String, ReleaseRequestDto) -> ReleaseResponseDto = { _, _ -> ReleaseResponseDto() },
    private val assignResponse: (String, AssignRequestDto) -> AssignResponseDto = { _, _ -> AssignResponseDto() },
    private val unassignResponse: (String, UnassignRequestDto) -> Response<UnassignResponseDto> =
        { _, _ -> Response.success(UnassignResponseDto()) },
    private val deleteResponse: (String, DeleteRequestDto) -> Response<DeleteResponseDto> =
        { _, _ -> Response.success(DeleteResponseDto()) },
    private val pendingAddResponse: (String, PendingAddRequestDto) -> Response<PendingAddResponseDto> =
        { _, _ -> Response.success(PendingAddResponseDto(added = true)) },
    private val uploadResponse: (String, String) -> Response<PendingAddResponseDto> =
        { _, _ -> Response.success(PendingAddResponseDto(added = true)) },
    private val fetchGutenbergResponse: (String, FetchGutenbergRequestDto) -> Response<FetchResponseDto> =
        { _, _ -> Response.success(FetchResponseDto()) },
    private val fetchHuggingfaceResponse: (String, FetchHuggingfaceRequestDto) -> Response<FetchResponseDto> =
        { _, _ -> Response.success(FetchResponseDto()) },
    private val getAdminConfigResponse: () -> Response<RegistryAdminConfigDto> =
        { Response.success(RegistryAdminConfigDto()) },
    private val putAdminConfigResponse: (JsonObject) -> Response<RegistryAdminConfigDto> =
        { Response.success(RegistryAdminConfigDto()) },
) : RegistryApiService {

    val releaseCalls = mutableListOf<Pair<String, ReleaseRequestDto>>()
    val assignCalls = mutableListOf<Pair<String, AssignRequestDto>>()
    val unassignCalls = mutableListOf<Pair<String, UnassignRequestDto>>()
    val deleteCalls = mutableListOf<Pair<String, DeleteRequestDto>>()
    val pendingAddCalls = mutableListOf<Pair<String, PendingAddRequestDto>>()
    val uploadCalls = mutableListOf<Triple<String, String, ByteArray>>()
    val fetchGutenbergCalls = mutableListOf<Pair<String, FetchGutenbergRequestDto>>()
    val fetchHuggingfaceCalls = mutableListOf<Pair<String, FetchHuggingfaceRequestDto>>()
    val putAdminConfigCalls = mutableListOf<JsonObject>()

    override suspend fun queue(group: String): QueueResponseDto = queueResponse(group)

    override suspend fun registry(group: String): RegistryResponseDto = registryResponse(group)

    override suspend fun runs(group: String): RunsResponseDto = runsResponse(group)

    override suspend fun history(group: String, modelId: String?): HistoryResponseDto = HistoryResponseDto()

    override suspend fun release(group: String, body: ReleaseRequestDto): ReleaseResponseDto {
        releaseCalls += group to body
        return releaseResponse(group, body)
    }

    override suspend fun assign(group: String, body: AssignRequestDto): AssignResponseDto {
        assignCalls += group to body
        return assignResponse(group, body)
    }

    override suspend fun unassign(group: String, body: UnassignRequestDto): Response<UnassignResponseDto> {
        unassignCalls += group to body
        return unassignResponse(group, body)
    }

    override suspend fun delete(group: String, body: DeleteRequestDto): Response<DeleteResponseDto> {
        deleteCalls += group to body
        return deleteResponse(group, body)
    }

    override suspend fun pendingAdd(group: String, body: PendingAddRequestDto): Response<PendingAddResponseDto> {
        pendingAddCalls += group to body
        return pendingAddResponse(group, body)
    }

    override suspend fun upload(
        group: String,
        filename: String,
        body: RequestBody,
    ): Response<PendingAddResponseDto> {
        val buffer = okio.Buffer()
        body.writeTo(buffer)
        uploadCalls += Triple(group, filename, buffer.readByteArray())
        return uploadResponse(group, filename)
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

    override suspend fun getAdminConfig(): Response<RegistryAdminConfigDto> = getAdminConfigResponse()

    override suspend fun putAdminConfig(body: JsonObject): Response<RegistryAdminConfigDto> {
        putAdminConfigCalls += body
        return putAdminConfigResponse(body)
    }
}
