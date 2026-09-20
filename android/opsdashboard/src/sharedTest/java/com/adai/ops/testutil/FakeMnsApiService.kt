package com.adai.ops.testutil

// @adai-status: beta        (TD-196 — kind filter + registerModel/linkWorldModel added)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-19


import com.adai.ops.network.MnsApiService
import com.adai.ops.network.dto.LinkWorldModelRequestDto
import com.adai.ops.network.dto.LinkWorldModelResultDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.MnsHealthDto
import com.adai.ops.network.dto.PromoteRequestDto
import com.adai.ops.network.dto.PromoteResultDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.network.dto.RegisterModelResultDto
import com.adai.ops.network.dto.ResolvedModelDto
import com.adai.ops.network.dto.RolesResponseDto
import com.adai.ops.network.dto.SetStateRequestDto
import kotlinx.serialization.json.JsonObject
import retrofit2.Response

class FakeMnsApiService(
    private val listModelsResponse: (String?, String?, String?, Int?) -> ModelsResponseDto =
        { _, _, _, _ -> ModelsResponseDto() },
    private val getModelResponse: (String) -> Response<ModelRecordDto> = { Response.success(ModelRecordDto()) },
    private val setStateResponse: (String, SetStateRequestDto) -> Response<ModelRecordDto> =
        { _, _ -> Response.success(ModelRecordDto()) },
    private val registerModelResponse: (RegisterModelRequestDto) -> Response<RegisterModelResultDto> =
        { Response.success(RegisterModelResultDto()) },
    private val linkWorldModelResponse: (String, LinkWorldModelRequestDto) -> Response<LinkWorldModelResultDto> =
        { _, body -> Response.success(LinkWorldModelResultDto(status = "ok", world_model_name = body.world_model_name)) },
    private val promoteResponse: (String, PromoteRequestDto) -> Response<PromoteResultDto> =
        { _, _ -> Response.success(PromoteResultDto()) },
    private val getAdminConfigResponse: () -> Response<MnsAdminConfigDto> = { Response.success(MnsAdminConfigDto()) },
    private val putAdminConfigResponse: (JsonObject) -> Response<MnsAdminConfigDto> =
        { Response.success(MnsAdminConfigDto()) },
) : MnsApiService {

    val setStateCalls = mutableListOf<Pair<String, SetStateRequestDto>>()
    val registerModelCalls = mutableListOf<RegisterModelRequestDto>()
    val linkWorldModelCalls = mutableListOf<Pair<String, LinkWorldModelRequestDto>>()
    val promoteCalls = mutableListOf<Pair<String, PromoteRequestDto>>()
    val putAdminConfigCalls = mutableListOf<JsonObject>()

    override suspend fun listModels(state: String?, role: String?, kind: String?, limit: Int?): ModelsResponseDto =
        listModelsResponse(state, role, kind, limit)

    override suspend fun getModel(name: String): Response<ModelRecordDto> = getModelResponse(name)

    override suspend fun registerModel(body: RegisterModelRequestDto): Response<RegisterModelResultDto> {
        registerModelCalls += body
        return registerModelResponse(body)
    }

    override suspend fun linkWorldModel(name: String, body: LinkWorldModelRequestDto): Response<LinkWorldModelResultDto> {
        linkWorldModelCalls += name to body
        return linkWorldModelResponse(name, body)
    }

    override suspend fun resolveModel(name: String): Response<ResolvedModelDto> = Response.success(ResolvedModelDto())

    override suspend fun setState(name: String, body: SetStateRequestDto): Response<ModelRecordDto> {
        setStateCalls += name to body
        return setStateResponse(name, body)
    }

    override suspend fun listRoles(): RolesResponseDto = RolesResponseDto()

    override suspend fun resolveRoleProduction(role: String): Response<ResolvedModelDto> = Response.success(ResolvedModelDto())

    override suspend fun promote(role: String, body: PromoteRequestDto): Response<PromoteResultDto> {
        promoteCalls += role to body
        return promoteResponse(role, body)
    }

    override suspend fun health(): MnsHealthDto = MnsHealthDto()

    override suspend fun getAdminConfig(): Response<MnsAdminConfigDto> = getAdminConfigResponse()

    override suspend fun putAdminConfig(body: JsonObject): Response<MnsAdminConfigDto> {
        putAdminConfigCalls += body
        return putAdminConfigResponse(body)
    }
}
