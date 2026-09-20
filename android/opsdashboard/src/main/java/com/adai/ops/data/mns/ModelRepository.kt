package com.adai.ops.data.mns

// @adai-status: beta        (TD-196 — kind filter + registerModel/linkWorldModel added)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-19


import com.adai.ops.network.ApiClientProvider
import com.adai.ops.network.ApiResult
import com.adai.ops.network.MnsApiService
import com.adai.ops.network.dto.LinkWorldModelRequestDto
import com.adai.ops.network.dto.LinkWorldModelResultDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.PromoteRequestDto
import com.adai.ops.network.dto.PromoteResultDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.network.dto.RegisterModelResultDto
import com.adai.ops.network.dto.ResolvedModelDto
import com.adai.ops.network.dto.RolesResponseDto
import com.adai.ops.network.dto.SetStateRequestDto
import com.adai.ops.network.safeApiCall
import com.adai.ops.network.safeResponseCall
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Client for mns_server. [clearStaleTrainingLock], [retireCandidate], and
 * [promoteToProduction] are the admin actions exposed by this app — all explicit,
 * confirm-before-send operations driven from the UI, never called automatically.
 */
class ModelRepository(
    private val apiClientProvider: ApiClientProvider,
    private val settingsRepository: OpsSettingsRepository,
) {
    private suspend fun service(): MnsApiService {
        val s = settingsRepository.settings.first()
        return apiClientProvider.serviceFor(
            s.effectiveMnsHost,
            s.mnsPort,
            MnsApiService::class.java,
            s.useHttpsRelay,
            s.accessClientId,
            s.accessClientSecret,
        )
    }

    suspend fun listModels(
        state: String? = null,
        role: String? = null,
        kind: String? = null,
        limit: Int? = null,
    ): ApiResult<ModelsResponseDto> = safeApiCall { service().listModels(state, role, kind, limit) }

    suspend fun getModel(name: String): ApiResult<ModelRecordDto> =
        safeResponseCall { service().getModel(name) }

    /**
     * TD-196: POST /models. Per-kind validation is entirely server-side (own layer count,
     * encoder/decoder dimensional match, world-model link compatibility) — a 400/409 surfaces
     * here as [ApiResult.ApiError]/[ApiResult.Conflict] with the server's own message, no
     * client-side re-validation needed beyond basic form completeness.
     */
    suspend fun registerModel(request: RegisterModelRequestDto): ApiResult<RegisterModelResultDto> =
        safeResponseCall { service().registerModel(request) }

    /**
     * Admin action: POST /models/{name}/link-world-model. Only valid on a chatbot-kind record;
     * an empty world_model_name in [request] detaches. 409 on a d_model mismatch whenever
     * world_model_inject_every_n_layers > 0.
     */
    suspend fun linkWorldModel(
        chatbotName: String,
        request: LinkWorldModelRequestDto,
    ): ApiResult<LinkWorldModelResultDto> = safeResponseCall { service().linkWorldModel(chatbotName, request) }

    suspend fun resolveModel(name: String): ApiResult<ResolvedModelDto> =
        safeResponseCall { service().resolveModel(name) }

    suspend fun listRoles(): ApiResult<RolesResponseDto> = safeApiCall { service().listRoles() }

    suspend fun resolveRoleProduction(role: String): ApiResult<ResolvedModelDto> =
        safeResponseCall { service().resolveRoleProduction(role) }

    /**
     * Admin action: PUT /models/{name}/state {"state":"candidate","run_id":runId}.
     * TD-147: ModelNameService.cpp's "candidate" transition has an ownership check that
     * fires whenever the model is currently "training" — the exact state this action
     * exists to recover from — and rejects the request with 409 unless [runId] matches
     * the model's own current run_id exactly (an *empty* run_id does NOT bypass the
     * check the way DatasetRegistry's release endpoint does; it fails the check like
     * any other mismatch). The previous version never sent a run_id at all, so every
     * real invocation — the button is only enabled when state=="training" — was
     * rejected. [runId] must be the run_id from the most recently fetched
     * [ModelRecordDto] for this model (the caller already has this from its polled
     * state); a stale/mismatched value correctly still 409s, which is the intended
     * protection against clobbering a run that was superseded by a new one.
     */
    suspend fun clearStaleTrainingLock(name: String, runId: String): ApiResult<ModelRecordDto> =
        safeResponseCall { service().setState(name, SetStateRequestDto(state = "candidate", run_id = runId)) }

    /**
     * Admin action: PUT /models/{name}/state {"state":"retired"}. For discarding a
     * candidate that isn't worth promoting; valid from any non-retired state.
     */
    suspend fun retireCandidate(name: String): ApiResult<ModelRecordDto> =
        safeResponseCall { service().setState(name, SetStateRequestDto(state = "retired")) }

    /**
     * Admin action: PUT /roles/{role}/production {"model_name":name}. Requires the
     * model to be in "candidate" state; retires the prior production model for the role.
     */
    suspend fun promoteToProduction(role: String, modelName: String): ApiResult<PromoteResultDto> =
        safeResponseCall { service().promote(role, PromoteRequestDto(model_name = modelName)) }

    /**
     * GET /admin/config. Response is inspected (not thrown), so a 403 (admin disabled on the
     * server) surfaces as [ApiResult.ApiError] rather than an exception.
     */
    suspend fun getAdminConfig(): ApiResult<MnsAdminConfigDto> = safeResponseCall { service().getAdminConfig() }

    /**
     * Admin action: PUT /admin/config with only {"registry_url": ...} — a single-key body,
     * never the full round-tripped object, since the server rejects any body containing an
     * immutable key name even unchanged (see MnsAdminConfigDto's doc comment).
     */
    suspend fun updateRegistryUrl(url: String): ApiResult<MnsAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("registry_url", url) }) }

    /** Admin action: PUT /admin/config with only {"registry_group": ...}. */
    suspend fun updateRegistryGroup(group: String): ApiResult<MnsAdminConfigDto> =
        safeResponseCall { service().putAdminConfig(buildJsonObject { put("registry_group", group) }) }
}
