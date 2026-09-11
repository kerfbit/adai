package com.adai.ops.data.mns

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.1
// @adai-reviewed: 2026-09-10


import com.adai.ops.network.ApiClientProvider
import com.adai.ops.network.ApiResult
import com.adai.ops.network.MnsApiService
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.PromoteRequestDto
import com.adai.ops.network.dto.PromoteResultDto
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

    suspend fun listModels(state: String? = null, role: String? = null, limit: Int? = null): ApiResult<ModelsResponseDto> =
        safeApiCall { service().listModels(state, role, limit) }

    suspend fun getModel(name: String): ApiResult<ModelRecordDto> =
        safeResponseCall { service().getModel(name) }

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
