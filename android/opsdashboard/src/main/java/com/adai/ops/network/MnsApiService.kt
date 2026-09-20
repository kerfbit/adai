package com.adai.ops.network

// @adai-status: beta        (TD-196 — kind filter + registerModel/linkWorldModel added)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-19


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
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query

/**
 * Client for mns_server (default port 8083). Read-only monitoring plus admin actions: setState
 * (used to clear a stale training lock), promote, registerModel, and linkWorldModel (TD-196).
 * Endpoints that can meaningfully 404/409 return Response<T> so callers can branch
 * on the status code without relying on exceptions.
 */
interface MnsApiService {

    @GET("models")
    suspend fun listModels(
        @Query("state") state: String? = null,
        @Query("role") role: String? = null,
        @Query("kind") kind: String? = null,
        @Query("limit") limit: Int? = null,
    ): ModelsResponseDto

    @GET("models/{name}")
    suspend fun getModel(@Path("name") name: String): Response<ModelRecordDto>

    /** TD-196: per-kind validation server-side — 400/409 surface via safeResponseCall. */
    @POST("models")
    suspend fun registerModel(@Body body: RegisterModelRequestDto): Response<RegisterModelResultDto>

    /** TD-196: empty world_model_name detaches. 400/404/409 surface via safeResponseCall. */
    @POST("models/{name}/link-world-model")
    suspend fun linkWorldModel(
        @Path("name") name: String,
        @Body body: LinkWorldModelRequestDto,
    ): Response<LinkWorldModelResultDto>

    @GET("models/{name}/resolve")
    suspend fun resolveModel(@Path("name") name: String): Response<ResolvedModelDto>

    @PUT("models/{name}/state")
    suspend fun setState(@Path("name") name: String, @Body body: SetStateRequestDto): Response<ModelRecordDto>

    @GET("roles")
    suspend fun listRoles(): RolesResponseDto

    @GET("roles/{role}/production")
    suspend fun resolveRoleProduction(@Path("role") role: String): Response<ResolvedModelDto>

    @PUT("roles/{role}/production")
    suspend fun promote(@Path("role") role: String, @Body body: PromoteRequestDto): Response<PromoteResultDto>

    @GET("health")
    suspend fun health(): MnsHealthDto

    /** 403 (Response inspected, not thrown) when the server was started with --admin-enabled=false. */
    @GET("admin/config")
    suspend fun getAdminConfig(): Response<MnsAdminConfigDto>

    /** [body] must contain only the field(s) actually changed — see MnsAdminConfigDto's doc comment. */
    @PUT("admin/config")
    suspend fun putAdminConfig(@Body body: JsonObject): Response<MnsAdminConfigDto>
}
