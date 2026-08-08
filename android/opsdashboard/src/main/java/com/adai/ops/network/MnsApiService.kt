package com.adai.ops.network

import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.MnsHealthDto
import com.adai.ops.network.dto.PromoteRequestDto
import com.adai.ops.network.dto.PromoteResultDto
import com.adai.ops.network.dto.ResolvedModelDto
import com.adai.ops.network.dto.RolesResponseDto
import com.adai.ops.network.dto.SetStateRequestDto
import kotlinx.serialization.json.JsonObject
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query

/**
 * Client for mns_server (default port 8083). Read-only monitoring plus two admin
 * actions: setState (used to clear a stale training lock) and promote.
 * Endpoints that can meaningfully 404/409 return Response<T> so callers can branch
 * on the status code without relying on exceptions.
 */
interface MnsApiService {

    @GET("models")
    suspend fun listModels(
        @Query("state") state: String? = null,
        @Query("role") role: String? = null,
        @Query("limit") limit: Int? = null,
    ): ModelsResponseDto

    @GET("models/{name}")
    suspend fun getModel(@Path("name") name: String): Response<ModelRecordDto>

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
