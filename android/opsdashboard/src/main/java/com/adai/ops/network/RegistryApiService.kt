package com.adai.ops.network

import com.adai.ops.network.dto.AssignRequestDto
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.FetchGutenbergRequestDto
import com.adai.ops.network.dto.FetchHuggingfaceRequestDto
import com.adai.ops.network.dto.FetchResponseDto
import com.adai.ops.network.dto.HistoryResponseDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.network.dto.RegistryHealthDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.ReleaseRequestDto
import com.adai.ops.network.dto.ReleaseResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import kotlinx.serialization.json.JsonObject
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query

/** Client for registry_server (default port 8082). Groups have no discovery endpoint. */
interface RegistryApiService {

    @GET("registry/{group}/queue")
    suspend fun queue(@Path("group") group: String): QueueResponseDto

    @GET("registry/{group}/registry")
    suspend fun registry(@Path("group") group: String): RegistryResponseDto

    @GET("registry/{group}/runs")
    suspend fun runs(@Path("group") group: String): RunsResponseDto

    @GET("registry/{group}/history")
    suspend fun history(
        @Path("group") group: String,
        @Query("model_id") modelId: String? = null,
    ): HistoryResponseDto

    @POST("registry/{group}/release")
    suspend fun release(@Path("group") group: String, @Body body: ReleaseRequestDto): ReleaseResponseDto

    @POST("registry/{group}/assign")
    suspend fun assign(@Path("group") group: String, @Body body: AssignRequestDto): AssignResponseDto

    @POST("registry/{group}/fetch/gutenberg")
    suspend fun fetchGutenberg(
        @Path("group") group: String,
        @Body body: FetchGutenbergRequestDto,
    ): Response<FetchResponseDto>

    @POST("registry/{group}/fetch/huggingface")
    suspend fun fetchHuggingface(
        @Path("group") group: String,
        @Body body: FetchHuggingfaceRequestDto,
    ): Response<FetchResponseDto>

    @GET("health")
    suspend fun health(): RegistryHealthDto

    /** 403 (Response inspected, not thrown) when the server was started with --admin-enabled=false. */
    @GET("admin/config")
    suspend fun getAdminConfig(): Response<RegistryAdminConfigDto>

    /** [body] must contain only the field(s) actually changed — see RegistryAdminConfigDto's doc comment. */
    @PUT("admin/config")
    suspend fun putAdminConfig(@Body body: JsonObject): Response<RegistryAdminConfigDto>
}
