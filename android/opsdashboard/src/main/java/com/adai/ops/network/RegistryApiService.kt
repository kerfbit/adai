package com.adai.ops.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-20

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
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path
import retrofit2.http.Query

/**
 * Client for registry_server (default port 8082). Groups have no discovery endpoint.
 *
 * TD-202/TD-205: every `registry/{groupPath}/...` route also accepts an optional kind segment
 * right after the group name — `/registry/{group}/{encoder|decoder|world_model|chatbot}/...` —
 * to target that trainable piece's own sub-pool instead of the legacy shared one. Rather than
 * doubling every method with a separate `kind` parameter, callers pass the already-combined
 * path segment as [groupPath] (`"group"` or `"group/kind"`) — see
 * [com.adai.ops.data.registry.RegistryRepository]'s own `groupPath()` helper, the single place
 * that builds this string, mirroring RemoteTransport::group_prefix_ (RegistryTransport.cpp) on
 * the C++ side. `encoded = true` lets the embedded "/" through Retrofit's path encoding.
 */
interface RegistryApiService {

    @GET("registry/{groupPath}/queue")
    suspend fun queue(@Path(value = "groupPath", encoded = true) groupPath: String): QueueResponseDto

    @GET("registry/{groupPath}/registry")
    suspend fun registry(@Path(value = "groupPath", encoded = true) groupPath: String): RegistryResponseDto

    @GET("registry/{groupPath}/runs")
    suspend fun runs(@Path(value = "groupPath", encoded = true) groupPath: String): RunsResponseDto

    @GET("registry/{groupPath}/history")
    suspend fun history(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Query("model_id") modelId: String? = null,
    ): HistoryResponseDto

    @POST("registry/{groupPath}/release")
    suspend fun release(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Body body: ReleaseRequestDto,
    ): ReleaseResponseDto

    @POST("registry/{groupPath}/assign")
    suspend fun assign(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Body body: AssignRequestDto,
    ): AssignResponseDto

    /** TD-205: the natural counterpart to [assign] — clears a pending entry's model_name. */
    @POST("registry/{groupPath}/unassign")
    suspend fun unassign(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Body body: UnassignRequestDto,
    ): Response<UnassignResponseDto>

    /** TD-205: permanently purge pending/trained entries, optionally unlinking the file. */
    @POST("registry/{groupPath}/delete")
    suspend fun delete(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Body body: DeleteRequestDto,
    ): Response<DeleteResponseDto>

    /**
     * TD-205: queue an already-existing (server-readable) path without fetching/uploading new
     * bytes — the manual-add half of "Manual add + Upload"; also how `migrateToKind()` and
     * `createSegments()` in the repository add each of their entries.
     */
    @POST("registry/{groupPath}/pending/add")
    suspend fun pendingAdd(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Body body: PendingAddRequestDto,
    ): Response<PendingAddResponseDto>

    /**
     * TD-205: upload a local file's raw bytes directly from the device — the upload half of
     * "Manual add + Upload". [body] must be a raw (non-JSON) RequestBody; see
     * RegistryRepository.upload() for how the caller builds one from a picked file's bytes.
     */
    @POST("registry/{groupPath}/upload")
    suspend fun upload(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Query("filename") filename: String,
        @Body body: RequestBody,
    ): Response<PendingAddResponseDto>

    @POST("registry/{groupPath}/fetch/gutenberg")
    suspend fun fetchGutenberg(
        @Path(value = "groupPath", encoded = true) groupPath: String,
        @Body body: FetchGutenbergRequestDto,
    ): Response<FetchResponseDto>

    @POST("registry/{groupPath}/fetch/huggingface")
    suspend fun fetchHuggingface(
        @Path(value = "groupPath", encoded = true) groupPath: String,
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
