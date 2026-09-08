package com.adai.ops.network

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import com.adai.ops.network.dto.ErrorResponseDto
import java.io.IOException
import kotlinx.serialization.json.Json
import retrofit2.HttpException
import retrofit2.Response

private val errorJson = Json { ignoreUnknownKeys = true }

private fun parseErrorMessage(body: String?): String {
    if (body.isNullOrBlank()) return "Unknown error"
    return runCatching { errorJson.decodeFromString<ErrorResponseDto>(body).error ?: body }.getOrDefault(body)
}

/** For suspend calls that return the DTO directly (Retrofit throws HttpException on non-2xx). */
suspend fun <T> safeApiCall(block: suspend () -> T): ApiResult<T> = try {
    ApiResult.Success(block())
} catch (e: HttpException) {
    val message = parseErrorMessage(e.response()?.errorBody()?.string())
    when (e.code()) {
        404 -> ApiResult.NotFound
        409 -> ApiResult.Conflict(message)
        else -> ApiResult.ApiError("HTTP ${e.code()}: $message")
    }
} catch (e: IOException) {
    ApiResult.NetworkError(e.message ?: "Network error")
} catch (e: Exception) {
    ApiResult.NetworkError(e.message ?: "Unknown error")
}

/** For suspend calls that return Response<T> directly, so 404/409 are inspected without exceptions. */
suspend fun <T> safeResponseCall(block: suspend () -> Response<T>): ApiResult<T> = try {
    val response = block()
    if (response.isSuccessful) {
        response.body()?.let { ApiResult.Success(it) } ?: ApiResult.ApiError("Empty response body")
    } else {
        val message = parseErrorMessage(response.errorBody()?.string())
        when (response.code()) {
            404 -> ApiResult.NotFound
            409 -> ApiResult.Conflict(message)
            else -> ApiResult.ApiError("HTTP ${response.code()}: $message")
        }
    }
} catch (e: IOException) {
    ApiResult.NetworkError(e.message ?: "Network error")
} catch (e: Exception) {
    ApiResult.NetworkError(e.message ?: "Unknown error")
}
