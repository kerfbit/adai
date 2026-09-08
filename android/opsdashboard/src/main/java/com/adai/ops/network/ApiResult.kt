package com.adai.ops.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


sealed interface ApiResult<out T> {
    data class Success<T>(val data: T) : ApiResult<T>
    data object NotFound : ApiResult<Nothing>
    data class Conflict(val message: String) : ApiResult<Nothing>
    data class ApiError(val message: String) : ApiResult<Nothing>
    data class NetworkError(val message: String) : ApiResult<Nothing>
}

fun ApiResult<*>.errorMessageOrNull(): String? = when (this) {
    is ApiResult.Success -> null
    is ApiResult.NotFound -> "Not found"
    is ApiResult.Conflict -> message
    is ApiResult.ApiError -> message
    is ApiResult.NetworkError -> message
}
