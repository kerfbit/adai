package com.adai.chatbot.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


sealed interface ApiResult<out T> {
    data class Success<T>(val data: T) : ApiResult<T>
    data class ApiError(val message: String) : ApiResult<Nothing>
    data class NetworkError(val message: String) : ApiResult<Nothing>
}
