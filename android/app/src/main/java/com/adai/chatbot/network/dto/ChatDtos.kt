package com.adai.chatbot.network.dto

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import kotlinx.serialization.Serializable

/**
 * All fields are nullable/defaulted because ChatbotAPI's hand-rolled JSON writer
 * omits keys rather than emitting nulls (e.g. error responses have no "response"
 * or "session_id" key, success responses have no "error" key).
 */
@Serializable
data class ChatSessionRequest(
    val session_id: String,
    val message: String,
)

@Serializable
data class ChatSessionResponse(
    val success: Boolean = false,
    val response: String? = null,
    val session_id: String? = null,
    val error: String? = null,
)

@Serializable
data class ChatRequest(
    val message: String,
)

@Serializable
data class ChatResponse(
    val success: Boolean = false,
    val response: String? = null,
    val error: String? = null,
)

@Serializable
data class ClearSessionRequest(
    val session_id: String,
)

@Serializable
data class ClearSessionResponse(
    val success: Boolean = false,
    val message: String? = null,
    val error: String? = null,
)

@Serializable
data class HealthResponse(
    val status: String? = null,
    val active_sessions: Int? = null,
)
