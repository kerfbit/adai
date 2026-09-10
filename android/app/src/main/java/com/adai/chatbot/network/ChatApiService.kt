package com.adai.chatbot.network

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.chatbot.network.dto.ChatRequest
import com.adai.chatbot.network.dto.ChatResponse
import com.adai.chatbot.network.dto.ChatSessionRequest
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.network.dto.ClearSessionRequest
import com.adai.chatbot.network.dto.ClearSessionResponse
import com.adai.chatbot.network.dto.HealthResponse
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface ChatApiService {

    @POST("chat/session")
    suspend fun chatSession(@Body request: ChatSessionRequest): ChatSessionResponse

    @POST("chat")
    suspend fun chat(@Body request: ChatRequest): ChatResponse

    @POST("clear-session")
    suspend fun clearSession(@Body request: ClearSessionRequest): ClearSessionResponse

    @GET("health")
    suspend fun health(): HealthResponse
}
