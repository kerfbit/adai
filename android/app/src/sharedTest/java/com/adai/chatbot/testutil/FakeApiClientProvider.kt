package com.adai.chatbot.testutil

// @adai-status: beta        (in-memory fakes backing both plain-JVM and instrumented tests, TD-048)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-13


import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.network.ChatApiService
import com.adai.chatbot.network.dto.ChatRequest
import com.adai.chatbot.network.dto.ChatResponse
import com.adai.chatbot.network.dto.ChatSessionRequest
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.network.dto.ClearSessionRequest
import com.adai.chatbot.network.dto.ClearSessionResponse
import com.adai.chatbot.network.dto.HealthResponse

/** Records every session_id sent, so tests can assert an empty one is never sent (the server-side echo bug). */
class RecordingFakeChatApiService(
    private val nextResponse: () -> ChatSessionResponse = {
        ChatSessionResponse(success = true, response = "ok", session_id = "")
    },
    private val clearSessionResponse: () -> ClearSessionResponse = {
        ClearSessionResponse(success = true, message = "Session cleared")
    },
) : ChatApiService {

    val sentSessionIds = mutableListOf<String>()
    val clearedSessionIds = mutableListOf<String>()

    override suspend fun chatSession(request: ChatSessionRequest): ChatSessionResponse {
        sentSessionIds += request.session_id
        return nextResponse()
    }

    override suspend fun chat(request: ChatRequest): ChatResponse = ChatResponse(success = true, response = "ok")

    override suspend fun clearSession(request: ClearSessionRequest): ClearSessionResponse {
        clearedSessionIds += request.session_id
        return clearSessionResponse()
    }

    override suspend fun health(): HealthResponse = HealthResponse(status = "ok", active_sessions = 0)
}

class FakeApiClientProvider(private val service: ChatApiService) : ApiClientProvider() {
    override fun serviceFor(
        host: String,
        port: Int,
        useHttps: Boolean,
        accessClientId: String,
        accessClientSecret: String,
    ): ChatApiService = service
}
