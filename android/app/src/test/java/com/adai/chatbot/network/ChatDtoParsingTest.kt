package com.adai.chatbot.network

import com.adai.chatbot.network.dto.ChatResponse
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.network.dto.ClearSessionResponse
import com.adai.chatbot.network.dto.HealthResponse
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * ChatbotAPI's hand-rolled JSON writer omits keys rather than emitting nulls
 * (ChatbotAPI.cpp create_json_response/create_error_response) — these tests
 * pin the app's Json config to tolerate exactly that shape.
 */
class ChatDtoParsingTest {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        explicitNulls = false
    }

    @Test
    fun `success chat-session response with no error key parses cleanly`() {
        val decoded = json.decodeFromString(
            ChatSessionResponse.serializer(),
            """{"success":true,"response":"hi","session_id":"abc-123"}""",
        )
        assertEquals(true, decoded.success)
        assertEquals("hi", decoded.response)
        assertEquals("abc-123", decoded.session_id)
        assertNull(decoded.error)
    }

    @Test
    fun `error chat-session response has no response or session_id key`() {
        val decoded = json.decodeFromString(
            ChatSessionResponse.serializer(),
            """{"success":false,"error":"Missing 'message' field in request"}""",
        )
        assertEquals(false, decoded.success)
        assertNull(decoded.response)
        assertNull(decoded.session_id)
        assertEquals("Missing 'message' field in request", decoded.error)
    }

    @Test
    fun `the confirmed session_id echo bug still parses as an empty string, not null`() {
        // A request with session_id:"" is echoed back verbatim by the server
        // (ChatbotAPI.cpp:205) rather than the real internal key it created.
        val decoded = json.decodeFromString(
            ChatSessionResponse.serializer(),
            """{"success":true,"response":"hi","session_id":""}""",
        )
        assertEquals("", decoded.session_id)
    }

    @Test
    fun `stateless chat response parses without a session_id field at all`() {
        val decoded = json.decodeFromString(ChatResponse.serializer(), """{"success":true,"response":"hi"}""")
        assertEquals(true, decoded.success)
        assertEquals("hi", decoded.response)
    }

    @Test
    fun `clear-session not-found error parses`() {
        val decoded = json.decodeFromString(
            ClearSessionResponse.serializer(),
            """{"success":false,"error":"Session not found"}""",
        )
        assertEquals(false, decoded.success)
        assertEquals("Session not found", decoded.error)
    }

    @Test
    fun `health response parses`() {
        val decoded = json.decodeFromString(HealthResponse.serializer(), """{"status":"ok","active_sessions":3}""")
        assertEquals("ok", decoded.status)
        assertEquals(3, decoded.active_sessions)
    }
}
