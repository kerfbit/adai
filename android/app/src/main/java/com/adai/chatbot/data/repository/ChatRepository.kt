package com.adai.chatbot.data.repository

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.6.0
// @adai-reviewed: 2026-10-27


import com.adai.chatbot.data.db.ConversationDao
import com.adai.chatbot.data.db.MessageDao
import com.adai.chatbot.data.db.MessageEntity
import com.adai.chatbot.data.db.MessageRole
import com.adai.chatbot.data.db.MessageStatus
import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.network.ApiResult
import com.adai.chatbot.network.dto.ChatSessionRequest
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.settings.ServerSettings
import com.adai.chatbot.settings.SettingsRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.Json
import retrofit2.HttpException
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

class ChatRepository(
    private val messageDao: MessageDao,
    private val conversationDao: ConversationDao,
    private val apiClientProvider: ApiClientProvider,
    private val settingsDataStore: SettingsRepository,
) {

    fun messagesFor(conversationId: Long): Flow<List<MessageEntity>> =
        messageDao.getMessagesForConversation(conversationId)

    /**
     * Sends [text] in [conversationId]'s conversation. TD-232: the server issues
     * session ids. A conversation with none sends "" and stores the id the reply
     * carries; when the server no longer holds the stored id (expiry, eviction, a
     * restart) it answers 404, and the message is sent again in a new session.
     */
    suspend fun sendMessage(conversationId: Long, text: String): ApiResult<String> {
        val conversation = conversationDao.getById(conversationId)
            ?: return ApiResult.NetworkError("Conversation not found")

        val now = System.currentTimeMillis()
        val userMessageId = messageDao.insert(
            MessageEntity(
                conversationId = conversationId,
                role = MessageRole.USER,
                content = text,
                timestamp = now,
                status = MessageStatus.SENDING,
            )
        )

        return dispatchToServer(conversationId, conversation.serverSessionId, text, userMessageId)
    }

    suspend fun retryMessage(conversationId: Long, messageId: Long, text: String): ApiResult<String> {
        messageDao.updateStatus(messageId, MessageStatus.SENDING)
        val conversation = conversationDao.getById(conversationId)
            ?: return ApiResult.NetworkError("Conversation not found")

        return dispatchToServer(conversationId, conversation.serverSessionId, text, messageId)
    }

    private suspend fun dispatchToServer(
        conversationId: Long,
        serverSessionId: String,
        text: String,
        userMessageId: Long,
    ): ApiResult<String> {
        val settings: ServerSettings = settingsDataStore.settings.first()
        if (!settings.isConfigured) {
            messageDao.updateStatus(userMessageId, MessageStatus.FAILED)
            return ApiResult.NetworkError("Server not configured — set a host in Settings")
        }

        return try {
            val service = apiClientProvider.serviceFor(
                settings.host,
                settings.port,
                settings.useHttps,
                settings.accessClientId,
                settings.accessClientSecret,
            )
            var sessionId = serverSessionId
            val response = try {
                service.chatSession(ChatSessionRequest(session_id = sessionId, message = text))
            } catch (e: HttpException) {
                if (e.code() != 404 || sessionId.isEmpty()) throw e
                conversationDao.updateServerSessionId(conversationId, "")
                sessionId = ""
                service.chatSession(ChatSessionRequest(session_id = sessionId, message = text))
            }

            val issued = response.session_id
            if (response.success && !issued.isNullOrEmpty() && issued != sessionId) {
                conversationDao.updateServerSessionId(conversationId, issued)
            }

            if (response.success && response.response != null) {
                messageDao.updateStatus(userMessageId, MessageStatus.SENT)
                messageDao.insert(
                    MessageEntity(
                        conversationId = conversationId,
                        role = MessageRole.ASSISTANT,
                        content = response.response,
                        timestamp = System.currentTimeMillis(),
                        status = MessageStatus.SENT,
                    )
                )
                conversationDao.touch(conversationId, System.currentTimeMillis())
                ApiResult.Success(response.response)
            } else {
                messageDao.updateStatus(userMessageId, MessageStatus.FAILED)
                ApiResult.ApiError(response.error ?: "Unknown server error")
            }
        } catch (e: HttpException) {
            // TD-236: the server answers errors with their HTTP status (400 bad input, 404 unknown
            // session, 413 batch too large, 500 generation failure) and the usual JSON body.
            messageDao.updateStatus(userMessageId, MessageStatus.FAILED)
            ApiResult.ApiError(serverErrorMessage(e))
        } catch (e: Exception) {
            messageDao.updateStatus(userMessageId, MessageStatus.FAILED)
            ApiResult.NetworkError(networkErrorMessage(e))
        }
    }

    companion object {
        private val errorJson = Json { ignoreUnknownKeys = true; isLenient = true }

        /** The "error" field of an error reply's body, or the status when the body has none. */
        fun serverErrorMessage(e: HttpException): String {
            val body = try {
                e.response()?.errorBody()?.string()
            } catch (_: IOException) {
                null
            }
            val error = body?.let {
                runCatching { errorJson.decodeFromString(ChatSessionResponse.serializer(), it).error }.getOrNull()
            }
            return error?.takeIf { it.isNotBlank() } ?: "Server error (HTTP ${e.code()})"
        }

        fun networkErrorMessage(e: Exception): String = when (e) {
            is UnknownHostException, is ConnectException ->
                "Can't reach the server — check the host/port in Settings"
            is SocketTimeoutException ->
                "The server took too long to respond"
            is IOException -> "Network error: ${e.message}"
            else -> e.message ?: "Unknown error"
        }
    }
}
