package com.adai.chatbot.data.repository

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.chatbot.data.db.ConversationDao
import com.adai.chatbot.data.db.MessageDao
import com.adai.chatbot.data.db.MessageEntity
import com.adai.chatbot.data.db.MessageRole
import com.adai.chatbot.data.db.MessageStatus
import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.network.ApiResult
import com.adai.chatbot.network.dto.ChatSessionRequest
import com.adai.chatbot.settings.ServerSettings
import com.adai.chatbot.settings.SettingsRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
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
     * Sends [text] in [conversationId]'s conversation. The conversation's
     * serverSessionId is fixed at creation and always non-empty, so the server
     * never falls into the empty-session_id echo bug (ChatbotAPI.cpp:156-205).
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
            val response = service.chatSession(
                ChatSessionRequest(session_id = serverSessionId, message = text)
            )

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
        } catch (e: Exception) {
            messageDao.updateStatus(userMessageId, MessageStatus.FAILED)
            ApiResult.NetworkError(networkErrorMessage(e))
        }
    }

    companion object {
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
