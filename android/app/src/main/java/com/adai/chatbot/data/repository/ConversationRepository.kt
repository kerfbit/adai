package com.adai.chatbot.data.repository

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import com.adai.chatbot.data.db.ConversationDao
import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.db.MessageDao
import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.network.dto.ClearSessionRequest
import com.adai.chatbot.settings.SettingsRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import java.util.UUID

class ConversationRepository(
    private val conversationDao: ConversationDao,
    private val messageDao: MessageDao,
    private val apiClientProvider: ApiClientProvider,
    private val settingsDataStore: SettingsRepository,
) {

    fun getAll(): Flow<List<ConversationEntity>> = conversationDao.getAll()

    suspend fun createConversation(title: String = DEFAULT_TITLE): Long {
        val now = System.currentTimeMillis()
        return conversationDao.insert(
            ConversationEntity(
                serverSessionId = UUID.randomUUID().toString(),
                title = title.ifBlank { DEFAULT_TITLE },
                createdAt = now,
                updatedAt = now,
            )
        )
    }

    /**
     * Titles a conversation from its first message, once. Safe to call on every
     * send (not just the first, tracked in-memory) since it re-checks the
     * persisted title — a fresh ChatViewModel after navigating back in would
     * otherwise have no way to know a title was already set.
     */
    suspend fun autoTitleFromFirstMessage(conversationId: Long, candidateTitle: String) {
        val conversation = conversationDao.getById(conversationId) ?: return
        if (conversation.title != DEFAULT_TITLE) return
        val trimmed = candidateTitle.trim().take(60)
        if (trimmed.isNotEmpty()) {
            conversationDao.updateTitle(conversationId, trimmed)
        }
    }

    companion object {
        const val DEFAULT_TITLE = "New chat"
    }

    /**
     * Best-effort server-side cleanup: /clear-session returns an error for a
     * session the server never saw (e.g. an empty local conversation), which is
     * harmless — local deletion is authoritative regardless of the server's response.
     */
    suspend fun deleteConversation(conversation: ConversationEntity) {
        val settings = settingsDataStore.settings.first()
        if (settings.isConfigured) {
            try {
                apiClientProvider.serviceFor(
                    settings.host,
                    settings.port,
                    settings.useHttps,
                    settings.accessClientId,
                    settings.accessClientSecret,
                ).clearSession(ClearSessionRequest(session_id = conversation.serverSessionId))
            } catch (_: Exception) {
                // Ignored — local deletion proceeds regardless.
            }
        }
        conversationDao.delete(conversation)
    }
}
