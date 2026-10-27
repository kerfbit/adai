package com.adai.chatbot.data.db

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-10-27


import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "conversations")
data class ConversationEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    /**
     * The session id chatbot_api_server issued for this conversation, or "" when it
     * has none yet. TD-232: the server issues ids (an id it didn't issue gets 404),
     * so a new conversation starts with "" and stores the id from the first reply.
     * Conversations created before this held a client-made UUID; the server answers
     * 404 for it and ChatRepository starts a new session.
     */
    val serverSessionId: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
)
