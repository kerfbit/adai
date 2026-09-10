package com.adai.chatbot.data.db

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "conversations")
data class ConversationEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    /**
     * Client-generated UUID, set once at creation and never blank. ChatbotAPI's
     * handle_chat_session echoes back whatever session_id string it was sent
     * without correcting for the internal id it actually stores the session
     * under when given an empty string (see ChatbotAPI.cpp:156-205) — sending a
     * non-empty, client-owned id sidesteps that bug entirely.
     */
    val serverSessionId: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
)
