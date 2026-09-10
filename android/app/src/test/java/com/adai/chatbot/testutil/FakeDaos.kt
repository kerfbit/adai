package com.adai.chatbot.testutil

import com.adai.chatbot.data.db.ConversationDao
import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.db.MessageDao
import com.adai.chatbot.data.db.MessageEntity
import com.adai.chatbot.data.db.MessageStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map
import java.util.concurrent.atomic.AtomicLong

class FakeConversationDao(seed: List<ConversationEntity> = emptyList()) : ConversationDao {
    private val nextId = AtomicLong((seed.maxOfOrNull { it.id } ?: 0) + 1)
    private val state = MutableStateFlow(seed.associateBy { it.id })

    override fun getAll() = state.map { it.values.sortedByDescending(ConversationEntity::updatedAt) }

    override suspend fun getById(id: Long): ConversationEntity? = state.value[id]

    override suspend fun insert(conversation: ConversationEntity): Long {
        val id = if (conversation.id != 0L) conversation.id else nextId.getAndIncrement()
        state.value = state.value + (id to conversation.copy(id = id))
        return id
    }

    override suspend fun update(conversation: ConversationEntity) {
        state.value = state.value + (conversation.id to conversation)
    }

    override suspend fun delete(conversation: ConversationEntity) {
        state.value = state.value - conversation.id
    }

    override suspend fun touch(id: Long, updatedAt: Long) {
        state.value[id]?.let { state.value = state.value + (id to it.copy(updatedAt = updatedAt)) }
    }

    override suspend fun updateTitle(id: Long, title: String) {
        state.value[id]?.let { state.value = state.value + (id to it.copy(title = title)) }
    }
}

class FakeMessageDao : MessageDao {
    private val nextId = AtomicLong(1)
    private val state = MutableStateFlow(emptyMap<Long, MessageEntity>())

    val all: List<MessageEntity> get() = state.value.values.sortedBy(MessageEntity::timestamp)

    override fun getMessagesForConversation(conversationId: Long) = state.map { messages ->
        messages.values.filter { it.conversationId == conversationId }.sortedBy(MessageEntity::timestamp)
    }

    override suspend fun insert(message: MessageEntity): Long {
        val id = nextId.getAndIncrement()
        state.value = state.value + (id to message.copy(id = id))
        return id
    }

    override suspend fun update(message: MessageEntity) {
        state.value = state.value + (message.id to message)
    }

    override suspend fun updateStatus(id: Long, status: MessageStatus) {
        state.value[id]?.let { state.value = state.value + (id to it.copy(status = status)) }
    }

    override suspend fun firstMessage(conversationId: Long): MessageEntity? =
        state.value.values.filter { it.conversationId == conversationId }.minByOrNull(MessageEntity::timestamp)
}
