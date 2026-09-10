package com.adai.chatbot.data.repository

import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.db.MessageRole
import com.adai.chatbot.data.db.MessageStatus
import com.adai.chatbot.network.ApiResult
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.testutil.FakeApiClientProvider
import com.adai.chatbot.testutil.FakeConversationDao
import com.adai.chatbot.testutil.FakeMessageDao
import com.adai.chatbot.testutil.FakeSettingsRepository
import com.adai.chatbot.testutil.RecordingFakeChatApiService
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.SocketTimeoutException
import java.util.UUID

class ChatRepositoryTest {

    private suspend fun FakeConversationDao.newConversation(): Long = insert(
        ConversationEntity(
            serverSessionId = UUID.randomUUID().toString(),
            title = "New chat",
            createdAt = 0,
            updatedAt = 0,
        )
    )

    @Test
    fun `sendMessage never sends an empty session_id, even for a brand new conversation`() = runTest {
        val conversationDao = FakeConversationDao()
        val messageDao = FakeMessageDao()
        val conversationId = conversationDao.newConversation()
        val fakeService = RecordingFakeChatApiService()
        val repository = ChatRepository(
            messageDao = messageDao,
            conversationDao = conversationDao,
            apiClientProvider = FakeApiClientProvider(fakeService),
            settingsDataStore = FakeSettingsRepository(),
        )

        repository.sendMessage(conversationId, "hello")

        assertTrue(fakeService.sentSessionIds.isNotEmpty())
        assertTrue(fakeService.sentSessionIds.all { it.isNotEmpty() })
    }

    @Test
    fun `sendMessage marks the user message SENT and inserts the assistant reply on success`() = runTest {
        val conversationDao = FakeConversationDao()
        val messageDao = FakeMessageDao()
        val conversationId = conversationDao.newConversation()
        val fakeService = RecordingFakeChatApiService(
            nextResponse = { ChatSessionResponse(success = true, response = "hi there", session_id = "") },
        )
        val repository = ChatRepository(
            messageDao = messageDao,
            conversationDao = conversationDao,
            apiClientProvider = FakeApiClientProvider(fakeService),
            settingsDataStore = FakeSettingsRepository(),
        )

        val result = repository.sendMessage(conversationId, "hello")

        assertTrue(result is ApiResult.Success)
        val messages = messageDao.all
        assertEquals(2, messages.size)
        assertEquals(MessageRole.USER, messages[0].role)
        assertEquals(MessageStatus.SENT, messages[0].status)
        assertEquals(MessageRole.ASSISTANT, messages[1].role)
        assertEquals("hi there", messages[1].content)
    }

    @Test
    fun `sendMessage marks the user message FAILED when the server reports success=false`() = runTest {
        val conversationDao = FakeConversationDao()
        val messageDao = FakeMessageDao()
        val conversationId = conversationDao.newConversation()
        val fakeService = RecordingFakeChatApiService(
            nextResponse = { ChatSessionResponse(success = false, error = "Missing 'message' field in request") },
        )
        val repository = ChatRepository(
            messageDao = messageDao,
            conversationDao = conversationDao,
            apiClientProvider = FakeApiClientProvider(fakeService),
            settingsDataStore = FakeSettingsRepository(),
        )

        val result = repository.sendMessage(conversationId, "hello")

        assertTrue(result is ApiResult.ApiError)
        assertEquals(MessageStatus.FAILED, messageDao.all.single().status)
    }

    @Test
    fun `sendMessage surfaces a network error and marks the message FAILED on timeout`() = runTest {
        val conversationDao = FakeConversationDao()
        val messageDao = FakeMessageDao()
        val conversationId = conversationDao.newConversation()
        val fakeService = RecordingFakeChatApiService(
            nextResponse = { throw SocketTimeoutException("timeout") },
        )
        val repository = ChatRepository(
            messageDao = messageDao,
            conversationDao = conversationDao,
            apiClientProvider = FakeApiClientProvider(fakeService),
            settingsDataStore = FakeSettingsRepository(),
        )

        val result = repository.sendMessage(conversationId, "hello")

        assertTrue(result is ApiResult.NetworkError)
        assertEquals(MessageStatus.FAILED, messageDao.all.single().status)
    }
}
