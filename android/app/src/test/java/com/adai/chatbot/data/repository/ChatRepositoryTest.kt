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
import okhttp3.ResponseBody.Companion.toResponseBody
import retrofit2.HttpException
import retrofit2.Response

class ChatRepositoryTest {

    private suspend fun FakeConversationDao.newConversation(serverSessionId: String = ""): Long = insert(
        ConversationEntity(
            serverSessionId = serverSessionId,
            title = "New chat",
            createdAt = 0,
            updatedAt = 0,
        )
    )

    private fun notFound() = HttpException(
        Response.error<Any>(404, """{"success":false,"error":"Session not found"}""".toResponseBody())
    )

    private fun repository(conversationDao: FakeConversationDao, service: RecordingFakeChatApiService) =
        ChatRepository(
            messageDao = FakeMessageDao(),
            conversationDao = conversationDao,
            apiClientProvider = FakeApiClientProvider(service),
            settingsDataStore = FakeSettingsRepository(),
        )

    // TD-232: the server issues session ids.
    @Test
    fun `a new conversation asks for a session and keeps the id the server issues`() = runTest {
        val conversationDao = FakeConversationDao()
        val conversationId = conversationDao.newConversation()
        val fakeService = RecordingFakeChatApiService(
            nextResponse = { ChatSessionResponse(success = true, response = "hi", session_id = "issued-1") },
        )
        val repository = repository(conversationDao, fakeService)

        repository.sendMessage(conversationId, "hello")
        repository.sendMessage(conversationId, "again")

        assertEquals(listOf("", "issued-1"), fakeService.sentSessionIds)
        assertEquals("issued-1", conversationDao.getById(conversationId)!!.serverSessionId)
    }

    @Test
    fun `a session the server no longer holds is replaced and the message still sent`() = runTest {
        val conversationDao = FakeConversationDao()
        val conversationId = conversationDao.newConversation(serverSessionId = "expired")
        var calls = 0
        val fakeService = RecordingFakeChatApiService(
            nextResponse = {
                if (calls++ == 0) throw notFound()
                ChatSessionResponse(success = true, response = "hi", session_id = "issued-2")
            },
        )

        val result = repository(conversationDao, fakeService).sendMessage(conversationId, "hello")

        assertTrue(result is ApiResult.Success)
        assertEquals(listOf("expired", ""), fakeService.sentSessionIds)
        assertEquals("issued-2", conversationDao.getById(conversationId)!!.serverSessionId)
    }

    @Test
    fun `a 404 for a new session is an error, not a retry loop`() = runTest {
        val conversationDao = FakeConversationDao()
        val conversationId = conversationDao.newConversation()
        val fakeService = RecordingFakeChatApiService(nextResponse = { throw notFound() })

        val result = repository(conversationDao, fakeService).sendMessage(conversationId, "hello")

        assertTrue(result is ApiResult.NetworkError)
        assertEquals(listOf(""), fakeService.sentSessionIds)
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
