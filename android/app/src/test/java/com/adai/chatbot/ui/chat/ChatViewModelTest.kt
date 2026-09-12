package com.adai.chatbot.ui.chat

import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.db.MessageRole
import com.adai.chatbot.data.repository.ChatRepository
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.testutil.FakeApiClientProvider
import com.adai.chatbot.testutil.FakeConversationDao
import com.adai.chatbot.testutil.FakeMessageDao
import com.adai.chatbot.testutil.FakeSettingsRepository
import com.adai.chatbot.testutil.RecordingFakeChatApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * TD-048: ChatViewModel had zero test coverage. A StandardTestDispatcher (not auto-advancing)
 * as Main lets these tests observe `_isSending` synchronously — send()/retry() flip it to true
 * *before* launching the coroutine that actually talks to the server, which is exactly what
 * the double-send guard depends on and exactly what these tests exercise directly.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ChatViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private class Fixture(
        chatService: RecordingFakeChatApiService = RecordingFakeChatApiService(),
    ) {
        val messageDao = FakeMessageDao()
        val conversationDao = FakeConversationDao()
        val settings = FakeSettingsRepository()
        val apiClientProvider = FakeApiClientProvider(chatService)
        val chatService = chatService
        val chatRepository = ChatRepository(messageDao, conversationDao, apiClientProvider, settings)
        val conversationRepository =
            ConversationRepository(conversationDao, messageDao, apiClientProvider, settings)

        suspend fun newConversation(): Long = conversationDao.insert(
            ConversationEntity(serverSessionId = "sess-1", title = "New chat", createdAt = 0, updatedAt = 0)
        )
    }

    @Test
    fun `send with blank text does not touch the repository or the server`() = runTest {
        val fixture = Fixture()
        val conversationId = fixture.newConversation()
        val viewModel = ChatViewModel(conversationId, fixture.chatRepository, fixture.conversationRepository)

        viewModel.send("   ")
        advanceUntilIdle()

        assertTrue("blank text must not create a message", fixture.messageDao.all.isEmpty())
        assertTrue("blank text must not reach the server", fixture.chatService.sentSessionIds.isEmpty())
        assertFalse(viewModel.isSending.value)
    }

    @Test
    fun `a second send while the first is still in flight is dropped`() = runTest {
        // isSending is flipped to true synchronously inside send(), before the launched
        // coroutine that calls the repository ever runs — so a second send() called before
        // advanceUntilIdle() genuinely exercises the "already sending" guard, not a race.
        val fixture = Fixture()
        val conversationId = fixture.newConversation()
        val viewModel = ChatViewModel(conversationId, fixture.chatRepository, fixture.conversationRepository)

        viewModel.send("hello")
        assertTrue("isSending must flip synchronously", viewModel.isSending.value)
        viewModel.send("this one should be dropped")
        advanceUntilIdle()

        assertEquals(1, fixture.chatService.sentSessionIds.size)
    }

    @Test
    fun `send success resets isSending and stores the assistant reply`() = runTest {
        val fixture = Fixture()
        val conversationId = fixture.newConversation()
        val viewModel = ChatViewModel(conversationId, fixture.chatRepository, fixture.conversationRepository)

        viewModel.send("hello")
        advanceUntilIdle()

        assertFalse(viewModel.isSending.value)
        assertEquals(1, fixture.chatService.sentSessionIds.size)
        // User message + assistant reply.
        assertEquals(2, fixture.messageDao.all.size)
    }

    @Test
    fun `send failure emits an error event and still resets isSending`() = runTest {
        val fixture = Fixture(
            chatService = RecordingFakeChatApiService(
                nextResponse = { ChatSessionResponse(success = false, error = "boom") },
            ),
        )
        val conversationId = fixture.newConversation()
        val viewModel = ChatViewModel(conversationId, fixture.chatRepository, fixture.conversationRepository)

        viewModel.send("hello")
        advanceUntilIdle()

        // _errorEvents is a Channel(BUFFERED): the value sent during advanceUntilIdle() above
        // is retained regardless of when a collector attaches, so first() here reads it back
        // without needing to race a collector against the send.
        assertEquals("boom", viewModel.errorEvents.first())
        assertFalse(viewModel.isSending.value)
    }

    @Test
    fun `retry while already sending is dropped`() = runTest {
        val fixture = Fixture()
        val conversationId = fixture.newConversation()
        val viewModel = ChatViewModel(conversationId, fixture.chatRepository, fixture.conversationRepository)

        // Complete one real send first so there's an actual message to retry — the message
        // isn't inserted until sendMessage() runs inside send()'s launched coroutine, so it
        // doesn't exist yet at the point send() returns, only after advancing.
        viewModel.send("hello")
        advanceUntilIdle()
        val message = fixture.messageDao.all.first { it.role == MessageRole.USER }
        assertEquals(1, fixture.chatService.sentSessionIds.size)

        // A second send() flips isSending back to true synchronously; retry() called before
        // advancing must see that and be dropped without touching the server again.
        viewModel.send("another message")
        assertTrue(viewModel.isSending.value)
        viewModel.retry(message)
        advanceUntilIdle()

        // Only the second send() reached the server a second time — retry() was dropped.
        assertEquals(2, fixture.chatService.sentSessionIds.size)
    }
}
