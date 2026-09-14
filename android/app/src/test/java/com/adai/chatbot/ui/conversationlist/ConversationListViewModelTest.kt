package com.adai.chatbot.ui.conversationlist

import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.settings.ServerSettings
import com.adai.chatbot.testutil.FakeApiClientProvider
import com.adai.chatbot.testutil.FakeConversationDao
import com.adai.chatbot.testutil.FakeMessageDao
import com.adai.chatbot.testutil.FakeSettingsRepository
import com.adai.chatbot.testutil.RecordingFakeChatApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * TD-048: ConversationListViewModel had zero test coverage -- only its Compose UI screen test
 * (ConversationListScreenTest) existed, discovered while auditing this same TD-048 entry's
 * "ViewModels done" claim (see TECHNICAL_DEBT.md's Correction).
 *
 * `conversations` is a `stateIn(SharingStarted.WhileSubscribed(5_000), emptyList())` over the
 * repository's Flow, so it only starts collecting the underlying dao once something actually
 * subscribes to it -- these tests start a `backgroundScope` collector and `runCurrent()`/
 * `advanceUntilIdle()` before reading `.value`, same shape as this module's poller-based
 * ViewModels collecting a StateFlow via `backgroundScope.launch { ... }`.
 *
 * createConversation()/deleteConversation() both use viewModelScope.launch directly
 * (fire-and-forget), which needs a real Main dispatcher -- hence the StandardTestDispatcher setup
 * below, same as ChatViewModelTest/ModelDetailViewModelTest.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ConversationListViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private class Fixture(
        seed: List<ConversationEntity> = emptyList(),
        chatService: RecordingFakeChatApiService = RecordingFakeChatApiService(),
        settings: FakeSettingsRepository = FakeSettingsRepository(),
    ) {
        val conversationDao = FakeConversationDao(seed)
        val messageDao = FakeMessageDao()
        val apiClientProvider = FakeApiClientProvider(chatService)
        val chatService = chatService
        val conversationRepository = ConversationRepository(conversationDao, messageDao, apiClientProvider, settings)

        fun viewModel() = ConversationListViewModel(conversationRepository)
    }

    private fun entity(id: Long, title: String, updatedAt: Long) =
        ConversationEntity(id = id, serverSessionId = "sess-$id", title = title, createdAt = updatedAt, updatedAt = updatedAt)

    @Test
    fun `conversations starts empty when the repository has nothing`() = runTest {
        val viewModel = Fixture().viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()

        assertTrue(viewModel.conversations.value.isEmpty())
        collectorJob.cancel()
    }

    @Test
    fun `conversations reflects the repository, sorted by most recently updated`() = runTest {
        val fixture = Fixture(
            seed = listOf(
                entity(id = 1, title = "Older", updatedAt = 100),
                entity(id = 2, title = "Newer", updatedAt = 200),
            ),
        )
        val viewModel = fixture.viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()

        assertEquals(listOf("Newer", "Older"), viewModel.conversations.value.map { it.title })
        collectorJob.cancel()
    }

    @Test
    fun `createConversation inserts a new conversation and invokes the callback with its id`() = runTest {
        val fixture = Fixture()
        val viewModel = fixture.viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()
        var createdId: Long? = null

        viewModel.createConversation(onCreated = { id -> createdId = id })
        advanceUntilIdle()

        assertEquals(createdId, viewModel.conversations.value.single().id)
        assertEquals("New chat", viewModel.conversations.value.single().title)
        collectorJob.cancel()
    }

    @Test
    fun `deleteConversation removes it from the list`() = runTest {
        val fixture = Fixture(seed = listOf(entity(id = 1, title = "To delete", updatedAt = 100)))
        val viewModel = fixture.viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()
        val conversation = viewModel.conversations.value.single()

        viewModel.deleteConversation(conversation)
        advanceUntilIdle()

        assertTrue(viewModel.conversations.value.isEmpty())
        collectorJob.cancel()
    }

    @Test
    fun `deleteConversation calls clear-session on the server when settings are configured`() = runTest {
        val fixture = Fixture(seed = listOf(entity(id = 1, title = "To delete", updatedAt = 100)))
        val viewModel = fixture.viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()
        val conversation = viewModel.conversations.value.single()

        viewModel.deleteConversation(conversation)
        advanceUntilIdle()

        assertEquals(listOf("sess-1"), fixture.chatService.clearedSessionIds)
    }

    @Test
    fun `deleteConversation still removes it locally when clear-session fails`() = runTest {
        // Matches ConversationRepository.deleteConversation()'s own doc comment: local deletion
        // is authoritative regardless of the server's response.
        val fixture = Fixture(
            seed = listOf(entity(id = 1, title = "To delete", updatedAt = 100)),
            chatService = RecordingFakeChatApiService(
                clearSessionResponse = { throw RuntimeException("server unreachable") },
            ),
        )
        val viewModel = fixture.viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()
        val conversation = viewModel.conversations.value.single()

        viewModel.deleteConversation(conversation)
        advanceUntilIdle()

        assertTrue(viewModel.conversations.value.isEmpty())
        collectorJob.cancel()
    }

    @Test
    fun `deleteConversation does not call clear-session when settings are not configured`() = runTest {
        val fixture = Fixture(
            seed = listOf(entity(id = 1, title = "To delete", updatedAt = 100)),
            settings = FakeSettingsRepository(ServerSettings()),
        )
        val viewModel = fixture.viewModel()
        val collectorJob = backgroundScope.launch { viewModel.conversations.collect {} }
        testScheduler.runCurrent()
        val conversation = viewModel.conversations.value.single()

        viewModel.deleteConversation(conversation)
        advanceUntilIdle()

        assertTrue(fixture.chatService.clearedSessionIds.isEmpty())
        assertTrue(viewModel.conversations.value.isEmpty())
        collectorJob.cancel()
    }
}
