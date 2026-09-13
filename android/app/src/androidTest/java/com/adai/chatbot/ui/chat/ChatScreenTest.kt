package com.adai.chatbot.ui.chat

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.db.MessageEntity
import com.adai.chatbot.data.db.MessageRole
import com.adai.chatbot.data.db.MessageStatus
import com.adai.chatbot.data.repository.ChatRepository
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.network.dto.ChatSessionResponse
import com.adai.chatbot.settings.ServerSettings
import com.adai.chatbot.testutil.FakeApiClientProvider
import com.adai.chatbot.testutil.FakeConversationDao
import com.adai.chatbot.testutil.FakeMessageDao
import com.adai.chatbot.testutil.FakeSettingsRepository
import com.adai.chatbot.testutil.RecordingFakeChatApiService
import kotlinx.coroutines.runBlocking
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: ChatScreen had zero coverage. Reuses ChatViewModelTest's own real-repository Fixture
 * shape (plain-JVM test, same sharedTest fakes) rather than a real Room database or network --
 * a *configured* FakeSettingsRepository (host="localhost") so ChatRepository's
 * dispatchToServer() actually reaches the fake RecordingFakeChatApiService instead of
 * short-circuiting on `!settings.isConfigured` (contrast ConversationListScreenTest's
 * deliberately-unconfigured settings, needed there for the opposite reason).
 */
class ChatScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private class Fixture(
        chatService: RecordingFakeChatApiService = RecordingFakeChatApiService(),
    ) {
        val messageDao = FakeMessageDao()
        val conversationDao = FakeConversationDao()
        val settings = FakeSettingsRepository(ServerSettings(host = "localhost", port = 8080))
        val apiClientProvider = FakeApiClientProvider(chatService)
        val chatRepository = ChatRepository(messageDao, conversationDao, apiClientProvider, settings)
        val conversationRepository =
            ConversationRepository(conversationDao, messageDao, apiClientProvider, settings)

        val conversationId: Long = runBlocking {
            conversationDao.insert(
                ConversationEntity(serverSessionId = "sess-1", title = "Chat", createdAt = 0, updatedAt = 0)
            )
        }

        fun viewModel() = ChatViewModel(conversationId, chatRepository, conversationRepository)
    }

    @Test
    fun sendingAMessage_showsUserMessageThenAssistantReply() {
        val fixture = Fixture()

        composeTestRule.setContent {
            ChatScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.onNode(hasSetTextAction()).performTextInput("hello there")
        composeTestRule.onNodeWithContentDescription("Send").performClick()

        composeTestRule.onNodeWithText("hello there").assertExists()
        // RecordingFakeChatApiService's default success response content is "ok".
        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("ok").fetchSemanticsNodes().isNotEmpty()
        }
    }

    @Test
    fun clickingBack_invokesOnBack() {
        val fixture = Fixture()
        var backInvoked = false

        composeTestRule.setContent {
            ChatScreen(viewModel = fixture.viewModel(), onBack = { backInvoked = true })
        }

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
    }

    @Test
    fun sendFailure_showsErrorBannerThenDismissHidesIt() {
        val fixture = Fixture(
            chatService = RecordingFakeChatApiService(
                nextResponse = { ChatSessionResponse(success = false, error = "boom") },
            ),
        )

        composeTestRule.setContent {
            ChatScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.onNode(hasSetTextAction()).performTextInput("hello")
        composeTestRule.onNodeWithContentDescription("Send").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("boom").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Dismiss").performClick()
        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("boom").fetchSemanticsNodes().isEmpty()
        }
    }

    @Test
    fun retryingAFailedMessage_resendsAndShowsAssistantReply() {
        val fixture = Fixture()
        val failedMessageId = runBlocking {
            fixture.messageDao.insert(
                MessageEntity(
                    conversationId = fixture.conversationId,
                    role = MessageRole.USER,
                    content = "never delivered",
                    timestamp = 0,
                    status = MessageStatus.FAILED,
                )
            )
        }

        composeTestRule.setContent {
            ChatScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.onNodeWithText("Failed — tap to retry").assertExists()
        composeTestRule.onNodeWithText("Failed — tap to retry").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            fixture.messageDao.all.any { it.id == failedMessageId && it.status == MessageStatus.SENT }
        }
        composeTestRule.onNodeWithText("Failed — tap to retry").assertDoesNotExist()
    }
}
