package com.adai.chatbot.ui.conversationlist

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.testutil.FakeConversationDao
import com.adai.chatbot.testutil.FakeMessageDao
import com.adai.chatbot.testutil.FakeSettingsRepository
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: ConversationListScreen had zero coverage — this repo's first real Compose UI test,
 * establishing the pattern for the rest of TD-048's "adopt Compose UI testing" chunk. Uses a
 * real ConversationListViewModel/ConversationRepository backed by the shared src/sharedTest
 * fakes (already used by the plain-JVM ViewModel/repository tests) rather than a real Room
 * database or network — a default (host-blank) FakeSettingsRepository means
 * ConversationRepository.deleteConversation() never actually calls ApiClientProvider (see its
 * own `if (settings.isConfigured)` guard), so a real, never-configured ApiClientProvider is
 * enough; nothing here needs a fake HTTP layer.
 */
class ConversationListScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun conversation(id: Long, title: String, updatedAt: Long = id) =
        ConversationEntity(
            id = id,
            serverSessionId = "session-$id",
            title = title,
            createdAt = updatedAt,
            updatedAt = updatedAt,
        )

    private fun repository(seed: List<ConversationEntity> = emptyList()) = ConversationRepository(
        conversationDao = FakeConversationDao(seed),
        messageDao = FakeMessageDao(),
        apiClientProvider = ApiClientProvider(),
        settingsDataStore = FakeSettingsRepository(),
    )

    @Test
    fun emptyState_showsPlaceholderMessage() {
        composeTestRule.setContent {
            ConversationListScreen(
                viewModel = ConversationListViewModel(repository()),
                onOpenConversation = {},
                onOpenSettings = {},
            )
        }

        composeTestRule.onNodeWithText("No conversations yet — tap + to start one.")
            .assertExists()
    }

    @Test
    fun populatedState_showsEachConversationTitleAndHidesEmptyMessage() {
        val repo = repository(
            listOf(conversation(1, "First chat"), conversation(2, "Second chat")),
        )

        composeTestRule.setContent {
            ConversationListScreen(
                viewModel = ConversationListViewModel(repo),
                onOpenConversation = {},
                onOpenSettings = {},
            )
        }

        composeTestRule.onNodeWithText("First chat").assertExists()
        composeTestRule.onNodeWithText("Second chat").assertExists()
        composeTestRule.onNodeWithText("No conversations yet — tap + to start one.")
            .assertDoesNotExist()
    }

    @Test
    fun clickingRow_invokesOnOpenConversationWithThatConversationsId() {
        val repo = repository(listOf(conversation(42, "Target chat")))
        var openedId: Long? = null

        composeTestRule.setContent {
            ConversationListScreen(
                viewModel = ConversationListViewModel(repo),
                onOpenConversation = { openedId = it },
                onOpenSettings = {},
            )
        }

        composeTestRule.onNodeWithText("Target chat").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { openedId != null }
        assert(openedId == 42L) { "expected onOpenConversation(42), got $openedId" }
    }

    @Test
    fun clickingSettingsIcon_invokesOnOpenSettings() {
        var settingsOpened = false

        composeTestRule.setContent {
            ConversationListScreen(
                viewModel = ConversationListViewModel(repository()),
                onOpenConversation = {},
                onOpenSettings = { settingsOpened = true },
            )
        }

        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        assert(settingsOpened) { "expected onOpenSettings() to have been invoked" }
    }

    @Test
    fun clickingDelete_removesTheConversationFromTheList() {
        val repo = repository(
            listOf(conversation(1, "Keep me"), conversation(2, "Delete me")),
        )

        composeTestRule.setContent {
            ConversationListScreen(
                viewModel = ConversationListViewModel(repo),
                onOpenConversation = {},
                onOpenSettings = {},
            )
        }

        // Both rows' delete buttons otherwise share one contentDescription ("Delete
        // conversation") -- id-qualified via testTag (see ConversationListScreen.kt) so this
        // targets conversation 2's row specifically, not "whichever renders first".
        composeTestRule.onNodeWithTag("delete_conversation_2").performClick()

        composeTestRule.onNodeWithText("Keep me").assertExists()
        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("Delete me").fetchSemanticsNodes().isEmpty()
        }
    }

    @Test
    fun clickingFab_createsANewConversationAndInvokesOnOpenConversation() {
        val repo = repository()
        var openedId: Long? = null

        composeTestRule.setContent {
            ConversationListScreen(
                viewModel = ConversationListViewModel(repo),
                onOpenConversation = { openedId = it },
                onOpenSettings = {},
            )
        }

        composeTestRule.onNodeWithContentDescription("New chat").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { openedId != null }
        assert(openedId != null) { "expected a new conversation id to be reported" }
    }
}
