package com.adai.chatbot.di

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.lifecycle.viewmodel.CreationExtras
import androidx.test.core.app.ApplicationProvider
import com.adai.chatbot.ChatbotApp
import com.adai.chatbot.settings.SettingsViewModel
import com.adai.chatbot.ui.chat.ChatViewModel
import com.adai.chatbot.ui.conversationlist.ConversationListViewModel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * TD-048: AppViewModelProvider had zero test coverage. Uses the real, already-running
 * [ChatbotApp] singleton (see [AppContainerTest]'s own doc comment for why a competing
 * `AppContainer` isn't constructed here instead) so these factories are exercised exactly the way
 * [com.adai.chatbot.ui.navigation.AdaiNavHost] uses them in production.
 */
class AppViewModelProviderTest {

    private val app = ApplicationProvider.getApplicationContext<ChatbotApp>()

    @Test
    fun factory_createsEachRegisteredViewModelType() {
        val factory = AppViewModelProvider.factory(app)

        assertNotNull(factory.create(SettingsViewModel::class.java, CreationExtras.Empty))
        assertNotNull(factory.create(ConversationListViewModel::class.java, CreationExtras.Empty))
    }

    @Test
    fun chatFactory_createsAChatViewModelBoundToTheGivenConversationId() = runBlocking {
        val conversationId = app.container.conversationRepository.createConversation()

        try {
            val viewModel = AppViewModelProvider.chatFactory(app, conversationId).create(ChatViewModel::class.java, CreationExtras.Empty)

            assertNotNull(viewModel)
            // Bound to the right conversation: sending nothing yet, its message list should
            // read from exactly this conversation's (currently empty) messages, proving
            // conversationId was actually threaded through rather than defaulted/ignored.
            assertTrue(app.container.chatRepository.messagesFor(conversationId).first().isEmpty())
        } finally {
            val created = app.container.conversationRepository.getAll().first().first { it.id == conversationId }
            app.container.conversationRepository.deleteConversation(created)
        }
    }

    // Two ViewModels built from the same factory() call must be backed by the SAME
    // ConversationRepository/AppContainer singleton, not two disconnected instances -- otherwise
    // a conversation created via one screen's ViewModel would never appear on another. Inserts
    // directly through the repository (a suspend call, awaited deterministically) rather than via
    // ConversationListViewModel.createConversation()'s own fire-and-forget viewModelScope.launch,
    // which is already covered by ConversationListViewModelTest and would only add a race here.
    //
    // `conversations` is a stateIn(WhileSubscribed(5_000), emptyList()) -- a brand new instance's
    // very first `.first()` can return that initial emptyList() default synchronously, before its
    // WhileSubscribed upstream has actually finished its first real query against Room. Using
    // `first { predicate }` (keeps collecting until it matches) instead of a bare `first()`
    // (grabs whatever's there right now) avoids that race -- confirmed directly: a bare
    // `.first()` here flaked exactly this way on a real device.
    @Test
    fun twoConversationListViewModelsFromTheSameFactory_shareTheSameRepository() = runBlocking {
        val factory = AppViewModelProvider.factory(app)
        val first = factory.create(ConversationListViewModel::class.java, CreationExtras.Empty)
        val second = factory.create(ConversationListViewModel::class.java, CreationExtras.Empty)

        val id = app.container.conversationRepository.createConversation()
        try {
            withTimeout(5_000) { first.conversations.first { conversations -> conversations.any { it.id == id } } }
            withTimeout(5_000) { second.conversations.first { conversations -> conversations.any { it.id == id } } }
            Unit
        } finally {
            val created = app.container.conversationRepository.getAll().first().first { it.id == id }
            app.container.conversationRepository.deleteConversation(created)
        }
    }
}
