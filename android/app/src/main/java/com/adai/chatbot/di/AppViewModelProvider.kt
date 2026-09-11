package com.adai.chatbot.di

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-10


import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.adai.chatbot.ChatbotApp
import com.adai.chatbot.settings.SettingsViewModel
import com.adai.chatbot.ui.chat.ChatViewModel
import com.adai.chatbot.ui.conversationlist.ConversationListViewModel

/**
 * Builds ViewModels from [ChatbotApp.container] instead of a Hilt graph — see
 * AppContainer's doc comment for why manual DI was chosen at this app's scope.
 */
object AppViewModelProvider {

    fun factory(app: ChatbotApp) = viewModelFactory {
        initializer {
            SettingsViewModel(app.container.settingsDataStore, app.container.apiClientProvider)
        }
        initializer {
            ConversationListViewModel(app.container.conversationRepository)
        }
    }

    fun chatFactory(app: ChatbotApp, conversationId: Long) = viewModelFactory {
        initializer {
            ChatViewModel(
                conversationId = conversationId,
                chatRepository = app.container.chatRepository,
                conversationRepository = app.container.conversationRepository,
            )
        }
    }
}
