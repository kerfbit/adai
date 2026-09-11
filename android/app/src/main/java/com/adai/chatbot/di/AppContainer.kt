package com.adai.chatbot.di

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-10


import android.content.Context
import androidx.room.Room
import com.adai.chatbot.data.db.AppDatabase
import com.adai.chatbot.data.repository.ChatRepository
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.settings.SettingsDataStore

/**
 * Small, manually-wired dependency container. The app has a handful of screens and
 * repositories, so a DI framework (Hilt) would add annotation-processor overhead
 * without buying much here.
 */
class AppContainer(context: Context) {

    val settingsDataStore = SettingsDataStore(context)

    val apiClientProvider = ApiClientProvider()

    private val database = Room.databaseBuilder(
        context.applicationContext,
        AppDatabase::class.java,
        "adai-chatbot.db",
    ).build()

    val conversationRepository = ConversationRepository(
        conversationDao = database.conversationDao(),
        messageDao = database.messageDao(),
        apiClientProvider = apiClientProvider,
        settingsDataStore = settingsDataStore,
    )

    val chatRepository = ChatRepository(
        messageDao = database.messageDao(),
        conversationDao = database.conversationDao(),
        apiClientProvider = apiClientProvider,
        settingsDataStore = settingsDataStore,
    )
}
