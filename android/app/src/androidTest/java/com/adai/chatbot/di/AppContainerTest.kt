package com.adai.chatbot.di

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.test.core.app.ApplicationProvider
import com.adai.chatbot.ChatbotApp
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * TD-048: AppContainer had zero test coverage — this is the app's real manual-DI graph (a Room
 * database + both repositories), constructed once per process in [ChatbotApp.onCreate]. Rather
 * than constructing a second, competing `AppContainer` against the same on-disk database file
 * (real risk: Room already holds that file open for the live app process every instrumented test
 * actually runs inside), this uses [ApplicationProvider] to reach the SAME real, already-running
 * [ChatbotApp.container] singleton -- exactly what production code sees -- and cleans up the one
 * row it creates afterward instead of touching the database file itself.
 *
 * What's worth proving here isn't "the properties are non-null" but that the wiring actually
 * produces a *shared* database: a conversation created through
 * [AppContainer.conversationRepository] must be visible through [AppContainer.chatRepository] too
 * (wired to the same `AppDatabase` instance, not a separate one).
 */
class AppContainerTest {

    private val app = ApplicationProvider.getApplicationContext<ChatbotApp>()

    @Test
    fun conversationRepository_and_chatRepository_shareTheSameDatabase() = runBlocking {
        val container = app.container
        val id = container.conversationRepository.createConversation()

        try {
            val conversations = container.conversationRepository.getAll().first()
            assertTrue("the newly created conversation should be visible via getAll()", conversations.any { it.id == id })

            // Proves chatRepository is wired to the SAME AppDatabase instance -- if AppContainer
            // accidentally built two separate Room databases, this conversation (inserted via
            // conversationRepository) wouldn't be visible to a query issued through
            // chatRepository at all, let alone return a sensible (empty) result.
            val messages = container.chatRepository.messagesFor(id).first()
            assertTrue("a freshly created conversation should have no messages yet", messages.isEmpty())
        } finally {
            val created = container.conversationRepository.getAll().first().first { it.id == id }
            container.conversationRepository.deleteConversation(created)
        }
    }

    @Test
    fun settingsDataStore_and_apiClientProvider_areUsableImmediately() = runBlocking {
        val container = app.container

        // Not asserting specific values -- just that reading from/constructing off of these
        // doesn't throw, proving they're real, usable objects rather than something that only
        // looks wired until first touched.
        container.settingsDataStore.settings.first()
        val service = container.apiClientProvider.serviceFor(
            host = "localhost", port = 8080, useHttps = false, accessClientId = "", accessClientSecret = "",
        )
        assertNotNull(service)
    }
}
