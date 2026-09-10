package com.adai.chatbot.data.db

import androidx.room.Room
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.settings.ServerSettings
import com.adai.chatbot.settings.SettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

private class UnconfiguredSettingsRepository : SettingsRepository {
    override val settings = MutableStateFlow(ServerSettings())
    override suspend fun save(
        host: String,
        port: Int,
        useHttps: Boolean,
        accessClientId: String,
        accessClientSecret: String,
    ) {}
}

@RunWith(AndroidJUnit4::class)
class ConversationRepositoryInstrumentedTest {

    private lateinit var db: AppDatabase
    private lateinit var repository: ConversationRepository

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).allowMainThreadQueries().build()
        repository = ConversationRepository(
            conversationDao = db.conversationDao(),
            messageDao = db.messageDao(),
            apiClientProvider = com.adai.chatbot.network.ApiClientProvider(),
            settingsDataStore = UnconfiguredSettingsRepository(),
        )
    }

    @After
    fun tearDown() {
        db.close()
    }

    @Test
    fun newConversationsAlwaysGetANonBlankServerSessionId() = runBlocking {
        val id = repository.createConversation()
        val conversation = db.conversationDao().getById(id)
        assertNotNull(conversation)
        assertTrue(conversation!!.serverSessionId.isNotBlank())
    }

    @Test
    fun deletingAConversationCascadesItsMessages() = runBlocking {
        val id = repository.createConversation()
        db.messageDao().insert(
            MessageEntity(
                conversationId = id,
                role = MessageRole.USER,
                content = "hi",
                timestamp = 0,
                status = MessageStatus.SENT,
            )
        )

        val conversation = db.conversationDao().getById(id)!!
        repository.deleteConversation(conversation)

        assertNull(db.conversationDao().getById(id))
        assertEquals(emptyList<MessageEntity>(), db.messageDao().getMessagesForConversation(id).first())
    }
}
