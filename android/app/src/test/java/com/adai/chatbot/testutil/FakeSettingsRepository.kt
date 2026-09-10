package com.adai.chatbot.testutil

import com.adai.chatbot.settings.ServerSettings
import com.adai.chatbot.settings.SettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow

class FakeSettingsRepository(initial: ServerSettings = ServerSettings(host = "localhost", port = 8080)) :
    SettingsRepository {

    private val state = MutableStateFlow(initial)
    override val settings = state

    override suspend fun save(
        host: String,
        port: Int,
        useHttps: Boolean,
        accessClientId: String,
        accessClientSecret: String,
    ) {
        state.value = ServerSettings(
            host = host,
            port = port,
            useHttps = useHttps,
            accessClientId = accessClientId,
            accessClientSecret = accessClientSecret,
        )
    }
}
