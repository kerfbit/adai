package com.adai.chatbot.settings

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-10


import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "settings")

data class ServerSettings(
    val host: String = "",
    val port: Int = DEFAULT_PORT,
    val useHttps: Boolean = false,
    val accessClientId: String = "",
    val accessClientSecret: String = "",
) {
    val isConfigured: Boolean get() = host.isNotBlank()

    companion object {
        const val DEFAULT_PORT = 8080
    }
}

/** Narrow surface repositories depend on, so tests can fake it without a real Context/DataStore. */
interface SettingsRepository {
    val settings: Flow<ServerSettings>
    suspend fun save(
        host: String,
        port: Int,
        useHttps: Boolean = false,
        accessClientId: String = "",
        accessClientSecret: String = "",
    )
}

class SettingsDataStore(private val context: Context) : SettingsRepository {

    private object Keys {
        val HOST = stringPreferencesKey("server_host")
        val PORT = intPreferencesKey("server_port")
        val USE_HTTPS = booleanPreferencesKey("server_use_https")
        val ACCESS_CLIENT_ID = stringPreferencesKey("server_access_client_id")
        val ACCESS_CLIENT_SECRET = stringPreferencesKey("server_access_client_secret")
    }

    override val settings: Flow<ServerSettings> = context.dataStore.data.map { prefs ->
        ServerSettings(
            host = prefs[Keys.HOST] ?: "",
            port = prefs[Keys.PORT] ?: ServerSettings.DEFAULT_PORT,
            useHttps = prefs[Keys.USE_HTTPS] ?: false,
            accessClientId = prefs[Keys.ACCESS_CLIENT_ID] ?: "",
            accessClientSecret = prefs[Keys.ACCESS_CLIENT_SECRET] ?: "",
        )
    }

    override suspend fun save(
        host: String,
        port: Int,
        useHttps: Boolean,
        accessClientId: String,
        accessClientSecret: String,
    ) {
        context.dataStore.edit { prefs ->
            prefs[Keys.HOST] = host
            prefs[Keys.PORT] = port
            prefs[Keys.USE_HTTPS] = useHttps
            prefs[Keys.ACCESS_CLIENT_ID] = accessClientId
            prefs[Keys.ACCESS_CLIENT_SECRET] = accessClientSecret
        }
    }
}
