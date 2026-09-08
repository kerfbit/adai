package com.adai.ops.settings

// @adai-status: beta
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "ops_settings")

/**
 * metrics_api_server, mns_server, and registry_server default to one host with
 * distinct ports (8081/8083/8082) in this repo's config.conf, but nothing in the
 * server code assumes co-location, so per-service host overrides stay available
 * even when [useSharedHost] is on.
 */
data class OpsSettings(
    val useSharedHost: Boolean = true,
    val sharedHost: String = "",
    val metricsHost: String = "",
    val metricsPort: Int = DEFAULT_METRICS_PORT,
    val mnsHost: String = "",
    val mnsPort: Int = DEFAULT_MNS_PORT,
    val registryHost: String = "",
    val registryPort: Int = DEFAULT_REGISTRY_PORT,
    val trainerHost: String = "",
    val trainerPort: Int = DEFAULT_TRAINER_PORT,
    val registryGroups: List<String> = emptyList(),
    val basePollIntervalMs: Long = DEFAULT_POLL_INTERVAL_MS,
    // Relay mode (Cloudflare Tunnel via kerfbit.dev) — one toggle shared across all
    // four "read-only-ish" services (metrics/mns/registry/trainer's host+port
    // resolution), consistent with useSharedHost. Only changes scheme and port
    // handling; the user still types the relay subdomain into the existing host
    // field(s) above.
    val useHttpsRelay: Boolean = false,
    // Access token for chat + the three read-only dashboards (metrics/mns/registry)
    // — see docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md's shared
    // "adai-tablet-relay" application. Deliberately NOT used for the trainer admin
    // API — see trainerAccessClientId/Secret below.
    val accessClientId: String = "",
    val accessClientSecret: String = "",
    // Separate Access token for trainer.kerfbit.dev ("adai-trainer-relay" in the
    // Cloudflare dashboard) — unlike the other four hostnames, the trainer admin API
    // can pause/resume/checkpoint a live training run, so it's gated by its own
    // Cloudflare Access application/service token that can be revoked independently
    // of chat/metrics/mns/registry access. Still gated by the same useHttpsRelay
    // toggle (only the credentials differ, not whether relay mode is on at all).
    val trainerAccessClientId: String = "",
    val trainerAccessClientSecret: String = "",
    // Watch face relay (see com.adai.ops.data.wearsync.WatchSyncRepository) — null override
    // means "Auto" (ActiveSessionSelector picks the live/most-recent session).
    val watchSyncEnabled: Boolean = true,
    val watchSyncSessionKeyOverride: String? = null,
) {
    val effectiveMetricsHost: String get() = if (useSharedHost) sharedHost else metricsHost
    val effectiveMnsHost: String get() = if (useSharedHost) sharedHost else mnsHost
    val effectiveRegistryHost: String get() = if (useSharedHost) sharedHost else registryHost
    val effectiveTrainerHost: String get() = if (useSharedHost) sharedHost else trainerHost

    companion object {
        const val DEFAULT_METRICS_PORT = 8081
        const val DEFAULT_MNS_PORT = 8083
        const val DEFAULT_REGISTRY_PORT = 8082
        // Matches TRAINER_ADMIN_PORT's default in config.trainer.conf — next free slot
        // after chatbot=8080/metrics=8081/registry=8082/mns=8083. Opt-in server-side
        // (TRAINER_ADMIN_ENABLED=false by default), so an unreachable host here is the
        // expected state until that's turned on.
        const val DEFAULT_TRAINER_PORT = 8084
        const val DEFAULT_POLL_INTERVAL_MS = 2000L
    }
}

/** Narrow surface repositories/ViewModels depend on, so tests can fake it without a real Context. */
interface OpsSettingsRepository {
    val settings: Flow<OpsSettings>
    suspend fun save(settings: OpsSettings)
}

class OpsSettingsDataStore(private val context: Context) : OpsSettingsRepository {

    private object Keys {
        val USE_SHARED_HOST = booleanPreferencesKey("use_shared_host")
        val SHARED_HOST = stringPreferencesKey("shared_host")
        val METRICS_HOST = stringPreferencesKey("metrics_host")
        val METRICS_PORT = intPreferencesKey("metrics_port")
        val MNS_HOST = stringPreferencesKey("mns_host")
        val MNS_PORT = intPreferencesKey("mns_port")
        val REGISTRY_HOST = stringPreferencesKey("registry_host")
        val REGISTRY_PORT = intPreferencesKey("registry_port")
        val TRAINER_HOST = stringPreferencesKey("trainer_host")
        val TRAINER_PORT = intPreferencesKey("trainer_port")
        val REGISTRY_GROUPS = stringPreferencesKey("registry_groups")
        val BASE_POLL_INTERVAL_MS = longPreferencesKey("base_poll_interval_ms")
        val USE_HTTPS_RELAY = booleanPreferencesKey("use_https_relay")
        val ACCESS_CLIENT_ID = stringPreferencesKey("access_client_id")
        val ACCESS_CLIENT_SECRET = stringPreferencesKey("access_client_secret")
        val TRAINER_ACCESS_CLIENT_ID = stringPreferencesKey("trainer_access_client_id")
        val TRAINER_ACCESS_CLIENT_SECRET = stringPreferencesKey("trainer_access_client_secret")
        val WATCH_SYNC_ENABLED = booleanPreferencesKey("watch_sync_enabled")
        val WATCH_SYNC_SESSION_KEY_OVERRIDE = stringPreferencesKey("watch_sync_session_key_override")
    }

    override val settings: Flow<OpsSettings> = context.dataStore.data.map { prefs ->
        OpsSettings(
            useSharedHost = prefs[Keys.USE_SHARED_HOST] ?: true,
            sharedHost = prefs[Keys.SHARED_HOST] ?: "",
            metricsHost = prefs[Keys.METRICS_HOST] ?: "",
            metricsPort = prefs[Keys.METRICS_PORT] ?: OpsSettings.DEFAULT_METRICS_PORT,
            mnsHost = prefs[Keys.MNS_HOST] ?: "",
            mnsPort = prefs[Keys.MNS_PORT] ?: OpsSettings.DEFAULT_MNS_PORT,
            registryHost = prefs[Keys.REGISTRY_HOST] ?: "",
            registryPort = prefs[Keys.REGISTRY_PORT] ?: OpsSettings.DEFAULT_REGISTRY_PORT,
            trainerHost = prefs[Keys.TRAINER_HOST] ?: "",
            trainerPort = prefs[Keys.TRAINER_PORT] ?: OpsSettings.DEFAULT_TRAINER_PORT,
            registryGroups = decodeGroups(prefs[Keys.REGISTRY_GROUPS]),
            basePollIntervalMs = prefs[Keys.BASE_POLL_INTERVAL_MS] ?: OpsSettings.DEFAULT_POLL_INTERVAL_MS,
            useHttpsRelay = prefs[Keys.USE_HTTPS_RELAY] ?: false,
            accessClientId = prefs[Keys.ACCESS_CLIENT_ID] ?: "",
            accessClientSecret = prefs[Keys.ACCESS_CLIENT_SECRET] ?: "",
            trainerAccessClientId = prefs[Keys.TRAINER_ACCESS_CLIENT_ID] ?: "",
            trainerAccessClientSecret = prefs[Keys.TRAINER_ACCESS_CLIENT_SECRET] ?: "",
            watchSyncEnabled = prefs[Keys.WATCH_SYNC_ENABLED] ?: true,
            watchSyncSessionKeyOverride = prefs[Keys.WATCH_SYNC_SESSION_KEY_OVERRIDE]?.takeIf { it.isNotBlank() },
        )
    }

    override suspend fun save(settings: OpsSettings) {
        context.dataStore.edit { prefs ->
            prefs[Keys.USE_SHARED_HOST] = settings.useSharedHost
            prefs[Keys.SHARED_HOST] = settings.sharedHost
            prefs[Keys.METRICS_HOST] = settings.metricsHost
            prefs[Keys.METRICS_PORT] = settings.metricsPort
            prefs[Keys.MNS_HOST] = settings.mnsHost
            prefs[Keys.MNS_PORT] = settings.mnsPort
            prefs[Keys.REGISTRY_HOST] = settings.registryHost
            prefs[Keys.REGISTRY_PORT] = settings.registryPort
            prefs[Keys.TRAINER_HOST] = settings.trainerHost
            prefs[Keys.TRAINER_PORT] = settings.trainerPort
            prefs[Keys.REGISTRY_GROUPS] = encodeGroups(settings.registryGroups)
            prefs[Keys.BASE_POLL_INTERVAL_MS] = settings.basePollIntervalMs
            prefs[Keys.USE_HTTPS_RELAY] = settings.useHttpsRelay
            prefs[Keys.ACCESS_CLIENT_ID] = settings.accessClientId
            prefs[Keys.ACCESS_CLIENT_SECRET] = settings.accessClientSecret
            prefs[Keys.TRAINER_ACCESS_CLIENT_ID] = settings.trainerAccessClientId
            prefs[Keys.TRAINER_ACCESS_CLIENT_SECRET] = settings.trainerAccessClientSecret
            prefs[Keys.WATCH_SYNC_ENABLED] = settings.watchSyncEnabled
            prefs[Keys.WATCH_SYNC_SESSION_KEY_OVERRIDE] = settings.watchSyncSessionKeyOverride ?: ""
        }
    }

    private fun encodeGroups(groups: List<String>): String = groups.joinToString(",")

    private fun decodeGroups(raw: String?): List<String> =
        raw?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() } ?: emptyList()
}
