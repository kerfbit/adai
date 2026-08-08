package com.adai.ops.data.wearsync

import android.content.Context
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.SampleRecordDto
import com.adai.ops.settings.OpsSettingsRepository
import com.adai.wearsync.WearSyncDefaults
import com.adai.wearsync.WearSyncKeys
import com.adai.wearsync.WearSyncPaths
import com.adai.wearsync.minMax
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.PutDataMapRequest
import com.google.android.gms.wearable.Wearable
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext

/**
 * Phone-side producer for the watch's loss/perplexity complications (see
 * `android/wearcomplications`): resolves which session to track (see [ActiveSessionSelector]),
 * fetches its whole-session sample history to compute normalization bounds via [minMax], and
 * pushes a single always-latest DataItem at [WearSyncPaths.TRAINING_SNAPSHOT]. Invoked both
 * from `SessionDetailViewModel`'s slow-cadence poll tick (foreground) and from a periodic
 * WorkManager job (background) — see `WatchSyncWorker`.
 *
 * The watch face itself (`android/wearface`, Watch Face Format) never touches the Data Layer
 * directly — it only reads the RANGED_VALUE complications the two services in
 * `wearcomplications` populate from this same pushed snapshot.
 */
class WatchSyncRepository(
    context: Context,
    private val metricsRepository: MetricsRepository,
    private val settingsRepository: OpsSettingsRepository,
) {
    private val dataClient = Wearable.getDataClient(context.applicationContext)

    /**
     * Returns true if a snapshot was successfully built and pushed. Returns false (rather than
     * throwing) if sync is disabled, no session could be resolved, or a fetch failed — the
     * caller (foreground tick or background worker) decides whether/how to log or retry.
     */
    suspend fun sync(): Boolean {
        val settings = settingsRepository.settings.first()
        if (!settings.watchSyncEnabled) return false

        val key = resolveSessionKey(settings.watchSyncSessionKeyOverride) ?: return false
        val status = (metricsRepository.sessionStatus(key) as? ApiResult.Success)?.data ?: return false
        val current = (metricsRepository.currentMetrics(key) as? ApiResult.Success)?.data
        val records = fetchWholeSessionRecords(key) ?: return false

        val lossBounds = minMax(records.map { it.loss })
        val perplexityBounds = minMax(records.map { it.perplexity })
        val serverLastUpdateMillis =
            System.currentTimeMillis() - (status.seconds_since_last_update * 1000).toLong()

        val putRequest = PutDataMapRequest.create(WearSyncPaths.TRAINING_SNAPSHOT).apply {
            dataMap.putInt(WearSyncKeys.SCHEMA_VERSION, WearSyncDefaults.SCHEMA_VERSION)
            dataMap.putString(WearSyncKeys.SESSION_KEY, key)
            dataMap.putString(
                WearSyncKeys.SESSION_LABEL,
                "$key · epoch ${status.current_epoch}/${status.total_epochs}",
            )
            dataMap.putBoolean(WearSyncKeys.IS_TRAINING, status.is_training)
            dataMap.putBoolean(WearSyncKeys.EFFECTIVE_IS_TRAINING, status.effective_is_training)
            dataMap.putInt(WearSyncKeys.CURRENT_EPOCH, status.current_epoch)
            dataMap.putInt(WearSyncKeys.TOTAL_EPOCHS, status.total_epochs)
            dataMap.putDouble(WearSyncKeys.CURRENT_LOSS, current?.current_loss ?: 0.0)
            dataMap.putDouble(WearSyncKeys.CURRENT_PERPLEXITY, current?.current_perplexity ?: 0.0)
            dataMap.putDouble(WearSyncKeys.LOSS_MIN, lossBounds.min)
            dataMap.putDouble(WearSyncKeys.LOSS_MAX, lossBounds.max)
            dataMap.putDouble(WearSyncKeys.PERPLEXITY_MIN, perplexityBounds.min)
            dataMap.putDouble(WearSyncKeys.PERPLEXITY_MAX, perplexityBounds.max)
            dataMap.putLong(WearSyncKeys.SERVER_LAST_UPDATE_EPOCH_MILLIS, serverLastUpdateMillis)
            dataMap.putLong(WearSyncKeys.RELAY_SYNC_EPOCH_MILLIS, System.currentTimeMillis())
        }.asPutDataRequest().setUrgent()

        return withContext(Dispatchers.IO) {
            runCatching { Tasks.await(dataClient.putDataItem(putRequest)) }.isSuccess
        }
    }

    private suspend fun resolveSessionKey(override: String?): String? {
        if (!override.isNullOrBlank()) return override
        val sessions = (metricsRepository.listSessions() as? ApiResult.Success)?.data?.sessions ?: return null
        return ActiveSessionSelector.select(sessions)?.key
    }

    private suspend fun fetchWholeSessionRecords(key: String): List<SampleRecordDto>? {
        val dbResult = metricsRepository.dbHistory(key)
        if (dbResult is ApiResult.Success) return dbResult.data.records

        // No DB backend configured (METRICS_STORAGE_BACKEND=file) — best available
        // approximation is the in-memory ring buffer, capped at 10k records.
        val fallback = metricsRepository.sampleHistory(key, maxRecords = 10_000)
        return (fallback as? ApiResult.Success)?.data?.records
    }
}
