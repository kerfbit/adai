package com.adai.wearcomplications

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import com.adai.wearsync.WearSyncDefaults
import com.adai.wearsync.WearSyncKeys
import com.google.android.gms.wearable.DataMap

/**
 * Scalar-only decode of the DataMap pushed by opsdashboard's `WatchSyncRepository`. Unlike
 * the old Canvas-renderer's snapshot, there's no series data here — a Watch Face Format
 * gauge only ever needs the current value plus its normalization bounds.
 */
data class TrainingSnapshot(
    val currentLoss: Double,
    val currentPerplexity: Double,
    val lossMin: Double,
    val lossMax: Double,
    val perplexityMin: Double,
    val perplexityMax: Double,
    val effectiveIsTraining: Boolean,
    val relaySyncEpochMillis: Long,
) {
    fun isStale(nowMillis: Long): Boolean =
        relaySyncEpochMillis <= 0L || (nowMillis - relaySyncEpochMillis) > WearSyncDefaults.STALE_AFTER_MILLIS

    companion object {
        fun fromDataMap(map: DataMap): TrainingSnapshot = TrainingSnapshot(
            currentLoss = map.getDouble(WearSyncKeys.CURRENT_LOSS, 0.0),
            currentPerplexity = map.getDouble(WearSyncKeys.CURRENT_PERPLEXITY, 0.0),
            lossMin = map.getDouble(WearSyncKeys.LOSS_MIN, 0.0),
            lossMax = map.getDouble(WearSyncKeys.LOSS_MAX, 0.0),
            perplexityMin = map.getDouble(WearSyncKeys.PERPLEXITY_MIN, 0.0),
            perplexityMax = map.getDouble(WearSyncKeys.PERPLEXITY_MAX, 0.0),
            effectiveIsTraining = map.getBoolean(WearSyncKeys.EFFECTIVE_IS_TRAINING, false),
            relaySyncEpochMillis = map.getLong(WearSyncKeys.RELAY_SYNC_EPOCH_MILLIS, 0L),
        )
    }
}
