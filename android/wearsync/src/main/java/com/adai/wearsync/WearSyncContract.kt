package com.adai.wearsync

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


/**
 * Shared contract between the opsdashboard phone app (producer, pushes via the Data Layer)
 * and the wearcomplications watch app (consumer, feeds two ComplicationDataSourceServices)
 * for relaying adai training metrics. The watch face itself is a Watch Face Format (WFF)
 * resource-only bundle with no code — it only ever reads data through system-delivered
 * RANGED_VALUE complications, never touches the Data Layer directly. Kept dependency-free
 * (no play-services-wearable, no Android types) so [minMax] is plain-JVM unit testable;
 * each side's own DataMap put/get calls reference [WearSyncKeys] by name.
 */
object WearSyncPaths {
    /** Single always-latest DataItem path — the relay never needs more than one. */
    const val TRAINING_SNAPSHOT = "/adai/wearsync/training"
}

object WearSyncKeys {
    const val SCHEMA_VERSION = "schema_version"
    const val SESSION_KEY = "session_key"
    const val SESSION_LABEL = "session_label"
    const val IS_TRAINING = "is_training"
    const val EFFECTIVE_IS_TRAINING = "effective_is_training"
    const val CURRENT_EPOCH = "current_epoch"
    const val TOTAL_EPOCHS = "total_epochs"
    const val CURRENT_SAMPLE = "current_sample"
    const val TOTAL_SAMPLES = "total_samples"
    const val CURRENT_LOSS = "current_loss"
    const val CURRENT_PERPLEXITY = "current_perplexity"
    const val LOSS_MIN = "loss_min"
    const val LOSS_MAX = "loss_max"
    const val PERPLEXITY_MIN = "perplexity_min"
    const val PERPLEXITY_MAX = "perplexity_max"
    const val SERVER_LAST_UPDATE_EPOCH_MILLIS = "server_last_update_epoch_millis"
    const val RELAY_SYNC_EPOCH_MILLIS = "relay_sync_epoch_millis"
}

object WearSyncDefaults {
    /** Current payload shape version; bump when adding/removing/renaming DataMap keys. */
    const val SCHEMA_VERSION = 2

    /** Past this age, complications should report stale/no data rather than a frozen value. */
    const val STALE_AFTER_MILLIS = 10 * 60 * 1000L
}

/** Normalization bounds for a RANGED_VALUE complication gauge. */
data class MinMax(val min: Double, val max: Double)

/**
 * Reduces a whole session's raw values down to the min/max a RANGED_VALUE complication gauge
 * needs for normalization — the full per-sample series itself has nowhere to go once rendered
 * as a single gauge (Watch Face Format has no primitive for an arbitrary multi-point polyline).
 * Non-finite values are dropped; an all-non-finite or empty input returns 0.0/0.0.
 */
fun minMax(values: List<Double>): MinMax {
    val finite = values.filter { it.isFinite() }
    if (finite.isEmpty()) return MinMax(0.0, 0.0)
    return MinMax(finite.min(), finite.max())
}
