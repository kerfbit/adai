package com.adai.ops.data.wearsync

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-13


sealed interface WatchFacePushResult {
    data class Success(val slotId: String) : WatchFacePushResult
    data class ValidationFailed(val reasons: List<String>) : WatchFacePushResult
    data class Failure(val message: String) : WatchFacePushResult
}

/**
 * Pushes the WFF `:wearface` bundle directly to the paired watch via Watch Face Push. See
 * [WearWatchFacePushRepository] for the real implementation (the only one wired in production,
 * via `AppContainer`) and `FakeWatchFacePushRepository`
 * (`android/opsdashboard/src/sharedTest/java/com/adai/ops/testutil/FakeWatchFacePushRepository.kt`)
 * for the in-memory fake used by `SettingsViewModelTest`/`SettingsScreenTest`. Split out as an
 * interface (TD-048) specifically so `SettingsViewModel` — the sole consumer — didn't need a real
 * Wear system service, `androidx.wear.watchfacepush`'s `WatchFacePushManagerFactory`, just to be
 * unit-testable.
 */
interface WatchFacePushRepository {
    fun isSupported(): Boolean

    /** Pushes (or updates, if already installed) the bundled watch face. Does not activate it
     * — call [setActive] separately, since the platform's own "set active" call can only be
     * invoked once per install and needs its own explicit user confirmation. */
    suspend fun pushWatchFace(): WatchFacePushResult

    /** Only callable once per install per the platform's own limit — surface that to the
     * user rather than silently retrying. */
    suspend fun setActive(slotId: String): Result<Unit>
}
