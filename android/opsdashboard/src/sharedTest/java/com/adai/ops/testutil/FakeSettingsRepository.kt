package com.adai.ops.testutil

// @adai-status: beta        (in-memory fake backing both plain-JVM and instrumented tests, TD-048)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import com.adai.ops.settings.OpsSettings
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow

class FakeSettingsRepository(initial: OpsSettings = OpsSettings(useSharedHost = true, sharedHost = "localhost")) :
    OpsSettingsRepository {

    private val state = MutableStateFlow(initial)
    override val settings = state

    override suspend fun save(settings: OpsSettings) {
        state.value = settings
    }
}
