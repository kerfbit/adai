package com.adai.ops.testutil

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
