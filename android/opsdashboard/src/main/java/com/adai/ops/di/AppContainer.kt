package com.adai.ops.di

import android.content.Context
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.data.wearsync.WatchFacePushRepository
import com.adai.ops.data.wearsync.WatchSyncRepository
import com.adai.ops.network.ApiClientProvider
import com.adai.ops.settings.OpsSettingsDataStore
import com.adai.ops.ui.common.AdminAuthGate
import com.adai.ops.ui.common.BiometricAdminAuthGate

/**
 * Small, manually-wired dependency container — mirrors the chatbot app's AppContainer.
 * This app has a handful of screens/repositories, so a DI framework would add
 * annotation-processor overhead without buying much here.
 */
class AppContainer(context: Context) {

    val settingsDataStore = OpsSettingsDataStore(context)

    val apiClientProvider = ApiClientProvider()

    val metricsRepository = MetricsRepository(apiClientProvider, settingsDataStore)
    val modelRepository = ModelRepository(apiClientProvider, settingsDataStore)
    val registryRepository = RegistryRepository(apiClientProvider, settingsDataStore)
    val watchSyncRepository = WatchSyncRepository(context, metricsRepository, settingsDataStore)
    val watchFacePushRepository = WatchFacePushRepository(context)

    // Singleton so BiometricAdminAuthGate's grace-period timer is shared across every
    // screen/action, not reset each time a screen recomposes.
    val adminAuthGate: AdminAuthGate = BiometricAdminAuthGate()
}
