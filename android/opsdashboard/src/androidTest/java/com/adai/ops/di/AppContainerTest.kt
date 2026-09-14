package com.adai.ops.di

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.test.platform.app.InstrumentationRegistry
import com.adai.ops.ui.common.BiometricAdminAuthGate
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * TD-048: AppContainer (opsdashboard module) had zero test coverage — this is the app's real
 * manual-DI graph, constructed once per real `Context` in `OpsApp.onCreate()`. Unlike the `app`
 * module's own AppContainerTest (see its doc comment), this constructs its own `AppContainer`
 * directly (not the app-module pattern of reaching `OpsApp.container`) since none of these
 * repositories touch a persistent on-disk store the way Room does -- `OpsSettingsDataStore` is
 * the one exception, backed by DataStore Preferences (a real file too), so its own read is only
 * checked for not throwing, not for round-tripping a written value (that's already covered by
 * `OpsSettingsDataStoreTest`).
 *
 * NOTE: same sandbox limitation as the other opsdashboard tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here (this module's
 * debug APK can't be installed on this sandbox's emulator at all, see TECHNICAL_DEBT.md's
 * `wear-sdk` note).
 */
class AppContainerTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun everyRepositoryAndGateIsConstructedAndUsable() = runBlocking {
        val container = AppContainer(context)

        assertNotNull(container.settingsDataStore)
        assertNotNull(container.apiClientProvider)
        assertNotNull(container.metricsRepository)
        assertNotNull(container.modelRepository)
        assertNotNull(container.registryRepository)
        assertNotNull(container.trainerRepository)
        assertNotNull(container.watchSyncRepository)
        assertNotNull(container.watchFacePushRepository)

        // Not asserting a specific value -- just that reading doesn't throw, proving
        // OpsSettingsDataStore is a real, usable DataStore-backed object rather than something
        // that only looks wired until first touched.
        container.settingsDataStore.settings.first()

        // BiometricAdminAuthGate specifically (not just "any AdminAuthGate") -- production wiring
        // shouldn't silently regress to a different implementation.
        assertTrue(container.adminAuthGate is BiometricAdminAuthGate)
    }
}
