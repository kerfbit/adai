package com.adai.ops.di

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.lifecycle.viewmodel.CreationExtras
import androidx.test.core.app.ApplicationProvider
import com.adai.ops.OpsApp
import com.adai.ops.settings.SettingsViewModel
import com.adai.ops.ui.admin.AdminViewModel
import com.adai.ops.ui.metrics.SessionDetailViewModel
import com.adai.ops.ui.metrics.SessionListViewModel
import com.adai.ops.ui.models.ModelDetailViewModel
import com.adai.ops.ui.models.ModelListViewModel
import com.adai.ops.ui.registry.GroupDetailViewModel
import com.adai.ops.ui.registry.GroupListViewModel
import com.adai.ops.ui.trainer.TrainerViewModel
import org.junit.Assert.assertNotNull
import org.junit.Test

/**
 * TD-048: AppViewModelProvider (opsdashboard module) had zero test coverage. Uses the real,
 * already-running [OpsApp] singleton (same reasoning as the `app` module's own
 * AppViewModelProviderTest -- reaching the live Application instance via [ApplicationProvider]
 * exercises these factories exactly the way [com.adai.ops.ui.navigation.OpsNavHost] uses them in
 * production, rather than constructing a second, competing `AppContainer`).
 *
 * `viewModelFactory { initializer { ... } }`'s legacy single-arg `create(Class<T>)` throws
 * `UnsupportedOperationException` for factories built this way -- confirmed directly on a real
 * device while building the `app` module's own equivalent test -- so every call here uses the
 * `create(Class<T>, CreationExtras)` overload with `CreationExtras.Empty` instead.
 *
 * NOTE: same sandbox limitation as the other opsdashboard tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class AppViewModelProviderTest {

    private val app = ApplicationProvider.getApplicationContext<OpsApp>()

    @Test
    fun factory_createsEachRegisteredViewModelType() {
        val factory = AppViewModelProvider.factory(app)

        assertNotNull(factory.create(SettingsViewModel::class.java, CreationExtras.Empty))
        assertNotNull(factory.create(SessionListViewModel::class.java, CreationExtras.Empty))
        assertNotNull(factory.create(ModelListViewModel::class.java, CreationExtras.Empty))
        assertNotNull(factory.create(GroupListViewModel::class.java, CreationExtras.Empty))
    }

    @Test
    fun sessionDetailFactory_createsASessionDetailViewModel() {
        val viewModel = AppViewModelProvider.sessionDetailFactory(app, "session-1")
            .create(SessionDetailViewModel::class.java, CreationExtras.Empty)

        assertNotNull(viewModel)
    }

    @Test
    fun modelDetailFactory_createsAModelDetailViewModel() {
        val viewModel = AppViewModelProvider.modelDetailFactory(app, "some-model")
            .create(ModelDetailViewModel::class.java, CreationExtras.Empty)

        assertNotNull(viewModel)
    }

    @Test
    fun groupDetailFactory_createsAGroupDetailViewModel() {
        val viewModel = AppViewModelProvider.groupDetailFactory(app, "some-group")
            .create(GroupDetailViewModel::class.java, CreationExtras.Empty)

        assertNotNull(viewModel)
    }

    @Test
    fun adminFactory_createsAnAdminViewModel() {
        val viewModel = AppViewModelProvider.adminFactory(app).create(AdminViewModel::class.java, CreationExtras.Empty)

        assertNotNull(viewModel)
    }

    @Test
    fun trainerFactory_createsATrainerViewModel() {
        val viewModel = AppViewModelProvider.trainerFactory(app).create(TrainerViewModel::class.java, CreationExtras.Empty)

        assertNotNull(viewModel)
    }
}
