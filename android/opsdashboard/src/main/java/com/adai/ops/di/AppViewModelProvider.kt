package com.adai.ops.di

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
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

/**
 * Builds ViewModels from [OpsApp.container] instead of a Hilt graph — see
 * AppContainer's doc comment for why manual DI was chosen at this app's scope.
 */
object AppViewModelProvider {

    fun factory(app: OpsApp) = viewModelFactory {
        initializer { SettingsViewModel(app.container.settingsDataStore, app.container.watchFacePushRepository) }
        initializer { SessionListViewModel(app.container.metricsRepository) }
        initializer { ModelListViewModel(app.container.modelRepository) }
        initializer { GroupListViewModel(app.container.registryRepository, app.container.settingsDataStore) }
    }

    fun sessionDetailFactory(app: OpsApp, sessionKey: String) = viewModelFactory {
        initializer {
            SessionDetailViewModel(
                sessionKey,
                app.container.metricsRepository,
                app.container.settingsDataStore,
                app.container.watchSyncRepository,
            )
        }
    }

    fun modelDetailFactory(app: OpsApp, modelName: String) = viewModelFactory {
        initializer { ModelDetailViewModel(modelName, app.container.modelRepository) }
    }

    fun groupDetailFactory(app: OpsApp, group: String) = viewModelFactory {
        initializer {
            GroupDetailViewModel(group, app.container.registryRepository, app.container.modelRepository)
        }
    }

    fun adminFactory(app: OpsApp) = viewModelFactory {
        initializer {
            AdminViewModel(app.container.modelRepository, app.container.registryRepository, app.container.metricsRepository)
        }
    }

    fun trainerFactory(app: OpsApp) = viewModelFactory {
        initializer { TrainerViewModel(app.container.trainerRepository) }
    }
}
