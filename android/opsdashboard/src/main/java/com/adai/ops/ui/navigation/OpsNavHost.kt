package com.adai.ops.ui.navigation

import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.adai.ops.OpsApp
import com.adai.ops.di.AppViewModelProvider
import com.adai.ops.settings.SettingsScreen
import com.adai.ops.settings.SettingsViewModel
import com.adai.ops.ui.admin.AdminRoute
import com.adai.ops.ui.metrics.SessionsRoute
import com.adai.ops.ui.models.ModelsRoute
import com.adai.ops.ui.registry.RegistryRoute

private const val SETTINGS_ROUTE = "settings"

@Composable
fun OpsNavHost(app: OpsApp, navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = OpsDestination.METRICS.route) {
        composable(OpsDestination.METRICS.route) {
            SessionsRoute(app, onOpenSettings = { navController.navigate(SETTINGS_ROUTE) })
        }
        composable(OpsDestination.MODELS.route) {
            ModelsRoute(app, onOpenSettings = { navController.navigate(SETTINGS_ROUTE) })
        }
        composable(OpsDestination.REGISTRY.route) {
            RegistryRoute(app, onOpenSettings = { navController.navigate(SETTINGS_ROUTE) })
        }
        composable(OpsDestination.ADMIN.route) {
            AdminRoute(app, onOpenSettings = { navController.navigate(SETTINGS_ROUTE) })
        }
        composable(SETTINGS_ROUTE) {
            val viewModel = viewModel<SettingsViewModel>(factory = AppViewModelProvider.factory(app))
            SettingsScreen(viewModel = viewModel, onBack = { navController.popBackStack() })
        }
    }
}
