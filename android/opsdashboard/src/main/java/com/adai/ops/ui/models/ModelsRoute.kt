package com.adai.ops.ui.models

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-10


import androidx.activity.compose.BackHandler
import androidx.compose.material3.adaptive.ExperimentalMaterial3AdaptiveApi
import androidx.compose.material3.adaptive.layout.AnimatedPane
import androidx.compose.material3.adaptive.layout.ListDetailPaneScaffold
import androidx.compose.material3.adaptive.layout.ListDetailPaneScaffoldRole
import androidx.compose.material3.adaptive.navigation.rememberListDetailPaneScaffoldNavigator
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import com.adai.ops.OpsApp
import com.adai.ops.di.AppViewModelProvider
import com.adai.ops.ui.common.EmptyDetailPlaceholder

@OptIn(ExperimentalMaterial3AdaptiveApi::class)
@Composable
fun ModelsRoute(app: OpsApp, onOpenSettings: () -> Unit) {
    val listViewModel = viewModel<ModelListViewModel>(factory = AppViewModelProvider.factory(app))
    val navigator = rememberListDetailPaneScaffoldNavigator<String>()

    BackHandler(navigator.canNavigateBack()) {
        navigator.navigateBack()
    }

    ListDetailPaneScaffold(
        directive = navigator.scaffoldDirective,
        value = navigator.scaffoldValue,
        listPane = {
            AnimatedPane {
                ModelListScreen(
                    viewModel = listViewModel,
                    onOpenModel = { name -> navigator.navigateTo(ListDetailPaneScaffoldRole.Detail, name) },
                    onOpenSettings = onOpenSettings,
                )
            }
        },
        detailPane = {
            AnimatedPane {
                val modelName = navigator.currentDestination?.content
                if (modelName != null) {
                    val detailViewModel = viewModel<ModelDetailViewModel>(
                        key = modelName,
                        factory = AppViewModelProvider.modelDetailFactory(app, modelName),
                    )
                    ModelDetailScreen(
                        modelName = modelName,
                        viewModel = detailViewModel,
                        onBack = { navigator.navigateBack() },
                    )
                } else {
                    EmptyDetailPlaceholder("Select a model")
                }
            }
        },
    )
}
