package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


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
fun RegistryRoute(app: OpsApp, onOpenSettings: () -> Unit) {
    val listViewModel = viewModel<GroupListViewModel>(factory = AppViewModelProvider.factory(app))
    val navigator = rememberListDetailPaneScaffoldNavigator<String>()

    BackHandler(navigator.canNavigateBack()) {
        navigator.navigateBack()
    }

    ListDetailPaneScaffold(
        directive = navigator.scaffoldDirective,
        value = navigator.scaffoldValue,
        listPane = {
            AnimatedPane {
                GroupListScreen(
                    viewModel = listViewModel,
                    onOpenGroup = { name -> navigator.navigateTo(ListDetailPaneScaffoldRole.Detail, name) },
                    onOpenSettings = onOpenSettings,
                )
            }
        },
        detailPane = {
            AnimatedPane {
                val group = navigator.currentDestination?.content
                if (group != null) {
                    val detailViewModel = viewModel<GroupDetailViewModel>(
                        key = group,
                        factory = AppViewModelProvider.groupDetailFactory(app, group),
                    )
                    GroupDetailScreen(
                        group = group,
                        viewModel = detailViewModel,
                        onBack = { navigator.navigateBack() },
                    )
                } else {
                    EmptyDetailPlaceholder("Select a registry group")
                }
            }
        },
    )
}
