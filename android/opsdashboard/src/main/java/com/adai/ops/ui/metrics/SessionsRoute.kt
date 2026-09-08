package com.adai.ops.ui.metrics

// @adai-status: experimental
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

/**
 * List-detail flow for Training Sessions: single-pane with back navigation on compact
 * width (phone portrait), side-by-side panes on expanded width (tablet / wide phone
 * landscape) — driven entirely by ListDetailPaneScaffold's adaptive directive, no
 * manual WindowSizeClass branching needed here.
 */
@OptIn(ExperimentalMaterial3AdaptiveApi::class)
@Composable
fun SessionsRoute(app: OpsApp, onOpenSettings: () -> Unit) {
    val listViewModel = viewModel<SessionListViewModel>(factory = AppViewModelProvider.factory(app))
    val navigator = rememberListDetailPaneScaffoldNavigator<String>()

    BackHandler(navigator.canNavigateBack()) {
        navigator.navigateBack()
    }

    ListDetailPaneScaffold(
        directive = navigator.scaffoldDirective,
        value = navigator.scaffoldValue,
        listPane = {
            AnimatedPane {
                SessionListScreen(
                    viewModel = listViewModel,
                    onOpenSession = { key -> navigator.navigateTo(ListDetailPaneScaffoldRole.Detail, key) },
                    onOpenSettings = onOpenSettings,
                )
            }
        },
        detailPane = {
            AnimatedPane {
                val sessionKey = navigator.currentDestination?.content
                if (sessionKey != null) {
                    val detailViewModel = viewModel<SessionDetailViewModel>(
                        key = sessionKey,
                        factory = AppViewModelProvider.sessionDetailFactory(app, sessionKey),
                    )
                    SessionDetailScreen(
                        sessionKey = sessionKey,
                        viewModel = detailViewModel,
                        onBack = { navigator.navigateBack() },
                        onEvicted = { navigator.navigateBack() },
                    )
                } else {
                    EmptyDetailPlaceholder("Select a session")
                }
            }
        },
    )
}
