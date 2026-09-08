package com.adai.ops.ui.admin

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import com.adai.ops.OpsApp
import com.adai.ops.di.AppViewModelProvider

/** Thin wrapper matching RegistryRoute/ModelsRoute — the Admin tab has no list/detail split. */
@Composable
fun AdminRoute(app: OpsApp, onOpenSettings: () -> Unit) {
    val viewModel = viewModel<AdminViewModel>(factory = AppViewModelProvider.adminFactory(app))
    AdminScreen(viewModel = viewModel, onOpenSettings = onOpenSettings)
}
