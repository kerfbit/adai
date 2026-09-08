package com.adai.ops.ui.trainer

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import com.adai.ops.OpsApp
import com.adai.ops.di.AppViewModelProvider

/** Thin wrapper matching AdminRoute — the Trainer tab has no list/detail split. */
@Composable
fun TrainerRoute(app: OpsApp, onOpenSettings: () -> Unit) {
    val viewModel = viewModel<TrainerViewModel>(factory = AppViewModelProvider.trainerFactory(app))
    TrainerScreen(viewModel = viewModel, onOpenSettings = onOpenSettings)
}
