package com.adai.ops.ui.common

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable

/** Settings action shown in each top-level section's TopAppBar. */
@Composable
fun SettingsAction(onOpenSettings: () -> Unit) {
    IconButton(onClick = onOpenSettings) {
        Icon(Icons.Filled.Settings, contentDescription = "Settings")
    }
}
