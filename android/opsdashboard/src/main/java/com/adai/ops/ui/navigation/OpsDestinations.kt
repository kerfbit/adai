package com.adai.ops.ui.navigation

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AdminPanelSettings
import androidx.compose.material.icons.filled.Dns
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.ui.graphics.vector.ImageVector

enum class OpsDestination(val route: String, val label: String, val icon: ImageVector) {
    METRICS("metrics", "Metrics", Icons.Filled.Timeline),
    MODELS("models", "Models", Icons.Filled.Dns),
    REGISTRY("registry", "Registry", Icons.Filled.Storage),
    TRAINER("trainer", "Trainer", Icons.Filled.ModelTraining),
    ADMIN("admin", "Admin", Icons.Filled.AdminPanelSettings),
}
