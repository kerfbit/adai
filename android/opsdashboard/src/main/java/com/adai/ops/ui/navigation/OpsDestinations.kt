package com.adai.ops.ui.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AdminPanelSettings
import androidx.compose.material.icons.filled.Dns
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.ui.graphics.vector.ImageVector

enum class OpsDestination(val route: String, val label: String, val icon: ImageVector) {
    METRICS("metrics", "Metrics", Icons.Filled.Timeline),
    MODELS("models", "Models", Icons.Filled.Dns),
    REGISTRY("registry", "Registry", Icons.Filled.Storage),
    ADMIN("admin", "Admin", Icons.Filled.AdminPanelSettings),
}
