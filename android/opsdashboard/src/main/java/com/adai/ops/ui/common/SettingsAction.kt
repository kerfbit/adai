package com.adai.ops.ui.common

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
