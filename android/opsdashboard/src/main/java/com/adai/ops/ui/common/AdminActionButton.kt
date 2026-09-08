package com.adai.ops.ui.common

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

/**
 * Visually distinct button for the app's admin (mutating) actions — always paired with
 * [ConfirmActionDialog] so nothing fires without an explicit, described confirmation.
 */
@Composable
fun AdminActionButton(
    label: String,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
        modifier = modifier,
    ) {
        Icon(Icons.Filled.Warning, contentDescription = null)
        Text(" $label")
    }
}
