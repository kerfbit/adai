package com.adai.ops.ui.common

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.fragment.app.FragmentActivity
import kotlinx.coroutines.launch

/**
 * Confirmation dialog for every admin action in the app (clear stale MNS lock, retire,
 * promote, registry force-release, metrics end-session, and admin-config saves). Always
 * states the literal HTTP call it will make so the operator knows exactly what they're
 * triggering before it happens, and gates [onConfirm] behind a device-credential
 * (PIN/pattern/password/biometric) check via [LocalAdminAuthGate] — this is the single seam
 * every admin action flows through, so no call site needs its own auth wiring.
 */
@Composable
fun ConfirmActionDialog(
    title: String,
    httpCallDescription: String,
    effectDescription: String,
    confirmLabel: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    val authGate = LocalAdminAuthGate.current
    val activity = LocalContext.current as FragmentActivity
    val scope = rememberCoroutineScope()
    var authInFlight by remember { mutableStateOf(false) }
    var authError by remember { mutableStateOf<String?>(null) }

    AlertDialog(
        onDismissRequest = onDismiss,
        icon = { Icon(Icons.Filled.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error) },
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(
                    httpCallDescription,
                    fontFamily = FontFamily.Monospace,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(6.dp))
                        .padding(8.dp),
                )
                Text(effectDescription, style = MaterialTheme.typography.bodyLarge)
                authError?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyLarge) }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    authError = null
                    authInFlight = true
                    scope.launch {
                        when (val result = authGate.authenticate(activity, reason = title)) {
                            is AdminAuthResult.Success -> onConfirm()
                            is AdminAuthResult.Cancelled -> Unit
                            is AdminAuthResult.Unavailable -> authError = result.message
                            is AdminAuthResult.Failed -> authError = result.message
                        }
                        authInFlight = false
                    }
                },
                enabled = !authInFlight,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
            ) {
                if (authInFlight) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                } else {
                    Text(confirmLabel)
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        },
    )
}
