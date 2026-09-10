package com.adai.chatbot.settings

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-10


import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(viewModel: SettingsViewModel, onBack: () -> Unit) {
    val state by viewModel.uiState.collectAsState()
    val coroutineScope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                "Point this app at your chatbot_api_server instance. Since the server " +
                    "uses plain HTTP with no authentication, only use it on a trusted " +
                    "LAN or VPN.",
                style = MaterialTheme.typography.bodyLarge,
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Switch(checked = state.useHttps, onCheckedChange = viewModel::onUseHttpsChanged)
                Text("Use secure relay (HTTPS via kerfbit.dev)", style = MaterialTheme.typography.bodyLarge)
            }

            OutlinedTextField(
                value = state.host,
                onValueChange = viewModel::onHostChanged,
                label = {
                    Text(
                        if (state.useHttps) "Host (e.g. chat.kerfbit.dev)"
                        else "Host (e.g. 192.168.1.16 or 10.0.2.2 for emulator)"
                    )
                },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            if (!state.useHttps) {
                OutlinedTextField(
                    value = state.port,
                    onValueChange = viewModel::onPortChanged,
                    label = { Text("Port") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            if (state.useHttps) {
                OutlinedTextField(
                    value = state.accessClientId,
                    onValueChange = viewModel::onAccessClientIdChanged,
                    label = { Text("Cloudflare Access Client ID") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )

                OutlinedTextField(
                    value = state.accessClientSecret,
                    onValueChange = viewModel::onAccessClientSecretChanged,
                    label = { Text("Cloudflare Access Client Secret") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                ConnectionStatusDot(status = state.connectionStatus)
                Text(
                    state.connectionMessage ?: when (state.connectionStatus) {
                        ConnectionStatus.CHECKING -> "Checking..."
                        else -> "Not tested"
                    },
                    style = MaterialTheme.typography.bodyLarge,
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = viewModel::testConnection) {
                    Text("Test Connection")
                }
                Button(onClick = {
                    coroutineScope.launch {
                        viewModel.save()
                        onBack()
                    }
                }) {
                    Text("Save")
                }
            }
        }
    }
}

@Composable
private fun ConnectionStatusDot(status: ConnectionStatus) {
    val color = when (status) {
        ConnectionStatus.UNKNOWN -> Color.Gray
        ConnectionStatus.CHECKING -> Color(0xFFFFC107)
        ConnectionStatus.CONNECTED -> Color(0xFF4CAF50)
        ConnectionStatus.FAILED -> Color(0xFFF44336)
    }
    Box(
        modifier = Modifier
            .size(12.dp)
            .background(color = color, shape = CircleShape),
    )
}
