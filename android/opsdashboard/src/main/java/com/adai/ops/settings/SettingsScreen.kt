package com.adai.ops.settings

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

private val POLL_INTERVAL_OPTIONS = listOf(1000L, 2000L, 5000L, 10000L)
private const val SET_ACTIVE_PERMISSION = "com.google.wear.permission.SET_PUSHED_WATCH_FACE_AS_ACTIVE"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(viewModel: SettingsViewModel, onBack: () -> Unit) {
    val state by viewModel.uiState.collectAsState()

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
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            Text(
                "Point this app at metrics_api_server, mns_server, and registry_server. " +
                    "All three use plain HTTP with no authentication — only use this on a " +
                    "trusted LAN or VPN.",
                style = MaterialTheme.typography.bodyLarge,
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Switch(checked = state.useSharedHost, onCheckedChange = viewModel::onUseSharedHostChanged)
                Text("Use one host for all services", style = MaterialTheme.typography.bodyLarge)
            }

            if (state.useSharedHost) {
                OutlinedTextField(
                    value = state.sharedHost,
                    onValueChange = viewModel::onSharedHostChanged,
                    label = {
                        Text(
                            if (state.useHttpsRelay) "Host (e.g. metrics.kerfbit.dev)"
                            else "Host (e.g. 192.168.1.16)"
                        )
                    },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            HorizontalDivider()

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Switch(checked = state.useHttpsRelay, onCheckedChange = viewModel::onUseHttpsRelayChanged)
                Text("Use secure relay (HTTPS via kerfbit.dev)", style = MaterialTheme.typography.bodyLarge)
            }

            if (state.useHttpsRelay) {
                Text(
                    "Chat, Metrics, Models, and Registry share one Access token:",
                    style = MaterialTheme.typography.bodySmall,
                )
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

            HorizontalDivider()

            ServiceHostRow(
                label = "Metrics (metrics_api_server)",
                host = state.metricsHost,
                port = state.metricsPort,
                showHost = !state.useSharedHost,
                showPort = !state.useHttpsRelay,
                onHostChanged = viewModel::onMetricsHostChanged,
                onPortChanged = viewModel::onMetricsPortChanged,
            )
            ServiceHostRow(
                label = "Models (mns_server)",
                host = state.mnsHost,
                port = state.mnsPort,
                showHost = !state.useSharedHost,
                showPort = !state.useHttpsRelay,
                onHostChanged = viewModel::onMnsHostChanged,
                onPortChanged = viewModel::onMnsPortChanged,
            )
            ServiceHostRow(
                label = "Registry (registry_server)",
                host = state.registryHost,
                port = state.registryPort,
                showHost = !state.useSharedHost,
                showPort = !state.useHttpsRelay,
                onHostChanged = viewModel::onRegistryHostChanged,
                onPortChanged = viewModel::onRegistryPortChanged,
            )
            ServiceHostRow(
                label = "Trainer (incremental_trainer serve, opt-in)",
                host = state.trainerHost,
                port = state.trainerPort,
                showHost = !state.useSharedHost,
                showPort = !state.useHttpsRelay,
                onHostChanged = viewModel::onTrainerHostChanged,
                onPortChanged = viewModel::onTrainerPortChanged,
            )
            if (state.useHttpsRelay) {
                Text(
                    "Trainer can pause/resume/checkpoint a live training run, so it uses its " +
                        "own separately-revocable Access token, not the one above:",
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedTextField(
                    value = state.trainerAccessClientId,
                    onValueChange = viewModel::onTrainerAccessClientIdChanged,
                    label = { Text("Trainer Access Client ID") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = state.trainerAccessClientSecret,
                    onValueChange = viewModel::onTrainerAccessClientSecretChanged,
                    label = { Text("Trainer Access Client Secret") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            HorizontalDivider()

            Text("Base poll interval (Metrics live view adapts faster automatically)", style = MaterialTheme.typography.bodyLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                POLL_INTERVAL_OPTIONS.forEach { ms ->
                    FilterChip(
                        selected = state.basePollIntervalMs == ms,
                        onClick = { viewModel.onBasePollIntervalChanged(ms) },
                        label = { Text("${ms / 1000}s") },
                    )
                }
            }

            HorizontalDivider()

            Text(
                "Dataset registry groups (there is no server-side way to discover " +
                    "these — add the group names you use)",
                style = MaterialTheme.typography.bodyLarge,
            )
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = state.newGroupInput,
                    onValueChange = viewModel::onNewGroupInputChanged,
                    label = { Text("Group name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedButton(
                    onClick = viewModel::addGroup,
                    modifier = Modifier.align(Alignment.End),
                ) {
                    Text("Add")
                }
            }
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.registryGroups) { group ->
                    FilterChip(
                        selected = false,
                        onClick = { },
                        label = { Text(group) },
                        trailingIcon = {
                            IconButton(onClick = { viewModel.removeGroup(group) }) {
                                Icon(Icons.Filled.Close, contentDescription = "Remove $group")
                            }
                        },
                    )
                }
            }

            HorizontalDivider()

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Switch(checked = state.watchSyncEnabled, onCheckedChange = viewModel::onWatchSyncEnabledChanged)
                Text("Sync training metrics to Galaxy Watch face", style = MaterialTheme.typography.bodyLarge)
            }
            if (state.watchSyncEnabled) {
                OutlinedTextField(
                    value = state.watchSyncSessionKeyOverride,
                    onValueChange = viewModel::onWatchSyncSessionKeyOverrideChanged,
                    label = { Text("Session key override (blank = Auto)") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "Auto tracks whichever session is currently training, or the most " +
                        "recently updated one if none are.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            HorizontalDivider()

            Text("Galaxy Watch face (Watch Face Format)", style = MaterialTheme.typography.titleLarge)
            Text(
                "Pushes the adai training-metrics watch face directly to the paired watch " +
                    "via Watch Face Push, bypassing Samsung's picker (which doesn't list " +
                    "sideloaded watch faces).",
                style = MaterialTheme.typography.bodySmall,
            )

            val context = LocalContext.current
            val activatePermissionLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestPermission(),
            ) { granted -> if (granted) viewModel.activateWatchFace() }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = viewModel::pushWatchFace, enabled = !state.watchFacePushInProgress) {
                    if (state.watchFacePushInProgress) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                    } else {
                        Text("Install / update")
                    }
                }
                if (state.pushedWatchFaceSlotId != null) {
                    OutlinedButton(onClick = {
                        val alreadyGranted = ContextCompat.checkSelfPermission(
                            context,
                            SET_ACTIVE_PERMISSION,
                        ) == PackageManager.PERMISSION_GRANTED
                        if (alreadyGranted) {
                            viewModel.activateWatchFace()
                        } else {
                            activatePermissionLauncher.launch(SET_ACTIVE_PERMISSION)
                        }
                    }) {
                        Text("Activate")
                    }
                }
            }
            state.watchFacePushMessage?.let { message ->
                Text(message, style = MaterialTheme.typography.bodyMedium)
            }

            Button(onClick = {
                viewModel.save()
                onBack()
            }) {
                Text("Save")
            }
        }
    }
}

@Composable
private fun ServiceHostRow(
    label: String,
    host: String,
    port: String,
    showHost: Boolean,
    showPort: Boolean,
    onHostChanged: (String) -> Unit,
    onPortChanged: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, style = MaterialTheme.typography.titleLarge)
        if (showHost) {
            OutlinedTextField(
                value = host,
                onValueChange = onHostChanged,
                label = { Text("Host") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        if (showPort) {
            OutlinedTextField(
                value = port,
                onValueChange = onPortChanged,
                label = { Text("Port") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}
