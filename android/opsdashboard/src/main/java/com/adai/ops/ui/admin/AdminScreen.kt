package com.adai.ops.ui.admin

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.adai.ops.network.dto.MetricsAdminConfigDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.SettingsAction
import kotlinx.coroutines.launch

/**
 * One section per daemon (MNS / Registry / Metrics) showing that daemon's current
 * /admin/config values, with a gated edit-and-save per mutable field. A disabled/absent
 * section (403 admin-disabled for MNS/Registry, 404 admin-routes-not-registered for
 * Metrics — see AdminUiState's doc comment) shows its error text instead of edit fields,
 * with no dead Save controls. Every save flows through [ConfirmActionDialog], which is
 * itself gated by the device-credential check — no separate auth wiring needed here.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AdminScreen(viewModel: AdminViewModel, onOpenSettings: () -> Unit) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    LaunchedEffect(state.actionMessage) {
        state.actionMessage?.let {
            scope.launch { snackbarHostState.showSnackbar(it) }
            viewModel.dismissActionMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Admin") },
                actions = { SettingsAction(onOpenSettings) },
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) { Snackbar(it) } },
    ) { padding ->
        if (state.isLoading) {
            FullScreenLoading(Modifier.padding(padding))
        } else {
            LazyColumn(modifier = Modifier.padding(padding).fillMaxSize().padding(16.dp)) {
                item { MnsSection(state.mnsConfig, state.mnsError, state.actionInProgress, viewModel) }
                item { HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp)) }
                item { RegistrySection(state.registryConfig, state.registryError, state.actionInProgress, viewModel) }
                item { HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp)) }
                item { MetricsSection(state.metricsConfig, state.metricsError, state.actionInProgress, viewModel) }
            }
        }
    }
}

@Composable
private fun MnsSection(config: MnsAdminConfigDto?, error: String?, actionInProgress: Boolean, viewModel: AdminViewModel) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("mns_server", style = MaterialTheme.typography.titleLarge)
        when {
            config == null -> Text(error ?: "Not loaded")
            !config.admin_enabled -> Text(
                "Admin mutation disabled on this server (started with --admin-enabled=false).",
                color = MaterialTheme.colorScheme.error,
            )
            else -> {
                AdminStringFieldRow(
                    label = "registry_url",
                    jsonKey = "registry_url",
                    value = config.registry_url,
                    enabled = !actionInProgress,
                    onSave = viewModel::updateRegistryUrl,
                )
                AdminStringFieldRow(
                    label = "registry_group",
                    jsonKey = "registry_group",
                    value = config.registry_group,
                    enabled = !actionInProgress,
                    onSave = viewModel::updateRegistryGroup,
                )
            }
        }
    }
}

@Composable
private fun RegistrySection(
    config: RegistryAdminConfigDto?,
    error: String?,
    actionInProgress: Boolean,
    viewModel: AdminViewModel,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("registry_server", style = MaterialTheme.typography.titleLarge)
        when {
            config == null -> Text(error ?: "Not loaded")
            !config.admin_enabled -> Text(
                "Admin mutation disabled on this server (started with --admin-enabled=false).",
                color = MaterialTheme.colorScheme.error,
            )
            else -> {
                AdminIntFieldRow(
                    label = "ftp_token_ttl_minutes",
                    jsonKey = "ftp_token_ttl_minutes",
                    value = config.ftp_token_ttl_minutes,
                    enabled = !actionInProgress,
                    onSave = viewModel::updateFtpTokenTtlMinutes,
                )
                AdminIntFieldRow(
                    label = "ftp_max_sessions_per_run",
                    jsonKey = "ftp_max_sessions_per_run",
                    value = config.ftp_max_sessions_per_run,
                    enabled = !actionInProgress,
                    onSave = viewModel::updateFtpMaxSessionsPerRun,
                )
            }
        }
    }
}

@Composable
private fun MetricsSection(
    config: MetricsAdminConfigDto?,
    error: String?,
    actionInProgress: Boolean,
    viewModel: AdminViewModel,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("metrics_api_server", style = MaterialTheme.typography.titleLarge)
        when {
            config == null -> Text(
                error?.let {
                    if (it == "Not found") {
                        "Admin routes not enabled on this server (allow_control=false at startup)."
                    } else {
                        it
                    }
                } ?: "Not loaded",
            )
            !config.allow_control -> Text(
                "Admin mutation disabled on this server (allow_control=false).",
                color = MaterialTheme.colorScheme.error,
            )
            else -> {
                AdminIntFieldRow("max_live_sessions", "max_live_sessions", config.max_live_sessions, !actionInProgress, viewModel::updateMaxLiveSessions)
                AdminIntFieldRow("completed_ttl_seconds", "completed_ttl_seconds", config.completed_ttl_seconds, !actionInProgress, viewModel::updateCompletedTtlSeconds)
                AdminIntFieldRow("sweep_interval_seconds", "sweep_interval_seconds", config.sweep_interval_seconds, !actionInProgress, viewModel::updateSweepIntervalSeconds)
                AdminIntFieldRow("persist_every_samples", "persist_every_samples", config.persist_every_samples, !actionInProgress, viewModel::updatePersistEverySamples)
                AdminIntFieldRow("persist_every_seconds", "persist_every_seconds", config.persist_every_seconds, !actionInProgress, viewModel::updatePersistEverySeconds)
                AdminIntFieldRow("max_records_in_memory", "max_records_in_memory", config.max_records_in_memory, !actionInProgress, viewModel::updateMaxRecordsInMemory)
                AdminIntFieldRow("max_records_on_disk", "max_records_on_disk", config.max_records_on_disk, !actionInProgress, viewModel::updateMaxRecordsOnDisk)
                AdminIntFieldRow("staleness_threshold_seconds", "staleness_threshold_seconds", config.staleness_threshold_seconds, !actionInProgress, viewModel::updateStalenessThresholdSeconds)
                AdminBoolFieldRow("enable_prometheus", "enable_prometheus", config.enable_prometheus, !actionInProgress, viewModel::updateEnablePrometheus)
            }
        }
    }
}

/** Editable Int field: pencil icon opens a value-entry dialog, then the auth-gated [ConfirmActionDialog]. */
@Composable
private fun AdminIntFieldRow(label: String, jsonKey: String, value: Int, enabled: Boolean, onSave: (Int) -> Unit) {
    var showEditDialog by remember { mutableStateOf(false) }
    var pendingValue by remember { mutableStateOf<Int?>(null) }

    FieldRow(label, value.toString(), enabled) { showEditDialog = true }

    if (showEditDialog) {
        EditIntValueDialog(
            label = label,
            initialValue = value,
            onSubmit = { newValue -> showEditDialog = false; pendingValue = newValue },
            onDismiss = { showEditDialog = false },
        )
    }

    pendingValue?.let { newValue ->
        ConfirmActionDialog(
            title = "Update $label?",
            httpCallDescription = "PUT /admin/config\n{\"$jsonKey\": $newValue}",
            effectDescription = "Changes $label from $value to $newValue and persists it, " +
                "overlaying config on the next restart.",
            confirmLabel = "Save",
            onConfirm = { pendingValue = null; onSave(newValue) },
            onDismiss = { pendingValue = null },
        )
    }
}

/** Editable String field — same two-step flow as [AdminIntFieldRow]. */
@Composable
private fun AdminStringFieldRow(label: String, jsonKey: String, value: String, enabled: Boolean, onSave: (String) -> Unit) {
    var showEditDialog by remember { mutableStateOf(false) }
    var pendingValue by remember { mutableStateOf<String?>(null) }

    FieldRow(label, value.ifEmpty { "(empty)" }, enabled) { showEditDialog = true }

    if (showEditDialog) {
        EditStringValueDialog(
            label = label,
            initialValue = value,
            onSubmit = { newValue -> showEditDialog = false; pendingValue = newValue },
            onDismiss = { showEditDialog = false },
        )
    }

    pendingValue?.let { newValue ->
        ConfirmActionDialog(
            title = "Update $label?",
            httpCallDescription = "PUT /admin/config\n{\"$jsonKey\": \"$newValue\"}",
            effectDescription = "Changes $label from '$value' to '$newValue' and persists it, " +
                "overlaying config on the next restart.",
            confirmLabel = "Save",
            onConfirm = { pendingValue = null; onSave(newValue) },
            onDismiss = { pendingValue = null },
        )
    }
}

/** Editable Boolean field — a Switch directly requests confirmation, no separate value-entry step. */
@Composable
private fun AdminBoolFieldRow(label: String, jsonKey: String, value: Boolean, enabled: Boolean, onSave: (Boolean) -> Unit) {
    var pendingValue by remember { mutableStateOf<Boolean?>(null) }

    Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        Switch(checked = value, onCheckedChange = { pendingValue = it }, enabled = enabled)
    }

    pendingValue?.let { newValue ->
        ConfirmActionDialog(
            title = "Update $label?",
            httpCallDescription = "PUT /admin/config\n{\"$jsonKey\": $newValue}",
            effectDescription = "Changes $label from $value to $newValue and persists it, " +
                "overlaying config on the next restart.",
            confirmLabel = "Save",
            onConfirm = { pendingValue = null; onSave(newValue) },
            onDismiss = { pendingValue = null },
        )
    }
}

@Composable
private fun FieldRow(label: String, valueDisplay: String, enabled: Boolean, onRequestEdit: () -> Unit) {
    Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
        Column {
            Text(label, style = MaterialTheme.typography.bodyLarge)
            Text(valueDisplay, style = MaterialTheme.typography.bodyMedium)
        }
        IconButton(onClick = onRequestEdit, enabled = enabled) {
            Icon(Icons.Filled.Edit, contentDescription = "Edit $label")
        }
    }
}

@Composable
private fun EditIntValueDialog(label: String, initialValue: Int, onSubmit: (Int) -> Unit, onDismiss: () -> Unit) {
    var text by remember { mutableStateOf(initialValue.toString()) }
    val parsed = text.toIntOrNull()
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit $label") },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                label = { Text(label) },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                isError = parsed == null,
                modifier = Modifier.fillMaxWidth(),
            )
        },
        confirmButton = {
            Button(onClick = { onSubmit(parsed!!) }, enabled = parsed != null) { Text("Next") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun EditStringValueDialog(label: String, initialValue: String, onSubmit: (String) -> Unit, onDismiss: () -> Unit) {
    var text by remember { mutableStateOf(initialValue) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit $label") },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                label = { Text(label) },
                modifier = Modifier.fillMaxWidth(),
            )
        },
        confirmButton = { Button(onClick = { onSubmit(text) }) { Text("Next") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
