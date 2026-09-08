package com.adai.ops.ui.trainer

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerStatusDto
import com.adai.ops.ui.common.AdminActionButton
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.SettingsAction
import com.adai.ops.ui.common.StatusBadge
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.launch

private enum class TrainerPendingAction { NONE, CHECKPOINT, PAUSE, RESUME }

/**
 * incremental_trainer serve's admin tab: live /admin/status (phase/progress, polled —
 * see TrainerViewModel.pollStatus), the three control actions (checkpoint/pause/resume,
 * each gated by [ConfirmActionDialog]'s device-credential check), the /admin/config
 * field editors (same pencil-icon → value dialog → confirm flow as AdminScreen), and a
 * color-coded activity log of the daemon's own real log entries fetched from
 * GET /admin/logs (see TrainerLogEvent's doc comment) — not a client-side summary.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainerScreen(viewModel: TrainerViewModel, onOpenSettings: () -> Unit) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var pendingAction by remember { mutableStateOf(TrainerPendingAction.NONE) }

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollStatus()
        }
    }

    LaunchedEffect(state.actionMessage) {
        state.actionMessage?.let {
            scope.launch { snackbarHostState.showSnackbar(it) }
            viewModel.dismissActionMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Trainer") },
                actions = { SettingsAction(onOpenSettings) },
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) { Snackbar(it) } },
    ) { padding ->
        if (state.isLoading) {
            FullScreenLoading(Modifier.padding(padding))
        } else {
            LazyColumn(
                modifier = Modifier.padding(padding).fillMaxSize().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                item { StatusSection(state.status, state.statusError) }
                item { HorizontalDivider() }
                item { Text("Actions", style = MaterialTheme.typography.titleLarge) }
                item {
                    AdminActionButton(
                        label = "Checkpoint now",
                        enabled = state.status != null && state.status?.phase != "idle" && !state.actionInProgress,
                        onClick = { pendingAction = TrainerPendingAction.CHECKPOINT },
                    )
                }
                item {
                    AdminActionButton(
                        label = "Pause",
                        enabled = state.status != null && state.status?.paused == false && !state.actionInProgress,
                        onClick = { pendingAction = TrainerPendingAction.PAUSE },
                    )
                }
                item {
                    AdminActionButton(
                        label = "Resume",
                        enabled = state.status != null && state.status?.paused == true && !state.actionInProgress,
                        onClick = { pendingAction = TrainerPendingAction.RESUME },
                    )
                }
                item { HorizontalDivider() }
                item { Text("Configuration", style = MaterialTheme.typography.titleLarge) }
                item {
                    ConfigSection(
                        config = state.config,
                        error = state.configError,
                        actionInProgress = state.actionInProgress,
                        viewModel = viewModel,
                    )
                }
                item { HorizontalDivider() }
                item { Text("Activity Log", style = MaterialTheme.typography.titleLarge) }
                if (state.events.isEmpty()) {
                    item { Text("No activity yet.", style = MaterialTheme.typography.bodyMedium) }
                } else {
                    items(state.events, key = { it.id }) { event -> LogEventRow(event) }
                }
            }
        }
    }

    if (pendingAction == TrainerPendingAction.CHECKPOINT) {
        ConfirmActionDialog(
            title = "Force a checkpoint now?",
            httpCallDescription = "POST /admin/checkpoint",
            effectDescription = "Forces a checkpoint write at the next optimizer-step boundary, " +
                "independent of the auto-save cadence. Fails if no training pass is currently active.",
            confirmLabel = "Checkpoint",
            onConfirm = { pendingAction = TrainerPendingAction.NONE; viewModel.requestCheckpoint() },
            onDismiss = { pendingAction = TrainerPendingAction.NONE },
        )
    }
    if (pendingAction == TrainerPendingAction.PAUSE) {
        ConfirmActionDialog(
            title = "Pause training?",
            httpCallDescription = "POST /admin/pause",
            effectDescription = "Drains the current training pass (if any) at its next " +
                "optimizer-step boundary, checkpoints it, and releases claimed dataset files back " +
                "to pending. The service keeps running and stays reachable — this does not stop " +
                "the process.",
            confirmLabel = "Pause",
            onConfirm = { pendingAction = TrainerPendingAction.NONE; viewModel.pauseTraining() },
            onDismiss = { pendingAction = TrainerPendingAction.NONE },
        )
    }
    if (pendingAction == TrainerPendingAction.RESUME) {
        ConfirmActionDialog(
            title = "Resume training?",
            httpCallDescription = "POST /admin/resume",
            effectDescription = "Clears the pause flag and immediately checks for pending work " +
                "instead of waiting out the idle-poll interval.",
            confirmLabel = "Resume",
            onConfirm = { pendingAction = TrainerPendingAction.NONE; viewModel.resumeTraining() },
            onDismiss = { pendingAction = TrainerPendingAction.NONE },
        )
    }
}

@Composable
private fun StatusSection(status: TrainerStatusDto?, error: String?) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Status", style = MaterialTheme.typography.titleLarge)
        if (status == null) {
            Text(error ?: "Not loaded", color = MaterialTheme.colorScheme.error)
            Text(
                "The trainer admin API is opt-in (TRAINER_ADMIN_ENABLED=false by default) and " +
                    "only runs under `incremental_trainer serve`. Confirm it's enabled and " +
                    "reachable at the host/port configured in Settings.",
                style = MaterialTheme.typography.bodySmall,
            )
            return
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatusBadge(label = status.phase, isPositive = status.phase == "training")
            if (status.paused) {
                StatusBadge(label = "paused", isPositive = false)
            }
        }
        error?.let {
            Text("Last status refresh failed: $it (showing last known state)", color = MaterialTheme.colorScheme.error)
        }
        DetailRow("Model", status.model_name.ifEmpty { "(none)" })
        DetailRow("Run ID", status.run_id.ifEmpty { "(none)" })
        DetailRow("Session ID", status.session_id.ifEmpty { "(none)" })
        DetailRow("Epoch", "${status.current_epoch} / ${status.total_epochs}")
        DetailRow("Samples trained (this pass)", status.samples_trained_this_pass.toString())
        DetailRow("Last loss", status.last_loss.toString())
        DetailRow("Best loss", status.best_loss.toString())
        DetailRow("Checkpoints written", status.checkpoints_written.toString())
        DetailRow("Last checkpoint", status.last_checkpoint_path.ifEmpty { "(none)" })
        if (status.last_checkpoint_time_unix > 0) {
            DetailRow("Last checkpoint time", formatUnixSeconds(status.last_checkpoint_time_unix))
        }
    }
}

@Composable
private fun ConfigSection(
    config: TrainerAdminConfigDto?,
    error: String?,
    actionInProgress: Boolean,
    viewModel: TrainerViewModel,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (config == null) {
            Text(error ?: "Not loaded", color = MaterialTheme.colorScheme.error)
        } else {
            TrainerBoolFieldRow(
                label = "auto_save_enabled",
                jsonKey = "auto_save_enabled",
                value = config.auto_save_enabled,
                enabled = !actionInProgress,
                onSave = viewModel::updateAutoSaveEnabled,
            )
            TrainerIntFieldRow(
                label = "auto_save_every_samples",
                jsonKey = "auto_save_every_samples",
                value = config.auto_save_every_samples,
                enabled = !actionInProgress,
                onSave = viewModel::updateAutoSaveEverySamples,
            )
            TrainerIntFieldRow(
                label = "auto_save_every_minutes",
                jsonKey = "auto_save_every_minutes",
                value = config.auto_save_every_minutes,
                enabled = !actionInProgress,
                onSave = viewModel::updateAutoSaveEveryMinutes,
            )
            TrainerIntFieldRow(
                label = "max_sessions_to_keep",
                jsonKey = "max_sessions_to_keep",
                value = config.max_sessions_to_keep,
                enabled = !actionInProgress,
                onSave = viewModel::updateMaxSessionsToKeep,
            )
        }
    }
}

/** Editable Int field: pencil icon opens a value-entry dialog, then the auth-gated [ConfirmActionDialog]. */
@Composable
private fun TrainerIntFieldRow(label: String, jsonKey: String, value: Int, enabled: Boolean, onSave: (Int) -> Unit) {
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

/** Editable Boolean field — a Switch directly requests confirmation, no separate value-entry step. */
@Composable
private fun TrainerBoolFieldRow(label: String, jsonKey: String, value: Boolean, enabled: Boolean, onSave: (Boolean) -> Unit) {
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
private fun DetailRow(label: String, value: String) {
    Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
}

/**
 * Color-coded activity-log row — the "color-coded messages from the log" this screen
 * shows, one per real entry fetched from GET /admin/logs (see TrainerLogEvent's doc
 * comment). Driven off MaterialTheme.colorScheme containers (matching StatusBadge's
 * themed-container approach) rather than hardcoded hex, so it follows dark/light theme
 * and dynamic color the same way the rest of the app does. Severity levels match
 * adai::Logger::Level exactly (debug/info/warn/error) rather than a bespoke UI scale.
 */
@Composable
private fun LogEventRow(event: TrainerLogEvent) {
    val containerColor = when (event.severity) {
        TrainerLogSeverity.DEBUG -> MaterialTheme.colorScheme.surfaceVariant
        TrainerLogSeverity.INFO -> MaterialTheme.colorScheme.primaryContainer
        TrainerLogSeverity.WARN -> MaterialTheme.colorScheme.tertiaryContainer
        TrainerLogSeverity.ERROR -> MaterialTheme.colorScheme.errorContainer
    }
    val contentColor = when (event.severity) {
        TrainerLogSeverity.DEBUG -> MaterialTheme.colorScheme.onSurfaceVariant
        TrainerLogSeverity.INFO -> MaterialTheme.colorScheme.onPrimaryContainer
        TrainerLogSeverity.WARN -> MaterialTheme.colorScheme.onTertiaryContainer
        TrainerLogSeverity.ERROR -> MaterialTheme.colorScheme.onErrorContainer
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(containerColor, RoundedCornerShape(8.dp))
            .padding(10.dp),
    ) {
        Text(event.message, color = contentColor, style = MaterialTheme.typography.bodyLarge)
        Text(formatEventTime(event.timestampMillis), color = contentColor, style = MaterialTheme.typography.bodySmall)
    }
}

private fun formatEventTime(millis: Long): String =
    SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(millis))

private fun formatUnixSeconds(seconds: Long): String =
    SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(seconds * 1000))
