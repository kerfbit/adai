package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
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
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.ui.common.AdminActionButton
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.ModelPickerDropdown
import com.adai.ops.ui.common.StatusBadge
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GroupDetailScreen(group: String, viewModel: GroupDetailViewModel, onBack: () -> Unit) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var runPendingRelease by remember { mutableStateOf<String?>(null) }
    var showGutenbergDialog by remember { mutableStateOf(false) }
    var showHuggingfaceDialog by remember { mutableStateOf(false) }
    var assignDialogEntry by remember { mutableStateOf<QueueEntryDto?>(null) }

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollGroup()
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
                title = { Text(group) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) { Snackbar(it) } },
    ) { padding ->
        when {
            // Only the very first load blocks the whole screen — once we have
            // anything to show, the Download section must stay reachable even
            // for a group with an empty queue, since that's exactly when an
            // operator needs it most (there's nothing pending yet to work with).
            state.isLoading && state.queueEntries.isEmpty() && state.runs.isEmpty() ->
                FullScreenLoading(Modifier.padding(padding))
            else -> GroupDetailContent(
                state = state,
                modifier = Modifier.padding(padding).fillMaxSize(),
                onRequestRelease = { runId -> runPendingRelease = runId },
                onRequestGutenbergFetch = { showGutenbergDialog = true },
                onRequestHuggingfaceFetch = { showHuggingfaceDialog = true },
                onRequestAssign = { entry -> assignDialogEntry = entry },
            )
        }
    }

    runPendingRelease?.let { runId ->
        val files = state.runs[runId].orEmpty()
        ConfirmActionDialog(
            title = "Force-release run '$runId'?",
            httpCallDescription = "POST /registry/$group/release\n{\"run_id\":\"\",\"files\":[...${files.size} file(s)...]}",
            effectDescription = "Using an empty run_id bypasses the owner check, so these " +
                "${files.size} file(s) return to the unassigned pool even if a live trainer " +
                "still holds them. Only do this if you've confirmed run '$runId' is dead.",
            confirmLabel = "Force release",
            onConfirm = {
                runPendingRelease = null
                viewModel.forceReleaseRun(runId, files)
            },
            onDismiss = { runPendingRelease = null },
        )
    }

    if (showGutenbergDialog) {
        FetchGutenbergDialog(
            models = state.models,
            onFetch = { bookId, numPairs, modelName ->
                showGutenbergDialog = false
                viewModel.fetchGutenberg(bookId, numPairs, modelName)
            },
            onDismiss = { showGutenbergDialog = false },
        )
    }

    if (showHuggingfaceDialog) {
        FetchHuggingfaceDialog(
            models = state.models,
            onFetch = { datasetId, numPairs, split, inputField, outputField, modelName ->
                showHuggingfaceDialog = false
                viewModel.fetchHuggingface(datasetId, numPairs, split, inputField, outputField, modelName)
            },
            onDismiss = { showHuggingfaceDialog = false },
        )
    }

    assignDialogEntry?.let { entry ->
        AssignModelDialog(
            path = entry.path,
            currentModelName = entry.model_name,
            models = state.models,
            onAssign = { modelName ->
                assignDialogEntry = null
                viewModel.assignModel(entry.path, modelName)
            },
            onDismiss = { assignDialogEntry = null },
        )
    }
}

@Composable
private fun GroupDetailContent(
    state: GroupDetailUiState,
    modifier: Modifier,
    onRequestRelease: (String) -> Unit,
    onRequestGutenbergFetch: () -> Unit,
    onRequestHuggingfaceFetch: () -> Unit,
    onRequestAssign: (QueueEntryDto) -> Unit,
) {
    LazyColumn(modifier = modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text("Download", style = MaterialTheme.typography.titleLarge) }
        item {
            Text(
                "Triggers registry_server to fetch and cache the source itself, then enqueue " +
                    "a rotating slice as a new pending file — nothing is downloaded to this phone.",
                style = MaterialTheme.typography.bodyLarge,
            )
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = onRequestGutenbergFetch) { Text("Fetch Gutenberg book") }
                Button(onClick = onRequestHuggingfaceFetch) { Text("Fetch HuggingFace dataset") }
            }
        }

        item { HorizontalDivider() }
        item { Text("Claimed by run", style = MaterialTheme.typography.titleLarge) }
        if (state.runs.isEmpty()) {
            item { Text("No files currently claimed by a run.") }
        } else {
            items(state.runs.entries.toList(), key = { "run-" + it.key }) { (runId, files) ->
                RunRow(runId, files.size, actionInProgress = state.actionInProgress) {
                    onRequestRelease(runId)
                }
            }
        }

        item { HorizontalDivider() }
        item { Text("Pending queue", style = MaterialTheme.typography.titleLarge) }
        if (state.queueEntries.isEmpty()) {
            item { Text("No pending files in this group.") }
        } else {
            item {
                Text(
                    "No timestamp is available for claims — registry_server does not track " +
                        "when a file was claimed, only by whom.",
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
            items(state.queueEntries, key = { "queue-" + it.path }) { entry ->
                QueueRow(entry, actionInProgress = state.actionInProgress, onAssign = { onRequestAssign(entry) })
                HorizontalDivider()
            }
        }

        item { HorizontalDivider() }
        item { Text("Trained files", style = MaterialTheme.typography.titleLarge) }
        if (state.registryEntries.isEmpty()) {
            item { Text("No trained files in this group yet.") }
        } else {
            items(state.registryEntries, key = { "registry-" + it.data_file }) { entry ->
                RegistryRow(entry)
                HorizontalDivider()
            }
        }
    }
}

@Composable
private fun RunRow(runId: String, fileCount: Int, actionInProgress: Boolean, onRelease: () -> Unit) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text("$runId — $fileCount file(s)", style = MaterialTheme.typography.bodyLarge)
        AdminActionButton(label = "Release", enabled = !actionInProgress, onClick = onRelease)
    }
}

@Composable
private fun QueueRow(entry: QueueEntryDto, actionInProgress: Boolean, onAssign: () -> Unit) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column {
            Text(entry.path, style = MaterialTheme.typography.bodyLarge)
            Text(
                "run: ${entry.run_id.ifEmpty { "unassigned" }} · " +
                    "model: ${entry.model_name.ifEmpty { "unassigned" }}",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                "${formatEntryCount(entry.num_entries)} · ${formatFileSize(entry.size_bytes)} · " +
                    "${entry.source.ifEmpty { "unknown source" }} · added ${formatAddedUtc(entry.added_utc)}",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        IconButton(onClick = onAssign, enabled = !actionInProgress) {
            Icon(Icons.Filled.Edit, contentDescription = "Assign model")
        }
    }
}

@Composable
private fun RegistryRow(entry: RegistryEntryDto) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column {
            Text(entry.data_file, style = MaterialTheme.typography.bodyLarge)
            Text(
                "${entry.num_samples} samples · ${entry.source.ifEmpty { "unknown source" }} · " +
                    "added ${formatAddedUtc(entry.added_utc)}",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        StatusBadge(label = if (entry.trained) "trained" else "pending", isPositive = entry.trained)
    }
}

// Human file size, e.g. "12.3 KB". 0/negative means "not locally readable by
// the registry at creation time" (see QueueEntryDto's doc comment) — not "empty file".
private fun formatFileSize(bytes: Long): String {
    if (bytes <= 0) return "size unknown"
    val units = arrayOf("B", "KB", "MB", "GB")
    var value = bytes.toDouble()
    var unitIndex = 0
    while (value >= 1024 && unitIndex < units.lastIndex) {
        value /= 1024
        unitIndex++
    }
    return if (unitIndex == 0) "$bytes B" else "%.1f %s".format(value, units[unitIndex])
}

private fun formatEntryCount(numEntries: Int): String =
    if (numEntries < 0) "entries unknown" else "$numEntries ${if (numEntries == 1) "entry" else "entries"}"

private val addedUtcFormatter: DateTimeFormatter =
    DateTimeFormatter.ofPattern("MMM d, yyyy HH:mm 'UTC'").withZone(ZoneOffset.UTC)

private fun formatAddedUtc(addedUtc: String): String {
    if (addedUtc.isEmpty())
        return "unknown"
    return runCatching { addedUtcFormatter.format(Instant.parse(addedUtc)) }.getOrDefault(addedUtc)
}

@Composable
private fun FetchGutenbergDialog(
    models: List<ModelRecordDto>,
    onFetch: (bookId: Int, numPairs: Int, modelName: String) -> Unit,
    onDismiss: () -> Unit,
) {
    var bookIdText by remember { mutableStateOf("") }
    var numPairsText by remember { mutableStateOf("500") }
    var modelName by remember { mutableStateOf("") }
    val bookId = bookIdText.toIntOrNull()
    val numPairs = numPairsText.toIntOrNull()

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Fetch Gutenberg book") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = bookIdText,
                    onValueChange = { bookIdText = it },
                    label = { Text("Gutenberg book ID") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    isError = bookIdText.isNotEmpty() && bookId == null,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = numPairsText,
                    onValueChange = { numPairsText = it },
                    label = { Text("Pairs to serve this call") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    isError = numPairsText.isNotEmpty() && numPairs == null,
                    modifier = Modifier.fillMaxWidth(),
                )
                ModelPickerDropdown(
                    models = models,
                    selectedModelName = modelName,
                    onModelSelected = { modelName = it },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onFetch(bookId!!, numPairs ?: 500, modelName) },
                enabled = bookId != null && bookId > 0,
            ) { Text("Fetch") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun FetchHuggingfaceDialog(
    models: List<ModelRecordDto>,
    onFetch: (
        datasetId: String,
        numPairs: Int,
        split: String,
        inputField: String,
        outputField: String,
        modelName: String,
    ) -> Unit,
    onDismiss: () -> Unit,
) {
    var datasetId by remember { mutableStateOf("") }
    var numPairsText by remember { mutableStateOf("500") }
    var split by remember { mutableStateOf("train") }
    var inputField by remember { mutableStateOf("") }
    var outputField by remember { mutableStateOf("") }
    var modelName by remember { mutableStateOf("") }
    val numPairs = numPairsText.toIntOrNull()

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Fetch HuggingFace dataset") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = datasetId,
                    onValueChange = { datasetId = it },
                    label = { Text("Dataset ID (e.g. tatsu-lab/alpaca)") },
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = numPairsText,
                    onValueChange = { numPairsText = it },
                    label = { Text("Pairs to serve this call") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    isError = numPairsText.isNotEmpty() && numPairs == null,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = split,
                    onValueChange = { split = it },
                    label = { Text("Split") },
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = inputField,
                    onValueChange = { inputField = it },
                    label = { Text("Input field (blank = auto-detect)") },
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = outputField,
                    onValueChange = { outputField = it },
                    label = { Text("Output field (blank = auto-detect)") },
                    modifier = Modifier.fillMaxWidth(),
                )
                ModelPickerDropdown(
                    models = models,
                    selectedModelName = modelName,
                    onModelSelected = { modelName = it },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onFetch(datasetId, numPairs ?: 500, split.ifEmpty { "train" }, inputField, outputField, modelName)
                },
                enabled = datasetId.isNotBlank(),
            ) { Text("Fetch") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun AssignModelDialog(
    path: String,
    currentModelName: String,
    models: List<ModelRecordDto>,
    onAssign: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var modelName by remember { mutableStateOf(currentModelName) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Assign model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(path, style = MaterialTheme.typography.bodyLarge)
                ModelPickerDropdown(
                    models = models,
                    selectedModelName = modelName,
                    onModelSelected = { modelName = it },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            Button(onClick = { onAssign(modelName) }, enabled = modelName.isNotEmpty()) {
                Text("Assign")
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
