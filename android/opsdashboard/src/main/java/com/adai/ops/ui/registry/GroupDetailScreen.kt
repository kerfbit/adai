package com.adai.ops.ui.registry

// @adai-status: beta        (TD-205 — full dataset management: kind filter chips, pool-health overview, unassign/delete/manual-add/upload/migrate/segment actions, all mutating actions now ConfirmActionDialog-gated; real-device run still unverified, see TD-048)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-20


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
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
import com.adai.ops.network.dto.SegmentTargetDto
import com.adai.ops.ui.common.AdminActionButton
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.ModelPickerDropdown
import com.adai.ops.ui.common.StatusBadge
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlinx.coroutines.launch

/** One pending mutating action awaiting ConfirmActionDialog's own auth gate. */
private sealed interface PendingAction {
    data object None : PendingAction
    data class Release(val runId: String, val files: List<String>) : PendingAction
    data class Assign(val entry: QueueEntryDto, val modelName: String) : PendingAction
    data class Unassign(val entry: QueueEntryDto) : PendingAction
    data class Delete(val path: String, val entry: QueueEntryDto?, val deleteFiles: Boolean) : PendingAction
    data class ManualAdd(val path: String) : PendingAction
    data class Upload(val filename: String, val bytes: ByteArray) : PendingAction
    data class Migrate(val entry: QueueEntryDto, val destKind: String) : PendingAction
    data class FetchGutenberg(val bookId: Int, val numPairs: Int, val modelName: String) : PendingAction
    data class FetchHuggingface(
        val datasetId: String,
        val numPairs: Int,
        val split: String,
        val inputField: String,
        val outputField: String,
        val modelName: String,
    ) : PendingAction
    data class Segment(val path: String, val ranges: List<Pair<Int, Int>>) : PendingAction
}

private val KIND_FILTERS = listOf(null, "chatbot", "encoder", "decoder", "world_model")
private fun kindLabel(kind: String?) = kind?.replaceFirstChar { it.uppercase() }?.replace('_', ' ') ?: "Legacy"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GroupDetailScreen(group: String, viewModel: GroupDetailViewModel, onBack: () -> Unit) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var pending by remember { mutableStateOf<PendingAction>(PendingAction.None) }
    var showGutenbergDialog by remember { mutableStateOf(false) }
    var showHuggingfaceDialog by remember { mutableStateOf(false) }
    var assignDialogEntry by remember { mutableStateOf<QueueEntryDto?>(null) }
    var deleteDialogEntry by remember { mutableStateOf<QueueEntryDto?>(null) }
    var deleteDialogTrained by remember { mutableStateOf<RegistryEntryDto?>(null) }
    var showManualAddDialog by remember { mutableStateOf(false) }
    var showUploadDialog by remember { mutableStateOf(false) }
    var migrateDialogEntry by remember { mutableStateOf<QueueEntryDto?>(null) }
    var segmentDialogEntry by remember { mutableStateOf<QueueEntryDto?>(null) }
    var showAddMenu by remember { mutableStateOf(false) }

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
                actions = {
                    IconButton(onClick = { showAddMenu = true }) {
                        Icon(Icons.Filled.Add, contentDescription = "Add data")
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
                onKindSelected = viewModel::setKindFilter,
                onRequestRelease = { runId -> pending = PendingAction.Release(runId, state.runs[runId].orEmpty()) },
                onRequestGutenbergFetch = { showGutenbergDialog = true },
                onRequestHuggingfaceFetch = { showHuggingfaceDialog = true },
                onRequestAssign = { entry -> assignDialogEntry = entry },
                onRequestUnassign = { entry -> pending = PendingAction.Unassign(entry) },
                onRequestDeletePending = { entry -> deleteDialogEntry = entry },
                onRequestDeleteTrained = { entry -> deleteDialogTrained = entry },
                onRequestMigrate = { entry -> migrateDialogEntry = entry },
                onRequestSegment = { entry -> segmentDialogEntry = entry },
            )
        }
    }

    if (showAddMenu) {
        AlertDialog(
            onDismissRequest = { showAddMenu = false },
            title = { Text("Add data") },
            text = {
                Column {
                    TextButton(
                        onClick = { showAddMenu = false; showManualAddDialog = true },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Queue existing path", modifier = Modifier.fillMaxWidth()) }
                    TextButton(
                        onClick = { showAddMenu = false; showUploadDialog = true },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Upload file", modifier = Modifier.fillMaxWidth()) }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { showAddMenu = false }) { Text("Cancel") } },
        )
    }

    if (showGutenbergDialog) {
        FetchGutenbergDialog(
            models = state.models,
            onFetch = { bookId, numPairs, modelName ->
                showGutenbergDialog = false
                pending = PendingAction.FetchGutenberg(bookId, numPairs, modelName)
            },
            onDismiss = { showGutenbergDialog = false },
        )
    }

    if (showHuggingfaceDialog) {
        FetchHuggingfaceDialog(
            models = state.models,
            onFetch = { datasetId, numPairs, split, inputField, outputField, modelName ->
                showHuggingfaceDialog = false
                pending = PendingAction.FetchHuggingface(datasetId, numPairs, split, inputField, outputField, modelName)
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
                pending = PendingAction.Assign(entry, modelName)
            },
            onDismiss = { assignDialogEntry = null },
        )
    }

    if (showManualAddDialog) {
        ManualAddDialog(
            onSubmit = { path -> showManualAddDialog = false; pending = PendingAction.ManualAdd(path) },
            onDismiss = { showManualAddDialog = false },
        )
    }

    if (showUploadDialog) {
        UploadFileDialog(
            onSubmit = { filename, bytes ->
                showUploadDialog = false
                pending = PendingAction.Upload(filename, bytes)
            },
            onDismiss = { showUploadDialog = false },
        )
    }

    deleteDialogEntry?.let { entry ->
        DeleteEntryDialog(
            path = entry.path,
            onSubmit = { deleteFiles ->
                deleteDialogEntry = null
                pending = PendingAction.Delete(entry.path, entry, deleteFiles)
            },
            onDismiss = { deleteDialogEntry = null },
        )
    }

    deleteDialogTrained?.let { entry ->
        DeleteEntryDialog(
            path = entry.data_file,
            onSubmit = { deleteFiles ->
                deleteDialogTrained = null
                pending = PendingAction.Delete(entry.data_file, null, deleteFiles)
            },
            onDismiss = { deleteDialogTrained = null },
        )
    }

    migrateDialogEntry?.let { entry ->
        MigrateToKindDialog(
            currentKind = state.selectedKind,
            onKindChosen = { kind ->
                migrateDialogEntry = null
                pending = PendingAction.Migrate(entry, kind)
            },
            onDismiss = { migrateDialogEntry = null },
        )
    }

    segmentDialogEntry?.let { entry ->
        CreateSegmentsDialog(
            path = entry.path,
            totalPairs = entry.num_entries,
            onSubmit = { ranges ->
                segmentDialogEntry = null
                pending = PendingAction.Segment(entry.path, ranges)
            },
            onDismiss = { segmentDialogEntry = null },
        )
    }

    when (val p = pending) {
        is PendingAction.None -> Unit
        is PendingAction.Release -> ConfirmActionDialog(
            title = "Force-release run '${p.runId}'?",
            httpCallDescription = "POST /registry/$group/release\n{\"run_id\":\"\",\"files\":[...${p.files.size} file(s)...]}",
            effectDescription = "Using an empty run_id bypasses the owner check, so these " +
                "${p.files.size} file(s) return to the unassigned pool even if a live trainer " +
                "still holds them. Only do this if you've confirmed run '${p.runId}' is dead.",
            confirmLabel = "Force release",
            onConfirm = { pending = PendingAction.None; viewModel.forceReleaseRun(p.runId, p.files) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.Assign -> ConfirmActionDialog(
            title = "Assign model?",
            httpCallDescription = "POST /registry/$group/assign\n{\"model_name\":\"${p.modelName}\",\"paths\":[\"${p.entry.path}\"]}",
            effectDescription = "Assigns '${p.entry.path}' to model '${p.modelName}'.",
            confirmLabel = "Assign",
            onConfirm = { pending = PendingAction.None; viewModel.assignModel(p.entry, p.modelName) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.Unassign -> ConfirmActionDialog(
            title = "Unassign model?",
            httpCallDescription = "POST /registry/$group/unassign\n{\"paths\":[\"${p.entry.path}\"],\"force\":true}",
            effectDescription = "Clears '${p.entry.path}' back to unassigned (was '${p.entry.model_name}').",
            confirmLabel = "Confirm unassign",
            onConfirm = { pending = PendingAction.None; viewModel.unassignModel(p.entry) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.Delete -> ConfirmActionDialog(
            title = "Delete entry?",
            httpCallDescription = "POST /registry/$group/delete\n{\"paths\":[\"${p.path}\"],\"force\":true," +
                "\"delete_files\":${p.deleteFiles}}",
            effectDescription = "Permanently purges '${p.path}' from the pending queue and trained registry." +
                if (p.deleteFiles) " The underlying file is also unlinked if the server owns it." else "",
            confirmLabel = "Delete",
            onConfirm = {
                pending = PendingAction.None
                val segment = p.entry?.takeIf { it.isSegment }
                    ?.let { SegmentTargetDto(it.path, it.segment_start, it.segment_count) }
                viewModel.deleteEntry(p.path, segment, p.deleteFiles)
            },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.ManualAdd -> ConfirmActionDialog(
            title = "Queue existing path?",
            httpCallDescription = "POST /registry/$group/pending/add\n{\"path\":\"${p.path}\"}",
            effectDescription = "Queues '${p.path}' as pending, assuming registry_server can already read it.",
            confirmLabel = "Queue",
            onConfirm = { pending = PendingAction.None; viewModel.manualAdd(p.path) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.Upload -> ConfirmActionDialog(
            title = "Upload file?",
            httpCallDescription = "POST /registry/$group/upload?filename=${p.filename}\n(${p.bytes.size} byte(s))",
            effectDescription = "Uploads '${p.filename}' to the registry server's own storage and queues it.",
            confirmLabel = "Upload",
            onConfirm = { pending = PendingAction.None; viewModel.upload(p.filename, p.bytes) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.Migrate -> ConfirmActionDialog(
            title = "Migrate to '${p.destKind}'?",
            httpCallDescription = "POST /registry/$group/${p.destKind}/pending/add\n{\"path\":\"${p.entry.path}\"}\n" +
                "POST /registry/$group/delete\n{\"paths\":[\"${p.entry.path}\"],\"force\":true}",
            effectDescription = "Moves '${p.entry.path}' from the legacy pool into '${p.destKind}'s own sub-pool" +
                if (p.entry.model_name.isNotEmpty()) ", preserving its assignment to '${p.entry.model_name}'." else ".",
            confirmLabel = "Confirm migrate",
            onConfirm = { pending = PendingAction.None; viewModel.migrateToKind(p.entry, p.destKind) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.Segment -> ConfirmActionDialog(
            title = "Create ${p.ranges.size} segment(s)?",
            httpCallDescription = "POST /registry/$group/pending/add ×${p.ranges.size}\n" +
                p.ranges.joinToString("\n") { (s, c) -> "{\"path\":\"${p.path}\",\"segment_start\":$s,\"segment_count\":$c}" },
            effectDescription = "Splits '${p.path}' into ${p.ranges.size} independently-manageable pending entries.",
            confirmLabel = "Create",
            onConfirm = { pending = PendingAction.None; viewModel.createSegments(p.path, p.ranges) },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.FetchGutenberg -> ConfirmActionDialog(
            title = "Fetch Gutenberg book?",
            httpCallDescription = "POST /registry/$group/fetch/gutenberg\n" +
                "{\"book_id\":${p.bookId},\"num_pairs\":${p.numPairs},\"model_name\":\"${p.modelName}\"}",
            effectDescription = "Downloads/caches book #${p.bookId} server-side and enqueues a slice as pending.",
            confirmLabel = "Fetch",
            onConfirm = {
                pending = PendingAction.None
                viewModel.fetchGutenberg(p.bookId, p.numPairs, p.modelName)
            },
            onDismiss = { pending = PendingAction.None },
        )
        is PendingAction.FetchHuggingface -> ConfirmActionDialog(
            title = "Fetch HuggingFace dataset?",
            httpCallDescription = "POST /registry/$group/fetch/huggingface\n" +
                "{\"dataset_id\":\"${p.datasetId}\",\"num_pairs\":${p.numPairs},\"split\":\"${p.split}\"}",
            effectDescription = "Downloads/caches '${p.datasetId}' server-side and enqueues a slice as pending.",
            confirmLabel = "Fetch",
            onConfirm = {
                pending = PendingAction.None
                viewModel.fetchHuggingface(p.datasetId, p.numPairs, p.split, p.inputField, p.outputField, p.modelName)
            },
            onDismiss = { pending = PendingAction.None },
        )
    }
}

@Composable
private fun GroupDetailContent(
    state: GroupDetailUiState,
    modifier: Modifier,
    onKindSelected: (String?) -> Unit,
    onRequestRelease: (String) -> Unit,
    onRequestGutenbergFetch: () -> Unit,
    onRequestHuggingfaceFetch: () -> Unit,
    onRequestAssign: (QueueEntryDto) -> Unit,
    onRequestUnassign: (QueueEntryDto) -> Unit,
    onRequestDeletePending: (QueueEntryDto) -> Unit,
    onRequestDeleteTrained: (RegistryEntryDto) -> Unit,
    onRequestMigrate: (QueueEntryDto) -> Unit,
    onRequestSegment: (QueueEntryDto) -> Unit,
) {
    LazyColumn(modifier = modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // TD-128: state.error was computed in the ViewModel but never displayed
        // anywhere in this screen — a failed refresh silently left the queue/runs/
        // registry sections showing stale data with no indication anything was
        // wrong. Matches the pattern TrainerScreen's StatusSection already uses.
        state.error?.let { error ->
            item {
                Text(
                    "Last refresh failed: $error (showing last known state)",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

        // TD-205: pool-health "planning" overview — every kind's pending/trained counts at a
        // glance, independent of the filter chips below. Tapping a row applies that kind.
        item { Text("Pool health", style = MaterialTheme.typography.titleLarge) }
        items(state.poolHealth, key = { "health-" + (it.kind ?: "legacy") }) { health ->
            PoolHealthRow(health, selected = health.kind == state.selectedKind, onClick = { onKindSelected(health.kind) })
        }

        item { HorizontalDivider() }
        item {
            KindFilterRow(selectedKind = state.selectedKind, onKindSelected = onKindSelected)
        }

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
            item { Text("No pending files in this pool.") }
        } else {
            item {
                Text(
                    "No timestamp is available for claims — registry_server does not track " +
                        "when a file was claimed, only by whom.",
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
            items(state.queueEntries, key = { "queue-" + it.path + "-" + it.segment_start }) { entry ->
                QueueRow(
                    entry,
                    actionInProgress = state.actionInProgress,
                    showMigrate = state.selectedKind == null,
                    onAssign = { onRequestAssign(entry) },
                    onUnassign = { onRequestUnassign(entry) },
                    onDelete = { onRequestDeletePending(entry) },
                    onMigrate = { onRequestMigrate(entry) },
                    onSegment = { onRequestSegment(entry) },
                )
                HorizontalDivider()
            }
        }

        item { HorizontalDivider() }
        item { Text("Trained files", style = MaterialTheme.typography.titleLarge) }
        if (state.registryEntries.isEmpty()) {
            item { Text("No trained files in this pool yet.") }
        } else {
            items(state.registryEntries, key = { "registry-" + it.data_file }) { entry ->
                RegistryRow(entry, onDelete = { onRequestDeleteTrained(entry) })
                HorizontalDivider()
            }
        }
    }
}

@Composable
private fun PoolHealthRow(health: KindPoolHealth, selected: Boolean, onClick: () -> Unit) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth(),
    ) {
        TextButton(onClick = onClick) {
            Text(
                kindLabel(health.kind),
                style = MaterialTheme.typography.bodyLarge,
                color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
            )
        }
        Text(
            "${health.pendingCount} pending · ${health.trainedCount} trained · ${health.totalSamples} samples",
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}

@Composable
private fun KindFilterRow(selectedKind: String?, onKindSelected: (String?) -> Unit) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        items(KIND_FILTERS) { kind ->
            FilterChip(
                selected = selectedKind == kind,
                onClick = { onKindSelected(kind) },
                label = { Text(kindLabel(kind)) },
            )
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
private fun QueueRow(
    entry: QueueEntryDto,
    actionInProgress: Boolean,
    showMigrate: Boolean,
    onAssign: () -> Unit,
    onUnassign: () -> Unit,
    onDelete: () -> Unit,
    onMigrate: () -> Unit,
    onSegment: () -> Unit,
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
            Column {
                Text(entry.path, style = MaterialTheme.typography.bodyLarge)
                if (entry.isSegment) {
                    Text(
                        "segment [${entry.segment_start}-${entry.segment_start + entry.segment_count - 1}]",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
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
            Row {
                IconButton(onClick = onAssign, enabled = !actionInProgress) {
                    Icon(Icons.Filled.Edit, contentDescription = "Assign model")
                }
                IconButton(onClick = onDelete, enabled = !actionInProgress) {
                    Icon(Icons.Filled.Delete, contentDescription = "Delete")
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (entry.model_name.isNotEmpty()) {
                TextButton(onClick = onUnassign, enabled = !actionInProgress) { Text("Unassign") }
            }
            if (showMigrate) {
                TextButton(onClick = onMigrate, enabled = !actionInProgress) { Text("Migrate") }
            }
            if (!entry.isSegment && entry.num_entries > 1) {
                TextButton(onClick = onSegment, enabled = !actionInProgress) { Text("Segment") }
            }
        }
    }
}

@Composable
private fun RegistryRow(entry: RegistryEntryDto, onDelete: () -> Unit) {
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
        Row {
            StatusBadge(label = if (entry.trained) "trained" else "pending", isPositive = entry.trained)
            IconButton(onClick = onDelete) {
                Icon(Icons.Filled.Delete, contentDescription = "Delete")
            }
        }
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
