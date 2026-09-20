package com.adai.ops.ui.registry

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-20

// TD-205: new data-entry dialogs for the dataset management segment expansion — unassign has
// no data to collect (goes straight to ConfirmActionDialog from GroupDetailScreen), so it has
// no dialog here. Each dialog below only collects/validates fields; the actual mutating call
// happens after ConfirmActionDialog's own auth gate, from GroupDetailScreen's onSubmit callback
// — same split FetchGutenbergDialog/FetchHuggingfaceDialog/AssignModelDialog already use.

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp

/** Delete a pending or trained entry. [onSubmit] hands back whether to also unlink the file. */
@Composable
fun DeleteEntryDialog(
    path: String,
    onSubmit: (deleteFiles: Boolean) -> Unit,
    onDismiss: () -> Unit,
) {
    var deleteFiles by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Delete entry") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(path, style = MaterialTheme.typography.bodyLarge)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(selected = deleteFiles, onClick = { deleteFiles = !deleteFiles }),
                ) {
                    RadioButton(selected = deleteFiles, onClick = { deleteFiles = !deleteFiles })
                    Text("Also unlink the underlying file (server-owned fetches/uploads only)")
                }
            }
        },
        confirmButton = { Button(onClick = { onSubmit(deleteFiles) }) { Text("Delete") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** Queue an already-existing (server-readable) path, without transferring any new bytes. */
@Composable
fun ManualAddDialog(onSubmit: (path: String) -> Unit, onDismiss: () -> Unit) {
    var path by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Queue existing path") },
        text = {
            OutlinedTextField(
                value = path,
                onValueChange = { path = it },
                label = { Text("Path (must be readable by registry_server)") },
                modifier = Modifier.fillMaxWidth(),
            )
        },
        confirmButton = {
            Button(onClick = { onSubmit(path) }, enabled = path.isNotBlank()) { Text("Queue") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** Upload a local file's bytes directly from the device. */
@Composable
fun UploadFileDialog(onSubmit: (filename: String, bytes: ByteArray) -> Unit, onDismiss: () -> Unit) {
    val context = LocalContext.current
    var filename by remember { mutableStateOf("") }
    var pickedBytes by remember { mutableStateOf<ByteArray?>(null) }
    var pickedUri by remember { mutableStateOf<Uri?>(null) }

    val pickFile = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        pickedUri = uri
        pickedBytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
        if (filename.isBlank()) {
            filename = queryDisplayName(context, uri) ?: uri.lastPathSegment.orEmpty()
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Upload file") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = { pickFile.launch("*/*") }, modifier = Modifier.fillMaxWidth()) {
                    Text(if (pickedUri == null) "Choose file…" else "Change file…")
                }
                pickedUri?.let { Text(it.lastPathSegment.orEmpty(), style = MaterialTheme.typography.bodyMedium) }
                OutlinedTextField(
                    value = filename,
                    onValueChange = { filename = it },
                    label = { Text("Filename (bare name, no path separators)") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { pickedBytes?.let { onSubmit(filename, it) } },
                enabled = pickedBytes != null && filename.isNotBlank() && !filename.contains('/'),
            ) { Text("Upload") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

private fun queryDisplayName(context: android.content.Context, uri: Uri): String? {
    val projection = arrayOf(android.provider.OpenableColumns.DISPLAY_NAME)
    return context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
        val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
        if (idx >= 0 && cursor.moveToFirst()) cursor.getString(idx) else null
    }
}

/**
 * Move a legacy pending entry into another kind's own sub-pool. Only meaningful while viewing
 * the legacy pool — [currentKind] is passed only to exclude it from the picker, though in
 * practice this dialog is only ever opened when [currentKind] is null (legacy).
 */
@Composable
fun MigrateToKindDialog(currentKind: String?, onKindChosen: (String) -> Unit, onDismiss: () -> Unit) {
    val kinds = listOf("encoder", "decoder", "world_model", "chatbot").filter { it != currentKind }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Migrate to kind") },
        text = {
            Column {
                kinds.forEach { kind ->
                    TextButton(onClick = { onKindChosen(kind) }, modifier = Modifier.fillMaxWidth()) {
                        Text(kindLabel(kind), modifier = Modifier.fillMaxWidth())
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

private fun kindLabel(kind: String) = kind.replaceFirstChar { it.uppercase() }.replace('_', ' ')

/**
 * Split [path]'s [totalPairs] pairs into either N near-equal parts or explicit ranges, each
 * queued as its own independently-assignable/deletable pending entry. [totalPairs] comes from
 * the originating [com.adai.ops.network.dto.QueueEntryDto.num_entries] — this dialog can't
 * count pairs itself (that needs local file access the app doesn't have for a path that lives
 * only on the registry_server's own storage), so it's only offered when that count is already
 * known and positive.
 */
@Composable
fun CreateSegmentsDialog(
    path: String,
    totalPairs: Int,
    onSubmit: (ranges: List<Pair<Int, Int>>) -> Unit,
    onDismiss: () -> Unit,
) {
    var useEqualSplit by remember { mutableStateOf(true) }
    var splitCountText by remember { mutableStateOf("2") }
    var rangesText by remember { mutableStateOf("") }
    val splitCount = splitCountText.toIntOrNull()

    val ranges: List<Pair<Int, Int>> = if (useEqualSplit) {
        val n = splitCount
        if (n == null || n <= 0 || n > totalPairs) {
            emptyList()
        } else {
            val base = totalPairs / n
            val extra = totalPairs % n
            var start = 0
            (0 until n).mapNotNull { i ->
                val count = base + if (i < extra) 1 else 0
                if (count <= 0) return@mapNotNull null
                (start to count).also { start += count }
            }
        }
    } else {
        parseRangesText(rangesText, totalPairs)
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Create segments") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("$path — $totalPairs pair(s)", style = MaterialTheme.typography.bodyLarge)
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(selected = useEqualSplit, onClick = { useEqualSplit = true }),
                ) {
                    RadioButton(selected = useEqualSplit, onClick = { useEqualSplit = true })
                    Text("Split into N equal parts")
                }
                if (useEqualSplit) {
                    OutlinedTextField(
                        value = splitCountText,
                        onValueChange = { splitCountText = it },
                        label = { Text("N") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        isError = splitCountText.isNotEmpty() && (splitCount == null || splitCount <= 0),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(selected = !useEqualSplit, onClick = { useEqualSplit = false }),
                ) {
                    RadioButton(selected = !useEqualSplit, onClick = { useEqualSplit = false })
                    Text("Custom ranges")
                }
                if (!useEqualSplit) {
                    OutlinedTextField(
                        value = rangesText,
                        onValueChange = { rangesText = it },
                        label = { Text("0-based inclusive ranges, e.g. 0-99,100-199") },
                        isError = rangesText.isNotEmpty() && ranges.isEmpty(),
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                if (ranges.isNotEmpty()) {
                    Text(
                        "Preview: " + ranges.joinToString(", ") { (s, c) -> "[$s-${s + c - 1}]" },
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        },
        confirmButton = {
            Button(onClick = { onSubmit(ranges) }, enabled = ranges.isNotEmpty()) { Text("Create") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** Parses "0-99,100-199" into [(0,100),(100,100)] — 0-based inclusive ranges, A<=B, A>=0. */
private fun parseRangesText(text: String, totalPairs: Int): List<Pair<Int, Int>> {
    if (text.isBlank()) return emptyList()
    val result = mutableListOf<Pair<Int, Int>>()
    for (token in text.split(',')) {
        val trimmed = token.trim()
        if (trimmed.isEmpty()) continue
        val dash = trimmed.indexOf('-')
        if (dash <= 0) return emptyList()
        val a = trimmed.substring(0, dash).toIntOrNull() ?: return emptyList()
        val b = trimmed.substring(dash + 1).toIntOrNull() ?: return emptyList()
        if (a < 0 || b < a || b >= totalPairs) return emptyList()
        result.add(a to (b - a + 1))
    }
    return result
}
