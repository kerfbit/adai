package com.adai.ops.ui.registry

// @adai-status: beta        (TD-048 — GroupListScreenTest.kt runs on an emulator since 2026-11-21)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-21


import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.adai.ops.ui.common.EmptyDetailPlaceholder
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.SettingsAction

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GroupListScreen(
    viewModel: GroupListViewModel,
    onOpenGroup: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollGroups()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Dataset Registry") },
                actions = { SettingsAction(onOpenSettings) },
            )
        },
    ) { padding ->
        when {
            state.isLoading && state.groups.isEmpty() -> FullScreenLoading(Modifier.padding(padding))
            state.groups.isEmpty() -> EmptyDetailPlaceholder(
                state.discoveryError?.let {
                    "Couldn't list the models' registry groups from MNS ($it). Check the Models " +
                        "server in Settings, or add group names there."
                } ?: ("No registry groups: no model in MNS has a run group, and none are added " +
                    "in Settings."),
                Modifier.padding(padding),
            )
            else -> LazyColumn(modifier = Modifier.padding(padding).fillMaxSize()) {
                state.discoveryError?.let { error ->
                    item {
                        Text(
                            "Showing only the groups in Settings: couldn't list models from MNS ($error).",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.padding(16.dp),
                        )
                    }
                }
                items(state.groups, key = { it.name }) { group ->
                    GroupRow(group, onClick = { onOpenGroup(group.name) })
                    HorizontalDivider()
                }
            }
        }
    }
}

@Composable
private fun GroupRow(group: GroupSummary, onClick: () -> Unit) {
    ListItem(
        headlineContent = { Text(group.name) },
        supportingContent = {
            Column {
                Text(group.error?.let { "Error: $it" } ?: "${group.pendingCount ?: 0} pending file(s)")
                // TD-580: which models train from this group (none for a group only in Settings).
                if (group.models.isNotEmpty()) {
                    Text("Models: ${group.models.joinToString(", ")}", style = MaterialTheme.typography.bodySmall)
                }
            }
        },
        modifier = Modifier.clickable(onClick = onClick),
    )
}
