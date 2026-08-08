package com.adai.ops.ui.registry

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ListItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
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
                "No registry groups configured. Add group names in Settings — " +
                    "registry_server has no way to list them for you.",
                Modifier.padding(padding),
            )
            else -> LazyColumn(modifier = Modifier.padding(padding).fillMaxSize()) {
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
            Text(group.error?.let { "Error: $it" } ?: "${group.pendingCount ?: 0} pending file(s)")
        },
        modifier = Modifier.clickable(onClick = onClick),
    )
}
