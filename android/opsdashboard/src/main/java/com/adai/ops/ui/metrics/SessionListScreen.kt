package com.adai.ops.ui.metrics

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


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
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.adai.ops.network.dto.SessionSummaryDto
import com.adai.ops.ui.common.EmptyDetailPlaceholder
import com.adai.ops.ui.common.FullScreenError
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.SettingsAction
import com.adai.ops.ui.common.StatusBadge
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SessionListScreen(
    viewModel: SessionListViewModel,
    onOpenSession: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollSessions()
        }
    }

    val visibleSessions = if (state.showIdle) state.sessions else state.sessions.filter { it.is_training }
    val hiddenIdleCount = state.sessions.size - state.sessions.count { it.is_training }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Training Sessions") },
                actions = {
                    TextButton(onClick = viewModel::toggleShowIdle) {
                        Text(if (state.showIdle) "Hide idle" else "Show idle")
                    }
                    SettingsAction(onOpenSettings)
                },
            )
        },
    ) { padding ->
        when {
            state.isLoading && state.sessions.isEmpty() -> FullScreenLoading(Modifier.padding(padding))
            state.error != null && state.sessions.isEmpty() ->
                FullScreenError(state.error ?: "Unknown error", Modifier.padding(padding))
            state.sessions.isEmpty() -> EmptyDetailPlaceholder("No sessions found", Modifier.padding(padding))
            visibleSessions.isEmpty() -> EmptyDetailPlaceholder(
                "No active sessions — $hiddenIdleCount idle session(s) hidden. Tap 'Show idle' to view them.",
                Modifier.padding(padding),
            )
            else -> LazyColumn(modifier = Modifier.padding(padding).fillMaxSize()) {
                items(visibleSessions, key = { it.key }) { session ->
                    SessionRow(session, onClick = { onOpenSession(session.key) })
                    HorizontalDivider()
                }
            }
        }
    }
}

@Composable
private fun SessionRow(session: SessionSummaryDto, onClick: () -> Unit) {
    ListItem(
        headlineContent = { Text(session.key) },
        supportingContent = {
            Text(
                "Epoch ${session.current_epoch}/${session.total_epochs} · loss " +
                    String.format(Locale.US, "%.4f", session.current_loss),
            )
        },
        trailingContent = {
            StatusBadge(
                label = if (session.is_training) "Training" else "Idle",
                isPositive = session.is_training,
            )
        },
        modifier = Modifier.clickable(onClick = onClick),
    )
}
