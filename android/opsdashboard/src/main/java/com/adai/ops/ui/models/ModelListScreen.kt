package com.adai.ops.ui.models

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-10


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
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.ui.common.EmptyDetailPlaceholder
import com.adai.ops.ui.common.FullScreenError
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.SettingsAction
import com.adai.ops.ui.common.StatusBadge

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelListScreen(
    viewModel: ModelListViewModel,
    onOpenModel: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollModels()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Models") },
                actions = { SettingsAction(onOpenSettings) },
            )
        },
    ) { padding ->
        when {
            state.isLoading && state.models.isEmpty() -> FullScreenLoading(Modifier.padding(padding))
            state.error != null && state.models.isEmpty() ->
                FullScreenError(state.error ?: "Unknown error", Modifier.padding(padding))
            state.models.isEmpty() -> EmptyDetailPlaceholder("No models registered", Modifier.padding(padding))
            else -> LazyColumn(modifier = Modifier.padding(padding).fillMaxSize()) {
                items(state.models, key = { it.model_id }) { model ->
                    ModelRow(model, onClick = { onOpenModel(model.model_name) })
                    HorizontalDivider()
                }
            }
        }
    }
}

@Composable
private fun ModelRow(model: ModelRecordDto, onClick: () -> Unit) {
    ListItem(
        headlineContent = { Text(model.model_name) },
        supportingContent = { Text("Role: ${model.role.ifEmpty { "(none)" }}") },
        trailingContent = { StatusBadge(label = model.state, isPositive = model.state == "production") },
        modifier = Modifier.clickable(onClick = onClick),
    )
}
