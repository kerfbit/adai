package com.adai.ops.ui.metrics

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.adai.ops.network.dto.CurrentMetricsDto
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.polling.PollerPhase
import com.adai.ops.ui.common.AdminActionButton
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.FullScreenError
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.MetricHistoryChart
import com.adai.ops.ui.common.StatusBadge
import java.util.Locale
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SessionDetailScreen(
    sessionKey: String,
    viewModel: SessionDetailViewModel,
    onBack: () -> Unit,
    onEvicted: () -> Unit,
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var showEndSessionConfirm by remember { mutableStateOf(false) }

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.poll()
        }
    }

    LaunchedEffect(viewModel) {
        viewModel.events.collect { event ->
            when (event) {
                SessionDetailEvent.SessionEvicted -> onEvicted()
            }
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
                title = { Text(sessionKey) },
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
            state.isLoading && state.status == null -> FullScreenLoading(Modifier.padding(padding))
            state.error != null && state.status == null ->
                FullScreenError(state.error ?: "Unknown error", Modifier.padding(padding))
            else -> SessionDetailContent(
                state = state,
                modifier = Modifier.padding(padding).fillMaxSize(),
                onRequestEndSession = { showEndSessionConfirm = true },
            )
        }
    }

    if (showEndSessionConfirm) {
        ConfirmActionDialog(
            title = "End this session?",
            httpCallDescription = "POST /api/sessions/$sessionKey/end",
            effectDescription = "Marks the session as complete and writes its final summary to " +
                "disk. Use this when the trainer crashed or hung and never called /end itself — " +
                "the session's last known data stays visible, but it will no longer be treated as " +
                "live/training.",
            confirmLabel = "End session",
            onConfirm = {
                showEndSessionConfirm = false
                viewModel.endSession()
            },
            onDismiss = { showEndSessionConfirm = false },
        )
    }
}

@Composable
private fun SessionDetailContent(
    state: SessionDetailUiState,
    modifier: Modifier,
    onRequestEndSession: () -> Unit,
) {
    val status = state.status
    val current = state.current
    val epochs = state.epochs

    LazyColumn(
        modifier = modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        item { PollerStatusRow(state) }

        status?.let {
            item { ProgressSection(it) }
            item { HorizontalDivider() }
        }

        current?.let {
            item { MetricsSection(it) }
            item { HorizontalDivider() }
            item { AdvancedDiagnosticsSection(it) }
            item { HorizontalDivider() }
            item { GenerationQualitySection(it) }
        }

        if (epochs != null && epochs.epoch_losses.size >= 2) {
            item {
                MetricHistoryChart(
                    title = "Loss History (per epoch)",
                    trainValues = epochs.epoch_losses,
                    validationValues = epochs.epoch_validation_losses,
                )
            }
        }

        if (epochs != null && epochs.epoch_perplexities.size >= 2) {
            item {
                MetricHistoryChart(
                    title = "Perplexity History (per epoch)",
                    trainValues = epochs.epoch_perplexities,
                    validationValues = epochs.epoch_validation_perplexities,
                )
            }
        }

        item { HorizontalDivider() }
        item { Text("Admin Actions", style = MaterialTheme.typography.titleLarge) }
        item {
            AdminActionButton(
                label = "End session",
                enabled = !state.actionInProgress,
                onClick = onRequestEndSession,
            )
        }
    }
}

@Composable
private fun PollerStatusRow(state: SessionDetailUiState) {
    val label = when (state.pollerPhase) {
        PollerPhase.LIVE -> "Live · polling every ${state.pollIntervalMs}ms"
        PollerPhase.RECONNECTING -> "Reconnecting (${state.retryCount})"
        PollerPhase.OFFLINE -> "Offline — retrying (${state.retryCount})"
        PollerPhase.EVICTED -> "Session ended"
    }
    StatusBadge(label = label, isPositive = state.pollerPhase == PollerPhase.LIVE)
}

@Composable
private fun ProgressSection(status: SessionStatusDto) {
    Text("Progress", style = MaterialTheme.typography.titleLarge)
    LinearProgressIndicator(
        progress = { (status.progress_percent / 100.0).toFloat().coerceIn(0f, 1f) },
        modifier = Modifier.fillMaxWidth(),
    )
    MetricRow("Epoch", "${status.current_epoch}/${status.total_epochs}")
    MetricRow("Sample", "${status.current_sample}/${status.total_samples}")
    MetricRow("Samples/sec", String.format(Locale.US, "%.2f", status.samples_per_second))
    MetricRow("ETA (s)", String.format(Locale.US, "%.0f", status.estimated_time_remaining_seconds))
    MetricRow(
        "Effective training",
        if (status.effective_is_training) "yes" else if (status.is_stale) "no — stale" else "no",
    )
}

@Composable
private fun MetricsSection(current: CurrentMetricsDto) {
    Text("Training Metrics", style = MaterialTheme.typography.titleLarge)
    MetricRow("Loss", String.format(Locale.US, "%.4f", current.current_loss))
    MetricRow("Validation loss", String.format(Locale.US, "%.4f", current.current_validation_loss))
    MetricRow("Running loss (avg)", String.format(Locale.US, "%.4f", current.running_loss))
    MetricRow("Best validation loss", String.format(Locale.US, "%.4f", current.best_validation_loss) + " (epoch ${current.best_epoch})")
    MetricRow("Learning rate", String.format(Locale.US, "%.6f", current.current_learning_rate))
    MetricRow("Perplexity", String.format(Locale.US, "%.2f", current.current_perplexity))
    MetricRow("Gradient norm", String.format(Locale.US, "%.4f", current.current_gradient_norm))
    if (current.current_validation_accuracy >= 0.0) {
        MetricRow("Validation accuracy", String.format(Locale.US, "%.4f", current.current_validation_accuracy))
    }
}

@Composable
private fun AdvancedDiagnosticsSection(current: CurrentMetricsDto) {
    Text("Advanced Diagnostics", style = MaterialTheme.typography.titleLarge)
    MetricRow("Gradient variance", String.format(Locale.US, "%.6f", current.gradient_variance))
    MetricRow("Compute time ratio", String.format(Locale.US, "%.4f", current.compute_time_ratio))
    MetricRow("Weight update ratio", String.format(Locale.US, "%.3e", current.weight_update_ratio))
    MetricRow("Activation saturation", String.format(Locale.US, "%.4f", current.activation_saturation_ratio))
}

@Composable
private fun GenerationQualitySection(current: CurrentMetricsDto) {
    if (current.current_bleu4 < 0.0) {
        Text("Generation Quality", style = MaterialTheme.typography.titleLarge)
        Text("Disabled (ENABLE_GENERATION_QUALITY_METRICS=false)", style = MaterialTheme.typography.bodyLarge)
        return
    }
    Text("Generation Quality", style = MaterialTheme.typography.titleLarge)
    MetricRow("BLEU-4", String.format(Locale.US, "%.4f", current.current_bleu4))
    MetricRow("ROUGE-1", String.format(Locale.US, "%.4f", current.current_rouge1))
    MetricRow("ROUGE-2", String.format(Locale.US, "%.4f", current.current_rouge2))
    MetricRow("ROUGE-L", String.format(Locale.US, "%.4f", current.current_rougeL))
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
}
