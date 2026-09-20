package com.adai.ops.ui.models

// @adai-status: beta        (TD-199 review fix — passes chatbotName into LinkWorldModelDialog so its confirm preview shows the real endpoint)
// @adai-version: 0.4.1
// @adai-reviewed: 2026-09-19


import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import com.adai.ops.network.dto.LinkWorldModelRequestDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.ui.common.AdminActionButton
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.FullScreenError
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.StatusBadge
import kotlinx.coroutines.launch

private enum class PendingAction { NONE, CLEAR_LOCK, RETIRE, PROMOTE, ENTER_LINK_WORLD_MODEL, CONFIRM_LINK_WORLD_MODEL, DETACH_WORLD_MODEL }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDetailScreen(
    modelName: String,
    viewModel: ModelDetailViewModel,
    onBack: () -> Unit,
    onOpenModel: (String) -> Unit = {},
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var pendingAction by remember { mutableStateOf(PendingAction.NONE) }
    var pendingLinkRequest by remember { mutableStateOf<LinkWorldModelRequestDto?>(null) }
    var pendingLinkPreview by remember { mutableStateOf("") }

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollModel()
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
                title = { Text(modelName) },
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
            state.isLoading && state.model == null -> FullScreenLoading(Modifier.padding(padding))
            state.error != null && state.model == null ->
                FullScreenError(state.error ?: "Unknown error", Modifier.padding(padding))
            state.model != null -> ModelDetailContent(
                model = state.model!!,
                actionInProgress = state.actionInProgress,
                modifier = Modifier.padding(padding).fillMaxSize(),
                onOpenModel = onOpenModel,
                onRequestClearLock = { pendingAction = PendingAction.CLEAR_LOCK },
                onRequestRetire = { pendingAction = PendingAction.RETIRE },
                onRequestPromote = { pendingAction = PendingAction.PROMOTE },
                onRequestLinkWorldModel = {
                    viewModel.loadWorldModelCandidates()
                    pendingAction = PendingAction.ENTER_LINK_WORLD_MODEL
                },
                onRequestDetachWorldModel = { pendingAction = PendingAction.DETACH_WORLD_MODEL },
            )
        }
    }

    val model = state.model
    if (pendingAction == PendingAction.CLEAR_LOCK && model != null) {
        ConfirmActionDialog(
            title = "Clear stale training lock?",
            httpCallDescription = "PUT /models/${model.model_name}/state\n{\"state\":\"candidate\"}",
            effectDescription = "The model moves from training to candidate state, releasing the " +
                "training lock and clearing its run_id. It stays eligible for promotion, since a " +
                "crashed run may still have produced a usable checkpoint. Only do this if you've " +
                "confirmed the owning training run is dead.",
            confirmLabel = "Clear lock",
            onConfirm = {
                pendingAction = PendingAction.NONE
                viewModel.clearStaleTrainingLock()
            },
            onDismiss = { pendingAction = PendingAction.NONE },
        )
    }
    if (pendingAction == PendingAction.RETIRE && model != null) {
        ConfirmActionDialog(
            title = "Retire this candidate?",
            httpCallDescription = "PUT /models/${model.model_name}/state\n{\"state\":\"retired\"}",
            effectDescription = "The model moves to retired state and is no longer eligible for " +
                "promotion. This does not delete the model or its artifact.",
            confirmLabel = "Retire",
            onConfirm = {
                pendingAction = PendingAction.NONE
                viewModel.retireCandidate()
            },
            onDismiss = { pendingAction = PendingAction.NONE },
        )
    }
    if (pendingAction == PendingAction.PROMOTE && model != null) {
        ConfirmActionDialog(
            title = "Promote to production?",
            httpCallDescription = "PUT /roles/${model.role}/production\n{\"model_name\":\"${model.model_name}\"}",
            effectDescription = "This immediately promotes '${model.model_name}' to production for " +
                "role '${model.role}' and retires the current production model for that role, if any. " +
                "Clients resolving this role will pick up the new model on their next resolve.",
            confirmLabel = "Promote",
            onConfirm = {
                pendingAction = PendingAction.NONE
                viewModel.promoteToProduction(model.role)
            },
            onDismiss = { pendingAction = PendingAction.NONE },
        )
    }
    if (pendingAction == PendingAction.ENTER_LINK_WORLD_MODEL && model != null) {
        LinkWorldModelDialog(
            chatbotName = model.model_name,
            worldModelCandidates = state.worldModelCandidates,
            currentConnection = model.connection,
            onSubmit = { request, preview ->
                pendingLinkRequest = request
                pendingLinkPreview = preview
                pendingAction = PendingAction.CONFIRM_LINK_WORLD_MODEL
            },
            onDismiss = { pendingAction = PendingAction.NONE },
        )
    }
    if (pendingAction == PendingAction.CONFIRM_LINK_WORLD_MODEL && model != null && pendingLinkRequest != null) {
        ConfirmActionDialog(
            title = "Link world model?",
            httpCallDescription = pendingLinkPreview,
            effectDescription = "Attaches '${pendingLinkRequest!!.world_model_name}' as the world model " +
                "for '${model.model_name}'. Rejected with a d_model mismatch error if the world model's " +
                "own dimension doesn't match this chatbot's, whenever inject_every_n_layers > 0.",
            confirmLabel = "Link",
            onConfirm = {
                pendingAction = PendingAction.NONE
                viewModel.linkWorldModel(pendingLinkRequest!!)
                pendingLinkRequest = null
            },
            onDismiss = { pendingAction = PendingAction.NONE },
        )
    }
    if (pendingAction == PendingAction.DETACH_WORLD_MODEL && model != null) {
        ConfirmActionDialog(
            title = "Detach world model?",
            httpCallDescription = "POST /models/${model.model_name}/link-world-model\n{\"world_model_name\":\"\"}",
            effectDescription = "Detaches '${model.connection.world_model_name}' from '${model.model_name}'. " +
                "The world model record itself is untouched and can be re-linked later.",
            confirmLabel = "Detach",
            onConfirm = {
                pendingAction = PendingAction.NONE
                viewModel.detachWorldModel()
            },
            onDismiss = { pendingAction = PendingAction.NONE },
        )
    }
}

@Composable
private fun ModelDetailContent(
    model: ModelRecordDto,
    actionInProgress: Boolean,
    modifier: Modifier,
    onOpenModel: (String) -> Unit,
    onRequestClearLock: () -> Unit,
    onRequestRetire: () -> Unit,
    onRequestPromote: () -> Unit,
    onRequestLinkWorldModel: () -> Unit,
    onRequestDetachWorldModel: () -> Unit,
) {
    // TD-196: a new-style chatbot resolves its architecture from its linked encoder/decoder —
    // its own inline `arch` fields are unused zeros, so show the links instead of dead zeros.
    val isLinkedChatbot = model.kind == "chatbot" &&
        model.connection.encoder_name.isNotEmpty() && model.connection.decoder_name.isNotEmpty()

    LazyColumn(modifier = modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                StatusBadge(label = model.state, isPositive = model.state == "production")
                StatusBadge(label = model.kind, isPositive = false)
            }
        }
        item { DetailRow("Model ID", model.model_id) }
        item { DetailRow("Role", model.role.ifEmpty { "(none)" }) }
        item { DetailRow("Run ID", model.run_id.ifEmpty { "(none)" }) }
        item { DetailRow("Created", model.created_utc) }
        item { DetailRow("Updated", model.updated_utc) }
        item { HorizontalDivider() }
        item { Text("Architecture", style = MaterialTheme.typography.titleLarge) }
        if (isLinkedChatbot) {
            item { ClickableDetailRow("Encoder", model.connection.encoder_name) { onOpenModel(model.connection.encoder_name) } }
            item { ClickableDetailRow("Decoder", model.connection.decoder_name) { onOpenModel(model.connection.decoder_name) } }
        } else {
            item { DetailRow("d_model", model.arch.d_model.toString()) }
            item { DetailRow("num_heads", model.arch.num_heads.toString()) }
            item { DetailRow("d_ff", model.arch.d_ff.toString()) }
            item { DetailRow("encoder / decoder layers", "${model.arch.num_encoder_layers} / ${model.arch.num_decoder_layers}") }
            item { DetailRow("max_seq_length", model.arch.max_seq_length.toString()) }
        }
        if (model.kind == "world_model") {
            item { DetailRow("sigreg_lambda", model.connection.sigreg_lambda.toString()) }
            item { DetailRow("sigreg_num_sketches", model.connection.sigreg_num_sketches.toString()) }
        }
        if (model.kind == "chatbot") {
            item { HorizontalDivider() }
            item { Text("World Model", style = MaterialTheme.typography.titleLarge) }
            if (model.connection.world_model_name.isNotEmpty()) {
                item { ClickableDetailRow("Linked to", model.connection.world_model_name) { onOpenModel(model.connection.world_model_name) } }
                item { DetailRow("Inject every N layers", model.connection.world_model_inject_every_n_layers.toString()) }
                item { DetailRow("Hippocampal memory", if (model.connection.hippocampal_memory_enabled) "enabled (capacity ${model.connection.hippocampal_memory_capacity})" else "disabled") }
                item { DetailRow("Repetition alpha / decay", "${model.connection.hippocampal_repetition_alpha} / ${model.connection.hippocampal_repetition_decay}") }
                item { DetailRow("Cross-reference alpha / association decay", "${model.connection.hippocampal_cross_reference_alpha} / ${model.connection.hippocampal_association_decay}") }
            } else {
                item { DetailRow("Linked to", "(none)") }
            }
        }
        item { HorizontalDivider() }
        item { Text("Artifact", style = MaterialTheme.typography.titleLarge) }
        item { DetailRow("Host", model.artifact.host.ifEmpty { "(local)" }) }
        item { DetailRow("Path", model.artifact.path.ifEmpty { "(none)" }) }
        item { DetailRow("Format", model.artifact.format) }
        if (model.training_history.isNotEmpty()) {
            item { HorizontalDivider() }
            item { Text("Training History", style = MaterialTheme.typography.titleLarge) }
            items(model.training_history.size) { i ->
                val h = model.training_history[i]
                DetailRow(
                    "Run ${h.run_id.ifEmpty { "(unknown)" }}",
                    "epochs=${h.epochs} loss=${h.final_loss} session=${h.metrics_session_key.ifEmpty { "(none)" }}",
                )
            }
        }
        item { HorizontalDivider() }
        item { Text("Admin Actions", style = MaterialTheme.typography.titleLarge) }
        item {
            AdminActionButton(
                label = "Clear stale training lock",
                enabled = model.state == "training" && !actionInProgress,
                onClick = onRequestClearLock,
            )
        }
        item {
            AdminActionButton(
                label = "Retire candidate",
                enabled = model.state == "candidate" && !actionInProgress,
                onClick = onRequestRetire,
            )
        }
        item {
            AdminActionButton(
                label = "Promote to production",
                enabled = model.state == "candidate" && !actionInProgress,
                onClick = onRequestPromote,
            )
        }
        if (model.kind == "chatbot") {
            item {
                AdminActionButton(
                    label = "Link world model",
                    enabled = !actionInProgress,
                    onClick = onRequestLinkWorldModel,
                )
            }
            item {
                AdminActionButton(
                    label = "Detach world model",
                    enabled = model.connection.world_model_name.isNotEmpty() && !actionInProgress,
                    onClick = onRequestDetachWorldModel,
                )
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        Text(value, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
private fun ClickableDetailRow(label: String, value: String, onClick: () -> Unit) {
    Row(
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
    ) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        Text(value, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.primary)
    }
}
