package com.adai.ops.ui.models

// @adai-status: beta        (TD-196 — kind filter chips + register-model flow added)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-09-19


import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
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
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.ui.common.ConfirmActionDialog
import com.adai.ops.ui.common.EmptyDetailPlaceholder
import com.adai.ops.ui.common.FullScreenError
import com.adai.ops.ui.common.FullScreenLoading
import com.adai.ops.ui.common.SettingsAction
import com.adai.ops.ui.common.StatusBadge
import kotlinx.coroutines.launch

private val KIND_FILTERS = listOf(null, "chatbot", "encoder", "decoder", "world_model")

private fun kindFilterLabel(kind: String?) = kind?.replaceFirstChar { it.uppercase() }?.replace('_', ' ') ?: "All"

private enum class RegisterStep { NONE, PICK_KIND, ENTER_ENCODER, ENTER_DECODER, ENTER_WORLD_MODEL, ENTER_CHATBOT, CONFIRM }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelListScreen(
    viewModel: ModelListViewModel,
    onOpenModel: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    var registerStep by remember { mutableStateOf(RegisterStep.NONE) }
    var pendingRequest by remember { mutableStateOf<RegisterModelRequestDto?>(null) }
    var pendingPreview by remember { mutableStateOf("") }

    LaunchedEffect(viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
            viewModel.pollModels()
        }
    }

    LaunchedEffect(state.registerMessage) {
        state.registerMessage?.let {
            scope.launch { snackbarHostState.showSnackbar(it) }
            viewModel.dismissRegisterMessage()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Models") },
                actions = {
                    IconButton(onClick = { registerStep = RegisterStep.PICK_KIND }) {
                        Icon(Icons.Filled.Add, contentDescription = "Register model")
                    }
                    SettingsAction(onOpenSettings)
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) { Snackbar(it) } },
    ) { padding ->
        Column(modifier = Modifier.padding(padding).fillMaxSize()) {
            KindFilterRow(selectedKind = state.selectedKind, onKindSelected = viewModel::setKindFilter)
            Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
                when {
                    state.isLoading && state.models.isEmpty() -> FullScreenLoading()
                    state.error != null && state.models.isEmpty() -> FullScreenError(state.error ?: "Unknown error")
                    state.models.isEmpty() -> EmptyDetailPlaceholder("No models registered")
                    else -> LazyColumn(modifier = Modifier.fillMaxSize()) {
                        items(state.models, key = { it.model_id }) { model ->
                            ModelRow(model, onClick = { onOpenModel(model.model_name) })
                            HorizontalDivider()
                        }
                    }
                }
            }
        }
    }

    if (registerStep == RegisterStep.PICK_KIND) {
        RegisterKindPickerDialog(
            onKindChosen = { kind ->
                registerStep = when (kind) {
                    "encoder" -> RegisterStep.ENTER_ENCODER
                    "decoder" -> RegisterStep.ENTER_DECODER
                    "world_model" -> RegisterStep.ENTER_WORLD_MODEL
                    else -> {
                        // TD-199 (review fix): fetch encoder/decoder candidates via their own
                        // dedicated, unfiltered-by-kind-chip calls — RegisterChatbotDialog used to
                        // be handed state.models directly, which is always restricted to whatever
                        // kind filter chip happens to be selected on this screen.
                        viewModel.loadChatbotLinkCandidates()
                        RegisterStep.ENTER_CHATBOT
                    }
                }
            },
            onDismiss = { registerStep = RegisterStep.NONE },
        )
    }

    val onEntered: (RegisterModelRequestDto, String) -> Unit = { request, preview ->
        pendingRequest = request
        pendingPreview = preview
        registerStep = RegisterStep.CONFIRM
    }
    when (registerStep) {
        RegisterStep.ENTER_ENCODER -> RegisterEncoderDialog(onSubmit = onEntered, onDismiss = { registerStep = RegisterStep.NONE })
        RegisterStep.ENTER_DECODER -> RegisterDecoderDialog(onSubmit = onEntered, onDismiss = { registerStep = RegisterStep.NONE })
        RegisterStep.ENTER_WORLD_MODEL -> RegisterWorldModelDialog(onSubmit = onEntered, onDismiss = { registerStep = RegisterStep.NONE })
        RegisterStep.ENTER_CHATBOT -> RegisterChatbotDialog(
            encoders = state.encoderCandidates,
            decoders = state.decoderCandidates,
            onSubmit = onEntered,
            onDismiss = { registerStep = RegisterStep.NONE },
        )
        else -> Unit
    }

    if (registerStep == RegisterStep.CONFIRM && pendingRequest != null) {
        ConfirmActionDialog(
            title = "Register '${pendingRequest!!.model_name}'?",
            httpCallDescription = pendingPreview,
            effectDescription = "Creates a new ${pendingRequest!!.kind} record in MNS. Architecture " +
                "is immutable after registration — this cannot be undone from the app (mns_cli delete " +
                "only works on an initializing or retired record).",
            confirmLabel = "Register",
            onConfirm = {
                viewModel.registerModel(pendingRequest!!)
                registerStep = RegisterStep.NONE
                pendingRequest = null
            },
            onDismiss = { registerStep = RegisterStep.NONE },
        )
    }
}

@Composable
private fun KindFilterRow(selectedKind: String?, onKindSelected: (String?) -> Unit) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        items(KIND_FILTERS) { kind ->
            FilterChip(
                selected = selectedKind == kind,
                onClick = { onKindSelected(kind) },
                label = { Text(kindFilterLabel(kind)) },
            )
        }
    }
}

@Composable
private fun RegisterKindPickerDialog(onKindChosen: (String) -> Unit, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Register a model") },
        text = {
            Column {
                // Distinct "Register <kind>" labels — not bare kind names — so they don't collide
                // with the list screen's own kind-filter chips (which sit behind this dialog).
                listOf("encoder" to "Register Encoder", "decoder" to "Register Decoder", "world_model" to "Register World Model", "chatbot" to "Register Chatbot").forEach { (kind, label) ->
                    TextButton(onClick = { onKindChosen(kind) }, modifier = Modifier.fillMaxWidth()) {
                        Text(label, modifier = Modifier.fillMaxWidth())
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun ModelRow(model: ModelRecordDto, onClick: () -> Unit) {
    ListItem(
        headlineContent = { Text(model.model_name) },
        supportingContent = { Text("Role: ${model.role.ifEmpty { "(none)" }} · Kind: ${model.kind}") },
        trailingContent = { StatusBadge(label = model.state, isPositive = model.state == "production") },
        modifier = Modifier.clickable(onClick = onClick),
    )
}
