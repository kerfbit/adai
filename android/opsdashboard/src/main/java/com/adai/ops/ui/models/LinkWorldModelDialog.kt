package com.adai.ops.ui.models

// @adai-status: experimental        (TD-199 review fix — preview now interpolates the real chatbot name instead of a literal "{name}" placeholder)
// @adai-version: 0.1.1
// @adai-reviewed: 2026-09-19


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.adai.ops.network.dto.ConnectionDto
import com.adai.ops.network.dto.LinkWorldModelRequestDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.ui.common.ModelPickerDropdown

/**
 * TD-196: data-entry dialog for POST /models/{name}/link-world-model, pre-filled from
 * [currentConnection] so re-linking (changing injection cadence or hippocampal tuning on an
 * already-linked chatbot) is really "edit in place". Fields beyond world_model_name are only
 * meaningful once a world model is actually selected, matching ConnectionDto's own defaults for
 * anything left untouched. Like the register dialogs, this hands its built request off to the
 * caller instead of calling the repository directly — ModelDetailScreen routes it through
 * ConfirmActionDialog's biometric/PIN gate before it actually fires.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LinkWorldModelDialog(
    chatbotName: String,
    worldModelCandidates: List<ModelRecordDto>,
    currentConnection: ConnectionDto,
    onSubmit: (LinkWorldModelRequestDto, String) -> Unit,
    onDismiss: () -> Unit,
) {
    var worldModelName by remember { mutableStateOf(currentConnection.world_model_name) }
    var injectText by remember { mutableStateOf(currentConnection.world_model_inject_every_n_layers.toString()) }
    var hippocampalEnabled by remember { mutableStateOf(currentConnection.hippocampal_memory_enabled) }
    var capacityText by remember { mutableStateOf(currentConnection.hippocampal_memory_capacity.toString()) }
    var repetitionAlphaText by remember { mutableStateOf(currentConnection.hippocampal_repetition_alpha.toString()) }
    var repetitionDecayText by remember { mutableStateOf(currentConnection.hippocampal_repetition_decay.toString()) }
    var crossReferenceAlphaText by remember { mutableStateOf(currentConnection.hippocampal_cross_reference_alpha.toString()) }
    var associationDecayText by remember { mutableStateOf(currentConnection.hippocampal_association_decay.toString()) }

    val inject = injectText.toLongOrNull()
    val capacity = capacityText.toLongOrNull()
    val repetitionAlpha = repetitionAlphaText.toFloatOrNull()
    val repetitionDecay = repetitionDecayText.toFloatOrNull()
    val crossReferenceAlpha = crossReferenceAlphaText.toFloatOrNull()
    val associationDecay = associationDecayText.toFloatOrNull()
    val canSubmit = worldModelName.isNotEmpty() && inject != null && capacity != null &&
        repetitionAlpha != null && repetitionDecay != null && crossReferenceAlpha != null && associationDecay != null

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Link world model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                ModelPickerDropdown(
                    models = worldModelCandidates,
                    selectedModelName = worldModelName,
                    onModelSelected = { worldModelName = it },
                    modifier = Modifier.fillMaxWidth(),
                    label = "World model",
                )
                NumberField("inject_every_n_layers", injectText) { injectText = it }
                Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                    Text("Hippocampal memory", style = MaterialTheme.typography.bodyLarge)
                    Switch(checked = hippocampalEnabled, onCheckedChange = { hippocampalEnabled = it })
                }
                NumberField("hippocampal_memory_capacity", capacityText) { capacityText = it }
                NumberField("hippocampal_repetition_alpha", repetitionAlphaText) { repetitionAlphaText = it }
                NumberField("hippocampal_repetition_decay", repetitionDecayText) { repetitionDecayText = it }
                NumberField("hippocampal_cross_reference_alpha", crossReferenceAlphaText) { crossReferenceAlphaText = it }
                NumberField("hippocampal_association_decay", associationDecayText) { associationDecayText = it }
            }
        },
        confirmButton = {
            Button(
                enabled = canSubmit,
                onClick = {
                    val request = LinkWorldModelRequestDto(
                        world_model_name = worldModelName,
                        world_model_inject_every_n_layers = inject!!,
                        hippocampal_memory_enabled = hippocampalEnabled,
                        hippocampal_memory_capacity = capacity!!,
                        hippocampal_repetition_alpha = repetitionAlpha!!,
                        hippocampal_repetition_decay = repetitionDecay!!,
                        hippocampal_cross_reference_alpha = crossReferenceAlpha!!,
                        hippocampal_association_decay = associationDecay!!,
                    )
                    val preview = "POST /models/$chatbotName/link-world-model\n{\"world_model_name\":\"$worldModelName\"," +
                        "\"world_model_inject_every_n_layers\":$inject,\"hippocampal_memory_enabled\":$hippocampalEnabled," +
                        "\"hippocampal_memory_capacity\":$capacity,\"hippocampal_repetition_alpha\":$repetitionAlpha," +
                        "\"hippocampal_repetition_decay\":$repetitionDecay,\"hippocampal_cross_reference_alpha\":$crossReferenceAlpha," +
                        "\"hippocampal_association_decay\":$associationDecay}"
                    onSubmit(request, preview)
                },
            ) { Text("Continue") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun NumberField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        isError = value.isNotEmpty() && value.toDoubleOrNull() == null,
        modifier = Modifier.fillMaxWidth(),
    )
}
