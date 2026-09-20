package com.adai.ops.ui.models

// @adai-status: experimental        (TD-199 review fix — RegisterChatbotDialog takes dedicated encoder/decoder candidate lists, not the list screen's own kind-filtered models)
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
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedTextField
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
import com.adai.ops.network.dto.ArchDto
import com.adai.ops.network.dto.ConnectionDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.ui.common.ModelPickerDropdown

/**
 * TD-196: one data-entry AlertDialog per registerable kind, following FetchGutenbergDialog's
 * shape (ui/registry/GroupDetailScreen.kt) — a Column of OutlinedTextFields + confirm/dismiss
 * buttons. Unlike that dialog (which fires directly), each of these hands its built
 * [RegisterModelRequestDto] and a human-readable HTTP-call preview to [onSubmit] instead of
 * calling the repository itself — the caller (ModelListScreen) always routes that through
 * ConfirmActionDialog's biometric/PIN gate before actually registering, matching every other
 * mutating action in this app.
 */

@Composable
fun RegisterEncoderDialog(onSubmit: (RegisterModelRequestDto, String) -> Unit, onDismiss: () -> Unit) {
    RegisterEncoderOrDecoderDialog(kind = "encoder", title = "Register Encoder", onSubmit = onSubmit, onDismiss = onDismiss)
}

@Composable
fun RegisterDecoderDialog(onSubmit: (RegisterModelRequestDto, String) -> Unit, onDismiss: () -> Unit) {
    RegisterEncoderOrDecoderDialog(kind = "decoder", title = "Register Decoder", onSubmit = onSubmit, onDismiss = onDismiss)
}

@Composable
private fun RegisterEncoderOrDecoderDialog(
    kind: String,
    title: String,
    onSubmit: (RegisterModelRequestDto, String) -> Unit,
    onDismiss: () -> Unit,
) {
    var modelName by remember { mutableStateOf("") }
    var role by remember { mutableStateOf("") }
    var runGroup by remember { mutableStateOf("") }
    var dModelText by remember { mutableStateOf("") }
    var numHeadsText by remember { mutableStateOf("") }
    var dFfText by remember { mutableStateOf("") }
    var numLayersText by remember { mutableStateOf("") }
    var maxSeqText by remember { mutableStateOf("") }

    val dModel = dModelText.toLongOrNull()
    val numHeads = numHeadsText.toLongOrNull()
    val dFf = dFfText.toLongOrNull()
    val numLayers = numLayersText.toLongOrNull()
    val maxSeq = maxSeqText.toLongOrNull()
    val canSubmit = modelName.isNotBlank() && dModel != null && numHeads != null && dFf != null &&
        numLayers != null && maxSeq != null

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                LabeledField("Model name", modelName) { modelName = it }
                LabeledField("Role (optional)", role) { role = it }
                LabeledField("Run group (optional)", runGroup) { runGroup = it }
                NumberField("d_model", dModelText) { dModelText = it }
                NumberField("num_heads", numHeadsText) { numHeadsText = it }
                NumberField("d_ff", dFfText) { dFfText = it }
                NumberField("num_layers", numLayersText) { numLayersText = it }
                NumberField("max_seq_length", maxSeqText) { maxSeqText = it }
            }
        },
        confirmButton = {
            Button(
                enabled = canSubmit,
                onClick = {
                    val arch = if (kind == "decoder") {
                        ArchDto(d_model = dModel!!, num_heads = numHeads!!, d_ff = dFf!!, num_encoder_layers = 0, num_decoder_layers = numLayers!!, max_seq_length = maxSeq!!)
                    } else {
                        ArchDto(d_model = dModel!!, num_heads = numHeads!!, d_ff = dFf!!, num_encoder_layers = numLayers!!, num_decoder_layers = 0, max_seq_length = maxSeq!!)
                    }
                    val request = RegisterModelRequestDto(model_name = modelName, role = role, kind = kind, run_group = runGroup, arch = arch)
                    val preview = "POST /models\n{\"model_name\":\"$modelName\",\"kind\":\"$kind\",\"arch\":{\"d_model\":$dModel," +
                        "\"num_heads\":$numHeads,\"d_ff\":$dFf,\"num_encoder_layers\":${arch.num_encoder_layers}," +
                        "\"num_decoder_layers\":${arch.num_decoder_layers},\"max_seq_length\":$maxSeq}}"
                    onSubmit(request, preview)
                },
            ) { Text("Register") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
fun RegisterWorldModelDialog(onSubmit: (RegisterModelRequestDto, String) -> Unit, onDismiss: () -> Unit) {
    var modelName by remember { mutableStateOf("") }
    var role by remember { mutableStateOf("") }
    var runGroup by remember { mutableStateOf("") }
    var dModelText by remember { mutableStateOf("") }
    var numHeadsText by remember { mutableStateOf("") }
    var dFfText by remember { mutableStateOf("") }
    var numLayersText by remember { mutableStateOf("") }
    var maxSeqText by remember { mutableStateOf("") }
    var sigregLambdaText by remember { mutableStateOf("1.0") }
    var sigregSketchesText by remember { mutableStateOf("64") }

    val dModel = dModelText.toLongOrNull()
    val numHeads = numHeadsText.toLongOrNull()
    val dFf = dFfText.toLongOrNull()
    val numLayers = numLayersText.toLongOrNull()
    val maxSeq = maxSeqText.toLongOrNull()
    val sigregLambda = sigregLambdaText.toFloatOrNull()
    val sigregSketches = sigregSketchesText.toLongOrNull()
    val canSubmit = modelName.isNotBlank() && dModel != null && numHeads != null && dFf != null &&
        numLayers != null && maxSeq != null && sigregLambda != null && sigregSketches != null

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Register World Model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                LabeledField("Model name", modelName) { modelName = it }
                LabeledField("Role (optional)", role) { role = it }
                LabeledField("Run group (optional)", runGroup) { runGroup = it }
                NumberField("d_model", dModelText) { dModelText = it }
                NumberField("num_heads", numHeadsText) { numHeadsText = it }
                NumberField("d_ff", dFfText) { dFfText = it }
                NumberField("num_layers (own encoder layers)", numLayersText) { numLayersText = it }
                NumberField("max_seq_length", maxSeqText) { maxSeqText = it }
                NumberField("sigreg_lambda", sigregLambdaText) { sigregLambdaText = it }
                NumberField("sigreg_num_sketches", sigregSketchesText) { sigregSketchesText = it }
            }
        },
        confirmButton = {
            Button(
                enabled = canSubmit,
                onClick = {
                    val arch = ArchDto(
                        d_model = dModel!!, num_heads = numHeads!!, d_ff = dFf!!,
                        num_encoder_layers = numLayers!!, num_decoder_layers = 0, max_seq_length = maxSeq!!,
                    )
                    val connection = ConnectionDto(sigreg_lambda = sigregLambda!!, sigreg_num_sketches = sigregSketches!!)
                    val request = RegisterModelRequestDto(
                        model_name = modelName, role = role, kind = "world_model",
                        run_group = runGroup, arch = arch, connection = connection,
                    )
                    val preview = "POST /models\n{\"model_name\":\"$modelName\",\"kind\":\"world_model\"," +
                        "\"arch\":{\"d_model\":$dModel,\"num_heads\":$numHeads,\"d_ff\":$dFf," +
                        "\"num_encoder_layers\":$numLayers,\"num_decoder_layers\":0,\"max_seq_length\":$maxSeq}," +
                        "\"connection\":{\"sigreg_lambda\":$sigregLambda,\"sigreg_num_sketches\":$sigregSketches}}"
                    onSubmit(request, preview)
                },
            ) { Text("Register") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisterChatbotDialog(
    encoders: List<ModelRecordDto>,
    decoders: List<ModelRecordDto>,
    onSubmit: (RegisterModelRequestDto, String) -> Unit,
    onDismiss: () -> Unit,
) {
    var modelName by remember { mutableStateOf("") }
    var role by remember { mutableStateOf("") }
    var runGroup by remember { mutableStateOf("") }
    var linked by remember { mutableStateOf(true) }

    // Linked mode
    var encoderName by remember { mutableStateOf("") }
    var decoderName by remember { mutableStateOf("") }

    // Legacy mode
    var dModelText by remember { mutableStateOf("") }
    var numHeadsText by remember { mutableStateOf("") }
    var dFfText by remember { mutableStateOf("") }
    var encLayersText by remember { mutableStateOf("") }
    var decLayersText by remember { mutableStateOf("") }
    var maxSeqText by remember { mutableStateOf("") }

    val canSubmit = modelName.isNotBlank() && if (linked) {
        encoderName.isNotEmpty() && decoderName.isNotEmpty()
    } else {
        listOf(dModelText, numHeadsText, dFfText, encLayersText, decLayersText, maxSeqText).all { it.toLongOrNull() != null }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Register Chatbot") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                LabeledField("Model name", modelName) { modelName = it }
                LabeledField("Role (optional)", role) { role = it }
                LabeledField("Run group (optional)", runGroup) { runGroup = it }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = linked, onClick = { linked = true }, label = { Text("Linked") })
                    FilterChip(selected = !linked, onClick = { linked = false }, label = { Text("Legacy") })
                }
                if (linked) {
                    ModelPickerDropdown(
                        models = encoders,
                        selectedModelName = encoderName,
                        onModelSelected = { encoderName = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = "Encoder",
                    )
                    ModelPickerDropdown(
                        models = decoders,
                        selectedModelName = decoderName,
                        onModelSelected = { decoderName = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = "Decoder",
                    )
                } else {
                    NumberField("d_model", dModelText) { dModelText = it }
                    NumberField("num_heads", numHeadsText) { numHeadsText = it }
                    NumberField("d_ff", dFfText) { dFfText = it }
                    NumberField("num_encoder_layers", encLayersText) { encLayersText = it }
                    NumberField("num_decoder_layers", decLayersText) { decLayersText = it }
                    NumberField("max_seq_length", maxSeqText) { maxSeqText = it }
                }
            }
        },
        confirmButton = {
            Button(
                enabled = canSubmit,
                onClick = {
                    val (request, preview) = if (linked) {
                        val connection = ConnectionDto(encoder_name = encoderName, decoder_name = decoderName)
                        val req = RegisterModelRequestDto(
                            model_name = modelName, role = role, kind = "chatbot",
                            run_group = runGroup, connection = connection,
                        )
                        val prev = "POST /models\n{\"model_name\":\"$modelName\",\"kind\":\"chatbot\"," +
                            "\"connection\":{\"encoder_name\":\"$encoderName\",\"decoder_name\":\"$decoderName\"}}"
                        req to prev
                    } else {
                        val arch = ArchDto(
                            d_model = dModelText.toLong(), num_heads = numHeadsText.toLong(), d_ff = dFfText.toLong(),
                            num_encoder_layers = encLayersText.toLong(), num_decoder_layers = decLayersText.toLong(),
                            max_seq_length = maxSeqText.toLong(),
                        )
                        val req = RegisterModelRequestDto(
                            model_name = modelName, role = role, kind = "chatbot", run_group = runGroup, arch = arch,
                        )
                        val prev = "POST /models\n{\"model_name\":\"$modelName\",\"kind\":\"chatbot\"," +
                            "\"arch\":{\"d_model\":${arch.d_model},\"num_heads\":${arch.num_heads},\"d_ff\":${arch.d_ff}," +
                            "\"num_encoder_layers\":${arch.num_encoder_layers},\"num_decoder_layers\":${arch.num_decoder_layers}," +
                            "\"max_seq_length\":${arch.max_seq_length}}}"
                        req to prev
                    }
                    onSubmit(request, preview)
                },
            ) { Text("Register") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun LabeledField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(value = value, onValueChange = onValueChange, label = { Text(label) }, modifier = Modifier.fillMaxWidth())
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
