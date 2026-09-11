package com.adai.ops.ui.common

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.1
// @adai-reviewed: 2026-09-10


import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.adai.ops.network.dto.ModelRecordDto

private const val UNASSIGNED_LABEL = "(unassigned)"

/**
 * Read-only dropdown over the live MNS model list, used everywhere the registry
 * screens need to pick a model_name — assigning a pending file or tagging a
 * server-side fetch's per-model rotating cursor. Always offers "(unassigned)",
 * mapping to an empty model_name.
 *
 * TD-148: this used to also claim an empty model_name "clears the assignment for
 * /assign" — wrong. RegistryServer.cpp's handle_assign flatly rejects an empty
 * model_name with 400 ("non-empty model_name matching [A-Za-z0-9._-]+ required");
 * there is no way to clear an existing assignment through /assign itself (the
 * server exposes a separate /unassign endpoint for that, not currently wired up
 * in RegistryApiService.kt). "(unassigned)" is only a genuinely valid choice for
 * the two fetch dialogs (handle_fetch_gutenberg/huggingface accept an empty
 * model_name — it just buckets into a shared, unassigned rotating cursor); for
 * the assign dialog it's presented so the current "no model assigned" state has
 * a visible label, but selecting it there correctly leaves the Assign button
 * disabled (see AssignModelDialog's `enabled = modelName.isNotEmpty()`) rather
 * than sending a request the server would 400 on.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelPickerDropdown(
    models: List<ModelRecordDto>,
    selectedModelName: String,
    onModelSelected: (String) -> Unit,
    modifier: Modifier = Modifier,
    label: String = "Model",
) {
    var expanded by remember { mutableStateOf(false) }

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = modifier,
    ) {
        OutlinedTextField(
            value = selectedModelName.ifEmpty { UNASSIGNED_LABEL },
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth(),
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            DropdownMenuItem(
                text = { Text(UNASSIGNED_LABEL) },
                onClick = {
                    onModelSelected("")
                    expanded = false
                },
            )
            models.forEach { model ->
                DropdownMenuItem(
                    text = { Text(model.model_name) },
                    onClick = {
                        onModelSelected(model.model_name)
                        expanded = false
                    },
                )
            }
        }
    }
}
