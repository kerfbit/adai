package com.adai.ops.ui.common

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
 * mapping to an empty model_name, since that's a valid choice server-side (buckets
 * into a shared cursor for fetches; simply clears the assignment for /assign).
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
