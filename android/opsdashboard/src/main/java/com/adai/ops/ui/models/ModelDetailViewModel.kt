package com.adai.ops.ui.models

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.1
// @adai-reviewed: 2026-09-10


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ModelDetailUiState(
    val model: ModelRecordDto? = null,
    val isLoading: Boolean = true,
    val error: String? = null,
    val actionInProgress: Boolean = false,
    val actionMessage: String? = null,
)

/**
 * Drives Model Detail plus the MNS admin actions: clearing a stale training lock
 * (PUT state=candidate), retiring a candidate (PUT state=retired), and promoting a
 * candidate to production (PUT roles/{role}/production). All are only invoked from an
 * explicit ConfirmActionDialog confirmation in the UI.
 */
class ModelDetailViewModel(
    private val modelName: String,
    private val modelRepository: ModelRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ModelDetailUiState())
    val uiState: StateFlow<ModelDetailUiState> = _uiState.asStateFlow()

    suspend fun pollModel() {
        FixedIntervalPoller(DETAIL_POLL_INTERVAL_MS).run { refresh() }
    }

    private suspend fun refresh() {
        val result = modelRepository.getModel(modelName)
        if (result is ApiResult.Success) {
            _uiState.update { it.copy(model = result.data, isLoading = false, error = null) }
        } else {
            _uiState.update { it.copy(isLoading = false, error = result.errorMessageOrNull()) }
        }
    }

    fun clearStaleTrainingLock() {
        // TD-147: the server's ownership check on this transition requires the model's
        // own current run_id — an empty one is rejected, not treated as an override —
        // so it must be threaded through from the most recently polled record.
        val runId = _uiState.value.model?.run_id.orEmpty()
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = modelRepository.clearStaleTrainingLock(modelName, runId)) {
                is ApiResult.Success -> _uiState.update {
                    it.copy(model = result.data, actionInProgress = false, actionMessage = "Lock cleared — model is now a candidate.")
                }
                else -> _uiState.update {
                    it.copy(actionInProgress = false, actionMessage = "Failed: ${result.errorMessageOrNull()}")
                }
            }
        }
    }

    fun retireCandidate() {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = modelRepository.retireCandidate(modelName)) {
                is ApiResult.Success -> _uiState.update {
                    it.copy(model = result.data, actionInProgress = false, actionMessage = "Model retired.")
                }
                else -> _uiState.update {
                    it.copy(actionInProgress = false, actionMessage = "Failed: ${result.errorMessageOrNull()}")
                }
            }
        }
    }

    fun promoteToProduction(role: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = modelRepository.promoteToProduction(role, modelName)) {
                is ApiResult.Success -> {
                    val retired = result.data.retired.ifEmpty { "none" }
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = "Promoted to production for role '${result.data.role}' (retired: $retired).",
                        )
                    }
                    refresh()
                }
                else -> _uiState.update {
                    it.copy(actionInProgress = false, actionMessage = "Failed: ${result.errorMessageOrNull()}")
                }
            }
        }
    }

    fun dismissActionMessage() {
        _uiState.update { it.copy(actionMessage = null) }
    }

    private companion object {
        const val DETAIL_POLL_INTERVAL_MS = 5000L
    }
}
