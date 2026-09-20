package com.adai.ops.ui.models

// @adai-status: beta        (TD-196 — kind-aware design view: link/detach world model added)
// @adai-version: 0.6.0
// @adai-reviewed: 2026-09-19


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.LinkWorldModelRequestDto
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
    val worldModelCandidates: List<ModelRecordDto> = emptyList(),
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

    /** TD-196: populates the world-model picker for the "Link World Model" dialog on demand. */
    fun loadWorldModelCandidates() {
        viewModelScope.launch {
            val result = modelRepository.listModels(kind = "world_model")
            if (result is ApiResult.Success) {
                _uiState.update { it.copy(worldModelCandidates = result.data.models) }
            }
        }
    }

    /**
     * Admin action: POST /models/{name}/link-world-model. Only meaningful on a chatbot-kind
     * record; the server rejects (409) a d_model mismatch whenever
     * world_model_inject_every_n_layers > 0 in [request].
     */
    fun linkWorldModel(request: LinkWorldModelRequestDto) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = modelRepository.linkWorldModel(modelName, request)) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(actionInProgress = false, actionMessage = "Linked world model '${result.data.world_model_name}'.")
                    }
                    refresh()
                }
                else -> _uiState.update {
                    it.copy(actionInProgress = false, actionMessage = "Failed: ${result.errorMessageOrNull()}")
                }
            }
        }
    }

    /** Admin action: POST /models/{name}/link-world-model with an empty world_model_name. */
    fun detachWorldModel() {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = modelRepository.linkWorldModel(modelName, LinkWorldModelRequestDto(world_model_name = ""))) {
                is ApiResult.Success -> {
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "World model detached.") }
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
