package com.adai.ops.ui.models

// @adai-status: beta        (TD-199 review fix — loadChatbotLinkCandidates() added, no longer reuses the list's own kind-filtered state.models)
// @adai-version: 0.3.1
// @adai-reviewed: 2026-09-19


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.RegisterModelRequestDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ModelListUiState(
    val models: List<ModelRecordDto> = emptyList(),
    val selectedKind: String? = null,
    val isLoading: Boolean = true,
    val error: String? = null,
    val registerInProgress: Boolean = false,
    val registerMessage: String? = null,
    val encoderCandidates: List<ModelRecordDto> = emptyList(),
    val decoderCandidates: List<ModelRecordDto> = emptyList(),
)

class ModelListViewModel(private val modelRepository: ModelRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(ModelListUiState())
    val uiState: StateFlow<ModelListUiState> = _uiState.asStateFlow()

    suspend fun pollModels() {
        FixedIntervalPoller(LIST_POLL_INTERVAL_MS).run { refresh() }
    }

    /** TD-196: All/Chatbot/Encoder/Decoder/World Model filter chips — null means "All". */
    fun setKindFilter(kind: String?) {
        _uiState.update { it.copy(selectedKind = kind) }
        viewModelScope.launch { refresh() }
    }

    /**
     * TD-199 (review fix): populates the encoder/decoder pickers for "Register Chatbot"'s Linked
     * mode with their own dedicated, unfiltered-by-kind-chip fetches — RegisterChatbotDialog used
     * to be handed [ModelListUiState.models] directly, which is always restricted to whatever
     * kind filter chip happens to be selected on the list screen (see refresh()), so picking e.g.
     * "Encoder" to browse the list before registering a chatbot silently emptied the Decoder
     * picker. Mirrors ModelDetailViewModel.loadWorldModelCandidates()'s own dedicated-fetch
     * pattern.
     */
    fun loadChatbotLinkCandidates() {
        viewModelScope.launch {
            val encoders = modelRepository.listModels(kind = "encoder")
            val decoders = modelRepository.listModels(kind = "decoder")
            if (encoders is ApiResult.Success) {
                _uiState.update { it.copy(encoderCandidates = encoders.data.models) }
            }
            if (decoders is ApiResult.Success) {
                _uiState.update { it.copy(decoderCandidates = decoders.data.models) }
            }
        }
    }

    /**
     * TD-196: POST /models, called after the caller's own kind-specific data-entry dialog and
     * ConfirmActionDialog (biometric/PIN gate) have both already run — see RegisterModelDialogs.kt.
     * One method handles every kind since they all resolve to the same request shape; only the
     * dialog that built [request] differs per kind.
     */
    fun registerModel(request: RegisterModelRequestDto) {
        viewModelScope.launch {
            _uiState.update { it.copy(registerInProgress = true, registerMessage = null) }
            when (val result = modelRepository.registerModel(request)) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            registerInProgress = false,
                            registerMessage = "Registered '${request.model_name}' (${request.kind}).",
                        )
                    }
                    refresh()
                }
                else -> _uiState.update {
                    it.copy(registerInProgress = false, registerMessage = "Failed: ${result.errorMessageOrNull()}")
                }
            }
        }
    }

    fun dismissRegisterMessage() {
        _uiState.update { it.copy(registerMessage = null) }
    }

    private suspend fun refresh() {
        val kind = _uiState.value.selectedKind
        val result = modelRepository.listModels(kind = kind)
        if (result is ApiResult.Success) {
            _uiState.update { it.copy(models = result.data.models, isLoading = false, error = null) }
        } else {
            _uiState.update { it.copy(isLoading = false, error = result.errorMessageOrNull()) }
        }
    }

    private companion object {
        const val LIST_POLL_INTERVAL_MS = 8000L
    }
}
