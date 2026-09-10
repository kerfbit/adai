package com.adai.ops.ui.models

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-10


import androidx.lifecycle.ViewModel
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class ModelListUiState(
    val models: List<ModelRecordDto> = emptyList(),
    val isLoading: Boolean = true,
    val error: String? = null,
)

class ModelListViewModel(private val modelRepository: ModelRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(ModelListUiState())
    val uiState: StateFlow<ModelListUiState> = _uiState.asStateFlow()

    suspend fun pollModels() {
        FixedIntervalPoller(LIST_POLL_INTERVAL_MS).run { refresh() }
    }

    private suspend fun refresh() {
        val result = modelRepository.listModels()
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
