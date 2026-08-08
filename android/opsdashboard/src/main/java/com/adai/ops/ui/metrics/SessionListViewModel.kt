package com.adai.ops.ui.metrics

import androidx.lifecycle.ViewModel
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.network.dto.SessionSummaryDto
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class SessionListUiState(
    val sessions: List<SessionSummaryDto> = emptyList(),
    val isLoading: Boolean = true,
    val error: String? = null,
    val showIdle: Boolean = false,
)

class SessionListViewModel(private val metricsRepository: MetricsRepository) : ViewModel() {

    private val _uiState = MutableStateFlow(SessionListUiState())
    val uiState: StateFlow<SessionListUiState> = _uiState.asStateFlow()

    /** Called from a repeatOnLifecycle(STARTED) block so it pauses/resumes with app foreground state. */
    suspend fun pollSessions() {
        FixedIntervalPoller(LIST_POLL_INTERVAL_MS).run { refresh() }
    }

    /** Idle (non-training) sessions are hidden by default — most are old/completed runs. */
    fun toggleShowIdle() {
        _uiState.update { it.copy(showIdle = !it.showIdle) }
    }

    private suspend fun refresh() {
        val result = metricsRepository.listSessions()
        if (result is ApiResult.Success) {
            _uiState.update { it.copy(sessions = result.data.sessions, isLoading = false, error = null) }
        } else {
            _uiState.update { it.copy(isLoading = false, error = result.errorMessageOrNull()) }
        }
    }

    private companion object {
        const val LIST_POLL_INTERVAL_MS = 5000L
    }
}
