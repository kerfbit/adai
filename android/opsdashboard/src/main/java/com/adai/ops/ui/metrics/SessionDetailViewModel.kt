package com.adai.ops.ui.metrics

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.data.wearsync.WatchSyncRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.CurrentMetricsDto
import com.adai.ops.network.dto.EpochHistoryDto
import com.adai.ops.network.dto.SampleRecordDto
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.AdaptivePoller
import com.adai.ops.polling.PollOutcome
import com.adai.ops.polling.PollerPhase
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SessionDetailUiState(
    val isLoading: Boolean = true,
    val status: SessionStatusDto? = null,
    val current: CurrentMetricsDto? = null,
    val epochs: EpochHistoryDto? = null,
    val sampleHistory: List<SampleRecordDto> = emptyList(),
    val pollerPhase: PollerPhase = PollerPhase.LIVE,
    val pollIntervalMs: Long = 2000L,
    val retryCount: Int = 0,
    val error: String? = null,
    val actionInProgress: Boolean = false,
    val actionMessage: String? = null,
)

sealed interface SessionDetailEvent {
    data object SessionEvicted : SessionDetailEvent
}

/**
 * Drives the Session Detail screen with the Tizen-app-derived AdaptivePoller: status +
 * current-metrics are fetched every tick, epoch/sample history at a slower cadence.
 * Polling is started via [poll], expected to be launched from a
 * repeatOnLifecycle(STARTED) block so backgrounding the app pauses it.
 */
class SessionDetailViewModel(
    private val sessionKey: String,
    private val metricsRepository: MetricsRepository,
    private val settingsRepository: OpsSettingsRepository,
    private val watchSyncRepository: WatchSyncRepository? = null,
) : ViewModel() {

    private val _uiState = MutableStateFlow(SessionDetailUiState())
    val uiState: StateFlow<SessionDetailUiState> = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<SessionDetailEvent>(extraBufferCapacity = 1)
    val events: SharedFlow<SessionDetailEvent> = _events.asSharedFlow()

    private var tick = 0

    suspend fun poll() = coroutineScope {
        val baseIntervalMs = settingsRepository.settings.first().basePollIntervalMs
        val poller = AdaptivePoller(baseIntervalMs)

        val stateCollector = launch {
            poller.state.collect { pollerState ->
                _uiState.update {
                    it.copy(
                        pollerPhase = pollerState.phase,
                        pollIntervalMs = pollerState.intervalMs,
                        retryCount = pollerState.retryCount,
                    )
                }
            }
        }

        poller.run { tickOnce() }

        stateCollector.cancel()
        _events.emit(SessionDetailEvent.SessionEvicted)
    }

    private suspend fun tickOnce(): PollOutcome {
        tick++

        val statusResult = metricsRepository.sessionStatus(sessionKey)
        if (statusResult is ApiResult.NotFound) return PollOutcome.NotFound

        val currentResult = metricsRepository.currentMetrics(sessionKey)
        if (currentResult is ApiResult.NotFound) return PollOutcome.NotFound

        val failureMessage = errorMessageOf(statusResult) ?: errorMessageOf(currentResult)
        if (failureMessage != null) {
            _uiState.update { it.copy(isLoading = false, error = failureMessage) }
            return PollOutcome.Failure(failureMessage)
        }

        val status = (statusResult as ApiResult.Success).data
        val current = (currentResult as ApiResult.Success).data

        if (tick % EPOCHS_REFRESH_EVERY_N_TICKS == 1) {
            val epochs = (metricsRepository.epochHistory(sessionKey) as? ApiResult.Success)?.data
            val samples = (metricsRepository.sampleHistory(sessionKey) as? ApiResult.Success)?.data?.records
            _uiState.update {
                it.copy(
                    epochs = epochs ?: it.epochs,
                    sampleHistory = samples ?: it.sampleHistory,
                )
            }

            // Fire-and-forget: the watch relay does its own (slower) full-session fetch and
            // Data Layer push, which shouldn't hold up this screen's own polling cadence.
            watchSyncRepository?.let { repo -> viewModelScope.launch { repo.sync() } }
        }

        _uiState.update { it.copy(isLoading = false, status = status, current = current, error = null) }
        return PollOutcome.Success(samplesPerSecond = status.samples_per_second.takeIf { it > 0.0 })
    }

    private fun errorMessageOf(result: ApiResult<*>): String? = when (result) {
        is ApiResult.NetworkError -> result.message
        is ApiResult.ApiError -> result.message
        is ApiResult.Conflict -> result.message
        else -> null
    }

    /**
     * Admin action: manually finalizes a session that crashed or hung and never called
     * its own /end, so it stops showing as "training" indefinitely.
     */
    fun endSession() {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = metricsRepository.endSession(sessionKey)) {
                is ApiResult.Success -> _uiState.update {
                    it.copy(actionInProgress = false, actionMessage = "Session ended.")
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
        const val EPOCHS_REFRESH_EVERY_N_TICKS = 5
    }
}
