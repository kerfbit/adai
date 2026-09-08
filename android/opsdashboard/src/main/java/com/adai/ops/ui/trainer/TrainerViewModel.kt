package com.adai.ops.ui.trainer

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.trainer.TrainerRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.TrainerLogEntryDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Drives the Trainer tab: GET /admin/config once on entry (like AdminViewModel —
 * config changes rarely, so a poll isn't worth it), and GET /admin/status +
 * GET /admin/logs together on one [FixedIntervalPoller] (phase/progress/log lines
 * all change continuously during training). The activity log shown by
 * [TrainerUiState.events] is the daemon's own real log (see [refreshLogs] and
 * TrainerLogEvent's doc comment) — this ViewModel does not fabricate its own
 * summary messages. Every control action (checkpoint/pause/resume/config edit) is
 * only invoked after ConfirmActionDialog's device-credential gate has already
 * succeeded — this ViewModel has no auth logic of its own.
 */
class TrainerViewModel(
    private val trainerRepository: TrainerRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(TrainerUiState())
    val uiState: StateFlow<TrainerUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch { loadConfig() }
    }

    suspend fun pollStatus() {
        FixedIntervalPoller(POLL_INTERVAL_MS).run {
            refreshStatus()
            refreshLogs()
        }
    }

    private suspend fun refreshStatus() {
        when (val result = trainerRepository.getStatus()) {
            is ApiResult.Success ->
                _uiState.update { it.copy(status = result.data, statusError = null, isLoading = false) }
            else -> _uiState.update { it.copy(statusError = result.errorMessageOrNull(), isLoading = false) }
        }
    }

    private suspend fun refreshLogs() {
        when (val result = trainerRepository.getLogs()) {
            is ApiResult.Success -> {
                // The server always returns its full current ring buffer (capped
                // at 200, oldest first) rather than an incremental "since last
                // poll" delta, so replacing wholesale (newest first for display)
                // is correct here, not just simplest — there's nothing to merge.
                val events = result.data.entries
                    .map { entry -> entry.toEvent() }
                    .sortedByDescending { it.id }
                _uiState.update { it.copy(events = events) }
            }
            // Deliberately not surfaced as its own error banner — statusError
            // (updated by refreshStatus, polled in the same tick) already covers
            // "the admin API is unreachable" for both endpoints at once.
            else -> Unit
        }
    }

    private fun TrainerLogEntryDto.toEvent(): TrainerLogEvent = TrainerLogEvent(
        id = id,
        timestampMillis = timestamp_unix_ms,
        severity = when (level.lowercase()) {
            "debug" -> TrainerLogSeverity.DEBUG
            "warn" -> TrainerLogSeverity.WARN
            "error" -> TrainerLogSeverity.ERROR
            else -> TrainerLogSeverity.INFO
        },
        message = message,
    )

    private suspend fun loadConfig() {
        when (val result = trainerRepository.getAdminConfig()) {
            is ApiResult.Success -> _uiState.update { it.copy(config = result.data, configError = null) }
            else -> _uiState.update { it.copy(configError = result.errorMessageOrNull()) }
        }
    }

    fun dismissActionMessage() {
        _uiState.update { it.copy(actionMessage = null) }
    }

    private fun runUpdate(fieldLabel: String, call: suspend () -> ApiResult<*>) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (call()) {
                is ApiResult.Success -> {
                    // Re-fetch rather than merging the single returned DTO in — simpler
                    // and admin-config reads are cheap/infrequent (matches AdminViewModel).
                    // The daemon logs "Admin config updated..." itself (see
                    // TrainerAdminAPI::handle_put_config), so the next log poll picks
                    // that up — no client-side event to append here.
                    loadConfig()
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Updated $fieldLabel.") }
                }
                else -> _uiState.update {
                    it.copy(actionInProgress = false, actionMessage = "Failed to update $fieldLabel.")
                }
            }
        }
    }

    fun updateAutoSaveEnabled(enabled: Boolean) =
        runUpdate("auto_save_enabled") { trainerRepository.updateAutoSaveEnabled(enabled) }
    fun updateAutoSaveEverySamples(samples: Int) =
        runUpdate("auto_save_every_samples") { trainerRepository.updateAutoSaveEverySamples(samples) }
    fun updateAutoSaveEveryMinutes(minutes: Int) =
        runUpdate("auto_save_every_minutes") { trainerRepository.updateAutoSaveEveryMinutes(minutes) }
    fun updateMaxSessionsToKeep(count: Int) =
        runUpdate("max_sessions_to_keep") { trainerRepository.updateMaxSessionsToKeep(count) }

    fun requestCheckpoint() {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = trainerRepository.requestCheckpoint()) {
                is ApiResult.Success -> {
                    val message = if (result.data.completed) {
                        "Checkpoint written to ${result.data.checkpoint_path.ifEmpty { "(unknown path)" }}."
                    } else {
                        "Checkpoint requested — will complete at the next optimizer-step boundary."
                    }
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = message) }
                    refreshStatus()
                    refreshLogs()
                }
                else -> {
                    val message = result.errorMessageOrNull() ?: "Unknown error"
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Checkpoint failed: $message") }
                }
            }
        }
    }

    fun pauseTraining() {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = trainerRepository.pause()) {
                is ApiResult.Success -> {
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Pause requested.") }
                    refreshStatus()
                    refreshLogs()
                }
                else -> {
                    val message = result.errorMessageOrNull() ?: "Unknown error"
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Failed to pause: $message") }
                }
            }
        }
    }

    fun resumeTraining() {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = trainerRepository.resume()) {
                is ApiResult.Success -> {
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Resume requested.") }
                    refreshStatus()
                    refreshLogs()
                }
                else -> {
                    val message = result.errorMessageOrNull() ?: "Unknown error"
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Failed to resume: $message") }
                }
            }
        }
    }

    private companion object {
        const val POLL_INTERVAL_MS = 5000L
    }
}
