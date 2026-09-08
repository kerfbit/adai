package com.adai.ops.ui.registry

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class GroupDetailUiState(
    val group: String,
    val queueEntries: List<QueueEntryDto> = emptyList(),
    val runs: Map<String, List<String>> = emptyMap(),
    val registryEntries: List<RegistryEntryDto> = emptyList(),
    val models: List<ModelRecordDto> = emptyList(),
    val isLoading: Boolean = true,
    val error: String? = null,
    val actionInProgress: Boolean = false,
    val actionMessage: String? = null,
)

/**
 * Drives Registry Group Detail: the pending queue/run claims, the trained dataset
 * registry (GET /registry/{group}/registry — Phase 15; previously fetched by nothing
 * in this app), force-releasing files claimed by a (presumed dead) run_id
 * (POST /registry/{group}/release with an empty run_id, which bypasses the owner
 * check), assigning a model to pending files (POST /registry/{group}/assign —
 * Phase 14; previously a no-op in remote mode, see RegistryRepository's doc
 * comment), and triggering server-side Gutenberg/HuggingFace fetches
 * (POST /registry/{group}/fetch/gutenberg|huggingface).
 *
 * PendingEntry's run_id carries no timestamp anywhere in the wire format (verified
 * against RegistryTransport.hpp), so there is no "claimed since" to show — only
 * "claimed by". added_utc (Phase 15) is unrelated — when the entry first entered
 * the system, not when it was claimed.
 */
class GroupDetailViewModel(
    private val group: String,
    private val registryRepository: RegistryRepository,
    private val modelRepository: ModelRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(GroupDetailUiState(group = group))
    val uiState: StateFlow<GroupDetailUiState> = _uiState.asStateFlow()

    suspend fun pollGroup() {
        FixedIntervalPoller(DETAIL_POLL_INTERVAL_MS).run { refresh() }
    }

    private suspend fun refresh() {
        val queueResult = registryRepository.queue(group)
        val runsResult = registryRepository.runs(group)
        val registryResult = registryRepository.registry(group)
        // Cheap and infrequently-changing — piggybacks on the same refresh tick
        // rather than a separate poller, purely to populate the model picker.
        val modelsResult = modelRepository.listModels()
        _uiState.update {
            it.copy(
                queueEntries = (queueResult as? ApiResult.Success)?.data?.entries ?: it.queueEntries,
                runs = (runsResult as? ApiResult.Success)?.data?.runs ?: it.runs,
                registryEntries = (registryResult as? ApiResult.Success)?.data?.entries
                    ?: it.registryEntries,
                models = (modelsResult as? ApiResult.Success)?.data?.models ?: it.models,
                isLoading = false,
                error = queueResult.errorMessageOrNull() ?: runsResult.errorMessageOrNull(),
            )
        }
    }

    fun forceReleaseRun(runId: String, files: List<String>) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = registryRepository.forceRelease(group, files)) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = "Released ${result.data.released} file(s) claimed by '$runId'.",
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

    /** Assigns [modelName] to a single pending file; empty [modelName] is rejected server-side. */
    fun assignModel(path: String, modelName: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = registryRepository.assignModel(group, modelName, listOf(path))) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = if (result.data.assigned > 0) {
                                "Assigned '$path' to model '$modelName'."
                            } else {
                                "No matching pending file found for '$path'."
                            },
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

    fun fetchGutenberg(bookId: Int, numPairs: Int, modelName: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = registryRepository.fetchGutenberg(group, bookId, numPairs, modelName)) {
                is ApiResult.Success -> {
                    val data = result.data
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = if (data.added) {
                                "Fetched book #$bookId: served sentences [${data.served_from_row}, " +
                                    "${data.next_row}) → ${data.pairs_written} pair(s)."
                            } else {
                                "Fetch failed: ${data.reason ?: "unknown reason"}"
                            },
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

    fun fetchHuggingface(
        datasetId: String,
        numPairs: Int,
        split: String,
        inputField: String,
        outputField: String,
        modelName: String,
    ) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (
                val result = registryRepository.fetchHuggingface(
                    group, datasetId, numPairs, split, inputField, outputField, modelName,
                )
            ) {
                is ApiResult.Success -> {
                    val data = result.data
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = if (data.added) {
                                "Fetched '$datasetId': served rows [${data.served_from_row}, " +
                                    "${data.next_row}) → ${data.pairs_written} pair(s)."
                            } else {
                                "Fetch failed: ${data.reason ?: "unknown reason"}"
                            },
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
        const val DETAIL_POLL_INTERVAL_MS = 6000L
    }
}
