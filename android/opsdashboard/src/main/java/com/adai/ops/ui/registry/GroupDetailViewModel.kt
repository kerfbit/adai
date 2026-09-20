package com.adai.ops.ui.registry

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.6.0
// @adai-reviewed: 2026-09-20


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.dto.SegmentTargetDto
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/** TD-205: one row of the pool-health overview — [kind] null means the legacy/unkinded pool. */
data class KindPoolHealth(
    val kind: String?,
    val pendingCount: Int = 0,
    val trainedCount: Int = 0,
    val totalSamples: Int = 0,
)

data class GroupDetailUiState(
    val group: String,
    // TD-202/TD-205: which pool's queue/runs/registry sections are currently shown; null =
    // the legacy/unkinded pool. poolHealth (below) always covers every kind regardless.
    val selectedKind: String? = null,
    val poolHealth: List<KindPoolHealth> = emptyList(),
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
 * TD-202/TD-205: [selectedKind] scopes every one of the above to that trainable piece's own
 * sub-pool. New actions round out this screen to full CLI parity: [unassignModel] (the
 * counterpart to [assignModel]), [deleteEntry] (real cleanup, not just organizing),
 * [manualAdd]/[upload] (the two ways to queue a new file besides Gutenberg/HuggingFace),
 * [migrateToKind] (move a legacy entry into a kind's own pool), and [createSegments] (split
 * a large file's pairs into independently-manageable row-range entries). [poolHealth] is the
 * "planning" deliverable: an at-a-glance view of where data is thin across all kinds, computed
 * alongside the normal per-tick refresh.
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

    fun setKindFilter(kind: String?) {
        _uiState.update { it.copy(selectedKind = kind) }
        viewModelScope.launch { refresh() }
    }

    private suspend fun refresh() {
        val kind = _uiState.value.selectedKind
        val queueResult = registryRepository.queue(group, kind)
        val runsResult = registryRepository.runs(group, kind)
        val registryResult = registryRepository.registry(group, kind)
        // Cheap and infrequently-changing — piggybacks on the same refresh tick
        // rather than a separate poller, purely to populate the model picker.
        val modelsResult = modelRepository.listModels()

        // TD-205: pool-health overview — one queue+registry pair per kind (including the
        // legacy pool), independent of the currently-selected filter, so an operator can see
        // every pool's shape at a glance without switching the filter chips one at a time.
        val poolHealth = ALL_KINDS.map { k ->
            val q = registryRepository.queue(group, k)
            val r = registryRepository.registry(group, k)
            val trainedEntries = (r as? ApiResult.Success)?.data?.entries.orEmpty()
            KindPoolHealth(
                kind = k,
                pendingCount = (q as? ApiResult.Success)?.data?.entries?.size ?: 0,
                trainedCount = trainedEntries.count { it.trained },
                totalSamples = trainedEntries.sumOf { it.num_samples },
            )
        }

        _uiState.update {
            it.copy(
                queueEntries = (queueResult as? ApiResult.Success)?.data?.entries ?: it.queueEntries,
                runs = (runsResult as? ApiResult.Success)?.data?.runs ?: it.runs,
                registryEntries = (registryResult as? ApiResult.Success)?.data?.entries
                    ?: it.registryEntries,
                models = (modelsResult as? ApiResult.Success)?.data?.models ?: it.models,
                poolHealth = poolHealth,
                isLoading = false,
                // TD-128: registryResult and modelsResult were fetched (and their
                // stale-on-failure fallback above already accounted for them) but
                // never checked here — only queueResult/runsResult's failures ever
                // reached `error`. registryResult in particular is the Phase 15
                // trained-files fetch this class's own doc comment calls out as a
                // newer addition; its errors were silently dropped ever since.
                error = queueResult.errorMessageOrNull()
                    ?: runsResult.errorMessageOrNull()
                    ?: registryResult.errorMessageOrNull()
                    ?: modelsResult.errorMessageOrNull(),
            )
        }
    }

    fun forceReleaseRun(runId: String, files: List<String>) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            when (val result = registryRepository.forceRelease(group, files, kind = kind)) {
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

    /** Assigns [modelName] to a pending entry; empty [modelName] is rejected server-side. */
    fun assignModel(entry: QueueEntryDto, modelName: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            val result = if (entry.isSegment) {
                registryRepository.assignModelToSegment(
                    group, modelName,
                    SegmentTargetDto(entry.path, entry.segment_start, entry.segment_count),
                    kind = kind,
                )
            } else {
                registryRepository.assignModel(group, modelName, listOf(entry.path), kind = kind)
            }
            when (result) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = if (result.data.assigned > 0) {
                                "Assigned '${entry.path}' to model '$modelName'."
                            } else {
                                "No matching pending file found for '${entry.path}'."
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

    /** TD-205: clears a pending entry's model assignment — the counterpart to [assignModel]. */
    fun unassignModel(entry: QueueEntryDto) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            val segment = if (entry.isSegment) {
                SegmentTargetDto(entry.path, entry.segment_start, entry.segment_count)
            } else {
                null
            }
            val paths = if (entry.isSegment) emptyList() else listOf(entry.path)
            when (
                val result = registryRepository.unassignModel(
                    group, paths = paths, force = true, segment = segment, kind = kind,
                )
            ) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = "Unassigned '${entry.path}' (was '${entry.model_name}').",
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

    /** TD-205: permanently purges a pending or trained entry, optionally unlinking the file. */
    fun deleteEntry(path: String, segment: SegmentTargetDto?, deleteFiles: Boolean) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            val paths = if (segment != null) emptyList() else listOf(path)
            when (
                val result = registryRepository.deleteEntries(
                    group, paths = paths, force = true, deleteFiles = deleteFiles,
                    segment = segment, kind = kind,
                )
            ) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = "Deleted ${result.data.deleted} entry(ies) for '$path'.",
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

    /** TD-205: queue an already-existing (server-readable) path — no new bytes transferred. */
    fun manualAdd(path: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            when (val result = registryRepository.pendingAdd(group, path, kind = kind)) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = if (result.data.added) {
                                "Queued '$path'."
                            } else {
                                "Not added: ${result.data.reason ?: "already pending"}"
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

    /** TD-205: upload a local file's raw bytes directly from the device. */
    fun upload(filename: String, bytes: ByteArray) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            when (val result = registryRepository.upload(group, filename, bytes, kind = kind)) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = if (result.data.added) {
                                "Uploaded '${result.data.path}'."
                            } else {
                                "Upload failed: ${result.data.reason ?: "unknown reason"}"
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

    /**
     * TD-205: move a legacy pending [entry] into [destKind]'s own sub-pool. Only meaningful
     * when currently viewing the legacy pool (selectedKind == null) — the destination kind is
     * always different from the (always-legacy) source, matching `dataset_manager migrate`'s
     * own design.
     */
    fun migrateToKind(entry: QueueEntryDto, destKind: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            when (val result = registryRepository.migrateToKind(group, entry, destKind)) {
                is ApiResult.Success -> {
                    _uiState.update {
                        it.copy(
                            actionInProgress = false,
                            actionMessage = "Migrated '${result.data}' into '$destKind'.",
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

    /**
     * TD-205: split [path]'s pairs into [ranges] (each a 0-based [start, start+count) index
     * range), queuing each as its own segment entry in the currently-selected kind's pool.
     */
    fun createSegments(path: String, ranges: List<Pair<Int, Int>>) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            val created = registryRepository.createSegments(group, path, ranges, kind = kind)
            _uiState.update {
                it.copy(
                    actionInProgress = false,
                    actionMessage = "Created $created/${ranges.size} segment(s) from '$path'.",
                )
            }
            refresh()
        }
    }

    fun fetchGutenberg(bookId: Int, numPairs: Int, modelName: String) {
        viewModelScope.launch {
            _uiState.update { it.copy(actionInProgress = true, actionMessage = null) }
            val kind = _uiState.value.selectedKind
            when (val result = registryRepository.fetchGutenberg(group, bookId, numPairs, modelName, kind = kind)) {
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
            val kind = _uiState.value.selectedKind
            when (
                val result = registryRepository.fetchHuggingface(
                    group, datasetId, numPairs, split, inputField, outputField, modelName, kind = kind,
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
        val ALL_KINDS: List<String?> = listOf(null, "encoder", "decoder", "world_model", "chatbot")
    }
}
