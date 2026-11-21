package com.adai.ops.ui.registry

// @adai-status: beta        (TD-048 resolved — see GroupListViewModelTest.kt)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-11-21


import androidx.lifecycle.ViewModel
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.errorMessageOrNull
import com.adai.ops.polling.FixedIntervalPoller
import com.adai.ops.settings.OpsSettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update

data class GroupSummary(
    val name: String,
    val pendingCount: Int? = null,
    val error: String? = null,
    // TD-580: the non-retired MNS models whose run_group this is (empty for a group only in Settings).
    val models: List<String> = emptyList(),
)

data class GroupListUiState(
    val groups: List<GroupSummary> = emptyList(),
    val isLoading: Boolean = true,
    // TD-580: why the groups couldn't be read from MNS, if they couldn't (Settings' groups still show).
    val discoveryError: String? = null,
)

/**
 * registry_server has no endpoint that lists its groups (RegistryServer.cpp creates one lazily on
 * first use), so the list is assembled here. TD-580: every model in MNS records the group it trains
 * from (run_group), so the groups of all non-retired models are listed, after the names added in
 * Settings (for groups no model names). Each is annotated with a live pending-file count. If MNS
 * can't be reached, Settings' groups still show, with the error in [GroupListUiState.discoveryError].
 */
class GroupListViewModel(
    private val registryRepository: RegistryRepository,
    private val settingsRepository: OpsSettingsRepository,
    private val modelRepository: ModelRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(GroupListUiState())
    val uiState: StateFlow<GroupListUiState> = _uiState.asStateFlow()

    suspend fun pollGroups() {
        FixedIntervalPoller(LIST_POLL_INTERVAL_MS).run { refresh() }
    }

    private suspend fun refresh() {
        val configured = settingsRepository.settings.first().registryGroups
        var discoveryError: String? = null
        val modelsByGroup: Map<String, List<String>> =
            when (val result = modelRepository.listModels(limit = MNS_LIST_LIMIT)) {
                is ApiResult.Success -> result.data.models
                    .filter { it.state != RETIRED_STATE && it.run_group.isNotBlank() }
                    .groupBy({ it.run_group }, { it.model_name })
                    .mapValues { (_, models) -> models.sorted() }
                else -> {
                    discoveryError = result.errorMessageOrNull()
                    emptyMap()
                }
            }
        val groupNames = (configured + modelsByGroup.keys.sorted()).distinct()
        val summaries = groupNames.map { name ->
            // TD-205: a group's pending count used to come from the legacy pool alone
            // (queue(name), no kind) — since TD-202 split each group into per-kind
            // sub-pools, a group whose data has all been migrated/queued under a kind
            // would show a misleading 0 here. Sum across the legacy pool plus every kind
            // instead, mirroring GroupDetailViewModel's own poolHealth computation
            // (a per-sub-pool failure just contributes 0, same as there).
            var total = 0
            var firstError: String? = null
            for (kind in ALL_KINDS) {
                when (val result = registryRepository.queue(name, kind)) {
                    is ApiResult.Success -> total += result.data.entries.size
                    else -> if (firstError == null) firstError = result.errorMessageOrNull()
                }
            }
            val models = modelsByGroup[name].orEmpty()
            if (firstError != null) {
                GroupSummary(name, error = firstError, models = models)
            } else {
                GroupSummary(name, pendingCount = total, models = models)
            }
        }
        _uiState.update { it.copy(groups = summaries, isLoading = false, discoveryError = discoveryError) }
    }

    private companion object {
        const val LIST_POLL_INTERVAL_MS = 8000L
        // MNS returns 50 models unless asked for more; ask for every model a deployment could have.
        const val MNS_LIST_LIMIT = 1000
        const val RETIRED_STATE = "retired"
        val ALL_KINDS: List<String?> = listOf(null, "encoder", "decoder", "world_model", "chatbot")
    }
}
