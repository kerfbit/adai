package com.adai.ops.ui.registry

// @adai-status: beta        (TD-048 resolved — see GroupListViewModelTest.kt)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-09-20


import androidx.lifecycle.ViewModel
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
)

data class GroupListUiState(
    val groups: List<GroupSummary> = emptyList(),
    val isLoading: Boolean = true,
)

/**
 * Registry groups have no server-side discovery endpoint (confirmed against
 * RegistryServer.cpp's lazily-created get_group()), so the group list comes from
 * user-maintained names in Settings; this screen just annotates each with a live
 * pending-file count.
 */
class GroupListViewModel(
    private val registryRepository: RegistryRepository,
    private val settingsRepository: OpsSettingsRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(GroupListUiState())
    val uiState: StateFlow<GroupListUiState> = _uiState.asStateFlow()

    suspend fun pollGroups() {
        FixedIntervalPoller(LIST_POLL_INTERVAL_MS).run { refresh() }
    }

    private suspend fun refresh() {
        val groupNames = settingsRepository.settings.first().registryGroups
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
            if (firstError != null) GroupSummary(name, error = firstError) else GroupSummary(name, pendingCount = total)
        }
        _uiState.update { it.copy(groups = summaries, isLoading = false) }
    }

    private companion object {
        const val LIST_POLL_INTERVAL_MS = 8000L
        val ALL_KINDS: List<String?> = listOf(null, "encoder", "decoder", "world_model", "chatbot")
    }
}
