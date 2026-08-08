package com.adai.ops.ui.registry

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
            when (val result = registryRepository.queue(name)) {
                is ApiResult.Success -> GroupSummary(name, pendingCount = result.data.entries.size)
                else -> GroupSummary(name, error = result.errorMessageOrNull())
            }
        }
        _uiState.update { it.copy(groups = summaries, isLoading = false) }
    }

    private companion object {
        const val LIST_POLL_INTERVAL_MS = 8000L
    }
}
