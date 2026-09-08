package com.adai.ops.ui.admin

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-07


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.ApiResult
import com.adai.ops.network.errorMessageOrNull
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * Drives the Admin tab: GET /admin/config for all three daemons in parallel on entry, and
 * one PUT /admin/config call per field edit — always a single-key body (see e.g.
 * RegistryRepository.updateFtpTokenTtlMinutes's doc comment), never a full round-tripped
 * object. Every mutating call here is only ever invoked after ConfirmActionDialog's
 * device-credential gate has already succeeded (see ui/common/ConfirmActionDialog.kt) —
 * this ViewModel has no auth logic of its own.
 */
class AdminViewModel(
    private val modelRepository: ModelRepository,
    private val registryRepository: RegistryRepository,
    private val metricsRepository: MetricsRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(AdminUiState())
    val uiState: StateFlow<AdminUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch { loadAll() }
    }

    suspend fun loadAll() = coroutineScope {
        val mnsDeferred = async { modelRepository.getAdminConfig() }
        val registryDeferred = async { registryRepository.getAdminConfig() }
        val metricsDeferred = async { metricsRepository.getAdminConfig() }
        val mnsResult = mnsDeferred.await()
        val registryResult = registryDeferred.await()
        val metricsResult = metricsDeferred.await()
        _uiState.update {
            it.copy(
                mnsConfig = (mnsResult as? ApiResult.Success)?.data ?: it.mnsConfig,
                mnsError = mnsResult.errorMessageOrNull(),
                registryConfig = (registryResult as? ApiResult.Success)?.data ?: it.registryConfig,
                registryError = registryResult.errorMessageOrNull(),
                metricsConfig = (metricsResult as? ApiResult.Success)?.data ?: it.metricsConfig,
                metricsError = metricsResult.errorMessageOrNull(),
                isLoading = false,
            )
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
                    // Re-fetch all three rather than merging the single returned DTO in —
                    // simpler and admin-config reads are cheap/infrequent.
                    loadAll()
                    _uiState.update { it.copy(actionInProgress = false, actionMessage = "Updated $fieldLabel.") }
                }
                else -> _uiState.update { it.copy(actionInProgress = false, actionMessage = "Failed to update $fieldLabel.") }
            }
        }
    }

    // --- mns_server ---
    fun updateRegistryUrl(url: String) = runUpdate("registry_url") { modelRepository.updateRegistryUrl(url) }
    fun updateRegistryGroup(group: String) = runUpdate("registry_group") { modelRepository.updateRegistryGroup(group) }

    // --- registry_server ---
    fun updateFtpTokenTtlMinutes(minutes: Int) =
        runUpdate("ftp_token_ttl_minutes") { registryRepository.updateFtpTokenTtlMinutes(minutes) }
    fun updateFtpMaxSessionsPerRun(count: Int) =
        runUpdate("ftp_max_sessions_per_run") { registryRepository.updateFtpMaxSessionsPerRun(count) }

    // --- metrics_api_server ---
    fun updateMaxLiveSessions(n: Int) = runUpdate("max_live_sessions") { metricsRepository.updateMaxLiveSessions(n) }
    fun updateCompletedTtlSeconds(seconds: Int) =
        runUpdate("completed_ttl_seconds") { metricsRepository.updateCompletedTtlSeconds(seconds) }
    fun updateSweepIntervalSeconds(seconds: Int) =
        runUpdate("sweep_interval_seconds") { metricsRepository.updateSweepIntervalSeconds(seconds) }
    fun updatePersistEverySamples(n: Int) =
        runUpdate("persist_every_samples") { metricsRepository.updatePersistEverySamples(n) }
    fun updatePersistEverySeconds(seconds: Int) =
        runUpdate("persist_every_seconds") { metricsRepository.updatePersistEverySeconds(seconds) }
    fun updateMaxRecordsInMemory(n: Int) =
        runUpdate("max_records_in_memory") { metricsRepository.updateMaxRecordsInMemory(n) }
    fun updateMaxRecordsOnDisk(n: Int) =
        runUpdate("max_records_on_disk") { metricsRepository.updateMaxRecordsOnDisk(n) }
    fun updateEnablePrometheus(enabled: Boolean) =
        runUpdate("enable_prometheus") { metricsRepository.updateEnablePrometheus(enabled) }
    fun updateStalenessThresholdSeconds(seconds: Int) =
        runUpdate("staleness_threshold_seconds") { metricsRepository.updateStalenessThresholdSeconds(seconds) }
}
