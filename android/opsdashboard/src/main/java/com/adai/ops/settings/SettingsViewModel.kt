package com.adai.ops.settings

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.6.0
// @adai-reviewed: 2026-11-21


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

data class SettingsUiState(
    val useSharedHost: Boolean = true,
    val sharedHost: String = "",
    val metricsHost: String = "",
    val metricsPort: String = OpsSettings.DEFAULT_METRICS_PORT.toString(),
    val mnsHost: String = "",
    val mnsPort: String = OpsSettings.DEFAULT_MNS_PORT.toString(),
    val registryHost: String = "",
    val registryPort: String = OpsSettings.DEFAULT_REGISTRY_PORT.toString(),
    val trainerHost: String = "",
    val trainerPort: String = OpsSettings.DEFAULT_TRAINER_PORT.toString(),
    val registryGroups: List<String> = emptyList(),
    val newGroupInput: String = "",
    val basePollIntervalMs: Long = OpsSettings.DEFAULT_POLL_INTERVAL_MS,
    val useHttpsRelay: Boolean = false,
    val accessClientId: String = "",
    val accessClientSecret: String = "",
    val trainerAccessClientId: String = "",
    val trainerAccessClientSecret: String = "",
    val saved: Boolean = false,
)

class SettingsViewModel(
    private val settingsRepository: OpsSettingsRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState: StateFlow<SettingsUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val current = settingsRepository.settings.first()
            _uiState.value = _uiState.value.copy(
                useSharedHost = current.useSharedHost,
                sharedHost = current.sharedHost,
                metricsHost = current.metricsHost,
                metricsPort = current.metricsPort.toString(),
                mnsHost = current.mnsHost,
                mnsPort = current.mnsPort.toString(),
                registryHost = current.registryHost,
                registryPort = current.registryPort.toString(),
                trainerHost = current.trainerHost,
                trainerPort = current.trainerPort.toString(),
                registryGroups = current.registryGroups,
                basePollIntervalMs = current.basePollIntervalMs,
                useHttpsRelay = current.useHttpsRelay,
                accessClientId = current.accessClientId,
                accessClientSecret = current.accessClientSecret,
                trainerAccessClientId = current.trainerAccessClientId,
                trainerAccessClientSecret = current.trainerAccessClientSecret,
            )
        }
    }

    fun onUseHttpsRelayChanged(value: Boolean) {
        _uiState.value = _uiState.value.copy(useHttpsRelay = value)
    }

    fun onAccessClientIdChanged(value: String) {
        _uiState.value = _uiState.value.copy(accessClientId = value)
    }

    fun onAccessClientSecretChanged(value: String) {
        _uiState.value = _uiState.value.copy(accessClientSecret = value)
    }

    fun onTrainerAccessClientIdChanged(value: String) {
        _uiState.value = _uiState.value.copy(trainerAccessClientId = value)
    }

    fun onTrainerAccessClientSecretChanged(value: String) {
        _uiState.value = _uiState.value.copy(trainerAccessClientSecret = value)
    }

    fun onUseSharedHostChanged(value: Boolean) {
        _uiState.value = _uiState.value.copy(useSharedHost = value)
    }

    fun onSharedHostChanged(value: String) {
        _uiState.value = _uiState.value.copy(sharedHost = value)
    }

    fun onMetricsHostChanged(value: String) {
        _uiState.value = _uiState.value.copy(metricsHost = value)
    }

    fun onMetricsPortChanged(value: String) {
        _uiState.value = _uiState.value.copy(metricsPort = value)
    }

    fun onMnsHostChanged(value: String) {
        _uiState.value = _uiState.value.copy(mnsHost = value)
    }

    fun onMnsPortChanged(value: String) {
        _uiState.value = _uiState.value.copy(mnsPort = value)
    }

    fun onRegistryHostChanged(value: String) {
        _uiState.value = _uiState.value.copy(registryHost = value)
    }

    fun onRegistryPortChanged(value: String) {
        _uiState.value = _uiState.value.copy(registryPort = value)
    }

    fun onTrainerHostChanged(value: String) {
        _uiState.value = _uiState.value.copy(trainerHost = value)
    }

    fun onTrainerPortChanged(value: String) {
        _uiState.value = _uiState.value.copy(trainerPort = value)
    }

    fun onNewGroupInputChanged(value: String) {
        _uiState.value = _uiState.value.copy(newGroupInput = value)
    }

    fun onBasePollIntervalChanged(ms: Long) {
        _uiState.value = _uiState.value.copy(basePollIntervalMs = ms)
    }

    fun addGroup() {
        val name = _uiState.value.newGroupInput.trim()
        if (name.isEmpty() || _uiState.value.registryGroups.contains(name)) return
        _uiState.value = _uiState.value.copy(
            registryGroups = _uiState.value.registryGroups + name,
            newGroupInput = "",
        )
    }

    fun removeGroup(name: String) {
        _uiState.value = _uiState.value.copy(
            registryGroups = _uiState.value.registryGroups.filterNot { it == name },
        )
    }

    /**
     * Persists the current settings. Deliberately a `suspend fun`, not a fire-and-forget
     * `viewModelScope.launch` — the Settings screen's Save button navigates back
     * immediately after calling this, and navigating back pops this screen's
     * NavBackStackEntry, which clears this ViewModel and cancels viewModelScope
     * (and everything running in it). A `viewModelScope.launch { settingsRepository
     * .save(...) }` here would race that cancellation: if the back-stack entry is
     * destroyed before the DataStore write's coroutine resumes, the write is
     * cancelled mid-flight and the save is silently lost — the UI already navigated
     * away as if it succeeded, with no error shown. Reproduced directly (see
     * TECHNICAL_DEBT_RESOLVED.md) — a fire-and-forget launch immediately followed by
     * cancelling its scope loses the write every time, while awaiting the same
     * suspend call before the caller proceeds to whatever might cancel that scope
     * always completes it first, since the two are sequenced in one coroutine.
     * The caller (SettingsScreen) is responsible for awaiting this in a scope of
     * its own (rememberCoroutineScope(), not viewModelScope) before calling onBack().
     */
    suspend fun save() {
        val state = _uiState.value
        val settings = OpsSettings(
            useSharedHost = state.useSharedHost,
            sharedHost = state.sharedHost.trim(),
            metricsHost = state.metricsHost.trim(),
            metricsPort = state.metricsPort.toIntOrNull() ?: OpsSettings.DEFAULT_METRICS_PORT,
            mnsHost = state.mnsHost.trim(),
            mnsPort = state.mnsPort.toIntOrNull() ?: OpsSettings.DEFAULT_MNS_PORT,
            registryHost = state.registryHost.trim(),
            registryPort = state.registryPort.toIntOrNull() ?: OpsSettings.DEFAULT_REGISTRY_PORT,
            trainerHost = state.trainerHost.trim(),
            trainerPort = state.trainerPort.toIntOrNull() ?: OpsSettings.DEFAULT_TRAINER_PORT,
            registryGroups = state.registryGroups,
            basePollIntervalMs = state.basePollIntervalMs,
            useHttpsRelay = state.useHttpsRelay,
            accessClientId = state.accessClientId.trim(),
            accessClientSecret = state.accessClientSecret.trim(),
            trainerAccessClientId = state.trainerAccessClientId.trim(),
            trainerAccessClientSecret = state.trainerAccessClientSecret.trim(),
        )
        settingsRepository.save(settings)
        _uiState.value = _uiState.value.copy(saved = true)
    }
}
