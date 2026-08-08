package com.adai.ops.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.ops.data.wearsync.WatchFacePushRepository
import com.adai.ops.data.wearsync.WatchFacePushResult
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
    val registryGroups: List<String> = emptyList(),
    val newGroupInput: String = "",
    val basePollIntervalMs: Long = OpsSettings.DEFAULT_POLL_INTERVAL_MS,
    val useHttpsRelay: Boolean = false,
    val accessClientId: String = "",
    val accessClientSecret: String = "",
    val watchSyncEnabled: Boolean = true,
    val watchSyncSessionKeyOverride: String = "",
    val saved: Boolean = false,
    val watchFacePushInProgress: Boolean = false,
    val watchFacePushMessage: String? = null,
    val pushedWatchFaceSlotId: String? = null,
)

class SettingsViewModel(
    private val settingsRepository: OpsSettingsRepository,
    private val watchFacePushRepository: WatchFacePushRepository,
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
                registryGroups = current.registryGroups,
                basePollIntervalMs = current.basePollIntervalMs,
                useHttpsRelay = current.useHttpsRelay,
                accessClientId = current.accessClientId,
                accessClientSecret = current.accessClientSecret,
                watchSyncEnabled = current.watchSyncEnabled,
                watchSyncSessionKeyOverride = current.watchSyncSessionKeyOverride ?: "",
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

    fun onNewGroupInputChanged(value: String) {
        _uiState.value = _uiState.value.copy(newGroupInput = value)
    }

    fun onBasePollIntervalChanged(ms: Long) {
        _uiState.value = _uiState.value.copy(basePollIntervalMs = ms)
    }

    fun onWatchSyncEnabledChanged(value: Boolean) {
        _uiState.value = _uiState.value.copy(watchSyncEnabled = value)
    }

    fun onWatchSyncSessionKeyOverrideChanged(value: String) {
        _uiState.value = _uiState.value.copy(watchSyncSessionKeyOverride = value)
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

    /** Pushes the bundled WFF watch face to the paired watch via Watch Face Push — does not
     * activate it (see [activateWatchFace]), since setting active needs its own runtime
     * permission grant with a max-one-rejection limit. */
    fun pushWatchFace() {
        _uiState.value = _uiState.value.copy(watchFacePushInProgress = true, watchFacePushMessage = null)
        viewModelScope.launch {
            val message: String
            var slotId: String? = null
            when (val result = watchFacePushRepository.pushWatchFace()) {
                is WatchFacePushResult.Success -> {
                    slotId = result.slotId
                    message = "Installed on the watch (slot ${result.slotId}). Tap \"Activate\" to set it as the active face."
                }
                is WatchFacePushResult.ValidationFailed ->
                    message = "Watch face failed validation:\n" + result.reasons.joinToString("\n")
                is WatchFacePushResult.Failure -> message = result.message
            }
            _uiState.value = _uiState.value.copy(
                watchFacePushInProgress = false,
                watchFacePushMessage = message,
                pushedWatchFaceSlotId = slotId ?: _uiState.value.pushedWatchFaceSlotId,
            )
        }
    }

    /** Call only after the SET_PUSHED_WATCH_FACE_AS_ACTIVE runtime permission has been
     * granted — the caller (Settings screen) owns that request since it needs an Activity. */
    fun activateWatchFace() {
        val slotId = _uiState.value.pushedWatchFaceSlotId ?: return
        viewModelScope.launch {
            val result = watchFacePushRepository.setActive(slotId)
            val message = if (result.isSuccess) {
                "Activated on the watch."
            } else {
                "Couldn't activate: ${result.exceptionOrNull()?.message ?: "unknown error"}"
            }
            _uiState.value = _uiState.value.copy(watchFacePushMessage = message)
        }
    }

    fun dismissWatchFacePushMessage() {
        _uiState.value = _uiState.value.copy(watchFacePushMessage = null)
    }

    fun save() {
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
            registryGroups = state.registryGroups,
            basePollIntervalMs = state.basePollIntervalMs,
            useHttpsRelay = state.useHttpsRelay,
            accessClientId = state.accessClientId.trim(),
            accessClientSecret = state.accessClientSecret.trim(),
            watchSyncEnabled = state.watchSyncEnabled,
            watchSyncSessionKeyOverride = state.watchSyncSessionKeyOverride.trim().takeIf { it.isNotEmpty() },
        )
        viewModelScope.launch {
            settingsRepository.save(settings)
            _uiState.value = _uiState.value.copy(saved = true)
        }
    }
}
