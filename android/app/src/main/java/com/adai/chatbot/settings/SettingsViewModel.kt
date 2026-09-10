package com.adai.chatbot.settings

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.5.0
// @adai-reviewed: 2026-09-10


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.chatbot.data.repository.ChatRepository
import com.adai.chatbot.network.ApiClientProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

enum class ConnectionStatus { UNKNOWN, CHECKING, CONNECTED, FAILED }

data class SettingsUiState(
    val host: String = "",
    val port: String = ServerSettings.DEFAULT_PORT.toString(),
    val useHttps: Boolean = false,
    val accessClientId: String = "",
    val accessClientSecret: String = "",
    val connectionStatus: ConnectionStatus = ConnectionStatus.UNKNOWN,
    val connectionMessage: String? = null,
)

class SettingsViewModel(
    private val settingsDataStore: SettingsRepository,
    private val apiClientProvider: ApiClientProvider,
) : ViewModel() {

    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState: StateFlow<SettingsUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val current = settingsDataStore.settings.first()
            _uiState.value = _uiState.value.copy(
                host = current.host,
                port = current.port.toString(),
                useHttps = current.useHttps,
                accessClientId = current.accessClientId,
                accessClientSecret = current.accessClientSecret,
            )
        }
    }

    fun onHostChanged(value: String) {
        _uiState.value = _uiState.value.copy(host = value, connectionStatus = ConnectionStatus.UNKNOWN)
    }

    fun onPortChanged(value: String) {
        _uiState.value = _uiState.value.copy(port = value, connectionStatus = ConnectionStatus.UNKNOWN)
    }

    fun onUseHttpsChanged(value: Boolean) {
        _uiState.value = _uiState.value.copy(useHttps = value, connectionStatus = ConnectionStatus.UNKNOWN)
    }

    fun onAccessClientIdChanged(value: String) {
        _uiState.value = _uiState.value.copy(accessClientId = value, connectionStatus = ConnectionStatus.UNKNOWN)
    }

    fun onAccessClientSecretChanged(value: String) {
        _uiState.value = _uiState.value.copy(accessClientSecret = value, connectionStatus = ConnectionStatus.UNKNOWN)
    }

    fun testConnection() {
        val host = _uiState.value.host.trim()
        val port = _uiState.value.port.toIntOrNull()
        if (host.isBlank() || port == null || port !in 1..65535) {
            _uiState.value = _uiState.value.copy(
                connectionStatus = ConnectionStatus.FAILED,
                connectionMessage = "Enter a valid host and port (1-65535)",
            )
            return
        }

        _uiState.value = _uiState.value.copy(connectionStatus = ConnectionStatus.CHECKING)
        viewModelScope.launch {
            try {
                val health = apiClientProvider.serviceFor(
                    host,
                    port,
                    _uiState.value.useHttps,
                    _uiState.value.accessClientId.trim(),
                    _uiState.value.accessClientSecret.trim(),
                ).health()
                _uiState.value = _uiState.value.copy(
                    connectionStatus = ConnectionStatus.CONNECTED,
                    connectionMessage = "Connected — ${health.active_sessions ?: 0} active session(s)",
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    connectionStatus = ConnectionStatus.FAILED,
                    connectionMessage = ChatRepository.networkErrorMessage(e),
                )
            }
        }
    }

    /**
     * TD-130: was a plain `fun` that fired `viewModelScope.launch { settingsDataStore.save(...) }`
     * and returned immediately — a fire-and-forget write. SettingsScreen's Save button called
     * `viewModel.save(); onBack()` back to back; `onBack()` pops this screen's NavBackStackEntry,
     * which clears its ViewModelStore, which cancels `viewModelScope` and anything still running
     * in it. If that cancellation landed before the launched write actually completed (a real
     * race — Compose Navigation's default transition is effectively instant), the save was lost
     * silently: the UI navigates back as if it saved, but nothing was persisted. Same mechanism as
     * TD-127 (opsdashboard's SettingsViewModel), independently present in this sibling module. Now
     * a suspend fun with no internal launch, so it only returns once the write has actually landed
     * — the call site (SettingsScreen.kt) sequences it before onBack() in one coroutine instead.
     */
    suspend fun save() {
        val host = _uiState.value.host.trim()
        val port = _uiState.value.port.toIntOrNull() ?: ServerSettings.DEFAULT_PORT
        val useHttps = _uiState.value.useHttps
        val accessClientId = _uiState.value.accessClientId.trim()
        val accessClientSecret = _uiState.value.accessClientSecret.trim()
        settingsDataStore.save(host, port, useHttps, accessClientId, accessClientSecret)
    }
}
