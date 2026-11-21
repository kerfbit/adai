package com.adai.ops.settings

import com.adai.ops.testutil.FakeSettingsRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * TD-048: SettingsViewModel (opsdashboard module) had zero test coverage — discovered while
 * building SettingsScreenTest, which had nothing to reuse a Fixture shape from.
 *
 * init{} uses viewModelScope.launch directly (fire-and-forget, same shape as
 * ModelDetailViewModel's action methods), which needs a real Main dispatcher — hence the
 * StandardTestDispatcher setup below.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class SettingsViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    private fun viewModel(initial: OpsSettings = OpsSettings()) =
        SettingsViewModel(FakeSettingsRepository(initial))

    @Test
    fun `initial state loads every field from the repository`() = runTest {
        val stored = OpsSettings(
            useSharedHost = false,
            sharedHost = "shared.example.com",
            metricsHost = "metrics.example.com",
            metricsPort = 9001,
            mnsHost = "mns.example.com",
            mnsPort = 9002,
            registryHost = "registry.example.com",
            registryPort = 9003,
            trainerHost = "trainer.example.com",
            trainerPort = 9004,
            registryGroups = listOf("group-a", "group-b"),
            basePollIntervalMs = 5000L,
            useHttpsRelay = true,
            accessClientId = "client-id",
            accessClientSecret = "client-secret",
            trainerAccessClientId = "trainer-client-id",
            trainerAccessClientSecret = "trainer-client-secret",
        )
        val vm = viewModel(stored)
        testScheduler.runCurrent()

        val state = vm.uiState.value
        assertEquals(false, state.useSharedHost)
        assertEquals("shared.example.com", state.sharedHost)
        assertEquals("metrics.example.com", state.metricsHost)
        assertEquals("9001", state.metricsPort)
        assertEquals("mns.example.com", state.mnsHost)
        assertEquals("9002", state.mnsPort)
        assertEquals("registry.example.com", state.registryHost)
        assertEquals("9003", state.registryPort)
        assertEquals("trainer.example.com", state.trainerHost)
        assertEquals("9004", state.trainerPort)
        assertEquals(listOf("group-a", "group-b"), state.registryGroups)
        assertEquals(5000L, state.basePollIntervalMs)
        assertEquals(true, state.useHttpsRelay)
        assertEquals("client-id", state.accessClientId)
        assertEquals("client-secret", state.accessClientSecret)
        assertEquals("trainer-client-id", state.trainerAccessClientId)
        assertEquals("trainer-client-secret", state.trainerAccessClientSecret)
    }

    @Test
    fun `each field-change handler updates its own state field independently`() = runTest {
        val vm = viewModel()
        testScheduler.runCurrent()

        vm.onUseSharedHostChanged(false)
        vm.onSharedHostChanged("shared-2")
        vm.onMetricsHostChanged("metrics-2")
        vm.onMetricsPortChanged("1111")
        vm.onMnsHostChanged("mns-2")
        vm.onMnsPortChanged("2222")
        vm.onRegistryHostChanged("registry-2")
        vm.onRegistryPortChanged("3333")
        vm.onTrainerHostChanged("trainer-2")
        vm.onTrainerPortChanged("4444")
        vm.onBasePollIntervalChanged(10000L)
        vm.onUseHttpsRelayChanged(true)
        vm.onAccessClientIdChanged("id-2")
        vm.onAccessClientSecretChanged("secret-2")
        vm.onTrainerAccessClientIdChanged("trainer-id-2")
        vm.onTrainerAccessClientSecretChanged("trainer-secret-2")
        vm.onNewGroupInputChanged("typed-group")

        val state = vm.uiState.value
        assertEquals(false, state.useSharedHost)
        assertEquals("shared-2", state.sharedHost)
        assertEquals("metrics-2", state.metricsHost)
        assertEquals("1111", state.metricsPort)
        assertEquals("mns-2", state.mnsHost)
        assertEquals("2222", state.mnsPort)
        assertEquals("registry-2", state.registryHost)
        assertEquals("3333", state.registryPort)
        assertEquals("trainer-2", state.trainerHost)
        assertEquals("4444", state.trainerPort)
        assertEquals(10000L, state.basePollIntervalMs)
        assertEquals(true, state.useHttpsRelay)
        assertEquals("id-2", state.accessClientId)
        assertEquals("secret-2", state.accessClientSecret)
        assertEquals("trainer-id-2", state.trainerAccessClientId)
        assertEquals("trainer-secret-2", state.trainerAccessClientSecret)
        assertEquals("typed-group", state.newGroupInput)
    }

    @Test
    fun `addGroup appends the trimmed name and clears the input`() = runTest {
        val vm = viewModel()
        testScheduler.runCurrent()

        vm.onNewGroupInputChanged("  nightly-batch  ")
        vm.addGroup()

        val state = vm.uiState.value
        assertEquals(listOf("nightly-batch"), state.registryGroups)
        assertEquals("", state.newGroupInput)
    }

    @Test
    fun `addGroup ignores a blank or whitespace-only input`() = runTest {
        val vm = viewModel()
        testScheduler.runCurrent()

        vm.onNewGroupInputChanged("   ")
        vm.addGroup()

        assertTrue(vm.uiState.value.registryGroups.isEmpty())
    }

    @Test
    fun `addGroup ignores a name already in the list`() = runTest {
        val vm = viewModel(OpsSettings(registryGroups = listOf("existing-group")))
        testScheduler.runCurrent()

        vm.onNewGroupInputChanged("existing-group")
        vm.addGroup()

        assertEquals(listOf("existing-group"), vm.uiState.value.registryGroups)
    }

    @Test
    fun `removeGroup removes only the matching group`() = runTest {
        val vm = viewModel(OpsSettings(registryGroups = listOf("group-a", "group-b")))
        testScheduler.runCurrent()

        vm.removeGroup("group-a")

        assertEquals(listOf("group-b"), vm.uiState.value.registryGroups)
    }

    @Test
    fun `save persists trimmed fields and falls back to defaults for unparseable ports`() = runTest {
        val settingsRepo = FakeSettingsRepository(OpsSettings())
        val vm = SettingsViewModel(settingsRepo)
        testScheduler.runCurrent()

        vm.onSharedHostChanged("  shared.example.com  ")
        vm.onMetricsPortChanged("not-a-number")
        vm.onMnsPortChanged("9002")
        vm.onAccessClientIdChanged("  client-id  ")

        vm.save()

        val saved = settingsRepo.settings.value
        assertEquals("shared.example.com", saved.sharedHost)
        assertEquals(OpsSettings.DEFAULT_METRICS_PORT, saved.metricsPort)
        assertEquals(9002, saved.mnsPort)
        assertEquals("client-id", saved.accessClientId)
        assertTrue(vm.uiState.value.saved)
    }
}
