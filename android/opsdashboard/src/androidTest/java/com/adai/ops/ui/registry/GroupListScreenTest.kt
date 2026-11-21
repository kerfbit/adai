package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-21


import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.settings.OpsSettings
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: GroupListScreen had zero coverage -- opsdashboard's first Compose UI test,
 * establishing the pattern for its own remaining screens (opsdashboard needed its own
 * androidx.compose.ui.test dependencies and src/sharedTest source set added first; see
 * :app's identical setup for ConversationListScreen). Reuses GroupListViewModelTest's own
 * real-repository Fixture shape (RegistryRepository backed by the shared src/sharedTest fakes)
 * rather than a real network. GroupListViewModel.pollGroups() runs one real refresh
 * immediately (FixedIntervalPoller.run() calls poll() before its first delay) and then loops
 * forever with an 8s delay between iterations -- these tests only need that first refresh to
 * land, observed via waitUntil() rather than relying on global Compose idle (which wouldn't
 * pause deterministically on a real, un-cancelled infinite polling coroutine anyway).
 */
class GroupListScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(
        settings: OpsSettings,
        fakeService: FakeRegistryApiService = FakeRegistryApiService(),
        mnsService: FakeMnsApiService = FakeMnsApiService(),
    ) = GroupListViewModel(
        registryRepository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository(settings)),
        settingsRepository = FakeSettingsRepository(settings),
        modelRepository = ModelRepository(FakeApiClientProvider(mnsService), FakeSettingsRepository(settings)),
    )

    private val configured = OpsSettings(useSharedHost = true, sharedHost = "localhost")

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun noConfiguredGroups_showsPlaceholderMessage() {
        val vm = viewModel(configured.copy(registryGroups = emptyList()))

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            textIsShowing("No registry groups: no model in MNS has a run group, and none are added in Settings.")
        }
    }

    // TD-580: a non-retired model's run_group is listed without being added in Settings.
    @Test
    fun groupOfAnMnsModel_isListedWithItsModels() {
        val mns = FakeMnsApiService(
            listModelsResponse = { _, _, _, _ ->
                ModelsResponseDto(
                    models = listOf(
                        ModelRecordDto(model_name = "bodacious-badger", run_group = "bb01"),
                        ModelRecordDto(model_name = "chatbot-v1", run_group = "chatbot-v1", state = "retired"),
                    ),
                )
            },
        )
        val vm = viewModel(configured.copy(registryGroups = emptyList()), mnsService = mns)

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("bb01") }
        composeTestRule.onNodeWithText("Models: bodacious-badger").assertExists()
        composeTestRule.onNodeWithText("chatbot-v1").assertDoesNotExist()
    }

    @Test
    fun unreachableMns_showsSettingsGroupsAndTheError() {
        val mns = FakeMnsApiService(listModelsResponse = { _, _, _, _ -> throw IOException("mns down") })
        val vm = viewModel(configured.copy(registryGroups = listOf("group-a")), mnsService = mns)

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("group-a") }
        composeTestRule.onNodeWithText(
            "Showing only the groups in Settings: couldn't list models from MNS (mns down).",
        ).assertExists()
    }

    @Test
    fun configuredGroups_showEachOnItsOwnRowWithPendingCount() {
        val fakeService = FakeRegistryApiService(
            // GroupListViewModel sums the legacy pool and every kind's sub-pool ("group-a",
            // "group-a/encoder", ...; TD-205), so only the legacy pool has entries here.
            queueResponse = { group ->
                when (group) {
                    "group-a" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "a1"), QueueEntryDto(path = "a2")))
                    "group-b" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "b1")))
                    else -> QueueResponseDto()
                }
            },
        )
        val vm = viewModel(configured.copy(registryGroups = listOf("group-a", "group-b")), fakeService)

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("2 pending file(s)") }
        composeTestRule.onNodeWithText("group-a").assertExists()
        composeTestRule.onNodeWithText("group-b").assertExists()
        composeTestRule.onNodeWithText("1 pending file(s)").assertExists()
    }

    @Test
    fun aGroupsFetchFailing_showsItsErrorWithoutHidingTheOthers() {
        val fakeService = FakeRegistryApiService(
            queueResponse = { group ->
                when {
                    group.startsWith("broken-group") -> throw IOException("connection refused")
                    group == "healthy-group" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "ok1")))
                    else -> QueueResponseDto()  // the healthy group's kind sub-pools (TD-205)
                }
            },
        )
        val vm = viewModel(configured.copy(registryGroups = listOf("broken-group", "healthy-group")), fakeService)

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Error: connection refused") }
        composeTestRule.onNodeWithText("healthy-group").assertExists()
        composeTestRule.onNodeWithText("1 pending file(s)").assertExists()
    }

    @Test
    fun clickingAGroupRow_invokesOnOpenGroupWithItsName() {
        val fakeService = FakeRegistryApiService(queueResponse = { QueueResponseDto() })
        val vm = viewModel(configured.copy(registryGroups = listOf("group-a")), fakeService)
        var opened: String? = null

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = { opened = it }, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("group-a") }
        composeTestRule.onNodeWithText("group-a").performClick()

        assert(opened == "group-a") { "expected onOpenGroup(\"group-a\"), got $opened" }
    }

    @Test
    fun clickingSettingsIcon_invokesOnOpenSettings() {
        val vm = viewModel(configured.copy(registryGroups = emptyList()))
        var settingsOpened = false

        composeTestRule.setContent {
            GroupListScreen(viewModel = vm, onOpenGroup = {}, onOpenSettings = { settingsOpened = true })
        }

        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        assert(settingsOpened) { "expected onOpenSettings() to have been invoked" }
    }
}
