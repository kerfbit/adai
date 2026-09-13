package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.settings.OpsSettings
import com.adai.ops.testutil.FakeApiClientProvider
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
    ) = GroupListViewModel(
        registryRepository = RegistryRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository(settings)),
        settingsRepository = FakeSettingsRepository(settings),
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
            textIsShowing(
                "No registry groups configured. Add group names in Settings — " +
                    "registry_server has no way to list them for you.",
            )
        }
    }

    @Test
    fun configuredGroups_showEachOnItsOwnRowWithPendingCount() {
        val fakeService = FakeRegistryApiService(
            queueResponse = { group ->
                when (group) {
                    "group-a" -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "a1"), QueueEntryDto(path = "a2")))
                    else -> QueueResponseDto(entries = listOf(QueueEntryDto(path = "b1")))
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
                if (group == "broken-group") throw IOException("connection refused")
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "ok1")))
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
