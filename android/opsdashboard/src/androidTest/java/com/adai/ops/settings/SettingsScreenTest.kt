package com.adai.ops.settings

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.wearsync.WatchFacePushResult
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.FakeWatchFacePushRepository
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: SettingsScreen (opsdashboard module) had zero coverage -- unlike every other screen
 * covered so far, there wasn't even a pre-existing SettingsViewModelTest to reuse a Fixture shape
 * from, so this test builds its own directly against SettingsViewModel's real constructor shape
 * (FakeSettingsRepository + FakeWatchFacePushRepository).
 *
 * WatchFacePushRepository was split into an interface (see its own doc comment) specifically so
 * SettingsViewModel didn't need a real Wear system service (androidx.wear.watchfacepush) just to
 * be testable -- FakeWatchFacePushRepository (shared src/sharedTest fake) now lets the
 * "Install / update" (pushWatchFace()) flow itself be exercised here. Still deliberately out of
 * scope: actually clicking "Activate" -- unlike "Install / update", it first checks a runtime
 * permission (SET_PUSHED_WATCH_FACE_AS_ACTIVE) and, if not yet granted, launches the system
 * permission dialog via rememberLauncherForActivityResult, which needs a real Activity/device to
 * interact with meaningfully; only that its button appears after a successful push is checked.
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class SettingsScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private val sharedHostLabel = "Host (e.g. 192.168.1.16)"

    private class Fixture(
        initial: OpsSettings = OpsSettings(),
        val watchFacePushRepository: FakeWatchFacePushRepository = FakeWatchFacePushRepository(),
    ) {
        val settingsRepo = FakeSettingsRepository(initial)

        fun viewModel() = SettingsViewModel(settingsRepo, watchFacePushRepository)
    }

    @Test
    fun defaultState_showsSharedHostFieldAndPerServicePortsOnly() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText(sharedHostLabel).fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithTag("switch_use_shared_host").assertIsOn()
        composeTestRule.onNodeWithText("Metrics (metrics_api_server)").assertExists()
        composeTestRule.onNodeWithText("Models (mns_server)").assertExists()
        composeTestRule.onNodeWithText("Registry (registry_server)").assertExists()
        // useSharedHost=true hides each service's own Host field; useHttpsRelay=false (default)
        // still shows each service's Port field -- 4 of them (metrics/mns/registry/trainer).
        assert(composeTestRule.onAllNodesWithText("Host").fetchSemanticsNodes().isEmpty()) {
            "expected no per-service Host fields while useSharedHost is on"
        }
        assert(composeTestRule.onAllNodesWithText("Port").fetchSemanticsNodes().size == 4) {
            "expected 4 per-service Port fields (metrics/mns/registry/trainer)"
        }
    }

    @Test
    fun togglingUseSharedHost_hidesSharedHostFieldAndShowsPerServiceHostFields() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText(sharedHostLabel).fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithTag("switch_use_shared_host").performClick()

        composeTestRule.onNodeWithText(sharedHostLabel).assertDoesNotExist()
        assert(composeTestRule.onAllNodesWithText("Host").fetchSemanticsNodes().size == 4) {
            "expected 4 per-service Host fields once useSharedHost is off"
        }
    }

    @Test
    fun togglingHttpsRelay_showsAccessClientFieldsAndHidesPortFields() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText(sharedHostLabel).fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithTag("switch_use_https_relay").performClick()

        composeTestRule.onNodeWithText("Cloudflare Access Client ID").assertExists()
        composeTestRule.onNodeWithText("Cloudflare Access Client Secret").assertExists()
        composeTestRule.onNodeWithText("Trainer Access Client ID").assertExists()
        composeTestRule.onNodeWithText("Trainer Access Client Secret").assertExists()
        assert(composeTestRule.onAllNodesWithText("Port").fetchSemanticsNodes().isEmpty()) {
            "expected no Port fields once useHttpsRelay is on (relay mode has no per-service port)"
        }
    }

    @Test
    fun addingGroup_appendsChipAndClearsInput() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText(sharedHostLabel).fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Group name").performTextInput("nightly-batch")
        composeTestRule.onNodeWithText("Add").performClick()

        composeTestRule.onNodeWithText("nightly-batch").assertExists()
        composeTestRule.onNodeWithContentDescription("Remove nightly-batch").assertExists()
    }

    @Test
    fun removingGroup_removesItsChip() {
        val fixture = Fixture(OpsSettings(registryGroups = listOf("group-a")))

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("group-a").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithContentDescription("Remove group-a").performClick()

        composeTestRule.onNodeWithText("group-a").assertDoesNotExist()
    }

    @Test
    fun selectingPollIntervalAndSaving_persistsSelectionAndInvokesOnBack() {
        val fixture = Fixture()
        var backInvoked = false

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = { backInvoked = true })
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText(sharedHostLabel).fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("5s").performClick()
        composeTestRule.onNodeWithText("Save").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { backInvoked }
        assert(fixture.settingsRepo.settings.value.basePollIntervalMs == 5000L) {
            "expected the selected 5s poll interval to have been persisted, got " +
                "${fixture.settingsRepo.settings.value.basePollIntervalMs}ms"
        }
    }

    @Test
    fun watchSyncToggle_hidesSessionKeyOverrideFieldWhenOff() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText(sharedHostLabel).fetchSemanticsNodes().isNotEmpty()
        }
        // watchSyncEnabled defaults to true (see OpsSettings), so the override field starts shown.
        composeTestRule.onNodeWithText("Session key override (blank = Auto)").assertExists()
        composeTestRule.onNodeWithTag("switch_watch_sync_enabled").assertIsOn()

        composeTestRule.onNodeWithTag("switch_watch_sync_enabled").performClick()

        composeTestRule.onNodeWithTag("switch_watch_sync_enabled").assertIsOff()
        composeTestRule.onNodeWithText("Session key override (blank = Auto)").assertDoesNotExist()
    }

    @Test
    fun clickingBack_invokesOnBackWithoutSaving() {
        val fixture = Fixture()
        var backInvoked = false

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = { backInvoked = true })
        }

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
        assert(fixture.settingsRepo.settings.value.basePollIntervalMs == OpsSettings.DEFAULT_POLL_INTERVAL_MS) {
            "back without Save must not persist anything"
        }
    }

    @Test
    fun watchFaceSection_rendersInstallButtonEnabledByDefault() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("Galaxy Watch face (Watch Face Format)").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Install / update").assertIsEnabled()
    }

    @Test
    fun clickingInstallUpdate_successShowsSlotIdMessageAndActivateButton() {
        val fixture = Fixture(
            watchFacePushRepository = FakeWatchFacePushRepository(
                pushResult = { WatchFacePushResult.Success("slot-7") },
            ),
        )

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("Install / update").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Install / update").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("slot-7", substring = true).fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Activate").assertExists()
    }

    @Test
    fun clickingInstallUpdate_validationFailureShowsReasons() {
        val fixture = Fixture(
            watchFacePushRepository = FakeWatchFacePushRepository(
                pushResult = { WatchFacePushResult.ValidationFailed(listOf("bad manifest")) },
            ),
        )

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("Install / update").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Install / update").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("bad manifest", substring = true).fetchSemanticsNodes().isNotEmpty()
        }
        // A failed validation must not offer to activate a face that was never installed.
        composeTestRule.onNodeWithText("Activate").assertDoesNotExist()
    }

    @Test
    fun clickingInstallUpdate_failureShowsMessage() {
        val fixture = Fixture(
            watchFacePushRepository = FakeWatchFacePushRepository(
                pushResult = { WatchFacePushResult.Failure("no watch paired") },
            ),
        )

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("Install / update").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Install / update").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("no watch paired").fetchSemanticsNodes().isNotEmpty()
        }
        composeTestRule.onNodeWithText("Activate").assertDoesNotExist()
    }
}
