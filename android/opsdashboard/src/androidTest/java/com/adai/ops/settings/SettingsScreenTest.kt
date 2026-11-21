package com.adai.ops.settings

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-21


import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import com.adai.ops.testutil.FakeSettingsRepository
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: SettingsScreen (opsdashboard module) had zero coverage -- unlike every other screen
 * covered so far, there wasn't even a pre-existing SettingsViewModelTest to reuse a Fixture shape
 * from, so this test builds its own directly against SettingsViewModel's real constructor shape
 * (FakeSettingsRepository).
 *
 * Runs on the Medium_Phone_API_35 AVD (`./gradlew :opsdashboard:connectedDebugAndroidTest`); it was
 * only compiled until November 21, 2026, when the APK stopped requiring `wear-sdk`.
 */
class SettingsScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private val sharedHostLabel = "Host (e.g. 192.168.1.16)"

    private class Fixture(initial: OpsSettings = OpsSettings()) {
        val settingsRepo = FakeSettingsRepository(initial)

        fun viewModel() = SettingsViewModel(settingsRepo)
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
        // The group controls are near the bottom of a scrolling column, below the fold on a
        // phone; a click on an off-screen node lands outside the window and does nothing.
        composeTestRule.onNodeWithText("Group name").performScrollTo().performTextInput("nightly-batch")
        composeTestRule.onNodeWithText("Add").performScrollTo().performClick()

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
        // The chip sits in a LazyRow inside the scrolling column; performScrollTo() on it scrolls
        // only the LazyRow, so bring the column's bottom (Save, just below the chips) on screen.
        composeTestRule.onNodeWithText("Save").performScrollTo()
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
        composeTestRule.onNodeWithText("5s").performScrollTo().performClick()
        // Save is at the bottom of a scrolling column, below the fold on a phone.
        composeTestRule.onNodeWithText("Save").performScrollTo().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { backInvoked }
        assert(fixture.settingsRepo.settings.value.basePollIntervalMs == 5000L) {
            "expected the selected 5s poll interval to have been persisted, got " +
                "${fixture.settingsRepo.settings.value.basePollIntervalMs}ms"
        }
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
}
