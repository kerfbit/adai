package com.adai.ops

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: MainActivity (opsdashboard's real launcher entry point) had zero test coverage.
 * Launches it for real via [createAndroidComposeRule] -- exercising `MainActivity.onCreate()` ->
 * `OpsAppScaffold`/`OpsNavHost` -> the real `AppViewModelProvider`/`AppContainer` DI graph
 * (including the real `LocalAdminAuthGate` -> `BiometricAdminAuthGate` composition-local wiring)
 * end to end, not a fake.
 *
 * `MainActivity` is a `FragmentActivity` (required for `ConfirmActionDialog`'s cast, see
 * `ConfirmDialogTestActivity`'s own doc comment), so it doesn't need that separate test
 * infrastructure here -- it already IS one. Deliberately doesn't click any admin action (that
 * would open a real `BiometricPrompt` this sandbox has no way to drive) -- only navigation
 * between top-level destinations is checked.
 *
 * NOTE: same sandbox limitation as the other opsdashboard tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here (this module's
 * debug APK can't be installed on this sandbox's emulator at all, see TECHNICAL_DEBT.md's
 * `wear-sdk` note).
 */
class MainActivityTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun launchingTheApp_showsTrainingSessionsAsTheStartDestination() {
        composeTestRule.onNodeWithText("Training Sessions").assertExists()
    }

    @Test
    fun navigatingToTheModelsTab_leavesTheSessionsScreen() {
        composeTestRule.onNodeWithText("Training Sessions").assertExists()

        composeTestRule.onNodeWithContentDescription("Models").performClick()

        composeTestRule.onNodeWithText("Training Sessions").assertDoesNotExist()
        // "Models" now appears at least twice: the nav-suite item's own label, plus the screen's
        // TopAppBar title -- onNodeWithText would be ambiguous, so just check both are present.
        assert(composeTestRule.onAllNodesWithText("Models").fetchSemanticsNodes().size >= 2) {
            "expected both the nav item's label and the Models screen's title to be showing"
        }
    }

    @Test
    fun navigatingToSettingsAndBack_returnsToTheStartDestination() {
        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        composeTestRule.onNodeWithText("Use one host for all services", substring = true).assertExists()
        composeTestRule.onNodeWithText("Save").assertExists()

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        composeTestRule.onNodeWithText("Training Sessions").assertExists()
    }
}
