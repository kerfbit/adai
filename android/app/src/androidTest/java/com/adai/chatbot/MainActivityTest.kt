package com.adai.chatbot

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: MainActivity (the app's real launcher entry point) had zero test coverage. Launches it
 * for real via [createAndroidComposeRule] -- exercising `MainActivity.onCreate()` ->
 * `AdaiNavHost` -> the real `AppViewModelProvider`/`AppContainer` DI graph end to end, not a
 * fake -- and checks that navigating between the two top-level destinations it wires
 * (conversation list <-> settings) actually works.
 *
 * Deliberately does not assert on the conversation list's specific contents (empty state vs.
 * populated), since this activity is backed by the real, shared on-disk database other
 * instrumented tests in this module also touch (see AppContainerTest's own doc comment) -- only
 * that the screen renders and both directions of navigation work, which holds regardless of
 * what's in the list.
 */
class MainActivityTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun launchingTheApp_showsTheConversationListAndNavigatesToSettingsAndBack() {
        composeTestRule.onNodeWithText("adai Chatbot").assertExists()

        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        composeTestRule.onNodeWithText("Settings").assertExists()
        composeTestRule.onNodeWithText("Save").assertExists()

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        composeTestRule.onNodeWithText("adai Chatbot").assertExists()
    }
}
