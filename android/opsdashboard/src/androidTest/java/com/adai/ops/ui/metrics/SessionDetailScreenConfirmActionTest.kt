package com.adai.ops.ui.metrics

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.onNodeWithContentDescriptionScrolled
import com.adai.ops.testutil.onNodeWithTextScrolled
import com.adai.ops.testutil.scrollToNode
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: exercises SessionDetailScreen's "End session" admin-action confirm-dialog flow using
 * the FragmentActivity/FakeAdminAuthGate infrastructure built for
 * ModelDetailScreenConfirmActionTest (see that file's own doc comment for the full explanation
 * and for the representative coverage of AdminAuthResult's Cancelled/Failed branches, which are
 * screen-agnostic and not re-tested here).
 *
 * Runs on the Medium_Phone_API_35 AVD (`./gradlew :opsdashboard:connectedDebugAndroidTest`); it was
 * only compiled until November 21, 2026, when the APK stopped requiring `wear-sdk`.
 */
class SessionDetailScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    // Scrolls the screen's LazyColumn to the text first: items below the fold aren't composed.
    private fun textIsShowing(text: String): Boolean =
        composeTestRule.scrollToNode(hasText(text, substring = true)) ||
            composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun confirmingEndSession_invokesEndSessionAndClosesTheDialog() {
        val fakeService = FakeMetricsApiService(sessionStatusResponse = { SessionStatusDto() })
        val viewModel = SessionDetailViewModel(
            sessionKey = "session-1",
            metricsRepository = MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
            settingsRepository = FakeSettingsRepository(),
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })

        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides authGate) {
                SessionDetailScreen(sessionKey = "session-1", viewModel = viewModel, onBack = {}, onEvicted = {})
            }
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("End session") }
        composeTestRule.onNodeWithTextScrolled("End session", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("End this session?").assertExists()

        composeTestRule.onNodeWithTextScrolled("End session").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.endSessionCalls.isNotEmpty() }
        assert(fakeService.endSessionCalls == listOf("session-1")) {
            "expected endSession('session-1'), got ${fakeService.endSessionCalls}"
        }
        composeTestRule.onNodeWithTextScrolled("End this session?").assertDoesNotExist()
    }
}
