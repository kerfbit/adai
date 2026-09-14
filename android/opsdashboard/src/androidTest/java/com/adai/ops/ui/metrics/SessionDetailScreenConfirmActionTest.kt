package com.adai.ops.ui.metrics

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
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
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class SessionDetailScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun textIsShowing(text: String): Boolean =
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
        composeTestRule.onNodeWithText("End session", substring = true).performClick()
        composeTestRule.onNodeWithText("End this session?").assertExists()

        composeTestRule.onNodeWithText("End session").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.endSessionCalls.isNotEmpty() }
        assert(fakeService.endSessionCalls == listOf("session-1")) {
            "expected endSession('session-1'), got ${fakeService.endSessionCalls}"
        }
        composeTestRule.onNodeWithText("End this session?").assertDoesNotExist()
    }
}
