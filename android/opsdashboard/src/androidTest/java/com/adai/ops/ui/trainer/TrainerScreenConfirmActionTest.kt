package com.adai.ops.ui.trainer

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.trainer.TrainerRepository
import com.adai.ops.network.dto.TrainerStatusDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.FakeTrainerApiService
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: exercises TrainerScreen's "Pause" control-action confirm-dialog flow (representative
 * of Checkpoint/Resume too, and of the config-field edits, which share the same
 * ConfirmActionDialog gate) using the FragmentActivity/FakeAdminAuthGate infrastructure built
 * for ModelDetailScreenConfirmActionTest (see that file's own doc comment for the representative
 * coverage of AdminAuthResult's Cancelled/Failed branches, which are screen-agnostic and not
 * re-tested here).
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class TrainerScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun confirmingPause_invokesPauseAndClosesTheDialog() {
        val fakeService = FakeTrainerApiService(
            getStatusResponse = { Response.success(TrainerStatusDto(phase = "training", paused = false)) },
        )
        val viewModel = TrainerViewModel(TrainerRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()))
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })

        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides authGate) {
                TrainerScreen(viewModel = viewModel, onOpenSettings = {})
            }
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("training") }
        composeTestRule.onNodeWithText("Pause", substring = true).performClick()
        composeTestRule.onNodeWithText("Pause training?").assertExists()

        composeTestRule.onNodeWithText("Pause").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.pauseCallCount > 0 }
        assert(fakeService.pauseCallCount == 1) { "expected pause() to be called exactly once, got ${fakeService.pauseCallCount}" }
        composeTestRule.onNodeWithText("Pause training?").assertDoesNotExist()
    }
}
