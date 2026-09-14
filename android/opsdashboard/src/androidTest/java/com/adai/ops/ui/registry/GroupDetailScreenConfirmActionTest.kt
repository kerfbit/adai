package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.RunsResponseDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: exercises GroupDetailScreen's "Force release" admin-action confirm-dialog flow using
 * the FragmentActivity/FakeAdminAuthGate infrastructure built for
 * ModelDetailScreenConfirmActionTest (see that file's own doc comment for the full explanation
 * and for the representative coverage of AdminAuthResult's Cancelled/Failed branches, which are
 * screen-agnostic and not re-tested here).
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class GroupDetailScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun confirmingForceRelease_invokesForceReleaseAndClosesTheDialog() {
        val fakeRegistryService = FakeRegistryApiService(
            runsResponse = { RunsResponseDto(runs = mapOf("run-1" to listOf("a.jsonl", "b.jsonl"))) },
        )
        val viewModel = GroupDetailViewModel(
            group = "group-x",
            registryRepository = RegistryRepository(FakeApiClientProvider(fakeRegistryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })

        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides authGate) {
                GroupDetailScreen(group = "group-x", viewModel = viewModel, onBack = {})
            }
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("run-1") }
        composeTestRule.onNodeWithText("Release", substring = true).performClick()
        composeTestRule.onNodeWithText("Force-release run 'run-1'?").assertExists()

        composeTestRule.onNodeWithText("Force release").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.releaseCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.releaseCalls.single()
        assert(group == "group-x" && body.run_id == "" && body.files == listOf("a.jsonl", "b.jsonl")) {
            "unexpected release call: group=$group body=$body"
        }
        composeTestRule.onNodeWithText("Force-release run 'run-1'?").assertDoesNotExist()
    }
}
