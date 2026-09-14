package com.adai.ops.ui.admin

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: exercises AdminScreen's field-edit admin-action confirm-dialog flow (mns_server's
 * registry_url, representative of every field on this screen -- they all share the same
 * pencil-icon -> value-entry-dialog -> ConfirmActionDialog shape) using the FragmentActivity/
 * FakeAdminAuthGate infrastructure built for ModelDetailScreenConfirmActionTest (see that file's
 * own doc comment for the representative coverage of AdminAuthResult's Cancelled/Failed
 * branches, which are screen-agnostic and not re-tested here).
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class AdminScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun confirmingAFieldEdit_invokesPutAdminConfigAndClosesBothDialogs() {
        val fakeMnsService = FakeMnsApiService(
            getAdminConfigResponse = { Response.success(MnsAdminConfigDto(admin_enabled = true, registry_url = "http://old")) },
        )
        val viewModel = AdminViewModel(
            modelRepository = ModelRepository(FakeApiClientProvider(fakeMnsService), FakeSettingsRepository()),
            registryRepository = RegistryRepository(FakeApiClientProvider(FakeRegistryApiService()), FakeSettingsRepository()),
            metricsRepository = MetricsRepository(FakeApiClientProvider(FakeMetricsApiService()), FakeSettingsRepository()),
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })

        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides authGate) {
                AdminScreen(viewModel = viewModel, onOpenSettings = {})
            }
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("http://old") }
        composeTestRule.onNodeWithContentDescription("Edit registry_url").performClick()
        composeTestRule.onNode(hasSetTextAction()).performTextClearance()
        composeTestRule.onNode(hasSetTextAction()).performTextInput("http://new")
        composeTestRule.onNodeWithText("Next").performClick()

        composeTestRule.onNodeWithText("Update registry_url?").assertExists()
        composeTestRule.onNodeWithText("Save").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeMnsService.putAdminConfigCalls.isNotEmpty() }
        assert(fakeMnsService.putAdminConfigCalls.single().toString().contains("http://new")) {
            "expected the PUT body to carry the new registry_url, got ${fakeMnsService.putAdminConfigCalls.single()}"
        }
        composeTestRule.onNodeWithText("Update registry_url?").assertDoesNotExist()
        composeTestRule.onNodeWithText("Edit registry_url").assertDoesNotExist()
    }
}
