package com.adai.ops.ui.admin

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.isToggleable
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.MetricsAdminConfigDto
import com.adai.ops.network.dto.MnsAdminConfigDto
import com.adai.ops.network.dto.RegistryAdminConfigDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: AdminScreen had zero coverage. Reuses AdminViewModelTest's own real-repository Fixture
 * shape (ModelRepository/RegistryRepository/MetricsRepository backed by the shared src/sharedTest
 * fakes).
 *
 * Deliberately out of scope for this pass, and more so than any prior screen: per this screen's
 * own doc comment, EVERY field edit here -- not just one admin action -- flows through
 * ConfirmActionDialog ("Every save flows through ConfirmActionDialog, which is itself gated by the
 * device-credential check"). ConfirmActionDialog does `LocalContext.current as FragmentActivity`
 * and reads `LocalAdminAuthGate.current` the instant it composes, which would crash a plain
 * createComposeRule() host the moment ANY field's edit-dialog "Next" button is clicked (that's
 * exactly what makes `pendingValue` non-null and composes ConfirmActionDialog) -- same
 * FragmentActivity/LocalAdminAuthGate infrastructure gap flagged for ModelDetailScreen/
 * SessionDetailScreen/GroupDetailScreen. So these tests only exercise: each section's read-only
 * rendering across its three states (populated / admin-disabled / error-with-no-config, including
 * the metrics-only "Not found" -> friendlier-message mapping); and the plain-AlertDialog edit
 * dialogs' own input validation (opening with the right title, the Int dialog's "Next" button
 * gated on parseable input) -- stopping BEFORE ever clicking "Next", which is the one action that
 * would compose ConfirmActionDialog.
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class AdminScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(
        mnsService: FakeMnsApiService = FakeMnsApiService(),
        registryService: FakeRegistryApiService = FakeRegistryApiService(),
        metricsService: FakeMetricsApiService = FakeMetricsApiService(),
    ) = AdminViewModel(
        modelRepository = ModelRepository(FakeApiClientProvider(mnsService), FakeSettingsRepository()),
        registryRepository = RegistryRepository(FakeApiClientProvider(registryService), FakeSettingsRepository()),
        metricsRepository = MetricsRepository(FakeApiClientProvider(metricsService), FakeSettingsRepository()),
    )

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    private fun <T> notFoundResponse(): Response<T> =
        Response.error(404, ResponseBody.create("application/json".toMediaType(), "{}"))

    @Test
    fun mnsSectionPopulated_showsRegistryUrlAndGroup() {
        val viewModel = viewModel(
            mnsService = FakeMnsApiService(
                getAdminConfigResponse = {
                    Response.success(MnsAdminConfigDto(admin_enabled = true, registry_url = "http://mns", registry_group = "group-a"))
                },
            ),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("http://mns") }
        composeTestRule.onNodeWithText("mns_server").assertExists()
        composeTestRule.onNodeWithText("registry_url").assertExists()
        composeTestRule.onNodeWithText("registry_group").assertExists()
        composeTestRule.onNodeWithText("group-a").assertExists()
    }

    @Test
    fun mnsSectionAdminDisabled_showsDisabledMessage() {
        val viewModel = viewModel(
            mnsService = FakeMnsApiService(getAdminConfigResponse = { Response.success(MnsAdminConfigDto(admin_enabled = false)) }),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            textIsShowing("Admin mutation disabled on this server (started with --admin-enabled=false).")
        }
    }

    @Test
    fun mnsSectionErrorWithNoConfig_showsErrorTextVerbatim() {
        val viewModel = viewModel(
            mnsService = FakeMnsApiService(getAdminConfigResponse = { throw IOException("connection refused") }),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("connection refused") }
    }

    @Test
    fun registrySectionPopulated_showsFtpFields() {
        val viewModel = viewModel(
            registryService = FakeRegistryApiService(
                getAdminConfigResponse = {
                    Response.success(RegistryAdminConfigDto(admin_enabled = true, ftp_token_ttl_minutes = 30, ftp_max_sessions_per_run = 4))
                },
            ),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("registry_server") }
        composeTestRule.onNodeWithText("ftp_token_ttl_minutes").assertExists()
        composeTestRule.onNodeWithText("30").assertExists()
        composeTestRule.onNodeWithText("ftp_max_sessions_per_run").assertExists()
        composeTestRule.onNodeWithText("4").assertExists()
    }

    @Test
    fun metricsSectionPopulated_showsFieldsIncludingTheBooleanSwitch() {
        val viewModel = viewModel(
            metricsService = FakeMetricsApiService(
                getAdminConfigResponse = {
                    Response.success(
                        MetricsAdminConfigDto(allow_control = true, max_live_sessions = 5, enable_prometheus = true),
                    )
                },
            ),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("metrics_api_server") }
        composeTestRule.onNodeWithText("max_live_sessions").assertExists()
        composeTestRule.onNodeWithText("5").assertExists()
        composeTestRule.onNodeWithText("enable_prometheus").assertExists()
        composeTestRule.onNode(isToggleable()).assertExists()
    }

    @Test
    fun metricsSectionNotFound_showsAdminRoutesDisabledFriendlyMessage() {
        // Distinct from mnsSectionErrorWithNoConfig above: the raw ApiResult.NotFound error
        // message is the terse "Not found" for every daemon, but only MetricsSection maps it to
        // a friendlier "admin routes not enabled" message (see AdminScreen.kt's MetricsSection).
        val viewModel = viewModel(
            metricsService = FakeMetricsApiService(getAdminConfigResponse = { notFoundResponse() }),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            textIsShowing("Admin routes not enabled on this server (allow_control=false at startup).")
        }
        composeTestRule.onNodeWithText("Not found").assertDoesNotExist()
    }

    @Test
    fun editIntDialog_opensAndGatesNextOnParseableInput() {
        val viewModel = viewModel(
            registryService = FakeRegistryApiService(
                getAdminConfigResponse = {
                    Response.success(RegistryAdminConfigDto(admin_enabled = true, ftp_token_ttl_minutes = 30))
                },
            ),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("ftp_token_ttl_minutes") }
        composeTestRule.onNodeWithContentDescription("Edit ftp_token_ttl_minutes").performClick()

        composeTestRule.onNodeWithText("Edit ftp_token_ttl_minutes").assertExists()
        composeTestRule.onNodeWithText("Next").assertIsEnabled()

        composeTestRule.onNode(hasSetTextAction()).performTextClearance()
        composeTestRule.onNode(hasSetTextAction()).performTextInput("not-a-number")
        composeTestRule.onNodeWithText("Next").assertIsNotEnabled()

        composeTestRule.onNode(hasSetTextAction()).performTextClearance()
        composeTestRule.onNode(hasSetTextAction()).performTextInput("99")
        composeTestRule.onNodeWithText("Next").assertIsEnabled()

        // Deliberately stop here -- clicking "Next" would compose ConfirmActionDialog, see the
        // class doc comment above.
        composeTestRule.onNodeWithText("Cancel").performClick()
        composeTestRule.onNodeWithText("Edit ftp_token_ttl_minutes").assertDoesNotExist()
    }

    @Test
    fun editStringDialog_opensAndAcceptsTypedInput() {
        val viewModel = viewModel(
            mnsService = FakeMnsApiService(
                getAdminConfigResponse = { Response.success(MnsAdminConfigDto(admin_enabled = true, registry_url = "http://old")) },
            ),
        )

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("http://old") }
        composeTestRule.onNodeWithContentDescription("Edit registry_url").performClick()

        composeTestRule.onNodeWithText("Edit registry_url").assertExists()
        composeTestRule.onNode(hasSetTextAction()).performTextClearance()
        composeTestRule.onNode(hasSetTextAction()).performTextInput("http://new")

        composeTestRule.onNodeWithText("http://new").assertExists()
        // Deliberately stop here -- clicking "Next" would compose ConfirmActionDialog.
        composeTestRule.onNodeWithText("Cancel").performClick()
    }

    @Test
    fun clickingSettingsIcon_invokesOnOpenSettings() {
        var settingsOpened = false
        val viewModel = viewModel()

        composeTestRule.setContent { AdminScreen(viewModel = viewModel, onOpenSettings = { settingsOpened = true }) }

        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        assert(settingsOpened) { "expected onOpenSettings() to have been invoked" }
    }
}
