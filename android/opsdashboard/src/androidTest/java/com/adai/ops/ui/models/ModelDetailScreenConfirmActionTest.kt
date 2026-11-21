package com.adai.ops.ui.models

// @adai-status: experimental        (TD-199/TD-200 review fix — asserts the link-world-model preview shows the real chatbot name, not a "{name}" placeholder)
// @adai-version: 0.2.1
// @adai-reviewed: 2026-09-19


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ArchDto
import com.adai.ops.network.dto.ArtifactDto
import com.adai.ops.network.dto.ConnectionDto
import com.adai.ops.network.dto.LinkWorldModelResultDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.onNodeWithContentDescriptionScrolled
import com.adai.ops.testutil.onNodeWithTextScrolled
import com.adai.ops.testutil.scrollToNode
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: exercises ModelDetailScreen's "Clear stale training lock" admin-action confirm-dialog
 * flow -- deliberately excluded from ModelDetailScreenTest itself (see that file's own doc
 * comment) pending the FragmentActivity/FakeAdminAuthGate infrastructure this file now uses
 * (com.adai.ops.testutil.ConfirmDialogTestActivity/FakeAdminAuthGate).
 *
 * This is also the representative test for [AdminAuthResult]'s branches -- ConfirmActionDialog's
 * own auth-result handling (Success/Cancelled/Failed/Unavailable) is screen-agnostic, so the
 * other four screens' own new confirm-dialog tests (SessionDetailScreen/GroupDetailScreen/
 * AdminScreen/TrainerScreen) only re-verify the Success happy path rather than re-testing every
 * branch redundantly.
 *
 * Runs on the Medium_Phone_API_35 AVD (`./gradlew :opsdashboard:connectedDebugAndroidTest`); it was
 * only compiled until November 21, 2026, when the APK stopped requiring `wear-sdk`.
 */
class ModelDetailScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun trainingModel() = ModelRecordDto(
        model_id = "model-1",
        model_name = "chatbot-main",
        state = "training",
        run_id = "run-42",
        artifact = ArtifactDto(),
        arch = ArchDto(),
    )

    private fun setContent(fakeService: FakeMnsApiService, authGate: FakeAdminAuthGate) {
        val viewModel = ModelDetailViewModel(
            modelName = "chatbot-main",
            modelRepository = ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
        )
        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides authGate) {
                ModelDetailScreen(modelName = "chatbot-main", viewModel = viewModel, onBack = {})
            }
        }
    }

    // Scrolls the screen's LazyColumn to the text first: items below the fold aren't composed.
    private fun textIsShowing(text: String): Boolean =
        composeTestRule.scrollToNode(hasText(text, substring = true)) ||
            composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun confirmingWithSuccessfulAuth_invokesTheAdminActionAndClosesTheDialog() {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(trainingModel()) },
            setStateResponse = { _, _ -> Response.success(trainingModel().copy(state = "candidate")) },
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock?").assertExists()

        composeTestRule.onNodeWithTextScrolled("Clear lock").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.setStateCalls.isNotEmpty() }
        val (name, body) = fakeService.setStateCalls.single()
        assert(name == "chatbot-main" && body.state == "candidate" && body.run_id == "run-42") {
            "unexpected setState call: name=$name body=$body"
        }
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock?").assertDoesNotExist()
        assert(authGate.authenticateCalls == listOf("Clear stale training lock?")) {
            "expected authenticate() to have been called with the dialog's own title as the reason"
        }
    }

    @Test
    fun cancellingTheDialog_doesNotAuthenticateOrInvokeTheAdminAction() {
        val fakeService = FakeMnsApiService(getModelResponse = { Response.success(trainingModel()) })
        val authGate = FakeAdminAuthGate()
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("Cancel").performClick()

        composeTestRule.onNodeWithTextScrolled("Clear stale training lock?").assertDoesNotExist()
        assert(authGate.authenticateCalls.isEmpty()) { "cancelling before confirming must not authenticate at all" }
        assert(fakeService.setStateCalls.isEmpty())
    }

    @Test
    fun authGateCancelled_leavesTheDialogOpenWithoutInvokingTheAdminAction() {
        val fakeService = FakeMnsApiService(getModelResponse = { Response.success(trainingModel()) })
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Cancelled })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("Clear lock").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { authGate.authenticateCalls.isNotEmpty() }
        // Cancelled is "not an error" (see AdminAuthResult's doc comment) -- the dialog is
        // neither dismissed nor does it show an error, just sits there for another attempt.
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock?").assertExists()
        assert(fakeService.setStateCalls.isEmpty())
    }

    @Test
    fun authGateFailed_showsTheErrorMessageInsideTheStillOpenDialog() {
        val fakeService = FakeMnsApiService(getModelResponse = { Response.success(trainingModel()) })
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Failed("sensor error") })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("Clear lock").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("sensor error") }
        composeTestRule.onNodeWithTextScrolled("Clear stale training lock?").assertExists()
        assert(fakeService.setStateCalls.isEmpty())
    }

    // ─────────────────────────────────────────────────────────────────────
    // TD-196: Link World Model / Detach World Model
    // ─────────────────────────────────────────────────────────────────────

    private fun chatbotModel(connection: ConnectionDto = ConnectionDto()) = ModelRecordDto(
        model_id = "model-1", model_name = "chatbot-main", kind = "chatbot", state = "production",
        artifact = ArtifactDto(), arch = ArchDto(), connection = connection,
    )

    @Test
    fun linkWorldModel_confirmedWithSuccessfulAuth_invokesLinkAndClosesDialog() {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(chatbotModel()) },
            listModelsResponse = { _, _, _, _ -> ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "wm-1", model_name = "my-wm", kind = "world_model"))) },
            linkWorldModelResponse = { _, body -> Response.success(LinkWorldModelResultDto(status = "ok", world_model_name = body.world_model_name)) },
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Link world model", substring = true).performClick()
        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("(unassigned)") }
        composeTestRule.onNodeWithTextScrolled("(unassigned)").performClick()
        composeTestRule.onNodeWithTextScrolled("my-wm").performClick()
        composeTestRule.onNodeWithTextScrolled("Continue").performClick()

        composeTestRule.onNodeWithTextScrolled("Link world model?").assertExists()
        // TD-199/TD-200 review fix: the preview must show the real chatbot name, not a literal
        // unsubstituted "{name}" placeholder.
        composeTestRule.onNodeWithTextScrolled("POST /models/chatbot-main/link-world-model", substring = true).assertExists()
        composeTestRule.onAllNodesWithText("{name}", substring = true).assertCountEquals(0)
        composeTestRule.onNodeWithTextScrolled("Link").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.linkWorldModelCalls.isNotEmpty() }
        val (name, body) = fakeService.linkWorldModelCalls.single()
        assert(name == "chatbot-main" && body.world_model_name == "my-wm") {
            "unexpected linkWorldModel call: name=$name body=$body"
        }
        composeTestRule.onNodeWithTextScrolled("Link world model?").assertDoesNotExist()
    }

    @Test
    fun linkWorldModel_serverRejectsMismatch_surfacesErrorMessage() {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(chatbotModel()) },
            listModelsResponse = { _, _, _, _ -> ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "wm-1", model_name = "my-wm", kind = "world_model"))) },
            linkWorldModelResponse = { _, _ ->
                Response.error(409, ResponseBody.create("application/json".toMediaType(), "{\"error\":\"d_model mismatch\"}"))
            },
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Link world model", substring = true).performClick()
        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("(unassigned)") }
        composeTestRule.onNodeWithTextScrolled("(unassigned)").performClick()
        composeTestRule.onNodeWithTextScrolled("my-wm").performClick()
        composeTestRule.onNodeWithTextScrolled("Continue").performClick()
        composeTestRule.onNodeWithTextScrolled("Link").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("d_model mismatch") }
    }

    @Test
    fun detachWorldModel_confirmedWithSuccessfulAuth_sendsEmptyWorldModelName() {
        val fakeService = FakeMnsApiService(
            getModelResponse = { Response.success(chatbotModel(ConnectionDto(world_model_name = "my-wm"))) },
            linkWorldModelResponse = { _, body -> Response.success(LinkWorldModelResultDto(status = "ok", world_model_name = body.world_model_name)) },
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithTextScrolled("Detach world model", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("Detach world model?").assertExists()
        composeTestRule.onNodeWithTextScrolled("Detach").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.linkWorldModelCalls.isNotEmpty() }
        val (name, body) = fakeService.linkWorldModelCalls.single()
        assert(name == "chatbot-main" && body.world_model_name == "") {
            "unexpected linkWorldModel (detach) call: name=$name body=$body"
        }
    }
}
