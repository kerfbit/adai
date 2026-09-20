package com.adai.ops.ui.models

// @adai-status: experimental        (TD-196 — new)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-19


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-196: exercises ModelListScreen's register-model confirm-dialog flow — needs
 * ConfirmDialogTestActivity/FakeAdminAuthGate the same way ModelDetailScreenConfirmActionTest.kt
 * does for the admin actions there (see that file's own doc comment for why this needs to be a
 * separate file from ModelListScreenTest.kt's plain createComposeRule()).
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests — compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class ModelListScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun setContent(fakeService: FakeMnsApiService, authGate: FakeAdminAuthGate) {
        val viewModel = ModelListViewModel(ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()))
        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides authGate) {
                ModelListScreen(viewModel = viewModel, onOpenModel = {}, onOpenSettings = {})
            }
        }
    }

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun registerEncoder_confirmedWithSuccessfulAuth_registersTheModel() {
        val fakeService = FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto() })
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No models registered") }
        composeTestRule.onNodeWithContentDescription("Register model").performClick()
        composeTestRule.onNodeWithText("Register Encoder").performClick()
        composeTestRule.onNodeWithText("Model name").performTextInput("my-encoder")
        composeTestRule.onNodeWithText("d_model").performTextInput("128")
        composeTestRule.onNodeWithText("num_heads").performTextInput("4")
        composeTestRule.onNodeWithText("d_ff").performTextInput("512")
        composeTestRule.onNodeWithText("num_layers").performTextInput("3")
        composeTestRule.onNodeWithText("max_seq_length").performTextInput("256")
        composeTestRule.onNodeWithText("Register").performClick()

        // ConfirmActionDialog previews the literal HTTP call before anything is sent.
        composeTestRule.onNodeWithText("Register 'my-encoder'?").assertExists()
        composeTestRule.onNodeWithText("POST /models", substring = true).assertExists()

        composeTestRule.onNodeWithText("Register").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeService.registerModelCalls.isNotEmpty() }
        val body = fakeService.registerModelCalls.single()
        assert(body.model_name == "my-encoder" && body.kind == "encoder" && body.arch.d_model == 128L) {
            "unexpected registerModel call: $body"
        }
        composeTestRule.onNodeWithText("Register 'my-encoder'?").assertDoesNotExist()
    }

    @Test
    fun registerEncoder_cancellingConfirmDialog_doesNotRegister() {
        val fakeService = FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto() })
        val authGate = FakeAdminAuthGate()
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No models registered") }
        composeTestRule.onNodeWithContentDescription("Register model").performClick()
        composeTestRule.onNodeWithText("Register Encoder").performClick()
        composeTestRule.onNodeWithText("Model name").performTextInput("my-encoder")
        composeTestRule.onNodeWithText("d_model").performTextInput("128")
        composeTestRule.onNodeWithText("num_heads").performTextInput("4")
        composeTestRule.onNodeWithText("d_ff").performTextInput("512")
        composeTestRule.onNodeWithText("num_layers").performTextInput("3")
        composeTestRule.onNodeWithText("max_seq_length").performTextInput("256")
        composeTestRule.onNodeWithText("Register").performClick()
        composeTestRule.onNodeWithText("Cancel").performClick()

        composeTestRule.onNodeWithText("Register 'my-encoder'?").assertDoesNotExist()
        assert(authGate.authenticateCalls.isEmpty())
        assert(fakeService.registerModelCalls.isEmpty())
    }

    @Test
    fun registerChatbot_serverRejectsMismatch_surfacesErrorInSnackbar() {
        val fakeService = FakeMnsApiService(
            listModelsResponse = { _, _, _, _ -> ModelsResponseDto() },
            registerModelResponse = {
                Response.error(
                    409,
                    ResponseBody.create(
                        "application/json".toMediaType(),
                        "{\"error\":\"encoder and decoder have incompatible d_model\"}",
                    ),
                )
            },
        )
        val authGate = FakeAdminAuthGate(result = { AdminAuthResult.Success })
        setContent(fakeService, authGate)

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No models registered") }
        composeTestRule.onNodeWithContentDescription("Register model").performClick()
        composeTestRule.onNodeWithText("Register Chatbot").performClick()
        composeTestRule.onNodeWithText("Model name").performTextInput("my-bot")
        composeTestRule.onNodeWithText("Legacy").performClick()
        composeTestRule.onNodeWithText("d_model").performTextInput("128")
        composeTestRule.onNodeWithText("num_heads").performTextInput("4")
        composeTestRule.onNodeWithText("d_ff").performTextInput("512")
        composeTestRule.onNodeWithText("num_encoder_layers").performTextInput("2")
        composeTestRule.onNodeWithText("num_decoder_layers").performTextInput("2")
        composeTestRule.onNodeWithText("max_seq_length").performTextInput("256")
        composeTestRule.onNodeWithText("Register").performClick()
        composeTestRule.onNodeWithText("Register").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("incompatible d_model") }
    }
}
