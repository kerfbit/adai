package com.adai.ops.ui.models

// @adai-status: experimental        (TD-196 — kind filter chips + register-flow field-validation coverage added)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-19


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: ModelListScreen had zero coverage. Reuses ModelListViewModelTest's own
 * real-repository Fixture shape (ModelRepository backed by the shared src/sharedTest fakes)
 * rather than a real network. NOTE (see GroupListScreenTest for the full explanation): this
 * sandbox cannot install/run opsdashboard's debug APK on its phone emulator at all
 * (INSTALL_FAILED_MISSING_SHARED_LIBRARY: wear-sdk, no Wear-capable device available), so this
 * test is compile-verified and hand-checked against the real production code paths, not
 * confirmed passing on a device. Deliberately does not attempt ModelListViewModelTest's own
 * "a failed *second* poll keeps the first poll's models" scenario -- that needs virtual time
 * control (advanceTimeBy past the 8s poll interval) that a real-device instrumented test can't
 * do without an actual ~9s real-time wait, and the ViewModel-level test already covers it with
 * precise control; only the first-refresh outcomes are worth this file's own coverage.
 */
class ModelListScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(fakeService: FakeMnsApiService) = ModelListViewModel(
        ModelRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
    )

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun noModelsRegistered_showsPlaceholderMessage() {
        val vm = viewModel(FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto() }))

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No models registered") }
    }

    @Test
    fun modelsPopulated_showEachRowWithRoleAndState() {
        val vm = viewModel(
            FakeMnsApiService(
                listModelsResponse = { _, _, _, _ ->
                    ModelsResponseDto(
                        models = listOf(
                            ModelRecordDto(model_id = "1", model_name = "chatbot-main", role = "production", state = "production"),
                            ModelRecordDto(model_id = "2", model_name = "chatbot-candidate", role = "", state = "training"),
                        ),
                    )
                },
            ),
        )

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithText("chatbot-candidate").assertExists()
        // TD-196: ModelRow's supporting text now also shows kind — both rows default to "chatbot".
        composeTestRule.onNodeWithText("Role: production · Kind: chatbot").assertExists()
        // A blank role renders as "(none)" (see ModelListScreen.kt's ModelRow).
        composeTestRule.onNodeWithText("Role: (none) · Kind: chatbot").assertExists()
        composeTestRule.onNodeWithText("production").assertExists()
        composeTestRule.onNodeWithText("training").assertExists()
    }

    @Test
    fun fetchFailsWithNoPriorModels_showsFullScreenError() {
        val vm = viewModel(
            FakeMnsApiService(listModelsResponse = { _, _, _, _ -> throw IOException("connection refused") }),
        )

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Couldn't load data") }
        composeTestRule.onNodeWithText("connection refused").assertExists()
    }

    @Test
    fun clickingAModelRow_invokesOnOpenModelWithItsName() {
        val vm = viewModel(
            FakeMnsApiService(
                listModelsResponse = { _, _, _, _ ->
                    ModelsResponseDto(models = listOf(ModelRecordDto(model_id = "1", model_name = "chatbot-main")))
                },
            ),
        )
        var opened: String? = null

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = { opened = it }, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("chatbot-main") }
        composeTestRule.onNodeWithText("chatbot-main").performClick()

        assert(opened == "chatbot-main") { "expected onOpenModel(\"chatbot-main\"), got $opened" }
    }

    @Test
    fun clickingSettingsIcon_invokesOnOpenSettings() {
        val vm = viewModel(FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto() }))
        var settingsOpened = false

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = { settingsOpened = true })
        }

        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        assert(settingsOpened) { "expected onOpenSettings() to have been invoked" }
    }

    // ─────────────────────────────────────────────────────────────────────
    // TD-196: kind filter chips + register-model flow
    // ─────────────────────────────────────────────────────────────────────

    @Test
    fun clickingAKindChip_refetchesWithThatKindFilter() {
        val capturedKinds = mutableListOf<String?>()
        val vm = viewModel(FakeMnsApiService(listModelsResponse = { _, _, kind, _ -> capturedKinds += kind; ModelsResponseDto() }))

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = {})
        }
        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No models registered") }

        composeTestRule.onNodeWithText("Encoder").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { capturedKinds.contains("encoder") }
    }

    @Test
    fun clickingAddButton_opensKindPicker() {
        val vm = viewModel(FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto() }))

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = {})
        }

        composeTestRule.onNodeWithContentDescription("Register model").performClick()

        composeTestRule.onNodeWithText("Register a model").assertExists()
        composeTestRule.onNodeWithText("Register Encoder").assertExists()
        composeTestRule.onNodeWithText("Register Chatbot").assertExists()
    }

    @Test
    fun registerEncoderDialog_registerDisabledUntilAllFieldsAreValid() {
        val vm = viewModel(FakeMnsApiService(listModelsResponse = { _, _, _, _ -> ModelsResponseDto() }))

        composeTestRule.setContent {
            ModelListScreen(viewModel = vm, onOpenModel = {}, onOpenSettings = {})
        }
        composeTestRule.onNodeWithContentDescription("Register model").performClick()
        composeTestRule.onNodeWithText("Register Encoder").performClick()

        composeTestRule.onNodeWithText("Register").assertIsNotEnabled()
        composeTestRule.onNodeWithText("Model name").performTextInput("my-encoder")
        composeTestRule.onNodeWithText("d_model").performTextInput("128")
        composeTestRule.onNodeWithText("num_heads").performTextInput("4")
        composeTestRule.onNodeWithText("d_ff").performTextInput("512")
        composeTestRule.onNodeWithText("num_layers").performTextInput("3")
        composeTestRule.onNodeWithText("max_seq_length").performTextInput("256")
        composeTestRule.onNodeWithText("Register").assertIsEnabled()
    }

    // NOTE: the flow beyond this point (clicking "Register" opens ConfirmActionDialog, which
    // requires a real FragmentActivity for its biometric/PIN gate) needs
    // createAndroidComposeRule<ConfirmDialogTestActivity>() — this plain createComposeRule() host
    // would crash rendering ConfirmActionDialog. See ModelListScreenConfirmActionTest.kt, mirroring
    // ModelDetailScreenConfirmActionTest.kt's own reasoning for the same split.
}
