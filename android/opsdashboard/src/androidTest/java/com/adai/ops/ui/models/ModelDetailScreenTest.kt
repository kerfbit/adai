package com.adai.ops.ui.models

// @adai-status: experimental        (TD-196 — kind-aware design view coverage: encoder/decoder/world-model links)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-19


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.network.dto.ArchDto
import com.adai.ops.network.dto.ArtifactDto
import com.adai.ops.network.dto.ConnectionDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.TrainingHistoryEntryDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: ModelDetailScreen had zero coverage. Reuses ModelDetailViewModelTest's own
 * real-repository Fixture shape (ModelRepository backed by the shared src/sharedTest fakes)
 * rather than a real network.
 *
 * Deliberately scoped to the read-only detail rendering, admin-button enabled/disabled state
 * (derived from model.state), and the back button -- NOT the admin-action confirm-dialog flow
 * (clicking "Clear stale training lock"/"Retire candidate"/"Promote to production" opens
 * ConfirmActionDialog, which unconditionally does `LocalContext.current as FragmentActivity` and
 * reads `LocalAdminAuthGate.current` the moment it composes -- both true even before the dialog's
 * own confirm button is clicked). A plain createComposeRule()'s default test host activity is a
 * bare ComponentActivity, not a FragmentActivity, so that cast would crash immediately; there is
 * also no fake AdminAuthGate provided via CompositionLocalProvider yet. Testing that flow for
 * real needs its own small addition (a minimal androidTest-only FragmentActivity host +
 * createAndroidComposeRule<...>() + a FakeAdminAuthGate) that doesn't exist in this codebase yet
 * -- flagged as a followup (see TECHNICAL_DEBT.md) rather than built speculatively here, since
 * this sandbox can't run any of it live anyway (opsdashboard's `wear-sdk` install blocker) to
 * confirm it's wired correctly.
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class ModelDetailScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(model: ModelRecordDto?, error: String? = null) = ModelDetailViewModel(
        modelName = model?.model_name ?: "some-model",
        modelRepository = ModelRepository(
            FakeApiClientProvider(
                FakeMnsApiService(
                    getModelResponse = { _ ->
                        if (error != null) throw IOException(error)
                        Response.success(model ?: throw IllegalStateException("no model and no error configured"))
                    },
                ),
            ),
            FakeSettingsRepository(),
        ),
    )

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    private fun fullModel(state: String = "candidate") = ModelRecordDto(
        model_id = "model-1",
        model_name = "chatbot-main",
        role = "chat",
        state = state,
        run_id = "run-42",
        created_utc = "2026-01-01T00:00:00Z",
        updated_utc = "2026-01-02T00:00:00Z",
        artifact = ArtifactDto(host = "storage.example.com", path = "/models/chatbot-main.bin", format = "adai-native"),
        arch = ArchDto(d_model = 512, num_heads = 8, d_ff = 2048, num_encoder_layers = 6, num_decoder_layers = 6, max_seq_length = 128),
        training_history = listOf(
            TrainingHistoryEntryDto(run_id = "run-1", metrics_session_key = "sess-1", epochs = 10, final_loss = 0.5),
        ),
    )

    @Test
    fun modelDetails_showAllFields() {
        val vm = viewModel(fullModel())

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("run-42").assertExists()
        composeTestRule.onNodeWithText("2026-01-01T00:00:00Z").assertExists()
        composeTestRule.onNodeWithText("512").assertExists()
        composeTestRule.onNodeWithText("8").assertExists()
        composeTestRule.onNodeWithText("2048").assertExists()
        composeTestRule.onNodeWithText("6 / 6").assertExists()
        composeTestRule.onNodeWithText("128").assertExists()
        composeTestRule.onNodeWithText("storage.example.com").assertExists()
        composeTestRule.onNodeWithText("/models/chatbot-main.bin").assertExists()
        composeTestRule.onNodeWithText("adai-native").assertExists()
    }

    @Test
    fun blankOptionalFields_renderTheirFallbackText() {
        val vm = viewModel(
            fullModel().copy(role = "", run_id = "", artifact = ArtifactDto(host = "", path = "", format = "adai-native")),
        )

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        // Role and Run ID both fall back to "(none)" (see ModelDetailContent's DetailRow calls) --
        // onNodeWithText requires exactly one match, so a plain onNodeWithText("(none)") would
        // itself fail here; assert the count directly instead.
        val noneCount = composeTestRule.onAllNodesWithText("(none)").fetchSemanticsNodes().size
        assert(noneCount >= 2) { "expected Role and Run ID to both fall back to \"(none)\", found $noneCount occurrence(s)" }
        composeTestRule.onNodeWithText("(local)").assertExists()
    }

    @Test
    fun trainingHistory_rendersEachEntry() {
        val vm = viewModel(fullModel())

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Training History") }
        composeTestRule.onNodeWithText("Run run-1").assertExists()
        composeTestRule.onNodeWithText("epochs=10 loss=0.5 session=sess-1").assertExists()
    }

    @Test
    fun noTrainingHistory_sectionIsNotShown() {
        val vm = viewModel(fullModel().copy(training_history = emptyList()))

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("Training History").assertDoesNotExist()
    }

    @Test
    fun trainingState_onlyClearLockButtonEnabled() {
        val vm = viewModel(fullModel(state = "training"))

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("Clear stale training lock", substring = true).assertIsEnabled()
        composeTestRule.onNodeWithText("Retire candidate", substring = true).assertIsNotEnabled()
        composeTestRule.onNodeWithText("Promote to production", substring = true).assertIsNotEnabled()
    }

    @Test
    fun candidateState_retireAndPromoteButtonsEnabled() {
        val vm = viewModel(fullModel(state = "candidate"))

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("Clear stale training lock", substring = true).assertIsNotEnabled()
        composeTestRule.onNodeWithText("Retire candidate", substring = true).assertIsEnabled()
        composeTestRule.onNodeWithText("Promote to production", substring = true).assertIsEnabled()
    }

    @Test
    fun productionState_allAdminButtonsDisabled() {
        val vm = viewModel(fullModel(state = "production"))

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("Clear stale training lock", substring = true).assertIsNotEnabled()
        composeTestRule.onNodeWithText("Retire candidate", substring = true).assertIsNotEnabled()
        composeTestRule.onNodeWithText("Promote to production", substring = true).assertIsNotEnabled()
    }

    @Test
    fun fetchFailsWithNoPriorModel_showsFullScreenError() {
        val vm = viewModel(model = null, error = "connection refused")

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Couldn't load data") }
        composeTestRule.onNodeWithText("connection refused").assertExists()
    }

    @Test
    fun clickingBack_invokesOnBack() {
        val vm = viewModel(fullModel())
        var backInvoked = false

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = { backInvoked = true })
        }

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
    }

    // ─────────────────────────────────────────────────────────────────────
    // TD-196: kind-aware design view — encoder/decoder/world-model links
    // ─────────────────────────────────────────────────────────────────────

    @Test
    fun linkedChatbot_showsEncoderAndDecoderRowsInsteadOfInlineArch() {
        val vm = viewModel(
            fullModel().copy(connection = ConnectionDto(encoder_name = "my-enc", decoder_name = "my-dec")),
        )

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("my-enc").assertExists()
        composeTestRule.onNodeWithText("my-dec").assertExists()
        // The legacy inline d_model row must not render for a linked chatbot.
        composeTestRule.onNodeWithText("512").assertDoesNotExist()
    }

    @Test
    fun clickingLinkedEncoderRow_invokesOnOpenModelWithItsName() {
        val vm = viewModel(
            fullModel().copy(connection = ConnectionDto(encoder_name = "my-enc", decoder_name = "my-dec")),
        )
        var opened: String? = null

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {}, onOpenModel = { opened = it })
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("my-enc") }
        composeTestRule.onNodeWithText("my-enc").performClick()

        assert(opened == "my-enc") { "expected onOpenModel(\"my-enc\"), got $opened" }
    }

    @Test
    fun chatbotWithLinkedWorldModel_showsWorldModelSectionAndDetachEnabled() {
        val vm = viewModel(
            fullModel().copy(
                connection = ConnectionDto(
                    world_model_name = "my-wm", world_model_inject_every_n_layers = 2,
                    hippocampal_memory_enabled = true, hippocampal_memory_capacity = 256,
                ),
            ),
        )

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("my-wm") }
        composeTestRule.onNodeWithText("my-wm").assertExists()
        composeTestRule.onNodeWithText("Link world model", substring = true).assertIsEnabled()
        composeTestRule.onNodeWithText("Detach world model", substring = true).assertIsEnabled()
    }

    @Test
    fun chatbotWithoutWorldModel_detachButtonDisabled() {
        val vm = viewModel(fullModel())

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("Link world model", substring = true).assertIsEnabled()
        composeTestRule.onNodeWithText("Detach world model", substring = true).assertIsNotEnabled()
    }

    @Test
    fun encoderKind_showsOwnArchitectureAndNoWorldModelSection() {
        val vm = viewModel(fullModel().copy(kind = "encoder", connection = ConnectionDto()))

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("512").assertExists()
        composeTestRule.onNodeWithText("World Model").assertDoesNotExist()
    }

    @Test
    fun worldModelKind_showsSigregFields() {
        val vm = viewModel(
            fullModel().copy(kind = "world_model", connection = ConnectionDto(sigreg_lambda = 2.5f, sigreg_num_sketches = 32)),
        )

        composeTestRule.setContent {
            ModelDetailScreen(modelName = "chatbot-main", viewModel = vm, onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("model-1") }
        composeTestRule.onNodeWithText("2.5").assertExists()
        composeTestRule.onNodeWithText("32").assertExists()
    }
}
