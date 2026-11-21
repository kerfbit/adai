package com.adai.ops.ui.trainer

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.isToggleable
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.trainer.TrainerRepository
import com.adai.ops.network.dto.TrainerAdminConfigDto
import com.adai.ops.network.dto.TrainerLogEntryDto
import com.adai.ops.network.dto.TrainerLogsResponseDto
import com.adai.ops.network.dto.TrainerStatusDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.FakeTrainerApiService
import com.adai.ops.testutil.onNodeWithContentDescriptionScrolled
import com.adai.ops.testutil.onNodeWithTextScrolled
import com.adai.ops.testutil.scrollToNode
import java.io.IOException
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048: TrainerScreen had zero coverage. Reuses TrainerViewModelTest's own real-repository
 * Fixture shape (TrainerRepository backed by the shared src/sharedTest fakes).
 *
 * Same infrastructure gap as AdminScreen (its closest sibling): every config field edit here
 * flows through ConfirmActionDialog too, so only the edit dialogs' own input validation is
 * exercised, stopping before "Next". The three control actions (Checkpoint/Pause/Resume) are
 * ALSO ConfirmActionDialog-gated -- only their default enabled/disabled rendering (derived from
 * status.phase/status.paused) is checked, not the click, same as the admin-action buttons on
 * ModelDetailScreen/SessionDetailScreen/GroupDetailScreen.
 *
 * Deliberately not attempted: a "status refresh succeeds once, then a later poll fails, so the
 * stale-data-with-error banner appears" scenario -- FixedIntervalPoller's real 5s interval would
 * need either virtual time control (unavailable at this level, see ModelListScreenTest's own
 * documented reason for skipping its analogous multi-tick scenario) or a slow, real-time wait.
 * Also not asserted: the exact formatted "Last checkpoint time" / log-entry timestamp text --
 * both use `SimpleDateFormat(..., Locale.getDefault())` against the device's default time zone,
 * so the formatted string isn't deterministic across environments; only that the row/entry
 * itself renders is checked.
 *
 * Runs on the Medium_Phone_API_35 AVD (`./gradlew :opsdashboard:connectedDebugAndroidTest`); it was
 * only compiled until November 21, 2026, when the APK stopped requiring `wear-sdk`.
 */
class TrainerScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(service: FakeTrainerApiService = FakeTrainerApiService()) =
        TrainerViewModel(TrainerRepository(FakeApiClientProvider(service), FakeSettingsRepository()))

    // Scrolls the screen's LazyColumn to the text first: items below the fold aren't composed.
    private fun textIsShowing(text: String): Boolean =
        composeTestRule.scrollToNode(hasText(text, substring = true)) ||
            composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun statusNotLoaded_showsErrorAndExplanatoryMessage() {
        val viewModel = viewModel(
            FakeTrainerApiService(getStatusResponse = { throw IOException("connection refused") }),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("connection refused") }
        composeTestRule.onNodeWithTextScrolled(
            "The trainer admin API is opt-in (TRAINER_ADMIN_ENABLED=false by default) and " +
                "only runs under `incremental_trainer serve`. Confirm it's enabled and " +
                "reachable at the host/port configured in Settings.",
        ).assertExists()
        // Exact label: "Checkpoint" as a substring also matches the "Checkpoints written" status row.
        composeTestRule.onNodeWithTextScrolled("Checkpoint now", substring = true).assertIsNotEnabled()
    }

    @Test
    fun statusPopulated_showsPhaseAndDetailFields() {
        val viewModel = viewModel(
            FakeTrainerApiService(
                getStatusResponse = {
                    Response.success(
                        TrainerStatusDto(
                            phase = "training",
                            paused = false,
                            model_name = "chatbot-main",
                            run_id = "run-1",
                            session_id = "session-1",
                            current_epoch = 3,
                            total_epochs = 10,
                            samples_trained_this_pass = 5000,
                            last_loss = 0.25,
                            best_loss = 0.2,
                            checkpoints_written = 4,
                            last_checkpoint_path = "/data/ckpt-4.bin",
                            last_checkpoint_time_unix = 1_700_000_000L,
                        ),
                    )
                },
            ),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("training") }
        composeTestRule.onNodeWithTextScrolled("chatbot-main").assertExists()
        composeTestRule.onNodeWithTextScrolled("run-1").assertExists()
        composeTestRule.onNodeWithTextScrolled("session-1").assertExists()
        composeTestRule.onNodeWithTextScrolled("3 / 10").assertExists()
        composeTestRule.onNodeWithTextScrolled("5000").assertExists()
        composeTestRule.onNodeWithTextScrolled("0.25").assertExists()
        composeTestRule.onNodeWithTextScrolled("0.2").assertExists()
        composeTestRule.onNodeWithTextScrolled("/data/ckpt-4.bin").assertExists()
        composeTestRule.onNodeWithTextScrolled("Last checkpoint time").assertExists()
        composeTestRule.onNodeWithTextScrolled("paused").assertDoesNotExist()
    }

    @Test
    fun pausedStatus_showsPausedBadgeAndCorrectButtonState() {
        val viewModel = viewModel(
            FakeTrainerApiService(getStatusResponse = { Response.success(TrainerStatusDto(phase = "training", paused = true)) }),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("paused") }
        composeTestRule.onNodeWithTextScrolled("Pause", substring = true).assertIsNotEnabled()
        composeTestRule.onNodeWithTextScrolled("Resume", substring = true).assertIsEnabled()
    }

    @Test
    fun idlePhase_disablesCheckpointButNotPause() {
        val viewModel = viewModel(
            FakeTrainerApiService(getStatusResponse = { Response.success(TrainerStatusDto(phase = "idle", paused = false)) }),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("idle") }
        // Exact label: "Checkpoint" as a substring also matches the "Checkpoints written" status row.
        composeTestRule.onNodeWithTextScrolled("Checkpoint now", substring = true).assertIsNotEnabled()
        composeTestRule.onNodeWithTextScrolled("Pause", substring = true).assertIsEnabled()
        composeTestRule.onNodeWithTextScrolled("Resume", substring = true).assertIsNotEnabled()
    }

    @Test
    fun configNotLoaded_showsErrorText() {
        val viewModel = viewModel(
            FakeTrainerApiService(getAdminConfigResponse = { throw IOException("config unreachable") }),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("config unreachable") }
    }

    @Test
    fun configPopulated_showsAllFourFields() {
        val viewModel = viewModel(
            FakeTrainerApiService(
                getAdminConfigResponse = {
                    Response.success(
                        TrainerAdminConfigDto(
                            auto_save_enabled = true,
                            auto_save_every_samples = 1000,
                            auto_save_every_minutes = 15,
                            max_sessions_to_keep = 5,
                        ),
                    )
                },
            ),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("auto_save_enabled") }
        composeTestRule.onNode(isToggleable()).assertExists()
        composeTestRule.onNodeWithTextScrolled("auto_save_every_samples").assertExists()
        composeTestRule.onNodeWithTextScrolled("1000").assertExists()
        composeTestRule.onNodeWithTextScrolled("auto_save_every_minutes").assertExists()
        composeTestRule.onNodeWithTextScrolled("15").assertExists()
        composeTestRule.onNodeWithTextScrolled("max_sessions_to_keep").assertExists()
        composeTestRule.onNodeWithTextScrolled("5").assertExists()
    }

    @Test
    fun editIntDialog_opensAndGatesNextOnParseableInput() {
        val viewModel = viewModel(
            FakeTrainerApiService(
                getAdminConfigResponse = { Response.success(TrainerAdminConfigDto(auto_save_every_samples = 1000)) },
            ),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("auto_save_every_samples") }
        composeTestRule.onNodeWithContentDescriptionScrolled("Edit auto_save_every_samples").performClick()

        composeTestRule.onNodeWithTextScrolled("Edit auto_save_every_samples").assertExists()
        composeTestRule.onNodeWithTextScrolled("Next").assertIsEnabled()

        composeTestRule.onNode(hasSetTextAction()).performTextClearance()
        composeTestRule.onNode(hasSetTextAction()).performTextInput("not-a-number")
        composeTestRule.onNodeWithTextScrolled("Next").assertIsNotEnabled()

        composeTestRule.onNode(hasSetTextAction()).performTextClearance()
        composeTestRule.onNode(hasSetTextAction()).performTextInput("2000")
        composeTestRule.onNodeWithTextScrolled("Next").assertIsEnabled()

        // Deliberately stop here -- clicking "Next" would compose ConfirmActionDialog, see the
        // class doc comment above.
        composeTestRule.onNodeWithTextScrolled("Cancel").performClick()
        composeTestRule.onNodeWithTextScrolled("Edit auto_save_every_samples").assertDoesNotExist()
    }

    @Test
    fun activityLogEmpty_showsNoActivityMessage() {
        val viewModel = viewModel(FakeTrainerApiService(getLogsResponse = { Response.success(TrainerLogsResponseDto()) }))

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Activity Log") }
        composeTestRule.onNodeWithTextScrolled("No activity yet.").assertExists()
    }

    @Test
    fun activityLogPopulated_rendersEachEntrysMessage() {
        val viewModel = viewModel(
            FakeTrainerApiService(
                getLogsResponse = {
                    Response.success(
                        TrainerLogsResponseDto(
                            entries = listOf(
                                TrainerLogEntryDto(id = 1, level = "info", message = "training started"),
                                TrainerLogEntryDto(id = 2, level = "warn", message = "gradient spike detected"),
                                TrainerLogEntryDto(id = 3, level = "error", message = "checkpoint write failed"),
                            ),
                        ),
                    )
                },
            ),
        )

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = {}) }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("checkpoint write failed") }
        composeTestRule.onNodeWithTextScrolled("training started").assertExists()
        composeTestRule.onNodeWithTextScrolled("gradient spike detected").assertExists()
        composeTestRule.onNodeWithTextScrolled("No activity yet.").assertDoesNotExist()
    }

    @Test
    fun clickingSettingsIcon_invokesOnOpenSettings() {
        var settingsOpened = false
        val viewModel = viewModel()

        composeTestRule.setContent { TrainerScreen(viewModel = viewModel, onOpenSettings = { settingsOpened = true }) }

        composeTestRule.onNodeWithContentDescriptionScrolled("Settings").performClick()

        assert(settingsOpened) { "expected onOpenSettings() to have been invoked" }
    }
}
