package com.adai.ops.ui.metrics

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.SessionSummaryDto
import com.adai.ops.network.dto.SessionsResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: SessionListScreen had zero coverage. Reuses SessionListViewModelTest's own
 * real-repository Fixture shape (MetricsRepository backed by the shared src/sharedTest fakes)
 * rather than a real network. The idle-session hide/show filtering itself
 * (`visibleSessions`/`hiddenIdleCount` in SessionListScreen.kt) lives entirely in the composable,
 * not the ViewModel (only the `showIdle` flag flip is ViewModel-level, already covered by
 * SessionListViewModelTest) -- genuinely new coverage, not a duplicate of an existing test.
 * NOTE: same sandbox limitation as GroupListScreenTest/ModelListScreenTest -- compile-verified
 * and hand-checked against the real production code paths, not run on a device here (see
 * GroupListScreenTest's doc comment for the full explanation).
 */
class SessionListScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(fakeService: FakeMetricsApiService) = SessionListViewModel(
        MetricsRepository(FakeApiClientProvider(fakeService), FakeSettingsRepository()),
    )

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty()

    private fun session(key: String, isTraining: Boolean, epoch: Int = 3, totalEpochs: Int = 10, loss: Double = 0.1234) =
        SessionSummaryDto(key = key, is_training = isTraining, current_epoch = epoch, total_epochs = totalEpochs, current_loss = loss)

    @Test
    fun noSessions_showsPlaceholderMessage() {
        val vm = viewModel(FakeMetricsApiService(listSessionsResponse = { SessionsResponseDto() }))

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No sessions found") }
    }

    @Test
    fun onlyIdleSessions_hiddenByDefaultShowsHiddenCountMessage() {
        val vm = viewModel(
            FakeMetricsApiService(
                listSessionsResponse = {
                    SessionsResponseDto(
                        sessions = listOf(session("idle-1", isTraining = false), session("idle-2", isTraining = false)),
                    )
                },
            ),
        )

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            textIsShowing("No active sessions — 2 idle session(s) hidden. Tap 'Show idle' to view them.")
        }
        composeTestRule.onNodeWithText("idle-1").assertDoesNotExist()
    }

    @Test
    fun clickingShowIdle_revealsIdleSessionsAndFlipsButtonLabel() {
        val vm = viewModel(
            FakeMetricsApiService(
                listSessionsResponse = { SessionsResponseDto(sessions = listOf(session("idle-1", isTraining = false))) },
            ),
        )

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Show idle") }
        composeTestRule.onNodeWithText("Show idle").performClick()

        composeTestRule.onNodeWithText("idle-1").assertExists()
        composeTestRule.onNodeWithText("Idle").assertExists()
        composeTestRule.onNodeWithText("Hide idle").assertExists()
    }

    @Test
    fun mixedSessions_defaultViewShowsOnlyTrainingRowWithEpochAndLoss() {
        val vm = viewModel(
            FakeMetricsApiService(
                listSessionsResponse = {
                    SessionsResponseDto(
                        sessions = listOf(
                            session("training-1", isTraining = true, epoch = 3, totalEpochs = 10, loss = 0.1234),
                            session("idle-1", isTraining = false),
                        ),
                    )
                },
            ),
        )

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("training-1") }
        composeTestRule.onNodeWithText("Epoch 3/10 · loss 0.1234").assertExists()
        composeTestRule.onNodeWithText("Training").assertExists()
        composeTestRule.onNodeWithText("idle-1").assertDoesNotExist()
    }

    @Test
    fun fetchFailsWithNoPriorSessions_showsFullScreenError() {
        val vm = viewModel(
            FakeMetricsApiService(listSessionsResponse = { throw IOException("connection refused") }),
        )

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = {}, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Couldn't load data") }
        composeTestRule.onNodeWithText("connection refused").assertExists()
    }

    @Test
    fun clickingASessionRow_invokesOnOpenSessionWithItsKey() {
        val vm = viewModel(
            FakeMetricsApiService(
                listSessionsResponse = { SessionsResponseDto(sessions = listOf(session("training-1", isTraining = true))) },
            ),
        )
        var opened: String? = null

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = { opened = it }, onOpenSettings = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("training-1") }
        composeTestRule.onNodeWithText("training-1").performClick()

        assert(opened == "training-1") { "expected onOpenSession(\"training-1\"), got $opened" }
    }

    @Test
    fun clickingSettingsIcon_invokesOnOpenSettings() {
        val vm = viewModel(FakeMetricsApiService(listSessionsResponse = { SessionsResponseDto() }))
        var settingsOpened = false

        composeTestRule.setContent {
            SessionListScreen(viewModel = vm, onOpenSession = {}, onOpenSettings = { settingsOpened = true })
        }

        composeTestRule.onNodeWithContentDescription("Settings").performClick()

        assert(settingsOpened) { "expected onOpenSettings() to have been invoked" }
    }
}
