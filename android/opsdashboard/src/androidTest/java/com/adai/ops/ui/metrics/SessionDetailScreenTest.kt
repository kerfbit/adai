package com.adai.ops.ui.metrics

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.CurrentMetricsDto
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.httpException
import java.io.IOException
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: SessionDetailScreen had zero coverage. Reuses SessionDetailViewModelTest's own
 * real-repository Fixture shape (MetricsRepository backed by the shared src/sharedTest fakes)
 * rather than a real network. AdaptivePoller.run() calls poll() immediately before any delay
 * (confirmed by reading AdaptivePoller.kt directly) -- same "first tick lands synchronously
 * enough for waitUntil() to observe" shape as the FixedIntervalPoller-based list screens, so no
 * virtual time control is needed here either, including for the eviction test below (a 404 on
 * the very first tick makes AdaptivePoller.run() return immediately, with zero delay, since
 * EVICTED short-circuits the loop before ever reaching its delay() call).
 *
 * Deliberately out of scope for this pass: the "End session" admin-action confirm-dialog flow --
 * same ConfirmActionDialog/FragmentActivity/LocalAdminAuthGate infrastructure gap already flagged
 * for ModelDetailScreenTest (see TECHNICAL_DEBT.md and that file's own doc comment for the full
 * explanation) -- only the button's default-enabled rendering is checked here, not the click.
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class SessionDetailScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(service: FakeMetricsApiService) = SessionDetailViewModel(
        sessionKey = "session-1",
        metricsRepository = MetricsRepository(FakeApiClientProvider(service), FakeSettingsRepository()),
        settingsRepository = FakeSettingsRepository(),
    )

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    private fun status(
        epoch: Int = 3,
        totalEpochs: Int = 10,
        sample: Int = 500,
        totalSamples: Int = 1000,
        effectiveTraining: Boolean = true,
        stale: Boolean = false,
    ) = SessionStatusDto(
        current_epoch = epoch,
        total_epochs = totalEpochs,
        current_sample = sample,
        total_samples = totalSamples,
        progress_percent = 50.0,
        samples_per_second = 0.0, // avoid AdaptivePoller's interval re-tuning entirely
        estimated_time_remaining_seconds = 120.0,
        effective_is_training = effectiveTraining,
        is_stale = stale,
    )

    private fun current(bleu4: Double = -1.0) = CurrentMetricsDto(
        current_loss = 0.1234,
        current_learning_rate = 0.000123,
        current_bleu4 = bleu4,
        current_rouge1 = 0.6,
        current_rouge2 = 0.4,
        current_rougeL = 0.5,
    )

    @Test
    fun liveSession_rendersPollerStatusProgressAndMetrics() {
        val vm = viewModel(
            FakeMetricsApiService(
                sessionStatusResponse = { status() },
                currentMetricsResponse = { current() },
            ),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Live · polling every 2000ms") }
        composeTestRule.onNodeWithText("3/10").assertExists()
        composeTestRule.onNodeWithText("500/1000").assertExists()
        composeTestRule.onNodeWithText("yes").assertExists()
        composeTestRule.onNodeWithText("0.1234").assertExists()
    }

    @Test
    fun staleAndNotEffectivelyTraining_showsStaleReason() {
        val vm = viewModel(
            FakeMetricsApiService(
                sessionStatusResponse = { status(effectiveTraining = false, stale = true) },
                currentMetricsResponse = { current() },
            ),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("no — stale") }
    }

    @Test
    fun generationQualityDisabled_showsDisabledMessage() {
        val vm = viewModel(
            FakeMetricsApiService(
                sessionStatusResponse = { status() },
                currentMetricsResponse = { current(bleu4 = -1.0) },
            ),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            textIsShowing("Disabled (ENABLE_GENERATION_QUALITY_METRICS=false)")
        }
        composeTestRule.onNodeWithText("BLEU-4").assertDoesNotExist()
    }

    @Test
    fun generationQualityEnabled_showsBleuAndRougeMetrics() {
        val vm = viewModel(
            FakeMetricsApiService(
                sessionStatusResponse = { status() },
                currentMetricsResponse = { current(bleu4 = 0.7) },
            ),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("BLEU-4") }
        composeTestRule.onNodeWithText("0.7000").assertExists()
        composeTestRule.onNodeWithText("ROUGE-1").assertExists()
    }

    @Test
    fun fetchFailsWithNoPriorStatus_showsFullScreenError() {
        val vm = viewModel(
            FakeMetricsApiService(sessionStatusResponse = { throw IOException("connection refused") }),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("Couldn't load data") }
        composeTestRule.onNodeWithText("connection refused").assertExists()
    }

    @Test
    fun sessionEvicted_invokesOnEvictedCallback() {
        val vm = viewModel(
            FakeMetricsApiService(sessionStatusResponse = { throw httpException(404) }),
        )
        var evicted = false

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = { evicted = true })
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { evicted }
    }

    @Test
    fun endSessionButton_visibleAndEnabledByDefault() {
        val vm = viewModel(
            FakeMetricsApiService(sessionStatusResponse = { status() }, currentMetricsResponse = { current() }),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("End session") }
        composeTestRule.onNodeWithText("End session", substring = true).assertIsEnabled()
    }

    @Test
    fun clickingBack_invokesOnBack() {
        val vm = viewModel(
            FakeMetricsApiService(sessionStatusResponse = { status() }, currentMetricsResponse = { current() }),
        )
        var backInvoked = false

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = { backInvoked = true }, onEvicted = {})
        }

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
    }
}
