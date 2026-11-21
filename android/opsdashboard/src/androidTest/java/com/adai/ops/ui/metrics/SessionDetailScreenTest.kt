package com.adai.ops.ui.metrics

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.performClick
import com.adai.ops.data.metrics.MetricsRepository
import com.adai.ops.network.dto.CurrentMetricsDto
import com.adai.ops.network.dto.SessionStatusDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMetricsApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.httpException
import com.adai.ops.testutil.onNodeWithContentDescriptionScrolled
import com.adai.ops.testutil.onNodeWithTextScrolled
import com.adai.ops.testutil.scrollToNode
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
 * Runs on the Medium_Phone_API_35 AVD (`./gradlew :opsdashboard:connectedDebugAndroidTest`); it was
 * only compiled until November 21, 2026, when the APK stopped requiring `wear-sdk`.
 */
class SessionDetailScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private fun viewModel(service: FakeMetricsApiService) = SessionDetailViewModel(
        sessionKey = "session-1",
        metricsRepository = MetricsRepository(FakeApiClientProvider(service), FakeSettingsRepository()),
        settingsRepository = FakeSettingsRepository(),
    )

    // Scrolls the screen's LazyColumn to the text first: items below the fold aren't composed.
    private fun textIsShowing(text: String): Boolean =
        composeTestRule.scrollToNode(hasText(text, substring = true)) ||
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
        composeTestRule.onNodeWithTextScrolled("3/10").assertExists()
        composeTestRule.onNodeWithTextScrolled("500/1000").assertExists()
        composeTestRule.onNodeWithTextScrolled("yes").assertExists()
        composeTestRule.onNodeWithTextScrolled("0.1234").assertExists()
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
        composeTestRule.onNodeWithTextScrolled("BLEU-4").assertDoesNotExist()
    }

    @Test
    fun generationQualityEnabled_showsBleuAndRougeMetrics() {
        val vm = viewModel(
            FakeMetricsApiService(
                sessionStatusResponse = { status() },
                // TD-380: BLEU is on the 0-100 scale, shown with two decimals.
                currentMetricsResponse = { current(bleu4 = 70.0) },
            ),
        )

        composeTestRule.setContent {
            SessionDetailScreen(sessionKey = "session-1", viewModel = vm, onBack = {}, onEvicted = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("BLEU-4") }
        composeTestRule.onNodeWithTextScrolled("70.00").assertExists()
        composeTestRule.onNodeWithTextScrolled("ROUGE-1").assertExists()
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
        composeTestRule.onNodeWithTextScrolled("connection refused").assertExists()
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
        composeTestRule.onNodeWithTextScrolled("End session", substring = true).assertIsEnabled()
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

        composeTestRule.onNodeWithContentDescriptionScrolled("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
    }
}
