package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-20


import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import java.io.IOException
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: GroupDetailScreen had zero coverage. Reuses GroupDetailViewModelTest's own
 * real-repository Fixture shape (RegistryRepository/ModelRepository backed by the shared
 * src/sharedTest fakes). FakeRegistryApiService gained a configurable `runsResponse` here (it was
 * previously hardcoded to always return an empty RunsResponseDto) so the "Claimed by run" section
 * could actually be exercised.
 *
 * TD-205: Assign and both Fetch dialogs are now ConfirmActionDialog-gated (retrofitted alongside
 * the new Unassign/Delete/Manual-add/Upload/Migrate/Segment actions, all gated from the start) —
 * this file only covers their own input-validation-gated confirm-button *enablement*, not the
 * click reaching the repository; see GroupDetailScreenConfirmActionTest.kt for the full
 * dialog-then-confirm-then-auth flows (mirrors ModelDetailScreenConfirmActionTest's own split).
 * Deliberately out of scope for this pass, same reason/infra gap as ModelDetailScreen/
 * SessionDetailScreen: the "Force release" admin-action confirm-dialog flow itself
 * (ConfirmActionDialog's FragmentActivity/LocalAdminAuthGate requirement) -- only the button's
 * default-enabled rendering is checked, not the click.
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class GroupDetailScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private class Fixture(
        val group: String = "group-x",
        val registryService: FakeRegistryApiService = FakeRegistryApiService(),
        mnsService: FakeMnsApiService = FakeMnsApiService(),
    ) {
        private val registryRepository = RegistryRepository(FakeApiClientProvider(registryService), FakeSettingsRepository())
        private val modelRepository = ModelRepository(FakeApiClientProvider(mnsService), FakeSettingsRepository())

        fun viewModel() = GroupDetailViewModel(group, registryRepository, modelRepository)
    }

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    @Test
    fun emptyGroup_showsPlaceholdersForEverySection() {
        val fixture = Fixture()

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No files currently claimed by a run.") }
        composeTestRule.onNodeWithText("No pending files in this pool.").assertExists()
        composeTestRule.onNodeWithText("No trained files in this pool yet.").assertExists()
    }

    @Test
    fun populatedQueueRunsAndRegistry_rendersEachSection() {
        val fixture = Fixture(
            registryService = FakeRegistryApiService(
                queueResponse = {
                    QueueResponseDto(
                        entries = listOf(
                            QueueEntryDto(
                                path = "file1.jsonl",
                                source = "gutenberg",
                                added_utc = "2026-01-01T00:00:00Z",
                                size_bytes = 2048,
                                num_entries = 10,
                            ),
                        ),
                    )
                },
                runsResponse = { RunsResponseDto(runs = mapOf("run-1" to listOf("a", "b"))) },
                registryResponse = {
                    RegistryResponseDto(
                        entries = listOf(RegistryEntryDto(data_file = "trained1.jsonl", num_samples = 100, trained = true)),
                    )
                },
            ),
        )

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("run-1") }
        composeTestRule.onNodeWithText("run-1 — 2 file(s)").assertExists()
        composeTestRule.onNodeWithText("file1.jsonl").assertExists()
        composeTestRule.onNodeWithText("run: unassigned · model: unassigned").assertExists()
        composeTestRule.onNodeWithText("10 entries · 2.0 KB · gutenberg · added Jan 1, 2026 00:00 UTC").assertExists()
        composeTestRule.onNodeWithText("trained1.jsonl").assertExists()
        composeTestRule.onNodeWithText("trained").assertExists()
    }

    @Test
    fun refreshFailure_showsLastRefreshFailedBanner() {
        val fixture = Fixture(
            registryService = FakeRegistryApiService(registryResponse = { throw IOException("connection refused") }),
        )

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            textIsShowing("Last refresh failed: connection refused (showing last known state)")
        }
    }

    @Test
    fun releaseButton_visibleAndEnabledByDefault() {
        val fixture = Fixture(
            registryService = FakeRegistryApiService(
                runsResponse = { RunsResponseDto(runs = mapOf("run-1" to listOf("a"))) },
            ),
        )

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("run-1") }
        composeTestRule.onNodeWithText("Release", substring = true).assertIsEnabled()
    }

    @Test
    fun fetchGutenbergDialog_fetchDisabledUntilAPositiveBookIdIsEntered() {
        val fixture = Fixture()

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No pending files") }
        composeTestRule.onNodeWithText("Fetch Gutenberg book").performClick()

        composeTestRule.onNodeWithText("Fetch").assertIsNotEnabled()
        composeTestRule.onNodeWithText("Gutenberg book ID").performTextInput("42")
        composeTestRule.onNodeWithText("Fetch").assertIsEnabled()
    }

    @Test
    fun assignModelDialog_confirmButtonEnabledOnceModelChosen() {
        val fixture = Fixture(
            registryService = FakeRegistryApiService(
                queueResponse = {
                    QueueResponseDto(entries = listOf(QueueEntryDto(path = "file1.jsonl", model_name = "model-a")))
                },
            ),
        )

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("file1.jsonl") }
        composeTestRule.onNodeWithContentDescription("Assign model").performClick()
        // TD-205: Assign is now ConfirmActionDialog-gated — tapping this button opens the
        // confirmation dialog rather than calling the repository directly; the actual
        // dialog-then-confirm-then-auth flow is covered in GroupDetailScreenConfirmActionTest.kt.
        composeTestRule.onNodeWithText("Assign").assertIsEnabled()
    }

    @Test
    fun clickingBack_invokesOnBack() {
        val fixture = Fixture()
        var backInvoked = false

        composeTestRule.setContent {
            GroupDetailScreen(group = fixture.group, viewModel = fixture.viewModel(), onBack = { backInvoked = true })
        }

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
    }
}
