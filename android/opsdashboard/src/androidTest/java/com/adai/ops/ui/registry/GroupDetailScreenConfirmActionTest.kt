package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-20


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
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
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import org.junit.Rule
import org.junit.Test

/**
 * TD-048/TD-205: exercises GroupDetailScreen's ConfirmActionDialog-gated flows using the
 * FragmentActivity/FakeAdminAuthGate infrastructure built for ModelDetailScreenConfirmActionTest
 * (see that file's own doc comment for the full explanation and for the representative coverage
 * of AdminAuthResult's Cancelled/Failed branches, which are screen-agnostic and not re-tested
 * here). TD-205 retrofitted Assign and both Fetch dialogs to also go through this gate (they
 * were previously ungated, an inconsistency with the rest of this app's admin actions), and
 * added Unassign/Delete/Manual-add/Upload/Migrate/Segment gated from the start.
 *
 * NOTE: same sandbox limitation as the other opsdashboard screen tests -- compile-verified and
 * hand-checked against the real production code paths, not run on a device here.
 */
class GroupDetailScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    private fun textIsShowing(text: String): Boolean =
        composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    private fun viewModelWith(registryService: FakeRegistryApiService, group: String = "group-x") =
        GroupDetailViewModel(
            group = group,
            registryRepository = RegistryRepository(FakeApiClientProvider(registryService), FakeSettingsRepository()),
            modelRepository = ModelRepository(FakeApiClientProvider(FakeMnsApiService()), FakeSettingsRepository()),
        )

    private fun setContentWithAuth(viewModel: GroupDetailViewModel, group: String = "group-x") {
        composeTestRule.setContent {
            CompositionLocalProvider(LocalAdminAuthGate provides FakeAdminAuthGate(result = { AdminAuthResult.Success })) {
                GroupDetailScreen(group = group, viewModel = viewModel, onBack = {})
            }
        }
    }

    @Test
    fun confirmingForceRelease_invokesForceReleaseAndClosesTheDialog() {
        val fakeRegistryService = FakeRegistryApiService(
            runsResponse = { RunsResponseDto(runs = mapOf("run-1" to listOf("a.jsonl", "b.jsonl"))) },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("run-1") }
        composeTestRule.onNodeWithText("Release", substring = true).performClick()
        composeTestRule.onNodeWithText("Force-release run 'run-1'?").assertExists()

        composeTestRule.onNodeWithText("Force release").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.releaseCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.releaseCalls.single()
        assert(group == "group-x" && body.run_id == "" && body.files == listOf("a.jsonl", "b.jsonl")) {
            "unexpected release call: group=$group body=$body"
        }
        composeTestRule.onNodeWithText("Force-release run 'run-1'?").assertDoesNotExist()
    }

    @Test
    fun confirmingFetchGutenberg_invokesFetchGutenberg() {
        val fakeRegistryService = FakeRegistryApiService()
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No pending files") }
        composeTestRule.onNodeWithText("Fetch Gutenberg book").performClick()
        composeTestRule.onNodeWithText("Gutenberg book ID").performTextInput("42")
        composeTestRule.onNodeWithText("Fetch").performClick()
        composeTestRule.onNodeWithText("POST /registry/group-x/fetch/gutenberg", substring = true).assertExists()

        composeTestRule.onNodeWithText("Fetch").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.fetchGutenbergCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.fetchGutenbergCalls.single()
        assert(group == "group-x" && body.book_id == 42) { "unexpected fetch call: group=$group body=$body" }
    }

    @Test
    fun confirmingAssign_invokesAssignWithPathAndChosenModel() {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "file1.jsonl", model_name = "model-a")))
            },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("file1.jsonl") }
        composeTestRule.onNodeWithContentDescription("Assign model").performClick()
        composeTestRule.onNodeWithText("Assign").performClick()
        composeTestRule.onNodeWithText("Assign model?").assertExists()

        composeTestRule.onNodeWithText("Assign").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.assignCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.assignCalls.single()
        assert(group == "group-x" && body.model_name == "model-a" && body.paths == listOf("file1.jsonl")) {
            "unexpected assign call: group=$group body=$body"
        }
    }

    @Test
    fun confirmingUnassign_invokesUnassignForTheEntry() {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "file1.jsonl", model_name = "model-a")))
            },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("file1.jsonl") }
        composeTestRule.onNodeWithText("Unassign").performClick()
        composeTestRule.onNodeWithText("Unassign model?").assertExists()

        composeTestRule.onNodeWithText("Confirm unassign").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.unassignCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.unassignCalls.single()
        assert(group == "group-x" && body.paths == listOf("file1.jsonl") && body.force) {
            "unexpected unassign call: group=$group body=$body"
        }
    }

    @Test
    fun confirmingDeletePending_invokesDeleteForTheEntry() {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = { QueueResponseDto(entries = listOf(QueueEntryDto(path = "file1.jsonl"))) },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("file1.jsonl") }
        composeTestRule.onNodeWithContentDescription("Delete").performClick()
        composeTestRule.onNodeWithText("Delete").performClick()
        composeTestRule.onNodeWithText("Delete entry?").assertExists()

        composeTestRule.onAllNodesWithText("Delete").onFirst().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.deleteCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.deleteCalls.single()
        assert(group == "group-x" && body.paths == listOf("file1.jsonl") && body.force && !body.delete_files) {
            "unexpected delete call: group=$group body=$body"
        }
    }

    @Test
    fun confirmingDeleteTrained_invokesDeleteForTheDataFile() {
        val fakeRegistryService = FakeRegistryApiService(
            registryResponse = {
                RegistryResponseDto(entries = listOf(RegistryEntryDto(data_file = "trained1.jsonl", trained = true)))
            },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("trained1.jsonl") }
        composeTestRule.onNodeWithContentDescription("Delete").performClick()
        composeTestRule.onNodeWithText("Delete").performClick()

        composeTestRule.onAllNodesWithText("Delete").onFirst().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.deleteCalls.isNotEmpty() }
        val (_, body) = fakeRegistryService.deleteCalls.single()
        assert(body.paths == listOf("trained1.jsonl")) { "unexpected delete call body: $body" }
    }

    @Test
    fun confirmingManualAdd_invokesPendingAddWithEnteredPath() {
        val fakeRegistryService = FakeRegistryApiService()
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No pending files") }
        composeTestRule.onNodeWithContentDescription("Add data").performClick()
        composeTestRule.onNodeWithText("Queue existing path").performClick()
        composeTestRule.onNodeWithText("Path (must be readable by registry_server)").performTextInput("/data/new.jsonl")
        composeTestRule.onNodeWithText("Queue").performClick()
        composeTestRule.onNodeWithText("Queue existing path?").assertExists()

        composeTestRule.onAllNodesWithText("Queue").onFirst().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.pendingAddCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.pendingAddCalls.single()
        assert(group == "group-x" && body.path == "/data/new.jsonl") {
            "unexpected pending/add call: group=$group body=$body"
        }
    }

    @Test
    fun confirmingMigrate_invokesPendingAddAssignAndDeleteInSequence() {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "file1.jsonl", model_name = "model-a")))
            },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("file1.jsonl") }
        composeTestRule.onNodeWithText("Migrate").performClick()
        composeTestRule.onNodeWithText("Chatbot").performClick()
        composeTestRule.onNodeWithText("Migrate to 'chatbot'?").assertExists()

        composeTestRule.onNodeWithText("Confirm migrate").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.deleteCalls.isNotEmpty() }
        val (addGroup, addBody) = fakeRegistryService.pendingAddCalls.single()
        assert(addGroup == "group-x/chatbot" && addBody.path == "file1.jsonl") {
            "unexpected pending/add call: group=$addGroup body=$addBody"
        }
        assert(fakeRegistryService.assignCalls.single().first == "group-x/chatbot") {
            "expected the model assignment to be restored in the destination kind's pool"
        }
        assert(fakeRegistryService.deleteCalls.single().first == "group-x") {
            "expected the source delete to target the legacy (unkinded) pool"
        }
    }

    @Test
    fun confirmingCreateSegments_invokesPendingAddOncePerRange() {
        val fakeRegistryService = FakeRegistryApiService(
            queueResponse = {
                QueueResponseDto(entries = listOf(QueueEntryDto(path = "big.jsonl", num_entries = 10)))
            },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("big.jsonl") }
        composeTestRule.onNodeWithText("Segment").performClick()
        composeTestRule.onNodeWithText("N").performTextInput("2")
        composeTestRule.onNodeWithText("Create").performClick()
        composeTestRule.onNodeWithText("Create 2 segment(s)?").assertExists()

        composeTestRule.onAllNodesWithText("Create").onFirst().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.pendingAddCalls.size >= 2 }
        val bodies = fakeRegistryService.pendingAddCalls.map { it.second }
        assert(bodies.all { it.path == "big.jsonl" }) { "unexpected segment bodies: $bodies" }
        assert(bodies.sumOf { it.segment_count } == 10) { "expected segments to cover all 10 pairs: $bodies" }
    }
}
