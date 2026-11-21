package com.adai.ops.ui.registry

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-19


import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.hasAnyAncestor
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.isDialog
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTextReplacement
import com.adai.ops.data.mns.ModelRepository
import com.adai.ops.data.registry.RegistryRepository
import com.adai.ops.network.dto.AssignResponseDto
import com.adai.ops.network.dto.DeleteResponseDto
import com.adai.ops.network.dto.ModelRecordDto
import com.adai.ops.network.dto.ModelsResponseDto
import com.adai.ops.network.dto.QueueEntryDto
import com.adai.ops.network.dto.QueueResponseDto
import com.adai.ops.network.dto.RegistryEntryDto
import com.adai.ops.network.dto.RegistryResponseDto
import com.adai.ops.network.dto.RequeueResponseDto
import com.adai.ops.network.dto.RunsResponseDto
import com.adai.ops.testutil.ConfirmDialogTestActivity
import com.adai.ops.testutil.FakeAdminAuthGate
import com.adai.ops.testutil.FakeApiClientProvider
import com.adai.ops.testutil.FakeMnsApiService
import com.adai.ops.testutil.FakeRegistryApiService
import com.adai.ops.testutil.FakeSettingsRepository
import com.adai.ops.testutil.onNodeWithContentDescriptionScrolled
import com.adai.ops.testutil.onNodeWithTextScrolled
import com.adai.ops.testutil.scrollToNode
import com.adai.ops.ui.common.AdminAuthResult
import com.adai.ops.ui.common.LocalAdminAuthGate
import org.junit.Rule
import org.junit.Test
import retrofit2.Response

/**
 * TD-048/TD-205: exercises GroupDetailScreen's ConfirmActionDialog-gated flows using the
 * FragmentActivity/FakeAdminAuthGate infrastructure built for ModelDetailScreenConfirmActionTest
 * (see that file's own doc comment for the full explanation and for the representative coverage
 * of AdminAuthResult's Cancelled/Failed branches, which are screen-agnostic and not re-tested
 * here). TD-205 retrofitted Assign and both Fetch dialogs to also go through this gate (they
 * were previously ungated, an inconsistency with the rest of this app's admin actions), and
 * added Unassign/Delete/Manual-add/Upload/Migrate/Segment gated from the start.
 *
 * Runs on the Medium_Phone_API_35 AVD (`./gradlew :opsdashboard:connectedDebugAndroidTest`); it was
 * only compiled until November 21, 2026, when the APK stopped requiring `wear-sdk`.
 */
class GroupDetailScreenConfirmActionTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ConfirmDialogTestActivity>()

    // Scrolls the screen's LazyColumn to the text first: items below the fold aren't composed.
    private fun textIsShowing(text: String): Boolean =
        composeTestRule.scrollToNode(hasText(text, substring = true)) ||
            composeTestRule.onAllNodesWithText(text, substring = true).fetchSemanticsNodes().isNotEmpty()

    private fun viewModelWith(
        registryService: FakeRegistryApiService,
        group: String = "group-x",
        mnsService: FakeMnsApiService = FakeMnsApiService(),
    ) = GroupDetailViewModel(
        group = group,
        registryRepository = RegistryRepository(FakeApiClientProvider(registryService), FakeSettingsRepository()),
        modelRepository = ModelRepository(FakeApiClientProvider(mnsService), FakeSettingsRepository()),
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
        composeTestRule.onNodeWithTextScrolled("Release", substring = true).performClick()
        composeTestRule.onNodeWithTextScrolled("Force-release run 'run-1'?").assertExists()

        composeTestRule.onNodeWithTextScrolled("Force release").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.releaseCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.releaseCalls.single()
        assert(group == "group-x" && body.run_id == "" && body.files == listOf("a.jsonl", "b.jsonl")) {
            "unexpected release call: group=$group body=$body"
        }
        composeTestRule.onNodeWithTextScrolled("Force-release run 'run-1'?").assertDoesNotExist()
    }

    @Test
    fun confirmingFetchGutenberg_invokesFetchGutenberg() {
        val fakeRegistryService = FakeRegistryApiService()
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No pending files") }
        composeTestRule.onNodeWithTextScrolled("Fetch Gutenberg book").performClick()
        composeTestRule.onNodeWithTextScrolled("Gutenberg book ID").performTextInput("42")
        composeTestRule.onNodeWithTextScrolled("Fetch").performClick()
        composeTestRule.onNodeWithTextScrolled("POST /registry/group-x/fetch/gutenberg", substring = true).assertExists()

        composeTestRule.onNodeWithTextScrolled("Fetch").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.fetchGutenbergCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.fetchGutenbergCalls.single()
        assert(group == "group-x" && body.book_id == 42) { "unexpected fetch call: group=$group body=$body" }
    }

    // TD-314: the dialog's config reaches the request, and the confirmation shows it.
    @Test
    fun confirmingFetchHuggingface_sendsTheConfig() {
        val fakeRegistryService = FakeRegistryApiService()
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No pending files") }
        composeTestRule.onNodeWithTextScrolled("Fetch HuggingFace dataset").performClick()
        composeTestRule.onNodeWithTextScrolled("Dataset ID (e.g. tatsu-lab/alpaca)").performTextInput("Salesforce/wikitext")
        composeTestRule.onNodeWithTextScrolled("Config (blank = default)").performTextInput("wikitext-2-raw-v1")
        composeTestRule.onNodeWithTextScrolled("Fetch").performClick()
        composeTestRule.onNodeWithTextScrolled("\"config\":\"wikitext-2-raw-v1\"", substring = true).assertExists()

        composeTestRule.onNodeWithTextScrolled("Fetch").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.fetchHuggingfaceCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.fetchHuggingfaceCalls.single()
        assert(group == "group-x" && body.dataset_id == "Salesforce/wikitext" && body.config == "wikitext-2-raw-v1") {
            "unexpected fetch call: group=$group body=$body"
        }
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
        composeTestRule.onNodeWithContentDescriptionScrolled("Assign model").performClick()
        composeTestRule.onNodeWithTextScrolled("Assign").performClick()
        composeTestRule.onNodeWithTextScrolled("Assign model?").assertExists()

        composeTestRule.onNodeWithTextScrolled("Assign").performClick()

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
        composeTestRule.onNodeWithTextScrolled("Unassign").performClick()
        composeTestRule.onNodeWithTextScrolled("Unassign model?").assertExists()

        composeTestRule.onNodeWithTextScrolled("Confirm unassign").performClick()

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
        composeTestRule.onNodeWithContentDescriptionScrolled("Delete").performClick()
        composeTestRule.onNodeWithTextScrolled("Delete").performClick()
        composeTestRule.onNodeWithTextScrolled("Delete entry?").assertExists()

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
        composeTestRule.onNodeWithContentDescriptionScrolled("Delete").performClick()
        composeTestRule.onNodeWithTextScrolled("Delete").performClick()

        composeTestRule.onAllNodesWithText("Delete").onFirst().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.deleteCalls.isNotEmpty() }
        val (_, body) = fakeRegistryService.deleteCalls.single()
        assert(body.paths == listOf("trained1.jsonl")) { "unexpected delete call body: $body" }
    }

    // TD-540: one trained file back to the pending queue, unassigned.
    @Test
    fun confirmingRequeue_requeuesTheTrainedFileUnassigned() {
        val fakeRegistryService = FakeRegistryApiService(
            registryResponse = {
                RegistryResponseDto(entries = listOf(RegistryEntryDto(data_file = "trained1.jsonl", trained = true)))
            },
            requeueResponse = { _, _ -> Response.success(RequeueResponseDto(requeued = 1)) },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("trained1.jsonl") }
        composeTestRule.onNodeWithContentDescriptionScrolled("Re-queue").performClick()
        composeTestRule.onNodeWithTextScrolled("Next").performClick()
        composeTestRule.onNodeWithTextScrolled("Re-queue 'trained1.jsonl'?").assertExists()
        composeTestRule.onNodeWithTextScrolled("Re-queue").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.requeueCalls.isNotEmpty() }
        val (group, body) = fakeRegistryService.requeueCalls.single()
        assert(group == "group-x" && body.paths == listOf("trained1.jsonl") && !body.all && body.model_name.isEmpty()) {
            "unexpected requeue call: group=$group body=$body"
        }
        composeTestRule.onNodeWithTextScrolled("Re-queue 'trained1.jsonl'?").assertDoesNotExist()
    }

    // TD-540: every trained file, assigned to a model chosen in the dialog.
    @Test
    fun confirmingRequeueAll_requeuesEveryTrainedFileForTheChosenModel() {
        val fakeRegistryService = FakeRegistryApiService(
            registryResponse = {
                RegistryResponseDto(
                    entries = listOf(
                        RegistryEntryDto(data_file = "t1.jsonl", trained = true),
                        RegistryEntryDto(data_file = "t2.jsonl", trained = true),
                    ),
                )
            },
        )
        val mns = FakeMnsApiService(
            listModelsResponse = { _, _, _, _ -> ModelsResponseDto(models = listOf(ModelRecordDto(model_name = "model-a"))) },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService, mnsService = mns))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("t2.jsonl") }
        composeTestRule.onNodeWithTextScrolled("Re-queue all").performClick()
        composeTestRule.onNodeWithTextScrolled("2 trained file(s) in this pool").assertExists()
        composeTestRule.onNodeWithTextScrolled("(unassigned)").performClick()
        composeTestRule.onNodeWithTextScrolled("model-a").performClick()
        composeTestRule.onNodeWithTextScrolled("Next").performClick()
        composeTestRule.onNodeWithTextScrolled("Re-queue every trained file?").assertExists()
        composeTestRule.onNodeWithTextScrolled("Re-queue").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.requeueCalls.isNotEmpty() }
        val (_, body) = fakeRegistryService.requeueCalls.single()
        assert(body.all && body.paths.isEmpty() && body.model_name == "model-a") { "unexpected requeue body: $body" }
    }

    @Test
    fun confirmingManualAdd_invokesPendingAddWithEnteredPath() {
        val fakeRegistryService = FakeRegistryApiService()
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("No pending files") }
        composeTestRule.onNodeWithContentDescriptionScrolled("Add data").performClick()
        composeTestRule.onNodeWithTextScrolled("Queue existing path").performClick()
        composeTestRule.onNodeWithTextScrolled("Path (must be readable by registry_server)").performTextInput("/data/new.jsonl")
        composeTestRule.onNodeWithTextScrolled("Queue").performClick()
        composeTestRule.onNodeWithTextScrolled("Queue existing path?").assertExists()

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
            // TD-326: migrate checks both results now, rolling back on a 0.
            assignResponse = { _, _ -> AssignResponseDto(assigned = 1) },
            deleteResponse = { _, _ -> Response.success(DeleteResponseDto(deleted = 1)) },
        )
        setContentWithAuth(viewModelWith(fakeRegistryService))

        composeTestRule.waitUntil(timeoutMillis = 5_000) { textIsShowing("file1.jsonl") }
        composeTestRule.onNodeWithTextScrolled("Migrate").performClick()
        // "Chatbot" is also the pool overview's row and a kind filter chip: pick the dialog's.
        composeTestRule.onNode(hasText("Chatbot") and hasAnyAncestor(isDialog())).performClick()
        composeTestRule.onNodeWithTextScrolled("Migrate to 'chatbot'?").assertExists()

        composeTestRule.onNodeWithTextScrolled("Confirm migrate").performClick()

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
        composeTestRule.onNodeWithTextScrolled("Segment").performClick()
        // N starts at 2; replace it rather than append (typing "2" made it 22).
        composeTestRule.onNodeWithTextScrolled("N").performTextReplacement("2")
        composeTestRule.onNodeWithTextScrolled("Create").performClick()
        composeTestRule.onNodeWithTextScrolled("Create 2 segment(s)?").assertExists()

        composeTestRule.onAllNodesWithText("Create").onFirst().performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { fakeRegistryService.pendingAddCalls.size >= 2 }
        val bodies = fakeRegistryService.pendingAddCalls.map { it.second }
        assert(bodies.all { it.path == "big.jsonl" }) { "unexpected segment bodies: $bodies" }
        assert(bodies.sumOf { it.segment_count } == 10) { "expected segments to cover all 10 pairs: $bodies" }
    }
}
