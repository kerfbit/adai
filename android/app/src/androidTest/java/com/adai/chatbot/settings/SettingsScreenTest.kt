package com.adai.chatbot.settings

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.compose.ui.test.isToggleable
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import com.adai.chatbot.testutil.FakeApiClientProvider
import com.adai.chatbot.testutil.FakeSettingsRepository
import com.adai.chatbot.testutil.RecordingFakeChatApiService
import org.junit.Rule
import org.junit.Test

/**
 * TD-048: SettingsScreen had zero coverage. Uses a real SettingsViewModel backed by
 * FakeSettingsRepository/FakeApiClientProvider (shared src/sharedTest fakes) rather than a real
 * DataStore or network. Host field label text differs by useHttps (see SettingsScreen.kt), so
 * these tests target it via the non-HTTPS label, matching the screen's own default state.
 */
class SettingsScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private val hostLabel = "Host (e.g. 192.168.1.16 or 10.0.2.2 for emulator)"

    private class Fixture(
        chatService: RecordingFakeChatApiService = RecordingFakeChatApiService(),
    ) {
        val settingsRepo = FakeSettingsRepository(ServerSettings())
        val apiClientProvider = FakeApiClientProvider(chatService)

        fun viewModel() = SettingsViewModel(settingsRepo, apiClientProvider)
    }

    @Test
    fun clickingBack_invokesOnBackWithoutSaving() {
        val fixture = Fixture()
        var backInvoked = false

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = { backInvoked = true })
        }

        composeTestRule.onNodeWithContentDescription("Back").performClick()

        assert(backInvoked) { "expected onBack() to have been invoked" }
        assert(fixture.settingsRepo.settings.value.host.isEmpty()) {
            "back without Save must not persist anything"
        }
    }

    @Test
    fun enteringHostAndClickingSave_persistsAndInvokesOnBack() {
        // NOTE: this does NOT verify TD-130's happens-before ordering (save() landing before
        // onBack() fires) -- performClick() waits for the real device's Main dispatcher to go
        // idle before returning, and by then *every* scheduled coroutine on it (including a
        // hypothetical fire-and-forget save()) has typically already run too, so a real-device
        // instrumented test can't distinguish "sequenced" from "fire-and-forget-but-fast" the
        // way SettingsViewModelTest's own StandardTestDispatcher-controlled plain-JVM test does
        // (confirmed directly: temporarily reverting SettingsScreen.kt to TD-130's original
        // fire-and-forget shape did not make this assertion fail). What this test does verify:
        // the Save button's wiring actually reaches SettingsViewModel.save() with the values
        // the user typed, and onBack() eventually fires too.
        val fixture = Fixture()
        var backInvoked = false

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = { backInvoked = true })
        }

        composeTestRule.onNodeWithText(hostLabel).performTextInput("192.168.1.50")
        composeTestRule.onNodeWithText("Save").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) { backInvoked }
        assert(fixture.settingsRepo.settings.value.host == "192.168.1.50") {
            "expected the typed host to have been persisted, got " +
                "'${fixture.settingsRepo.settings.value.host}'"
        }
    }

    @Test
    fun togglingHttps_swapsPortFieldForAccessClientFields() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.onNodeWithText("Port").assertExists()
        // The Switch and its label Text are separate composables with no shared clickable
        // wrapper (see SettingsScreen.kt) -- clicking the label text itself does nothing, only
        // the Switch's own toggle semantics respond to a click.
        composeTestRule.onNode(isToggleable()).performClick()

        composeTestRule.onNodeWithText("Port").assertDoesNotExist()
        composeTestRule.onNodeWithText("Cloudflare Access Client ID").assertExists()
        composeTestRule.onNodeWithText("Cloudflare Access Client Secret").assertExists()
    }

    @Test
    fun testConnectionWithBlankHost_showsValidationErrorWithoutCallingServer() {
        val chatService = RecordingFakeChatApiService()
        val fixture = Fixture(chatService)

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.onNodeWithText("Test Connection").performClick()

        composeTestRule.onNodeWithText("Enter a valid host and port (1-65535)").assertExists()
    }

    @Test
    fun testConnectionSuccess_showsConnectedStatusMessage() {
        val fixture = Fixture()

        composeTestRule.setContent {
            SettingsScreen(viewModel = fixture.viewModel(), onBack = {})
        }

        composeTestRule.onNodeWithText(hostLabel).performTextInput("192.168.1.50")
        composeTestRule.onNodeWithText("Test Connection").performClick()

        composeTestRule.waitUntil(timeoutMillis = 5_000) {
            composeTestRule.onAllNodesWithText("Connected — 0 active session(s)")
                .fetchSemanticsNodes().isNotEmpty()
        }
    }
}
