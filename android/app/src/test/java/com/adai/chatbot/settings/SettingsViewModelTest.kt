package com.adai.chatbot.settings

import com.adai.chatbot.network.ApiClientProvider
import com.adai.chatbot.testutil.FakeSettingsRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * TD-130: save() used to be a plain `fun` that fired `viewModelScope.launch { ... }` and
 * returned immediately, so SettingsScreen's `viewModel.save(); onBack()` could pop the
 * NavBackStackEntry (clearing viewModelScope) before the write ever landed. These tests use
 * a controlled (non-auto-advancing) StandardTestDispatcher as Main so a fire-and-forget launch
 * is provably still pending — not yet run — the instant save() returns, distinguishing it from
 * a genuinely-suspending, sequenced write.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class SettingsViewModelTest {

    @Before
    fun setUp() {
        Dispatchers.setMain(StandardTestDispatcher())
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `save persists synchronously, before it returns, not via a fire-and-forget launch`() = runTest {
        val fakeRepo = FakeSettingsRepository()
        val viewModel = SettingsViewModel(fakeRepo, ApiClientProvider())
        advanceUntilIdle() // let init{}'s load-from-repo settle first

        viewModel.onHostChanged("192.168.1.50")
        viewModel.onPortChanged("9090")

        viewModel.save()

        // No advanceUntilIdle()/runCurrent() here on purpose: if save() were still a
        // fire-and-forget viewModelScope.launch, the write would be scheduled but not yet
        // run under a StandardTestDispatcher, and this assertion would fail with the fake
        // repository still holding its construction-time default.
        assertEquals("192.168.1.50", fakeRepo.settings.value.host)
        assertEquals(9090, fakeRepo.settings.value.port)
    }
}
