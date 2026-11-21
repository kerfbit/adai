package com.adai.ops.testutil

// @adai-status: beta        (TD-048 — debug-build-only FragmentActivity test host, see doc comment below)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-21


import androidx.fragment.app.FragmentActivity

/**
 * Minimal debug-build-only [FragmentActivity] host for `createAndroidComposeRule<...>()`,
 * built specifically to unblock testing any screen's admin-action confirm-dialog flow.
 *
 * [com.adai.ops.ui.common.ConfirmActionDialog] unconditionally does
 * `LocalContext.current as FragmentActivity` the instant it composes — true even before its own
 * confirm button is clicked — because [com.adai.ops.ui.common.BiometricAdminAuthGate] needs a
 * `FragmentActivity` to host `BiometricPrompt`. The plain `createComposeRule()` used by every
 * other screen test in this module hosts content in a bare `ComponentActivity`
 * (`androidx.compose.ui.test.junit4`'s own default test activity), which is not a
 * `FragmentActivity` — only [com.adai.ops.MainActivity] itself is — so that cast crashes
 * immediately. This class exists purely to be a `FragmentActivity` for
 * `createAndroidComposeRule<ConfirmDialogTestActivity>()` to launch; it deliberately does nothing
 * in `onCreate` — the test itself calls `composeTestRule.setContent { ... }`, the same as with
 * `createComposeRule()`.
 *
 * Lives in the `debug` source set and is declared in `src/debug/AndroidManifest.xml`, so it is
 * part of the debug app APK (never the release one), the same way `ui-test-manifest` supplies
 * `ComponentActivity` for `createComposeRule()`. It used to be declared in the androidTest
 * manifest, which put it in the test APK: the test then launched an activity of the app's
 * package that resolved to the `com.adai.ops.test` process and failed ("resolved to different
 * process"), the first time these tests ran on a device (November 21, 2026). Pair with [FakeAdminAuthGate], provided via
 * `CompositionLocalProvider(LocalAdminAuthGate provides ...)` around the screen under test, to
 * exercise a confirm-dialog click all the way through without touching real biometric hardware.
 */
class ConfirmDialogTestActivity : FragmentActivity()
