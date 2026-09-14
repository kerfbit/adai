package com.adai.ops.testutil

// @adai-status: beta        (TD-048 — androidTest-only FragmentActivity host, see doc comment below)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.fragment.app.FragmentActivity

/**
 * Minimal `androidTest`-only [FragmentActivity] host for `createAndroidComposeRule<...>()`,
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
 * Declared in `src/androidTest/AndroidManifest.xml` (test-only activities need their own manifest
 * entry; this is never merged into the shipped app). Pair with [FakeAdminAuthGate], provided via
 * `CompositionLocalProvider(LocalAdminAuthGate provides ...)` around the screen under test, to
 * exercise a confirm-dialog click all the way through without touching real biometric hardware.
 */
class ConfirmDialogTestActivity : FragmentActivity()
