package com.adai.ops.testutil

// @adai-status: beta        (in-memory fake backing both plain-JVM and instrumented tests, TD-048)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import androidx.fragment.app.FragmentActivity
import com.adai.ops.ui.common.AdminAuthGate
import com.adai.ops.ui.common.AdminAuthResult

/**
 * Stands in for [com.adai.ops.ui.common.BiometricAdminAuthGate] in tests, so
 * [com.adai.ops.ui.common.ConfirmActionDialog]'s confirm-click flow can be exercised without
 * touching real `BiometricPrompt`/device-credential UI. Pair with
 * [com.adai.ops.testutil.ConfirmDialogTestActivity] (`androidTest`-only — provides the real
 * `FragmentActivity` the dialog casts `LocalContext.current` to).
 */
class FakeAdminAuthGate(
    private val result: (reason: String) -> AdminAuthResult = { AdminAuthResult.Success },
) : AdminAuthGate {

    val authenticateCalls = mutableListOf<String>()

    override suspend fun authenticate(activity: FragmentActivity, reason: String): AdminAuthResult {
        authenticateCalls += reason
        return result(reason)
    }
}
