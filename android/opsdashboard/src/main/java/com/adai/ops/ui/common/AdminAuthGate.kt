package com.adai.ops.ui.common

import androidx.compose.runtime.staticCompositionLocalOf
import androidx.fragment.app.FragmentActivity

/**
 * Result of a device-credential (PIN/pattern/password/biometric) verification attempt,
 * gating every admin action in the app. See [BiometricAdminAuthGate] for the real
 * implementation and [ConfirmActionDialog] for the single call site that invokes it.
 */
sealed interface AdminAuthResult {
    /** Verified just now, or covered by an existing grace-period window. */
    data object Success : AdminAuthResult

    /** User dismissed the system prompt (back button, negative button) — not an error. */
    data object Cancelled : AdminAuthResult

    /** No PIN/pattern/password/biometric is enrolled on this device, or hardware is missing. */
    data class Unavailable(val message: String) : AdminAuthResult

    /** The prompt could not be completed for a reason other than user cancellation. */
    data class Failed(val message: String) : AdminAuthResult
}

/**
 * Client-side, tablet-local verification gate. Deliberately independent of any server-side
 * auth (none of the three daemons have any) — this only verifies that whoever is holding the
 * tablet can unlock it, before an admin HTTP call is allowed to fire.
 */
interface AdminAuthGate {
    suspend fun authenticate(activity: FragmentActivity, reason: String): AdminAuthResult
}

/**
 * Provided once, app-wide, from [MainActivity][com.adai.ops.MainActivity] so the grace-period
 * timer in [BiometricAdminAuthGate] is shared across every screen rather than reset per-composable.
 */
val LocalAdminAuthGate = staticCompositionLocalOf<AdminAuthGate> {
    error("No AdminAuthGate provided — wrap content in CompositionLocalProvider(LocalAdminAuthGate provides ...)")
}
