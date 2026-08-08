package com.adai.ops.ui.common

import android.os.SystemClock
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricManager.Authenticators.BIOMETRIC_WEAK
import androidx.biometric.BiometricManager.Authenticators.DEVICE_CREDENTIAL
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine

private const val ALLOWED_AUTHENTICATORS = BIOMETRIC_WEAK or DEVICE_CREDENTIAL

/**
 * [AdminAuthGate] backed by [BiometricPrompt], accepting the device's screen-lock credential
 * (PIN/pattern/password) or biometric — whichever the tablet has enrolled. A single successful
 * check covers any admin action for [GRACE_PERIOD_MS] afterward, so an operator doing several
 * actions in a row isn't re-prompted each time; the timer is shared app-wide since this class
 * is a singleton in AppContainer, not recreated per screen.
 */
class BiometricAdminAuthGate : AdminAuthGate {

    @Volatile
    private var lastSuccessAtMs: Long = 0L

    override suspend fun authenticate(activity: FragmentActivity, reason: String): AdminAuthResult {
        if (SystemClock.elapsedRealtime() - lastSuccessAtMs < GRACE_PERIOD_MS) {
            return AdminAuthResult.Success
        }

        val canAuthenticate = BiometricManager.from(activity).canAuthenticate(ALLOWED_AUTHENTICATORS)
        if (canAuthenticate == BiometricManager.BIOMETRIC_ERROR_NONE_ENROLLED ||
            canAuthenticate == BiometricManager.BIOMETRIC_ERROR_NO_HARDWARE ||
            canAuthenticate == BiometricManager.BIOMETRIC_ERROR_HW_UNAVAILABLE ||
            canAuthenticate == BiometricManager.BIOMETRIC_ERROR_UNSUPPORTED
        ) {
            return AdminAuthResult.Unavailable(
                "No screen lock is set up on this tablet. Set a PIN, pattern, or password in " +
                    "Android Settings to use admin actions.",
            )
        }

        val result = showPrompt(activity, reason)
        if (result is AdminAuthResult.Success) {
            lastSuccessAtMs = SystemClock.elapsedRealtime()
        }
        return result
    }

    private suspend fun showPrompt(activity: FragmentActivity, reason: String): AdminAuthResult =
        suspendCancellableCoroutine { continuation ->
            val executor = ContextCompat.getMainExecutor(activity)
            val callback = object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                    if (continuation.isActive) continuation.resume(AdminAuthResult.Success)
                }

                override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                    if (!continuation.isActive) return
                    val cancelled = errorCode == BiometricPrompt.ERROR_USER_CANCELED ||
                        errorCode == BiometricPrompt.ERROR_CANCELED ||
                        errorCode == BiometricPrompt.ERROR_NEGATIVE_BUTTON
                    continuation.resume(
                        if (cancelled) AdminAuthResult.Cancelled else AdminAuthResult.Failed(errString.toString()),
                    )
                }

                // Wrong PIN/biometric on a single attempt — the system prompt stays open and
                // lets the user retry, so we don't resume here.
                override fun onAuthenticationFailed() = Unit
            }

            val prompt = BiometricPrompt(activity, executor, callback)
            val promptInfo = BiometricPrompt.PromptInfo.Builder()
                .setTitle("Admin verification required")
                .setSubtitle(reason)
                .setAllowedAuthenticators(ALLOWED_AUTHENTICATORS)
                .build()
            prompt.authenticate(promptInfo)

            continuation.invokeOnCancellation { prompt.cancelAuthentication() }
        }

    private companion object {
        const val GRACE_PERIOD_MS = 120_000L
    }
}
