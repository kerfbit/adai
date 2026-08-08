package com.adai.ops.data.wearsync

import android.content.Context
import android.os.ParcelFileDescriptor
import androidx.wear.watchfacepush.WatchFacePushManager
import androidx.wear.watchfacepush.WatchFacePushManagerFactory
import com.google.android.wearable.watchface.validator.client.DwfValidatorFactory
import java.io.File
import java.io.FileOutputStream

sealed interface WatchFacePushResult {
    data class Success(val slotId: String) : WatchFacePushResult
    data class ValidationFailed(val reasons: List<String>) : WatchFacePushResult
    data class Failure(val message: String) : WatchFacePushResult
}

/**
 * Pushes the WFF `:wearface` bundle (bundled as a raw asset — see opsdashboard's
 * `copyWearFaceApk` Gradle task) directly to the paired watch via Watch Face Push
 * (`androidx.wear.watchfacepush`), bypassing Samsung's One UI Watch picker entirely — see
 * the `project_wearface_samsung_picker_limitation` memory note for why that's necessary.
 *
 * The validation token `addWatchFace`/`updateWatchFace` require is generated on-device, at
 * push time, via the offline `DwfValidator` library — no Play Console or external tool step.
 */
class WatchFacePushRepository(private val context: Context) {

    private val wearFacePackageName = "com.adai.ops.watchfacepush.training"

    fun isSupported(): Boolean = WatchFacePushManagerFactory.isSupported()

    /** Pushes (or updates, if already installed) the bundled watch face. Does not activate it
     * — call [setActive] separately, since [WatchFacePushManager.setWatchFaceAsActive] can
     * only be invoked once per install and needs its own explicit user confirmation. */
    suspend fun pushWatchFace(): WatchFacePushResult {
        if (!isSupported()) {
            return WatchFacePushResult.Failure("Watch Face Push isn't supported on the paired watch")
        }
        val apkFile = extractWearFaceAsset()
            ?: return WatchFacePushResult.Failure("Bundled watch face asset missing — rebuild opsdashboard")

        val validation = DwfValidatorFactory.create().validate(apkFile, context.packageName)
        if (validation.failures().isNotEmpty()) {
            return WatchFacePushResult.ValidationFailed(
                validation.failures().map { "${it.name()}: ${it.failureMessage()}" },
            )
        }
        val token = validation.validationToken()
        val manager = WatchFacePushManagerFactory.createWatchFacePushManager(context)

        return try {
            ParcelFileDescriptor.open(apkFile, ParcelFileDescriptor.MODE_READ_ONLY).use { pfd ->
                val existingSlotId = manager.listWatchFaces()
                    .installedWatchFaceDetails
                    .firstOrNull { it.packageName == wearFacePackageName }
                    ?.slotId

                val details = if (existingSlotId != null) {
                    manager.updateWatchFace(existingSlotId, pfd, token)
                } else {
                    manager.addWatchFace(pfd, token)
                }
                WatchFacePushResult.Success(details.slotId)
            }
        } catch (e: WatchFacePushManager.AddWatchFaceException) {
            WatchFacePushResult.Failure("Install failed (code ${e.errorCode}): ${e.message}")
        } catch (e: WatchFacePushManager.UpdateWatchFaceException) {
            WatchFacePushResult.Failure("Update failed (code ${e.errorCode}): ${e.message}")
        } catch (e: Exception) {
            WatchFacePushResult.Failure(e.message ?: "Unknown error pushing watch face")
        }
    }

    /** Only callable once per install per the platform's own limit — surface that to the
     * user rather than silently retrying. */
    suspend fun setActive(slotId: String): Result<Unit> = try {
        WatchFacePushManagerFactory.createWatchFacePushManager(context).setWatchFaceAsActive(slotId)
        Result.success(Unit)
    } catch (e: Exception) {
        Result.failure(e)
    }

    private fun extractWearFaceAsset(): File? = try {
        val out = File(context.cacheDir, "wearface.apk")
        context.assets.open(ASSET_NAME).use { input ->
            FileOutputStream(out).use { output -> input.copyTo(output) }
        }
        out
    } catch (e: Exception) {
        null
    }

    private companion object {
        const val ASSET_NAME = "wearface.apk"
    }
}
