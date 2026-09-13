package com.adai.ops.data.wearsync

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


import android.content.Context
import android.os.ParcelFileDescriptor
import androidx.wear.watchfacepush.WatchFacePushManager
import androidx.wear.watchfacepush.WatchFacePushManagerFactory
import com.google.android.wearable.watchface.validator.client.DwfValidatorFactory
import java.io.File
import java.io.FileOutputStream

/**
 * The real [WatchFacePushRepository] — bundles the WFF `:wearface` bundle as a raw asset (see
 * opsdashboard's `copyWearFaceApk` Gradle task) and pushes it directly to the paired watch via
 * Watch Face Push (`androidx.wear.watchfacepush`), bypassing Samsung's One UI Watch picker
 * entirely — see the `project_wearface_samsung_picker_limitation` memory note for why that's
 * necessary.
 *
 * The validation token `addWatchFace`/`updateWatchFace` require is generated on-device, at
 * push time, via the offline `DwfValidator` library — no Play Console or external tool step.
 */
class WearWatchFacePushRepository(private val context: Context) : WatchFacePushRepository {

    private val wearFacePackageName = "com.adai.ops.watchfacepush.training"

    override fun isSupported(): Boolean = WatchFacePushManagerFactory.isSupported()

    override suspend fun pushWatchFace(): WatchFacePushResult {
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

    override suspend fun setActive(slotId: String): Result<Unit> = try {
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
