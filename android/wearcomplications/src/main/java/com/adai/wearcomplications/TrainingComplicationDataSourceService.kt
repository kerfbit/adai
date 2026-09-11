package com.adai.wearcomplications

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.1
// @adai-reviewed: 2026-09-10


import android.util.Log
import androidx.wear.watchface.complications.data.ComplicationData
import androidx.wear.watchface.complications.data.ComplicationType
import androidx.wear.watchface.complications.data.NoDataComplicationData
import androidx.wear.watchface.complications.data.PlainComplicationText
import androidx.wear.watchface.complications.data.RangedValueComplicationData
import androidx.wear.watchface.complications.datasource.ComplicationDataSourceService
import androidx.wear.watchface.complications.datasource.ComplicationDataSourceService.ComplicationRequestListener
import androidx.wear.watchface.complications.datasource.ComplicationRequest
import com.adai.wearsync.WearSyncPaths
import com.google.android.gms.wearable.DataItemBuffer
import com.google.android.gms.wearable.DataMapItem
import com.google.android.gms.wearable.Wearable
import java.util.Locale

/**
 * Shared logic for the loss/perplexity RANGED_VALUE complications: read whatever the phone
 * relay last pushed to the Data Layer (a fast local cache read, not a network call — Play
 * services already syncs the DataItem to this device), and answer with either a gauge value
 * or [NoDataComplicationData] if nothing's been pushed yet or the data's gone stale.
 *
 * No persistent Data Layer listener is registered here: `onComplicationRequest` already fires
 * whenever the watch face becomes visible (raise-to-wake, tap) in addition to the declared
 * `UPDATE_PERIOD_SECONDS` — plenty fresh given the phone only pushes every 15-60 minutes.
 */
abstract class TrainingComplicationDataSourceService : ComplicationDataSourceService() {

    protected abstract val label: String
    protected abstract fun currentValue(snapshot: TrainingSnapshot): Double
    protected abstract fun minValue(snapshot: TrainingSnapshot): Double
    protected abstract fun maxValue(snapshot: TrainingSnapshot): Double

    // TD-146: `onComplicationRequest` is dispatched by the platform via a Handler bound to this
    // service's main thread (confirmed directly against the pinned
    // watchface-complications-data-source:1.2.1 library bytecode — IComplicationProviderWrapper's
    // onUpdate2 calls getMainThreadHandler().post { ...onComplicationRequest... }), not a
    // background thread. The previous implementation called `Tasks.await(dataClient.dataItems, 2,
    // TimeUnit.SECONDS)` here — a synchronous block of up to 2 seconds on that same main thread,
    // every single time either complication is requested (raise-to-wake, tap, or the periodic
    // update), on every real device. Fixed by switching to the non-blocking
    // addOnSuccessListener/addOnFailureListener form of the same Task, which answers the
    // ComplicationRequestListener from the callback instead of blocking the calling thread —
    // matching the documented non-blocking contract `onComplicationRequest` expects, and Google's
    // own recommended pattern for Data-Layer-backed complications.
    override fun onComplicationRequest(request: ComplicationRequest, listener: ComplicationRequestListener) {
        val dataClient = Wearable.getDataClient(applicationContext)
        dataClient.dataItems
            .addOnSuccessListener { buffer -> listener.onComplicationData(dataFromBuffer(buffer)) }
            .addOnFailureListener { e ->
                Log.w(TAG, "Failed to read latest training snapshot", e)
                listener.onComplicationData(NoDataComplicationData())
            }
    }

    private fun dataFromBuffer(buffer: DataItemBuffer): ComplicationData = try {
        val snapshot = buffer
            .firstOrNull { it.uri.path == WearSyncPaths.TRAINING_SNAPSHOT }
            ?.let { TrainingSnapshot.fromDataMap(DataMapItem.fromDataItem(it).dataMap) }
        if (snapshot == null || snapshot.isStale(System.currentTimeMillis())) {
            NoDataComplicationData()
        } else {
            buildRangedValue(snapshot)
        }
    } catch (e: Exception) {
        Log.w(TAG, "Failed to decode latest training snapshot", e)
        NoDataComplicationData()
    } finally {
        buffer.release()
    }

    override fun getPreviewData(type: ComplicationType): ComplicationData? {
        if (type != ComplicationType.RANGED_VALUE) return null
        return RangedValueComplicationData.Builder(
            0.5f,
            0f,
            1f,
            PlainComplicationText.Builder(label).build(),
        ).build()
    }

    private fun buildRangedValue(snapshot: TrainingSnapshot): ComplicationData {
        val value = currentValue(snapshot)
        val min = minValue(snapshot)
        val max = maxValue(snapshot)
        // A degenerate/empty session's min==max would make the gauge's range invalid.
        val safeMax = if (max > min) max else min + 1.0

        return RangedValueComplicationData.Builder(
            value.toFloat(),
            min.toFloat(),
            safeMax.toFloat(),
            PlainComplicationText.Builder(label).build(),
        ).setText(PlainComplicationText.Builder(String.format(Locale.US, "%.2f", value)).build())
            .build()
    }

    private companion object {
        const val TAG = "TrainingComplication"
    }
}
