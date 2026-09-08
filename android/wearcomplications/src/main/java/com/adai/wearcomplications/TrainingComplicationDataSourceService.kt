package com.adai.wearcomplications

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


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
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.DataMapItem
import com.google.android.gms.wearable.Wearable
import java.util.Locale
import java.util.concurrent.TimeUnit

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

    override fun onComplicationRequest(request: ComplicationRequest, listener: ComplicationRequestListener) {
        val snapshot = readLatestSnapshot()
        val data: ComplicationData = if (snapshot == null || snapshot.isStale(System.currentTimeMillis())) {
            NoDataComplicationData()
        } else {
            buildRangedValue(snapshot)
        }
        listener.onComplicationData(data)
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

    private fun readLatestSnapshot(): TrainingSnapshot? = try {
        val dataClient = Wearable.getDataClient(applicationContext)
        val buffer = Tasks.await(dataClient.dataItems, READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        try {
            buffer
                .firstOrNull { it.uri.path == WearSyncPaths.TRAINING_SNAPSHOT }
                ?.let { TrainingSnapshot.fromDataMap(DataMapItem.fromDataItem(it).dataMap) }
        } finally {
            buffer.release()
        }
    } catch (e: Exception) {
        Log.w(TAG, "Failed to read latest training snapshot", e)
        null
    }

    private companion object {
        const val TAG = "TrainingComplication"
        const val READ_TIMEOUT_SECONDS = 2L
    }
}
