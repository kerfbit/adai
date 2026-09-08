package com.adai.ops.data.wearsync

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.adai.ops.OpsApp
import java.util.concurrent.TimeUnit

/**
 * Background fallback for the watch face relay so the chart doesn't go stale for hours if
 * opsdashboard isn't foregrounded — [SessionDetailViewModel][com.adai.ops.ui.metrics.SessionDetailViewModel]
 * already syncs opportunistically while a session detail screen is open. 15 minutes is
 * WorkManager's enforced floor for periodic work; coarse, but fine since the chart is a
 * slow-moving whole-session trend rather than a live tick.
 */
class WatchSyncWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as OpsApp
        // Not treated as a hard failure when nothing was synced (e.g. sync disabled, or no
        // session currently available) — there's nothing actionable for WorkManager to retry.
        app.container.watchSyncRepository.sync()
        return Result.success()
    }

    companion object {
        private const val UNIQUE_WORK_NAME = "watch_sync_periodic"

        fun schedule(context: Context) {
            val request = PeriodicWorkRequestBuilder<WatchSyncWorker>(15, TimeUnit.MINUTES).build()
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                UNIQUE_WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request,
            )
        }
    }
}
