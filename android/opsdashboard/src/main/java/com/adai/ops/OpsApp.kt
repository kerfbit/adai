package com.adai.ops

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import android.app.Application
import com.adai.ops.data.wearsync.WatchSyncWorker
import com.adai.ops.di.AppContainer

class OpsApp : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        WatchSyncWorker.schedule(this)
    }
}
