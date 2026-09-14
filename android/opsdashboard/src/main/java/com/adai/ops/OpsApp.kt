package com.adai.ops

// @adai-status: beta        (TD-048 — exercised indirectly by AppContainerTest.kt/AppViewModelProviderTest.kt/MainActivityTest.kt; real-device run still unverified, see below)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-13


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
