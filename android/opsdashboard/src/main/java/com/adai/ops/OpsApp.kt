package com.adai.ops

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
