package com.adai.ops

// @adai-status: beta        (TD-048 — exercised indirectly by AppContainerTest.kt/AppViewModelProviderTest.kt/MainActivityTest.kt; real-device run still unverified, see below)
// @adai-version: 0.3.0
// @adai-reviewed: 2026-11-21


import android.app.Application
import com.adai.ops.di.AppContainer

class OpsApp : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
