package com.adai.chatbot

// @adai-status: beta        (TD-048 — exercised indirectly by AppContainerTest.kt/AppViewModelProviderTest.kt/MainActivityTest.kt, all verified on a real device)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-13


import android.app.Application
import com.adai.chatbot.di.AppContainer

class ChatbotApp : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }
}
