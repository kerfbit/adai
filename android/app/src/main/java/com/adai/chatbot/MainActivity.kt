package com.adai.chatbot

// @adai-status: experimental        (capped by TD-048 — see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-07


import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.adai.chatbot.ui.navigation.AdaiNavHost
import com.adai.chatbot.ui.theme.AdaiChatbotTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AdaiChatbotTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    AdaiNavHost(app = application as ChatbotApp)
                }
            }
        }
    }
}
