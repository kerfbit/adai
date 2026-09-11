package com.adai.chatbot.ui.navigation

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-10


import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.adai.chatbot.ChatbotApp
import com.adai.chatbot.di.AppViewModelProvider
import com.adai.chatbot.settings.SettingsScreen
import com.adai.chatbot.settings.SettingsViewModel
import com.adai.chatbot.ui.chat.ChatScreen
import com.adai.chatbot.ui.chat.ChatViewModel
import com.adai.chatbot.ui.conversationlist.ConversationListScreen
import com.adai.chatbot.ui.conversationlist.ConversationListViewModel

private object Routes {
    const val CONVERSATIONS = "conversations"
    const val SETTINGS = "settings"
    const val CHAT = "chat/{conversationId}"
    const val CHAT_ARG = "conversationId"
    fun chat(conversationId: Long) = "chat/$conversationId"
}

@Composable
fun AdaiNavHost(app: ChatbotApp, navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = Routes.CONVERSATIONS) {
        composable(Routes.CONVERSATIONS) {
            val viewModel = viewModel<ConversationListViewModel>(
                factory = AppViewModelProvider.factory(app),
            )
            ConversationListScreen(
                viewModel = viewModel,
                onOpenConversation = { id -> navController.navigate(Routes.chat(id)) },
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
            )
        }
        composable(
            route = Routes.CHAT,
            arguments = listOf(navArgument(Routes.CHAT_ARG) { type = NavType.LongType }),
        ) { backStackEntry ->
            val conversationId = backStackEntry.arguments?.getLong(Routes.CHAT_ARG) ?: return@composable
            val viewModel = viewModel<ChatViewModel>(
                factory = AppViewModelProvider.chatFactory(app, conversationId),
            )
            ChatScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() },
            )
        }
        composable(Routes.SETTINGS) {
            val viewModel = viewModel<SettingsViewModel>(
                factory = AppViewModelProvider.factory(app),
            )
            SettingsScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() },
            )
        }
    }
}
