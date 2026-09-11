package com.adai.chatbot.ui.conversationlist

// @adai-status: beta        (capped by TD-047 — see TECHNICAL_DEBT.md)
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-10


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.chatbot.data.db.ConversationEntity
import com.adai.chatbot.data.repository.ConversationRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class ConversationListViewModel(
    private val conversationRepository: ConversationRepository,
) : ViewModel() {

    val conversations: StateFlow<List<ConversationEntity>> = conversationRepository.getAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun createConversation(onCreated: (Long) -> Unit) {
        viewModelScope.launch {
            val id = conversationRepository.createConversation()
            onCreated(id)
        }
    }

    fun deleteConversation(conversation: ConversationEntity) {
        viewModelScope.launch {
            conversationRepository.deleteConversation(conversation)
        }
    }
}
