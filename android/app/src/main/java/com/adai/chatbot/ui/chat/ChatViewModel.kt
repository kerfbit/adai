package com.adai.chatbot.ui.chat

// @adai-status: beta        (TD-048 resolved — see ChatViewModelTest.kt)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-12


import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.adai.chatbot.data.db.MessageEntity
import com.adai.chatbot.data.repository.ChatRepository
import com.adai.chatbot.data.repository.ConversationRepository
import com.adai.chatbot.network.ApiResult
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch

class ChatViewModel(
    private val conversationId: Long,
    private val chatRepository: ChatRepository,
    private val conversationRepository: ConversationRepository,
) : ViewModel() {

    val messages: Flow<List<MessageEntity>> = chatRepository.messagesFor(conversationId)

    private val _isSending = MutableStateFlow(false)
    val isSending: StateFlow<Boolean> = _isSending.asStateFlow()

    private val _errorEvents = Channel<String>(Channel.BUFFERED)
    val errorEvents: Flow<String> = _errorEvents.receiveAsFlow()

    fun send(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty() || _isSending.value) return

        _isSending.value = true
        viewModelScope.launch {
            conversationRepository.autoTitleFromFirstMessage(conversationId, trimmed)
            when (val result = chatRepository.sendMessage(conversationId, trimmed)) {
                is ApiResult.ApiError -> _errorEvents.send(result.message)
                is ApiResult.NetworkError -> _errorEvents.send(result.message)
                is ApiResult.Success -> Unit
            }
            _isSending.value = false
        }
    }

    fun retry(message: MessageEntity) {
        if (_isSending.value) return
        _isSending.value = true
        viewModelScope.launch {
            when (val result = chatRepository.retryMessage(conversationId, message.id, message.content)) {
                is ApiResult.ApiError -> _errorEvents.send(result.message)
                is ApiResult.NetworkError -> _errorEvents.send(result.message)
                is ApiResult.Success -> Unit
            }
            _isSending.value = false
        }
    }
}
