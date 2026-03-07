# Fix: Input Length Expanding During Generation

## Problem

When using the chatbot GUI, the input sequence length kept growing from 513 to 685+ tokens, even for a simple "hello" input. This caused warnings:

```text
Warning: Input sequence length (513) exceeds max_len (512). Positions beyond max_len will not receive positional encodings.
Warning: Input sequence length (514) exceeds max_len (512)...
...
Warning: Input sequence length (685) exceeds max_len (512)...
```

## Root Cause

The chatbot GUI uses `ConversationContext` which accumulates **all conversation history**. When generating a response, it was:

1. **Encoding the entire conversation history** (all previous messages) as the encoder input
2. **Starting with 513 tokens** from accumulated context (previous conversations/messages)
3. **Decoder growing** as it generates, but the encoder context was already >512 tokens
4. **Exceeding the positional encoding max_len** of 512 tokens

### Code Flow

```cpp
// ChatbotGUI.cpp - generateResponse()
context->add_user_message(user_input);  // Add "hello" to history

std::string formatted_context = context->format_with_special_tokens();
// ↑ This formats ALL messages in history, resulting in 513+ tokens!

model->generate_response_with_strategy(
    formatted_context,  // 513+ tokens sent to encoder
    max_response_length,
    ...
);
```

### Positional Encoding Issue

```cpp
// Decoder.cpp - forward_with_encoder()
cached_embeddings = token_embedding->forward(token_ids);
cached_pos_encoded = positional_encoding->forward(cached_embeddings);
// ↑ Positional encoding has max_len=512, but input is 513+!
```

## Solution

### 1. Reduced Context Max Tokens

Changed from 2048 to 480 tokens to ensure context stays under the model's max_len:

```cpp
// Before: context = std::make_unique<ConversationContext>(20, 2048);
// After:
context = std::make_unique<ConversationContext>(20, 480);
```

**Why 480?** Leaves headroom below 512 for special tokens (BOS, EOS, separators).

### 2. Added Clear Conversation Button

Added a "Clear" button in the GUI to allow users to reset the conversation history:

```cpp
clearButton = new QPushButton("Clear");
clearButton->setMinimumWidth(80);
clearButton->setMinimumHeight(40);
clearButton->setToolTip("Clear conversation history");
connect(clearButton, &QPushButton::clicked, this, &ChatbotGUI::onClearConversation);
```

The `onClearConversation()` method:

- Clears the chat display
- Resets the conversation context
- Shows a confirmation dialog

## Benefits

### ✅ Prevents Exceeding Max Length

- Context now limited to 480 tokens (< 512 max_len)
- No more positional encoding warnings
- Model operates within designed limits

### ✅ User Control

- Clear button lets users start fresh conversations
- Prevents context from growing indefinitely
- Better memory management

### ✅ Improved Generation Quality

- Staying within positional encoding limits ensures proper attention
- Context doesn't get diluted by very old messages
- More relevant recent context for generation

## Usage

### Run the Fixed GUI

```bash
./build/src/chatbot_gui --vocab vocab.txt --model chatbot_model.bin
```

### Clear Conversation History

Click the **"Clear"** button to reset the conversation when:

- Context gets too large
- Want to start a new topic
- Experiencing poor responses due to cluttered context

### Monitor Context Size

The context automatically truncates when it reaches:

- **20 messages** (max_messages limit)
- **480 tokens** (max_tokens limit)

Oldest messages are removed first (FIFO).

## Technical Details

### ConversationContext Truncation

The context automatically manages its size:

```cpp
void ConversationContext::truncate_to_limits() {
    // Truncate by message count
    if (max_messages > 0) {
        while (static_cast<int>(messages.size()) > max_messages) {
            remove_oldest_message();
        }
    }

    // Truncate by token count
    if (max_tokens > 0) {
        while (total_tokens > max_tokens && !messages.empty()) {
            remove_oldest_message();
        }
    }
}
```

### Model Architecture Constraints

```text
Encoder:         max_seq_length = 1024, but practical limit = 512 (positional encoding)
Decoder:         max_seq_length = 1024, but practical limit = 512 (positional encoding)
Positional Enc:  max_len = 512 (hardcoded in PositionalEncoding constructor)
```

**Important:** Even though the model supports 1024 sequence length, the positional encoding is limited to 512, which becomes the effective bottleneck.

## Future Improvements

### Option 1: Increase Positional Encoding Max Length

Modify `PositionalEncoding` to support longer sequences:

```cpp
// In model initialization
positional_encoding = std::make_unique<PositionalEncoding>(d_model, 1024);  // Instead of 512
```

Trade-offs:

- ✅ Allows longer context
- ❌ More memory usage
- ❌ Slower inference (attention is O(n²))

### Option 2: Sliding Window Context

Only keep the most recent N tokens of context:

```cpp
std::vector<int> truncate_to_window(std::vector<int> tokens, int window_size) {
    if (tokens.size() > window_size) {
        return std::vector<int>(tokens.end() - window_size, tokens.end());
    }
    return tokens;
}
```

### Option 3: Context Compression

Summarize older messages before encoding:

- Keep last 3-5 messages verbatim
- Summarize older messages into condensed form
- Reduces token count while preserving information

## Verification

After rebuilding, the GUI should:

1. ✅ Not show warnings about exceeding max_len
2. ✅ Have a "Clear" button between input field and "Send"
3. ✅ Start with empty conversation (no 513 token context)
4. ✅ Auto-truncate when reaching 480 tokens or 20 messages

## Files Modified

- **`src/ChatbotGUI.cpp`**:
  - Reduced `max_tokens` from 2048 to 480
  - Added `clearButton` to input area
  - Connected clear button to `onClearConversation()`

- **`src/ChatbotCLI.cpp`**:
  - Reduced `max_tokens` from 2048 to 480
  - CLI already has `/clear` command to reset context

- **`src/ChatbotGUI.hpp`**:
  - Already had `clearButton` and `onClearConversation()` declarations

## Testing

### Test GUI

```bash
# 1. Build GUI
./scripts/build_and_vocab.sh build-gui

# 2. Run GUI
./build/src/chatbot_gui --vocab vocab.txt --model chatbot_model.bin

# 3. Send a message
# Type "hello" and press Send

# 4. Verify no warnings about exceeding max_len in terminal

# 5. Send multiple messages to accumulate context

# 6. Click "Clear" button to reset conversation
```

### Test CLI

```bash
# 1. Build CLI
cd build && make chatbot -j$(nproc)

# 2. Run CLI
./build/src/chatbot --vocab vocab.txt --model chatbot_model.bin

# 3. Send a message
You> hello

# 4. Verify no warnings about exceeding max_len

# 5. Check conversation stats
You> /stats

# 6. Clear conversation
You> /clear
```

### Expected Behavior (Both GUI & CLI)

- No warnings about sequence length exceeding 512
- Context automatically truncates at 480 tokens
- Clear command/button resets conversation
- Context token count resets to 0

---

**Summary:** The input length was expanding because the conversation context was accumulating all message history (starting at 513 tokens). The fix limits context to 480 tokens and adds a Clear button for manual resets.
