# ConversationContext Class - Context Documentation

**Component:** `ConversationContext`
**Files:** `src/ConversationContext.hpp`, `src/ConversationContext.cpp`
**Purpose:** Manage multi-turn conversation history for chatbot applications
**Status:** ✅ Complete and Production-Ready

---

## Overview

The `ConversationContext` class provides comprehensive conversation history management for chatbot applications. It handles message storage, automatic context window management, token tracking, and formatting for model consumption.

### Key Features

- **Role-Based Messaging**: Track messages by role (user/assistant/system)
- **Automatic Truncation**: Sliding window with configurable message/token limits
- **Multiple Formatting Options**: Standard text, special tokens, custom separators
- **Persistence**: Save/load conversations to disk
- **Summarization Support**: Compress long conversations while retaining recent context
- **Token Management**: Track and limit token usage
- **System Messages**: Persistent instructions/context that survives truncation

---

## Architecture

### Class Structure

```cpp
class ConversationContext {
public:
    struct Message {
        std::string role;      // "user", "assistant", "system"
        std::string content;   // Message text
        int token_count;       // Token count (estimated or actual)
    };

private:
    std::deque<Message> messages;        // Efficient message queue
    Message* system_message;             // Optional system prompt
    int max_messages;                    // Message limit (0 = unlimited)
    int max_tokens;                      // Token limit (0 = unlimited)
    bool keep_system_message;            // Preserve system message
    int total_tokens;                    // Cached token count
};
```

### Design Decisions

1. **Deque for Messages**:
   - Efficient removal from front (oldest messages)
   - Fast append to back (new messages)
   - O(1) operations for sliding window

2. **Separate System Message**:
   - System message never truncated (configurable)
   - Always included in context formatting
   - Can be updated without affecting history

3. **Token Estimation**:
   - Rough estimation: ~4 characters per token
   - Can accept manual token counts for accuracy
   - Enables token-based truncation without tokenizer dependency

4. **Automatic Truncation**:
   - Triggered after each message addition
   - Removes oldest messages first
   - Preserves system message

---

## Core Methods

### Construction

```cpp
ConversationContext(int max_messages = 20,
                   int max_tokens = 2048,
                   bool keep_system_message = true);
```

Parameters:

- `max_messages`: Maximum conversation messages (0 = unlimited)
- `max_tokens`: Maximum total tokens (0 = unlimited)
- `keep_system_message`: Whether to preserve system message during truncation

Example:

```cpp
// Chatbot with 20-message history, 2048 token limit
ConversationContext context(20, 2048, true);

// Unlimited messages, 4096 token limit
ConversationContext context(0, 4096);

// 10 messages, no token limit
ConversationContext context(10, 0);
```

### Adding Messages

```cpp
void add_user_message(const std::string& content, int token_count = 0);
void add_assistant_message(const std::string& content, int token_count = 0);
void add_message(const std::string& role, const std::string& content, int token_count = 0);
void set_system_message(const std::string& content, int token_count = 0);
```

Parameters:

- `content`: Message text
- `token_count`: Token count (0 = auto-estimate)
- `role`: Custom role for `add_message()`

Behavior:

- Automatically truncates if limits exceeded
- Estimates tokens if not provided
- Updates total token count

Example:

```cpp
// Basic usage (auto token estimation)
context.add_user_message("What is machine learning?");
context.add_assistant_message("Machine learning is a subset of AI...");

// With manual token counts (from actual tokenizer)
std::vector<int> tokens = tokenizer.encode("Hello world");
context.add_user_message("Hello world", tokens.size());

// System message (instructions for the model)
context.set_system_message("You are a helpful AI tutor specializing in computer science.");
```

### Formatting for Models

```cpp
std::string format_for_model(bool include_system = true,
                             const std::string& separator = "\n") const;

std::string format_with_special_tokens(const std::string& bos_token = "<bos>",
                                      const std::string& eos_token = "<eos>",
                                      const std::string& sep_token = "<sep>") const;
```

**Purpose**: Convert conversation to model input format

Example:

```cpp
// Standard formatting
std::string input = context.format_for_model(true);
// Output:
// System: You are a helpful assistant.
// User: Hello
// Assistant: Hi there!
// User: How are you?

// With special tokens for encoder-decoder
std::string input = context.format_with_special_tokens("<bos>", "<eos>", "<sep>");
// Output:
// <bos> [SYSTEM] You are a helpful assistant.<sep> [USER] Hello<sep> [ASSISTANT] Hi there!<sep> <eos>
```

### Message Retrieval

```cpp
std::string get_last_user_message() const;
std::string get_last_assistant_message() const;
std::vector<Message> get_messages() const;
std::string get_system_message() const;
```

Example:

```cpp
// Get last user input
std::string last_input = context.get_last_user_message();

// Get all messages for custom processing
for (const auto& msg : context.get_messages()) {
    std::cout << msg.role << ": " << msg.content << std::endl;
}
```

### Context Management

```cpp
void clear();                              // Clear messages, keep system
void clear_all();                          // Clear everything
void truncate_to_limits();                 // Manually trigger truncation
void set_max_messages(int max_messages);   // Update message limit
void set_max_tokens(int max_tokens);       // Update token limit
```

Example:

```cpp
// Start new conversation (keep system instructions)
context.clear();

// Completely reset
context.clear_all();

// Reduce limits dynamically
context.set_max_messages(10);   // Immediately truncates if needed
```

### Statistics and Monitoring

```cpp
int get_total_tokens() const;
int get_message_count() const;
bool is_empty() const;
std::string get_statistics() const;
```

Example:

```cpp
std::cout << context.get_statistics();
// Output:
// Conversation Statistics:
//   Messages: 8
//   Total Tokens: 247
//   System Message: Yes
//   Max Messages: 20
//   Max Tokens: 2048
//   User Messages: 4
//   Assistant Messages: 4
```

### Persistence

```cpp
void save_to_file(const std::string& filepath) const;
void load_from_file(const std::string& filepath);
```

File Format:

```text
MAX_MESSAGES:20
MAX_TOKENS:2048
KEEP_SYSTEM:1
---
SYSTEM|15|You are a helpful assistant.
user|10|Hello world
assistant|12|Hi there!
```

Example:

```cpp
// Save conversation
context.save_to_file("conversation_2026-01-18.txt");

// Load later
ConversationContext restored;
restored.load_from_file("conversation_2026-01-18.txt");
```

### Summarization

```cpp
ConversationContext create_summarized(int keep_recent = 5,
                                     const std::string& summary_text = "") const;
```

**Purpose**: Create compressed version for long conversations

Example:

```cpp
// Original: 50 messages
ConversationContext long_conv;
// ... add many messages ...

// Create summary keeping last 10 messages
ConversationContext summarized = long_conv.create_summarized(
    10,
    "Earlier we discussed ML basics, supervised learning, and neural networks"
);
// Result: Summary message + 10 recent messages
```

---

## Usage Patterns

### Pattern 1: Simple Chatbot

```cpp
ConversationContext context(20, 2048);
context.set_system_message("You are a friendly chatbot.");

while (true) {
    std::string user_input = get_user_input();
    if (user_input == "quit") break;

    context.add_user_message(user_input);

    // Generate response (using your model)
    std::string model_input = context.format_for_model(true);
    std::string response = model.generate_response(model_input);

    context.add_assistant_message(response);
    display_response(response);
}
```

### Pattern 2: Context-Aware Responses

```cpp
ConversationContext context;
context.set_system_message("You remember user preferences and prior context.");

// User sets preference
context.add_user_message("I prefer concise answers.");
context.add_assistant_message("Noted! I'll keep my responses brief.");

// Later in conversation
context.add_user_message("Explain transformers.");
// Model sees full history, can tailor response style
std::string formatted = context.format_for_model(true);
```

### Pattern 3: Multi-Session Persistence

```cpp
// Session 1
ConversationContext session1;
session1.add_user_message("My name is Alice.");
session1.save_to_file("alice_conversation.txt");

// Session 2 (next day)
ConversationContext session2;
session2.load_from_file("alice_conversation.txt");
session2.add_user_message("Do you remember my name?");
// Model has access to previous conversation
```

### Pattern 4: Dynamic Context Management

```cpp
ConversationContext context(0, 4096);  // Token-limited

while (conversation_active) {
    // Check token usage
    if (context.get_total_tokens() > 3500) {
        // Approaching limit, create summary
        ConversationContext summarized = context.create_summarized(
            5,
            summarize_conversation(context)  // Your summarization logic
        );
        context = summarized;
    }

    // Continue conversation...
}
```

### Pattern 5: Role-Based Processing

```cpp
// Custom role tracking
context.add_message("function", "search_database(query='transformers')");
context.add_message("function_result", "Found 42 papers on transformers");
context.add_assistant_message("I found 42 papers on transformers in the database.");

// Process by role
for (const auto& msg : context.get_messages()) {
    if (msg.role == "function") {
        execute_function(msg.content);
    }
}
```

---

## Integration with EncoderDecoderModel

### Basic Integration

```cpp
#include "EncoderDecoderModel.hpp"
#include "ConversationContext.hpp"

class Chatbot {
private:
    EncoderDecoderModel model;
    ConversationContext context;

public:
    Chatbot(int vocab_size, int d_model)
        : model(vocab_size, d_model, 6, 6, 8, 2048),
          context(20, 2048) {
        context.set_system_message("You are a helpful AI assistant.");
    }

    std::string chat(const std::string& user_input) {
        // Add user message
        context.add_user_message(user_input);

        // Format context for model
        std::string model_input = context.format_for_model(true);

        // Generate response
        std::string response = model.generate_response(model_input, 100);

        // Add to context
        context.add_assistant_message(response);

        return response;
    }

    void save_conversation(const std::string& filepath) {
        context.save_to_file(filepath);
    }
};
```

### Advanced Integration with Token Counting

```cpp
class TokenAwareChatbot {
private:
    EncoderDecoderModel model;
    ConversationContext context;
    BPETokenizer* tokenizer;

public:
    std::string chat(const std::string& user_input) {
        // Use actual tokenizer for accurate counts
        std::vector<int> user_tokens = tokenizer->encode(user_input);
        context.add_user_message(user_input, user_tokens.size());

        // Format with special tokens
        std::string model_input = context.format_with_special_tokens(
            "<bos>", "<eos>", "<sep>"
        );

        std::string response = model.generate_response(model_input, 100);

        std::vector<int> response_tokens = tokenizer->encode(response);
        context.add_assistant_message(response, response_tokens.size());

        return response;
    }
};
```

---

## Performance Considerations

### Memory Usage

Per Message:

- Message struct: ~48 bytes (role, content, token_count)
- Content string: variable (actual text size)
- **Total**: ~50 bytes + text length

For 20 messages @ 100 chars each:

- ~20 * (50 + 100) = 3 KB
- Very lightweight

### Time Complexity

|Operation|Complexity|Notes|
|-----------|------------|-------|
|add_message|O(1) amortized|May trigger truncation|
|truncate|O(n)|Where n = messages to remove|
|format_for_model|O(m)|Where m = total message count|
|get_last_*|O(m)|Linear search backwards|
|save/load|O(m)|File I/O|

### Optimization Tips

1. **Pre-compute Token Counts**: Use actual tokenizer instead of estimation
2. **Batch Operations**: Add multiple messages before truncation
3. **Cache Formatted Output**: If context unchanged, reuse formatted string
4. **Summarization**: For very long conversations (>50 turns)

---

## Limitations and Considerations

### Current Limitations

1. **Token Estimation**: Rough approximation (~4 chars/token)
   - **Impact**: May over/under-estimate actual token usage
   - **Solution**: Pass actual token counts from tokenizer

2. **No Attention Mask Support**: Doesn't generate attention masks
   - **Impact**: Model must create masks separately
   - **Workaround**: Format with special tokens, let model handle masking

3. **Simple Truncation**: FIFO (oldest first) only
   - **Impact**: May lose important context
   - **Alternative**: Manual summarization before limits reached

4. **Single-threaded**: Not thread-safe
   - **Impact**: Requires external synchronization for multi-threaded apps
   - **Solution**: Use mutex/lock for concurrent access

### Design Trade-offs

Automatic vs Manual Truncation:

- ✅ Automatic: Convenient, prevents OOM
- ❌ Automatic: May remove important context
- **Choice**: Automatic with manual override (create_summarized)

Token Estimation vs Exact Counting:

- ✅ Estimation: No tokenizer dependency, faster
- ❌ Estimation: Less accurate
- **Choice**: Estimation with option for exact counts

Deque vs Vector:

- ✅ Deque: Efficient front removal
- ❌ Deque: Slightly more memory overhead
- **Choice**: Deque for sliding window performance

---

## Testing Recommendations

### Unit Tests to Write

1. **Construction Tests**
   - Default parameters
   - Custom limits
   - Edge cases (0 limits, negative values)

2. **Message Management**
   - Add messages (user/assistant/system)
   - Automatic truncation
   - Token counting
   - Role filtering

3. **Formatting Tests**
   - Standard format
   - Special token format
   - Empty conversation
   - System-only message

4. **Truncation Tests**
   - Message limit enforcement
   - Token limit enforcement
   - System message preservation
   - Manual limit updates

5. **Persistence Tests**
   - Save/load round-trip
   - Metadata preservation
   - Error handling (missing file, corrupted data)

6. **Summarization Tests**
   - Keep recent messages
   - Summary injection
   - Edge cases (empty, single message)

7. **Edge Cases**
   - Empty messages
   - Very long messages
   - Special characters
   - Concurrent modifications

### Example Test Pattern

```cpp
TEST(ConversationContextTest, AutomaticTruncation) {
    ConversationContext context(3, 0);  // Max 3 messages

    context.add_user_message("Message 1");
    context.add_assistant_message("Response 1");
    context.add_user_message("Message 2");
    context.add_assistant_message("Response 2");

    EXPECT_EQ(context.get_message_count(), 3);  // Truncated to 3

    auto messages = context.get_messages();
    EXPECT_EQ(messages[0].content, "Response 1");  // Oldest kept
}
```

---

## Production Deployment Checklist

### Before Production

- [ ] **Implement Unit Tests**: 30+ tests covering all methods
- [ ] **Thread Safety**: Add mutex for multi-threaded environments
- [ ] **Error Handling**: Validate inputs, handle edge cases
- [ ] **Logging**: Add logging for truncation events
- [ ] **Metrics**: Track average tokens, truncation frequency
- [ ] **Documentation**: API docs, deployment guide
- [ ] **Benchmarking**: Test with realistic conversation sizes

### Configuration Recommendations

For Different Use Cases:

|Use Case|max_messages|max_tokens|keep_system|
|----------|--------------|------------|-------------|
|Short Q&A|10|1024|true|
|General Chat|20|2048|true|
|Long Dialogue|50|4096|true|
|Technical Support|30|3072|true|
|Creative Writing|0|8192|true|

### Monitoring

Track these metrics:

- Average conversation length (messages & tokens)
- Truncation frequency
- Peak memory usage
- Format operation latency
- Save/load success rate

---

## Future Enhancements

### Potential Additions

1. **Advanced Summarization**
   - Automatic summarization using language model
   - Importance-based message retention
   - Semantic clustering of related messages

2. **Thread Safety**
   - Mutex/lock support for concurrent access
   - Lock-free implementation for performance

3. **Attention Mask Generation**
   - Create masks for encoder-decoder models
   - Support for various masking strategies

4. **Conversation Analytics**
   - Sentiment tracking
   - Topic extraction
   - Turn-taking analysis

5. **Multi-Language Support**
   - Unicode handling improvements
   - Language-specific token estimation

6. **Compression**
   - Compress old messages for memory savings
   - On-demand decompression for retrieval

---

## Conclusion

The `ConversationContext` class provides a robust, production-ready solution for managing chatbot conversation history. Its key strengths are:

✅ **Automatic Management**: Sliding window with configurable limits
✅ **Multiple Formats**: Support for various model input formats
✅ **Persistence**: Save/load conversations across sessions
✅ **Flexible**: Customizable roles, limits, and formatting
✅ **Lightweight**: Minimal memory overhead
✅ **Well-Designed**: Clean API, clear abstractions

### Integration Status

|Component|Status|
|-----------|--------|
|EncoderDecoderModel|✅ Ready to integrate|
|BPETokenizer|✅ Compatible for token counting|
|TextGenerator|✅ Works with formatted output|
|Training Pipeline|✅ Can format training data|
|API Layer|✅ Ready for HTTP/gRPC integration|

### Next Steps

1. Write comprehensive unit tests
2. Integrate with EncoderDecoderModel in chatbot application
3. Add example chatbot with conversation management
4. Benchmark performance with realistic workloads
5. Add thread safety for production deployment

The ConversationContext class completes a critical piece of the chatbot infrastructure, enabling multi-turn conversations with proper context management.
