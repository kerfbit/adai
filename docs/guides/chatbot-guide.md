# Chatbot CLI User Guide

A comprehensive guide to using the ADAI transformer-based chatbot command-line interface.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation and Setup](#installation-and-setup)
- [Basic Usage](#basic-usage)
- [Commands Reference](#commands-reference)
- [Generation Strategies](#generation-strategies)
- [Configuration Parameters](#configuration-parameters)
- [Conversation Management](#conversation-management)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)
- [Internals and Testing](#internals-and-testing)

---

## Quick Start

### Building the Chatbot

```bash
cd /path/to/adai
mkdir -p build && cd build
cmake ..
make chatbot -j$(nproc)
```

### Running the Chatbot

```bash
# From build directory
./src/chatbot

# Or with custom paths
./src/chatbot vocab.txt model.bin conversation.txt
```

### Your First Conversation

```text
╔═══════════════════════════════════════════════════════════╗
║          🤖 ADAI Transformer Chatbot CLI v1.0            ║
╚═══════════════════════════════════════════════════════════╝

You: Hello!
Bot: Hi there! How can I help you today?

You: What can you do?
Bot: I'm an AI assistant powered by a transformer model...

You: /exit
💾 Saving conversation...
✅ Conversation saved to: conversation_history.txt
👋 Goodbye!
```

---

## Installation and Setup

### Prerequisites

- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2017+)
- CMake 3.15 or higher
- Required libraries: `adai_models`, `adai_nlp`

### Building

```bash
# Clone repository
git clone https://github.com/rjv717/adai.git
cd adai

# Build
mkdir -p build && cd build
cmake ..
make chatbot

# Run
./src/chatbot
```

### File Requirements

**Vocabulary File** (`vocab.txt`):

```text
hello 100
world 50
test 25
...
```

**Model File** (`chatbot_model.bin`):

- Pre-trained transformer weights
- Binary format from training
- Optional (will use random initialization if missing)

### Default File Paths

| File | Default Path | Purpose |
| ------ | ------------- | --------- |
| Vocabulary | `vocab.txt` | BPE tokenizer vocabulary |
| Model | `chatbot_model.bin` | Pre-trained weights |
| Conversation | `conversation_history.txt` | Auto-saved conversations |

---

## Basic Usage

### Starting the Chatbot

```bash
# Use defaults
./src/chatbot

# Custom vocabulary
./src/chatbot my_vocab.txt

# Custom vocabulary and model
./src/chatbot my_vocab.txt my_model.bin

# All custom paths
./src/chatbot my_vocab.txt my_model.bin my_conversations.txt
```

### Command-Line Help

```bash
./src/chatbot --help
# or
./src/chatbot -h
```

Output:

```text
Usage: chatbot [vocab_file] [model_file] [conversation_save_file]

Default values:
  vocab_file: vocab.txt
  model_file: chatbot_model.bin
  conversation_save_file: conversation_history.txt
```

### Interactive Mode

Once running, simply type your messages:

```text
You: [your message here]
Bot: [response]
```

Press `Enter` to send. Type `/exit` or `/quit` to leave.

---

## Commands Reference

All commands start with `/` and are case-sensitive.

### Help and Information

#### `/help`

Display available commands and usage information.

```text
You: /help
```

#### `/stats`

Show conversation statistics.

```text
You: /stats

📊 Conversation Statistics:
  Total messages: 12
  Estimated tokens: 487
```

#### `/settings`

Display current generation parameters.

```text
You: /settings

⚙️  Current Settings:
  Strategy: nucleus
  Max length: 100
  Temperature: 1.0
  Top-p (nucleus): 0.9
  Top-k: 50
  Beam width: 5
```

### Conversation Management

#### `/clear`

Clear conversation history (resets context).

```text
You: /clear
✅ Conversation history cleared
```

#### `/save`

Manually save conversation to file.

```text
You: /save
✅ Conversation saved to: conversation_history.txt
```

#### `/load`

Load previously saved conversation.

```text
You: /load
✅ Conversation loaded from: conversation_history.txt
```

### Configuration

#### `/set <parameter> <value>`

Change generation parameters during conversation.

```text
You: /set temperature 0.7
✅ Temperature set to: 0.7

You: /set strategy greedy
✅ Generation strategy set to: greedy

You: /set max_length 150
✅ Max response length set to: 150
```

#### `/system <message>`

Set system message for conversation context.

```text
You: /system You are a helpful programming assistant
✅ System message set
```

### Exit

#### `/exit` or `/quit`

Exit the chatbot (auto-saves conversation).

```text
You: /exit
💾 Saving conversation...
✅ Conversation saved to: conversation_history.txt
👋 Goodbye!
```

---

## Generation Strategies

The chatbot supports 5 text generation strategies:

### 1. Greedy (Deterministic)

**Strategy:** Always selects highest probability token

```text
You: /set strategy greedy
```

**Best for:**

- Consistent, predictable responses
- Factual information retrieval
- Deterministic testing

**Pros:** Fast, deterministic
**Cons:** Can be repetitive, lacks creativity

---

### 2. Beam Search

**Strategy:** Maintains multiple candidate sequences

```text
You: /set strategy beam
You: /set beam_width 5
```

**Best for:**

- High-quality, coherent responses
- Translation tasks
- Structured output

**Pros:** Better quality than greedy
**Cons:** Slower, requires more memory

**Parameters:**

- `beam_width`: Number of beams (default: 5)

---

### 3. Sampling (Temperature-based)

**Strategy:** Samples from probability distribution

```text
You: /set strategy sampling
You: /set temperature 0.8
```

**Best for:**

- Creative responses
- Varied outputs
- Exploration

**Pros:** Diverse, creative
**Cons:** Can be inconsistent

**Parameters:**

- `temperature`: Controls randomness (0.1-2.0)
  - Low (0.1-0.5): More focused
  - Medium (0.6-1.0): Balanced
  - High (1.1-2.0): More random

---

### 4. Top-K Sampling

**Strategy:** Samples from top K most likely tokens

```text
You: /set strategy top-k
You: /set top_k 40
```

**Best for:**

- Controlled diversity
- Filtering unlikely tokens
- Quality + variety balance

**Pros:** Good balance of quality and diversity
**Cons:** Fixed cutoff can be limiting

**Parameters:**

- `top_k`: Number of top tokens (default: 50)

---

### 5. Nucleus (Top-P) Sampling ⭐ **Recommended**

**Strategy:** Samples from smallest set of tokens with cumulative probability ≥ p

```text
You: /set strategy nucleus
You: /set top_p 0.9
```

**Best for:**

- General conversation (default)
- Natural-sounding responses
- Adaptive quality control

**Pros:** Adaptive, high quality, natural
**Cons:** Slightly slower than greedy

**Parameters:**

- `top_p`: Cumulative probability threshold (0.0-1.0)
  - 0.9: Recommended default
  - 0.95: More diverse
  - 0.8: More focused

---

## Configuration Parameters

### Complete Parameter Reference

| Parameter | Aliases | Type | Range | Default | Description |
| ----------- | --------- | ------ | ------- | --------- | ------------- |
| `strategy` | - | string | See above | `nucleus` | Generation strategy |
| `length` | `max_length` | int | 1-1024 | 100 | Maximum response tokens |
| `temperature` | `temp` | float | 0.1-2.0 | 1.0 | Sampling randomness |
| `top_p` | `top-p` | float | 0.0-1.0 | 0.9 | Nucleus threshold |
| `top_k` | `top-k` | int | 1-500 | 50 | Top-K limit |
| `beam_width` | `beam-width` | int | 1-20 | 5 | Beam search width |

### Setting Parameters

```bash
# Using primary name
You: /set temperature 0.7

# Using alias
You: /set temp 0.7

# Both work identically
```

### Recommended Configurations

#### **Creative Writing**
```text
/set strategy nucleus
/set temperature 1.2
/set top_p 0.95
/set max_length 200
```

#### **Factual Q&A**
```text
/set strategy nucleus
/set temperature 0.5
/set top_p 0.8
/set max_length 100
```

#### **Code Generation**
```text
/set strategy beam
/set beam_width 5
/set temperature 0.3
/set max_length 150
```

#### **Casual Conversation**
```text
/set strategy nucleus
/set temperature 1.0
/set top_p 0.9
/set max_length 100
```

---

## Conversation Management

### Automatic Saving

Conversations are **automatically saved** when you exit:

```text
You: /exit
💾 Saving conversation...
✅ Conversation saved to: conversation_history.txt
```

### Manual Saving

Save at any time:

```text
You: /save
✅ Conversation saved to: conversation_history.txt
```

### Loading Previous Conversations

```text
You: /load
✅ Conversation loaded from: conversation_history.txt
```

**Note:** Loading replaces current conversation history.

### Conversation Limits

- **Max messages:** 20 (oldest auto-pruned)
- **Max tokens:** 2048 (context window)

View current usage:

```text
You: /stats
📊 Conversation Statistics:
  Total messages: 12
  Estimated tokens: 487
```

---

## Advanced Usage

### System Messages

Set context or personality:

```text
You: /system You are a helpful Python programming expert
✅ System message set

You: How do I read a file?
Bot: To read a file in Python, use the `open()` function...
```

### Multiple Conversations

Use different save files for different topics:

```bash
# Work conversations
./src/chatbot vocab.txt model.bin work_chat.txt

# Personal conversations
./src/chatbot vocab.txt model.bin personal_chat.txt
```

### Parameter Experimentation

Try different settings mid-conversation:

```text
You: Tell me a story
Bot: [creative story with default nucleus]

You: /set strategy greedy
You: /set temperature 0.3
You: Tell me a technical explanation
Bot: [more focused, deterministic response]
```

### Batch Processing

For scripted interactions, redirect input:

```bash
echo -e "Hello\nWhat is AI?\n/exit" | ./src/chatbot
```

---

## Troubleshooting

### Common Issues

#### **Error: Failed to load tokenizer**
```text
❌ Failed to load tokenizer from: vocab.txt
```

**Solution:** Ensure `vocab.txt` exists and is formatted correctly:

```text
token1 frequency1
token2 frequency2
...
```

#### **Warning: Model not found**
```text
ℹ️  No pre-trained model found. Using random initialization.
   (Train the model first for better results)
```

**Solution:** Either:

1. Provide a trained model file
2. Continue with random weights (for testing only)

#### **Conversation not loading**
```text
❌ Failed to load conversation
```

**Solution:**

- Check file exists and has read permissions
- Verify file format is correct
- Try `/clear` and start fresh

#### **Invalid command**
```text
❓ Unknown command. Type /help for available commands.
```

**Solution:** Check command spelling (commands are case-sensitive)

### Performance Tips

1. **Faster responses:** Use `greedy` strategy
2. **Lower memory:** Reduce `beam_width`
3. **Better quality:** Use `nucleus` with `top_p=0.9`
4. **Longer responses:** Increase `max_length`

---

## Examples

### Example 1: Basic Conversation

```text
You: Hello!
Bot: Hi there! How can I help you today?

You: What's the weather like?
Bot: I don't have access to real-time weather data, but I can help you find weather information or discuss weather-related topics.

You: /stats
📊 Conversation Statistics:
  Total messages: 4
  Estimated tokens: 156
```

### Example 2: Adjusting Creativity

```text
You: Tell me a creative story about a robot
Bot: [creative, diverse story]

You: /set temperature 1.5
You: Tell me another creative story
Bot: [more wild and creative story]

You: /set temperature 0.3
You: Summarize the stories
Bot: [focused, concise summary]
```

### Example 3: Role-Playing

```text
You: /system You are a Shakespearean actor
✅ System message set

You: Tell me about thy day
Bot: Ah, 'twas a day most splendid, good fellow! The morn did break with golden rays...

You: /clear
✅ Conversation history cleared

You: /system You are a pirate
✅ System message set

You: Tell me about your day
Bot: Arrr, matey! 'Twas a fine day on the high seas...
```

### Example 4: Comparing Strategies

```text
You: Explain quantum computing

You: /set strategy greedy
Bot: [deterministic explanation]

You: /set strategy nucleus
You: Explain quantum computing
Bot: [more natural explanation]

You: /set strategy beam
You: Explain quantum computing
Bot: [higher quality explanation]
```

---

## Internals and Testing

For developers and advanced users:

- **Implementation Details:** `chatbot-cli-internals.md`
- **Testing Documentation:** `../testing/chatbot-cli-tests.md`
- **Header File:** `src/ChatbotCLI.hpp`
- **Source Code:** `src/ChatbotCLI.cpp`
- **Main Entry:** `src/ChatbotCLI_main.cpp`
- **Test Suite:** `tests/chatbotcli_improved_test.cpp` (23 tests)
- **Legacy Tests:** `tests/chatbotcli_test.cpp` (83 tests)

### Building Tests

```bash
cd build
make chatbotcliImprovedTests
./tests/chatbotcliImprovedTests
```

### Running All Tests

```bash
cd build
ctest -R ChatbotCLI
```

---

## Summary

The ADAI Chatbot CLI provides:

✅ **5 generation strategies** for different use cases
✅ **Flexible configuration** via runtime commands
✅ **Conversation management** with auto-save
✅ **System messages** for context control
✅ **Colored output** for better UX
✅ **Comprehensive help** system
✅ **Modern C++** implementation (smart pointers, string_view)
✅ **Fully tested** (106 total tests across 2 suites)

**Perfect for:**

- Testing transformer models
- Interactive AI conversations
- Experimenting with generation strategies
- Educational demonstrations
- Production chatbot deployments

**Get Started:**

```bash
./src/chatbot
You: Hello!
```

Enjoy your conversations! 🤖
