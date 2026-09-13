# Chatbot CLI User Guide

A comprehensive guide to using the ADAI transformer-based chatbot command-line interface.

> **Note (TD-053, resolved September 13, 2026):** `ChatbotCLI` was re-architected at some point
> from a standalone CLI that loaded its own vocabulary/model files directly into a thin HTTP
> client for `chatbot_api_server` (it now takes `[server_url] [conversation_save_file]`, not
> `[vocab_file] [model_file] [conversation_save_file]`), and this guide was never fully updated to
> match at the time — the vocabulary/model file examples and error messages below have now been
> corrected to describe the current API-client architecture. Separately, `/save`, `/load`, and
> auto-save on exit were previously non-functional stubs (`ChatbotCLI` has no local conversation
> state — history lives entirely server-side); they are now real, and are documented as such
> throughout this guide, including in the "Conversation Management" section below. This pass was
> a targeted correction of known-stale text, not a line-by-line re-verification against a live
> `chatbot` + `chatbot_api_server` pair — flag anything else that looks off.

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
# From build directory — connects to chatbot_api_server at the default URL
# (http://localhost:8080), which must already be running
./src/chatbot

# Or with a custom server URL and conversation-save path
./src/chatbot http://localhost:8080 conversation.txt
```

Note: `chatbot` is an API client for `chatbot_api_server`, not a standalone binary that loads a
vocabulary/model file itself — `chatbot_api_server` must be running first (see
[../deployment/README.md](../deployment/README.md)).

### Your First Conversation

```text
╔═══════════════════════════════════════════════════════════╗
║          🤖 ADAI Transformer Chatbot CLI v1.0             ║
╚═══════════════════════════════════════════════════════════╝

You: Hello!
Bot: Hi there! How can I help you today?

You: What can you do?
Bot: I'm an AI assistant powered by a transformer model...

You: /exit
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

`chatbot` itself needs no vocabulary or model file — it's an HTTP client, and
`chatbot_api_server` (the process actually holding the vocabulary and model) is what needs
those. See [../deployment/README.md](../deployment/README.md) for `chatbot_api_server`'s
requirements.

### Default File Paths (for `chatbot`)

|Argument|Default|Purpose|
|------|-------------|---------|
|`server_url`|`http://localhost:8080`|`chatbot_api_server` to connect to|
|`conversation_save_file`|`conversation_history.txt`|Local path used by `/save`, `/load`, and auto-save on exit|

---

## Basic Usage

### Starting the Chatbot

```bash
# Use defaults (connects to http://localhost:8080)
./src/chatbot

# Custom server URL
./src/chatbot http://192.168.1.10:8080

# Custom server URL and conversation-save path
./src/chatbot http://192.168.1.10:8080 my_conversations.txt
```

### Command-Line Help

```bash
./src/chatbot --help
# or
./src/chatbot -h
```

Output (verified against `src/ChatbotCLI_main.cpp` September 8, 2026):

```text
Usage: chatbot [server_url] [conversation_save_file]

Default values:
  server_url: http://localhost:8080
  conversation_save_file: conversation_history.txt

Example: chatbot http://localhost:8080
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

#### `/save` and `/load`

`ChatbotCLI` holds no conversation history of its own — the full conversation lives server-side,
in `chatbot_api_server`'s `Session`. `/save` asks the server to export the active session's
history (over `POST /chat/session/export`) and writes the result to the local
`conversation_save_file` path; `/load` reads that local file back and asks the server to import
it (`POST /chat/session/import`) into a session — reusing the current one if there is one, or
having the server allocate a fresh one otherwise.

```text
You: /save
✅ Conversation saved to conversation_history.txt
```

```text
You: /load
✅ Conversation loaded from conversation_history.txt (4 messages)
```

`/save` before any message has been sent (no active session yet) reports the same "nothing to
save" condition rather than writing an empty file:

```text
You: /save
❌ Nothing to save yet — send a message first.
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

Exit the chatbot. `/exit` and `/quit` automatically save the conversation first (equivalent to
running `/save`), skipped silently if no message was ever sent in the session:

```text
You: /exit
✅ Conversation saved to conversation_history.txt
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

Best for:

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

Best for:

- High-quality, coherent responses
- Translation tasks
- Structured output

**Pros:** Better quality than greedy
**Cons:** Slower, requires more memory

Parameters:

- `beam_width`: Number of beams (default: 5)

---

### 3. Sampling (Temperature-based)

**Strategy:** Samples from probability distribution

```text
You: /set strategy sampling
You: /set temperature 0.8
```

Best for:

- Creative responses
- Varied outputs
- Exploration

**Pros:** Diverse, creative
**Cons:** Can be inconsistent

Parameters:

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

Best for:

- Controlled diversity
- Filtering unlikely tokens
- Quality + variety balance

**Pros:** Good balance of quality and diversity
**Cons:** Fixed cutoff can be limiting

Parameters:

- `top_k`: Number of top tokens (default: 50)

---

### 5. Nucleus (Top-P) Sampling ⭐ **Recommended**

**Strategy:** Samples from smallest set of tokens with cumulative probability ≥ p

```text
You: /set strategy nucleus
You: /set top_p 0.9
```

Best for:

- General conversation (default)
- Natural-sounding responses
- Adaptive quality control

**Pros:** Adaptive, high quality, natural
**Cons:** Slightly slower than greedy

Parameters:

- `top_p`: Cumulative probability threshold (0.0-1.0)
  - 0.9: Recommended default
  - 0.95: More diverse
  - 0.8: More focused

---

## Configuration Parameters

### Complete Parameter Reference

|Parameter|Aliases|Type|Range|Default|Description|
|-----------|---------|------|-------|---------|-------------|
|`strategy`|-|string|See above|`nucleus`|Generation strategy|
|`length`|`max_length`|int|1-1024|100|Maximum response tokens|
|`temperature`|`temp`|float|0.1-2.0|1.0|Sampling randomness|
|`top_p`|`top-p`|float|0.0-1.0|0.9|Nucleus threshold|
|`top_k`|`top-k`|int|1-500|50|Top-K limit|
|`beam_width`|`beam-width`|int|1-20|5|Beam search width|

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

## Conversation History

Conversation history itself lives entirely server-side, in `chatbot_api_server`'s `Session` for
this client's `session_id` — `ChatbotCLI` holds no copy of its own. `/save` and `/load` transport
that server-side state to and from a local file over HTTP (see [`/save` and
`/load`](#save-and-load) above); there is no other local persistence.

### Automatic Saving

Conversations are **automatically saved** when you exit, unless no message was ever sent:

```text
You: /exit
✅ Conversation saved to conversation_history.txt
👋 Goodbye!
```

### Manual Saving

Save at any time:

```text
You: /save
✅ Conversation saved to conversation_history.txt
```

### Loading Previous Conversations

```text
You: /load
✅ Conversation loaded from conversation_history.txt (4 messages)
```

**Note:** Loading replaces the current session's conversation history on the server (or, if this
client has no active session yet, creates a new one populated from the file).

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
./src/chatbot http://localhost:8080 work_chat.txt

# Personal conversations
./src/chatbot http://localhost:8080 personal_chat.txt
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

#### **Error: cannot connect to the server**

`chatbot` is an HTTP client — it loads no tokenizer or model file itself, so a failure here means
`chatbot_api_server` isn't reachable at the configured `server_url`, not a local file problem:

```text
Error: Connection failed
```

**Solution:**

1. Confirm `chatbot_api_server` is running (`curl http://localhost:8080/health`)
2. Check the `server_url` argument matches where it's actually listening
3. See [../deployment/README.md](../deployment/README.md) for starting `chatbot_api_server`,
   including its own tokenizer/model file requirements (`chatbot` itself has none)

#### **Conversation not loading**

```text
❌ No saved conversation found at 'conversation_history.txt'
```

or, if the server rejects the import (e.g. the file's data is corrupt or empty):

```text
❌ Saved conversation file 'conversation_history.txt' is empty
```

Solution:

- Check the file exists (relative to the directory `chatbot` was run from) and has read permissions
- Verify the file is one `/save` actually wrote — it holds `ConversationContext::serialize()`'s
  own format, not arbitrary text
- Confirm `chatbot_api_server` is reachable — `/load` needs a live round trip to import the data,
  same as `/save` needs one to export it
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

Perfect for:

- Testing transformer models
- Interactive AI conversations
- Experimenting with generation strategies
- Educational demonstrations
- Production chatbot deployments

Get Started:

```bash
./src/chatbot
You: Hello!
```

Enjoy your conversations! 🤖
