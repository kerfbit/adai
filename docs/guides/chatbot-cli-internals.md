# ChatbotCLI Context Documentation

## Purpose

`ChatbotCLI` is an interactive command-line interface for the ADAI transformer-based chatbot. It provides a user-friendly terminal application with conversation management, multiple generation strategies, configurable parameters, and persistent conversation history.

## File Location

**Implementation:** `src/ChatbotCLI.cpp`

## Dependencies

### Core Components
- **EncoderDecoderModel:** Main transformer model for text generation
- **ConversationContext:** Manages conversation history and context
- **TextGenerator:** Text generation utilities (imported but model's built-in generation used)
- **BPETokenizer:** Byte-pair encoding tokenizer

### Standard Libraries
```cpp
#include <iostream>      // Console I/O
#include <string>        // String handling
#include <fstream>       // File I/O
#include <ctime>         // Time utilities
#include <iomanip>       // I/O formatting
```

## ANSI Color Codes

The CLI uses ANSI escape codes for colored terminal output:

```cpp
#define COLOR_RESET   "\033[0m"      // Reset to default
#define COLOR_USER    "\033[1;36m"   // Cyan (user messages)
#define COLOR_BOT     "\033[1;32m"   // Green (bot responses)
#define COLOR_SYSTEM  "\033[1;33m"   // Yellow (system messages)
#define COLOR_ERROR   "\033[1;31m"   // Red (error messages)
```

**Visual Example:**
```
You: Hello!                    (Cyan)
Bot: Hi there! How can I help? (Green)
✅ Message saved               (Yellow)
❌ Error loading file          (Red)
```

## ChatbotCLI Class

### Private Members

```cpp
// Core components
BPETokenizer* tokenizer;              // Vocabulary and tokenization
EncoderDecoderModel* model;           // Transformer model
ConversationContext* context;         // Conversation history manager

// File paths
std::string model_path;               // Path to model weights file
std::string vocab_path;               // Path to vocabulary file
std::string conversation_save_path;   // Path to save conversations

// Generation parameters
int max_response_length;              // Maximum tokens in response (default: 100)
float temperature;                    // Sampling temperature (default: 1.0)
float top_p;                          // Nucleus sampling threshold (default: 0.9)
int top_k;                            // Top-k sampling limit (default: 50)
int beam_width;                       // Beam search width (default: 5)
std::string generation_strategy;      // Current strategy (default: "nucleus")
```

### Constructor

```cpp
ChatbotCLI(const std::string& vocab_file, 
           const std::string& model_file,
           const std::string& conv_save_file = "conversation_history.txt")
```

**Parameters:**
- `vocab_file`: Path to BPE vocabulary file
- `model_file`: Path to pre-trained model weights
- `conv_save_file`: Path for saving conversation history (optional, default: "conversation_history.txt")

**Initialization:**
- Sets file paths
- Initializes default generation parameters
- Sets all component pointers to `nullptr`

**Default Generation Parameters:**
```cpp
max_response_length = 100
temperature = 1.0f
top_p = 0.9f
top_k = 50
beam_width = 5
generation_strategy = "nucleus"
```

### Destructor

```cpp
~ChatbotCLI()
```

**Purpose:** Clean up allocated resources

**Calls:** `cleanup()` to delete all dynamically allocated objects

### Public Methods

#### initialize()

```cpp
bool initialize()
```

**Purpose:** Initialize all chatbot components

**Process:**

1. **Load Tokenizer**
   ```cpp
   tokenizer = new BPETokenizer();
   tokenizer->load_vocab(vocab_path);
   ```
   - Creates tokenizer instance
   - Loads vocabulary from file
   - Reports vocabulary size

2. **Create Model**
   ```cpp
   model = new EncoderDecoderModel(
       512,    // d_model
       8,      // num_heads
       2048,   // d_ff
       6,      // num_encoder_layers
       6,      // num_decoder_layers
       tokenizer->get_vocab_size(),
       1024    // max_seq_length
   );
   ```

3. **Load Pre-trained Weights** (if available)
   ```cpp
   if (model_file exists) {
       model->load_model(model_path);
   } else {
       // Use random initialization (warn user)
   }
   ```

4. **Create Conversation Context**
   ```cpp
   context = new ConversationContext(
       20,     // max 20 messages
       2048    // max 2048 tokens
   );
   ```

**Returns:** `true` if successful

**Output Messages:**
```
🤖 Initializing Chatbot...
📚 Loading tokenizer from: vocab.txt
✅ Tokenizer loaded (vocab size: 5000)
🧠 Initializing transformer model...
💾 Loading model weights from: chatbot_model.bin
✅ Model weights loaded successfully!
✅ Text generator initialized
✅ Conversation manager initialized
🎉 Chatbot ready!
```

#### cleanup()

```cpp
void cleanup()
```

**Purpose:** Release allocated memory

**Process:**
```cpp
if (model) delete model;
if (tokenizer) delete tokenizer;
if (context) delete context;
```

**Called By:** Destructor

#### print_welcome()

```cpp
void print_welcome()
```

**Purpose:** Display welcome message and help information

**Output:**
```
╔═══════════════════════════════════════════════════════════╗
║          🤖 ADAI Transformer Chatbot CLI v1.0            ║
╚═══════════════════════════════════════════════════════════╝

Commands:
  /help         - Show this help message
  /clear        - Clear conversation history
  /save         - Save conversation to file
  /load         - Load conversation from file
  /stats        - Show conversation statistics
  /settings     - Show current settings
  /set <param>  - Change generation parameter
  /system <msg> - Set system message
  /exit, /quit  - Exit the chatbot

Generation strategies:
  greedy, beam, sampling, top-k, nucleus
```

#### print_stats()

```cpp
void print_stats()
```

**Purpose:** Display conversation statistics

**Output:**
```
📊 Conversation Statistics:
  Total messages: 10
  Estimated tokens: 245
```

**Data Source:**
- `context->get_message_count()` - Total messages in conversation
- `context->get_total_tokens()` - Estimated token count

#### print_settings()

```cpp
void print_settings()
```

**Purpose:** Display current generation settings

**Output:**
```
⚙️  Current Settings:
  Strategy: nucleus
  Max length: 100
  Temperature: 1.0
  Top-p (nucleus): 0.9
  Top-k: 50
  Beam width: 5
```

#### handle_command()

```cpp
void handle_command(const std::string& command)
```

**Purpose:** Process user commands

**Supported Commands:**

| Command | Description | Action |
|---------|-------------|--------|
| `/help` | Show help message | Calls `print_welcome()` |
| `/clear` | Clear conversation | `context->clear()` |
| `/save` | Save conversation | `context->save_to_file(path)` |
| `/load` | Load conversation | `context->load_from_file(path)` |
| `/stats` | Show statistics | Calls `print_stats()` |
| `/settings` | Show settings | Calls `print_settings()` |
| `/set <param> <value>` | Change parameter | Calls `handle_setting()` |
| `/system <message>` | Set system message | `context->set_system_message()` |
| `/exit`, `/quit` | Exit chatbot | Exits main loop |

**Error Handling:**
- Unknown commands display error message
- Save/load failures caught and reported

**Examples:**
```
/help                    # Show help
/clear                   # Clear history
/save                    # Save to default file
/stats                   # Show statistics
/set strategy beam       # Change to beam search
/set temperature 0.7     # Set temperature
/system You are helpful  # Set system message
```

#### handle_setting()

```cpp
void handle_setting(const std::string& setting)
```

**Purpose:** Change generation parameters

**Format:** `<parameter> <value>`

**Parameters:**

| Parameter | Aliases | Type | Description | Example |
|-----------|---------|------|-------------|---------|
| `strategy` | - | string | Generation strategy | `strategy nucleus` |
| `length` | `max_length` | int | Max response tokens | `length 150` |
| `temperature` | `temp` | float | Sampling temperature | `temp 0.8` |
| `top_p` | `top-p` | float | Nucleus threshold | `top_p 0.95` |
| `top_k` | `top-k` | int | Top-k limit | `top_k 40` |
| `beam_width` | `beam-width` | int | Beam search width | `beam_width 10` |

**Valid Strategies:**
- `greedy` - Always select highest probability token
- `beam` - Beam search with configurable width
- `sampling` - Temperature-based sampling
- `top-k` - Sample from top-k tokens
- `nucleus` - Nucleus (top-p) sampling (recommended)

**Validation:**
- Strategy must be one of the valid options
- Numeric values parsed with `std::stoi()` / `std::stof()`
- Invalid parameters report error

**Examples:**
```cpp
/set strategy greedy         // Use greedy decoding
/set temperature 0.7         // More focused responses
/set top_p 0.95             // Wider nucleus sampling
/set length 200             // Longer responses
/set beam_width 10          // Wider beam search
```

#### generate_response()

```cpp
std::string generate_response(const std::string& user_input)
```

**Purpose:** Generate chatbot response to user input

**Process:**

1. **Add User Message to Context**
   ```cpp
   context->add_user_message(user_input);
   ```

2. **Format Context for Model**
   ```cpp
   std::string formatted_context = context->format_with_special_tokens();
   ```
   - Includes conversation history
   - Adds special tokens for model

3. **Generate Response**
   ```cpp
   response = model->generate_response_with_strategy(
       formatted_context,
       max_response_length,
       generation_strategy,
       temperature,
       top_p,
       top_k,
       beam_width
   );
   ```

4. **Add Response to Context**
   ```cpp
   context->add_assistant_message(response);
   ```

5. **Return Response**

**Error Handling:**
```cpp
try {
    // Generate response
} catch (const std::exception& e) {
    response = "[Error generating response: " + std::string(e.what()) + "]";
}
```

**Returns:** Generated response string

#### run()

```cpp
void run()
```

**Purpose:** Main chatbot loop

**Process:**

1. **Initialize Components**
   ```cpp
   if (!initialize()) {
       // Error and exit
   }
   ```

2. **Display Welcome**
   ```cpp
   print_welcome();
   ```

3. **Main Loop**
   ```cpp
   while (running) {
       // Display prompt
       std::cout << "You: ";
       
       // Get user input
       std::getline(std::cin, user_input);
       
       // Trim whitespace
       // Skip empty input
       
       // Check for exit commands
       if (user_input == "/exit" || user_input == "/quit") {
           running = false;
           continue;
       }
       
       // Handle commands
       if (user_input[0] == '/') {
           handle_command(user_input);
           continue;
       }
       
       // Generate and display response
       std::string response = generate_response(user_input);
       std::cout << "Bot: " << response << std::endl;
   }
   ```

4. **Save on Exit**
   ```cpp
   context->save_to_file(conversation_save_path);
   ```

**Input Processing:**
- Trims leading/trailing whitespace
- Skips empty lines
- Commands start with `/`
- Normal messages generate responses

**Exit Conditions:**
- `/exit` command
- `/quit` command
- Ctrl+D (EOF)

## Main Function

```cpp
int main(int argc, char* argv[])
```

**Purpose:** Entry point, parse arguments, run chatbot

**Command-Line Arguments:**

```bash
./chatbot [vocab_file] [model_file] [conversation_save_file]
```

| Argument | Position | Default | Description |
|----------|----------|---------|-------------|
| `vocab_file` | 1 | `vocab.txt` | Vocabulary file path |
| `model_file` | 2 | `chatbot_model.bin` | Model weights path |
| `conversation_save_file` | 3 | `conversation_history.txt` | Conversation save path |

**Help Option:**
```bash
./chatbot --help
./chatbot -h
```

**Output:**
```
Usage: ./chatbot [vocab_file] [model_file] [conversation_save_file]

Default values:
  vocab_file: vocab.txt
  model_file: chatbot_model.bin
  conversation_save_file: conversation_history.txt
```

**Execution:**
```cpp
ChatbotCLI chatbot(vocab_path, model_path, conv_save_path);
chatbot.run();
```

## Generation Strategies

### 1. Greedy Decoding

**Strategy:** `greedy`

**Description:** Always select the token with highest probability

**Characteristics:**
- Deterministic (same input → same output)
- Fast (no sampling overhead)
- May produce repetitive text
- Good for factual responses

**Use Cases:**
- Factual question answering
- Predictable responses needed
- Testing/debugging

**Parameters Used:** None

### 2. Beam Search

**Strategy:** `beam`

**Description:** Maintain top-N candidate sequences

**Characteristics:**
- Explores multiple hypotheses
- Better quality than greedy
- Slower than greedy
- Configurable width

**Use Cases:**
- High-quality text generation
- Translation tasks
- Balanced quality/diversity

**Parameters Used:**
- `beam_width` - Number of beams to maintain

**Example:**
```
/set strategy beam
/set beam_width 5
```

### 3. Temperature Sampling

**Strategy:** `sampling`

**Description:** Sample from probability distribution with temperature scaling

**Characteristics:**
- Random (different each time)
- Temperature controls randomness
- More diverse than greedy
- Can be incoherent if temp too high

**Use Cases:**
- Creative text generation
- Diverse responses
- Conversational variety

**Parameters Used:**
- `temperature` - Controls randomness (0.1-2.0)
  - Low (0.1-0.5): Focused, conservative
  - Medium (0.7-1.0): Balanced
  - High (1.2-2.0): Creative, random

**Example:**
```
/set strategy sampling
/set temperature 0.8
```

### 4. Top-K Sampling

**Strategy:** `top-k`

**Description:** Sample from top-k most probable tokens

**Characteristics:**
- Limits sampling to likely tokens
- Prevents unlikely words
- Configurable k value
- Good balance of quality/diversity

**Use Cases:**
- Controlled creativity
- Avoiding nonsense
- Moderate diversity

**Parameters Used:**
- `top_k` - Number of top tokens to consider
- `temperature` - Additional temperature scaling

**Example:**
```
/set strategy top-k
/set top_k 40
/set temperature 0.9
```

### 5. Nucleus Sampling (Top-P)

**Strategy:** `nucleus` (recommended)

**Description:** Sample from smallest set of tokens whose cumulative probability exceeds p

**Characteristics:**
- Dynamic vocabulary size
- Adapts to probability distribution
- State-of-the-art for text generation
- Recommended default

**Use Cases:**
- General conversation (default)
- High-quality diverse responses
- Production chatbots

**Parameters Used:**
- `top_p` - Cumulative probability threshold (0.0-1.0)
  - 0.9: Recommended default
  - 0.95: More diverse
  - 0.8: More focused
- `temperature` - Additional temperature scaling

**Example:**
```
/set strategy nucleus
/set top_p 0.9
/set temperature 1.0
```

## Model Architecture

**Fixed Configuration:**
```cpp
d_model = 512              // Model dimension
num_heads = 8              // Attention heads
d_ff = 2048                // Feed-forward dimension
num_encoder_layers = 6     // Encoder depth
num_decoder_layers = 6     // Decoder depth
max_seq_length = 1024      // Maximum sequence length
```

**Note:** Architecture is hardcoded in `initialize()`. To use different architectures, model file must match these dimensions.

## Conversation Context

**Configuration:**
```cpp
max_messages = 20          // Maximum messages in history
max_tokens = 2048          // Maximum tokens in context
```

**Features:**
- Automatic history truncation
- System message support
- Special token formatting
- Token counting
- Save/load functionality

**Context Format:**
```
[System message if set]
User: <message 1>
Assistant: <response 1>
User: <message 2>
Assistant: <response 2>
...
```

## File Operations

### Vocabulary File

**Location:** Specified by first argument or default `vocab.txt`

**Format:** BPE vocabulary (one token per line)
```
<unk>
<pad>
<s>
</s>
token1
token2
...
```

**Loading:**
```cpp
tokenizer->load_vocab(vocab_path);
```

### Model File

**Location:** Specified by second argument or default `chatbot_model.bin`

**Format:** Binary model weights (EncoderDecoderModel format)

**Loading:**
```cpp
model->load_model(model_path);
```

**Behavior:**
- If file exists: Load weights
- If file missing: Use random initialization (with warning)
- If load fails: Catch exception, use random initialization

### Conversation History File

**Location:** Specified by third argument or default `conversation_history.txt`

**Format:** Plain text with message markers

**Operations:**
- **Save:** `/save` command or automatic on exit
- **Load:** `/load` command

**Example Format:**
```
USER: Hello!
ASSISTANT: Hi there! How can I help you?
USER: What's the weather?
ASSISTANT: I don't have access to real-time weather data.
```

## Usage Examples

### Basic Usage

```bash
# Use default files
./chatbot

# Specify custom files
./chatbot my_vocab.txt my_model.bin my_history.txt

# Show help
./chatbot --help
```

### Interactive Session

```
You: Hello!
Bot: Hi there! How can I help you today?

You: /stats
📊 Conversation Statistics:
  Total messages: 2
  Estimated tokens: 25

You: /set strategy greedy
✅ Generation strategy set to: greedy

You: /set temperature 0.7
✅ Temperature set to: 0.7

You: Tell me a joke
Bot: Why did the programmer quit his job? Because he didn't get arrays!

You: /save
✅ Conversation saved to: conversation_history.txt

You: /exit
💾 Saving conversation...
✅ Conversation saved to: conversation_history.txt
👋 Goodbye!
```

### Advanced Configuration

```
You: /settings
⚙️  Current Settings:
  Strategy: nucleus
  Max length: 100
  Temperature: 1.0
  Top-p (nucleus): 0.9
  Top-k: 50
  Beam width: 5

You: /set strategy beam
✅ Generation strategy set to: beam

You: /set beam_width 10
✅ Beam width set to: 10

You: /set length 150
✅ Max response length set to: 150

You: /system You are a helpful assistant specialized in Python programming.
✅ System message set
```

## Command Reference

### Information Commands

| Command | Description | Output |
|---------|-------------|--------|
| `/help` | Display help | Welcome message and command list |
| `/stats` | Show statistics | Message count, token count |
| `/settings` | Show settings | Current generation parameters |

### Conversation Commands

| Command | Description | Effect |
|---------|-------------|--------|
| `/clear` | Clear history | Removes all messages from context |
| `/save` | Save conversation | Writes to conversation file |
| `/load` | Load conversation | Reads from conversation file |
| `/system <msg>` | Set system message | Adds context/instructions for bot |

### Configuration Commands

| Command | Example | Effect |
|---------|---------|--------|
| `/set strategy <name>` | `/set strategy nucleus` | Change generation strategy |
| `/set length <n>` | `/set length 150` | Set max response length |
| `/set temperature <f>` | `/set temperature 0.8` | Set sampling temperature |
| `/set top_p <f>` | `/set top_p 0.95` | Set nucleus threshold |
| `/set top_k <n>` | `/set top_k 40` | Set top-k limit |
| `/set beam_width <n>` | `/set beam_width 10` | Set beam width |

### Exit Commands

| Command | Effect |
|---------|--------|
| `/exit` | Save conversation and exit |
| `/quit` | Save conversation and exit |
| Ctrl+D | Exit (may not save) |

## Best Practices

### 1. Model Selection

**Use Pre-trained Model:**
```bash
# Train model first
./ChatbotTrainer --data conversations.txt --build-vocab 5000 --epochs 20 --output my_model.bin

# Then use in CLI
./chatbot vocab.txt my_model.bin
```

**Random Initialization:**
- Only for testing/debugging
- Responses will be nonsensical
- Shows "⚠️ Using random initialization" warning

### 2. Generation Strategy Selection

**Recommended Settings by Use Case:**

| Use Case | Strategy | Temperature | Top-P | Top-K |
|----------|----------|-------------|-------|-------|
| General chat | `nucleus` | 0.9-1.0 | 0.9 | - |
| Factual Q&A | `greedy` | - | - | - |
| Creative writing | `sampling` | 1.2-1.5 | - | - |
| Balanced quality | `beam` | - | - | beam_width=5 |
| Controlled diversity | `top-k` | 0.9 | - | 40 |

### 3. Context Management

**Clear Long Conversations:**
```
/clear    # Reset when conversation gets too long or off-topic
```

**Use System Messages:**
```
/system You are a helpful coding assistant specializing in Python.
```

**Save Important Conversations:**
```
/save     # Before /clear or /exit
```

### 4. Performance Tuning

**Faster Responses:**
- Use `greedy` strategy
- Reduce `max_length`
- Reduce `beam_width`

**Better Quality:**
- Use `nucleus` or `beam`
- Increase `beam_width` (5-10)
- Adjust `temperature` (0.7-0.9)

**More Diversity:**
- Increase `temperature` (1.0-1.2)
- Increase `top_p` (0.95)
- Use `sampling` strategy

### 5. Troubleshooting

**Repetitive Responses:**
```
/set strategy nucleus
/set temperature 1.0
/set top_p 0.9
```

**Incoherent Responses:**
```
/set temperature 0.7
/set strategy beam
```

**Slow Generation:**
```
/set strategy greedy
/set length 50
```

**Response Too Short:**
```
/set length 200
```

## Error Handling

### Initialization Errors

**Vocabulary Not Found:**
```
Failed to load tokenizer: Cannot open vocab.txt
```
**Solution:** Ensure vocabulary file exists and path is correct

**Model Load Failure:**
```
⚠️ Failed to load model weights. Using random initialization.
```
**Solution:** Check model file path and format compatibility

### Runtime Errors

**Generation Error:**
```
Bot: [Error generating response: <error message>]
```
**Causes:**
- Model dimension mismatch
- Out of memory
- Invalid token IDs

**Save/Load Error:**
```
❌ Failed to save conversation
❌ Failed to load conversation
```
**Causes:**
- File permissions
- Disk space
- Invalid file format

### Command Errors

**Unknown Command:**
```
❓ Unknown command. Type /help for available commands.
```

**Invalid Parameter:**
```
❌ Unknown parameter: <param>
Available: strategy, length, temperature, top_p, top_k, beam_width
```

**Invalid Strategy:**
```
❌ Invalid strategy. Use: greedy, beam, sampling, top-k, or nucleus
```

## Integration with Other Components

### EncoderDecoderModel

**Methods Used:**
```cpp
// Constructor
EncoderDecoderModel(d_model, num_heads, d_ff, enc_layers, dec_layers, vocab_size, max_len);

// Model loading
void load_model(filepath);

// Generation (primary method)
std::string generate_response_with_strategy(
    context, max_length, strategy, temperature, top_p, top_k, beam_width
);
```

### ConversationContext

**Methods Used:**
```cpp
// Constructor
ConversationContext(max_messages, max_tokens);

// Message management
void add_user_message(text);
void add_assistant_message(text);
void set_system_message(text);
void clear();

// Formatting
std::string format_with_special_tokens();

// Statistics
int get_message_count();
int get_total_tokens();

// Persistence
void save_to_file(filepath);
void load_from_file(filepath);
```

### BPETokenizer

**Methods Used:**
```cpp
// Vocabulary loading
void load_vocab(filepath);

// Vocabulary info
int get_vocab_size();
```

## Limitations & Future Enhancements

### Current Limitations

1. **Fixed Model Architecture**
   - Hardcoded dimensions in `initialize()`
   - Cannot load models with different architectures
   - **Future:** Read architecture from model file

2. **No Multi-turn Context Validation**
   - Assumes context fits in max_seq_length
   - May truncate without warning
   - **Future:** Smart truncation with warnings

3. **No Response Streaming**
   - Full response generated before display
   - Long responses have noticeable delay
   - **Future:** Token-by-token streaming output

4. **Limited Error Recovery**
   - Some errors cause full exit
   - No retry mechanisms
   - **Future:** Graceful degradation

5. **No Configuration File**
   - All settings via commands
   - Settings not persisted
   - **Future:** Config file support

6. **Single Conversation**
   - Only one active conversation
   - No conversation switching
   - **Future:** Multiple conversation sessions

### Planned Enhancements

1. **Enhanced Features**
   - Response streaming
   - Typing indicators
   - Token usage tracking
   - Cost estimation
   - Response rating/feedback

2. **Better Context Management**
   - Automatic summarization
   - Smart truncation
   - Context compression
   - Multi-document context

3. **Advanced Configuration**
   - Configuration file (.chatbotrc)
   - Profiles (casual, professional, creative)
   - Persistent settings
   - Per-conversation settings

4. **Improved UX**
   - Autocomplete for commands
   - Command history (up/down arrows)
   - Multi-line input
   - Rich text formatting
   - Progress indicators

5. **Additional Commands**
   - `/retry` - Regenerate last response
   - `/edit` - Edit last message
   - `/undo` - Remove last exchange
   - `/export` - Export in various formats
   - `/conversations` - List saved conversations

## Related Documentation

- **EncoderDecoderModel:** `Context Documentation/ENCODERDECODERMODEL_CONTEXT.md`
- **ConversationContext:** `Context Documentation/CONVERSATIONCONTEXT_CONTEXT.md`
- **BPETokenizer:** `Context Documentation/BPE_TOKENIZER_CONTEXT.md`
- **TextGenerator:** `Context Documentation/TEXTGENERATOR_CONTEXT.md`
- **ChatbotTrainer:** `Context Documentation/CHATBOTTRAINER_CONTEXT.md`
- **Quickstart Guide:** `CHATBOT_CLI_QUICKSTART.md`
- **README:** `CHATBOT_CLI_README.md`

## Summary

`ChatbotCLI` provides a feature-rich command-line interface for the ADAI transformer chatbot with:

✅ **Interactive Chat:** Real-time conversation with transformer model  
✅ **Multiple Strategies:** 5 generation strategies (greedy, beam, sampling, top-k, nucleus)  
✅ **Configurable Parameters:** Temperature, top-p, top-k, beam width, max length  
✅ **Conversation Management:** History tracking, save/load, statistics  
✅ **Rich Commands:** 15+ commands for control and configuration  
✅ **Colored Output:** ANSI color codes for better UX  
✅ **Persistent History:** Automatic save on exit  
✅ **Error Handling:** Graceful failures with informative messages  
✅ **Flexible Configuration:** Command-line arguments and runtime settings  

**Ideal For:**
- Testing trained chatbot models
- Interactive conversations
- Experimenting with generation strategies
- Debugging model behavior
- Production chatbot deployment (terminal environments)

**Key Strength:** Combines ease of use (simple CLI) with power (5 generation strategies, extensive configuration) for transformer-based chatbots.
