# ADAI Chatbot GUI Guide

## Overview

The ADAI Chatbot GUI is a graphical client for `chatbot_api_server`, built with Qt (C++). Like the `chatbot` CLI,
it sends your messages to the model the server serves, and the server keeps the conversation. It provides a chat
window, generation settings and conversation management.

> **Since November 4, 2026 (TD-260)** the GUI no longer loads a vocabulary or model of its own (it used to take
> `[vocab_file] [model_file]` and run a separate copy of the model on the CPU, which never actually loaded a
> trained checkpoint). It now takes the server's URL, so it always talks to the same trained model as every other
> client, with the server's architecture (MNS), GPU and prompt format.

## Quick Install & Run

### Install on Ubuntu/Debian

```bash
# Install Qt5
sudo apt-get install qt5-default qtbase5-dev

# Build
cd build
cmake .. -DBUILD_GUI=ON
make chatbot_gui -j$(nproc)

# Run (with chatbot_api_server running on localhost:8080)
./src/chatbot_gui
```

### Install on macOS

```bash
# Install Qt
brew install qt@5

# Build
cd build
cmake .. -DBUILD_GUI=ON -DCMAKE_PREFIX_PATH=$(brew --prefix qt@5)
make chatbot_gui -j$(nproc)

# Run (with chatbot_api_server running on localhost:8080)
./src/chatbot_gui
```

## Features

### Chat Interface

- **Clean message display** with color-coded user and bot messages
- **Timestamp** for each message
- **Auto-scrolling** to latest messages
- **Responsive while the server generates**: each request runs on a worker thread (TD-263)

### Generation Settings Panel

Each setting starts at **server default**: it isn't sent, so `chatbot_api_server`'s configuration applies.
Change one and it is sent with every message after that (the server checks it against the same ranges);
set it back to server default to stop sending it.

- **Generation Strategy**: Server default, Nucleus (Top-p), Top-k, Greedy, Beam Search, Sampling (Temperature)
- **Temperature**: Control randomness (0 - 2; 0 is greedy)
- **Top-p (Nucleus)**: Probability mass cutoff (0 - 1)
- **Top-k**: Number of top tokens to consider (1 - 1000)
- **Max Response Length**: New tokens in the response (1 - 4096)
- **Beam Width**: For beam search strategy (1 - 16)

### Conversation Management

- **Clear Chat**: Clear the conversation on the server
- **Save Conversation**: Export the server's copy of the conversation to a file (the `chatbot` CLI's format)
- **Load Conversation**: Send a saved conversation to the server, which continues it

## Screenshots

```text
╔═══════════════════════════════════════════════════════════════════╗
║  Chat Area                      │  Settings Panel                 ║
║ ─────────────────────────────   │ ───────────────────────────     ║
║  You: Hello!                    │  Generation Strategy:           ║
║  Bot: Hi! How can I help?       │  [Nucleus (Top-p)    ▼]         ║
║                                 │                                 ║
║  You: Tell me about AI          │  Temperature: [1.0  ]           ║
║  Bot: AI is...                  │  Top-p:       [0.90 ]           ║
║                                 │  Top-k:       [50   ]           ║
║ ─────────────────────────────   │  Max Length:  [100  ]           ║
║  [Type message...        ]      │  Beam Width:  [5    ]           ║
║  [Send]                         │                                 ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Generation Strategies

|Strategy|Best For|Speed|
|---------------|----------------------------|-------|
|**Nucleus**|Balanced, creative responses|Fast|
|**Top-k**|Controlled creativity|Fast|
|**Greedy**|Deterministic, predictable|Fastest|
|**Beam Search**|High quality, coherent|Slower|
|**Sampling**|Random, diverse|Fast|

## Recommended Settings

### Creative Writing

- Strategy: Nucleus
- Temperature: 1.2
- Top-p: 0.9

### Technical Q&A

- Strategy: Greedy
- Temperature: 0.7
- Top-p: 0.95

### General Chat

- Strategy: Nucleus
- Temperature: 1.0
- Top-p: 0.9

## Prerequisites

System Requirements:

- **Qt5/Qt6**: Qt5 recommended for stability
- **Disk Space**: At least 2GB free space
- **Memory**: Minimum 2GB RAM, recommended 4GB+
- **C++17 Compiler**: GCC 7+, Clang 5+, or MSVC 2019+

### Required Dependencies

#### Ubuntu/Debian

```bash
# Qt5 (recommended)
sudo apt-get update
sudo apt-get install qt5-default qtbase5-dev

# Or Qt6
sudo apt-get install qt6-base-dev
```

#### Fedora/RHEL

```bash
# Qt5
sudo dnf install qt5-qtbase-devel

# Or Qt6
sudo dnf install qt6-qtbase-devel
```

#### macOS

```bash
# Using Homebrew
brew install qt@5

# Or Qt6
brew install qt@6
```

#### Windows

Download and install Qt from [Qt Downloads](https://www.qt.io/download).

## Building the GUI

### Standard Build

```bash
cd /path/to/adai
mkdir -p build && cd build

# Configure with CMake
cmake .. -DBUILD_GUI=ON

# Build the GUI application
make chatbot_gui -j$(nproc)
```

### Build Options

```bash
# Disable GUI build
cmake .. -DBUILD_GUI=OFF

# Specify Qt version explicitly
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt

# Build with optimizations
cmake .. -DBUILD_GUI=ON -DCMAKE_BUILD_TYPE=Release
make chatbot_gui -j$(nproc)
```

## Running the GUI

### Command-Line Usage

Start `chatbot_api_server` first (see the [CLI guide](chatbot-guide.md)), then:

```bash
# From the build directory; connects to http://localhost:8080
./src/chatbot_gui

# Another server
./src/chatbot_gui http://ai-machine:8080
```

### Command-Line Options

```bash
# Show help (works without a display)
./src/chatbot_gui --help

# Example output:
# Usage: chatbot_gui [server_url]
#
# A desktop client for chatbot_api_server, like the chatbot CLI: messages go to the model
# the server serves, and the server keeps the conversation.
#
# Default values:
#   server_url: http://localhost:8080
```

The old `chatbot_gui vocab.txt model.bin` form exits with an error saying the GUI no longer loads them.

When the window opens, the chat pane says whether the server was reached. If the server is running an
untrained model (`ALLOW_UNTRAINED_MODEL=true`, which makes `/health` answer 503 `"untrained"`), it warns that
the replies are random. If the server can't be reached, it says so; messages are sent once the server is up.

`chatbot_gui` is a small launcher that runs `chatbot_gui_binary` beside it. It removes what a snap (such as a snap
VS Code's terminal) leaks into the environment: the `SNAP*` and `GTK_PATH` variables, a snap `GIO_MODULE_DIR`, and
the snap entries of `LD_LIBRARY_PATH` and `QT_PLUGIN_PATH`, keeping your own entries in order. Under snap it also
points Qt at the platform plugins of the Qt the GUI was built with (Qt5 or Qt6). It sets no architecture- or
distribution-specific paths (TD-264).

## Using the GUI

### Basic Usage

1. **Type your message** in the input field
2. **Press Enter or click Send**
3. **Wait for response** (Send shows "Generating..."; input and settings are disabled until the reply arrives,
   and the window stays responsive)
4. **Adjust settings** in real-time on the right panel

### Starting a Conversation

1. **Start `chatbot_api_server`, then launch the application**

   ```bash
   ./src/chatbot_gui http://localhost:8080
   ```

2. **Type your message** in the input field at the bottom

3. **Press "Send"** or hit **Enter** to send your message

4. **Wait for the response** - the bot will generate and display a reply

### Adjusting Settings

1. **Open the Settings Panel** on the right side of the window

2. **Select a generation strategy** (or leave it at **Server default**):
   - **Nucleus (Top-p)**: Best for creative, diverse responses
   - **Top-k Sampling**: Good balance of quality and diversity
   - **Greedy**: Most predictable, deterministic output
   - **Beam Search**: High quality but slower
   - **Sampling**: More random and creative

3. **Adjust parameters**:
   - **Temperature**: Higher = more creative/random (try 0.8-1.2)
   - **Top-p**: 0.9 is a good default for most use cases
   - **Top-k**: 50 is standard, lower for more focused responses
   - **Max Length**: Limit response size (100 tokens ≈ 75 words)

### Managing Conversations

#### Save Conversation

1. Click **"Save Conversation"** button
2. Choose a location and filename
3. The server's copy of the conversation is written there (replacing the file only once it's fully written)

#### Load Conversation

1. Click **"Load Conversation"** button
2. Select a previously saved conversation file (from the GUI or the `chatbot` CLI)
3. The server continues that conversation. Its earlier messages aren't redisplayed yet (TD-266)

#### Clear Chat

1. Click **"Clear Chat"** button
2. Confirm the action
3. The server clears the conversation and the chat display is reset

## Keyboard Shortcuts

- **Enter**: Send message
- **Ctrl+S**: Save conversation (when implemented)
- **Ctrl+L**: Load conversation (when implemented)
- **Ctrl+Q**: Quit application

## Troubleshooting

### Qt Not Found

**Problem**: CMake cannot find Qt

**Solution**:

```bash
# Find Qt installation
find /usr -name "Qt5Config.cmake" 2>/dev/null
find /usr -name "Qt6Config.cmake" 2>/dev/null

# Set Qt path explicitly
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt -DBUILD_GUI=ON
```

### Server Not Reachable

**Problem**: "Could not reach chatbot_api_server at …"

**Solution**:

1. Start `chatbot_api_server` (see the [CLI guide](chatbot-guide.md))
2. Or pass its URL: `./src/chatbot_gui http://<host>:8080`

### Gibberish Replies or "untrained model" Warning

The server's model is untrained or undertrained. Train it with `incremental_trainer` and restart the server
on the new checkpoint; the GUI needs no change.

### Slow Response Generation

Generation runs in the server, so its speed is the server's (GPU, threads, model size). The GUI stays
responsive while waiting. Reduce **Max Response Length** or use **Greedy** for faster replies.

### Black/Empty Chat Display

**Problem**: Messages don't appear in chat window

**Solution**:

1. Check Qt version compatibility
2. Check the chat pane's connection message
3. Check terminal output for error messages

## Configuration

The GUI reads no configuration file. The model, vocabulary, architecture and default generation settings are
`chatbot_api_server`'s (`config.chatbot.conf`, or MNS). Saved conversations go where you choose.

## Architecture

### Components

1. **ChatbotGUI.hpp/cpp**: Main GUI implementation
   - Qt widgets and layout management
   - Signal/slot connections
   - Message display formatting
   - Settings panel controls

2. **ChatbotGUI_main.cpp**: Application entry point
   - Qt application initialization
   - Command-line argument parsing (the server URL)
   - Window creation and display

3. **ChatbotClient.hpp/cpp** (TD-260): the HTTP session with `chatbot_api_server`, shared with the `chatbot`
   CLI
   - `POST /chat/session` with the changed settings, `/clear-session`, `/chat/session/export` and `/import`
   - Starts a new session when the server has forgotten one (it answers 404)

4. **ChatbotGuiLogic.hpp**: the Qt-free parts (strategy names, which settings are sent, connection messages),
   unit-tested

### Qt Features Used

- **QMainWindow**: Main application window
- **QTextEdit**: Rich text display for messages
- **QLineEdit**: User input field
- **QComboBox**: Strategy selection dropdown
- **QSpinBox/QDoubleSpinBox**: Numeric parameter controls
- **QGroupBox**: Settings organization
- **QSplitter**: Resizable layout sections

### Architecture Diagram

```text
ChatbotGUI (Qt QMainWindow)
    ├── Chat Display (QTextEdit)
    │   └── Formatted HTML messages
    ├── Input Area (QLineEdit + QPushButton)
    │   └── User message entry
    ├── Settings Panel (QGroupBox widgets)
    │   ├── Strategy (QComboBox)
    │   ├── Temperature (QDoubleSpinBox)
    │   ├── Top-p (QDoubleSpinBox)
    │   ├── Top-k (QSpinBox)
    │   ├── Max Length (QSpinBox)
    │   └── Beam Width (QSpinBox)
    └── ChatbotClient (worker thread)
        └── HTTP → chatbot_api_server (model, tokenizer, conversation)
```

### Files Created

- `src/ChatbotGUI.hpp` - GUI header with Qt class definition
- `src/ChatbotGUI.cpp` - GUI implementation with Qt widgets
- `src/ChatbotGUI_main.cpp` - Application entry point
- `src/ChatbotClient.hpp/cpp` - HTTP client of chatbot_api_server, shared with the CLI
- `src/ChatbotGuiLogic.hpp` - Qt-free logic
- `src/CMakeLists.txt` - Build configuration (Qt target)

## Performance Tips

Generation runs in `chatbot_api_server`; these settings change how much work it does per reply.

1. **Use Greedy or Nucleus** for fastest generation
2. **Limit max_response_length** to 50-100 tokens for real-time feel
3. **Lower temperature** (0.7-0.9) for faster, more predictable responses
4. **Avoid Beam Search** with high beam width (use 3-5 max)

## Future Enhancements

Planned features for future versions:

- [x] Asynchronous generation (non-blocking UI) (TD-263, November 4, 2026)
- [ ] Redisplay a loaded conversation's messages (TD-266)
- [ ] Stop a reply in progress (the server has no way to cancel a generation yet)
- [ ] Conversation export to multiple formats (JSON, Markdown)
- [ ] Syntax highlighting for code in messages
- [ ] Custom themes and color schemes
- [ ] Message editing and regeneration
- [ ] Token count display
- [ ] Response streaming (token-by-token)
- [ ] System message/prompt configuration in GUI

## Comparison with CLI

|Feature|GUI|CLI|
|---------|-----|-----|
|**Ease of Use**|⭐⭐⭐⭐⭐|⭐⭐⭐|
|**Visual Appeal**|⭐⭐⭐⭐⭐|⭐⭐|
|**Configuration**|Real-time sliders|Command syntax|
|**Message History**|Scrollable display|Terminal scrollback|
|**Conversation Save**|GUI dialog|Auto-save|
|**Model**|chatbot_api_server|chatbot_api_server|
|**Performance**|Same (the server's)|Same (the server's)|
|**Portability**|Requires Qt|Terminal only|

## Examples

### Creative Writing Assistant

```text
Settings:
- Strategy: Nucleus
- Temperature: 1.2
- Top-p: 0.9
- Max Length: 200

Message: "Write a short poem about artificial intelligence"
```

### Technical Q&A Example

```text
Settings:
- Strategy: Greedy
- Temperature: 0.7
- Top-p: 0.95
- Max Length: 150

Message: "Explain how transformers work in machine learning"
```

### Casual Conversation

```text
Settings:
- Strategy: Nucleus
- Temperature: 1.0
- Top-p: 0.9
- Max Length: 100

Message: "What's your favorite hobby?"
```

## Requirements

- **Qt5** or **Qt6** (Qt5 recommended)
- **A running `chatbot_api_server`** serving a trained model
- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2019+)

## Next Steps

1. **Train a model** with `incremental_trainer` if you haven't (see the training guide)

2. **Start the server** on it: `chatbot_api_server` (see the [CLI guide](chatbot-guide.md))

3. **Run the GUI**:

   ```bash
   ./src/chatbot_gui http://localhost:8080
   ```

4. **Experiment with settings** to find what works best for your use case

5. **Save interesting conversations** for later review

## See Also

- [CLI Guide](chatbot-guide.md) - Command-line interface
- [Training Guide](../reference/chatbot-completeness.md) - Model training

## Support

For issues, questions, or contributions:

- Check the main ADAI documentation
- Review ChatbotCLI guide for similar functionality
- Examine the source code in `src/ChatbotGUI.cpp`

The GUI and the CLI are both clients of `chatbot_api_server`, through the same `ChatbotClient`, so:

- Both talk to the same model, and generation quality is identical
- Saved conversations are interchangeable
- Settings behave the same way (only changed settings are sent)

For detailed information about generation strategies and parameters, see the main chatbot documentation.

## License

Same license as the main ADAI project.
