# ADAI Chatbot GUI Guide

## Overview

The ADAI Chatbot GUI is a modern, user-friendly graphical interface for interacting with the transformer-based chatbot. Built with Qt (C++), it provides an intuitive chat experience with real-time configuration and conversation management.

## Quick Install & Run

### Install on Ubuntu/Debian

```bash
# Install Qt5
sudo apt-get install qt5-default qtbase5-dev

# Build
cd build
cmake .. -DBUILD_GUI=ON
make chatbot_gui -j$(nproc)

# Run
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

# Run
./src/chatbot_gui
```

## Features

### Chat Interface

- **Clean message display** with color-coded user and bot messages
- **Timestamp** for each message
- **Auto-scrolling** to latest messages
- **Real-time response generation**

### Generation Settings Panel

Configure all generation parameters in real-time:

- **Generation Strategy**: Nucleus (Top-p), Top-k, Greedy, Beam Search, Sampling
- **Temperature**: Control randomness (0.1 - 2.0)
- **Top-p (Nucleus)**: Probability mass cutoff (0.1 - 1.0)
- **Top-k**: Number of top tokens to consider (1 - 200)
- **Max Response Length**: Maximum tokens in response (10 - 500)
- **Beam Width**: For beam search strategy (1 - 10)

### Conversation Management

- **Clear Chat**: Reset conversation history
- **Save Conversation**: Export chat history to file
- **Load Conversation**: Import previous conversations

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

| Strategy        | Best For                     | Speed   |
| --------------- | ---------------------------- | ------- |
| **Nucleus**     | Balanced, creative responses | Fast    |
| **Top-k**       | Controlled creativity        | Fast    |
| **Greedy**      | Deterministic, predictable   | Fastest |
| **Beam Search** | High quality, coherent       | Slower  |
| **Sampling**    | Random, diverse              | Fast    |

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

**System Requirements:**

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

```bash
# From the build directory
./src/chatbot_gui

# With custom vocabulary
./src/chatbot_gui vocab.txt

# With custom vocabulary and model
./src/chatbot_gui vocab.txt chatbot_model.bin
```

### Command-Line Options

```bash
# Show help
./src/chatbot_gui --help

# Example output:
# Usage: chatbot_gui [vocab_file] [model_file]
#
# Default values:
#   vocab_file: vocab.txt
#   model_file: chatbot_model.bin
```

## Using the GUI

### Basic Usage

1. **Type your message** in the input field
2. **Press Enter or click Send**
3. **Wait for response** (GUI shows "Generating...")
4. **Adjust settings** in real-time on the right panel

### Starting a Conversation

1. **Launch the application**

   ```bash
   ./src/chatbot_gui
   ```

2. **Type your message** in the input field at the bottom

3. **Press "Send"** or hit **Enter** to send your message

4. **Wait for the response** - the bot will generate and display a reply

### Adjusting Settings

1. **Open the Settings Panel** on the right side of the window

2. **Select a generation strategy**:
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
3. Conversation is saved in plain text format

#### Load Conversation

1. Click **"Load Conversation"** button
2. Select a previously saved conversation file
3. Context is restored for the chatbot

#### Clear Chat

1. Click **"Clear Chat"** button
2. Confirm the action
3. Chat display and conversation context are reset

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

### Model Loading Fails

**Problem**: "Failed to initialize chatbot components"

**Solution**:

1. Ensure `vocab.txt` exists in the working directory
2. Train a model first using `chatbot_trainer`
3. Or specify correct paths:

   ```bash
   ./src/chatbot_gui /path/to/vocab.txt /path/to/model.bin
   ```

### Slow Response Generation

**Problem**: GUI freezes during response generation

**Explanation**: Response generation happens in the main thread. For large models or long responses, this may take time.

**Solutions**:

- Reduce **Max Response Length**
- Use **Greedy** strategy instead of Beam Search
- Train a smaller model for faster inference
- Future enhancement: Add threading for async generation

### Black/Empty Chat Display

**Problem**: Messages don't appear in chat window

**Solution**:

1. Check Qt version compatibility
2. Verify model initialization succeeded
3. Check terminal output for error messages

## Configuration Files

### Default Locations

- **Vocabulary**: `vocab.txt` (in working directory)
- **Model**: `chatbot_model.bin` (in working directory)
- **Saved Conversations**: User-specified location

### Custom Paths

Always use absolute paths or run from the correct directory:

```bash
# Run from build directory
cd build
./src/chatbot_gui

# Or use absolute paths
./src/chatbot_gui ~/adai/vocab.txt ~/adai/models/chatbot_model.bin
```

## Architecture

### Components

1. **ChatbotGUI.hpp/cpp**: Main GUI implementation
   - Qt widgets and layout management
   - Signal/slot connections
   - Message display formatting
   - Settings panel controls

2. **ChatbotGUI_main.cpp**: Application entry point
   - Qt application initialization
   - Command-line argument parsing
   - Window creation and display

3. **Integration**:
   - Uses existing `BPETokenizer` for text processing
   - Uses `EncoderDecoderModel` for generation
   - Uses `ConversationContext` for multi-turn dialogue

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
    │   └── Max Length (QSpinBox)
    └── Chatbot Backend
        ├── BPETokenizer
        ├── EncoderDecoderModel
        └── ConversationContext
```

### Files Created

- `src/ChatbotGUI.hpp` - GUI header with Qt class definition
- `src/ChatbotGUI.cpp` - GUI implementation with Qt widgets
- `src/ChatbotGUI_main.cpp` - Application entry point
- `src/CMakeLists.txt` - Updated build configuration (Qt target)

## Performance Tips

1. **Use Greedy or Nucleus** for fastest generation
2. **Limit max_response_length** to 50-100 tokens for real-time feel
3. **Lower temperature** (0.7-0.9) for faster, more predictable responses
4. **Avoid Beam Search** with high beam width (use 3-5 max)

## Future Enhancements

Planned features for future versions:

- [ ] Asynchronous generation (non-blocking UI)
- [ ] Conversation export to multiple formats (JSON, Markdown)
- [ ] Syntax highlighting for code in messages
- [ ] Custom themes and color schemes
- [ ] Message editing and regeneration
- [ ] Token count display
- [ ] Response streaming (token-by-token)
- [ ] System message/prompt configuration in GUI

## Comparison with CLI

| Feature | GUI | CLI |
| --------- | ----- | ----- |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Visual Appeal** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Configuration** | Real-time sliders | Command syntax |
| **Message History** | Scrollable display | Terminal scrollback |
| **Conversation Save** | GUI dialog | Auto-save |
| **Performance** | Same | Same |
| **Portability** | Requires Qt | Terminal only |

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
- **Trained chatbot model** (`chatbot_model.bin`)
- **Vocabulary file** (`vocab.txt`)
- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2019+)

## Next Steps

1. **Train a model** if you haven't:

   ```bash
   ./src/chatbot_trainer --data training_data.txt --vocab vocab.txt
   ```

2. **Run the GUI**:

   ```bash
   ./src/chatbot_gui
   ```

3. **Experiment with settings** to find what works best for your use case

4. **Save interesting conversations** for later review

## See Also

- [CLI Guide](chatbot-guide.md) - Command-line interface
- [Training Guide](../reference/chatbot-completeness.md) - Model training
- [Dataset Batch Processing](BATCH_PROCESSING_QUICK_REFERENCE.md) - Data handling

## Support

For issues, questions, or contributions:

- Check the main ADAI documentation
- Review ChatbotCLI guide for similar functionality
- Examine the source code in `src/ChatbotGUI.cpp`

The GUI uses the same underlying components as the CLI, so:

- Generation quality is identical
- All features from CLI are available
- Settings behave the same way

For detailed information about generation strategies and parameters, see the main chatbot documentation.

## License

Same license as the main ADAI project.
