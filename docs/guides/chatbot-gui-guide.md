# Chatbot GUI Guide

## Overview

The ADAI Chatbot GUI is a modern, user-friendly graphical interface for interacting with the transformer-based chatbot. Built with Qt (C++), it provides an intuitive chat experience with real-time configuration and conversation management.

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

## Prerequisites

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

Download and install Qt from: https://www.qt.io/download

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

### Basic Usage

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

### Technical Q&A

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

## Support

For issues, questions, or contributions:

- Check the main ADAI documentation
- Review ChatbotCLI guide for similar functionality
- Examine the source code in `src/ChatbotGUI.cpp`

## License

Same license as the main ADAI project.
