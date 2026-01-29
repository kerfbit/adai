# ADAI Chatbot GUI - Quick Start

A modern Qt-based graphical interface for the ADAI transformer chatbot.

## Quick Install & Run

### Ubuntu/Debian
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

### macOS
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

✨ **Modern Chat Interface**
- Color-coded messages (blue for user, green for bot)
- Timestamps for all messages
- Auto-scrolling message display

⚙️ **Real-Time Settings**
- Generation strategy selection
- Temperature, top-p, top-k controls
- Max response length adjustment
- Beam width configuration

💾 **Conversation Management**
- Save conversations to file
- Load previous conversations
- Clear chat history

## Screenshots

```
╔═══════════════════════════════════════════════════════════════════╗
║  Chat Area                      │  Settings Panel                 ║
║ ─────────────────────────────   │ ───────────────────────────     ║
║  You: Hello!                    │  Generation Strategy:           ║
║  Bot: Hi! How can I help?       │  [Nucleus (Top-p)    ▼]        ║
║                                 │                                 ║
║  You: Tell me about AI          │  Temperature: [1.0  ]          ║
║  Bot: AI is...                  │  Top-p:      [0.90 ]          ║
║                                 │  Top-k:      [50   ]          ║
║ ─────────────────────────────   │  Max Length: [100  ]          ║
║  [Type message...        ]      │  Beam Width: [5    ]          ║
║  [Send]                         │                                 ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Basic Usage

1. **Type your message** in the input field
2. **Press Enter or click Send**
3. **Wait for response** (GUI shows "Generating...")
4. **Adjust settings** in real-time on the right panel

## Generation Strategies

| Strategy | Best For | Speed |
|----------|----------|-------|
| **Nucleus** | Balanced, creative responses | Fast ⚡⚡⚡ |
| **Top-k** | Controlled creativity | Fast ⚡⚡⚡ |
| **Greedy** | Deterministic, predictable | Fastest ⚡⚡⚡⚡ |
| **Beam Search** | High quality, coherent | Slower ⚡⚡ |
| **Sampling** | Random, diverse | Fast ⚡⚡⚡ |

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

## Requirements

- **Qt5** or **Qt6** (Qt5 recommended)
- **Trained chatbot model** (`chatbot_model.bin`)
- **Vocabulary file** (`vocab.txt`)
- **C++17 compiler** (GCC 7+, Clang 5+, MSVC 2019+)

## Troubleshooting

### Qt Not Found
```bash
# Find Qt location
find /usr -name "Qt5Config.cmake" 2>/dev/null

# Tell CMake where Qt is
cmake .. -DCMAKE_PREFIX_PATH=/path/to/qt -DBUILD_GUI=ON
```

### Model Not Loading
```bash
# Use absolute paths
./src/chatbot_gui /absolute/path/to/vocab.txt /absolute/path/to/model.bin

# Or run from correct directory
cd /path/with/vocab/and/model
/path/to/build/src/chatbot_gui
```

### Slow Generation
- Reduce max response length (50-100 tokens)
- Use Greedy or Nucleus strategy
- Avoid Beam Search with high beam width

## Files Created

- `src/ChatbotGUI.hpp` - GUI header with Qt class definition
- `src/ChatbotGUI.cpp` - GUI implementation with Qt widgets
- `src/ChatbotGUI_main.cpp` - Application entry point
- `src/CMakeLists.txt` - Updated build configuration (Qt target)
- `docs/guides/chatbot-gui-guide.md` - Complete documentation

## Architecture

```
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

- [Complete GUI Guide](chatbot-gui-guide.md) - Full documentation
- [CLI Guide](chatbot-guide.md) - Command-line interface
- [Training Guide](../reference/chatbot-completeness.md) - Model training

## Support

The GUI uses the same underlying components as the CLI, so:
- Generation quality is identical
- All features from CLI are available
- Settings behave the same way

For detailed information about generation strategies and parameters, see the main chatbot documentation.
