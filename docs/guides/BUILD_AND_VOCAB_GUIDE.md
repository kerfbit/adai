# ADAI Build & Vocabulary Creation Guide

This guide explains how to build the ADAI project and create vocabularies for training chatbot models.

## Quick Start

### Option 1: Interactive Build Script (Recommended)

```bash
# Make script executable (first time only)
chmod +x build_and_vocab.sh

# Launch interactive menu
./build_and_vocab.sh

# Or use direct commands
./build_and_vocab.sh build          # Build in release mode
./build_and_vocab.sh vocab 5000     # Create 5000-token vocabulary
./build_and_vocab.sh train 10       # Train for 10 epochs
./build_and_vocab.sh full           # Complete workflow
```

### Option 2: Manual Build

```bash
# Create and enter build directory
mkdir -p build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build with all cores
make -j$(nproc)

# Executables will be in build/bin/
```

## Build Tools Overview

### Main Executables

| Executable | Purpose |
| ----------- | --------- |
| `chatbot` | Interactive chatbot CLI |
| `chatbot_trainer` | Model training tool |
| `vocab_builder` | Vocabulary creation utility |
| `chatbot_api_server` | REST API server |
| `chatbot_gui` | Qt-based graphical interface |

## Vocabulary Creation

### Method 1: Using VocabBuilder (Standalone Tool)

The `vocab_builder` is a dedicated tool for creating BPE vocabularies.

#### Basic Usage

```bash
# Build from plain text
./build/bin/vocab_builder \
    --input training_data.txt \
    --output vocab.txt \
    --vocab-size 5000

# Build from chatbot training data (INPUT/RESPONSE format)
./build/bin/vocab_builder \
    --input sample_training_data.txt \
    --output vocab.txt \
    --vocab-size 8000 \
    --format pairs \
    --stats

# Build from multiple files
./build/bin/vocab_builder \
    --input file1.txt \
    --input file2.txt \
    --input file3.txt \
    --output vocab.txt \
    --vocab-size 10000 \
    --stats
```

#### Input Formats

**Plain Text** (`--format plain`)

```text
One sentence per line.
Each line is treated as a separate sample.
This is the default format.
```

**Pairs Format** (`--format pairs`)

```text
INPUT: User message here
RESPONSE: Assistant response here

INPUT: Another user message
RESPONSE: Another response
```

**JSON Format** (`--format json`)

```json
[
  "First text sample",
  "Second text sample",
  "Third text sample"
]
```

#### VocabBuilder Options

| Option | Description | Default |
| -------- | ------------- | --------- |
| `--input <file>` | Input text file (can use multiple times) | Required |
| `--output <file>` | Output vocabulary file | Required |
| `--vocab-size <N>` | Target vocabulary size | 5000 |
| `--threshold <N>` | Minimum character frequency | 1 |
| `--format <type>` | Input format: plain, pairs, json | plain |
| `--stats` | Show vocabulary statistics | Off |
| `--help` | Show help message | - |

### Method 2: Using ChatbotTrainer

The trainer can also build vocabularies as part of the training process.

```bash
./build/bin/chatbot_trainer \
    --data sample_training_data.txt \
    --build-vocab \
    --vocab-size 5000 \
    --output-vocab vocab.txt
```

### Method 3: Using the Build Script

```bash
./build_and_vocab.sh vocab 5000 my_vocab.txt
```

## Training a Model

### Prerequisites

1. Training data in INPUT/RESPONSE format
2. Vocabulary file (created using one of the methods above)

### Training Command

```bash
./build/bin/chatbot_trainer \
    --data sample_training_data.txt \
    --vocab vocab.txt \
    --output chatbot_model.bin \
    --epochs 10 \
    --learning-rate 0.001 \
    --batch-size 4
```

### Training Data Format

Create a file `training_data.txt` with this format:

```text
INPUT: Hello!
RESPONSE: Hi there! How can I help you today?

INPUT: How are you?
RESPONSE: I'm doing well, thank you for asking!

INPUT: What is your name?
RESPONSE: I'm an AI assistant created to help you.
```

### Training Parameters

| Parameter | Description | Default | Recommended |
| ----------- | ------------- | --------- | ------------- |
| `--epochs` | Number of training epochs | 10 | 10-50 |
| `--learning-rate` | Learning rate | 0.001 | 0.0001-0.01 |
| `--batch-size` | Training batch size | 4 | 4-32 |
| `--vocab-size` | Vocabulary size | 5000 | 5000-15000 |
| `--d-model` | Model dimension | 256 | 256-512 |
| `--num-heads` | Number of attention heads | 8 | 8-16 |

## Complete Workflow Example

### 1. Prepare Training Data

```bash
# Create or use existing training data
cat > my_training_data.txt << 'EOF'
INPUT: Hello
RESPONSE: Hi there! How can I help you?

INPUT: What's your name?
RESPONSE: I'm an AI assistant.

INPUT: Tell me a joke
RESPONSE: Why did the scarecrow win an award? He was outstanding in his field!

INPUT: Goodbye
RESPONSE: Goodbye! Have a great day!
EOF
```

### 2. Build the Project

```bash
# Using the build script
./build_and_vocab.sh build

# Or manually
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

### 3. Create Vocabulary

```bash
# Using vocab_builder
./build/bin/vocab_builder \
    --input my_training_data.txt \
    --output my_vocab.txt \
    --vocab-size 5000 \
    --format pairs \
    --stats
```

### 4. Train the Model

```bash
./build/bin/chatbot_trainer \
    --data my_training_data.txt \
    --vocab my_vocab.txt \
    --output my_model.bin \
    --epochs 20 \
    --learning-rate 0.001
```

### 5. Test the Model

```bash
# Interactive chatbot
./build/bin/chatbot \
    --vocab my_vocab.txt \
    --model my_model.bin

# Or load a specific epoch checkpoint
./build/bin/chatbot \
    --vocab my_vocab.txt \
    --model my_model.bin.epoch10
```

## Build Configuration Options

### CMake Build Types

```bash
# Release build (optimized, recommended for production)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build (with symbols, for development)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# RelWithDebInfo (optimized with debug symbols)
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

### Optional Features

```bash
# Enable all features
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DBUILD_EXAMPLES=ON \
      -DBUILD_GUI=ON \
      -DBUILD_API_SERVER=ON \
      ..

# Minimal build (core only)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_GUI=OFF \
      -DBUILD_API_SERVER=OFF \
      ..
```

## Troubleshooting

### Build Issues

**Problem**: CMake not found

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake
```

**Problem**: Compiler not found

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS (install Xcode Command Line Tools)
xcode-select --install
```

**Problem**: Qt not found (for GUI)

```bash
# Ubuntu/Debian
sudo apt-get install qt5-default qtbase5-dev

# macOS
brew install qt@5
```

**Problem**: httplib not found (for API server)

```bash
# Ubuntu/Debian
sudo apt-get install libhttplib-dev

# Or download manually
git clone https://github.com/yhirose/cpp-httplib.git external/cpp-httplib
```

### Vocabulary Issues

**Problem**: "No text data loaded"

- Check input file exists and is not empty
- Verify correct format (use `--format pairs` for INPUT/RESPONSE format)
- Try with `--help` to see usage examples

**Problem**: "Vocabulary too small"

- Increase `--vocab-size` parameter
- Provide more training data
- Lower `--threshold` value

### Training Issues

**Problem**: "Vocabulary file not found"

- Create vocabulary first using `vocab_builder`
- Check file path is correct

**Problem**: "Model generates gibberish"

- Train for more epochs (10-50)
- Increase training data size
- Use lower temperature for generation (0.3-0.7)
- Verify training loss is decreasing

## Performance Tips

### Build Performance

1. **Use Release mode**: `cmake -DCMAKE_BUILD_TYPE=Release`
2. **Parallel compilation**: `make -j$(nproc)`
3. **Enable ccache**: `cmake -DENABLE_CCACHE=ON` (if installed)

### Vocabulary Creation

1. **Larger vocabulary** = Better word coverage but slower training
2. **Smaller vocabulary** = Faster training but may miss uncommon words
3. **Recommended sizes**:
   - Small projects: 2,000-5,000 tokens
   - Medium projects: 5,000-15,000 tokens
   - Large projects: 15,000-50,000 tokens

### Training Performance

1. **Batch size**: Larger = faster training (if RAM allows)
2. **Learning rate**: Start with 0.001, adjust if needed
3. **Epochs**: Monitor validation loss to avoid overfitting
4. **Save checkpoints**: Models saved per epoch for recovery

## Example Scripts

### Quick Build and Test

```bash
#!/bin/bash
# quick_test.sh

# Build
./build_and_vocab.sh build

# Create small vocabulary for testing
./build/bin/vocab_builder \
    --input sample_training_data.txt \
    --output test_vocab.txt \
    --vocab-size 1000 \
    --format pairs

# Train for 5 epochs (quick test)
./build/bin/chatbot_trainer \
    --data sample_training_data.txt \
    --vocab test_vocab.txt \
    --output test_model.bin \
    --epochs 5

# Test the model
./build/bin/chatbot \
    --vocab test_vocab.txt \
    --model test_model.bin.epoch5
```

### Production Build

```bash
#!/bin/bash
# production_build.sh

# Clean build
rm -rf build
mkdir build && cd build

# Configure for maximum performance
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CCACHE=ON \
      -DBUILD_TESTING=OFF \
      -DBUILD_EXAMPLES=OFF \
      ..

# Build
make -j$(nproc)

cd ..

# Create production vocabulary
./build/bin/vocab_builder \
    --input production_data.txt \
    --output production_vocab.txt \
    --vocab-size 15000 \
    --format pairs

# Train production model
./build/bin/chatbot_trainer \
    --data production_data.txt \
    --vocab production_vocab.txt \
    --output production_model.bin \
    --epochs 50 \
    --learning-rate 0.0005

echo "✅ Production build complete!"
```

## Additional Resources

- [Training Guide](docs/guides/training-internals.md)
- [BPE Tokenizer API](docs/api/nlp/tokenizer.md)
- [ChatbotTrainer Documentation](docs/guides/chatbot-cli-internals.md)
- [Model Architecture](docs/architecture/transformer-design.md)

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review documentation in `docs/`
3. Check example programs in `src/*Example.cpp`
