# ADAI - Advanced Deep Learning AI

A modern C++ implementation of transformer-based neural networks and natural language processing tools, featuring encoder-decoder architectures, multi-head attention, and a complete chatbot framework.

## 🌟 Features

- **Transformer Architecture**: Full encoder-decoder implementation with multi-head attention
- **Optimization Algorithms**: SGD, SGD with Momentum, Adam, and AdamW optimizers
- **NLP Tools**: BPE tokenization, text generation, and conversation management
- **Chatbot Framework**: Complete CLI and training tools for conversational AI
- **Comprehensive Testing**: 18 test suites with extensive coverage
- **Production Ready**: Memory-efficient, well-documented, and thoroughly tested

## 🚀 Quick Start

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14 or higher
- Google Test (included)

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/adai.git
cd adai

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run tests
ctest
```

### Running the Chatbot

```bash
# From build directory
./src/chatbot

# Or train a new chatbot
./src/chatbot_trainer --data ../sample_training_data.txt --vocab ../vocab.txt
```

## 📚 Documentation

Full documentation is available in the [`docs/`](docs/) directory:

- **[Documentation Index](docs/README.md)** - Complete documentation guide
- **[Quick Start Guide](docs/guides/quickstart.md)** - Get up and running quickly
- **[API Reference](docs/api/)** - Detailed API documentation
- **[Architecture Guide](docs/architecture/)** - System design and patterns
- **[User Guides](docs/guides/)** - Training and usage guides

## 🏗️ Project Structure

```
adai/
├── src/                    # Source code
│   ├── Core components (Matrix, Activation, Optimizer)
│   ├── Attention mechanisms (MultiHeadAttention, CrossAttention)
│   ├── Transformer blocks (Encoder, Decoder)
│   ├── NLP tools (BPETokenizer, TextGenerator)
│   └── Applications (ChatbotCLI, ChatbotTrainer)
├── tests/                  # Comprehensive test suites
├── docs/                   # Documentation
│   ├── api/               # API reference
│   ├── architecture/      # Design documentation
│   ├── guides/            # User guides
│   ├── testing/           # Test documentation
│   └── reference/         # Additional references
├── scripts/               # Build and utility scripts
├── legacy/                # Legacy implementations
└── tools/                 # Development tools
```

## 🧪 Testing

Run all tests:
```bash
cd build
ctest
```

Run specific test suites:
```bash
./tests/matrixTests
./tests/optimizerTests
./tests/multiheadattentionTests
./tests/chatbotcliTests
```

## 📊 Project Metrics

- **79** C++ source/header files
- **~32,500** lines of code
- **18** comprehensive test suites
- **59** documentation files
- **11** executable targets
- **25+** transformer/neural network components

## 🔧 Core Components

### Neural Network Core
- **Matrix**: Efficient matrix operations with operator overloading
- **Activation**: ReLU, Softmax, Tanh, Sigmoid functions
- **Optimizer**: SGD, Adam, AdamW with gradient clipping
- **Neural Network**: Base network with forward/backward propagation

### Transformer Architecture
- **Multi-Head Attention**: Scaled dot-product attention
- **Cross Attention**: Encoder-decoder attention mechanism
- **Encoder/Decoder Blocks**: Complete transformer components
- **Layer Normalization**: Stabilized training
- **Positional Encoding**: Position-aware embeddings

### NLP Pipeline
- **BPE Tokenizer**: Byte-Pair Encoding with vocabulary management
- **Text Generator**: Beam search and sampling-based generation
- **Conversation Context**: Multi-turn conversation handling

## 🎯 Use Cases

- **Chatbot Development**: Build conversational AI systems
- **Text Generation**: Generate coherent text sequences
- **Sequence-to-Sequence**: Translation, summarization, etc.
- **Research**: Experiment with transformer architectures
- **Education**: Learn transformer implementation details

## 🤝 Contributing

Contributions are welcome! Please refer to:
- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) for development guidelines
- [Documentation Guide](docs/README.md) for documentation standards

## 📈 Performance

- Optimized matrix operations
- Efficient memory management
- Gradient clipping and numerical stability
- Batch processing support

## 🗺️ Roadmap

- [ ] GPU acceleration support
- [ ] Additional optimization algorithms
- [ ] Pre-trained model zoo
- [ ] Python bindings
- [ ] Distributed training

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

Built with modern C++ and following best practices in deep learning implementation.

## 📞 Contact

[Your contact information]

---

**For detailed documentation, see [docs/README.md](docs/README.md)**
