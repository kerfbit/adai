# ADAI Documentation

Welcome to the Advanced Deep Learning AI (ADAI) documentation. This guide will help you navigate through all aspects of the project, from getting started to deep architectural understanding.

## 🚀 Getting Started

- **[Quick Start Guide](guides/quickstart.md)** - Get running in 5 minutes
- **[Architecture Overview](architecture/neural-network.md)** - System design and components
- **[Training Guide](guides/training-guide.md)** - How to train models

## 📚 User Guides

### Chatbot
- **[Chatbot User Guide](guides/chatbot-guide.md)** - Using the chatbot CLI
- **[Chatbot CLI Internals](guides/chatbot-cli-internals.md)** - How the CLI works
- **[Training Internals](guides/training-internals.md)** - Training system details

### Development
- **[Building ADAI](guides/building.md)** - Build instructions, requirements, and troubleshooting
- **[Contributing Guide](guides/contributing.md)** - How to contribute to the project
- **[Git Workflow](guides/git-workflow.md)** - Branching, commits, and PR process
- **[Technical Debt Management](guides/technical-debt-management.md)** - How to track and resolve technical debt
- **[Training Example](guides/training-example.md)** - Complete training workflow
- **[Save/Load Models](guides/save-load.md)** - Model persistence
- **[Implementation Checklist](guides/implementation-checklist.md)** - Development checklist
- **[Implementation Guide](guides/implementation-guide.md)** - Implementation best practices

## 🔧 API Reference

### Core Components
- **[Matrix](api/core/matrix.md)** - Matrix operations and linear algebra
- **[Activation](api/core/activation.md)** - Activation functions (ReLU, Softmax, etc.)
- **[Optimizer](api/core/optimizer.md)** - SGD, Adam, AdamW optimizers
- **[Neural Network](api/core/neural-network.md)** - Base neural network implementation
- **[Neuron](api/core/neuron.md)** - Individual neuron component
- **[Neuron Layer](api/core/neuron-layer.md)** - Dense layer implementation

### Attention Mechanisms
- **[Multi-Head Attention](api/attention/multihead-attention.md)** - Self-attention implementation
- **[Cross Attention](api/attention/cross-attention.md)** - Encoder-decoder attention

### Transformer Components
- **[Encoder Block](api/transformer/encoder-block.md)** - Transformer encoder
- **[Decoder Block](api/transformer/decoder-block.md)** - Transformer decoder
- **[Encoder](api/transformer/encoder.md)** - Full encoder stack
- **[Decoder](api/transformer/decoder.md)** - Full decoder stack
- **[Encoder-Decoder Model](api/transformer/encoder-decoder-model.md)** - Complete transformer model
- **[Layer Normalization](api/transformer/layer-norm.md)** - Layer norm implementation
- **[Feed Forward](api/transformer/feed-forward.md)** - Position-wise FFN
- **[Positional Encoding](api/transformer/positional-encoding.md)** - Position embeddings
- **[Token Embedding](api/transformer/token-embedding.md)** - Token embeddings
- **[Language Model Head](api/transformer/language-model-head.md)** - Output projection layer

### NLP Utilities
- **[BPE Tokenizer](api/nlp/tokenizer.md)** - Byte-Pair Encoding tokenization
- **[Text Generator](api/nlp/text-generator.md)** - Text generation utilities
- **[Conversation Context](api/nlp/conversation-context.md)** - Conversation management

## 🏗️ Architecture Deep Dives

- **[Transformer Design](architecture/transformer-design.md)** - Complete transformer architecture
- **[Decoder Architecture](architecture/decoder-architecture.md)** - Decoder design patterns
- **[Decoder Design](architecture/decoder-design.md)** - Detailed decoder implementation
- **[Encoder-Decoder Comparison](architecture/encoder-decoder-comparison.md)** - Architectural comparison

## 🧪 Testing

- **[BPE Tokenizer Tests](testing/bpe-tokenizer-tests.md)** - Tokenizer test suite
- **[Chatbot CLI Tests](testing/chatbot-cli-tests.md)** - CLI testing
- **[Chatbot Trainer Tests](testing/chatbot-trainer-tests.md)** - Trainer testing
- **[Encoder-Decoder Tests](testing/encoder-decoder-tests.md)** - Model testing
- **[Neural Network Tests](testing/neural-network-tests.md)** - Core network tests
- **[Neuron Tests](testing/neuron-tests.md)** - Neuron component tests
- **[Neuron Layer Tests](testing/neuron-layer-tests.md)** - Layer testing
- **[Optimizer Integration Tests](testing/optimizer-integration-tests.md)** - Optimizer testing

## 📖 Reference

- **[Chatbot Completeness](reference/chatbot-completeness.md)** - Feature completeness analysis
- **[Gradient Operations](reference/GRADIENT_OPERATIONS_WITHOUT_OPTIMIZER.md)** - Manual gradient computation

## 🔄 Development & Process

- **[Process Improvement Plan](../PROCESS_IMPROVEMENT_PLAN.md)** - Development process improvements
- **[Archive Summary](../ARCHIVE_SUMMARY.md)** - Project history and archive

## 📊 Project Metrics

- **79** C++ source/header files
- **~32,500** lines of code
- **18** comprehensive test suites
- **52** documentation files
- **11** executable targets
- **25+** transformer/neural network components

## 📂 Documentation Structure

```
docs/
├── README.md (this file)        # Documentation index
├── architecture/                # System architecture and design
├── api/                         # API reference documentation
│   ├── core/                   # Core components
│   ├── attention/              # Attention mechanisms
│   ├── transformer/            # Transformer components
│   └── nlp/                    # NLP utilities
├── guides/                      # User and developer guides
├── testing/                     # Test documentation
└── reference/                   # Additional reference materials
```

## 🤝 Contributing

- **[Technical Debt Tracker](../TECHNICAL_DEBT.md)** - Known issues and improvement opportunities
- **[Process Improvement Plan](../PROCESS_IMPROVEMENT_PLAN.md)** - Development guidelines and roadmap

When adding new documentation:
1. Place API docs in the appropriate `api/` subdirectory
2. User guides go in `guides/`
3. Architecture docs go in `architecture/`
4. Test docs go in `testing/`
5. Update this index when adding new major documents

## 📝 License

See [LICENSE](../LICENSE) file in the root directory.
