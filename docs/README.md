# ADAI Documentation

Welcome to the Advanced Deep Learning AI (ADAI) documentation. This guide will help you navigate through all aspects of the project, from getting started to deep architectural understanding.

## 📁 Documentation Structure

The documentation is organized into two main sections:

### 👨‍💻 [Development Documentation](development/)

For developers, contributors, and architects

Contains all coding and development-related documentation:

- **[API Documentation](development/api/)** - Component APIs and interfaces
- **[Architecture](development/architecture/)** - System design and architecture
- **[Testing](development/testing/)** - Test specifications and coverage
- **[Reference](development/reference/)** - Technical reference materials
- **[Developer Guides](development/guides/)** - Implementation guides, CI/CD, git workflow, internals
- **[Archive](development/archive/)** - Historical documentation and phase summaries

### 👥 [Operations Documentation](operations/)

For users, data scientists, and system administrators

Contains all user-facing and operational documentation:

- **[User Guides](operations/guides/)** - How to use the chatbot, train models, and configure features
- **[Deployment](operations/deployment/)** - Docker and infrastructure setup
- **[Troubleshooting](operations/guides/troubleshooting/)** - Common issues and solutions
- **[Quick References](operations/guides/quick-reference/)** - Quick reference cards and cheat sheets

## 🚀 Quick Start

### For New Users

- **[Quick Start Guide](operations/guides/quickstart.md)** - Get running in 5 minutes
- **[Chatbot User Guide](operations/guides/chatbot-guide.md)** - Using the chatbot
- **[Training Guide](operations/guides/training-guide.md)** - How to train models
- **[Docker Deployment](operations/deployment/docker.md)** - Containerized deployment

### For New Developers

- **[Contributing Guide](development/guides/contributing.md)** - How to contribute
- **[Building ADAI](development/guides/building.md)** - Build instructions
- **[Git Workflow](development/guides/git-workflow.md)** - Git best practices
- **[API Reference](development/api/)** - Component APIs

## 📖 For Developers

Documentation for those working on the codebase:

### Essential Guides

- [Contributing Guide](development/guides/contributing.md) - Contribution guidelines
- [Building ADAI](development/guides/building.md) - Build instructions and requirements
- [Git Workflow](development/guides/git-workflow.md) - Branching, commits, and PR process
- [CI/CD Pipeline](development/guides/ci-cd.md) - Continuous integration

### API & Architecture

- [API Documentation](development/api/) - Complete API reference
- [Architecture](development/architecture/) - System design documentation
- [Testing](development/testing/) - Test documentation

### Implementation Guides

- [Implementation Guide](development/guides/implementation-guide.md) - Best practices
- [RAG Implementation](development/guides/RAG_IMPLEMENTATION_GUIDE.md) - Retrieval-Augmented Generation
- [OpenMP Implementation](development/guides/OPENMP_IMPLEMENTATION.md) - Parallel processing
- [Training Internals](development/guides/training-internals.md) - Training system details
- [Technical Debt Management](development/guides/technical-debt-management.md) - Managing technical debt

## 👥 For Users and Operators

Documentation for using and operating the chatbot:

### Getting Started

- [Quick Start](operations/guides/quickstart.md) - 5-minute quick start
- [Chatbot Guide](operations/guides/chatbot-guide.md) - Using the chatbot CLI
- [Chatbot GUI Guide](operations/guides/chatbot-gui-guide.md) - Using the GUI

### Training

- [Training Guide](operations/guides/training-guide.md) - Complete training guide
- [Incremental Training](operations/guides/incremental-training-guide.md) - Continuous training
- [Gutenberg Training](operations/guides/gutenberg-training-guide.md) - Training with books

### Configuration & Optimization

- [Build and Vocab Guide](operations/guides/BUILD_AND_VOCAB_GUIDE.md) - Setup guide
- [Inference Optimization](operations/guides/inference-optimization.md) - Performance tuning
- [RAG Quick Reference](operations/guides/RAG_QUICK_REFERENCE.md) - RAG usage

### Deployment

- [Docker Deployment](operations/deployment/docker.md) - Production deployment

### Troubleshooting

- [Troubleshooting Index](operations/guides/troubleshooting/README.md) - Common issues
- [Training Fix Strategy](operations/guides/troubleshooting/TRAINING_FIX_STRATEGY.md)
- [GUI Troubleshooting](operations/guides/troubleshooting/CHATBOT_GUI_TROUBLESHOOTING.md)

## 📊 Project Highlights

- **87** C++ source/header files
- **~39,000** lines of code
- **730+** comprehensive test suites (100% pass rate)
- **65+** documentation files (~18,000+ lines)
- **30+** transformer/neural network components
- Production-ready AI chatbot with advanced features

## 🗂️ Directory Layout

```text
docs/
├── README.md (this file)
├── development/              # Developer & coding documentation
│   ├── api/                 # API reference
│   ├── architecture/        # System architecture
│   ├── testing/             # Test documentation
│   ├── reference/           # Technical reference
│   ├── guides/              # Developer guides
│   └── archive/             # Historical docs
└── operations/              # User & operations documentation
    ├── guides/              # User guides
    ├── deployment/          # Deployment docs
    ├── troubleshooting/     # Troubleshooting
    └── quick-reference/     # Quick refs
```

## 📄 License

See [../LICENSE](../LICENSE) for details.
