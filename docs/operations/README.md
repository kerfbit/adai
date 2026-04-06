# Operations Documentation

This folder contains all **user-facing and operational documentation** for the ADAI project.

## Contents

### 📄 [OPERATIONS_MANUAL.md](OPERATIONS_MANUAL.md)

Consolidated single-document operations manual covering all topics below — build, vocabulary, training, running, configuration, RAG, deployment, monitoring, and troubleshooting.

---

### 📁 [guides/](guides/)

User guides and operational how-to documentation:

#### Getting Started

- **[quickstart.md](guides/quickstart.md)** - Quick start guide for new users
- **[chatbot-guide.md](guides/chatbot-guide.md)** - Using the chatbot CLI
- **[chatbot-gui-guide.md](guides/chatbot-gui-guide.md)** - Using the chatbot GUI
- **[COMMANDS.md](guides/COMMANDS.md)** - Command reference

#### Training

- **[training-guide.md](guides/training-guide.md)** - How to train models
- **[incremental-training-guide.md](guides/incremental-training-guide.md)** - Incremental training
- **[gutenberg-training-guide.md](guides/gutenberg-training-guide.md)** - Training with Project Gutenberg data

#### Building & Setup

- **[BUILD_AND_VOCAB_GUIDE.md](guides/BUILD_AND_VOCAB_GUIDE.md)** - Building and vocabulary setup
- **[WINDOWS_BUILD_QUICK_REFERENCE.md](guides/WINDOWS_BUILD_QUICK_REFERENCE.md)** - Windows build quick reference
- **[MODEL_SERVICE_MANAGER.md](guides/MODEL_SERVICE_MANAGER.md)** - Guide to using the model_service.sh script for background service management

#### Troubleshooting

- **[troubleshooting/README.md](guides/troubleshooting/README.md)** - Troubleshooting index
- **[CHATBOT_GUI_TROUBLESHOOTING.md](guides/troubleshooting/CHATBOT_GUI_TROUBLESHOOTING.md)** - GUI common issues and solutions
- **[CPP_WRAPPER_SOLUTION.md](guides/troubleshooting/CPP_WRAPPER_SOLUTION.md)** - C++ wrapper fix for pthread/library conflicts
- **[FIXING_UNK_GENERATION.md](guides/troubleshooting/FIXING_UNK_GENERATION.md)** - Resolving `<unk>` token generation after vocabulary repair
- **[INPUT_LENGTH_FIX.md](guides/troubleshooting/INPUT_LENGTH_FIX.md)** - Fixing input sequence length overflow
- **[MODEL_LOADING_FIX.md](guides/troubleshooting/MODEL_LOADING_FIX.md)** - Resolving model file loading failures
- **[SPECIAL_TOKEN_ISSUES.md](guides/troubleshooting/SPECIAL_TOKEN_ISSUES.md)** - Diagnosing and fixing special token ID mismatches
- **[THREAD_ERROR_FIX.md](guides/troubleshooting/THREAD_ERROR_FIX.md)** - Thread/pthread symbol error resolution
- **[TRAINING_FIX_STRATEGY.md](guides/troubleshooting/TRAINING_FIX_STRATEGY.md)** - Training loss and perplexity issue fixes

#### Quick Reference

- **[quick-reference/README.md](guides/quick-reference/README.md)** - Quick reference index
- **[QUICK_REFERENCE.md](guides/quick-reference/QUICK_REFERENCE.md)** - Build, vocabulary, training, and chatbot command cheat sheet
- **[GUI_QUICK_REFERENCE.md](guides/quick-reference/GUI_QUICK_REFERENCE.md)** - GUI command quick reference card
- **[GUI_QUICK_START.md](guides/quick-reference/GUI_QUICK_START.md)** - GUI quick start with parallel processing
- **[PARALLEL_STATUS.md](guides/quick-reference/PARALLEL_STATUS.md)** - Parallel processing (OpenMP) status summary

### 📁 [deployment/](deployment/)

Deployment and infrastructure documentation:

- **[deployment/README.md](deployment/README.md)** - Deployment overview and scenario guide
- **[docker.md](deployment/docker.md)** - Docker deployment guide (concise)
- **[DOCKER_DEPLOYMENT.md](deployment/DOCKER_DEPLOYMENT.md)** - Full Docker and Docker Compose deployment guide
- **[SYSTEMD_DEPLOYMENT.md](deployment/SYSTEMD_DEPLOYMENT.md)** - systemd service deployment guide

## Target Audience

This documentation is intended for:

- **End users** using the chatbot
- **Data scientists** training models
- **System administrators** deploying the system
- **Operations teams** maintaining production systems

## Related Documentation

For development and coding documentation, see [../development/](../development/)
