# Operations Documentation

This folder contains all **user-facing and operational documentation** for the ADAI project.

## Contents

### 📄 [OPERATIONS_MANUAL.md](OPERATIONS_MANUAL.md)

Consolidated single-document operations manual covering all topics below — build, vocabulary, training, running, configuration, RAG, deployment, monitoring, and troubleshooting.

---

### 📁 [guides/](guides/)

User guides and operational how-to documentation:

#### Getting Started

- **[chatbot-guide.md](guides/chatbot-guide.md)** - Using the chatbot CLI
- **[chatbot-gui-guide.md](guides/chatbot-gui-guide.md)** - Using the chatbot GUI
- **[COMMANDS.md](guides/COMMANDS.md)** - Command reference

#### Training

- **[training-guide.md](guides/training-guide.md)** - How to train models (includes incremental training and Gutenberg data)

#### Building & Setup

- **[WINDOWS_BUILD_QUICK_REFERENCE.md](guides/quick-reference/WINDOWS_BUILD_QUICK_REFERENCE.md)** - Windows build quick reference
- **[MODEL_SERVICE_MANAGER.md](guides/MODEL_SERVICE_MANAGER.md)** - Guide to using the model_service.sh script for background service management

#### Troubleshooting

- **[troubleshooting/README.md](guides/troubleshooting/README.md)** - Troubleshooting index
- **[CHATBOT_GUI_TROUBLESHOOTING.md](guides/troubleshooting/CHATBOT_GUI_TROUBLESHOOTING.md)** - GUI common issues and solutions
- **[CPP_WRAPPER_SOLUTION.md](guides/troubleshooting/CPP_WRAPPER_SOLUTION.md)** - C++ wrapper fix for pthread/library conflicts
- **[MODEL_LOADING_FIX.md](guides/troubleshooting/MODEL_LOADING_FIX.md)** - Resolving model file loading failures
- **[THREAD_ERROR_FIX.md](guides/troubleshooting/THREAD_ERROR_FIX.md)** - Thread/pthread symbol error resolution
- **[TRAINING_FIX_STRATEGY.md](guides/troubleshooting/TRAINING_FIX_STRATEGY.md)** - Training loss and perplexity issue fixes

#### Quick Reference

- **[quick-reference/README.md](guides/quick-reference/README.md)** - Quick reference index
- **[GUI_QUICK_REFERENCE.md](guides/quick-reference/GUI_QUICK_REFERENCE.md)** - GUI command quick reference card
- **[GUI_QUICK_START.md](guides/quick-reference/GUI_QUICK_START.md)** - GUI quick start with parallel processing

### 📁 [deployment/](deployment/)

Deployment and infrastructure documentation:

- **[deployment/README.md](deployment/README.md)** - Deployment overview and scenario guide
- **[docker.md](deployment/docker.md)** - Docker and Docker Compose deployment guide
- **[SYSTEMD_DEPLOYMENT.md](deployment/SYSTEMD_DEPLOYMENT.md)** - systemd service deployment guide

## Target Audience

This documentation is intended for:

- **End users** using the chatbot
- **Data scientists** training models
- **System administrators** deploying the system
- **Operations teams** maintaining production systems

## Related Documentation

For development and coding documentation, see [../development/](../development/)
