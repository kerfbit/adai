# ADAI Reference Documentation

This directory contains reference materials, technical specifications, and implementation details for the ADAI project.

## 📖 Contents

### Core References

- **[Source file references](source/)** - Per-file, code-traced docs: every function, where it's called, and why it matters
  - [Activation](source/Activation.md) (`src/Activation.{hpp,cpp}`)
  - [BatchedInferenceEngine](source/BatchedInferenceEngine.md) (`src/BatchedInferenceEngine.hpp`)
  - [BatchProcessor](source/BatchProcessor.md) (`src/BatchProcessor.hpp`)
  - [BPETokenizer](source/BPETokenizer.md) (`src/BPETokenizer.{hpp,cpp}`)
  - [ChatbotAPI](source/ChatbotAPI.md) (`src/ChatbotAPI.{hpp,cpp}`)
  - [ChatbotApiServerArgs](source/ChatbotApiServerArgs.md) (`src/ChatbotApiServerArgs.{hpp,cpp}`)
  - [ChatbotAPIServer](source/ChatbotAPIServer.md) (`src/ChatbotAPIServer.cpp`, `chatbot_api_server`'s `main()`)
  - [ChatbotCLI](source/ChatbotCLI.md) (`src/ChatbotCLI.{hpp,cpp}`, `src/ChatbotCLI_main.cpp`; the `chatbot` client)
  - [ChatbotGUI](source/ChatbotGUI.md) (`src/ChatbotGUI.{hpp,cpp}`, `ChatbotGuiLogic.hpp`, `ChatbotGUI_main.cpp`, `ChatbotGUI_wrapper.cpp`; `chatbot_gui`)
  - [ChatbotTrainer](source/ChatbotTrainer.md) (`src/ChatbotTrainer.{hpp,cpp}`; the supervised training pass)
  - [ChildProcess](source/ChildProcess.md) (`src/ChildProcess.{hpp,cpp}`; `trainer_service`'s child launcher)
  - [Config](source/Config.md) (`src/Config.{hpp,cpp}`; `ServiceConfig`/`ConfigLoader`)
  - [ConversationContext](source/ConversationContext.md) (`src/ConversationContext.{hpp,cpp}`; multi-turn chat history)

- **[KVCache API Reference](kvcache.md)** - Complete API documentation for the Key-Value cache system
  - Single-layer and multi-layer caching
  - Usage patterns and best practices
  - Performance characteristics
  - Integration examples
  - Troubleshooting guide

- **[PerformanceProfiler API Reference](performanceprofiler.md)** - Complete API documentation for profiling tools
  - Timer, ScopedTimer, and Profiler classes
  - Statistical analysis (mean, median, percentiles)
  - Automated benchmarking
  - Before/after optimization comparison
  - Production monitoring patterns
  - Troubleshooting guide

- **[Dataset Batch Processing](../api/data/dataset-batch-processing.md)** - Complete guide to efficient batch processing ✨ NEW
  - Dataset batch methods (get_batch_with_padding, get_dynamic_batches)
  - TokenBatchLoader for multi-threaded loading
  - Dynamic batching optimization (20-40% token reduction)
  - Parallel loading (2-6x throughput improvement)
  - Training pipeline integration
  - Complete API reference and examples

### Project Planning

- **[Chatbot Completeness](chatbot-completeness.md)** - Feature completeness tracking and roadmap
  - Implementation status
  - Phase-by-phase development plan
  - Feature requirements

### Technical Documentation

- **[Gradient Operations Without Optimizer](GRADIENT_OPERATIONS_WITHOUT_OPTIMIZER.md)** - Manual gradient computation reference
  - Low-level gradient calculations
  - Implementation details

## 🔗 Related Documentation

### For API Usage

- See [API Reference](../api/README.md) for component APIs
- See [Guides](../guides/README.md) for usage tutorials

### For Performance Optimization

- [Inference Optimization Guide](../guides/inference-optimization.md) - Complete optimization guide
- [Quick Start](../guides/inference-optimization-quickstart.md) - 5-minute tutorial
- [KVCache API](kvcache.md) - Detailed cache API reference

### For Development

- [Architecture Documentation](../architecture/) - System design and patterns
- [Testing Documentation](../testing/) - Test suites and validation

## 📝 Adding New Reference Docs

When adding new reference documentation:

1. **Create the document** in this directory
2. **Update this README** with a link and brief description
3. **Update the main docs README** (`docs/README.md`) if it's a major reference
4. **Cross-link** from related guides and API docs

### Document Types

This directory is for:

- ✅ API reference documentation
- ✅ Technical specifications
- ✅ Implementation details
- ✅ Performance characteristics
- ✅ Project planning documents

This directory is NOT for:

- ❌ User guides (use `guides/` instead)
- ❌ Architecture overviews (use `architecture/` instead)
- ❌ Test documentation (use `testing/` instead)
- ❌ API component docs (use `api/` instead)

---

**Last Updated:** January 25, 2026
