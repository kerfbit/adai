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
  - [CrossAttention](source/CrossAttention.md) (`src/CrossAttention.{hpp,cpp}`; encoder-decoder attention)
  - [DaemonConfigStore](source/DaemonConfigStore.md) (`src/DaemonConfigStore.{hpp,cpp}`; persisted `/admin/config` overrides and session counters)
  - [DataFetcher](source/DataFetcher.md) (`src/DataFetcher.{hpp,cpp}`; Gutenberg/HuggingFace import for `registry_server`/`dataset_manager`)
  - [Dataset](source/Dataset.md) (`src/Dataset.hpp`, **retired** October 11, 2026; kept as a record for TD-014)
  - [DatasetManagerArgs](source/DatasetManagerArgs.md) (`src/DatasetManagerArgs.{hpp,cpp}`; `dataset_manager` per-command argument parsing)
  - [DatasetManagerTool](source/DatasetManagerTool.md) (`src/DatasetManagerTool.cpp`; `dataset_manager` CLI `main()`)
  - [DatasetRegistry](source/DatasetRegistry.md) (`src/DatasetRegistry.{hpp,cpp}`; pending queue, trained registry, training-file parser)
  - [DataTransport](source/DataTransport.md) (`src/DataTransport.hpp`; trainer-side FTP/FTPS download client)
  - [Decoder](source/Decoder.md) (`src/Decoder.{hpp,cpp}`; `LLMDecoder` stack: embedding, blocks, final norm, KV-cache and GPU decode)
  - [DecoderBlock](source/DecoderBlock.md) (`src/DecoderBlock.{hpp,cpp}`; Pre-LN decoder layer with gated world-model/hippocampal paths)
  - [DocumentStore](source/DocumentStore.md) (`src/DocumentStore.{hpp,cpp}`; RAG retrieval index)
  - [EfficientBatching](source/EfficientBatching.md) (`src/EfficientBatching.hpp`, **retired** October 11, 2026; bucketing kept as a v2-plan design note)
  - [EncoderBlock](source/EncoderBlock.md) (`src/EncoderBlock.{hpp,cpp}`; Pre-LN encoder layer)
  - [EncoderDecoderModel](source/EncoderDecoderModel.md) (`src/EncoderDecoderModel.{hpp,cpp}`; full chatbot model: training, generation, checkpoints, GPU path)
  - [HippocampalMemory](source/HippocampalMemory.md) (`src/HippocampalMemory.{hpp,cpp}`; bounded episodic memory with least-used eviction, swap file and associations)
  - [IMetricsReporter](source/IMetricsReporter.md) (`src/IMetricsReporter.hpp`; trainer-side metrics interface and `NullMetricsReporter`)
  - [IncrementalTrainer](source/IncrementalTrainer.md) (`src/IncrementalTrainer.{hpp,cpp}`; session/checkpoint manager around `ChatbotTrainer` for every training pass)
  - [IncrementalTrainerArgs](source/IncrementalTrainerArgs.md) (`src/IncrementalTrainerArgs.{hpp,cpp}`; `incremental_trainer`/`trainer_service` global flag parser and helpers)
  - [IncrementalTrainingTool](source/IncrementalTrainingTool.md) (`src/IncrementalTrainingTool.cpp`; `incremental_trainer` main: commands, fork-to-background, LeJEPA pass)
  - [LLMEncoder](source/LLMEncoder.md) (`src/encoder.hpp`, `src/LLMEncoder.cpp`; encoder stack: embedding, blocks, final norm; chat, pipeline and RAG encoders)
  - [FeedForward](source/FeedForward.md) (`src/FeedForward.{hpp,cpp}`; position-wise FFN in every encoder/decoder block and the LeJEPA predictor)
  - [FtpDataServer](source/FtpDataServer.md) (`src/FtpDataServer.hpp`; read-only FTP/FTPS server embedded in `registry_server` for per-file dataset delivery)
  - [GenerationQualityMetrics](source/GenerationQualityMetrics.md) (`src/GenerationQualityMetrics.{hpp,cpp}`; BLEU/ROUGE scorer for per-epoch generation quality)
  - [GPUUtils](source/GPUUtils.md) (`src/gpu/GPUUtils.hpp`; GPU backend gateway: CUDA `GPUManager`/`GPUMemory`, no-GPU stub, SYCL include)
  - [GPUUtils_SYCL](source/GPUUtils_SYCL.md) (`src/gpu/sycl/GPUUtils_SYCL.hpp`; SYCL `GPUManager` (in-order queue) and USM `GPUMemory`)
  - [MatrixGPU_SYCL](source/MatrixGPU_SYCL.md) (`src/gpu/sycl/MatrixGPU_SYCL.hpp`; SYCL kernel declarations and the SYCL `GPUMatrix`)
  - [MatrixGPU_SYCL.cpp](source/MatrixGPU_SYCL_cpp.md) (`src/gpu/sycl/MatrixGPU_SYCL.cpp`; SYCL/oneMKL kernels, run and verified on a oneAPI CPU device)
  - [MatrixGPU](source/MatrixGPU.md) (`src/gpu/MatrixGPU.hpp`; GPU kernel API, `GPUMatrix`, GPU KV caches)
  - [MatrixGPU.cu](source/MatrixGPU_cu.md) (`src/gpu/MatrixGPU.cu`; CUDA kernels and cuBLAS wrappers behind `GPUMatrix`)

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

- **[Dataset](source/Dataset.md)** - retired `Dataset` class (loading, splitting, batch processing), kept as a record
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
