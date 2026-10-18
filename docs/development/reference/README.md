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
  - [IntegratedInferenceEngine](source/IntegratedInferenceEngine.md) (`src/IntegratedInferenceEngine.hpp`; `--integrated-inference` three-thread request pipeline)
  - [KVCache](source/KVCache.md) (`src/KVCache.hpp`; CPU key/value cache for incremental decoding)
  - [LanguageModelHead](source/LanguageModelHead.md) (`src/LanguageModelHead.{hpp,cpp}`; output projection to vocabulary logits)
  - [LayerNorm](source/LayerNorm.md) (`src/LayerNorm.{hpp,cpp}`; per-token layer normalization)
  - [LeJEPAEncoder](source/LeJEPAEncoder.md) (`src/LeJEPAEncoder.{hpp,cpp}`; LeJEPA world-model encoder: pretraining step, save/load)
  - [Logger](source/Logger.md) (`src/Logger.{hpp,cpp}`; the spdlog wrapper every binary logs through)
  - [LoRA](source/LoRA.md) (`src/LoRA.hpp`; low-rank adapters for the attention projections)
  - [Matrix](source/Matrix.md) (`src/Matrix.{hpp,cpp}`; the dense float matrix: `operator*` dispatch, SIMD/OpenMP/BLAS/GPU paths)
  - [MatrixSIMD](source/MatrixSIMD.md) (`src/MatrixSIMD.hpp`; compile-time SIMD macros, CPUID detection, reduction helpers; which presets actually get AVX2)
  - [MetricsApiServerArgs](source/MetricsApiServerArgs.md) (`src/MetricsApiServerArgs.{hpp,cpp}`; `metrics_api_server`'s config layering: file, saved admin overrides, CLI)
  - [MetricsDatabase](source/MetricsDatabase.md) (`src/MetricsDatabase.hpp`; the metrics server's storage interface and backend factory)
  - [MetricsPushClient](source/MetricsPushClient.md) (`src/MetricsPushClient.{hpp,cpp}`; the trainer's queued HTTP push to `metrics_api_server`, plus the session-key helpers)
  - [MetricsSessionRegistry](source/MetricsSessionRegistry.md) (`src/MetricsSessionRegistry.hpp`; the metrics server's session map: cap, TTL eviction, archiving)
  - [MetricsTracker](source/MetricsTracker.md) (`src/MetricsTracker.hpp`; per-pass epoch history, CSV sidecar and overfitting warning)
  - [MnsCliCommands](source/MnsCliCommands.md) (`src/MnsCliCommands.{hpp,cpp}`; `mns_cli`'s argument parsing and request building)
  - [MnsCliTool](source/MnsCliTool.md) (`src/MnsCliTool.cpp`; `mns_cli`'s `main()`: options, config, dispatch, HTTP)
  - [MnsJsonHelpers](source/MnsJsonHelpers.md) (`src/MnsJsonHelpers.hpp`; the MNS Manager GUI's minimal JSON reader/writer and request bodies)
  - [MnsManagerGUI](source/MnsManagerGUI.md) (`src/MnsManagerGUI.{hpp,cpp}`, `MnsManagerGUI_main.cpp`; the optional Qt MNS manager)
  - [MnsServerArgs](source/MnsServerArgs.md) (`src/MnsServerArgs.{hpp,cpp}`; `mns_server`'s CLI and config layering)
  - [ModelNameClient](source/ModelNameClient.md) (`src/ModelNameClient.{hpp,cpp}`; the MNS client the trainer, chat server and `dataset_manager` use)
  - [ModelNameService](source/ModelNameService.md) (`src/ModelNameService.{hpp,cpp}`; the MNS server: records, lifecycle, roles, persistence)
  - [ModelNameServiceServer](source/ModelNameServiceServer.md) (`src/ModelNameServiceServer.cpp`; `mns_server`'s `main()`)
  - [ModelSerializer](source/ModelSerializer.md) (`src/ModelSerializer.{hpp,cpp}`; SafeTensors export/import with T5 tensor names)
  - [MultiHeadAttention](source/MultiHeadAttention.md) (`src/MultiHeadAttention.{hpp,cpp}`; per-head self-attention, KV-cached decode, LoRA hooks)
  - [Optimizer](source/Optimizer.md) (`src/Optimizer.{hpp,cpp}`; SGD/momentum/Adam/AdamW over parameter groups, clipping)
  - [ParquetReader](source/ParquetReader.md) (`src/ParquetReader.{hpp,cpp}`; dependency-free Parquet → JSONL for HuggingFace fetches)
  - [PerformanceProfiler](source/PerformanceProfiler.md) (`src/PerformanceProfiler.hpp`; timers, `Profiler`, `PROFILE_SCOPE`, `Benchmark`)
  - [PipelineInferenceEngine](source/PipelineInferenceEngine.md) (`src/PipelineInferenceEngine.hpp`; two-thread encoder/decoder pipeline behind `--pipeline-inference`)
  - [PortableSocket](source/PortableSocket.md) (`src/PortableSocket.hpp`; POSIX/Winsock socket shim for `FtpDataServer`)
  - [PortableTime](source/PortableTime.md) (`src/PortableTime.hpp`; portable UTC `gmtime`/`timegm`/`strptime`)
  - [PositionalEncoding](source/PositionalEncoding.md) (`src/PositionalEncoding.{hpp,cpp}`; fixed sinusoidal position encoding)
  - [PostgresMetricsDatabase](source/PostgresMetricsDatabase.md) (`src/PostgresMetricsDatabase.{hpp,cpp}`; PostgreSQL metrics backend, opt-in build)
  - [PPOOptimizer](source/PPOOptimizer.md) (`src/PPOOptimizer.hpp`; RLHF value function, GAE and clipped objective)
  - [Predictor](source/Predictor.md) (`src/Predictor.{hpp,cpp}`; LeJEPA embedding-space predictor)
  - [Quantization](source/Quantization.md) (`src/Quantization.hpp`; INT8/INT4 per-tensor quantization, not used in production)
  - [RAGInference](source/RAGInference.md) (`src/RAGInference.{hpp,cpp}`; retrieval-augmented generation for `chatbot_api_server`'s `RAG_ENABLED` path)
  - [RegistryServerArgs](source/RegistryServerArgs.md) (`src/RegistryServerArgs.{hpp,cpp}`; `registry_server` argument parsing and config precedence)
  - [RegistryServer](source/RegistryServer.md) (`src/RegistryServer.cpp`; `registry_server` endpoints, storage and `main()`)
  - [RegistryTransport](source/RegistryTransport.md) (`src/RegistryTransport.{hpp,cpp}`; queue/registry storage: `LocalTransport` flat files and `RemoteTransport` HTTP client)
  - [RewardModel](source/RewardModel.md) (`src/RewardModel.hpp`; Bradley–Terry RLHF reward MLP, no production entry point)
  - [RLHFTrainer](source/RLHFTrainer.md) (`src/RLHFTrainer.{hpp,cpp}`; RLHF rollout, reward scoring and policy-gradient step, no production entry point)
  - [SIGReg](source/SIGReg.md) (`src/SIGReg.{hpp,cpp}`; LeJEPA's sketched isotropic-Gaussian regularizer)
  - [SpecialTokens](source/SpecialTokens.md) (`src/SpecialTokens.hpp`; PAD/UNK/BOS/EOS constants and helpers)
  - [SpeculativeDecoding](source/SpeculativeDecoding.md) (`src/SpeculativeDecoding.hpp`; draft/verify generation behind `--draft-model`)
  - [SQLiteMetricsDatabase](source/SQLiteMetricsDatabase.md) (`src/SQLiteMetricsDatabase.{hpp,cpp}`; default metrics store and backend factory)
  - [StartupSweep](source/StartupSweep.md) (`src/StartupSweep.hpp`; `incremental_trainer`'s stale-download cleanup)
  - [TextGenerator](source/TextGenerator.md) (`src/TextGenerator.{hpp,cpp}`; greedy, sampling, top-k, nucleus and beam decoding)
  - [TokenEmbedding](source/TokenEmbedding.md) (`src/TokenEmbedding.{hpp,cpp}`; learnable token lookup table)
  - [TrainerAdminAPI](source/TrainerAdminAPI.md) (`src/TrainerAdminAPI.{hpp,cpp}`; per-pass admin HTTP API behind `trainer_service`)
  - [TrainerControlState](source/TrainerControlState.md) (`src/TrainerControlState.hpp`; per-pass state shared by the trainer and its admin API)
  - [TrainerServiceControlState](source/TrainerServiceControlState.md) (`src/TrainerServiceControlState.hpp`; `trainer_service`'s process-lifetime pause/stop state)
  - [TrainerServiceMain](source/TrainerServiceMain.md) (`src/TrainerServiceMain.cpp`; `trainer_service` supervisory loop)
  - [TrainerServiceProxy](source/TrainerServiceProxy.md) (`src/TrainerServiceProxy.{hpp,cpp}`; `trainer_service`'s public admin listener and child proxy)
  - [TrainingMetricsAPI](source/TrainingMetricsAPI.md) (`src/TrainingMetricsAPI.{hpp,cpp}`; `metrics_api_server`'s HTTP layer; endpoint catalogue in [TRAINING_METRICS_API.md](../TRAINING_METRICS_API.md))
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

- **[KVCache Reference](source/KVCache.md)** - Code-traced reference for the CPU key/value cache
  - Single-layer and multi-layer caching
  - Usage patterns and best practices
  - Performance characteristics
  - Integration examples
  - Troubleshooting guide

- **[PerformanceProfiler](source/PerformanceProfiler.md)** - Timing and profiling utilities (code-traced reference)
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
- [KVCache API](source/KVCache.md) - Detailed cache API reference

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
