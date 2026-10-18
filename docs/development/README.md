# Development Documentation

Coding and development documentation for the ADAI project.

## Subdirectories

### [api/](api/)

Component API documentation. See [api/README.md](api/README.md) for the full index.

- **Core** — moved to [reference/source/Activation.md](reference/source/Activation.md), [reference/source/Matrix.md](reference/source/Matrix.md) and [reference/source/Optimizer.md](reference/source/Optimizer.md)
- **Transformer** — all moved to [reference/source/](reference/README.md): [LLMEncoder](reference/source/LLMEncoder.md), [Decoder](reference/source/Decoder.md), [EncoderDecoderModel](reference/source/EncoderDecoderModel.md), [FeedForward](reference/source/FeedForward.md), [PositionalEncoding](reference/source/PositionalEncoding.md), [TokenEmbedding](reference/source/TokenEmbedding.md), [LayerNorm](reference/source/LayerNorm.md), [LanguageModelHead](reference/source/LanguageModelHead.md)
- **Attention** — moved to [reference/source/MultiHeadAttention.md](reference/source/MultiHeadAttention.md) and [reference/source/CrossAttention.md](reference/source/CrossAttention.md)
- **NLP** — all moved: TextGenerator, BPETokenizer and ConversationContext to [reference/source/TextGenerator.md](reference/source/TextGenerator.md), [reference/source/BPETokenizer.md](reference/source/BPETokenizer.md) and [reference/source/ConversationContext.md](reference/source/ConversationContext.md)
- **Memory / RAG** — moved to [reference/source/RAGInference.md](reference/source/RAGInference.md) and [reference/source/DocumentStore.md](reference/source/DocumentStore.md)
- **Data** — `batch-processing.md`, `batch-processing-quickref.md` (`Dataset` retired; record in [reference/source/Dataset.md](reference/source/Dataset.md))
- **Advanced** — all moved: SpeculativeDecoding, RewardModel, PPOOptimizer, LoRA and Quantization to [reference/source/SpeculativeDecoding.md](reference/source/SpeculativeDecoding.md), [reference/source/RewardModel.md](reference/source/RewardModel.md), [reference/source/PPOOptimizer.md](reference/source/PPOOptimizer.md), [reference/source/LoRA.md](reference/source/LoRA.md) and [reference/source/Quantization.md](reference/source/Quantization.md)
- **REST API** — `rest-api.md`

### [architecture/](architecture/)

System architecture and design documentation:

- [decoder-architecture.md](architecture/decoder-architecture.md) — Decoder architecture overview
- [decoder-design.md](architecture/decoder-design.md) — Decoder design details
- [encoder-decoder-comparison.md](architecture/encoder-decoder-comparison.md) — Encoder vs decoder-only comparison
- [transformer-design.md](architecture/transformer-design.md) — Transformer design patterns

### [guides/](guides/)

Developer guides and implementation documentation:

- **[building/](guides/building/)** — Building, CI/CD, Windows cross-compilation
- **[workflow/](guides/workflow/)** — Git workflow, branch protection, contributing
- **[training/](guides/training/)** — Training examples, enhanced pipeline, data pipeline, dataset features
- **[features/](guides/features/)** — Augmentation, batch processing, OpenMP, RAG, inference optimization, save/load
- **[internals/](guides/internals/)** — Chatbot CLI and incremental trainer architecture
- **[quick-reference/](guides/quick-reference/)** — Cheat sheets for augmentation, batch processing, dataset, inference, OpenMP, RAG
- **Implementation** — [implementation-guide.md](guides/implementation-guide.md), [implementation-checklist.md](guides/implementation-checklist.md), [phase5-advanced-features.md](guides/phase5-advanced-features.md)
- **Technical debt** — [TECHNICAL_DEBT.md](guides/TECHNICAL_DEBT.md), [technical-debt-management.md](guides/technical-debt-management.md), [PRIORITY1_CHECKLIST.md](guides/PRIORITY1_CHECKLIST.md)
- **File status & versioning** — [file-status-standard.md](guides/file-status-standard.md) — the per-file `@adai-status`/`@adai-version` tag; see [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for the generated dashboard

### [reference/](reference/)

Technical reference documentation. See [reference/README.md](reference/README.md) for the full index.

- [kvcache.md](reference/source/KVCache.md) — KV cache implementation details
- [source/BatchProcessor.md](reference/source/BatchProcessor.md) — Batch padding/grouping utilities (code-traced)
- [PerformanceProfiler.md](reference/source/PerformanceProfiler.md) — Performance profiler API
- [GRADIENT_OPERATIONS_WITHOUT_OPTIMIZER.md](reference/GRADIENT_OPERATIONS_WITHOUT_OPTIMIZER.md) — Gradient operations reference
- [VOCAB_TRAINING_ANALYSIS.md](reference/VOCAB_TRAINING_ANALYSIS.md) — Vocabulary training analysis
- [chatbot-completeness.md](reference/chatbot-completeness.md) — Component completeness tracking

### [testing/](testing/)

Testing documentation and test specifications:

- [bpe-tokenizer-tests.md](testing/bpe-tokenizer-tests.md)
- [chatbot-cli-tests.md](testing/chatbot-cli-tests.md)
- [chatbot-trainer-tests.md](testing/chatbot-trainer-tests.md)
- [encoder-decoder-tests.md](testing/encoder-decoder-tests.md)
- [neural-network-tests.md](testing/neural-network-tests.md)
- [neuron-layer-tests.md](testing/neuron-layer-tests.md)
- [neuron-tests.md](testing/neuron-tests.md)
- [optimizer-integration-tests.md](testing/optimizer-integration-tests.md)
- [vocab-test.md](testing/vocab-test.md)
- **Coverage reports** — [testing/coverage/](testing/coverage/) — per-component coverage reports for Matrix, Decoder, Encoder, Attention, FeedForward, TextGenerator, and more

### [../proposals/](../proposals/)

Design proposals and planning documents (unified directory). See [proposals/README.md](../proposals/README.md) for the index.

### [archive/](archive/)

Historical documentation, completed phase summaries, and implementation reports. See [archive/README.md](archive/README.md) for the index.

## Root-Level Documents

| File | Description |
| --- | --- |
| [configuration_guide.md](configuration_guide.md) | Configuration guide (includes parameter order and service script usage) |
| [LOG_FILE_STANDARDS.md](LOG_FILE_STANDARDS.md) | Log file standards |
| [TRAINING_METRICS_API.md](TRAINING_METRICS_API.md) | Training metrics REST API |
| [TRAINING_METRICS_SERVICE.md](TRAINING_METRICS_SERVICE.md) | Training metrics service internals |
| [reference/source/SpecialTokens.md](reference/source/SpecialTokens.md) | Special token constants and helpers (replaces SPECIAL_TOKEN_CONSOLIDATION.md) |
| [TEST_COVERAGE_IMPROVEMENTS.md](TEST_COVERAGE_IMPROVEMENTS.md) | Test coverage improvement tracker |
| [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) | Generated per-file status dashboard — see [file-status-standard.md](guides/file-status-standard.md) |
| [transformer_introspection_plan.md](../proposals/transformer_introspection_plan.md) | Transformer introspection design (proposal) |
| [code-citations.md](code-citations.md) | Code citations and references |

## Related Documentation

For operational and deployment documentation, see [../operations/](../operations/)
