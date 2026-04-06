# Development Documentation

Coding and development documentation for the ADAI project.

## Subdirectories

### [api/](api/)

Component API documentation. See [api/README.md](api/README.md) for the full index.

- **Core** — `matrix.md`, `optimizer.md`, `activation.md`
- **Transformer** — encoder, decoder, encoder-decoder model, attention heads, feed-forward, positional encoding, token embedding, layer norm, language model head
- **Attention** — `multihead-attention.md`, `cross-attention.md`
- **NLP** — `tokenizer.md`, `text-generator.md`, `conversation-context.md`
- **Memory / RAG** — `DocumentStore.md`, `RAGInference.md`
- **Data** — `batch-processing.md`, `batch-processing-quickref.md`, `dataset-batch-processing.md`
- **Advanced** — `quantization.md`, `speculative-decoding.md`, `ppo-optimizer.md`, `lora.md`, `reward-model.md`
- **REST API** — `rest-api.md`

### [architecture/](architecture/)

System architecture and design documentation:

- [decoder-architecture.md](architecture/decoder-architecture.md) — Decoder architecture overview
- [decoder-design.md](architecture/decoder-design.md) — Decoder design details
- [encoder-decoder-comparison.md](architecture/encoder-decoder-comparison.md) — Encoder vs decoder-only comparison
- [transformer-design.md](architecture/transformer-design.md) — Transformer design patterns

### [guides/](guides/)

Developer guides and implementation documentation:

- **Build & CI/CD** — [building.md](guides/building.md), [ci-cd.md](guides/ci-cd.md), [windows-cross-compilation.md](guides/windows-cross-compilation.md)
- **Git workflow** — [git-workflow.md](guides/git-workflow.md), [branch-protection.md](guides/branch-protection.md), [contributing.md](guides/contributing.md)
- **Implementation** — [implementation-guide.md](guides/implementation-guide.md), [implementation-checklist.md](guides/implementation-checklist.md), [phase5-advanced-features.md](guides/phase5-advanced-features.md)
- **Internals** — [chatbot-cli-internals.md](guides/chatbot-cli-internals.md), [incremental-trainer-internals.md](guides/incremental-trainer-internals.md)
- **Feature guides** — [AUGMENTATION_IMPLEMENTATION.md](guides/AUGMENTATION_IMPLEMENTATION.md), [BATCH_PROCESSING_INTEGRATION.md](guides/BATCH_PROCESSING_INTEGRATION.md), [OPENMP_IMPLEMENTATION.md](guides/OPENMP_IMPLEMENTATION.md), [RAG_IMPLEMENTATION_GUIDE.md](guides/RAG_IMPLEMENTATION_GUIDE.md)
- **Quick references** — [AUGMENTATION_QUICK_REFERENCE.md](guides/AUGMENTATION_QUICK_REFERENCE.md), [BATCH_PROCESSING_QUICK_REFERENCE.md](guides/BATCH_PROCESSING_QUICK_REFERENCE.md), [OPENMP_QUICK_REFERENCE.md](guides/OPENMP_QUICK_REFERENCE.md), [RAG_QUICK_REFERENCE.md](guides/RAG_QUICK_REFERENCE.md), [PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md](guides/PARALLEL_OPTIMIZATIONS_QUICK_REFERENCE.md), [incremental-trainer-quick-reference.md](guides/incremental-trainer-quick-reference.md), [dataset-quick-reference.md](guides/dataset-quick-reference.md)
- **Training** — [training-example.md](guides/training-example.md), [enhanced-training-pipeline.md](guides/enhanced-training-pipeline.md), [inference-optimization.md](guides/inference-optimization.md), [inference-optimization-quickstart.md](guides/inference-optimization-quickstart.md)
- **Data pipeline** — [data-pipeline-enhancement.md](guides/data-pipeline-enhancement.md), [dataset-enhanced-features.md](guides/dataset-enhanced-features.md)
- **Model management** — [save-load.md](guides/save-load.md)
- **Technical debt** — [TECHNICAL_DEBT.md](guides/TECHNICAL_DEBT.md), [technical-debt-management.md](guides/technical-debt-management.md), [PROCESS_IMPROVEMENT_PLAN.md](guides/PROCESS_IMPROVEMENT_PLAN.md), [PRIORITY1_CHECKLIST.md](guides/PRIORITY1_CHECKLIST.md)
- **Checklists** — [AUGMENTATION_CHECKLIST.md](guides/AUGMENTATION_CHECKLIST.md)

### [reference/](reference/)

Technical reference documentation. See [reference/README.md](reference/README.md) for the full index.

- [kvcache.md](reference/kvcache.md) — KV cache implementation details
- [batchprocessor.md](reference/batchprocessor.md) — Batch processor internals
- [performanceprofiler.md](reference/performanceprofiler.md) — Performance profiler API
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

### [proposals/](proposals/)

Design proposals and planning documents:

- [advanced_training_metrics_plan.md](proposals/advanced_training_metrics_plan.md) — Training metrics system design
- [llm_operations_tooling_plan.md](proposals/llm_operations_tooling_plan.md) — LLM operations tooling plan

### [archive/](archive/)

Historical documentation and completed phase summaries. See [archive/README.md](archive/README.md) for the index.

## Root-Level Documents

| File | Description |
| --- | --- |
| [configuration_guide.md](configuration_guide.md) | Configuration guide |
| [CONFIG_README.md](CONFIG_README.md) | Config system overview |
| [CONFIG_HOT_RELOAD_COMPLETE.md](CONFIG_HOT_RELOAD_COMPLETE.md) | Hot-reload implementation summary |
| [configuration_verification.md](configuration_verification.md) | Configuration verification notes |
| [DAEMON_IMPLEMENTATION_COMPLETE.md](DAEMON_IMPLEMENTATION_COMPLETE.md) | Daemon mode implementation summary |
| [daemon_service_plan.md](daemon_service_plan.md) | Daemon service design |
| [LOG_FILE_STANDARDS.md](LOG_FILE_STANDARDS.md) | Log file standards |
| [LOG_FILE_ROTATION_COMPLETE.md](LOG_FILE_ROTATION_COMPLETE.md) | Log rotation implementation summary |
| [TRAINING_METRICS_API.md](TRAINING_METRICS_API.md) | Training metrics REST API |
| [TRAINING_METRICS_SERVICE.md](TRAINING_METRICS_SERVICE.md) | Training metrics service internals |
| [SPECIAL_TOKEN_CONSOLIDATION.md](SPECIAL_TOKEN_CONSOLIDATION.md) | Special token consolidation notes |
| [VOCABULARY_PROTECTION_REPORT.md](VOCABULARY_PROTECTION_REPORT.md) | Vocabulary protection implementation |
| [TEST_COVERAGE_IMPROVEMENTS.md](TEST_COVERAGE_IMPROVEMENTS.md) | Test coverage improvement tracker |
| [transformer_introspection_plan.md](transformer_introspection_plan.md) | Transformer introspection design |
| [signal_handling_verification.md](signal_handling_verification.md) | Signal handling verification |
| [MIGRATION_COMPLETE.md](MIGRATION_COMPLETE.md) | Migration completion summary |

## Related Documentation

For operational and deployment documentation, see [../operations/](../operations/)
