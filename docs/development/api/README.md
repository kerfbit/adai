# API Documentation

Component and REST API reference documentation for the ADAI project.

## REST API

- [rest-api.md](rest-api.md) — Complete HTTP API reference (endpoints, request/response schemas, CLI flags, client examples)
- [batch-processing.md](batch-processing.md) — Batch processing API guide
- [batch-processing-quickref.md](batch-processing-quickref.md) — Batch processing quick reference
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) — API server implementation summary

## Training Metrics API

The `TrainingMetricsAPI` is a separate HTTP server (default port 8081) for real-time monitoring of training jobs. Full reference documentation lives alongside the service:

- [../TRAINING_METRICS_API.md](../TRAINING_METRICS_API.md) — Full REST API reference (12 endpoints, request/response schemas, CLI flags, integration examples)
- [../TRAINING_METRICS_SERVICE.md](../TRAINING_METRICS_SERVICE.md) — `TrainingMetricsService` C++ API, configuration, export formats, and integration guide

### Metrics API Quick Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/metrics/current` | Current snapshot — loss, perplexity, validation accuracy, BLEU/ROUGE, and more |
| `GET` | `/api/metrics/summary` | Aggregated training summary |
| `GET` | `/api/metrics/history` | Historical records (`max_records`, `session_id` query params) |
| `GET` | `/api/metrics/abnormal` | Samples flagged as anomalous by outlier detection |
| `GET` | `/api/metrics/generation-quality` | Current and per-epoch BLEU-4 / ROUGE-1/2/L scores (TD-016) |
| `GET` | `/api/metrics/prometheus` | Prometheus text format for scraping |
| `GET` | `/api/metrics/csv` | Current metrics in CSV |
| `GET` | `/api/session/status` | Session status and progress percent |
| `GET` | `/api/session/epochs` | Per-epoch loss, validation, perplexity arrays |
| `POST` | `/api/control/flush` | Force immediate disk flush |
| `POST` | `/api/control/clear` | Clear in-memory history |
| `GET` | `/health` | Server health check |

## Core Components

| File | Class | Description |
| --- | --- | --- |
| [../reference/source/Matrix.md](../reference/source/Matrix.md) | `Matrix` | Matrix operations and linear algebra |
| [core/optimizer.md](core/optimizer.md) | `Optimizer` | SGD, Adam, and optimizer base |
| [../reference/source/Activation.md](../reference/source/Activation.md) | Activations | ReLU, GELU, softmax functions |

## Transformer Components

| File | Class | Description |
| --- | --- | --- |
| [LLMEncoder.md](../reference/source/LLMEncoder.md) | `LLMEncoder` | Encoder stack (code-traced reference) |
| [EncoderBlock.md](../reference/source/EncoderBlock.md) | `EncoderBlock` | Single encoder layer (code-traced reference) |
| [Decoder.md](../reference/source/Decoder.md) | `LLMDecoder` | Decoder stack (code-traced reference) |
| [DecoderBlock.md](../reference/source/DecoderBlock.md) | `DecoderBlock` | Single decoder layer (code-traced reference) |
| [EncoderDecoderModel.md](../reference/source/EncoderDecoderModel.md) | `EncoderDecoderModel` | Full seq2seq model (code-traced reference) |
| [FeedForward.md](../reference/source/FeedForward.md) | `FeedForward` | Position-wise FFN (code-traced reference) |
| [transformer/positional-encoding.md](transformer/positional-encoding.md) | `PositionalEncoding` | Sinusoidal position encoding |
| [transformer/token-embedding.md](transformer/token-embedding.md) | `TokenEmbedding` | Token embedding layer |
| [LayerNorm.md](../reference/source/LayerNorm.md) | `LayerNorm` | Layer normalization (code-traced reference) |
| [LanguageModelHead.md](../reference/source/LanguageModelHead.md) | `LanguageModelHead` | Output projection to vocabulary logits (code-traced reference) |

## Attention

| File | Class | Description |
| --- | --- | --- |
| [attention/multihead-attention.md](attention/multihead-attention.md) | `MultiHeadAttention` | Multi-head self-attention |
| [CrossAttention.md](../reference/source/CrossAttention.md) | `CrossAttention` | Encoder-decoder cross-attention (code-traced reference) |

## NLP Components

| File | Class | Description |
| --- | --- | --- |
| [../reference/source/BPETokenizer.md](../reference/source/BPETokenizer.md) | `BPETokenizer` | Byte-pair encoding tokenizer |
| [nlp/text-generator.md](nlp/text-generator.md) | `TextGenerator` | Inference and decoding strategies |
| [../reference/source/ConversationContext.md](../reference/source/ConversationContext.md) | `ConversationContext` | Session and conversation management |

## Memory / RAG

| File | Class | Description |
| --- | --- | --- |
| [DocumentStore.md](../reference/source/DocumentStore.md) | `DocumentStore` | Document storage for RAG (code-traced reference) |
| [memory/RAGInference.md](memory/RAGInference.md) | `RAGInference` | Retrieval-augmented generation inference |

## Data

| File | Description |
| --- | --- |
| [Dataset.md](../reference/source/Dataset.md) | `Dataset` (retired October 11, 2026; record of the former API) |

## Advanced Features

| File | Class | Description |
| --- | --- | --- |
| [advanced/quantization.md](advanced/quantization.md) | `Quantization` | INT8/INT4 weight quantization |
| [advanced/speculative-decoding.md](advanced/speculative-decoding.md) | `SpeculativeDecoder` | Draft-model speculative decoding |
| [advanced/ppo-optimizer.md](advanced/ppo-optimizer.md) | `PPOOptimizer` | Proximal Policy Optimization |
| [LoRA.md](../reference/source/LoRA.md) | `LoRA` | Low-rank adaptation (code-traced reference) |
| [advanced/reward-model.md](advanced/reward-model.md) | `RewardModel` | RLHF reward model |

## Quick Reference

### Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Server health check |
| `POST` | `/chat` | Single-turn conversation |
| `POST` | `/chat/session` | Multi-turn conversation |
| `POST` | `/clear-session` | Clear session history |

### Server CLI Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--port <n>` | 8080 | Listening port |
| `--vocab <path>` | — | Vocabulary file (required) |
| `--model <path>` | — | Pre-trained model file |
| `--timeout <n>` | 30 | Session timeout (minutes) |
| `--d-model <n>` | 512 | Model dimension |
| `--num-heads <n>` | 8 | Attention heads |
| `--d-ff <n>` | 2048 | Feed-forward dimension |
| `--enc-layers <n>` | 6 | Encoder layers |
| `--dec-layers <n>` | 6 | Decoder layers |
| `--max-gen-len <n>` | 100 | Max response tokens |
| `--temperature <f>` | 1.0 | Sampling temperature |
| `--top-p <f>` | 0.9 | Nucleus sampling threshold |
| `--strategy <str>` | nucleus | `greedy`, `beam`, `temperature`, `top_k`, `nucleus` |

## Related Documentation

- [../reference/](../reference/) — KV cache, batch processor, performance profiler internals
- [../guides/RAG_IMPLEMENTATION_GUIDE.md](../guides/RAG_IMPLEMENTATION_GUIDE.md) — RAG integration guide
- [../guides/inference-optimization.md](../guides/inference-optimization.md) — Inference optimization
- [../../operations/deployment/](../../operations/deployment/) — Deployment guides
- [Architecture Overview](../architecture/) (TODO)
