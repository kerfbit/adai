# API Documentation

Component and REST API reference documentation for the ADAI project.

## REST API

- [rest-api.md](rest-api.md) — Complete HTTP API reference (endpoints, request/response schemas, CLI flags, client examples)
- [batch-processing.md](batch-processing.md) — Batch processing API guide
- [batch-processing-quickref.md](batch-processing-quickref.md) — Batch processing quick reference
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) — API server implementation summary

## Core Components

| File | Class | Description |
| --- | --- | --- |
| [core/matrix.md](core/matrix.md) | `Matrix` | Matrix operations and linear algebra |
| [core/optimizer.md](core/optimizer.md) | `Optimizer` | SGD, Adam, and optimizer base |
| [core/activation.md](core/activation.md) | Activations | ReLU, GELU, softmax functions |

## Transformer Components

| File | Class | Description |
| --- | --- | --- |
| [transformer/encoder.md](transformer/encoder.md) | `Encoder` | Encoder stack |
| [transformer/encoder-block.md](transformer/encoder-block.md) | `EncoderBlock` | Single encoder layer |
| [transformer/decoder.md](transformer/decoder.md) | `Decoder` | Decoder stack |
| [transformer/decoder-block.md](transformer/decoder-block.md) | `DecoderBlock` | Single decoder layer |
| [transformer/encoder-decoder-model.md](transformer/encoder-decoder-model.md) | `EncoderDecoderModel` | Full seq2seq model |
| [transformer/feed-forward.md](transformer/feed-forward.md) | `FeedForward` | Position-wise FFN |
| [transformer/positional-encoding.md](transformer/positional-encoding.md) | `PositionalEncoding` | Sinusoidal position encoding |
| [transformer/token-embedding.md](transformer/token-embedding.md) | `TokenEmbedding` | Token embedding layer |
| [transformer/layer-norm.md](transformer/layer-norm.md) | `LayerNorm` | Layer normalization |
| [transformer/language-model-head.md](transformer/language-model-head.md) | `LanguageModelHead` | Output projection and softmax |

## Attention

| File | Class | Description |
| --- | --- | --- |
| [attention/multihead-attention.md](attention/multihead-attention.md) | `MultiHeadAttention` | Multi-head self-attention |
| [attention/cross-attention.md](attention/cross-attention.md) | `CrossAttention` | Encoder-decoder cross-attention |

## NLP Components

| File | Class | Description |
| --- | --- | --- |
| [nlp/tokenizer.md](nlp/tokenizer.md) | `BPETokenizer` | Byte-pair encoding tokenizer |
| [nlp/text-generator.md](nlp/text-generator.md) | `TextGenerator` | Inference and decoding strategies |
| [nlp/conversation-context.md](nlp/conversation-context.md) | `ConversationContext` | Session and conversation management |

## Memory / RAG

| File | Class | Description |
| --- | --- | --- |
| [memory/DocumentStore.md](memory/DocumentStore.md) | `DocumentStore` | Document storage for RAG |
| [memory/RAGInference.md](memory/RAGInference.md) | `RAGInference` | Retrieval-augmented generation inference |

## Data

| File | Description |
| --- | --- |
| [data/dataset-batch-processing.md](data/dataset-batch-processing.md) | Dataset batch processing integration |

## Advanced Features

| File | Class | Description |
| --- | --- | --- |
| [advanced/quantization.md](advanced/quantization.md) | `Quantization` | INT8/INT4 weight quantization |
| [advanced/speculative-decoding.md](advanced/speculative-decoding.md) | `SpeculativeDecoder` | Draft-model speculative decoding |
| [advanced/ppo-optimizer.md](advanced/ppo-optimizer.md) | `PPOOptimizer` | Proximal Policy Optimization |
| [advanced/lora.md](advanced/lora.md) | `LoRA` | Low-rank adaptation fine-tuning |
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
