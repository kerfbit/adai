# Proposal: Expand Lessons for Full Transformer Coverage

**Status:** Proposed
**Date:** June 27, 2026

## Context

The `docs/lessons/` directory contains 15 high-quality single-point lessons covering training planning, optimization, stability, and methodology. Two large architecture references (`architecting-attention.md` and `core-model-components.md`) cover transformer internals comprehensively.

However, the ADAI codebase now includes many advanced features — LoRA, quantization, speculative decoding, RAG, RLHF, KV cache, and multiple generation strategies — that have no corresponding lesson. The lessons also have no coverage of inference-time topics or the encoder-vs-decoder architectural design space.

This proposal adds **12 new lessons** organized into 3 new sections, plus a restructuring of the README to incorporate them.

## Gap Analysis Summary

**Currently covered (15 lessons):**
Training planning (3), optimization hyperparameters (4), stability/generalization (2), methodology (3), architecture reference (2)

**Not covered:**
Architecture design choices, loss functions, generation/decoding, inference optimization, fine-tuning, and all advanced features (LoRA, quantization, RAG, RLHF, speculative decoding)

## Proposed New Lessons

### New Section: Architecture & Design

These fill the gap between the high-level training lessons and the low-level `core-model-components.md` reference.

#### 1. `encoder-decoder-architecture.md`
**Summary:** Encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) — when each is appropriate, how attention masking differs (causal vs bidirectional), and the implications for task design.

**Key sections:**
- The three architectures and their attention patterns
- Causal masking vs bidirectional masking
- Cross-attention: connecting encoder and decoder
- Task-architecture alignment (classification, generation, translation)
- ADAI's encoder-decoder design and when decoder-only is a better fit

**See also:** `architecting-attention.md` Ch 1 (attention masking), `core-model-components.md` Ch 7 (model assemblies)

#### 2. `loss-functions-and-objectives.md`
**Summary:** Cross-entropy loss from first principles, why it works for language modeling, and the variants that modify it — label smoothing, focal loss, and sequence-level objectives.

**Key sections:**
- Cross-entropy for next-token prediction
- Token-level vs sequence-level loss
- Label smoothing mechanics (connects to `regularization-strategy.md`)
- Loss scaling for mixed precision (connects to `precision-and-stability.md`)
- Reading the loss curve: what different shapes mean

**See also:** `regularization-strategy.md` (label smoothing), `precision-and-stability.md` (loss scaling)

#### 3. `generation-strategies.md`
**Summary:** How to turn a trained model into useful output — greedy, beam search, top-k, nucleus (top-p), and temperature scaling. When each is appropriate and the trade-offs between quality, diversity, and speed.

**Key sections:**
- Greedy decoding: fast but degenerate
- Beam search: quality at the cost of diversity
- Temperature scaling: controlling the softmax sharpness
- Top-k and nucleus (top-p) sampling
- Repetition penalties and length normalization
- Choosing a strategy by use case (chatbot vs translation vs code)

**ADAI reference:** `src/TextGenerator.hpp` implements all five strategies. `docs/development/api/nlp/text-generator.md` documents the API.

#### 4. `kv-cache-and-inference-memory.md`
**Summary:** How KV caching eliminates redundant computation during autoregressive generation, its memory implications, and techniques to manage cache size — MQA, GQA, paged attention, and quantized caches.

**Key sections:**
- Why autoregressive generation recomputes attention
- KV cache: the core optimization
- Memory cost: `2 × layers × seq_len × d_model × batch_size`
- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
- Paged attention and dynamic allocation
- Quantized KV caches (INT8 keys/values)

**See also:** `architecting-attention.md` Ch 6 (efficiency optimizations), `docs/development/reference/kvcache.md`

### New Section: Advanced Techniques

Lessons for features implemented in ADAI that extend the base transformer.

#### 5. `quantization-strategy.md`
**Summary:** Reducing model precision from fp32/fp16 to INT8/INT4 for faster inference and lower memory, with calibration and accuracy trade-offs.

**Key sections:**
- Post-training quantization (PTQ) vs quantization-aware training (QAT)
- Symmetric vs asymmetric quantization
- Calibration: choosing representative data
- Per-tensor vs per-channel granularity
- When to quantize and expected accuracy impact
- ADAI's quantization modes and calibration API

**ADAI reference:** `src/Quantization.hpp` implements multiple quantization modes. `docs/development/api/advanced/quantization.md` documents the API.

#### 6. `lora-and-efficient-finetuning.md`
**Summary:** LoRA (Low-Rank Adaptation) and the principle of parameter-efficient fine-tuning — adding small trainable adapters instead of updating all parameters.

**Key sections:**
- Full fine-tuning cost and catastrophic forgetting
- Low-rank decomposition: W + BA where B and A are small
- Rank selection: the capacity vs efficiency trade-off
- Alpha/scale tuning and its interaction with learning rate
- Which layers to adapt (attention projections vs FFN)
- When LoRA is appropriate vs full fine-tuning

**ADAI reference:** `src/LoRA.hpp`. `docs/development/api/advanced/lora.md`.

#### 7. `retrieval-augmented-generation.md`
**Summary:** Augmenting generation with retrieved context — how RAG works, when it beats a larger model, and the design decisions in the retrieval and integration pipeline.

**Key sections:**
- Why retrieval: knowledge freshness and factual grounding
- The retrieval-then-generate pipeline
- Document store design: chunking, embedding, and indexing
- Context integration: prepend vs cross-attention vs fusion
- Retrieval quality vs generation quality trade-offs
- RAG vs fine-tuning on domain data

**ADAI reference:** `src/RAGInference.hpp`, `src/DocumentStore.hpp`. `docs/development/api/memory/`.

#### 8. `reward-models-and-rlhf.md`
**Summary:** Training language models from human preferences — reward model design, PPO mechanics, KL divergence penalties, and the alignment tax.

**Key sections:**
- The RLHF pipeline: SFT → reward model → PPO
- Reward model training from preference pairs
- PPO for language models: the policy gradient objective
- KL penalty: preventing reward hacking
- The alignment tax: capability vs safety
- Practical considerations: reward model size, data requirements

**ADAI reference:** `src/RewardModel.hpp`, `src/PPOOptimizer.hpp`. `docs/development/api/advanced/`.

#### 9. `speculative-decoding.md`
**Summary:** Using a small "draft" model to propose multiple tokens that the large model verifies in parallel, achieving 2-3x speedup without changing output distribution.

**Key sections:**
- The autoregressive bottleneck
- Draft-then-verify: the core algorithm
- Acceptance probability and the rejection criterion
- Draft model selection: size, architecture, distillation
- When speculative decoding helps (and when it doesn't)
- Integration with KV cache and batching

**See also:** `kv-cache-and-inference-memory.md`, `architecting-attention.md` Ch 6.5

### New Section: Fine-tuning & Transfer

#### 10. `fine-tuning-best-practices.md`
**Summary:** Adapting a pretrained model to a specific task or domain without destroying what it already knows.

**Key sections:**
- Transfer learning: what the pretrained model already knows
- Learning rate for fine-tuning (10-100x lower than pretraining)
- Layer freezing and gradual unfreezing
- Catastrophic forgetting: what it is and how to mitigate it
- Validation strategy for fine-tuning (small val sets, early stopping)
- Full fine-tuning vs LoRA vs prompt tuning decision tree

**See also:** `selecting-a-learning-rate.md`, `lora-and-efficient-finetuning.md`, `regularization-strategy.md`

#### 11. `curriculum-and-data-ordering.md`
**Summary:** Whether the order in which training data is presented matters, and when structured curricula (easy-to-hard) outperform random shuffling.

**Key sections:**
- Random shuffling as the default baseline
- Curriculum learning: sorting by difficulty
- Difficulty metrics: length, perplexity, label noise
- Self-paced learning: dynamic curriculum from model confidence
- Domain sequencing for multi-domain corpora
- When curriculum learning is not worth the complexity

**See also:** `data-quality-and-selection.md`, `batch-size-and-gradient-accumulation.md`

#### 12. `distributed-training-fundamentals.md`
**Summary:** Splitting training across multiple GPUs or machines — data parallelism, gradient synchronization, and the communication bottleneck.

**Key sections:**
- Data parallelism: replicate model, split batches
- Gradient all-reduce: synchronous vs asynchronous
- The communication bottleneck and gradient compression
- Tensor parallelism and pipeline parallelism (brief overview)
- Scaling efficiency: linear scaling rule revisited
- ADAI's OpenMP parallelism as a single-node analogue

**See also:** `batch-size-and-gradient-accumulation.md` (effective batch size across workers)

## Updated README Structure

```
## Training Planning (existing - 3 lessons)
## Optimization (existing - 4 lessons)
## Stability and Generalization (existing - 2 lessons)
## Methodology (existing - 3 lessons)

## Architecture & Design (NEW - 4 lessons)
- Encoder-Decoder Architecture
- Loss Functions and Objectives
- Generation Strategies
- KV Cache and Inference Memory

## Advanced Techniques (NEW - 5 lessons)
- Quantization Strategy
- LoRA and Efficient Fine-tuning
- Retrieval-Augmented Generation
- Reward Models and RLHF
- Speculative Decoding

## Fine-tuning & Transfer (NEW - 3 lessons)
- Fine-tuning Best Practices
- Curriculum and Data Ordering
- Distributed Training Fundamentals

## Architecture Reference (existing - 2 docs)
```

## Lesson Format

Each new lesson follows the established format:
- Title + `*ADAI Training — Single-Point Lesson*`
- `## The Core Idea` — 2-3 paragraph introduction
- Topic-specific H2 sections with formulas, code snippets, and diagrams where appropriate
- `## Common Mistakes` — pitfalls to avoid
- `## Quick Decision Checklist` — actionable summary in code block
- `*See also:*` — cross-references to related lessons

Target length: 5-10KB per lesson (matching existing lessons). The architecture reference docs remain separate at their larger sizes.

## Priority Order

**Write first** (core understanding, no existing coverage):
1. `loss-functions-and-objectives.md`
2. `generation-strategies.md`
3. `encoder-decoder-architecture.md`
4. `kv-cache-and-inference-memory.md`

**Write second** (advanced features already implemented in ADAI):
5. `quantization-strategy.md`
6. `lora-and-efficient-finetuning.md`
7. `fine-tuning-best-practices.md`

**Write third** (specialized topics):
8. `retrieval-augmented-generation.md`
9. `reward-models-and-rlhf.md`
10. `speculative-decoding.md`
11. `curriculum-and-data-ordering.md`
12. `distributed-training-fundamentals.md`

## Verification

After each lesson is written:
1. Check all `*See also:*` cross-references point to valid files
2. Verify any ADAI code references (`src/*.hpp`) still exist in the codebase
3. Update `docs/lessons/README.md` to include the new lesson in the correct section
4. Ensure the Reading Order section is updated for first-run planning
