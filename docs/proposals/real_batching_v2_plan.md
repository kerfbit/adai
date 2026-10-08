# Proposal: Real Batched Inference (BatchProcessor / BatchedInferenceEngine v2)

- **Status:** Proposed — owner decision October 8, 2026 to plan real batching as a v2 upgrade
- **Tracks:** [TD-171](../development/guides/TECHNICAL_DEBT.md#td-171-no-batch-dimension-anywhere-in-the-model-stack--real-parallel-batched-training-not-supported) (inference half),
  [TD-211](../development/guides/TECHNICAL_DEBT.md#td-211-batchedinferenceengine-queues-and-serializes-requests-but-never-batches-the-model),
  [TD-218](../development/guides/TECHNICAL_DEBT.md#td-218-batchprocessor-helpers-have-dead-surface-lost-ordering-and-unchecked-inputs);
  closes the loop on [TD-217](../development/guides/TECHNICAL_DEBT.md#td-217-chatbatch-reports-hypothetical-batching-stats-as-if-they-were-real)
- **Code-traced background:** [BatchedInferenceEngine.md](../development/reference/source/BatchedInferenceEngine.md),
  [BatchProcessor.md](../development/reference/source/BatchProcessor.md)

---

## 1. Problem

Two `stable` components are named for batching but can't batch:

- `BatchedInferenceEngine` (`chatbot_api_server --batched-inference`) queues requests and generates
  them **one at a time** on a single worker thread (TD-211).
- `BatchProcessor.hpp` builds padded `TokenBatch`es that **nothing can consume** (TD-218). Its only
  production use is the hypothetical padding `stats` in `/chat/batch` responses (TD-217).

Both are blocked on the same root cause: every layer from `EncoderDecoderModel::forward()` down to
`Matrix` processes exactly one sequence (TD-171).

**Decision (October 8, 2026):** keep both components, rather than retiring them the way TD-170
retired `TokenBatchLoader`, and deliver real batching as a **v2** of each. Until then, v1.x gets
honesty and robustness fixes only (§6).

## 2. Goal and non-goals

**Goal:** serve N concurrent generation requests with one matrix multiply per weight matrix per
decode step across all N, instead of N separate ones. That's where batching's throughput comes
from: better use of CPU SIMD/cache and GPU occupancy on the same GEMMs.

**Non-goals for v2:**
- Batched **training** (TD-171's other half). The training loop already gets an effective batch
  size from gradient accumulation; true batched backward is a separate project.
- Beam search inside a batch. Beam requests keep the existing per-request path.
- Changing model math or checkpoint format. Batched output must equal unbatched output (§5).

## 3. Recommended design: stacked ("packed") rows, not a 3-D tensor

Most layers here already treat each row of a `Matrix` independently:

| Layer | Row-independent today? | Batched change needed |
|---|---|---|
| `FeedForward` (`x·W1 + b1 → GELU → ·W2 + b2`) | Yes | None |
| `LayerNorm` (per-row mean/var) | Yes | None |
| `LanguageModelHead` (`x·W + b`) | Yes | None |
| `TokenEmbedding::forward(ids)` | Yes (lookup per id) | None, given concatenated ids |
| `PositionalEncoding::forward(x)` | **No**: uses the row's index in the matrix as its position | Take per-sequence offsets so positions restart at 0 for each sequence |
| `MultiHeadAttention` / `CrossAttention` | **No**: every row attends to every other row | Attend only within each sequence's own row range (block-diagonal) |

So if B sequences of lengths `L1…LB` are **stacked** into one `[ΣLi, d_model]` matrix, the
expensive dense layers run as one large GEMM unchanged, and only positional encoding and attention
need to know where each sequence starts. That's a far smaller change than adding a batch axis to
`Matrix` and every layer, the scope TD-171 originally assumed.

It also needs **no padding**: stacked rows are ragged, so no work is wasted on `PAD` tokens. Padding
and `create_padding_mask()` stay useful mainly for GPU attention kernels that want rectangular
tiles (§4, phase B6).

**Attention over a stack.** Per head, compute scores only inside each sequence's block
(`Q_i·K_iᵀ`), never across sequences. On CPU that's a loop over sequences inside the existing
per-head loop. It costs the same as today's per-sequence attention, so the gain comes entirely
from the dense layers. Masks become per-sequence (causal for self-attention; source-length for
cross-attention).

**Validate before building.** The whole case rests on "one `[ΣLi, d]×[d, d_ff]` GEMM is meaningfully
faster than B small ones." Phase B0 measures that first, on this repo's own `Matrix` and on the
SYCL path, before any layer changes.

## 4. Phases

| # | Phase | Delivers | Depends on | Resolves |
|---|---|---|---|---|
| **B0** | Feasibility spike | Benchmark: stacked GEMM vs. B separate GEMMs for realistic shapes (`d_model` 768, `d_ff` 3072, B ∈ {1,4,8,16,32}, short decode-step rows) on CPU and SYCL. Go/no-go with numbers in this doc | — | — |
| **B1** | `BatchProcessor` v2 (`@adai-version 2.0.0`) | `TokenBatch` gains `source_indices` and a packed form (`offsets` into a stacked id vector) alongside the padded form; per-sequence `[q, k]` mask builder (causal and/or length); bounds-checked `unbatch`/`unstack`; `Logger`-based stats (`print()` removed) | B0 go | TD-218 |
| **B2** | Stacked forward (inference only) | `PositionalEncoding` with per-sequence offsets; `MultiHeadAttention`/`CrossAttention` "stacked" forward that attends within blocks; `Encoder`/`Decoder`/`EncoderDecoderModel::forward_stacked()`. No backward | B1 | TD-171 (inference forward) |
| **B3** | Batched decode loop | Step all active sequences together: one stacked decoder forward per step; per-sequence EOS/`max_length`; per-sequence sampling config **and RNG**; one `DecoderKVCache` per sequence, with the KV-cached step stacking only the new token rows | B2 | TD-214 (RNG) |
| **B4** | `BatchedInferenceEngine` v2 (`@adai-version 2.0.0`) | `process_batch()` drives B3; **continuous batching** (admit queued requests between decode steps, retire finished ones); real per-request latency from `submit_time` | B3, plus v1.x TD-212 fixes already in | TD-211, TD-213 |
| **B5** | `/chat/batch` on real batching | `generate_batch_responses()` uses B3; the `padding_estimate` block (TD-217) is replaced or joined by measured figures | B3 | TD-217 follow-up |
| **B6** | GPU path | Stacked `GPUMatrix` forward; per-sequence attention kernels (or padded tiles using B1's padded form and masks); `GPUDecoderKVCache` per sequence. Validated on ai-machine (remote-only: needs a prepared package and run instructions) | B2–B4 | TD-171 GPU half for inference |

Each phase ends with full `ctest` green and, from B2 on, the equivalence tests in §5.

## 5. Acceptance criteria

- **Equivalence.** For any set of inputs, `forward_stacked()` rows for sequence *i* match
  `forward()` on sequence *i* alone, within float tolerance (CPU and GPU). Greedy generation
  through B3/B4 produces token-identical output to the unbatched path.
- **Isolation.** One request's error, EOS, or long generation never changes another's output or
  fails it (builds on TD-212).
- **Throughput.** `batched_inference_benchmark`, rewritten to use the real model rather than a
  sleep stub, shows a measured speedup at B ≥ 8 on the target host. The number goes in this doc,
  and the "10–20×" / "27.80×" claims are replaced with it.
- **Versioning.** `BatchProcessor.hpp` and `BatchedInferenceEngine.hpp` go to `2.0.0` with
  `@adai-reviewed` updated, per the [file-status standard](../development/guides/file-status-standard.md).
  `--batched-inference` keeps its name; its `--help` text describes the real behaviour.

## 6. Interim v1.x work (do now, independent of v2)

These don't wait for v2:

- **TD-212** (MEDIUM): per-request error isolation, full shutdown drain, `submit()`/shutdown race.
  B4 depends on these anyway.
- **TD-211 / TD-218 honesty pass:** rewrite file/class comments and `--help` text to describe
  today's queue-and-serialize and padding-stats behaviour, and point at this plan. Add bounds checks
  to `unbatch_outputs()`. Leave the dead config fields and helpers in place, documented as
  reserved for v2.
- **TD-217:** relabel `/chat/batch`'s `stats` as a padding estimate (decided October 8, 2026).
- **TD-213, TD-214, TD-219:** as filed.

## 7. Risks

- **B0 says no.** If the stacked GEMM isn't meaningfully faster on CPU (plausible for small
  `d_model` or when `Matrix` multiply is already memory-bound), v2 narrows to the GPU path only, or
  stops. Then TD-211/TD-218 fall back to the v1.x honesty fixes as the final state.
- **Attention dominates at long context.** Block-diagonal attention gets no batching win, so gains
  shrink as sequence length grows. B0 should include a long-context shape.
- **Hidden per-instance state.** Layers cache activations for backward (`cached_input` etc.) and,
  on GPU, persistent `GPUState`. A stacked forward must not corrupt the single-sequence path's
  caches. Keep forward-only stacked entry points separate, and keep `model_mutex_` (TD-156/TD-033)
  around them.
- **KV-cache bookkeeping.** Per-sequence caches with ragged lengths are the most bug-prone part of
  B3. Do greedy-equivalence tests on it before sampling.
