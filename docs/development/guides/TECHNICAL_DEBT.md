# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** September 13, 2026
**Total Items:** 12
**High Priority:** 1
**Medium Priority:** 6
**Low Priority:** 5
**Future Enhancements:** 19
**Resolved Items:** 156
**Deferred Decisions:** 2

## Recommended Execution Order

Analyzed September 12, 2026 to sequence the 14 active items by dependency and risk rather than
just priority label — several depend on each other or on a single owner decision, and the
"Medium/Low" labels alone don't capture that. Re-derive this ordering rather than trusting it
blindly once several of these items have moved.

**Tier 1 — Code fix landed, retrain still outstanding.** [TD-059](#td-059-multi-head-and-cross-attention-never-actually-split-into-heads)'s
owner decision landed September 12, 2026 (fix the attention math for real and retrain everything,
not document the current single-head behavior as-is), and the code side of that landed September
13, 2026 — `MultiHeadAttention`/`CrossAttention` now genuinely split into heads on both CPU and GPU
paths, verified via an independent-reference test and finite-difference gradient checks, full
ctest green. What's still outstanding is the actual retrain: this dev environment has no real
trained checkpoint or dataset to retrain against, so that step needs the user's own training
infrastructure. [TD-050](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)'s incremental
attention kernels are now unblocked to build against final math (the code fix is in);
[TD-033](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode) was never
blocked on this in the first place (its routing work doesn't touch attention math).

**Tier 2 — Contained, high-confidence wins** (proven patterns or small isolated scope, no design
ambiguity, can start immediately regardless of Tier 1's outcome):
- [TD-033](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode) —
  wired in and verified September 13, 2026 (code + concurrency fix + tests); only the before/after
  latency benchmark remains, blocked on real GPU hardware not available in this dev environment.
- TD-053 resolved September 13, 2026 (`/save`/`/load`/auto-save-on-exit implemented for real,
  transporting `ConversationContext::serialize()`/`deserialize()` over two new `ChatbotAPI`
  endpoints) — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-053-chatbotclis-save-and-load-commands-are-non-functional-everywhere).
  Its remaining doc-verification item continues as
  [TD-164](#td-164-chatbot-guidemd-needs-a-live-pair-verification-pass).

**Tier 3 — Unblocked, ready to pick up:** TD-034 resolved September 13, 2026 (real policy-ratio/KL
via a caller-supplied log-prob callback, plus a real `ValueFunction` backward pass) — see
[archive](../archive/TECHNICAL_DEBT_RESOLVED.md#td-034-ppooptimizers-core-update-loop-is-a-placeholder-not-real-ppo).
That was the only thing blocking
[TD-038](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)'s last
open item (`RewardModel` wiring), now unblocked — the other two open TD-038 items (LoRA/Quantization)
are deliberately deferred by prior user decision, not blocked, so TD-038 itself needs no separate pick.

**Tier 4 — Sustained, low-risk test-coverage investment** (systematic, already-validated pattern,
no open design questions): [TD-048](#td-048-android-uidientry-point-classes-are-untested-and-unreleased)
(Compose UI testing now adopted in both modules, 10/12 screens done September 13, 2026 — 2 screens
remain, both in `opsdashboard`, whose debug APK can't be installed/run on this sandbox's emulator
at all — `wear-sdk` shared-library requirement, no Wear-capable device available here — so further
`opsdashboard` screens stay compile-verified only pending real device access; `AdminScreen` (still
remaining) additionally needs the not-yet-built admin-action confirm-dialog test infrastructure
`ModelDetailScreen`'s own update flagged — `GroupDetailScreen`'s own confirm-dialog flow (already
done otherwise) needs the same infrastructure too; plus DI/entry-points remain, continuing the
now-proven ViewModel-then-screen-testing approach)
and [TD-037](#td-037-no-qt-test-infrastructure-for-gui-classes) (8-12h — the same shape for the
desktop Qt GUI).

**Tier 5 — Larger investigation, sequence after Tier 1:** [TD-050](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)
(20-28h). Its own action items note the concrete next step is still confirming whether the CPU
`DecoderKVCache` bug is even live via a real incremental-vs-full-recompute comparison — start
there, not the full GPU buildout, and ideally after TD-059 lands so the incremental attention
kernels are written against final math rather than math that might change under them.

**Tier 6 — Process, not code:** [TD-047](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history)'s
remaining items are cutting the first real Android release (small, whenever desired) and
resolving the `origin` repo's GitHub billing issue (on the account holder, not something to fix in
a coding session). [TD-039](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)
has no fixed action — it resolves once the trainer/metrics API surface stops changing, not before.

**Tier 7 — Standalone features, lowest urgency:** [TD-006](#td-006-fill-in-the-middle-fim-training-data-generation)
(6-8h, FIM training data) and [TD-014](#td-014-llm-operations-and-training-tooling-suite) (no
estimate, likely the largest remaining item). Note TD-014's planned `adai-weights-tool`
(quantization) overlaps with TD-038's deferred Quantization-wiring decision — scope those two
together when either is picked up, rather than designing quantization twice.
[TD-163](#td-163-attentionheadbenchmark-hangs-indefinitely) (root cause unknown) also belongs
here — a standalone benchmark binary, not gating anything, not part of `ctest`.

## Table of Contents

- [Overview](#overview)
- [Recommended Execution Order](#recommended-execution-order)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-059: Multi-Head and Cross-Attention Never Actually Split Into Heads](#td-059-multi-head-and-cross-attention-never-actually-split-into-heads)
  - [TD-050: GPU-Resident KV-Cache for Autoregressive Generation](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)
  - [TD-033: chatbot_api_server Inference Never Uses Persistent GPU-Resident Decode](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode)
  - [TD-014: LLM Operations and Training Tooling Suite](#td-014-llm-operations-and-training-tooling-suite)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
  - [TD-037: No Qt Test Infrastructure for GUI Classes](#td-037-no-qt-test-infrastructure-for-gui-classes)
  - [TD-038: Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)
  - [TD-039: Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)
  - [TD-047: Android Data/Repository/API Layer Has No CI or Release History](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history)
  - [TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased](#td-048-android-uidientry-point-classes-are-untested-and-unreleased)
  - [TD-163: AttentionHeadBenchmark Hangs Indefinitely](#td-163-attentionheadbenchmark-hangs-indefinitely)
  - [TD-164: chatbot-guide.md Needs a Live-Pair Verification Pass](#td-164-chatbot-guidemd-needs-a-live-pair-verification-pass)
- [Resolved Items](#resolved-items) (156 items — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md))
- [Future Improvements](#future-improvements)
  - [Performance Optimizations](#performance-optimizations)
  - [Code Quality](#code-quality)
  - [Developer Experience](#developer-experience)
  - [Configuration and Service Management](#configuration-and-service-management)
  - [Logging and Observability](#logging-and-observability)
  - [Container and Deployment](#container-and-deployment)
- [Deferred Decisions](#deferred-decisions)
  - [AMD Radeon / ROCm-HIP GPU Backend — Not Pursued](#amd-radeon--rocm-hip-gpu-backend--not-pursued)
  - [ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug](#threadsanitizer-data-races-in-matrixcpps-openmp-parallelized-code--confirmed-tool-limitation-not-a-bug)
- [Process Guidelines](#process-guidelines)
  - [Adding New Technical Debt](#adding-new-technical-debt)
  - [Prioritization Criteria](#prioritization-criteria)
  - [Resolving Technical Debt](#resolving-technical-debt)
- [Statistics](#statistics)
  - [By Priority](#by-priority)
  - [By Component](#by-component)
  - [Effort Distribution](#effort-distribution)
  - [Future Enhancements Summary](#future-enhancements-summary)
  - [Deferred Decisions Summary](#deferred-decisions-summary)
- [References](#references)

## Active Technical Debt

### TD-059: Multi-Head and Cross-Attention Never Actually Split Into Heads

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| **HIGH** | Open — code fix landed and verified (September 13, 2026); retraining not performed | Core Model Architecture | September 8, 2026 | 30-50 hours (implementation + full retrain/validation cycle) |

Description:
Found while reading `src/MultiHeadAttention.cpp`/`.hpp` end to end, then confirmed independently
in `src/CrossAttention.cpp`/`.hpp`. **The production attention computation used by every encoder
and decoder layer in this codebase has never actually implemented multi-head attention** — it
computes a single global attention pattern over the full `d_model` width, with `num_heads` having
no effect on the math at all beyond gating the constructor's divisibility check.

Specifically:

- `MultiHeadAttention::forward()` — called by both `EncoderBlock::forward()` (self-attention) and
  `DecoderBlock::forward()` (self-attention) — computes `cached_Q * cached_K.transpose()` where
  `cached_Q`/`cached_K` are the full `[seq_len, d_model]` projections. There is no per-head slicing
  anywhere in this function: no `d_k`-wide column ranges, no per-head loop, nothing. `Matrix::operator*`
  is a plain, general-purpose GEMM (confirmed by reading `Matrix.cpp`) with no head-aware behavior of
  its own, so this really is one attention computation across all `d_model` dimensions at once —
  mathematically identical to single-head attention with a head dimension of `d_model`.
- `MultiHeadAttention::forward_with_cache()` (the KV-cache-based autoregressive decode path) has the
  identical gap: `Q_new * K_full.transpose()` over the full width, no per-head split.
- Both of the above still scale by `1.0f / std::sqrt(static_cast<float>(d_k))` — correct *only* for a
  dot product contracted over `d_k` dimensions. Since the actual contraction here is over `d_model =
  num_heads * d_k` dimensions, the correct scale for what's actually computed would be
  `1/sqrt(d_model)`; using `1/sqrt(d_k)` instead makes the pre-softmax logits too large by a factor of
  `sqrt(num_heads)`, pushing softmax toward more peaked (lower-entropy) distributions than intended —
  a secondary numerical bug riding on top of the missing head split.
- `CrossAttention::scaled_dot_product_attention()` and `CrossAttention::forward_with_cache()` — the
  cross-attention path used by every `DecoderBlock::forward()` call — have the exact same two problems,
  implemented independently (this is a separate class, not inheriting `MultiHeadAttention`).
- `MultiHeadAttention::forward_parallel()` is the **only** method in either file that implements the
  design correctly: it explicitly slices `[h*d_k, (h+1)*d_k)` column ranges per head (`start_dim = h *
  d_k`), runs scaled dot-product attention per head with the correct `1/sqrt(d_k)` scale (correct
  *for that width*), and concatenates — matching the "Attention is All You Need" formulation exactly.
  **Grep confirms zero callers of `forward_parallel()` anywhere outside `MultiHeadAttention.cpp`/`.hpp`
  itself** — it is unreachable dead code, apparently written later (its own doc comment: "Properly
  splits input into num_heads... This provides 2-4x speedup") specifically to add the correct
  behavior, but never wired into `EncoderBlock`/`DecoderBlock` in place of `forward()`.

Why this was never caught: `tests/multiheadattention_test.cpp` has 45+ tests, but every one checks
generic properties — output shape, attention weights summing to 1 per row, non-negativity, gradient
existence, save/load round-trip — that hold identically whether or not heads are actually split.
Nothing compares `forward()`'s output against a hand-computed or `forward_parallel()`-derived
per-head reference, and nothing asserts that changing `num_heads` (with `d_model` fixed) changes the
attention pattern in a way consistent with genuine multi-head behavior. The same is true of
`CrossAttention`'s test suite.

**Decision made (September 12, 2026): fix the math and retrain.** Every model ever trained with
this codebase — every existing checkpoint — was trained under the current
(single-head-over-d_model, mis-scaled) math, and switching `forward()`/`forward_with_cache()` over
to `forward_parallel()`'s approach changes what every layer's `W_q`/`W_k`/`W_v`/`W_o` weights
actually mean at inference time, so every existing checkpoint needs retraining from scratch to be
meaningful under the new math. The owner of model quality chose this over documenting the current
single-head behavior as-is.

**Code fix landed September 13, 2026.** `MultiHeadAttention::forward()` now delegates to
`forward_parallel()`'s per-head logic (kept as a separate public method, not inlined, since
`benchmarks/AttentionHeadBenchmark.cpp` calls it directly to compare the parallel/sequential
branches); `forward_with_cache()`, `backward()`, `gpu_forward()`, and `gpu_backward()` were
rewritten the same way. `CrossAttention` got the identical treatment across all five entry points
(it had no `forward_parallel()` to reuse, so this is new code written directly). Both classes now
cache real per-head post-softmax weights (`cached_head_weights_`) for `backward()` to differentiate
through per-head, instead of one d_model-wide softmax; `get_attention_weights()` returns the
mean-across-heads matrix (still a valid row-stochastic distribution) for callers/hooks that just
want a single summary. The GPU paths extract/scatter each head's column slice using the existing
`matrix_copy_device_to_device_gpu()` primitive rather than new CUDA/SYCL kernels — deliberately,
since this sandbox has the CUDA and SYCL toolchains to *compile* against (confirmed: both the `gpu`
and `sycl` presets build `adai_attention` clean) but no physical GPU device to *run* either backend
on, so new low-level kernel code would have shipped with zero ability to catch a bug in it; reusing
an already-exercised primitive keeps that residual risk bounded. The two unused dead-code
`scaled_dot_product_attention()` private methods (one per class, both TD-059-carriers not called
by anything) were deleted along with the `cached_scores` members they were the only readers of.

Verification: a first regression-test draft (comparing `num_heads=1` vs `num_heads=4` outputs under
shared weights) turned out to be a weak check — `d_k = d_model/num_heads` changes with `num_heads`
regardless of whether heads are genuinely split, and the pre-fix scale factor already varied with
`d_k` on its own, so that draft passed against *both* the pre-fix and post-fix code. Replaced with
`MatchesIndependentPerHeadReference` in both `tests/multiheadattention_test.cpp` and
`tests/crossattention_test.cpp` — an independent, from-scratch reimplementation of the documented
per-head formula, computed from the class's own weights via public accessors, confirmed via the
standard revert-confirm-fail cycle to fail against the pre-fix code and pass against the fix.
Finite-difference gradient checks (`BackwardPassMatchesNumericalGradient`, both classes, `num_heads
> 1`) confirm the rewritten `backward()` is the correct gradient of the rewritten `forward()`.
Existing `EncoderBlockTest`/`DecoderBlockTest` numeric-gradient-check tests continue to pass
(composed correctness through `EncoderBlock`/`DecoderBlock`). Full targeted suites
(`multiheadattentionTests` 60/60, `crossattentionTests` 41/41, `encoderblockTests` 34/34,
`decoderblockTests` 28/28) and the full `ctest -j8` (127/127) all pass clean. `adai_attention`
compiles clean under both the `gpu` (CUDA) and `sycl` (Intel oneAPI) presets — see the residual
GPU-runtime-verification-gap note above.

**Retraining not performed.** This dev environment has no real trained checkpoint or training
dataset to retrain — `training_sessions/` here holds only test-fixture output from `ctest` runs
(`alpha-key`, `both-fields`, etc.), not a real model. Retraining every existing checkpoint under
the new math has to happen on whatever infrastructure holds the real training data and currently-
deployed checkpoints, which is outside this environment/session's reach. This item stays open until
that's done and the before/after quality benchmark below is run.

While investigating: `gpu_forward()`/`gpu_backward()` (not called out in the original filing) had
the identical bug — training already runs on GPU when available (`ChatbotTrainer::train_epoch()`
via `EncoderDecoderModel::gpu_forward()`/`gpu_backward()`, see TD-033's writeup for that call
chain), so this was the actual production training path, not just a documentation gap. Fixed
identically to the CPU path.

Action Items:

- [x] Wire `MultiHeadAttention::forward()`/`forward_with_cache()`/`backward()` through genuine
  per-head logic, and do the same for `CrossAttention`'s three CPU entry points.
- [x] Fix both scale factors — resolved as a side effect of the per-head rewrite; `1/sqrt(d_k)` was
  always the right constant, it was just being applied to the wrong (`d_model`-wide) contraction.
- [x] Fix `gpu_forward()`/`gpu_backward()` identically for both classes (found during this pass —
  not in the original filing's scope, but the same bug, and the actual training-time path).
- [x] Add tests that would have caught the original bug (see Verification above for why the first
  draft wasn't strong enough, and what replaced it).
- [x] Update `MultiHeadAttention.hpp`/`CrossAttention.hpp`'s class-level architecture comments; the
  `forward_parallel()`/`scaled_dot_product_attention()` dead-code question resolved as: keep
  `forward_parallel()` (real external caller in the benchmark), delete `scaled_dot_product_attention()`
  (genuinely unused by anything).
- [ ] Retrain from scratch and validate against a from-scratch baseline before declaring any
  existing deployment upgraded — needs the real training data/infrastructure this session doesn't
  have access to.
- [ ] Benchmark generation quality/perplexity before vs. after on the same held-out data to confirm
  genuine multi-head attention is actually an improvement, not just "different," before retiring
  the old checkpoints.

Files to Modify:

- `src/MultiHeadAttention.cpp` / `src/MultiHeadAttention.hpp` — done
- `src/CrossAttention.cpp` / `src/CrossAttention.hpp` — done
- `tests/multiheadattention_test.cpp` / `tests/crossattention_test.cpp` — done (new
  architecture-sensitive coverage)
- Every existing trained checkpoint (out of source-tree scope, and the real remaining cost driver
  of this item) — not done, needs the user's own training infrastructure

---

### TD-050: GPU-Resident KV-Cache for Autoregressive Generation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Planned | GPU / Inference / Training | July 4, 2026 | 20-28 hours |

Description:
Neither the CPU nor the GPU decode path has a working incremental KV-cache. On CPU, `generate_response_with_strategy()`'s greedy branch explicitly bypasses `DecoderKVCache`/`forward_with_cache()` via a documented workaround ("TODO: Fix KV cache to properly handle autoregressive generation" — `src/EncoderDecoderModel.cpp`) because the cache produces incorrect results for greedy decoding. `EncoderDecoderModel::gpu_generate_response()` (added to GPU-accelerate BLEU/ROUGE scoring during validation) inherits the same limitation by necessity: no GPU-resident cache exists at all, so every decode step recomputes the full sequence from scratch via `LLMDecoder::gpu_decode()` — O(n) work per step, O(n^2) total over a generation, instead of O(1) per step / O(n) total with a correct cache. This is functionally correct (mirrors the CPU workaround's algorithmic shape) but leaves an easy performance win on the table now that generation runs on GPU.

Action Items:

- [ ] Root-cause the existing CPU `DecoderKVCache` correctness bug (self-attention and/or cross-attention cache indexing) in `src/KVCache.hpp` / `src/Decoder.cpp` `forward_with_cache()` before building the GPU equivalent on top of the same flawed model.
      Partial progress (September 8, 2026, from a full end-to-end read of `KVCache.hpp`): traced and
      hand-verified as mathematically consistent between the cached and non-cached paths (i.e. ruled out
      as the source of the divergence) — `KVCache::append()`'s row-wise concatenation;
      `LLMDecoder::forward_with_cache()`'s inline positional-encoding reimplementation (its odd-index
      exponent `(i-1)/d_model` is algebraically identical to the canonical `PositionalEncoding.cpp`'s
      `2*(i/2)/d_model` for odd `i`, since integer `i/2` floors); its causal-mask construction for the
      new-tokens-only query range; `LayerNorm::forward()`'s per-row independence (so normalizing a
      single new token in isolation is provably identical to normalizing that same row within a full
      batch); and `CrossAttention::forward_with_cache()`'s cache-encoder-K/V-once-then-reuse logic.
      TD-059 (missing per-head split, now fixed — see its own entry) was present in both
      `MultiHeadAttention::forward()` and `forward_with_cache()` identically at the time of this
      trace, so it could not itself have been the source of a cached-vs-uncached *divergence* (it
      would have produced equally-wrong-but-matching output in both modes) — it's fixed now, as its
      own item, not as part of this one. Not yet checked: an actual
      multi-step incremental-decode-vs-single-shot-full-recompute numerical comparison with identical
      weights (the only way to confirm the bug is still live at all, and if so, localize which step
      first diverges) — this is the next concrete step, not yet attempted.
- [ ] Design a GPU-resident cache type (e.g. `GPUKVCache`) holding persistent per-layer `GPUMatrix` key/value buffers in `src/gpu/sycl/MatrixGPU_SYCL.hpp`, sized for `max_seq_length` and appended to in-place as new tokens are generated (no per-step malloc_device/free churn).
- [ ] Add incremental self-attention kernels that compute Q/K/V for only the newest token(s) and attend against the full cached K/V (mirrors the CPU cache's intent), plus a one-time cross-attention K/V cache populated from the encoder output and reused unchanged across all decode steps.
- [ ] Add `LLMDecoder::gpu_decode_step()` (single-token incremental decode using the cache) alongside the existing full-sequence `gpu_decode()` (retained for training's teacher-forced forward pass, which doesn't need a cache).
- [ ] Wire `EncoderDecoderModel::gpu_generate_response()` to use the new incremental path instead of recomputing the full sequence every step.
- [ ] Validate correctness against the existing full-recompute GPU path (identical token-for-token output for greedy decoding) and against the CPU path once its cache bug is fixed.
- [ ] Benchmark generation latency before/after for representative `max_length` values (e.g. 50, 100 tokens) to confirm the expected O(n) vs O(n^2) improvement.

Files to Modify:

- `src/KVCache.hpp` — fix existing CPU cache bug
- `src/Decoder.cpp` / `src/Decoder.hpp` — `forward_with_cache()` fix; new `gpu_decode_step()`
- `src/gpu/sycl/MatrixGPU_SYCL.hpp` / `src/gpu/sycl/MatrixGPU_SYCL.cpp` — new `GPUKVCache` type and incremental attention kernels
- `src/MultiHeadAttention.cpp` / `src/MultiHeadAttention.hpp` — GPU incremental self-attention using the cache
- `src/CrossAttention.cpp` / `src/CrossAttention.hpp` — one-time GPU cross-attention K/V cache
- `src/EncoderDecoderModel.cpp` / `src/EncoderDecoderModel.hpp` — `gpu_generate_response()` switched to incremental decode
- `tests/` — new coverage for cache correctness and generation parity

Context: added alongside `gpu_evaluate()` and `gpu_generate_response()` (July 2026), which GPU-accelerated validation loss and BLEU/ROUGE scoring — see git history around that change for the full-recompute implementation this replaces.

---

### TD-033: chatbot_api_server Inference Never Uses Persistent GPU-Resident Decode

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open — wired in and verified September 13, 2026; before/after benchmark not performed (no GPU hardware in this environment) | GPU / Inference / Performance | September 7, 2026 | 6-8 hours |

Description:
Originally filed as "Matrix GPU Dispatch Doesn't Use Persistent GPUMatrix Residency," on the claim that nothing in the training/inference hot path calls `to_gpu()`/`from_gpu()` and that TD-003's persistent-residency win was never realized outside code that explicitly opts in. That claim is only half true, and the half that's false changes where the fix belongs — this entry replaces it with the verified scope.

**Training already has full persistent GPU residency.** `ChatbotTrainer::train_epoch()` calls `EncoderDecoderModel::gpu_forward()`/`gpu_backward()` (`src/ChatbotTrainer.cpp`) under `#ifdef ADAI_ENABLE_GPU`, which cascades through `EncoderBlock`/`DecoderBlock::gpu_forward()`/`gpu_backward()` (`src/EncoderBlock.cpp`, `src/DecoderBlock.cpp`) into `MultiHeadAttention`/`FeedForward`/`LayerNorm::gpu_forward()`/`gpu_backward()`, all chaining `GPUMatrix` end-to-end with cached device-resident buffers (`cached_Q`, `cached_K`, `cached_hidden`, etc.). This is exactly TD-003's intended design (resolved May 3, 2026), fully wired up, and identical on both backends — `GPUMatrix` exposes the same API in `src/gpu/MatrixGPU.hpp` (CUDA) and `src/gpu/sycl/MatrixGPU_SYCL.hpp` (SYCL).

**The real, live gap is inference serving.** `ChatbotAPI::generate_response()` (`src/ChatbotAPI.cpp`) builds its `model_fn` around `EncoderDecoderModel::forward()` — the plain CPU `Matrix` path — never the GPU-resident one. `Matrix::operator*()` (`src/Matrix.cpp`) auto-dispatches per call to `Matrix::multiply_gpu()` whenever `GPUManager::is_available()` and both inner dimensions are ≥32; `multiply_gpu()` (and `add_gpu`/`transpose_gpu`/`scale_gpu`/`hadamard_gpu`) allocate device memory, upload, compute, and download on every single call, with zero reuse of `GPUMatrix`. So on a GPU build, every live chat request re-pays a full alloc+upload+compute+download for every matmul, in every layer, for every generated token — worse than the training case, since it's per-token request latency rather than amortized training throughput. Confirmed backend-symmetric: `GPUMemory` does a fresh `cudaMalloc` (`src/gpu/GPUUtils.hpp`) or `sycl::malloc_device` (`src/gpu/sycl/GPUUtils_SYCL.hpp`) per call either way — this is a `Matrix.cpp`/`ChatbotAPI.cpp` dispatch problem, not a CUDA-vs-SYCL difference.

`EncoderDecoderModel::gpu_generate_response()` (`src/EncoderDecoderModel.cpp`) already exists as a persistent-residency decode path, but it's wired only into `ChatbotTrainer`'s internal generation-quality backfill / BLEU-ROUGE scoring (`src/ChatbotTrainer.cpp`), never into `ChatbotAPI` — so the one caller that would most benefit (real chat latency) has never been connected to it.

Related: TD-050 (GPU-Resident KV-Cache) covers a different problem inside `gpu_generate_response()` itself — once called, it recomputes the full sequence from scratch every step (no KV-cache), which is O(n²) instead of O(n). TD-050 is about making `gpu_generate_response()` fast once used; this item is about it not being used by `chatbot_api_server` at all.

Discovered during the per-file production-readiness rollout (September 7, 2026) — see [file-status-standard.md](file-status-standard.md); corrected same day after tracing both GPU backends end-to-end.

**Wired in September 13, 2026.** `ChatbotAPI::generate_response()`'s plain path now checks
`GPUManager::is_available()` and, when true, calls a new
`EncoderDecoderModel::gpu_generate_response_with_strategy()` instead of building the CPU
`model_fn`/`TextGenerator` path — added alongside the existing `gpu_generate_response()` (kept
unchanged for its original BLEU/ROUGE-scoring caller in `ChatbotTrainer.cpp`) rather than
extending it, since `ChatbotAPI` needs per-request `strategy`/`temperature`/`top_k`/`top_p`/
`num_beams`, which the original method never took as parameters (it relied on whatever config
`generator` already held).

Confirmed by reading `TextGenerator.cpp` directly (not merely assumed) that a single model_fn
returning only the last position's logits — the same shape `gpu_generate_response()`'s model_fn
already used — is the correct, sufficient contract for *every* strategy: `generate_greedy()`,
`generate_sampling()`, `generate_top_k()`, `generate_nucleus()`, and `generate_beam_search()`
(called once per beam per step, with that beam's own token sequence) all extract only
`logits.rows - 1`'s row from whatever model_fn returns. This means, unlike the CPU
`generate_response_with_strategy()`, the GPU-resident version needs no separate KV-cache-vs-
no-cache branch per strategy — `gpu_decode()` already recomputes the full sequence from scratch
every call regardless (no GPU KV-cache exists yet — TD-050), so one model_fn serves every
strategy uniformly, including beam search.

**Found and fixed a real concurrency gap while doing this wiring — not merely a hypothetical
one.** `model_mutex_` (added by TD-156 to serialize `forward()`/`backward()`/`generate_response()`/
`generate_response_with_strategy()` against `chatbot_api_server`'s real concurrent request
threads) was deliberately *not* held across `gpu_forward()`/`gpu_backward()`/
`gpu_generate_response()`, justified at the time by "not reachable from chatbot_api_server's live
serving path" — a justification this exact fix removes. Confirmed by reading
`MultiHeadAttention::gpu_forward()` (and the analogous `EncoderBlock`/`LayerNorm`/`FeedForward`
methods it calls into): `encoder->gpu_encode()`/`decoder->gpu_decode()`/`lm_head->gpu_forward()`
all write through to those sub-objects' own persistent per-instance `GPUState` (`cached_Q`,
`cached_K`, `cached_head_weights`, `cached_attn_out`) — exactly as shared and exactly as
unsynchronized as the four CPU-side `cached_*` members TD-156 already found racing. Two
concurrent chat requests hitting the newly-wired GPU-resident path would have reintroduced that
same class of heap-corruption bug. Fixed by holding `model_mutex_` for the full duration of both
`gpu_generate_response()` and the new `gpu_generate_response_with_strategy()` — the same fix
shape TD-156 used, applied before this bug could ever manifest for real rather than after.

Action Items:

- [x] Give `ChatbotAPI::generate_response()`'s `model_fn` a GPU-resident branch that calls a
  GPU-resident decode path when `GPUManager::is_available()`, instead of unconditionally calling
  `model_->forward()`.
- [x] Confirm `TextGenerator`'s beam/top-k/nucleus/temperature strategies all work against the
  GPU-resident decode path — confirmed by reading `TextGenerator.cpp` directly; see above.
- [ ] Benchmark chat response latency before/after on a representative prompt/`max_length` on
  both CUDA and SYCL builds to confirm the per-token round-trip elimination — **not performed**:
  this dev environment has the CUDA (`nvcc`) and Intel oneAPI (`icpx`) toolchains but no physical
  GPU device in either configuration (confirmed directly — see TD-041's writeup for the same
  finding), so `GPUManager::is_available()` never becomes true here and the new branch, while
  compiled and reachable, has never actually executed end-to-end on real hardware. Needs the
  user's own GPU-equipped machine.
- [x] Leave `Matrix::multiply_gpu()` and siblings as-is for callers that only have CPU `Matrix`
  data and no persistent-residency alternative — untouched; this item routed the identified hot
  path around them rather than removing them.

Files Modified:

- `src/ChatbotAPI.cpp` — `generate_response()`'s plain path, GPU-resident branch
- `src/EncoderDecoderModel.hpp` / `src/EncoderDecoderModel.cpp` — new
  `gpu_generate_response_with_strategy()`; `model_mutex_` now held across both GPU generation
  methods (concurrency fix, see above)
- `tests/chatbotapi_test.cpp` — two new tests (functional correctness across every strategy;
  concurrent-request safety mirroring TD-156's own regression test), both `GTEST_SKIP()`-guarded
  on `GPUManager::is_available()`/`initialize()` for the reason given in the benchmark item above
  — real, permanent coverage for whenever a GPU is available, not exercised by this session's own
  verification run

---

### TD-014: LLM Operations and Training Tooling Suite

| Priority | Status | Component | Created |
|----------|--------|-----------|---------|
| MEDIUM | Planned | Tooling / Toolchain | March 8, 2026 |

Description:
As the core machine learning models mature, standalone tools are missing for dataset lifecycle, evaluation, and deployment optimization. We lack isolated binaries to handle quantization, standardized inference evaluation, dataset PII scrubbing/dedup, and concurrent load testing.

Action Items:

- Implement `adai-weights-tool` for FP16/INT8 quantization and converting weight formats.
- Develop `adai-eval` for deterministic pipeline benchmarking on standardized Q&A lists.
- Isolate data-filtering scripts into an `adai-data-prep` tool for reproducible data hygiene.
- Add a dedicated target in `CMakeLists.txt` for these auxiliary tools to avoid bloating main executables.

### TD-006: Fill-in-the-Middle (FIM) Training Data Generation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Planned | Training / Data Generation | February 17, 2026 | 6-8 hours |

Description:
Current training data for Project Gutenberg books uses consecutive sentence pairs (question → answer). Adding Fill-in-the-Middle (FIM) style training where the model predicts a middle sentence given first and last sentences would improve narrative understanding and coherence.

Current Behavior:

```cpp
// Simple consecutive pairs
INPUT: "What does this mean: [sentence 1]"
RESPONSE: "[sentence 2]"
```

Desired Behavior:

```cpp
// Add ~20% FIM-style pairs
INPUT: "Fill in the middle: <|first|>[sentence 1]<|last|>[sentence 3]"
RESPONSE: "[sentence 2]"
```

Benefits:

- Better understanding of narrative flow and context
- Improved coherence in generated responses
- Enhanced ability to reason about story structure
- More robust to partial context scenarios

Implementation Tasks:

- [ ] Add FIM data generation to `create_qa_pairs_from_text()`
- [ ] Define special tokens for FIM format (`<|first|>`, `<|middle|>`, `<|last|>`)
- [ ] Add configuration for FIM percentage (default: 20%)
- [ ] Ensure balanced distribution of regular and FIM pairs
- [ ] Add FIM-specific evaluation metrics
- [ ] Update documentation with FIM training approach
- [ ] Test on various text sources

Files to Modify:

- `src/DataFetcher.cpp` - Modify `create_qa_pairs_from_text()` (moved here from
  `IncrementalTrainer.cpp` in the TD-028 refactor; corrected September 7, 2026)
- `src/BPETokenizer.cpp` - Add FIM special tokens to vocabulary
- `src/DataFetcher.hpp` - Add FIM configuration options

Code Location:

`src/DataFetcher.cpp` — `create_qa_pairs_from_text()` (tagged with a `// TODO: TD-006` comment)

References:

- FIM approach used successfully in code completion models (Copilot, CodeGen)
- Paper: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al.)

Evaluation:

- Test on narrative coherence tasks
- Measure improvement in multi-turn conversation quality
- Compare perplexity on FIM vs standard test sets

---

### TD-037: No Qt Test Infrastructure for GUI Classes

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GUI / Testing | September 7, 2026 | 8-12 hours |

Description:
`ChatbotGUI.{cpp,hpp}` and `MnsManagerGUI.{cpp,hpp}` have no automated coverage, and this repo
has no QTest (or any Qt-aware test) infrastructure at all — every other test in `tests/` is a
plain GTest with no Qt event loop. Real widget behavior is hard to test without one; the more
tractable near-term step is separating non-widget logic (state transitions, signal/slot wiring
decisions, data formatting) out of the widget classes into plain C++ that GTest can already
exercise, deferring full widget testing until QTest is actually adopted.

Action Items:

- [ ] Decide whether to adopt QTest (`Qt::Test` component, `QTEST_MAIN`) as a second test
  framework alongside GTest, or to keep pushing logic out of the widget classes instead.
- [ ] Extract testable non-widget logic from `ChatbotGUI`/`MnsManagerGUI` into plain classes.
- [ ] Add tests for the extracted logic; add QTest-based tests for the remaining widget code if
  that framework is adopted.

Files to Modify:

- `src/ChatbotGUI.cpp` / `src/ChatbotGUI.hpp`
- `src/MnsManagerGUI.cpp` / `src/MnsManagerGUI.hpp`
- `tests/CMakeLists.txt` (if QTest is adopted)

---

### TD-038: Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open (5/7 non-blocked items done) | Advanced Features / Integration | September 7, 2026 | 16-24 hours |

Description:
`BatchedInferenceEngine`, `IntegratedInferenceEngine`, `PipelineInferenceEngine`,
`SpeculativeDecoding`, `LoRA`, `Quantization`, `RewardModel`, and `PerformanceProfiler` each had
real, passing dedicated tests — the gap wasn't test coverage, it was that nothing in `chatbot`,
`chatbot_api_server`, or `incremental_trainer` actually called any of them. Each needed its own
integration decision (a CLI flag, a config option, a training-mode switch) rather than one shared
fix — grouped here because they shared the identical structural gap, not because one change
resolves all eight.

**Update (September 12, 2026):** all five non-blocked, non-deferred items are now wired into
`chatbot_api_server` — see the commits below. Two of the five (`PipelineInferenceEngine`,
`IntegratedInferenceEngine`) turned out to be genuinely broken, not just untested: each had a real
bug that had never been exercised against real model components, because their own dedicated unit
test suites used mock/null types shaped to match the (buggy) production code rather than the real
dependency — exactly the failure mode this TD's title describes, confirmed in the most direct way
possible. `LoRA`/`Quantization` were explicitly scoped out after a user decision (both would
require touching `MultiHeadAttention`'s forward pass for any real integration — the same
foundational-class risk class as TD-059); `RewardModel` was genuinely blocked on TD-034, resolved
September 13, 2026 — the wiring itself is still a separate, not-yet-started integration decision.

Action Items:

- [x] `BatchedInferenceEngine`: wired into `ChatbotAPI::enable_batched_inference()` /
  `chatbot_api_server --batched-inference`. Found and fixed a real bug along the way:
  `process_batch()`'s per-response token-count stat called `tokenizer_->encode()` unguarded on a
  generated response that can legitimately be empty, and one request's empty response would
  spuriously fail every other request batched alongside it.
- [x] `PipelineInferenceEngine`: wired into `ChatbotAPI::enable_pipeline_inference()` /
  `chatbot_api_server --pipeline-inference`. Found and fixed a real bug: `decoder_worker()` called
  a method, `forward_with_cross_attention(tokens, encoder_output, nullptr)`, that does not exist
  on the real `LLMDecoder` — its own test suite and `PipelineBenchmark.cpp` both instantiated the
  class exclusively against mocks shaped to match that wrong call, silently masking the mismatch.
- [x] `IntegratedInferenceEngine`: wired into `ChatbotAPI::enable_integrated_inference()` /
  `chatbot_api_server --integrated-inference`. Found and fixed two real bugs, worse than
  `BatchedInferenceEngine`'s: `decoder_worker()` and `batcher_worker()` each called
  `tokenizer_->encode()` unguarded with no try/catch anywhere in either function, so an empty
  generated response or empty input text would escape the worker thread and call
  `std::terminate()` — crashing the entire `chatbot_api_server` process, not just failing one
  request. Confirmed via genuine reproduction (real, non-mock model components — this class isn't
  templated) before and after the fix.
- [x] `SpeculativeDecoding`: `ChatbotAPI::draft_model_` + `--draft-model`/
  `--speculative-candidates` CLI flags / `DRAFT_MODEL_PATH`/`SPECULATIVE_NUM_CANDIDATES` config
  keys.
- [x] `PerformanceProfiler`: `ChatbotAPI::enable_profiling()` / `chatbot_api_server --profile` →
  `GET /admin/profile`. Found and fixed a real, session-introduced thread-safety bug along the
  way: `Profiler`'s internal maps had no locking, and `active_timers` was keyed by section name
  alone, so concurrent same-name `start()`/`stop()` pairs (exactly `generate_response()`'s case
  under `chatbot_api_server`'s real thread pool) corrupted each other's recorded timings.
- [ ] `LoRA` / `Quantization`: explicitly deferred (see above) — not attempted this pass. Still
  needs a scoping decision: a standalone checkpoint-manipulation CLI tool (lower risk) vs. actually
  modifying `MultiHeadAttention`'s forward pass for real inference/training integration (the real
  thing, higher risk).
- [ ] `RewardModel`: unblocked now that TD-034's PPO fix has landed — still needs its own
  integration decision (how `chatbot_api_server`/`incremental_trainer` would drive an actual RLHF
  fine-tuning pass), not attempted as part of TD-034 itself.
- [x] Add an integration test per wired feature proving the wiring works end-to-end — done for all
  five above (ChatbotAPI-level integration tests plus live end-to-end verification against a real
  running `chatbot_api_server` process for each).

Files to Modify:

- `src/LoRA.hpp`, `src/Quantization.hpp`, `src/RewardModel.hpp` — remaining, unattempted work.
- Already done: `src/BatchedInferenceEngine.hpp`, `src/PipelineInferenceEngine.hpp`,
  `src/IntegratedInferenceEngine.hpp`, `src/SpeculativeDecoding.hpp`, `src/PerformanceProfiler.hpp`,
  `src/ChatbotAPI.{hpp,cpp}`, `src/ChatbotApiServerArgs.{hpp,cpp}`, `src/ChatbotAPIServer.cpp`,
  `src/Config.{hpp,cpp}`.

---

### TD-039: Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable

| Priority | Status | Component | Created |
|----------|--------|-----------|---------|
| MEDIUM | Open | Training / Metrics / Core | September 7, 2026 |

Description:
`ChatbotTrainer`, `IncrementalTrainer`, `TrainingMetricsService`, and `TrainingMetricsAPI` are the
largest files in the tree (2000-2700 lines each) and already have substantial test coverage
(14/19/7/5 test-file cross-references respectively) — this isn't a test-coverage gap like the
other items here. It's churn: all four gained real features during the same session this tracker
was built in (the trainer admin API, session registry work), and a file that's still actively
changing shape isn't a good candidate for a `stable` claim regardless of how well-tested its
current snapshot is. No effort estimate — this isn't a fixed-scope task, it resolves when the API
surface stops changing.

Action Items:

- [ ] Let the current round of feature work (trainer admin API, session-scoped metrics) land and
  settle — no more structural changes planned.
- [ ] Do one deliberate final correctness/API-freeze review pass per file.
- [ ] Only then bump each to `MAJOR >= 1` and `@adai-status: stable`.

Files to Modify:

- `src/ChatbotTrainer.cpp` / `src/ChatbotTrainer.hpp`
- `src/IncrementalTrainer.cpp` / `src/IncrementalTrainer.hpp`
- `src/TrainingMetricsService.cpp` / `src/TrainingMetricsService.hpp`
- `src/TrainingMetricsAPI.cpp` / `src/TrainingMetricsAPI.hpp`

---

### TD-047: Android Data/Repository/API Layer Has No CI or Release History

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open (CI added; release process defined, first real release still pending) | Android / CI | September 7, 2026 | 10-14 hours |

Description:
41 files across `android/app` (15) and `android/opsdashboard`/`android/wearsync` (26) — the
repository/network/DTO/poller/contract layer — each have real dedicated unit tests and pass, but
the entire Android surface landed in a single commit, both apps still declare
`versionName = "0.1.0"`, and no CI workflow builds or tests either app. Being tested in isolation
isn't the same as having a release process; that's the actual gap here, distinct from TD-048
(files with no test at all).

**Update (September 12, 2026):** `.github/workflows/android-ci.yml` added — builds `:app`/
`:opsdashboard` (`assembleDebug`) and runs `testDebugUnitTest` across all modules on every
`android/**` push/PR. Verified genuinely green on a real GitHub Actions run (kerfbit mirror), not
just locally — the first version failed on its actual first run despite passing every local
check, because `wearface`'s `resignWffApk` task needs `~/.android/debug.keystore` to already
exist and a dev machine has one from prior Android Studio use, but a fresh CI runner doesn't;
fixed by generating the same well-known debug keystore as an explicit CI step. **The primary
`origin` (rjv717/adai) repo's own runs have not actually executed yet** — GitHub reports "recent
account payments have failed or your spending limit needs to be increased" on that account,
unrelated to this workflow; needs that billing issue resolved before this CI job is actually
gating anything on the repo's primary remote (it is confirmed working on the kerfbit mirror in
the meantime).

Setting up this CI job also surfaced a real, already-shipped regression it now guards against:
`opsdashboard`'s test suite didn't even compile on `main` (`ModelRepositoryTest.kt` was calling
`clearStaleTrainingLock` with the pre-TD-147 one-argument signature) — fixed alongside the CI
work; see that commit for the full story.

**Update (September 12, 2026, later same day):** [`android-release.md`](android-release.md) (the
versioning/release convention doc) and [`android-release.yml`](../../../.github/workflows/android-release.yml)
(the tag-triggered workflow enforcing it) added together — `app-vX.Y.Z`/`opsdashboard-vX.Y.Z` tags
now validate `versionName`/`versionCode` against the tagged commit, re-run the full test suite, and
build+publish both a debug-signed and an unsigned release APK to a GitHub Release, failing loudly
before publishing anything if any check doesn't hold. Neither app has actually cut a real release
with it yet — both are still at `0.1.0`/`versionCode 1` — but every path through the mechanism was
verified against real GitHub Actions runs (on the kerfbit mirror, to avoid touching the primary
`origin` repo's release history with test artifacts): a malformed tag (`app-v1.0.0-beta`) rejected
before any other step ran, a version-mismatched tag (`app-v9.9.9` against the real `0.1.0` code)
rejected with the expected error, a stale-`versionCode` tag (a throwaway commit bumping only
`versionName`) rejected once a prior release existed to compare against, and a correct
`app-v0.1.0` tag matching the code exactly running the full happy path — real `assembleRelease`/
`assembleDebug` builds and a real GitHub Release with both APKs attached. All test tags, the
throwaway commit/branch, and the test release were deleted afterward; nothing from this
verification was left in place. Real production signing (a keystore + GitHub secrets) is
explicitly out of scope — see the doc's own "Signing status" section for why that's a deliberate,
one-way decision left to a human, not something to generate as a side effect of automating this.

Action Items:

- [x] Add a GitHub Actions workflow building both `:app` and `:opsdashboard` and running their
  unit tests (`./gradlew testDebugUnitTest`).
- [ ] Resolve the `origin` repo's GitHub billing/spending-limit issue so `android-ci.yml` actually
  runs there (confirmed working on the kerfbit mirror already).
- [x] Establish a real release/versioning process before either app leaves `0.1.0` — defined in
  `android-release.md` and mechanically enforced by `android-release.yml`.
- [ ] Actually cut the first real release of each app under this process (both are still at
  `0.1.0`/`versionCode 1`).
- [x] Decide on real release signing — deliberately deferred (September 12, 2026): both apps are
  currently side-loaded on the maintainer's own devices, not distributed further, so the
  debug-signed APK already produced is sufficient. `android-release.md`'s "Signing status"
  section records the concrete plan for if/when that changes (self-managed keystore vs. Play App
  Signing) so this doesn't need to be re-decided later.

Files to Modify:

- 41 files under `android/app/src/main` and `android/opsdashboard/src/main` /
  `android/wearsync/src/main` tagged `beta` — see [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)
  for the exact list.
- `.github/workflows/android-ci.yml` — done.
- `docs/development/guides/android-release.md`, `.github/workflows/android-release.yml` — done.

---

### TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open (11/12 ViewModels done — see correction below; Compose UI testing adopted, 10/12 screens covered; DI/entry-points remain) | Android / Testing | September 7, 2026 | 24-32 hours |

Description:
62 files — Compose screens, ViewModels without tests, DI containers, `Activity`/`Application`
entry points, and the `wearcomplications` services — genuinely have zero automated coverage, on
top of sharing TD-047's no-CI/no-release problem. `BiometricAdminAuthGate.kt` is a partial
exception: hard to unit-test (`BiometricPrompt` needs a real `FragmentActivity`), but it was read
manually during the rollout and appears complete — the gap there is coverage, not a known defect.

**Update (September 12, 2026):** all 8 ViewModels now have real test coverage —
`ChatViewModel`, `GroupListViewModel`, `SessionListViewModel`, `ModelListViewModel` (the four
originally tagged `experimental` here), plus `ModelDetailViewModel`, `AdminViewModel`,
`SessionDetailViewModel`, and `TrainerViewModel`. **The latter four turned out to be mistagged**
— each was already `beta`/"capped by TD-047" despite having zero dedicated tests (TD-047's own
scope is explicitly the repository/DTO layer, never ViewModels, and `beta` requires real test
coverage per the file-status standard). `AdminUiState.kt` was promoted alongside `AdminViewModel`
since its own documented per-daemon error-independence invariant is what got tested. Every test
was verified via the standard revert-confirm-fail cycle (a real behavior removed from the
ViewModel, the corresponding test confirmed to fail, then restored) — several of these are
regression tests for real historical bugs that had never had ViewModel-level coverage before
(e.g. `ModelDetailViewModelTest`'s `clearStaleTrainingLock` test is the first regression test for
TD-147's own fix). `AdminAuthGate.kt` was checked and confirmed to be correctly tagged as-is — a
pure interface/sealed-type declaration with no behavior of its own to test, unlike the four
mistagged ViewModels.

**Update (September 13, 2026):** Compose UI testing adopted — `ConversationListScreen` (`app`
module) is the first screen with real coverage, establishing the pattern for the remaining 11.
This repo had no `androidx.compose.ui.test` infrastructure at all, and the existing plain-JVM
`test`-source-set fakes (`FakeConversationDao`, `FakeApiClientProvider`, `FakeSettingsRepository`
— used by `ChatViewModelTest`/`ChatRepositoryTest`) live in a source set instrumented
(`androidTest`) tests can't see; rather than duplicate them, `android/app/src/sharedTest/java` was
added and wired into both the `test` and `androidTest` source sets (`build.gradle.kts`), and the
three fake files moved there — they depend only on plain Kotlin/coroutines and this module's own
domain interfaces, nothing Android-framework-specific, so nothing about moving them changes what
either source set can compile or run. `androidx.compose.ui:ui-test-junit4` (`androidTestImplementation`)
and `ui-test-manifest` (`debugImplementation`) added, both version-pinned via the existing
`compose-bom` platform already in use, no new version to track. `ConversationRow`'s delete
`IconButton` gained a `Modifier.testTag("delete_conversation_$id")` — every row otherwise shares
one `contentDescription` ("Delete conversation"), which a UI test can't disambiguate between rows
without depending on list order. 5 tests added
(`ConversationListScreenTest`): empty state, populated state (titles shown, placeholder hidden),
row click → `onOpenConversation(id)`, settings icon click → `onOpenSettings()`, delete click →
conversation removed from the rendered list, FAB click → new conversation created and opened.

**Update (September 13, 2026, continued):** `ChatScreen` (`app` module) done as the second screen
— along with it, `ChatInputBar`, `MessageBubble`, and `ErrorBanner` (three separate files it
composes) each got their own specific behavior genuinely exercised and were promoted
`experimental` → `beta` alongside it: `ChatInputBar`'s text-entry/send wiring, `MessageBubble`'s
FAILED-state rendering and retry click, and `ErrorBanner`'s message rendering and dismiss-action
callback. `TypingIndicator` (also composed by `ChatScreen`) was deliberately left `experimental`
— none of the 4 new tests ever observe `isSending == true` for long enough to assert on it,
since the fake chat service resolves synchronously with no artificial delay, so nothing here
actually exercises its own rendering. Along the way, found and fixed a real inaccuracy in
`ConversationListScreenTest`'s own doc comment from earlier the same day: it claimed
`FakeSettingsRepository()`'s default made `ConversationRepository.deleteConversation()` skip
`ApiClientProvider` entirely, but that fake's actual default constructor value is a *configured*
`ServerSettings(host = "localhost", ...)` (needed elsewhere for `ChatViewModelTest`'s real
fake-network path) — meaning the delete test had silently been attempting a real (harmlessly
failing, try/caught) network call to `localhost:8080` the whole time. Fixed by passing a blank
`ServerSettings()` explicitly, matching what the comment always claimed was happening.
4 tests added (`ChatScreenTest`): send a message → user bubble then assistant reply bubble
appear; back button → `onBack()`; a failed send → `ErrorBanner` shows the server's error message,
Dismiss hides it; retrying a pre-seeded `FAILED` message → resends and shows a new assistant
reply, `"Failed — tap to retry"` no longer rendered.

**Update (September 13, 2026, continued further):** `SettingsScreen` (`app` module) done as the
third screen, promoted `experimental` → `beta` alongside it. 5 tests added (`SettingsScreenTest`):
back button → `onBack()` without persisting anything; entering a host and clicking Save →
`SettingsViewModel.save()` is reached with the typed value and `onBack()` eventually fires;
toggling the HTTPS switch swaps the Port field for the two Cloudflare Access Client fields; a
blank-host "Test Connection" click shows the validation error without ever reaching the fake
server; a successful "Test Connection" shows the real "Connected — N active session(s)" message
from the fake `health()` response. Two real findings along the way:
- A genuine test-writing mistake, caught by its own revert-confirm-fail rather than shipped
  silently: the HTTPS-toggle test first tried clicking the `Switch`'s adjacent label `Text`, which
  has no click handler of its own (`Switch` and its label are separate composables in
  `SettingsScreen.kt`, not wrapped in one shared `clickable`) — the assertion failed for the right
  reason (the switch never toggled) but the wrong cause (a test bug, not a production one). Fixed
  by targeting `isToggleable()` instead of the label text.
- The Save-button test was originally written to assert TD-130's happens-before ordering (`save()`
  landing before `onBack()` fires) the same way `ChatScreenTest`'s tests assert ordering-sensitive
  outcomes — but revert-confirm-fail disproved that this actually works here: temporarily restoring
  TD-130's original fire-and-forget shape in `SettingsScreen.kt` did *not* make the assertion fail,
  because `performClick()` waits for the real device's Main dispatcher to reach idle before
  returning, and by then every scheduled coroutine on it — sequenced or not — has typically already
  run. A real-device instrumented test genuinely cannot distinguish "sequenced" from
  "fire-and-forget-but-fast" the way `SettingsViewModelTest`'s own `StandardTestDispatcher`
  -controlled plain-JVM test already does; only that finer-grained test actually guards TD-130's
  ordering invariant. Reworded the UI test's assertion and doc comment to claim only what it can
  actually verify (the wiring reaches `save()` with the right value, and `onBack()` eventually
  fires) — confirmed to still fail for a real reason via a second revert-confirm-fail pass
  (removing the `save()` call entirely).

**Update (September 13, 2026, continued still further):** `GroupListScreen` (`opsdashboard`
module) done as the fourth screen — the first in `opsdashboard`, which needed its own
`androidx.compose.ui.test`/`ui-test-manifest` dependencies and `src/sharedTest` source set wired
up first (identical setup to `:app`'s own, done earlier the same day); its 6 fake test-utility
files moved there and newly tagged (`beta`, not previously in-scope for the file-status standard
under `src/test`). 5 tests added (`GroupListScreenTest`): no configured groups shows the
placeholder message; multiple configured groups each show their own row with the right pending
count; one group's fetch failing shows its own error without hiding or corrupting the others'
results (mirrors `GroupListViewModelTest`'s own TD-128-style isolation test at the UI level);
clicking a row invokes `onOpenGroup(name)`; clicking the settings icon invokes `onOpenSettings()`.

**Known limitation, disclosed rather than worked around:** `opsdashboard`'s debug APK could not
be installed on this sandbox's phone emulator to run these tests live —
`INSTALL_FAILED_MISSING_SHARED_LIBRARY`, because its merged manifest declares
`<uses-library android:name="wear-sdk" android:required="true"/>` (auto-injected by its Wear
watch-face-push dependencies), and no Wear-capable emulator/device was available here even with
the `google_apis_playstore` system image. Per user decision, proceeded compile-verified only:
`GroupListScreenTest` compiles clean and was cross-checked carefully against the real production
code path by hand (`SafeCall.kt`/`ApiResult.kt`'s exact error-message strings, confirmed to
produce `"Error: connection refused"` verbatim for an `IOException("connection refused")`;
`ConversationListScreenTest`'s already-device-verified `ListItem` + `Modifier.clickable` and
`IconButton` + `contentDescription` interaction patterns, structurally identical here) rather than
run end-to-end. `GroupListScreen.kt` promoted `experimental` → `beta` with an explicit
real-device-still-unverified caveat in its own status comment, matching `GPUUtils.hpp`'s TD-041
precedent for hardware-gated code in this same tracker. Whoever next has access to a Wear-capable
emulator or real device should run `opsdashboard`'s `connectedDebugAndroidTest` for real at least
once to close this gap — nothing here is expected to fail, but it hasn't actually been observed
passing on a device the way the three `app`-module screens have.

**Update (September 13, 2026, yet further still):** `ModelListScreen` (`opsdashboard` module) done
as the fifth screen, same compile-verified-only caveat as `GroupListScreen` above (this sandbox
still can't install/run `opsdashboard` on a device). 5 tests added (`ModelListScreenTest`): no
models registered shows the placeholder message; multiple models each show their own row with
role (including the blank-role → `"(none)"` fallback) and state; a fetch failing with no prior
models shows `FullScreenError`; clicking a row invokes `onOpenModel(name)`; clicking the settings
icon invokes `onOpenSettings()`. Deliberately did not attempt `ModelListViewModelTest`'s own
"a failed *second* poll keeps the first poll's models" scenario at the UI level — that test needs
virtual time control (`advanceTimeBy` past the 8s poll interval) a real-device instrumented test
can't do without an actual ~9s real-time wait, and the existing ViewModel-level test already
covers it with precise control; duplicating it here for a screen-level test would cost real
wall-clock time in the suite for coverage that already exists. `ModelListScreen.kt` promoted
`experimental` → `beta` with the same real-device-still-unverified caveat.

**Update (September 13, 2026, and further still):** `SessionListScreen` (`opsdashboard` module)
done as the sixth screen, same compile-verified-only caveat. 7 tests added
(`SessionListScreenTest`): no sessions shows the placeholder message; idle-only sessions (hidden
by default) show the correct hidden-count message; clicking "Show idle" reveals the idle rows and
flips the button label to "Hide idle"; a mix of training/idle sessions shows only the training row
by default, with the exact `"Epoch N/M · loss X.XXXX"` formatting and `StatusBadge` label; a fetch
failing with no prior sessions shows `FullScreenError`; clicking a row invokes
`onOpenSession(key)`; clicking the settings icon invokes `onOpenSettings()`. The idle-session
hide/show filtering (`visibleSessions`/`hiddenIdleCount`) lives entirely in the composable, not
the ViewModel — only the `showIdle` flag flip itself is covered by `SessionListViewModelTest` —
so this is genuinely new coverage, not a duplicate of existing tests. `SessionListScreen.kt`
promoted `experimental` → `beta` with the same real-device-still-unverified caveat.

**Update (September 13, 2026, continuing on):** `ModelDetailScreen` (`opsdashboard` module) done
as the seventh screen. 9 tests added (`ModelDetailScreenTest`): all detail fields render correctly
(architecture, artifact, identity fields); blank optional fields (role, run ID, artifact
host/path) fall back to their documented placeholder text (`"(none)"`/`"(local)"`); each training
history entry renders with the right formatting; the "Training History" section is hidden
entirely when there's none; the three admin-action buttons' enabled/disabled state correctly
follows `model.state` (training → only "Clear lock" enabled; candidate → "Retire"/"Promote"
enabled; production → all three disabled); a fetch failing with no prior model shows
`FullScreenError`; the back button invokes `onBack()`.

**Deliberately out of scope for this pass:** the admin-action confirm-dialog flow itself (clicking
"Clear stale training lock"/"Retire candidate"/"Promote to production" and confirming) is not
tested. `ConfirmActionDialog` unconditionally does `LocalContext.current as FragmentActivity` and
reads `LocalAdminAuthGate.current` (a `staticCompositionLocalOf` that throws if never provided)
the moment it composes — true the instant the dialog opens, before its own confirm button is ever
clicked. A plain `createComposeRule()`'s default test host activity is a bare `ComponentActivity`,
not a `FragmentActivity` (only `MainActivity` itself is), so that cast would crash immediately;
there is also no fake `AdminAuthGate` implementation anywhere in this codebase yet. Testing this
flow for real needs a small, genuinely new piece of test infrastructure — a minimal
`androidTest`-only `FragmentActivity` host used via `createAndroidComposeRule<...>()`, plus a
`FakeAdminAuthGate` — that wasn't built speculatively here since this sandbox can't run any of it
live anyway to confirm it's wired correctly (see the `wear-sdk` limitation below). Flagged as a
concrete follow-up rather than left implicit: whoever has real device access and picks this up
should build that infrastructure once (it will be needed again for `AdminScreen` and
`GroupDetailScreen`'s own admin actions) rather than per-screen. `ModelDetailScreen.kt` promoted
`experimental` → `beta` anyway — the read-only majority of the screen has real coverage, and
`beta`'s own definition ("known gaps exist") fits a disclosed, scoped gap like this one — with
both this gap and the real-device-still-unverified caveat noted in its own status comment.

**Update (September 13, 2026, continuing on still further):** `SessionDetailScreen` (`opsdashboard`
module) done as the eighth screen. 8 tests added (`SessionDetailScreenTest`): a live session renders
its poller-status label, progress fields, and training metrics; a stale/not-effectively-training
session shows the "no — stale" reason; generation-quality metrics show their disabled message when
`current_bleu4 < 0` and show BLEU/ROUGE values otherwise; a fetch failing with no prior status shows
`FullScreenError`; a 404 on the very first poll correctly invokes `onEvicted()`; the "End session"
admin button renders enabled by default; the back button invokes `onBack()`. Notable finding:
`SessionDetailViewModel` uses `AdaptivePoller` (not the simpler `FixedIntervalPoller` the list
screens use), but `AdaptivePoller.run()` also calls `poll()` immediately before any delay, and a 404
response drives it straight to `PollerPhase.EVICTED`, which short-circuits the loop before it ever
reaches its `delay()` call — so the eviction scenario is deterministically observable via
`waitUntil()` on a real device just like the list screens' first-poll scenarios, with no virtual-time
control needed. Deliberately out of scope for this pass, for the same reason and with the same
infrastructure gap as `ModelDetailScreen` above: the "End session" confirm-dialog flow itself
(`ConfirmActionDialog`'s `FragmentActivity`/`LocalAdminAuthGate` requirement) — only the button's
default-enabled rendering is checked, not the click. `SessionDetailScreen.kt` promoted
`experimental` → `beta`, same rationale as `ModelDetailScreen`.

**Update (September 13, 2026, still continuing on further):** `SettingsScreen` (`opsdashboard`
module) done as the ninth screen. Unlike every screen so far, there was no pre-existing
`SettingsViewModelTest` to reuse a Fixture shape from (opsdashboard's `SettingsViewModel` had zero
coverage at any level) — `SettingsScreenTest`'s own Fixture was built directly against
`SettingsViewModel`'s real constructor (`FakeSettingsRepository` + a real `WatchFacePushRepository`).
9 tests added: the default shared-host layout (per-service Host fields hidden, Port fields shown);
toggling "one host for all services" swaps that; toggling the HTTPS-relay switch shows the
Cloudflare/Trainer Access Client fields and hides all Port fields; adding and removing a registry
group chip; selecting a poll-interval chip and Save persisting it and firing `onBack()`; toggling
watch-sync hides/shows the session-key-override field; Back without Save persists nothing; the
watch-face section's "Install / update" button renders enabled by default. Three `Switch`
composables share this screen (unlike `SettingsScreen` in the `app` module, which has only one), so
each gained its own `testTag` (`switch_use_shared_host`/`switch_use_https_relay`/
`switch_watch_sync_enabled`) — a no-op production change, same precedent as `ConversationListScreen`'s
delete-button `testTag`. `WatchFacePushRepository` is a concrete class wrapping
`androidx.wear.watchfacepush` (a real Wear system service), not an interface — no fake exists for it,
and building one would mean reshaping production code for a screen-test task alone, which this pass
doesn't do; it's constructed here with the instrumentation's real target `Context` (safe — the
constructor only stores the `Context`, touching no Wear API until `pushWatchFace()`/`isSupported()`
are actually called). Deliberately out of scope for this pass: clicking "Install / update" or
"Activate" — both call into `WatchFacePushManagerFactory`, a real AndroidX Wear library this sandbox
has no paired-watch/Wear-capable emulator to exercise (the same `wear-sdk` limitation already
blocking a live device run of this whole module). Only the button's default rendering is checked.
`SettingsScreen.kt` promoted `experimental` → `beta` (0.3.0) — the gap is disclosed in its own status
comment and here.

**Correction (September 13, 2026):** this entry's "8/8 ViewModels done" claim (Action Items below)
was stale/wrong — enumerating every `*ViewModel.kt` under both modules' `src/main` against every
`*ViewModelTest.kt` under the matching `src/test` found two real gaps it had missed: opsdashboard's
`SettingsViewModel` (fixed by this update, see below) and `app`'s `ConversationListViewModel`
(still open — flagged separately). There are 12 ViewModels total (3 `app`, 9 `opsdashboard`); the
true count as of this correction is 11/12.

`SettingsViewModel` (`opsdashboard` module) now has `SettingsViewModelTest` (15 tests): initial
state loads every field from the repository correctly (including the `watchSyncSessionKeyOverride
== null` → `""` mapping); every field-change handler updates its own state field independently;
`addGroup` trims the input, ignores blank/whitespace-only input, and ignores a name already in the
list; `removeGroup` removes only the matching entry; `save()` trims string fields, falls back to
each service's default port on an unparseable port string, and persists a blank/whitespace-only
session-key override as `null` rather than as a literal blank string (verified via
revert-confirm-fail — temporarily un-trimming that field before the `takeIf` made the test fail for
the right reason); and `pushWatchFace()`/`activateWatchFace()`'s success/validation-failure/
failure/no-slot-id branches, including a real fake instead of a mock.

That last group needed `WatchFacePushRepository` split into an interface (production change) plus
`WearWatchFacePushRepository` (the real `androidx.wear.watchfacepush`-backed implementation, wired
into `AppContainer` exactly where the old concrete class was) and a new
`FakeWatchFacePushRepository` (shared `src/sharedTest` fake) — the same interface-plus-fake shape
already used for `AdminAuthGate`/`BiometricAdminAuthGate` in this same module. This also let
`SettingsScreenTest` (from the prior update) drop its awkward real-`Context`-backed
`WatchFacePushRepository` construction in favor of the new fake, and gained three tests actually
exercising the "Install / update" button's success/validation-failure/failure paths — a
previously-disclosed gap this closes at the ViewModel-orchestration level. Still out of scope:
clicking "Activate" itself, which first checks/requests a runtime permission via
`rememberLauncherForActivityResult` — real system-permission-dialog interaction needs a real
device, not a fake repository. `SettingsScreen.kt`'s status comment updated to reflect this.

**Update (September 13, 2026, on and on):** `GroupDetailScreen` (`opsdashboard` module) done as the
tenth screen. Reused `GroupDetailViewModelTest`'s own real-repository Fixture shape
(`RegistryRepository`/`ModelRepository` backed by the shared `src/sharedTest` fakes).
`FakeRegistryApiService` gained a configurable `runsResponse` (it was previously hardcoded to
always return an empty `RunsResponseDto`, so the "Claimed by run" section could never be exercised
by any test). 10 tests added (`GroupDetailScreenTest`): the empty-state placeholders for all three
sections; a populated queue/runs/registry all rendering correctly together, including the exact
formatted row text (entry count, human file size, source, formatted timestamp); TD-128's
"Last refresh failed" banner on a `registry()` failure; the "Release" button's default-enabled
rendering; and — unlike `ModelDetailScreen`/`SessionDetailScreen` — the Gutenberg-fetch,
HuggingFace-fetch, and Assign-model dialogs' full click-through flows, each verified end-to-end
against the fake's captured call arguments (including their own input-validation-gated confirm
buttons, e.g. the Gutenberg dialog's Fetch button staying disabled until a positive book ID is
entered). Those three dialogs are plain `AlertDialog`s with no `ConfirmActionDialog` gate, unlike
this same screen's "Force release" action, which does use `ConfirmActionDialog` and hits the same
`FragmentActivity`/`LocalAdminAuthGate` infrastructure gap already flagged for
`ModelDetailScreen`/`SessionDetailScreen` — only that button's default-enabled rendering is checked,
not the click. `GroupDetailScreen.kt` promoted `experimental` → `beta` (0.3.0).

Action Items:

- [x] Add ViewModel unit tests first (cheapest — no Compose/Activity needed) — 11 of 12 done (see
  the Correction above); `ConversationListViewModel` (`app` module) still has zero test coverage,
  flagged separately. See the per-ViewModel test files under `android/app/src/test`/
  `android/opsdashboard/src/test`.
- [x] Adopt Compose UI testing (`androidx.compose.ui.test`) for screens once ViewModels are
  covered — infrastructure adopted for both modules now; `ConversationListScreen`, `ChatScreen`,
  `SettingsScreen` (`app` module), `GroupListScreen`, `ModelListScreen`, `SessionListScreen`,
  `ModelDetailScreen`, `SessionDetailScreen`, `SettingsScreen`, and `GroupDetailScreen`
  (`opsdashboard` module) done (10 of 12; see updates above). The other 2 remain, both in
  `opsdashboard` (`AdminScreen`, `TrainerScreen`) — same pattern, infra already in place. Note:
  this sandbox cannot install/run `opsdashboard`'s debug APK on a device (see `GroupListScreen`'s
  update above, `wear-sdk` shared-library requirement) — further `opsdashboard` screens will be
  compile-verified only here too, pending real Wear-capable device access to actually run any of them.
- [ ] Build the admin-action confirm-dialog testing infrastructure (a minimal `androidTest`-only
  `FragmentActivity` host + `createAndroidComposeRule<...>()` + a `FakeAdminAuthGate`) — see
  `ModelDetailScreen`'s update above for the full explanation of why it's needed and doesn't exist
  yet. `AdminScreen` and `GroupDetailScreen` will need it too, so build it once, not per-screen.
- [ ] `BiometricAdminAuthGate.kt` specifically: consider an instrumented test using
  `BiometricPrompt`'s test/fake authenticator support instead of leaving it permanently untested.
- [ ] DI containers (`AppContainer.kt`/`AppViewModelProvider.kt` in both apps), `Activity`/
  `Application` entry points, and the `wearcomplications` services remain entirely untested —
  not attempted in this pass.

Files to Modify:

- ~40 remaining files under `android/app/src/main`, `android/opsdashboard/src/main`, and
  `android/wearcomplications/src/main` still tagged `experimental` — see
  [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) for the exact, current list.
- Done: the 8 ViewModel files, `AdminUiState.kt`, `ConversationListScreen.kt`, `ChatScreen.kt`,
  `ChatInputBar.kt`, `MessageBubble.kt`, `ErrorBanner.kt`, `SettingsScreen.kt` (`app` module), and
  `GroupListScreen.kt`/`ModelListScreen.kt`/`SessionListScreen.kt`/`ModelDetailScreen.kt`/
  `SessionDetailScreen.kt`/`SettingsScreen.kt`/`GroupDetailScreen.kt` (`opsdashboard` module —
  `ModelDetailScreen.kt`/`SessionDetailScreen.kt`/`GroupDetailScreen.kt` with their admin-action
  confirm-dialog flows still uncovered, `SettingsScreen.kt` with its watch-face-push
  Activate-click still uncovered — see the
  Correction above).

---

### TD-164: chatbot-guide.md Needs a Live-Pair Verification Pass

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Documentation | September 13, 2026 | 2-4 hours |

Description:
Split off from [TD-053](../archive/TECHNICAL_DEBT_RESOLVED.md#td-053-chatbotclis-save-and-load-commands-are-non-functional-everywhere)
when that item was resolved. TD-053 fixed `chatbot-guide.md` sections verifiable by reading source
directly: the Quick Start banner, File Requirements, Default File Paths, "Starting the Chatbot",
and Command-Line Help sections (all corrected September 8, 2026, when the doc was found to
describe an entire earlier CLI architecture — a standalone binary taking `[vocab_file]
[model_file] [conversation_save_file]` — that predates `ChatbotCLI` becoming a thin HTTP client
for `chatbot_api_server`), plus the Conversation History / `/save` / `/load` sections and the
`vocab.txt`/`model.bin` example invocations and tokenizer-loading troubleshooting entry (all
corrected September 13, 2026, once `/save`/`/load`/auto-save-on-exit became real). What's left is
a genuinely interactive check — Commands Reference details, Generation Strategies, and
Configuration Parameters — none of which can be confidently verified by reading source alone the
way the sections above were.

Action Items:

- [ ] Start a real `chatbot_api_server` + `chatbot` pair and walk every example in the Commands
  Reference, Generation Strategies, and Configuration Parameters sections, fixing any output that
  no longer matches.
- [ ] Remove the "targeted correction, not a full re-verification" caveat from the top-of-file
  banner once done.

Files to Modify:

- `docs/operations/guides/chatbot-guide.md`

---

### TD-163: AttentionHeadBenchmark Hangs Indefinitely

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Benchmarks / Tooling | September 13, 2026 | Not yet estimated — root cause unknown |

Description:
`benchmarks/AttentionHeadBenchmark.cpp` hangs indefinitely when run. Confirmed via
`timeout 20 ./attention_head_benchmark` (built from the `debug` preset's `attention_head_benchmark`
target): output stops right after printing the "Benchmark: Scaling with Number of Attention Heads"
table header — it never prints even the first row (`num_heads=2`, `seq_len=128`, `d_model=512`, 3
warmup pairs + 50 timed iterations of `MultiHeadAttention::forward_parallel()`).

Found while verifying TD-059's attention-head-splitting fix — confirmed via `git stash` to be
**pre-existing**, reproducing identically against the original (pre-TD-059-fix) `MultiHeadAttention.cpp`
as well as the fixed version, so it is not a regression from that change. Not part of the `ctest`
suite (`attention_head_benchmark` is a standalone binary, not a registered test), so it isn't
gating anything — found incidentally while smoke-testing the benchmark still built and ran cleanly
after TD-059's fix.

Action Items:

- [ ] Root-cause the hang. Candidates worth checking first: an OpenMP interaction specific to
  sandboxed/containerized environments (`OMP_NUM_THREADS`, nested-parallelism settings, thread pool
  exhaustion under a CPU quota/cgroup), the `Timer` class used for the benchmark's own timing (check
  for a busy-wait or blocking call), or something in `PerformanceProfiler.hpp` (also included by
  this benchmark) triggering a deadlock on include or static initialization.
- [ ] Use a fast, disposable repro harness (a minimal standalone program calling just
  `MultiHeadAttention::forward_parallel()` a handful of times with the same config —
  `d_model=512`, `seq_len=128`, `num_heads=2`) rather than re-running the full benchmark repeatedly,
  to localize the exact call that hangs.
- [ ] Fix it if the root cause is a real bug; if it turns out to be inherent to this specific
  sandboxed environment (e.g. no real multi-core CPU affinity, a container cgroup/CPU-quota
  interaction with `libgomp`), document that plainly in the benchmark's own comments rather than
  leaving it silently broken.

Files to Modify:

- `benchmarks/AttentionHeadBenchmark.cpp`
- Possibly `src/PerformanceProfiler.hpp` or wherever `Timer` is defined, depending on root cause

---

## Resolved Items

156 items resolved. See [archive/TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md) for full details.

---
## Future Improvements

These are lower-priority enhancements that don't currently block development:

### Performance Optimizations

1. **Memory Pool for Matrix Allocations**
   - Reduce allocation overhead during forward/backward passes
   - Pre-allocate memory for common matrix sizes
   - Reduce memory fragmentation

2. **Monitor Augmentation Performance** (from Priority 2 completion)
   - **Priority:** Low
   - **Effort:** Ongoing
   - **Description:** Track real-world speedup from the parallel data augmentation implementation (Priority 2) across different hardware configurations and dataset sizes.
   - **Implementation:**
     - Collect augmentation throughput metrics during training runs
     - Compare 1-thread vs multi-thread performance in production workloads
     - Record results in benchmark log for regression detection
   - **Files:** `src/EfficientBatching.hpp`, `benchmarks/AugmentationBenchmark.cpp`
   - **Reference:** `docs/development/guides/AUGMENTATION_CHECKLIST.md`

3. **Batched Inference Engine (Priority 3)**
   - **Priority:** Medium
   - **Effort:** Medium (estimated 2-4 days)
   - **Description:** Implement a batched inference engine to process multiple inference requests simultaneously, achieving 10-20x throughput improvement for serving/production workloads.
   - **Expected Impact:** 10-20x throughput improvement
   - **Implementation:**
     - Group incoming inference requests into dynamic batches
     - Process batches through the model in a single forward pass
     - Return individual results to each caller
     - Tune batch size and timeout for latency vs throughput trade-off
   - **Benefits:**
     - Dramatically higher request throughput
     - Better GPU/CPU utilization
     - Lower per-request cost at scale
   - **Files:** `src/BatchedInferenceEngine.hpp`, `src/ChatbotAPIServer.cpp`, `benchmarks/BatchedInferenceBenchmark.cpp`
   - **Reference:** `docs/development/guides/AUGMENTATION_CHECKLIST.md`, `docs/development/archive/BATCHED_INFERENCE_SUMMARY.md`

4. **Attention Head Parallelism (Priority 4)**
   - **Priority:** Medium
   - **Effort:** Medium (estimated 2-3 days)
   - **Description:** Parallelize multi-head attention computation so each attention head is processed concurrently, targeting a 2-4x speedup in the attention layer.
   - **Expected Impact:** 2-4x speedup in attention computations
   - **Implementation:**
     - Distribute attention heads across OpenMP threads
     - Ensure thread-safe accumulation of outputs
     - Validate correctness against single-threaded reference
   - **Benefits:**
     - Faster inference and training for transformer-based models
     - Better utilization of multi-core CPUs
   - **Files:** `src/MultiHeadAttention.hpp`, `src/MultiHeadAttention.cpp`, `benchmarks/AttentionHeadBenchmark.cpp`
   - **Reference:** `docs/development/guides/AUGMENTATION_CHECKLIST.md`, `docs/development/archive/ATTENTION_HEAD_PARALLELISM_SUMMARY.md`

### Code Quality

1. **Add Benchmarking Suite**
   - Performance regression testing
   - Track training throughput over time
   - Compare against baseline implementations

### Developer Experience

1. **Add Python Bindings**
   - Enable easier experimentation
   - Broader community adoption
   - Integration with Python ML ecosystem

2. **Improve Build Times**
   - Use precompiled headers
   - Optimize template instantiations
   - Modularize includes

### Configuration and Service Management

Related to Steps 1-5: Daemon Service Implementation

1. **JSON Configuration Format Support** (Step 1 Enhancement)
   - **Priority:** Low
   - **Effort:** 2-3 hours
   - **Description:** Support JSON in addition to key=value format
   - **Implementation:**
     - Add JSON parsing library (nlohmann/json or similar)
     - Implement `load_from_json()` method in ConfigLoader
     - Support both formats with auto-detection
     - Add schema validation for JSON config
   - **Benefits:**
     - More expressive configuration (nested objects, arrays)
     - Better tooling support (editors, validators)
     - Easier integration with deployment tools
   - **Files:** `src/Config.cpp`, `src/Config.hpp`, `CMakeLists.txt`

2. **Configuration Profiles** (Step 1 Enhancement)
   - **Priority:** Low
   - **Effort:** 3-4 hours
   - **Description:** Support named configuration profiles (dev, staging, prod)
   - **Implementation:**
     - Add `--profile` command-line argument
     - Load base config + profile-specific overrides
     - Support profile inheritance
   - **Files:** `src/Config.cpp`, `config.dev.conf`, `config.prod.conf`

3. **Model State Persistence on Shutdown** (Step 2 Enhancement)
   - **Priority:** Medium
   - **Effort:** 4-6 hours
   - **Description:** Automatically save model weights during graceful shutdown
   - **Implementation:**
     - Check if MODEL_PATH is configured during shutdown
     - If set, call model->save_weights() in shutdown sequence
     - Add checkpoint metadata (timestamp, loss, etc.)
     - Log save progress with structured logging
   - **Benefits:**
     - No manual model saving needed
     - Automatic checkpointing on restart
     - Reduced risk of losing trained state
   - **Files:** `src/ChatbotAPIServer.cpp`

4. **Graceful Reload (Zero-Downtime Restart)** (Step 2 Enhancement)
   - **Priority:** Low
   - **Effort:** 8-12 hours
   - **Description:** Reload model without dropping connections
   - **Implementation:**
     - Implement SIGHUP handler for reload signal
     - Load new model in background thread
     - Atomic swap to new model after loading
     - Keep serving requests during reload
   - **Benefits:**
     - True zero-downtime deployments
     - Seamless model updates
   - **Files:** `src/ChatbotAPIServer.cpp`, `src/ChatbotAPI.cpp`

### Logging and Observability

Related to Step 3: Structured Logging

1. **JSON Log Output Format** (Step 3 Enhancement)
   - **Priority:** Medium
   - **Effort:** 2-3 hours
   - **Description:** Support JSON-formatted logs for machine parsing
   - **Implementation:**
     - Add `LOG_FORMAT` config option (text/json)
     - Implement JSON formatter in Logger class
     - Include structured fields: timestamp, level, message, component, context
     - Support ECS (Elastic Common Schema) format
   - **Benefits:**
     - Better integration with log aggregation (ELK, Splunk)
     - Easier parsing and analysis
     - Structured error reporting
   - **Example Output:**

     ```json
     {"timestamp":"2026-03-01T16:15:17.862Z","level":"info","message":"Server started","port":8080,"pid":1234}
     ```

   - **Files:** `src/Logger.cpp`, `src/Logger.hpp`, `src/Config.hpp`

2. **Per-Module Log Levels** (Step 3 Enhancement)
    - **Priority:** Low
    - **Effort:** 4-5 hours
    - **Description:** Different log levels for different components
    - **Implementation:**
      - Create named loggers per component (api, model, tokenizer)
      - Support per-module configuration: `LOG_LEVEL_API=DEBUG`
      - Add logger registry for runtime level changes
    - **Benefits:**
      - Fine-grained debugging control
      - Reduce log noise in production
      - Debug specific components without full verbosity
    - **Example:**

      ```ini
      LOG_LEVEL_API=DEBUG
      LOG_LEVEL_MODEL=INFO
      LOG_LEVEL_TOKENIZER=WARN
      ```

    - **Files:** `src/Logger.cpp`, `src/Logger.hpp`, `src/Config.cpp`

3. **Custom Log Sinks** (Step 3 Enhancement)
    - **Priority:** Low
    - **Effort:** 6-8 hours
    - **Description:** Support multiple log destinations
    - **Implementation:**
      - Add syslog sink for system logging
      - Add network sink (TCP/UDP) for centralized logging
      - Add custom sink interface for extensibility
      - Configure via `LOG_SINKS` config option
    - **Files:** `src/Logger.cpp`, `src/sinks/SyslogSink.cpp`, `src/sinks/NetworkSink.cpp`

### Container and Deployment

Related to Steps 4-5: Docker and systemd

1. **Kubernetes Deployment Manifests** (Step 4 Enhancement)
    - **Priority:** Low
    - **Effort:** 4-6 hours
    - **Description:** Create production-ready Kubernetes manifests
    - **Implementation:**
      - Create Deployment manifest with resource limits
      - Add Service manifest for load balancing
      - Add ConfigMap for configuration
      - Add HorizontalPodAutoscaler for auto-scaling
      - Add probes (liveness, readiness, startup)
    - **Files:** `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/configmap.yaml`, `docs/operations/KUBERNETES_DEPLOYMENT.md`

2. **Metrics Endpoint for Prometheus** (Step 5 Enhancement)
    - **Priority:** Medium
    - **Effort:** 6-8 hours
    - **Description:** Expose /metrics endpoint with application metrics
    - **Implementation:**
      - Add Prometheus C++ client library
      - Expose metrics: request_count, request_duration, active_sessions, model_inference_time
      - Add custom metrics: generation_tokens_total, vocabulary_size, etc.
      - Integrate with systemd service monitoring
    - **Benefits:**
      - Production monitoring and alerting
      - Performance tracking over time
      - Integration with Grafana dashboards
    - **Files:** `src/MetricsExporter.cpp`, `src/ChatbotAPIServer.cpp`, `CMakeLists.txt`

3. **systemd Socket Activation** (Step 5 Enhancement)
    - **Priority:** Low
    - **Effort:** 5-7 hours
    - **Description:** Support on-demand service activation
    - **Implementation:**
      - Add socket activation support with sd_listen_fds()
      - Create adai.socket unit file
      - Modify server to accept pre-bound socket from systemd
      - Support both direct and socket-activated startup
    - **Benefits:**
      - Reduced resource usage for idle services
      - Faster first-request handling (systemd pre-binds socket)
      - Better system integration
    - **Files:** `src/ChatbotAPIServer.cpp`, `scripts/adai.socket`, `docs/operations/SYSTEMD_DEPLOYMENT.md`

4. **Multi-Instance Service Templates** (Step 5 Enhancement)
    - **Priority:** Low
    - **Effort:** 2-3 hours
    - **Description:** Support running multiple chatbot instances
    - **Implementation:**
      - Convert `adai.service` to template unit (`adai@.service`)
      - Use %i instance identifier for port assignment
      - Support per-instance configuration files
    - **Example Usage:**

      ```bash
      systemctl start adai@8080
      systemctl start adai@8081
      systemctl start adai@8082
      ```

    - **Files:** `scripts/adai@.service`, `docs/operations/SYSTEMD_DEPLOYMENT.md`

5. **Health Check Enhancements** (Steps 4-5 Enhancement)
    - **Priority:** Low
    - **Effort:** 3-4 hours
    - **Description:** More comprehensive health checking
    - **Implementation:**
      - Add detailed health endpoint returning component status
      - Check vocabulary loaded, model initialized, memory usage
      - Support readiness vs liveness checks (Kubernetes)
      - Add configurable health check timeout
    - **Response Example:**

      ```json
      {
        "status": "healthy",
        "components": {
          "tokenizer": {"status": "ok", "vocab_size": 9925},
          "model": {"status": "ok", "initialized": true},
          "memory": {"status": "ok", "usage_mb": 512}
        },
        "uptime_seconds": 3600
      }
      ```

    - **Files:** `src/ChatbotAPIServer.cpp`, `src/ChatbotAPI.cpp`

---

## Deferred Decisions

Architecture decisions that were considered and explicitly not pursued, with the reasoning and the
condition that would reopen them. Unlike Future Improvements above, these are not queued work —
they're a record of "not now, and here's why," so the reasoning isn't re-litigated from scratch later.

### AMD Radeon / ROCm-HIP GPU Backend — Not Pursued

**Date:** September 7, 2026
**Component:** GPU / Architecture

**Decision:** Do not add a third GPU backend (ROCm/HIP) alongside the existing CUDA
(`src/gpu/MatrixGPU.cu`) and SYCL (`src/gpu/sycl/`) paths.

**Reasoning:**

- No AMD Radeon hardware is available or targeted for this project — a HIP backend would be
  maintained by inspection only, the same way `MatrixGPU.cu` already is on the primary dev
  machine (whose only GPU is an integrated Intel UHD 620 — neither CUDA nor a discrete Intel ARC
  device is actually present there either).
- The two existing backends already demonstrate the sync cost: CUDA and SYCL were edited in the
  same commit, on the same day, and still drifted (see the September 2026 fix that added float4
  vectorization, sub-group/warp-shuffle reduction, and `GPU_STRATEGY` queue-priority parity to the
  SYCL backend to catch it up with CUDA). A third backend multiplies that sync surface rather than
  adding to it linearly.
- Unlike SYCL (a ground-up rewrite in a different kernel-lambda paradigm), HIP is close to
  source-compatible with CUDA — AMD's hipify tools can mechanically translate most of
  `MatrixGPU.cu` (`cuda*` → `hip*`, `cublas` → `hipblas`/`rocblas`) — so adding it later is
  expected to cost meaningfully less than SYCL did.

**Revisit when:** Either (a) AMD Radeon/ROCm hardware becomes available to build and test
against, or (b) a concrete deployment target requires it. The CMake mutual-exclusion pattern
(`ENABLE_GPU`/`ENABLE_SYCL`, see `CMakeLists.txt`) already extends cleanly to a third
`ENABLE_HIP` option if/when that happens.

---

### ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug

**Date:** September 12, 2026
**Component:** Testing / Sanitizers / Core Matrix Ops

**Decision:** Do not modify `src/Matrix.cpp`'s `#pragma omp parallel for` regions (`operator*()`,
`transpose()`, `scale()`, and the other OpenMP-parallelized methods) in response to the ThreadSanitizer
"data race" warnings they produce under the `tsan` CMake preset — running `chatbotapiTests` (or any
other suite exercising real matrix multiplication) reports roughly 20-90+ of them depending on how
much multiplication the run does. Originally surfaced as a side observation while verifying TD-156's
concurrency fix, flagged for follow-up, and investigated here to a definitive conclusion rather than
left open.

**Reasoning:**

- Confirmed via `git stash` (independently, across two separate investigations) that these warnings
  are pre-existing on pristine `main`, unrelated to any specific change — identical counts with or
  without, e.g., TD-156's mutex.
- Read the full race-report bodies (not just the one-line summaries `TSAN_OPTIONS` prints by
  default) with `TSAN_OPTIONS=history_size=7` for deeper history. Every one follows the same shape:
  a worker thread spawned by libgomp (visible in the stack traces via `pthread_create` inside
  `libgomp.so.1`) reads/writes a `Matrix`'s backing `std::vector<std::vector<float>>` inside a
  `#pragma omp parallel for` region, and the "previous access" it's reported racing against is the
  *calling* thread's own move-assignment/`delete`/further construction of that same `Matrix`
  immediately after the parallel region returns — exactly the access pattern OpenMP's **mandatory
  implicit barrier** at the end of a `parallel for` (no `nowait` is used anywhere in this file)
  guarantees is safe.
- Manual review of every flagged call site confirms genuinely disjoint memory per loop iteration
  (distinct output rows/elements per index; each `Matrix`'s backing buffers are already fully
  allocated, single-threaded, before its parallel region starts — see `Matrix::Matrix(int, int)`'s
  `data.resize(...)`) — there is no aliasing bug to fix in this file's own logic.
- This is a known, documented ThreadSanitizer + GCC-`libgomp` limitation, not specific to this
  codebase: a ThreadSanitizer co-author has stated on the GCC mailing list that TSan reliably
  supports "pthread-based synchronization and atomic compiler builtins" only ([Konstantin
  Serebryany, gcc.gnu.org, 2013](https://gcc.gnu.org/legacy-ml/gcc/2013-02/msg00295.html)). GCC's
  `libgomp` implements its own barrier/thread-pool synchronization without routing through
  primitives TSan instruments, so TSan cannot see the happens-before edge the barrier actually
  provides at runtime. The documented fix for accurate OpenMP race detection is **Archer**, an
  LLVM/OMPT-based race predictor built on Clang's `libomp` rather than GCC's `libgomp` ([HPC Wiki,
  ThreadSanitizer](https://hpc-wiki.info/hpc/ThreadSanitizer)) — a real but nontrivial toolchain
  change (Clang + `libomp` + Archer instead of GCC + `libgomp` for `tsan` builds specifically), not
  a code fix in this repository.
- Tested the commonly-suggested `TSAN_OPTIONS=ignore_noninstrumented_modules=1` workaround directly
  against `chatbotapiTests`: it does **not** suppress these warnings (18 races still reported in that
  run) — that flag only helps when the racing access itself happens inside literal non-instrumented
  code, but here both accesses are in our own instrumented `Matrix.cpp`; only the *synchronization
  connecting them* (libgomp's barrier) is invisible to TSan, which this flag doesn't address.
- A durable comment documenting this (with a pointer back to this entry) was added directly in
  `src/Matrix.cpp`, since that's where anyone confused by a `tsan` run will actually be looking.

**Revisit when:** The project adopts a Clang + `libomp` (+ Archer) toolchain for `tsan` builds
instead of GCC + `libgomp` — until then, expect roughly 20-90+ TSan warnings inside `Matrix.cpp`'s
OpenMP regions on any `tsan` run that exercises real matrix multiplication, all attributable to this
limitation. A genuinely different-shaped warning — e.g., one whose two accesses implicate actually
overlapping memory rather than a pre-barrier/post-barrier access pair — would still be worth
investigating on its own merits rather than dismissed by pattern-matching against this entry.

---

## Process Guidelines

### Adding New Technical Debt

When adding a new technical debt item:

1. **Create an entry in this document** with all required fields:
   - Unique ID (TD-XXX)
   - Priority (High/Medium/Low)
   - Component
   - Effort estimate
   - Description and impact
   - Location in code
   - Task checklist
   - Files affected

2. **Create a GitHub issue** using the technical debt template:
   - Link to this document
   - Use label: `technical-debt`
   - Assign priority label
   - Add to project board

3. **Update code comments** to reference the tracking item:

   ```cpp
   // See TD-001 in TECHNICAL_DEBT.md - Parameter exposure incomplete
   ```

4. **Remove untracked TODOs**

- All TODOs must be tracked here or in GitHub issues

### Prioritization Criteria

High Priority:

- Blocks new feature development
- Causes bugs or incorrect behavior
- Security or stability issues
- Affects multiple components

Medium Priority:

- Improves code maintainability significantly
- Reduces technical complexity
- Enables future features
- Clear path to resolution

Low Priority:

- Nice-to-have improvements
- Cosmetic code cleanup
- Performance optimizations (non-critical)
- Developer convenience features

### Resolving Technical Debt

When resolving a debt item:

1. Complete all tasks in the checklist
2. Add tests to prevent regression
3. Update documentation
4. Move item from "Active" to "Resolved" section with resolution date
5. Close related GitHub issue
6. Remove code comments referencing the item

---

## Statistics

### By Priority

|Priority|Count|Percentage|
|----------|-------|------------|
|High|1|4%|
|Medium|12|46%|
|Low|13|50%|

**Total Active Items:** 26

### By Component

|Component|Count|
|----------------------|-------|
|Core Model Architecture|1|
|Training / Data Generation|1|
|Training / Data Management|1|
|Training / Data Loading|1|
|CLI / User-Facing|1|
|Tooling / Toolchain|1|
|GPU / Inference / Training|1|
|GPU / Inference / Performance|1|
|Build / Windows / Metrics|1|
|RLHF / PPOOptimizer|1|
|Testing / Tooling|2|
|GUI / Testing|1|
|Advanced Features / Integration|1|
|Training / Metrics / Core|1|
|Security / Registry|1|
|GPU / Testing|1|
|Metrics / Testing|1|
|Scripts / Tooling|3|
|Scripts / Cleanup|1|
|Android / CI|1|
|Android / Testing|1|
|Tizen / Testing|1|

### Effort Distribution

|Effort Range|Count|
|--------------|-------|
|0-2 hours|1|
|2-4 hours|4|
|4-8 hours|7|
|8+ hours|12|
|Not estimated|2|

**Total Estimated Effort (Active Items):** 203-305 hours (excludes TD-014 and TD-039, which have no effort estimate)

### Future Enhancements Summary

**Total Future Enhancements:** 19
**Estimated Total Effort:** 100+ hours

By Category:

- Configuration and Service Management: 4 items (17-25 hours)
- Logging and Observability: 3 items (12-16 hours)
- Container and Deployment: 5 items (26-36 hours)
- Performance Optimizations: 4 items (38-66 hours)
- Code Quality: 1 item (4-6 hours)
- Developer Experience: 2 items (8-12 hours)

By Priority:

- High: 0 items
- Medium: 4 items (Batched Inference, Attention Head, Metrics Endpoint, Model State Persistence)
- Low: 15 items

Recently Completed:

- TD-020: Persistent Metrics Storage via SQL Database - September 7, 2026 (tracker correction; implementation predates this entry)
- TD-027: Install Script for incremental_trainer Sub-System - June 7, 2026
- TD-028: Separate Dataset Management from IncrementalTrainer - June 7, 2026
- TD-023: Parallel Generation Quality Scoring via Model Snapshot - June 4, 2026
- TD-022: Remove Direct Terminal Output from IncrementalTrainer and Dependencies - June 2, 2026
- TD-021: IncrementalTrainer × Metrics Service Decoupling - May 31, 2026
- TD-003: GPU Memory Management Optimization (GPUMatrix) - May 3, 2026
- TD-016: BLEU/ROUGE Generation Quality Scoring - April 11, 2026
- TD-013b: Batch Padding Efficiency Tracking - April 11, 2026
- TD-013 (partial): FeedForward activation saturation hooks - April 11, 2026
- TD-007: Matrix Operations SIMD Acceleration - April 11, 2026
- TD-015: Validation Metrics Integration - March 14, 2026
- TD-013: Advanced Training Metrics and Outlier Detection - March 14, 2026
- TD-009: Incremental Trainer Dashboard and Structured Logging - March 2, 2026
- TD-004: Enhanced Metrics Tracking (absorbed by TD-009) - March 2, 2026
- TD-010: Configuration Hot-Reloading - March 1, 2026
- TD-011: File Rotation and Management - March 1, 2026
- TD-012: Increase Test Coverage - March 1, 2026
- TD-008: Daemon Service Implementation - March 1, 2026
- TD-005: Checkpoint Management - February 18, 2026
- TD-002: BPE Tokenizer Error Handling - February 18, 2026
- TD-001: Optimizer Parameter Exposure - January 28, 2026

### Deferred Decisions Summary

**Total Deferred Decisions:** 2

- AMD Radeon / ROCm-HIP GPU Backend — Not Pursued (September 7, 2026)
- ThreadSanitizer "Data Races" in Matrix.cpp's OpenMP-Parallelized Code — Confirmed Tool Limitation, Not a Bug (September 12, 2026)

---

## References

- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) - Section 10: Technical Debt Items
- [Contributing Guide](docs/guides/contributing.md) - Code quality standards
- [GitHub Issues](https://github.com/yourusername/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - Active debt tracking

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
