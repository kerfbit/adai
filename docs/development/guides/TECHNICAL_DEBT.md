# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** September 8, 2026
**Total Items:** 24
**High Priority:** 0
**Medium Priority:** 11
**Low Priority:** 13
**Future Enhancements:** 19
**Resolved Items:** 34
**Deferred Decisions:** 1

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-050: GPU-Resident KV-Cache for Autoregressive Generation](#td-050-gpu-resident-kv-cache-for-autoregressive-generation)
  - [TD-033: chatbot_api_server Inference Never Uses Persistent GPU-Resident Decode](#td-033-chatbot_api_server-inference-never-uses-persistent-gpu-resident-decode)
  - [TD-032: Bundle SQLite3 Amalgamation for Windows Cross-Compilation](#td-032-bundle-sqlite3-amalgamation-for-windows-cross-compilation)
  - [TD-014: LLM Operations and Training Tooling Suite](#td-014-llm-operations-and-training-tooling-suite)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
  - [TD-034: PPOOptimizer's Core Update Loop Is a Placeholder, Not Real PPO](#td-034-ppooptimizers-core-update-loop-is-a-placeholder-not-real-ppo)
  - [TD-035: Shipped Daemon/CLI Binaries Have No Dedicated Test](#td-035-shipped-daemoncli-binaries-have-no-dedicated-test)
  - [TD-036: Thin main() Wrappers Have No Smoke Test](#td-036-thin-main-wrappers-have-no-smoke-test)
  - [TD-037: No Qt Test Infrastructure for GUI Classes](#td-037-no-qt-test-infrastructure-for-gui-classes)
  - [TD-038: Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary](#td-038-advanced-features-tested-in-isolation-never-wired-into-a-shipped-binary)
  - [TD-039: Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable](#td-039-core-trainingmetrics-classes-too-large-and-fast-moving-to-certify-stable)
  - [TD-040: FtpDataServer's Auth Path Unreviewed; RegistryServer Untested in Isolation](#td-040-ftpdataservers-auth-path-unreviewed-registryserver-untested-in-isolation)
  - [TD-041: GPUUtils Has No Dedicated Test on Either Backend](#td-041-gpuutils-has-no-dedicated-test-on-either-backend)
  - [TD-042: PostgresMetricsDatabase Has Zero Test Coverage](#td-042-postgresmetricsdatabase-has-zero-test-coverage)
  - [TD-043: Deployment-Critical Scripts Have No Automated Test](#td-043-deployment-critical-scripts-have-no-automated-test)
  - [TD-044: Manual-QA Launcher Scripts Have No Automated Test](#td-044-manual-qa-launcher-scripts-have-no-automated-test)
  - [TD-045: Standalone Dev-Utility Scripts Have No Test or Integration](#td-045-standalone-dev-utility-scripts-have-no-test-or-integration)
  - [TD-046: Orphaned/Superseded Scripts Should Be Removed or Reconciled](#td-046-orphanedsuperseded-scripts-should-be-removed-or-reconciled)
  - [TD-047: Android Data/Repository/API Layer Has No CI or Release History](#td-047-android-datarepositoryapi-layer-has-no-ci-or-release-history)
  - [TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased](#td-048-android-uidientry-point-classes-are-untested-and-unreleased)
  - [TD-049: No JS Test Framework for the Tizen TV App](#td-049-no-js-test-framework-for-the-tizen-tv-app)
  - [TD-051: IncrementalTrainer::load_conversation_pairs() Is an Unmigrated Duplicate](#td-051-incrementaltrainerload_conversation_pairs-is-an-unmigrated-duplicate)
  - [TD-052: ParallelDataLoader's Batches Use Character Codes, Not Real Tokens](#td-052-paralleldataloaders-batches-use-character-codes-not-real-tokens)
  - [TD-053: ChatbotCLI's /save and /load Commands Are Non-Functional Everywhere](#td-053-chatbotclis-save-and-load-commands-are-non-functional-everywhere)
- [Resolved Items](#resolved-items) (34 items — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md))
- [Future Improvements](#future-improvements)
  - [Performance Optimizations](#performance-optimizations)
  - [Code Quality](#code-quality)
  - [Developer Experience](#developer-experience)
  - [Configuration and Service Management](#configuration-and-service-management)
  - [Logging and Observability](#logging-and-observability)
  - [Container and Deployment](#container-and-deployment)
- [Deferred Decisions](#deferred-decisions)
  - [AMD Radeon / ROCm-HIP GPU Backend — Not Pursued](#amd-radeon--rocm-hip-gpu-backend--not-pursued)
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

### TD-050: GPU-Resident KV-Cache for Autoregressive Generation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Planned | GPU / Inference / Training | July 4, 2026 | 20-28 hours |

Description:
Neither the CPU nor the GPU decode path has a working incremental KV-cache. On CPU, `generate_response_with_strategy()`'s greedy branch explicitly bypasses `DecoderKVCache`/`forward_with_cache()` via a documented workaround ("TODO: Fix KV cache to properly handle autoregressive generation" — `src/EncoderDecoderModel.cpp`) because the cache produces incorrect results for greedy decoding. `EncoderDecoderModel::gpu_generate_response()` (added to GPU-accelerate BLEU/ROUGE scoring during validation) inherits the same limitation by necessity: no GPU-resident cache exists at all, so every decode step recomputes the full sequence from scratch via `LLMDecoder::gpu_decode()` — O(n) work per step, O(n^2) total over a generation, instead of O(1) per step / O(n) total with a correct cache. This is functionally correct (mirrors the CPU workaround's algorithmic shape) but leaves an easy performance win on the table now that generation runs on GPU.

Action Items:

- [ ] Root-cause the existing CPU `DecoderKVCache` correctness bug (self-attention and/or cross-attention cache indexing) in `src/KVCache.hpp` / `src/Decoder.cpp` `forward_with_cache()` before building the GPU equivalent on top of the same flawed model.
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
| MEDIUM | Open | GPU / Inference / Performance | September 7, 2026 | 6-8 hours |

Description:
Originally filed as "Matrix GPU Dispatch Doesn't Use Persistent GPUMatrix Residency," on the claim that nothing in the training/inference hot path calls `to_gpu()`/`from_gpu()` and that TD-003's persistent-residency win was never realized outside code that explicitly opts in. That claim is only half true, and the half that's false changes where the fix belongs — this entry replaces it with the verified scope.

**Training already has full persistent GPU residency.** `ChatbotTrainer::train_epoch()` calls `EncoderDecoderModel::gpu_forward()`/`gpu_backward()` (`src/ChatbotTrainer.cpp`) under `#ifdef ADAI_ENABLE_GPU`, which cascades through `EncoderBlock`/`DecoderBlock::gpu_forward()`/`gpu_backward()` (`src/EncoderBlock.cpp`, `src/DecoderBlock.cpp`) into `MultiHeadAttention`/`FeedForward`/`LayerNorm::gpu_forward()`/`gpu_backward()`, all chaining `GPUMatrix` end-to-end with cached device-resident buffers (`cached_Q`, `cached_K`, `cached_hidden`, etc.). This is exactly TD-003's intended design (resolved May 3, 2026), fully wired up, and identical on both backends — `GPUMatrix` exposes the same API in `src/gpu/MatrixGPU.hpp` (CUDA) and `src/gpu/sycl/MatrixGPU_SYCL.hpp` (SYCL).

**The real, live gap is inference serving.** `ChatbotAPI::generate_response()` (`src/ChatbotAPI.cpp`) builds its `model_fn` around `EncoderDecoderModel::forward()` — the plain CPU `Matrix` path — never the GPU-resident one. `Matrix::operator*()` (`src/Matrix.cpp`) auto-dispatches per call to `Matrix::multiply_gpu()` whenever `GPUManager::is_available()` and both inner dimensions are ≥32; `multiply_gpu()` (and `add_gpu`/`transpose_gpu`/`scale_gpu`/`hadamard_gpu`) allocate device memory, upload, compute, and download on every single call, with zero reuse of `GPUMatrix`. So on a GPU build, every live chat request re-pays a full alloc+upload+compute+download for every matmul, in every layer, for every generated token — worse than the training case, since it's per-token request latency rather than amortized training throughput. Confirmed backend-symmetric: `GPUMemory` does a fresh `cudaMalloc` (`src/gpu/GPUUtils.hpp`) or `sycl::malloc_device` (`src/gpu/sycl/GPUUtils_SYCL.hpp`) per call either way — this is a `Matrix.cpp`/`ChatbotAPI.cpp` dispatch problem, not a CUDA-vs-SYCL difference.

`EncoderDecoderModel::gpu_generate_response()` (`src/EncoderDecoderModel.cpp`) already exists as a persistent-residency decode path, but it's wired only into `ChatbotTrainer`'s internal generation-quality backfill / BLEU-ROUGE scoring (`src/ChatbotTrainer.cpp`), never into `ChatbotAPI` — so the one caller that would most benefit (real chat latency) has never been connected to it.

Related: TD-050 (GPU-Resident KV-Cache) covers a different problem inside `gpu_generate_response()` itself — once called, it recomputes the full sequence from scratch every step (no KV-cache), which is O(n²) instead of O(n). TD-050 is about making `gpu_generate_response()` fast once used; this item is about it not being used by `chatbot_api_server` at all.

Discovered during the per-file production-readiness rollout (September 7, 2026) — see [file-status-standard.md](file-status-standard.md); corrected same day after tracing both GPU backends end-to-end.

Action Items:

- [ ] Give `ChatbotAPI::generate_response()`'s `model_fn` a GPU-resident branch that calls `EncoderDecoderModel::gpu_generate_response()` (or an equivalent single-pass GPU decode) when `GPUManager::is_available()`, instead of unconditionally calling `model_->forward()`.
- [ ] Confirm `TextGenerator`'s beam/top-k/nucleus/temperature strategies all work against the GPU-resident decode path — `gpu_generate_response()` was built for BLEU/ROUGE scoring and may only need to support greedy today; verify before switching real callers over, extend if needed.
- [ ] Benchmark chat response latency before/after on a representative prompt/`max_length` on both CUDA and SYCL builds to confirm the per-token round-trip elimination.
- [ ] Leave `Matrix::multiply_gpu()` and siblings as-is for callers that only have CPU `Matrix` data and no persistent-residency alternative — this item is about routing the identified hot path around them, not removing them.

Files to Modify:

- `src/ChatbotAPI.cpp` / `src/ChatbotAPI.hpp` — `generate_response()`'s `model_fn`
- `src/EncoderDecoderModel.cpp` / `src/EncoderDecoderModel.hpp` — adjust `gpu_generate_response()` for a non-training caller if needed (e.g. strategy support)
- `src/TextGenerator.cpp` / `src/TextGenerator.hpp` — verify/extend strategy support against the GPU-resident decode path
- `tests/` — coverage confirming identical output between the CPU and GPU-resident serving paths

---

### TD-032: Bundle SQLite3 Amalgamation for Windows Cross-Compilation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Build / Windows / Metrics | September 7, 2026 | 2-4 hours |

Description:
TD-020 (Persistent Metrics Storage via SQL Database, resolved — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md)) links SQLite3 via `find_path`/`find_library` in `src/CMakeLists.txt`, which works on Linux/macOS but has no equivalent for the MinGW Windows cross-compilation path (`scripts/build_windows.sh`, `scripts/package_windows.sh`). TD-020's original proposal called for bundling the public-domain SQLite amalgamation specifically to cover this case; that one item was the only part of the proposal not carried out. Everything else in TD-020 shipped and is verified working.

Action Items:

- [ ] Vendor the SQLite amalgamation into `external/sqlite3/` (`sqlite3.c`, `sqlite3.h`).
- [ ] Add a CMake option to build it as a static lib on MinGW/Windows targets when system SQLite3 isn't found.
- [ ] Verify `metrics_api_server` builds and runs under Wine or a Windows VM with the SQLite backend enabled.

Files to Modify:

- `external/sqlite3/` (new)
- `src/CMakeLists.txt`
- `scripts/build_windows.sh`

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

### TD-034: PPOOptimizer's Core Update Loop Is a Placeholder, Not Real PPO

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | RLHF / PPOOptimizer | September 7, 2026 | 10-16 hours |

Description:
`PPOOptimizer::train()`'s minibatch loop never recomputes log-probabilities under the current
policy: `float new_log_prob = batch_old_log_probs[i];  // Placeholder` makes the clipped-ratio term
`exp(new_log_prob - batch_old_log_probs[i])` always evaluate to `exp(0) = 1`, so the policy loss is
not actually PPO's clipped surrogate objective — it silently trains as something else. The KL-based
early-stopping check has the same problem: `float approx_kl = 0.0f;  // Placeholder` means the
`> 1.5 * kl_target` early-stop condition can never fire. `PPOOptimizer` is not wired into any
shipped binary (`phase5_test.cpp` exercises it in isolation only — see
[PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)), so this has no effect on any current
training path, but it would silently misbehave the moment RLHF fine-tuning is wired up.

**A second, more severe bug in the same file, found during a later re-audit (September 8, 2026):**
`ValueFunction::update()` allocates `weight_grads`/`bias_grads` (zero-initialized), then its
per-sample loop computes a local `grad` variable (`// Backward pass (simplified...)`) that is
**never written into `weight_grads`/`bias_grads` at all** — the variable is computed and
immediately discarded. The weight-update loop below then does
`weights_[i](r, c) -= learning_rate * weight_grads[i](r, c)` against gradients that are still
exactly zero. Net effect: `update()` returns a real, correctly-computed MSE loss (so a caller
would see a plausible-looking "loss" value) but **never changes a single weight** — the value
function is permanently frozen at its random initialization, no matter how many times `update()`
is called. This is strictly worse than the ratio/KL issue above (that one degrades to a wrong-but-
nonzero gradient signal; this one is a complete no-op) and was missed on the first pass because
the file's `@adai-status` tag already flagged it as broken for the ratio/KL reason — nobody
checked whether *every* function in the file had the same problem.

Discovered during the per-file production-readiness rollout (September 7, 2026) — see
[file-status-standard.md](file-status-standard.md).

Action Items:

- [ ] Recompute `new_log_prob` via a real forward pass of the current policy over `batch_states[i]`,
  not a copy of `batch_old_log_probs[i]`.
- [ ] Track actual per-minibatch KL divergence between old and current policy for `approx_kl`
  instead of the hardcoded `0.0f`.
- [ ] Implement `ValueFunction::update()`'s backward pass for real: accumulate `grad` into
  `weight_grads`/`bias_grads` via actual backprop through the network's cached activations,
  instead of computing and discarding it.
- [ ] Add a test that trains `ValueFunction` on a toy regression target and asserts its weights
  actually move and its loss actually decreases over iterations — the current test suite doesn't
  catch a permanently-frozen value function because it likely only checks that `update()` runs
  without crashing and returns a plausible loss value.
- [ ] Add a test that trains `PPOOptimizer` end-to-end on a toy environment/reward and asserts the
  policy actually changes (a no-op ratio would previously have passed any test that doesn't check
  this).

Files to Modify:

- `src/PPOOptimizer.hpp`
- `tests/phase5_test.cpp`

---

### TD-035: Shipped Daemon/CLI Binaries Have No Dedicated Test

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Testing / Tooling | September 7, 2026 | 12-16 hours |

Description:
Six of the binaries in CLAUDE.md's own "Executable Targets" table — the actual shipped
product — have no automated test at all: `chatbot_api_server`, `dataset_manager`, `mns_cli`,
`mns_server`, `metrics_api_server`, and `incremental_trainer` each have their `main()`-hosting
`.cpp` file doing real work (argument parsing, command dispatch, server lifecycle) with zero
coverage. `ChatbotCLI` already shows the fix: `ChatbotCLI.cpp`/`.hpp` hold the testable logic
(and have a real test) while `ChatbotCLI_main.cpp` is a thin, untested wrapper (tracked separately
as TD-036, since a 40-line argv shim isn't the same problem as a whole untested server).

Action Items:

- [ ] For each binary, extract its command-dispatch / server-setup logic out of `main()` into a
  testable function or class, the way `ChatbotCLI` already separates from `ChatbotCLI_main`.
- [ ] Add a dedicated test per binary covering: argument parsing, config loading, and (for the
  three daemons) the request-handling entry points not already covered by a live/integration test.
- [ ] `RegistryServer.cpp` is a partial case — already exercised indirectly by
  `dataset_registry_live_test.cpp`/`trainer_admin_api_test.cpp` as a live instance; needs a
  dedicated unit test of its own request-handling logic in isolation, not a from-scratch test.

Files to Modify:

- `src/ChatbotAPIServer.cpp`
- `src/DatasetManagerTool.cpp`
- `src/MnsCliTool.cpp`
- `src/ModelNameServiceServer.cpp`
- `src/TrainingMetricsAPIServer.cpp`
- `src/IncrementalTrainingTool.cpp`
- `src/RegistryServer.cpp` (see note above — narrower scope)
- `tests/` — one new test file per binary

---

### TD-036: Thin main() Wrappers Have No Smoke Test

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Testing / Tooling | September 7, 2026 | 3-5 hours |

Description:
`ChatbotCLI_main.cpp`, `ChatbotGUI_main.cpp`, `ChatbotGUI_wrapper.cpp`, and
`MnsManagerGUI_main.cpp` are all 40-75 line argv-parsing shims that construct and delegate to
already-tested library classes. A GTest unit test isn't a good fit for a `main()` — the realistic
path is a scripted smoke test (invoke the binary, check exit code and `--help` output), the same
category as the manual QA scripts already in `scripts/` (`test_chatbot_gui.sh`, etc.), just made
automated and part of the suite instead of a manual step.

Action Items:

- [ ] Add a lightweight smoke-test script (or a CTest `add_test` entry wrapping one) per binary:
  invoke with `--help`/`--version`, assert exit code 0 and non-empty output.
- [ ] Wire these into `ctest` (e.g. via `add_test(... COMMAND sh -c "...")`) so they run alongside
  the rest of the suite instead of living only as manual `scripts/test_*.sh` invocations.

Files to Modify:

- `src/ChatbotCLI_main.cpp`
- `src/ChatbotGUI_main.cpp`
- `src/ChatbotGUI_wrapper.cpp`
- `src/MnsManagerGUI_main.cpp`
- `tests/CMakeLists.txt`

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
| LOW | Open | Advanced Features / Integration | September 7, 2026 | 16-24 hours |

Description:
`BatchedInferenceEngine`, `IntegratedInferenceEngine`, `PipelineInferenceEngine`,
`SpeculativeDecoding`, `LoRA`, `Quantization`, `RewardModel`, and `PerformanceProfiler` each have
real, passing dedicated tests — the gap isn't test coverage, it's that nothing in `chatbot`,
`chatbot_api_server`, or `incremental_trainer` actually calls any of them. They're complete,
correct implementations of a scoped feature sitting unreachable from the actual product. Each
needs its own integration decision (a CLI flag, a config option, a training-mode switch) rather
than one shared fix — grouped here because they share the identical structural gap, not because
one change resolves all eight.

Action Items:

- [ ] `BatchedInferenceEngine` / `PipelineInferenceEngine`: wire into `chatbot_api_server`'s
  request path as an opt-in serving mode.
- [ ] `SpeculativeDecoding`: add a CLI/config flag to `chatbot`/`chatbot_api_server` enabling it
  for generation.
- [ ] `LoRA` / `Quantization`: hook into `ModelSerializer` save/load or `IncrementalTrainer`'s
  fine-tuning flow.
- [ ] `RewardModel`: wire into an actual RLHF training command (blocked on TD-034's PPO fix landing
  first — no point integrating a reward model into a policy-update loop that doesn't work yet).
- [ ] `PerformanceProfiler`: wire into at least one binary's `--profile` flag or equivalent.
- [ ] Add an integration test per feature proving the wiring works end-to-end, not just the
  existing isolated unit test.

Files to Modify:

- `src/BatchedInferenceEngine.hpp`, `src/IntegratedInferenceEngine.hpp`,
  `src/PipelineInferenceEngine.hpp`, `src/SpeculativeDecoding.hpp`, `src/LoRA.hpp`,
  `src/Quantization.hpp`, `src/RewardModel.hpp`, `src/PerformanceProfiler.hpp`
- `src/ChatbotAPI.cpp`, `src/IncrementalTrainer.cpp` (likely integration points)

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

### TD-040: FtpDataServer's Auth Path Unreviewed; RegistryServer Untested in Isolation

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Security / Registry | September 7, 2026 | 6-10 hours |

Description:
`FtpDataServer.hpp` implements a hand-rolled FTP server with its own authentication, token, and
virtual-user logic (`IssuedToken`, `TokenStore`, `VirtualUser`), embedded directly in
`registry_server`'s startup path (`RegistryServer.cpp` constructs it in `main()`). It has a real
602-line test and no demonstrated defect, but no one has done a focused security read of the
auth/token path specifically — a hand-rolled network-facing auth implementation deserves that
before being called production-ready, regardless of general test coverage. `RegistryServer.cpp`
itself is only exercised as a live instance via `dataset_registry_live_test.cpp` and
`trainer_admin_api_test.cpp`, not tested in isolation (see also TD-035, which covers the same file
from the "no dedicated unit test" angle — this item is specifically about the FTP auth review).

Action Items:

- [ ] Security-focused read of `FtpDataServer.hpp`'s authentication, token issuance/validation, and
  virtual-user permission logic.
- [ ] Confirm token expiry/scope enforcement can't be bypassed and credentials aren't logged.
- [ ] Add a dedicated `RegistryServer` unit test isolating its request-handling logic from the live
  server it's normally only exercised through.

Files to Modify:

- `src/FtpDataServer.hpp`
- `src/RegistryServer.cpp`

---

### TD-041: GPUUtils Has No Dedicated Test on Either Backend

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | GPU / Testing | September 7, 2026 | 3-5 hours |

Description:
Neither `gpu/GPUUtils.hpp` (CUDA) nor `gpu/sycl/GPUUtils_SYCL.hpp` (SYCL) has a dedicated test —
both `GPUManager`/`GPUMemory` are only exercised incidentally through `Matrix`'s GPU dispatch
tests. `gpu/GPUUtils.hpp` was incorrectly tagged `stable` during the original per-file rollout
(the "stable requires tests" rule was violated); corrected to `beta` as part of filing this item.
`gpu/sycl/GPUUtils_SYCL.hpp` could not be built or tested during that rollout — no SYCL toolchain
(Intel oneAPI `icpx`) was available in that environment — so its status is asserted from reading
the code, not from a passing build.

Action Items:

- [ ] Add a dedicated test for `GPUManager`/`GPUMemory` (device init, allocation, the CPU-only
  stub path) — one for CUDA, one for SYCL, both gated behind their respective `ENABLE_GPU`/
  `ENABLE_SYCL` CMake options like the rest of the GPU-specific tests.
- [ ] Build and run the SYCL variant on a machine with Intel oneAPI installed to confirm it
  actually compiles — this has not been verified since the file was last touched.

Files to Modify:

- `src/gpu/GPUUtils.hpp`
- `src/gpu/sycl/GPUUtils_SYCL.hpp`
- `tests/CMakeLists.txt`

---

### TD-042: PostgresMetricsDatabase Has Zero Test Coverage

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Metrics / Testing | September 7, 2026 | 4-6 hours |

Description:
`PostgresMetricsDatabase.{cpp,hpp}` is the optional Postgres backend selected by
`ENABLE_POSTGRES_METRICS`; it isn't built by default and has no test, unlike its SQLite sibling
(`SQLiteMetricsDatabase`, thoroughly covered by `MetricsDatabaseTest.cpp`). Since it implements the
same `IMetricsDatabase` interface, `MetricsDatabaseTest.cpp`'s existing fixtures are a natural
starting point rather than a from-scratch design.

Compiled standalone against the system `libpq` (`g++ -std=c++17 -DENABLE_POSTGRES_METRICS`,
September 7, 2026) to check whether it even builds, since no configured build here has
`ENABLE_POSTGRES_METRICS=ON` — it compiled clean, zero errors or warnings. So this is purely a
coverage/CI gap, not a known code defect; the action items below are unchanged.

Action Items:

- [ ] Parameterize (or duplicate) `MetricsDatabaseTest.cpp`'s `IMetricsDatabase` test cases to run
  against `PostgresMetricsDatabase` when `ENABLE_POSTGRES_METRICS` is on.
- [ ] Add a CI job (or document a local setup) that builds with `ENABLE_POSTGRES_METRICS=ON`
  against a real Postgres instance, since this configuration is currently untested by any CI path.

Files to Modify:

- `src/PostgresMetricsDatabase.cpp` / `src/PostgresMetricsDatabase.hpp`
- `tests/MetricsDatabaseTest.cpp`

---

### TD-043: Deployment-Critical Scripts Have No Automated Test

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Scripts / Tooling | September 7, 2026 | 14-20 hours |

Description:
16 scripts that the project actually depends on for building, packaging, and deploying — every
`install_*.sh`, `package_*.sh`/`package-sycl.sh`, `build_windows.sh`, `docker_build.sh`,
`model_service.sh`, plus `check_tech_debt.sh`, `run_tests.sh`, and this standard's own
`check_file_status.py`/`gen_status_report.py` — have no automated test of the script itself.
Several are documented as the sanctioned way to do something (`CLAUDE.md`, `SERVER_BUNDLE_DEPLOYMENT.md`)
but nothing verifies the script's own argument parsing, error handling, or output stays correct
across changes.

Action Items:

- [ ] Add a lightweight test harness for shell scripts (e.g. bats-core or a plain
  assert-and-diff wrapper) and wire it into `ctest` or a dedicated CI job.
- [ ] For the two Python tools (`check_file_status.py`, `gen_status_report.py`), add a real
  `pytest`/`unittest` suite — they currently rely only on the manual test cases run interactively
  during this rollout, not anything repeatable.
- [ ] Prioritize `install_server_bundle.sh` and `check_tech_debt.sh`/`run_tests.sh` first — the
  most central of the sixteen.

Files to Modify:

- `scripts/build_windows.sh`, `scripts/docker_build.sh`, `scripts/install_chatbot_API.sh`,
  `scripts/install_incremental_trainer.sh`, `scripts/install_metrics_service.sh`,
  `scripts/install_mns_server.sh`, `scripts/install_oneapi_libs.sh`,
  `scripts/install_server_bundle.sh`, `scripts/model_service.sh`, `scripts/package-sycl.sh`,
  `scripts/package_server_bundle.sh`, `scripts/package_windows.sh`, `scripts/check_tech_debt.sh`,
  `scripts/run_tests.sh`, `scripts/check_file_status.py`, `scripts/gen_status_report.py`

---

### TD-044: Manual-QA Launcher Scripts Have No Automated Test

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Scripts / Tooling | September 7, 2026 | 6-10 hours |

Description:
11 scripts covering chatbot/GUI launching and manual regression checks for already-shipped
features (config hot-reload, log rotation, signal handling, parallel processing) — same "no
automated test of the script itself" gap as TD-043, split out because these are lower-stakes,
developer-facing manual QA tools rather than the deployment path.

Action Items:

- [ ] Same harness as TD-043, applied here once that's stood up — no need to design a second
  approach.
- [ ] Lower priority than TD-043; address opportunistically.

Files to Modify:

- `scripts/chatbot_gui_fixed.sh`, `scripts/manual_test_reload.sh`, `scripts/run_chatbot.sh`,
  `scripts/run_chatbot_gui.sh`, `scripts/test_chatbot_gui.sh`,
  `scripts/test_chatbot_gui_comprehensive.sh`, `scripts/test_config_reload.sh`,
  `scripts/test_log_rotation.sh`, `scripts/test_signal_handling.sh`,
  `scripts/verify_cli_parallel.sh`, `scripts/verify_gui_parallel.sh`

---

### TD-045: Standalone Dev-Utility Scripts Have No Test or Integration

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Scripts / Tooling | September 7, 2026 | 4-6 hours |

Description:
6 self-contained utility scripts (an example API client, a driver-update monitor, a markdown
linter, a training dashboard, a static file server, a port checker) with no test and no other
script depending on them. `check_ports.sh` additionally has a known, specific bug beyond "no
test": its hardcoded port list (`8080 8081 8082`) omits `mns_server` (8083) and the trainer admin
API (8084), so it under-reports what's actually listening.

Action Items:

- [ ] Fix `check_ports.sh`'s port list to include 8083/8084.
- [ ] Add a minimal smoke test per script (invoke it, assert it doesn't crash) — full unit tests
  aren't warranted for tools this small and low-stakes.

Files to Modify:

- `scripts/batch_api_client.py`, `scripts/check_intel_driver_updates.py`, `scripts/check_ports.sh`,
  `scripts/fix_markdown_lint.py`, `scripts/monitor_training.py`, `scripts/serve_dashboard.py`

---

### TD-046: Orphaned/Superseded Scripts Should Be Removed or Reconciled

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Scripts / Cleanup | September 7, 2026 | 2-3 hours |

Description:
6 scripts that aren't a testing gap at all — they're dead weight or duplicates, found during the
per-file rollout: `scan_todos.sh` duplicates `check_tech_debt.sh` almost line-for-line and isn't
documented anywhere; `test_sigint.sh` is a near-identical, less-complete twin of
`test_signal_handling.sh`; `install_registry_server.sh` isn't called by
`install_server_bundle.sh` (which now handles `registry_server` directly) and isn't documented;
`docker_deploy.sh` isn't referenced by current docs (`docker.md` documents `docker_build.sh` +
`docker-compose` instead); `apply_narrowing_fixes.py` and `verify_special_token_fixes.py` are
explicitly one-off scripts tied to a specific already-completed past fix. This item is a decision
to make (delete vs. keep for reference), not a fix to implement.

Action Items:

- [ ] Confirm each is genuinely superseded (spot-checked already during the rollout — see
  [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)) and delete, or explicitly document why
  it's being kept.

Files to Modify:

- `scripts/apply_narrowing_fixes.py`, `scripts/docker_deploy.sh`,
  `scripts/install_registry_server.sh`, `scripts/scan_todos.sh`, `scripts/test_sigint.sh`,
  `scripts/verify_special_token_fixes.py`

---

### TD-047: Android Data/Repository/API Layer Has No CI or Release History

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Android / CI | September 7, 2026 | 10-14 hours |

Description:
41 files across `android/app` (15) and `android/opsdashboard`/`android/wearsync` (26) — the
repository/network/DTO/poller/contract layer — each have real dedicated unit tests and pass, but
the entire Android surface landed in a single commit, both apps still declare
`versionName = "0.1.0"`, and no CI workflow builds or tests either app. Being tested in isolation
isn't the same as having a release process; that's the actual gap here, distinct from TD-048
(files with no test at all).

Action Items:

- [ ] Add a GitHub Actions workflow building both `:app` and `:opsdashboard` and running their
  unit tests (`./gradlew testDebugUnitTest`) — no such job currently exists anywhere in
  `.github/workflows/`.
- [ ] Establish a real release/versioning process before either app leaves `0.1.0`.

Files to Modify:

- 41 files under `android/app/src/main` and `android/opsdashboard/src/main` /
  `android/wearsync/src/main` tagged `beta` — see [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md)
  for the exact list.
- `.github/workflows/` (new Android CI job)

---

### TD-048: Android UI/DI/Entry-Point Classes Are Untested and Unreleased

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Android / Testing | September 7, 2026 | 24-32 hours |

Description:
62 files — Compose screens, ViewModels without tests, DI containers, `Activity`/`Application`
entry points, and the `wearcomplications` services — genuinely have zero automated coverage, on
top of sharing TD-047's no-CI/no-release problem. `BiometricAdminAuthGate.kt` is a partial
exception: hard to unit-test (`BiometricPrompt` needs a real `FragmentActivity`), but it was read
manually during the rollout and appears complete — the gap there is coverage, not a known defect.

Action Items:

- [ ] Add ViewModel unit tests first (cheapest — no Compose/Activity needed), following the
  pattern already established for `TrainerViewModel`, `SettingsViewModel`, etc.
- [ ] Adopt Compose UI testing (`androidx.compose.ui.test`) for screens once ViewModels are
  covered.
- [ ] `BiometricAdminAuthGate.kt` specifically: consider an instrumented test using
  `BiometricPrompt`'s test/fake authenticator support instead of leaving it permanently untested.

Files to Modify:

- 62 files under `android/app/src/main`, `android/opsdashboard/src/main`, and
  `android/wearcomplications/src/main` tagged `experimental` — see
  [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) for the exact list.

---

### TD-049: No JS Test Framework for the Tizen TV App

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Tizen / Testing | September 7, 2026 | 4-6 hours |

Description:
`tizen-metrics-app/js/{app,chart,navigation}.js` have no test framework at all — unlike the rest
of the tree, this app has never had one. All three are wired into `index.html`, iterated on over
24 commits, and actually deployed to Samsung TV hardware, so this isn't a correctness concern,
just a gap: nothing would catch a regression before it ships to the TV.

Action Items:

- [ ] Add a minimal JS test setup (e.g. a small assertion helper run via Node, or a headless
  browser harness) for the pure-logic pieces — `chart.js`'s coordinate math and `navigation.js`'s
  key-code mapping are the most testable without a real DOM/TV remote.

Files to Modify:

- `tizen-metrics-app/js/app.js`, `tizen-metrics-app/js/chart.js`,
  `tizen-metrics-app/js/navigation.js`

---

### TD-051: IncrementalTrainer::load_conversation_pairs() Is an Unmigrated Duplicate

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| LOW | Open | Training / Data Management | September 7, 2026 | 1-2 hours |

Description:
Found while auditing every active TD entry for an appropriate in-code TODO. TD-028's dataset
management refactor (resolved June 7, 2026) added `DatasetRegistry::load_conversation_pairs()`
as the intended new home for this logic — confirmed byte-for-byte identical to
`IncrementalTrainer::load_conversation_pairs()` (same 73 lines, only a renamed parameter,
`filepath` vs `path`) — but never actually removed the original or redirected its two call sites
(`IncrementalTrainer.cpp:926,993`), which still call the old copy. A `// TODO(TD-028): Move to
DatasetRegistry::load_conversation_pairs()` comment has sat on the unmigrated copy since — but
TD-028 itself is closed, so this was effectively an orphaned, never-completed sub-task inside an
otherwise-resolved item.

Action Items:

- [ ] Redirect `IncrementalTrainer.cpp:926,993` to call `DatasetRegistry::load_conversation_pairs()`.
- [ ] Delete `IncrementalTrainer::load_conversation_pairs()` and its declaration.
- [ ] Verify `incrementaltrainerTests` still pass after the redirect.

Files to Modify:

- `src/IncrementalTrainer.cpp` / `src/IncrementalTrainer.hpp`

---

### TD-052: ParallelDataLoader's Batches Use Character Codes, Not Real Tokens

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | Training / Data Loading | September 8, 2026 | 4-6 hours |

Description:
`ParallelDataLoader.hpp` was incorrectly tagged `stable` in the original per-file
production-readiness rollout — corrected to `experimental` when this was found during a
broader re-audit. Its batch-generation code is explicit about being a placeholder:
`// Note: For now we'll create dummy token sequences from the text` / `// In a real
implementation, this should use a tokenizer`, followed by `// Create simple token sequence
(char codes for demonstration)` — each character of the input text is converted to its raw
`unsigned char` value and used directly as a "token ID," with no BPE tokenizer involved at all.
`ParallelDataLoaderTest`'s tests do exercise this code path (`next_batch()`), but they only
check batch shape/counting, not token content — so the tests passing gave false confidence
during the original rollout that this class was production-ready. It is also not included by
any production `src/*.cpp` file — only by its own test.

Action Items:

- [ ] Replace the char-code loop with a real `BPETokenizer::encode()` call.
- [ ] Add a test asserting batch contents are valid vocabulary token IDs, not raw byte values.
- [ ] Re-evaluate whether `ParallelDataLoader` should be wired into a real training path once
  fixed, or whether `EfficientBatching`/`Dataset` already cover this need and it should be
  retired instead.

Files to Modify:

- `src/ParallelDataLoader.hpp`
- `tests/paralleldataloader_test.cpp`

---

### TD-053: ChatbotCLI's /save and /load Commands Are Non-Functional Everywhere

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | CLI / User-Facing | September 8, 2026 | 6-10 hours |

Description:
`docs/operations/guides/chatbot-guide.md` documents an entire "Conversation History" feature set
— automatic save-on-exit, manual `/save`, manual `/load` — as real and working, with example
usage shown twice. None of it exists: `ChatbotCLI.cpp`'s `/exit`/`/quit` handler only sets
`running = false` (no save call, no "conversation_history.txt" string anywhere in the file), and
`/save`/`/load` share one handler that unconditionally prints `"Save/Load not supported in API
client mode yet."` The message's wording implies a working alternative mode exists; it doesn't —
there is no other code path anywhere in `ChatbotCLI.cpp`/`.hpp` that saves or loads a
conversation. A user following the documented examples would hit a dead end on all three.

**Broader context found while fixing the doc:** `chatbot-guide.md` turned out to describe an
entire earlier CLI architecture — a standalone binary taking `[vocab_file] [model_file]
[conversation_save_file]` — that predates `ChatbotCLI` becoming a thin HTTP client for
`chatbot_api_server` (current args, verified against `ChatbotCLI_main.cpp`:
`[server_url] [conversation_save_file]`). The Quick Start banner, File Requirements, Default
File Paths, "Starting the Chatbot", and Command-Line Help sections have been corrected to match
current behavior (September 8, 2026); a banner at the top of the doc flags that later sections
(Commands Reference details, Generation Strategies, Configuration Parameters) have not been
re-verified against a live `chatbot`/`chatbot_api_server` pair and may have the same problem.

Action Items:

- [ ] Either implement save-on-exit and `/save`/`/load` (serialize/restore `ConversationContext`
  to/from disk) or formally decide they're out of scope for the API-client CLI and remove the
  "planned" framing from the doc instead of leaving it aspirational indefinitely.
- [ ] Add a CLI test asserting the actual current behavior (clear error on `/save`/`/load`, no
  crash or silent no-op on exit) so this doesn't regress silently either way.
- [ ] Do a full pass over the rest of `chatbot-guide.md` (Commands Reference, Generation
  Strategies, Configuration Parameters) against a live `chatbot` + `chatbot_api_server` pair —
  the sections already fixed were the ones verifiable by reading source directly; the rest need
  interactive verification. Remove the "partially stale" banner once done.

Files to Modify:

- `src/ChatbotCLI.cpp`
- `docs/operations/guides/chatbot-guide.md`

---

## Resolved Items

34 items resolved. See [archive/TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md) for full details.

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
|High|0|0%|
|Medium|11|46%|
|Low|13|54%|

**Total Active Items:** 24

### By Component

|Component|Count|
|----------------------|-------|
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
|4-8 hours|6|
|8+ hours|11|
|Not estimated|2|

**Total Estimated Effort (Active Items):** 171-251 hours (excludes TD-014 and TD-039, which have no effort estimate)

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

**Total Deferred Decisions:** 1

- AMD Radeon / ROCm-HIP GPU Backend — Not Pursued (September 7, 2026)

---

## References

- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) - Section 10: Technical Debt Items
- [Contributing Guide](docs/guides/contributing.md) - Code quality standards
- [GitHub Issues](https://github.com/yourusername/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - Active debt tracking

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
