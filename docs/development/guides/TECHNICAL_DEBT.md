# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** September 7, 2026
**Total Items:** 7
**High Priority:** 0
**Medium Priority:** 3
**Low Priority:** 4
**Future Enhancements:** 19
**Resolved Items:** 32
**Deferred Decisions:** 1

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-030: GPU-Resident KV-Cache for Autoregressive Generation](#td-030-gpu-resident-kv-cache-for-autoregressive-generation)
  - [TD-029: Fix GCC 13 ICE in raginference\_test.cpp](#td-029-fix-gcc-13-ice-in-raginference_testcpp)
  - [TD-033: Matrix GPU Dispatch Doesn't Use Persistent GPUMatrix Residency](#td-033-matrix-gpu-dispatch-doesnt-use-persistent-gpumatrix-residency)
  - [TD-032: Bundle SQLite3 Amalgamation for Windows Cross-Compilation](#td-032-bundle-sqlite3-amalgamation-for-windows-cross-compilation)
  - [TD-014: LLM Operations and Training Tooling Suite](#td-014-llm-operations-and-training-tooling-suite)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
  - [TD-034: PPOOptimizer's Core Update Loop Is a Placeholder, Not Real PPO](#td-034-ppooptimizers-core-update-loop-is-a-placeholder-not-real-ppo)
- [Resolved Items](#resolved-items) (32 items — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md))
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

### TD-030: GPU-Resident KV-Cache for Autoregressive Generation

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

### TD-029: Fix GCC 13 ICE in raginference\_test.cpp

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|-----------------|
| LOW | Open | Tests / RAGInference | June 7, 2026 | 1-2 hours |

Description:
`tests/raginference_test.cpp` triggers an Internal Compiler Error (ICE) in GCC 13 (`cc1plus` exits with SIGSEGV) during a full build, preventing the `raginferenceTests` target from being compiled. All other targets build and test cleanly. The root cause is a C++ construct in the 504-line test file that tickles a GCC 13 code-generation bug. The fix is to identify the offending construct (likely a complex template instantiation or lambda capture) and either simplify it or add a targeted workaround.

Action Items:

- [ ] Bisect `raginference_test.cpp` to isolate the construct that triggers the ICE (binary-search by commenting out half the file, rebuild, repeat).
- [ ] Apply the minimal fix: restructure the construct or break it into smaller translation units.
- [ ] Verify build succeeds with GCC 13 and add a CI note to monitor for recurrence.

---

### TD-033: Matrix GPU Dispatch Doesn't Use Persistent GPUMatrix Residency

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Open | GPU / Performance | September 7, 2026 | 8-12 hours |

Description:
TD-003 (resolved May 3, 2026 — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md)) added the persistent-residency `GPUMatrix` type and `Matrix::to_gpu()` / `Matrix::from_gpu()` conversions. However, `Matrix::multiply_gpu()` — the path `Matrix::multiply()` dispatches to whenever `GPUManager::is_available()` — still allocates, uploads, computes, and downloads on every single call, the exact per-operation round-trip TD-003 set out to eliminate. Nothing in the training/inference hot path (`EncoderDecoderModel`, `ChatbotTrainer`) calls `to_gpu()`/`from_gpu()` to keep model weights resident across a chain of GPU operations, so TD-003's intended performance win was never realized outside of code that explicitly opts in to the persistent-residency API.

Discovered during the per-file production-readiness rollout (September 7, 2026) — see [file-status-standard.md](file-status-standard.md).

Action Items:

- [ ] Identify the hot-path call chains (training forward/backward, `EncoderDecoderModel` layers) where consecutive `Matrix` operations run back-to-back on the same data.
- [ ] Convert those chains to stay on `GPUMatrix` across the chain via `to_gpu()`/`from_gpu()` instead of one round-trip per op.
- [ ] Benchmark before/after on a representative training step to confirm the reduction in host↔device transfers.

Files to Modify:

- `src/Matrix.cpp` / `src/Matrix.hpp`
- `src/EncoderDecoderModel.cpp`
- `src/ChatbotTrainer.cpp` (call sites)

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

- `src/IncrementalTrainer.cpp` - Modify `create_qa_pairs_from_text()` (line 920)
- `src/BPETokenizer.cpp` - Add FIM special tokens to vocabulary
- `src/IncrementalTrainer.hpp` - Add FIM configuration options

Code Location:

`src/IncrementalTrainer.cpp:920`

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
| LOW | Open | RLHF / PPOOptimizer | September 7, 2026 | 6-10 hours |

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

Discovered during the per-file production-readiness rollout (September 7, 2026) — see
[file-status-standard.md](file-status-standard.md).

Action Items:

- [ ] Recompute `new_log_prob` via a real forward pass of the current policy over `batch_states[i]`,
  not a copy of `batch_old_log_probs[i]`.
- [ ] Track actual per-minibatch KL divergence between old and current policy for `approx_kl`
  instead of the hardcoded `0.0f`.
- [ ] Add a test that trains on a toy environment/reward and asserts the policy actually changes
  (a no-op ratio would previously have passed any test that doesn't check this).

Files to Modify:

- `src/PPOOptimizer.hpp`
- `tests/phase5_test.cpp`

---

## Resolved Items

32 items resolved. See [archive/TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md) for full details.

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
|Medium|3|43%|
|Low|4|57%|

**Total Active Items:** 7

### By Component

|Component|Count|
|----------------------|-------|
|Training / Data Generation|1|
|Tooling / Toolchain|1|
|Tests / RAGInference|1|
|GPU / Inference / Training|1|
|GPU / Performance|1|
|Build / Windows / Metrics|1|
|RLHF / PPOOptimizer|1|

### Effort Distribution

|Effort Range|Count|
|--------------|-------|
|0-2 hours|1|
|2-4 hours|1|
|4-8 hours|1|
|8+ hours|3|
|Not estimated|1|

**Total Estimated Effort (Active Items):** 43-64 hours (excludes TD-014, which has no effort estimate)

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
