# Technical Debt Tracker

This document tracks all known technical debt items, TODOs, and improvement opportunities in the ADAI codebase. Items are prioritized and linked to GitHub issues for tracking.

## Overview

**Last Updated:** May 21, 2026
**Total Items:** 3
**High Priority:** 0
**Medium Priority:** 2
**Low Priority:** 1
**Future Enhancements:** 19
**Resolved Items:** 31

## Table of Contents

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Active Technical Debt](#active-technical-debt)
  - [TD-029: Fix GCC 13 ICE in raginference\_test.cpp](#td-029-fix-gcc-13-ice-in-raginference_testcpp)
  - [TD-020: Persistent Metrics Storage via SQL Database](#td-020-persistent-metrics-storage-via-sql-database)
  - [TD-014: LLM Operations and Training Tooling Suite](#td-014-llm-operations-and-training-tooling-suite)
  - [TD-006: Fill-in-the-Middle (FIM) Training Data Generation](#td-006-fill-in-the-middle-fim-training-data-generation)
- [Resolved Items](#resolved-items) (31 items — see [archive](../archive/TECHNICAL_DEBT_RESOLVED.md))
- [Future Improvements](#future-improvements)
  - [Performance Optimizations](#performance-optimizations)
  - [Code Quality](#code-quality)
  - [Developer Experience](#developer-experience)
  - [Configuration and Service Management](#configuration-and-service-management)
  - [Logging and Observability](#logging-and-observability)
  - [Container and Deployment](#container-and-deployment)
- [Process Guidelines](#process-guidelines)
  - [Adding New Technical Debt](#adding-new-technical-debt)
  - [Prioritization Criteria](#prioritization-criteria)
  - [Resolving Technical Debt](#resolving-technical-debt)
- [Statistics](#statistics)
  - [By Priority](#by-priority)
  - [By Component](#by-component)
  - [Effort Distribution](#effort-distribution)
  - [Future Enhancements Summary](#future-enhancements-summary)
- [References](#references)

## Active Technical Debt

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

### TD-020: Persistent Metrics Storage via SQL Database

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Planned | Training / Metrics / API / Infrastructure | May 31, 2026 | 20-30 hours |

Description:
The metrics stack persists all data to flat JSONL/JSON files per session. This prevents time-range queries, limits history served via the API to the 10,000-record in-memory ring buffer, and makes cross-session analytics impossible without reading multiple files externally. A SQL persistence layer (SQLite by default, PostgreSQL as a compile-time option) would remove these constraints while an `IMetricsDatabase` abstraction keeps both backends interchangeable.

Proposal: `docs/development/proposals/persistent-metrics-sql-storage.md`

Action Items:

- [ ] Define `IMetricsDatabase` interface and `SessionRecord` struct in `src/MetricsDatabase.hpp`.
- [ ] Implement `SQLiteMetricsDatabase` with WAL mode and prepared statements (`src/SQLiteMetricsDatabase.hpp/.cpp`).
- [ ] Implement optional `PostgresMetricsDatabase` with connection pool and retry logic (`src/PostgresMetricsDatabase.hpp/.cpp`), guarded by `ENABLE_POSTGRES_METRICS` CMake option.
- [ ] Add `MetricsDatabaseFactory::create(Config)` to select backend from `METRICS_STORAGE_BACKEND` config key.
- [ ] Wire `IMetricsDatabase*` into `TrainingMetricsService`: dual-write in `persist_metrics()` / `persist_summary()`, DB-first restore in `restore_from_summary()`.
- [ ] Have `MetricsSessionRegistry` own and initialise the database instance; inject into each session at creation.
- [ ] Add four new REST endpoints: time-range history, cross-session metric compare, status-filtered session list, full history export.
- [ ] Bundle SQLite amalgamation at `external/sqlite3/` for Windows/MinGW builds; update `adai/CMakeLists.txt`.
- [ ] Add `METRICS_STORAGE_BACKEND`, `METRICS_DB_PATH`, `METRICS_DB_URL`, `METRICS_DB_POOL_SIZE` to `config.conf`.
- [ ] Write `tests/MetricsDatabaseTest.cpp` covering schema bootstrap, WAL mode, round-trip insert/query, time-range filter, dual-write path, and DB-fallback restore.

Files to Create:

- `src/MetricsDatabase.hpp`
- `src/SQLiteMetricsDatabase.hpp` / `src/SQLiteMetricsDatabase.cpp`
- `src/PostgresMetricsDatabase.hpp` / `src/PostgresMetricsDatabase.cpp`
- `src/MetricsDatabaseFactory.hpp`
- `external/sqlite3/sqlite3.c` / `external/sqlite3/sqlite3.h` (amalgamation)
- `tests/MetricsDatabaseTest.cpp`

Files to Modify:

- `src/TrainingMetricsService.hpp` / `src/TrainingMetricsService.cpp`
- `src/MetricsSessionRegistry.hpp` / `src/MetricsSessionRegistry.cpp`
- `src/TrainingMetricsAPI.hpp` / `src/TrainingMetricsAPI.cpp`
- `src/TrainingMetricsAPIServer.cpp`
- `adai/CMakeLists.txt` / `adai/src/CMakeLists.txt`
- `config.conf` / `config-remote.conf`
- `docs/development/TRAINING_METRICS_API.md`

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
- `include/IncrementalTrainer.hpp` - Add FIM configuration options

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

### TD-018: Multi-Instance Training Metrics Service

| Priority | Status | Component | Created | Effort Estimate |
|----------|--------|-----------|---------|------------------|
| MEDIUM | Planned | Metrics / API / Config / Training | May 21, 2026 | 16-24 hours |

Description:
The current `TrainingMetricsService` is built around a single active training session: one `TrainingMetricsSnapshot`, one `std::atomic<int> current_session_id_`, global file paths, and a `GlobalMetricsService` singleton. Launching a second training job (e.g., a hyperparameter sweep, parallel fine-tune, or multi-GPU node) corrupts metrics of the first job because all callers share the same in-process object and write to the same files.

Full proposal: `docs/development/proposals/multi-instance-metrics-service.md`

Action Items:

- [ ] **Step 1 — Per-session file paths in `TrainingMetricsService`**: Verify that `MetricsServiceConfig` carries all path fields (`metrics_file`, `metrics_summary_file`, `metrics_prometheus_file`, `abnormal_samples_file`) so each session can receive its own derived paths at construction time. No constructor-signature changes needed.
- [ ] **Step 2 — Implement `MetricsSessionRegistry`** (`src/MetricsSessionRegistry.hpp`): new header-only class owning `std::unordered_map<std::string, std::shared_ptr<TrainingMetricsService>> sessions_`, a `std::shared_mutex` reader-writer lock, `size_t max_live_sessions_` (default 16), `create_or_get_session(key)`, `get_session(key)`, `list_sessions()`, and `evict_completed_sessions(max_age_seconds)`. Session key format: `^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$`; reject invalid keys with HTTP 400. Key `"0-default"` maps to legacy file paths for backwards compatibility.
- [ ] **Step 3 — Update `TrainingMetricsAPI`**: Replace the single `TrainingMetricsService*` member with a `MetricsSessionRegistry*`. Register all session-scoped routes under `"/api/sessions/{key}/..."` prefix (using httplib path-param capture or manual prefix matching). Add `GET /api/sessions` (session index) and `GET /api/metrics/aggregate` (cross-session snapshot) endpoints. Preserve all old flat routes (`POST /api/session/start`, `GET /api/metrics/current`, etc.) as backwards-compat aliases mapping to key `"0-default"`, emitting `Deprecation: true` header.
- [ ] **Step 4 — Update `TrainingMetricsAPIServer`**: Construct a `MetricsSessionRegistry` (seeded with the legacy `"0-default"` session from existing `MetricsServiceConfig`) and inject it into `TrainingMetricsAPI` instead of a single `TrainingMetricsService`.
- [ ] **Step 5 — New config keys in `Config`**: Add `METRICS_SESSION_KEY` (string, default `"0-default"`), `METRICS_MAX_LIVE_SESSIONS` (int, default 16), `METRICS_COMPLETED_TTL_SECONDS` (int, default 3600), `METRICS_SWEEP_INTERVAL_SECONDS` (int, default 60). Parse in `src/Config.cpp` alongside existing metrics keys.
- [ ] **Step 6 — Trainer HTTP client changes**: `ChatbotTrainer` and `IncrementalTrainer` read `metrics_session_key` from their config and prefix every HTTP push base URL with `/api/sessions/{key}` (e.g., `POST /api/sessions/42-gpu0/metrics/sample`).
- [ ] **Step 7 — Tests**: Unit tests for `MetricsSessionRegistry` (create, evict, capacity cap, concurrent access, key validation). Integration test: two `TrainingMetricsService` instances writing to different file paths do not interfere.

Files to Create:

- `src/MetricsSessionRegistry.hpp` — new session registry header
- `tests/metrics_session_registry_test.cpp` — unit tests

Files to Modify:

- `src/TrainingMetricsService.hpp` — verify per-session path fields in `MetricsServiceConfig`; remove/replace `GlobalMetricsService` singleton
- `src/TrainingMetricsAPI.hpp` / `src/TrainingMetricsAPI.cpp` — registry injection, session-keyed routes, new endpoints, backwards-compat aliases
- `src/TrainingMetricsAPIServer.cpp` — construct `MetricsSessionRegistry`; inject into `TrainingMetricsAPI`
- `src/Config.hpp` / `src/Config.cpp` — four new config keys
- `src/ChatbotTrainer.hpp` / `src/ChatbotTrainer.cpp` — `metrics_session_key` field; prefix push URLs
- `src/IncrementalTrainer.cpp` — map `metrics_session_key` from `ServiceConfig`; prefix push URLs
- `tests/CMakeLists.txt` — add `metricsSessionRegistryTests` target

---

## Resolved Items

31 items resolved. See [archive/TECHNICAL_DEBT_RESOLVED.md](../archive/TECHNICAL_DEBT_RESOLVED.md) for full details.

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
|Medium|2|67%|
|Low|1|33%|

**Total Active Items:** 3

### By Component

|Component|Count|
|----------------------|-------|
|Training / Data Generation|1|
|Tooling / Toolchain|1|
|Metrics / API / Config / Training|1|

### Effort Distribution

|Effort Range|Count|
|--------------|-------|
|0-2 hours|1|
|2-4 hours|1|
|4-8 hours|1|
|8+ hours|1|

**Total Estimated Effort (Active Items):** 22-32 hours

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

---

## References

- [Process Improvement Plan](PROCESS_IMPROVEMENT_PLAN.md) - Section 10: Technical Debt Items
- [Contributing Guide](docs/guides/contributing.md) - Code quality standards
- [GitHub Issues](https://github.com/yourusername/adai/issues?q=is%3Aissue+label%3Atechnical-debt) - Active debt tracking

---

**Maintenance Note:** This document should be reviewed monthly and updated as items are added or resolved.
