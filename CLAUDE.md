# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All builds are out-of-source via CMake presets. The build directory is `build/<preset>/`.

```bash
# Configure + build (first time or after CMakeLists changes)
cmake --preset=<preset>
cmake --build --preset=<preset> -j$(nproc)

# Subsequent builds (already configured)
cmake --build --preset=<preset> -j$(nproc)

# Build a specific target
cmake --build --preset=<preset> --target <target>
```

| Preset | Purpose |
|---|---|
| `debug` | Debug symbols, no optimization |
| `release` | Optimized, `-march=native` |
| `portable` | Optimized, `-march=x86-64` — use for binaries deployed to other machines |
| `sycl` | Intel ARC GPU via oneAPI (`icpx`); builds only `incremental_trainer` |
| `gpu` | CUDA GPU; builds only `incremental_trainer` |
| `asan` / `ubsan` / `tsan` | Sanitizer builds (inherit from debug) |
| `coverage` | Coverage instrumentation |

**Key CMake flags** (when configuring without presets):
- `-DENABLE_SYCL=ON` — Intel SYCL backend (mutually exclusive with `ENABLE_GPU`)
- `-DENABLE_GPU=ON` — CUDA backend
- `-DPORTABLE_BUILD=ON` — Disables `-march=native`
- `-DBUILD_TESTING=ON` — Enables Google Test suite
- `-DBUILD_GUI=ON` — Qt5/Qt6 GUI targets

**Code formatting:**
```bash
./scripts/format_code.sh   # runs clang-format on all src/
```

## Running Tests

Tests use Google Test (fetched at build time). Build with `debug` or any non-GPU preset.

```bash
# Run all tests
cd build/debug && ctest --output-on-failure -j$(nproc)

# Run tests matching a pattern
ctest -R "matrixTests" --output-on-failure

# Run a test binary directly (for GTest filter/verbose)
./build/debug/tests/matrixTests --gtest_filter="*Multiply*"
./build/debug/tests/encoderblockTests
```

Test binaries live in `build/<preset>/tests/`. Named after the component: `matrixTests`, `multiheadattentionTests`, `encoderblockTests`, `decoderblockTests`, `optimizerTests`, `tokenizer_test`, `layernormTests`, etc.

For sanitizer testing: `./scripts/run_tests.sh --asan|--ubsan|--tsan|--coverage`

## Architecture

### Executable Targets

| Binary | Purpose | Port |
|---|---|---|
| `chatbot` | Interactive CLI client | — |
| `chatbot_api_server` | REST inference API + session management | 8080 |
| `chatbot_gui` | Qt GUI (optional) | — |
| `incremental_trainer` | Online/incremental training with GPU support | — |
| `metrics_api_server` | Training metrics collection + export | 8081 |
| `registry_server` | Distributed dataset queue coordination | 8082 |
| `mns_server` | Model Name Service: model identity + role registry | 8083 |
| `mns_cli` | CLI for MNS management | — |
| `dataset_manager` | Local dataset queue management | — |
| `vocab_builder` | BPE vocabulary creation | — |

### Static Library Dependency Graph

```
adai_models          (EncoderDecoderModel, ModelSerializer)
  └── adai_transformer (EncoderBlock, DecoderBlock)
        ├── adai_attention  (MultiHeadAttention, CrossAttention)
        ├── adai_feedforward
        └── adai_layers     (LayerNorm, PositionalEncoding, TokenEmbedding)
              └── adai_core (Matrix, Optimizer, Metrics, Registry, Logger, Config)
                    └── adai_gpu  [optional: CUDA or SYCL]

adai_nlp             (BPETokenizer, TextGenerator, ConversationContext)
  └── adai_core

adai_metrics_api     (TrainingMetricsAPI — HTTP routes)
  └── adai_core

adai_mns             (ModelNameService, ModelNameClient)
  └── adai_core
```

### GPU Backend

Two mutually exclusive backends, selected at compile time:

- **SYCL** (`-DENABLE_SYCL=ON`): `src/gpu/sycl/` — Intel ARC via oneAPI `icpx` + oneMKL. Uses an in-order `sycl::queue`; **do not add `.wait()` to intermediate kernels** — the queue enforces ordering without CPU stalls.
- **CUDA** (`-DENABLE_GPU=ON`): `src/gpu/MatrixGPU.cu` — static `libcudart`, dynamic `libcublas`.

Both backends expose the same interface through `src/gpu/MatrixGPU.hpp` (`GPUMatrix`, `GPUMemory<T>`, `GPUManager`). `Matrix.cpp` dispatches to GPU when `GPUManager::is_available()` and matrix dimensions meet a minimum threshold.

**TD-003** (active): GPU memory round-trips. Currently each `Matrix` operation uploads data, runs a kernel, and downloads the result. The `GPUMatrix` persistent-residency type and `Matrix::to_gpu()` / `Matrix::from_gpu()` are the scaffolding for fixing this — model weights should stay device-resident throughout a training run.

**`GPU_STRATEGY`** (config key): `background` (low-priority queue, default) or `full` (normal priority).

### Training Pipeline

```
DatasetRegistry (JSONL pending queue)
  └── [RemoteTransport → registry_server | LocalTransport]
        └── [FTP download if ftp_server_host non-empty]
IncrementalTrainer
  ├── BPETokenizer → token IDs
  ├── ChatbotTrainer (forward + backward + optimizer step)
  │     └── EncoderDecoderModel
  ├── TrainingMetricsService → MetricsPushClient → metrics_api_server
  └── ModelNameClient → mns_server  (set_training / set_candidate states)
```

`IncrementalConfig` is separate from `ServiceConfig`. `IncrementalTrainer::make_incremental_config(svc)` maps `ServiceConfig` → `IncrementalConfig`; any new config field added to `ServiceConfig` must also be added to `IncrementalConfig` and mapped there.

### Metrics Session Lifecycle

The dashboard discovers which metrics session belongs to a model via the `metrics_session_key` field stored in the MNS model record. This link is written by `mns_client_->set_training(model_name, run_id, session_key)`. If that call returns 409 (model locked by a different `run_id`), the link is never written and the dashboard sees no active session. Clear stale MNS locks with `mns-cli` or `PUT /models/{name}/state`.

Sessions are marked stale after `METRICS_STALENESS_THRESHOLD_SECONDS` (default: 60) of no metric ingest — `effective_is_training` becomes false and the dashboard drops the session. `MetricsPushClient` sends a `/heartbeat` every `METRICS_HEARTBEAT_INTERVAL_MS` (default: 30 000 ms) when idle to prevent this during pre-processing.

### Distributed Dataset Registry

`DatasetRegistry` selects its transport at construction from `ServiceConfig`:
- `registry_server_url` non-empty → `RemoteTransport` (HTTP to `registry_server`)
- Otherwise → `LocalTransport` (flat files: `data_registry.txt`, `pending_files.txt`)

`dataset_manager status` calls `load_pending` (returns all entries). The trainer calls `acquire` (returns only unassigned entries where `run_id` is empty). A file visible in `status` but not acquired means its `run_id` is still set from a previous run. Release with `POST /registry/{group}/release {"run_id":"","files":[...]}` (empty `run_id` bypasses the owner check).

## Configuration

`config.conf` uses `KEY=VALUE` format. Loading priority: **env vars → config.conf → hardcoded defaults**. Hot-reload via `SIGHUP`.

Architecturally significant keys:

| Key | Notes |
|---|---|
| `D_MODEL`, `NUM_HEADS`, `D_FF`, `NUM_ENCODER_LAYERS`, `NUM_DECODER_LAYERS`, `MAX_SEQ_LENGTH` | Model architecture — must match checkpoint |
| `BATCH_SIZE` | Samples per gradient update; increase for GPU efficiency |
| `GPU_ENABLED`, `GPU_DEVICE_ID`, `GPU_MEMORY_FRACTION`, `GPU_STRATEGY` | GPU control |
| `METRICS_SERVER_URL` | URL of `metrics_api_server`; empty = no push |
| `METRICS_STALENESS_THRESHOLD_SECONDS` | Seconds idle before dashboard drops session (default: 60) |
| `METRICS_HEARTBEAT_INTERVAL_MS` | Idle heartbeat period from trainer (default: 30 000) |
| `NAME_SERVICE_URL`, `MODEL_NAME`, `MODEL_ROLE` | MNS connection |
| `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID` | Distributed dataset registry |

## Code Conventions

- **Logging**: `adai::Logger::info/warn/error/debug(fmt, args...)` — never `std::cout` in library code.
- **Headers**: `#pragma once`, no implementation in headers except templates.
- **Ownership**: `std::unique_ptr` / `std::shared_ptr`; no raw owning pointers.
- **New component**: `src/Component.{cpp,hpp}` + `tests/component_test.cpp` + register in `src/CMakeLists.txt` and `tests/CMakeLists.txt`.
- **Artifacts**: trained weights go in `training_sessions/` (gitignored); feature proposals in `docs/development/proposals/`.

## Active Technical Debt Tags

The codebase uses `// TD-NNN` comment tags. Notable open items:

| Tag | Description |
|---|---|
| **TD-003** | Persistent GPU-resident matrices — model weights should stay on device; `GPUMatrix` / `to_gpu()` / `from_gpu()` are scaffolding |
| TD-013 | Advanced metrics & outlier detection in `TrainingMetricsService` |
| TD-017 | Adaptive gradient clipping — persist per-epoch clip threshold |
| TD-018 | Replace single `current_session_id_` with per-session registry in `TrainingMetricsService` |
| TD-020 | Cross-session metric comparison from SQLite DB |
| TD-021 | Per-session labelled Prometheus output; `MetricsPushClient` push client |
| TD-023 | Background generation-quality scoring thread |
| TD-028 | Multiple references in metrics / training path (see grep for current locations) |
