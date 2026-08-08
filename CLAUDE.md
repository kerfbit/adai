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
  └── ModelNameClient → mns_server  (begin_run / set_training / push_progress / set_candidate)
```

`IncrementalConfig` is separate from `ServiceConfig`. `IncrementalTrainer::make_incremental_config(svc)` maps `ServiceConfig` → `IncrementalConfig`; any new config field added to `ServiceConfig` must also be added to `IncrementalConfig` and mapped there. `IncrementalConfig::dataset` (a `DatasetConfig`) is populated the same way — `IncrementalTrainer`'s 3-arg constructor (the one `incremental_trainer`'s `train`/`retrain`/`resume`/`reset` commands all use) copies it into `dataset_config_`, so `resume_last_session()`/`reset_all()` see the real `REGISTRY_SERVER_URL`/`MODEL_NAME` instead of an all-default `DatasetConfig`.

### MNS/registry-authoritative run and session numbering

**MNS and the dataset registry are the definitive source for run/session identity and training progress
whenever they're configured and reachable — client-local values are only a fallback for when they're
unavailable** (`NAME_SERVICE_URL`/`REGISTRY_SERVER_URL` unset at startup), the same standard used for
[MNS-authoritative architecture](#configuration) above. A single transient mid-run request failure
still just surfaces as a warning/no-op — it does **not** trigger a silent fallback to local files, since
swapping data sources mid-run risks duplicate/inconsistent training across a distributed pool.

- **`IncrementalTrainer::begin_run(is_retrain)`** — called once per `train`/`retrain`/`resume` invocation,
  before any data is acquired. Calls `ModelNameClient::set_training(model_name, new_run)`, which now
  returns the **server-allocated** `run_id` instead of taking a client-supplied one: `"run-01"` the first
  time a model ever trains (regardless of `new_run`), incrementing only when `new_run=true` (`retrain`;
  `train`/`resume` continue the current run). That same `run_id` is then used both for the MNS training
  lock/history *and* as the `run_id` passed to `DatasetRegistry::acquire_pending`/`mark_trained`/
  `release_pending` — one canonical identifier across both systems, instead of the two unrelated ones
  used before.
- `begin_run` also calls `DatasetRegistry::next_session(model_name, run_id)` →
  `registry_server`'s `POST /registry/{group}/session/next` (or `LocalTransport`'s equivalent local
  counter in standalone mode), which allocates `"session-01"`, `"session-02"`, … per `(model_name, run_id)`
  pair — a run_id never seen before naturally starts its session counter at 1, so a new MNS-allocated run
  resets it for free.
- **Crash-safe progress**: after every epoch, `IncrementalTrainer` calls
  `ModelNameClient::push_progress(model_name, run_id, session_id, epoch, loss, best_loss)` →
  `PUT /models/{name}/progress`, so a `SIGKILL`'d or crashed trainer still leaves an accurate last-known
  state on the MNS record (`progress.epoch`/`progress.loss`/`progress.best_loss`). If a new
  `set_training` call finds the record still `"training"` under a previous (never-completed) run, MNS
  archives that snapshot into `training_history` tagged `"incomplete":true` before allocating the new run
  — see `ModelNameService::handle_state_transition`'s `"training"` branch.

### Metrics Session Lifecycle

The dashboard discovers which metrics session belongs to a model via the `metrics_session_key` field stored in the MNS model record. This link is written by `mns_client_->set_training(model_name, new_run, session_key)`. Clear stale MNS locks with `mns-cli` or `PUT /models/{name}/state`.

Sessions are marked stale after `METRICS_STALENESS_THRESHOLD_SECONDS` (default: 60) of no metric ingest — `effective_is_training` becomes false and the dashboard drops the session. `MetricsPushClient` sends a `/heartbeat` every `METRICS_HEARTBEAT_INTERVAL_MS` (default: 30 000 ms) when idle to prevent this during pre-processing.

### Distributed Dataset Registry

`DatasetRegistry` selects its transport at construction from `ServiceConfig`:
- `registry_server_url` non-empty → `RemoteTransport` (HTTP to `registry_server`)
- Otherwise → `LocalTransport` (flat files: `data_registry.txt`, `pending_files.txt`)

`dataset_manager status` calls `load_pending` (returns all entries). The trainer calls `acquire` (returns only untrained entries — `run_id` empty — that are also unassigned or assigned to the caller's own `model_name`; a caller with no `model_name` can only claim unassigned entries, never one assigned to a specific model). A file visible in `status` but not acquired means its `run_id` is still set from a previous run, or it's assigned to a different model (`dataset_manager assign`/`unassign`). Release with `POST /registry/{group}/release {"run_id":"","files":[...]}` (empty `run_id` bypasses the owner check).

## Configuration

Config files use `KEY=VALUE` format, parsed by the single `ServiceConfig`/`ConfigLoader` in
`src/Config.{hpp,cpp}`. Loading priority: **env vars → config file → hardcoded defaults**; explicit CLI
flags (where a binary has them) override all three. Client binaries (`chatbot_api_server`,
`incremental_trainer`) additionally hot-reload their file via `SIGHUP`.

Configuration is split into 5 service-scoped files, each read only by the binaries that need those
keys (non-architecture keys used by more than one binary, e.g. GPU/MNS-client settings, are duplicated
across files rather than shared):

| File | Read by | Covers |
|---|---|---|
| `config.chatbot.conf` | `chatbot_api_server`, `chatbot_gui` | server/log, architecture (fallback), generation, RAG, GPU, MNS client |
| `config.trainer.conf` | `incremental_trainer` | architecture (fallback), training hyperparameters, GPU, metrics client, MNS client, registry client |
| `config.metrics.conf` | `metrics_api_server` | metrics daemon: port, persistence, DB backend, session registry limits |
| `config.mns.conf` | `mns_server` | MNS daemon: port, data dir, registry proxy target |
| `config.registry.conf` | `registry_server`, `dataset_manager` | registry daemon: port, data dir, FTP server; registry client keys |

Each binary discovers its file via `ConfigLoader::discover_config_path(explicit, service_filename)`
(`src/Config.cpp`): `--config <path>` > `./config.<service>.conf` > `/etc/adai/config.<service>.conf` >
`./config.conf` (legacy, pre-split) > `/etc/adai/config.conf` (legacy). The legacy fallback exists so
old single-file deployments keep working; new deployments should use the per-service files.

**Model architecture is MNS-authoritative.** `D_MODEL`, `NUM_HEADS`, `D_FF`, `NUM_ENCODER_LAYERS`,
`NUM_DECODER_LAYERS`, `MAX_SEQ_LENGTH` in `config.chatbot.conf`/`config.trainer.conf` are only a
fallback. When `NAME_SERVICE_URL` + `MODEL_NAME` resolve successfully, `chatbot_api_server` and
`incremental_trainer` overwrite these 6 fields from the model's `ModelRecord` in MNS
(`ModelNameClient::get_architecture()`) at startup — this is what keeps a chatbot and its trainer in
checkpoint-compatible lockstep without hand-syncing two files. Architecture is set once at
`mns_cli register` and is immutable thereafter (no update endpoint — changing it would break the
checkpoint). The local values are only used standalone (no MNS) or to bootstrap a not-yet-registered
model.

Other architecturally significant keys:

| Key | Notes |
|---|---|
| `BATCH_SIZE` | Samples per gradient update; increase for GPU efficiency |
| `GPU_ENABLED`, `GPU_DEVICE_ID`, `GPU_MEMORY_FRACTION`, `GPU_STRATEGY` | GPU control |
| `METRICS_SERVER_URL` | URL of `metrics_api_server`; empty = no push |
| `METRICS_STALENESS_THRESHOLD_SECONDS` | Seconds idle before dashboard drops session (default: 60) |
| `METRICS_HEARTBEAT_INTERVAL_MS` | Idle heartbeat period from trainer (default: 30 000) |
| `NAME_SERVICE_URL`, `MODEL_NAME`, `MODEL_ROLE` | MNS connection |
| `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID` | Distributed dataset registry |
| `REGISTRY_LISTEN_PORT`, `REGISTRY_DATA_DIR` | `registry_server`'s own listen port / data dir (server-side, distinct from the client-side `REGISTRY_SERVER_URL` above) |

### Daemon admin config API

`metrics_api_server`, `mns_server`, and `registry_server` now also load their `config.<service>.conf`
at startup and expose `GET`/`PUT /admin/config` for a documented allow-list of settings that are safe
to change without restarting (things baked into an already-open socket, DB handle, or FTP listener —
e.g. `port`, `data_dir`, `db_path` — are excluded and file/CLI-only). `PUT` writes accepted keys to a
`daemon_config` SQLite table (`src/DaemonConfigStore.{hpp,cpp}`, one `daemon_config.db` per daemon,
isolated from that daemon's own data store) which overlays the config file on the next restart. The
endpoint is gated behind `--admin-enabled` (`registry_server`, `mns_server`) or the existing
`METRICS_API_ALLOW_CONTROL` (`metrics_api_server`), all default-true.

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
