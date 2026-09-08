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
| `incremental_trainer` | Online/incremental training with GPU support | 8084 (admin API, `serve` only, opt-in) |
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

**TD-033** (active): `chatbot_api_server` inference never uses persistent GPU-resident decode. Training already has full GPU residency — `ChatbotTrainer::gpu_forward()`/`gpu_backward()` chain `GPUMatrix` end-to-end via TD-003's `to_gpu()`/`from_gpu()`. But `ChatbotAPI::generate_response()` still calls the plain CPU `forward()` path, so `Matrix::multiply_gpu()` uploads, computes, and downloads on every call, for every matmul, for every generated token — the one caller that most needs the existing `EncoderDecoderModel::gpu_generate_response()` persistent-residency path has never been wired to use it.

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

`IncrementalConfig` is separate from `ServiceConfig`. `IncrementalTrainer::make_incremental_config(svc)` maps `ServiceConfig` → `IncrementalConfig`; any new config field added to `ServiceConfig` must also be added to `IncrementalConfig` and mapped there. `IncrementalConfig::dataset` (a `DatasetConfig`) is populated the same way — `IncrementalTrainer`'s 3-arg constructor (the one `incremental_trainer`'s `train`/`retrain`/`resume`/`reset`/`serve` commands all use) copies it into `dataset_config_`, so `resume_last_session()`/`reset_all()` see the real `REGISTRY_SERVER_URL`/`MODEL_NAME` instead of an all-default `DatasetConfig`.

`incremental_trainer serve` (recommended for `adai-trainer.service`, see `scripts/adai-trainer.service`) is a distinct top-level command, not routed through `train`/`retrain`/`resume`'s fork+daemonize path — it never forks and stays alive for the process's entire lifetime, internally looping between checking for pending work and running a pass via `resume_last_session()`'s existing logic once per iteration (default 45s idle-poll interval). This is what lets it host the always-on admin HTTP API below — see "Incremental trainer admin API".

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
| `AUTO_SAVE_ENABLED`, `AUTO_SAVE_EVERY_SAMPLES`, `AUTO_SAVE_EVERY_MINUTES`, `MAX_SESSIONS_TO_KEEP` | Checkpoint cadence / retention — map into `IncrementalConfig`'s matching fields via `make_incremental_config()`; live-tunable under `serve` via `PUT /admin/config`, see below |
| `TRAINER_ADMIN_ENABLED`, `TRAINER_ADMIN_PORT`, `TRAINER_ADMIN_HOST`, `TRAINER_ADMIN_DIR` | `incremental_trainer serve`'s admin HTTP API — see below |

### Daemon admin config API

`metrics_api_server`, `mns_server`, and `registry_server` now also load their `config.<service>.conf`
at startup and expose `GET`/`PUT /admin/config` for a documented allow-list of settings that are safe
to change without restarting (things baked into an already-open socket, DB handle, or FTP listener —
e.g. `port`, `data_dir`, `db_path` — are excluded and file/CLI-only). `PUT` writes accepted keys to a
`daemon_config` SQLite table (`src/DaemonConfigStore.{hpp,cpp}`, one `daemon_config.db` per daemon,
isolated from that daemon's own data store) which overlays the config file on the next restart. The
endpoint is gated behind `--admin-enabled` (`registry_server`, `mns_server`) or the existing
`METRICS_API_ALLOW_CONTROL` (`metrics_api_server`), all default-true.

### Incremental trainer admin API

`incremental_trainer serve` — not `train`/`retrain`/`resume` — is the only command that hosts this;
those three remain simple one-shot CLI invocations with no HTTP server at all. Unlike the three admin
daemons above, `serve`'s admin port is **opt-in** (`TRAINER_ADMIN_ENABLED=false` by default) since it's
the first thing to open a network port on a host that previously had none, and it's **genuinely
always-on**: bound once at `serve` startup and kept alive for the whole process lifetime, independent of
any single training pass — the design point that makes it different from a lighter "control-file" or
"only reachable while a worker process happens to be up between systemd restarts" alternative. See
`src/TrainerControlState.hpp`/`src/TrainerAdminAPI.{hpp,cpp}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/admin/config` | current `auto_save_enabled`/`auto_save_every_samples`/`auto_save_every_minutes`/`max_sessions_to_keep` |
| PUT | `/admin/config` | mutate the same four keys; persisted to `TRAINER_ADMIN_DIR/daemon_config.db` |
| GET | `/admin/status` | phase (`idle`/`loading_data`/`tokenizing`/`training`/`checkpointing`/`pausing`), run/session identity, epoch/sample/loss progress, `paused`, checkpoint counters |
| POST | `/admin/checkpoint[?wait_ms=N]` | force a checkpoint at the next optimizer-step boundary; 409 if idle (no active pass to checkpoint) |
| POST | `/admin/pause` | drain the current pass (if any) via `ChatbotTrainer::set_abort_flag()`, checkpoint, release claimed files back to pending, return to idle — the supervisory loop keeps serving, it does not exit |
| POST | `/admin/resume` | clear pause, wake the idle-poll sleep so pending work is checked immediately |

No HTTP shutdown endpoint exists or is planned — `systemctl stop`/SIGTERM stays the sole way to end the
process, unchanged from `train`/`retrain`/`resume`. No companion CLI wraps this API (matches
`mns_cli`/`dataset_manager` not wrapping their daemons' `/admin/config` either) — `curl` is the
documented interface, e.g. `curl -s http://127.0.0.1:8084/admin/status`.

## Code Conventions

- **Logging**: `adai::Logger::info/warn/error/debug(fmt, args...)` — never `std::cout` in library code.
- **Headers**: `#pragma once`, no implementation in headers except templates.
- **Ownership**: `std::unique_ptr` / `std::shared_ptr`; no raw owning pointers.
- **New component**: `src/Component.{cpp,hpp}` + `tests/component_test.cpp` + register in `src/CMakeLists.txt` and `tests/CMakeLists.txt`; new files start tagged `@adai-status: experimental`, `@adai-version: 0.1.0` (see below).
- **Artifacts**: trained weights go in `training_sessions/` (gitignored); feature proposals in `docs/development/proposals/`.
- **File status tag**: every in-scope file (`src/`, Android `src/main`, `tizen-metrics-app/js/`, operational `scripts/`) carries a header comment block — `@adai-status` (`experimental`/`beta`/`stable`/`deprecated`/`legacy`), `@adai-version` (per-file SemVer, `stable` requires `MAJOR >= 1`), `@adai-reviewed` (ISO date). See [file-status-standard.md](docs/development/guides/file-status-standard.md); validate with `./scripts/check_file_status.py --changed` and regenerate the dashboard with `./scripts/gen_status_report.py`.

## Active Technical Debt Tags

The codebase uses `// TD-NNN` comment tags, though most that remain inline today cite
already-resolved items kept as historical design-rationale footnotes, not pending work — check
[TECHNICAL_DEBT.md](docs/development/guides/TECHNICAL_DEBT.md) for what's actually open rather than
trusting a `grep TD-NNN` alone. Currently active items:

| Tag | Description |
|---|---|
| TD-050 | GPU-resident KV-cache for autoregressive generation — CPU cache has a known correctness bug; no GPU cache exists at all |
| **TD-033** | `chatbot_api_server` inference never uses the persistent GPU-resident decode path — training already does |
| **TD-034** | `PPOOptimizer::train()`'s ratio/KL terms are a placeholder, and `ValueFunction::update()` never writes its computed gradient into the weight update — the value function's weights never change |
| TD-032 | SQLite amalgamation not bundled for Windows/MinGW cross-compilation |
| TD-014 | Missing standalone tooling (quantization, eval, data-prep binaries) |
| TD-006 | Fill-in-the-Middle (FIM) training data generation not implemented |
