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
| `incremental_trainer` | Online/incremental training with GPU support (CLI only — see `trainer_service` for the always-on service) | — |
| `trainer_service` | Process supervisor (TD-172) that launches `incremental_trainer` as a single-pass child, repeatedly, for the always-on training service; never touches GPU/CUDA/SYCL objects itself | 8084 (admin API proxy, opt-in) |
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

**TD-033** (mostly resolved September 13, 2026): `ChatbotAPI::generate_response()`'s plain path now checks `GPUManager::is_available()` and routes through a new `EncoderDecoderModel::gpu_generate_response_with_strategy()` — a persistent-residency decode path supporting every generation strategy — instead of unconditionally calling the plain CPU `forward()` path. `model_mutex_` is now held across it (and the pre-existing `gpu_generate_response()`), closing a concurrency gap this exact wiring would otherwise have reintroduced (encoder/decoder/lm_head's own `gpu_forward()` calls write through to persistent per-instance `GPUState`, shared across concurrent chat requests — the same class of bug TD-156 fixed for the CPU path). Only the before/after latency benchmark remains open, blocked on real GPU hardware not available in the environment this was developed in.

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

**TD-172 (September 14, 2026):** the always-on training service is `trainer_service`, a separate
binary from `incremental_trainer` — not a top-level command of it (the old `incremental_trainer
serve` command has been removed). `trainer_service` is a thin process supervisor: it launches
`incremental_trainer --foreground --admin-port <N> resume` as a single-pass child, waits for it to
exit, and launches another immediately if that pass did work (more may be pending) or after a poll
interval (default 45s) if not — the same idle-poll loop shape `serve` used to run in-process.
`trainer_service` never links `adai_models`/`adai_nlp` and never touches GPU/CUDA/SYCL objects at
all; every actual training pass runs inside the child. This is what lets a GPU-driver crash during
a pass take down only that child process — `trainer_service` itself (and its admin API, which
proxies to whichever child is alive) keeps running and launches a fresh child on the next poll
cycle. See "Incremental trainer admin API" below for how the admin surface works across this
process boundary, and [TECHNICAL_DEBT.md](docs/development/guides/TECHNICAL_DEBT.md)'s TD-172
entry for the full design rationale (why a process supervisor over in-process reuse, why loopback
HTTP over a file-based channel).

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
| `TRAINER_ADMIN_ENABLED`, `TRAINER_ADMIN_PORT`, `TRAINER_ADMIN_HOST`, `TRAINER_ADMIN_DIR` | `trainer_service`'s admin HTTP API — see below |
| `TRAINER_CHILD_ADMIN_PORT` | Loopback-only port `trainer_service` assigns each single-pass child it launches — see below |

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

**TD-172:** this API's HTTP surface is unchanged, but which process answers it changed —
`trainer_service` hosts the public-facing listener below (`TrainerServiceProxy`), and proxies every
`/admin/*` request to whichever `incremental_trainer --admin-port` child is currently running its own
private `TrainerAdminAPI` (bound to `127.0.0.1:TRAINER_CHILD_ADMIN_PORT`, never exposed directly).
`train`/`retrain`/plain `resume` (no `--admin-port`) remain simple one-shot CLI invocations with no
HTTP server at all — only a `resume` launched by `trainer_service` itself hosts one, for that single
pass's duration. Unlike the three admin daemons elsewhere in this table, this admin port is **opt-in**
(`TRAINER_ADMIN_ENABLED=false` by default) since it's the first thing to open a network port on a host
that previously had none, and it's **genuinely always-on** despite each *child*'s `TrainerAdminAPI`
living only as long as one pass: `trainer_service`'s own listener is bound once at its own startup and
kept alive for the whole process lifetime, transparently proxying to whichever child is alive (or
answering an idle default when none is) — the design point that makes it different from a lighter
"only reachable while a worker process happens to be up between systemd restarts" alternative. See
`src/TrainerControlState.hpp`/`src/TrainerAdminAPI.{hpp,cpp}` (used unchanged, by the child) and
`src/TrainerServiceProxy.{hpp,cpp}`/`src/ChildProcess.{hpp,cpp}`/`src/TrainerServiceMain.cpp` (the
supervisor's own proxy and process-launch/monitor logic).

**TD-173:** pause/resume act at *two* levels, both real. `src/TrainerServiceControlState.hpp` is
`trainer_service`'s own process-lifetime control state (distinct from `TrainerControlState`, which
is reconstructed fresh per child pass) — `paused` there persists across every child launch and is
checked by the supervisory loop *before* starting the next one, the same check `serve`'s own
single, process-lifetime `TrainerControlState` used to make. `/admin/pause`/`/admin/resume` always
set/clear that flag (the real, service-level effect: stop/resume launching passes at all) and
additionally proxy to a live child if one exists (so an in-flight pass also drains/resumes
promptly). Before this fix, pause only drained whichever pass happened to be running and the
supervisor immediately launched a fresh, unpaused child right after — see TECHNICAL_DEBT.md's
TD-173 entry for the full incident writeup. `/admin/status` also reports several
`trainer_service`-only observability fields (`supervisor_paused`, `total_passes_launched`,
`total_passes_did_work`, `total_passes_crashed`, `last_exit_code`, `service_uptime_seconds`) as
additional top-level keys alongside the pre-existing shape — additive only, so existing clients
(the Android ops dashboard's `TrainerStatusDto` already sets `ignoreUnknownKeys = true`) are
unaffected.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness — always answered by `trainer_service` itself, never proxied |
| GET | `/admin/config` | current `auto_save_enabled`/`auto_save_every_samples`/`auto_save_every_minutes`/`max_sessions_to_keep` — read from the current child if one is alive, else from the shared `TRAINER_ADMIN_DIR/daemon_config.db` directly |
| PUT | `/admin/config` | mutate the same four keys; persisted to `TRAINER_ADMIN_DIR/daemon_config.db` either way (via the live child's own overlay, or directly when idle) — either way the next child launched picks it up |
| GET | `/admin/status` | phase (`idle`/`loading_data`/`tokenizing`/`training`/`checkpointing`/`pausing`), run/session identity, epoch/sample/loss progress, real (not hardcoded) `paused`, checkpoint counters, plus `trainer_service`'s own observability fields (see above) |
| POST | `/admin/checkpoint[?wait_ms=N]` | force a checkpoint at the next optimizer-step boundary; 409 if idle (no active pass to checkpoint) |
| POST | `/admin/pause` | sets `trainer_service`'s own persistent pause flag (stops it launching another pass) and, best-effort, drains the current pass (if any) via `ChatbotTrainer::set_abort_flag()`/checkpoint/release-claimed-files — `trainer_service` keeps serving regardless, it does not exit |
| POST | `/admin/resume` | clears the persistent pause flag and wakes the idle-poll sleep immediately, plus resumes a live child if one happens to be paused too |

No HTTP shutdown endpoint exists or is planned — `systemctl stop`/SIGTERM stays the sole way to end the
`trainer_service` process, which in turn forwards a graceful-stop request to whatever child is
currently running, escalating to a force kill (`ChildProcess::stop_and_wait()`) if it doesn't exit
within 10s — protects against a genuinely wedged child (this host's own documented GPU-driver
*hang*, not just crash) even without systemd's own external `TimeoutStopSec` as a backstop. No
companion CLI wraps this API (matches `mns_cli`/`dataset_manager` not wrapping their daemons'
`/admin/config` either) — `curl` is the documented interface, e.g.
`curl -s http://127.0.0.1:8084/admin/status`.

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
| **TD-059** (HIGH) | Fixed September 13, 2026: `MultiHeadAttention`/`CrossAttention` now genuinely split into per-head slices on both CPU and GPU paths (previously every self- and cross-attention call was single-head attention over the full `d_model` width with a mismatched softmax scale). Retraining every existing checkpoint under the new math is still outstanding — needs the user's own training infrastructure, not available in a dev session. |
| TD-050 | GPU-resident KV-cache for autoregressive generation — CPU cache has a known correctness bug; no GPU cache exists at all |
| **TD-033** | Mostly resolved — wired in and concurrency-safe; only the before/after GPU latency benchmark remains, blocked on real hardware |
| TD-014 | Missing standalone tooling (quantization, eval, data-prep binaries) |
| TD-006 | Fill-in-the-Middle (FIM) training data generation not implemented |
| TD-162 | `IntegratedInferenceEngine`/`BatchedInferenceEngine` fulfill a request's `std::promise` before finishing that request's mutex-guarded stats update — a client's `f.wait_for()`/`f.get()` can race `get_stats()` against the still-in-flight counter increment on another thread. Confirmed via a real flake (`total_requests` undercounted by one under full-suite `ctest -j8`). |
| TD-172 | Implemented September 14, 2026: `trainer_service` (process supervisor) replaces `incremental_trainer serve`; see "Incremental trainer admin API" above. Code implemented, unit-tested, and verified end-to-end locally (real training pass, real proxying, clean shutdown) — a live deployed host still needs its own `adai-trainer.service` file and binaries updated to actually cut over. |
| TD-173 | Fixed September 14, 2026 (same day): a review of TD-172's control pattern found pause/resume had no real service-level effect (only drained/proxied to whichever child happened to be alive, so a paused pass was immediately followed by a fresh, unpaused one) and that a wedged child could hang `trainer_service`'s own shutdown indefinitely. `TrainerServiceControlState` (new, process-lifetime) plus `ChildProcess::stop_and_wait()`'s SIGKILL escalation fix both — see "Incremental trainer admin API" above. |
