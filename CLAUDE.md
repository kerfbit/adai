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

**TD-202: per-trainable-piece sub-pools (`dataset_kind`).** Within one `run_group`, each
trainable piece of the overall model system gets its own physical sub-pool — `data_dir/<group>/<kind>/`
server-side (its own `data_registry.txt`/`pending_files.txt`), or `<session_dir>/<kind>/` for
`LocalTransport` — instead of one shared pool with no way to separate, say, `--objective=lejepa`'s
world-model pretraining data from plain chatbot fine-tuning pairs (they used to draw from the exact
same pending pool with no isolation at all). `kind` reuses MNS's own vocabulary
(`encoder`/`decoder`/`world_model`/`chatbot`, TD-196); empty (the default) reproduces every
pre-TD-202 group's behavior byte-for-byte. `RemoteTransport` encodes it as a URL segment
(`/registry/{group}/{kind}/...`, added via a route regex's optional capture group — an
unrecognized kind value simply can't match any route, so it 404s rather than needing its own
validation check); `DatasetConfig::dataset_kind`/`ServiceConfig::dataset_kind` (config key
`DATASET_KIND`) carry it from config/env through to `DatasetRegistry::build_transport()`.
`incremental_trainer` sets it automatically from the training objective (`--objective=lejepa` →
`"world_model"`, every other objective → `"chatbot"`) right after `svc_config` loads, **unconditionally
— a config-file/env `DATASET_KIND` has no effect on this binary's automatic selection** (TD-204: a
static config value can't vary between a plain `train` and a `--objective=lejepa` run against the
same `config.trainer.conf`, so letting it win would silently reunite the two objectives' pools —
only the per-invocation `--dataset-kind <kind>` flag can override the automatic default).
`dataset_manager` gets a global `--kind <kind>` flag (selects which
sub-pool every command in that invocation targets, and correctly reads `DATASET_KIND` from config
when `--kind` isn't passed, since it has no per-objective mapping to protect) and a
`migrate <kind> [file ...] [--count N]`
command for moving already-queued legacy (unkinded) pending files into a kind's own sub-pool —
the one-time upgrade path for data queued before this feature existed (a migrated entry's model
assignment, if any, is restored via a follow-up `assign` call, since the underlying add-pending
wire format has no model_name field of its own).

**TD-205: row-range segments.** Within one physical file, a `PendingEntry`/`FileToken` can now
carry `segment_start`/`segment_count` (both `-1` = whole file, the default — every pre-TD-205
entry/caller is unaffected byte-for-byte). This makes one file's pairs independently
addressable/assignable/deletable in row-range slices instead of only as a whole unit — the
primitive behind both "use only part of a large dataset" and staged/incremental release (queue a
file once, then unassign its held-back segments to training in stages, reusing the ordinary
assign/unassign primitives — no new status/state machine). Identity for dedup/lookup is
`(path, segment_start, segment_count)`, not bare `path`; a `SegmentTarget{path, segment_start,
segment_count}` addresses one exact segment via an additive `segments` field alongside every
existing `paths`-based request (assign/unassign/delete/release), so no pre-TD-205 caller needed to
change. Segmentation only applies to JSONL files (range-sliced by *parsed pair count*, matching
`DatasetRegistry::count_pairs()`); a legacy `INPUT:`/`RESPONSE:` file can't be cheaply sliced by
line-number arithmetic and always falls back to whole-file loading regardless of a requested
range. `dataset_manager segment <path> [--count N | --ranges A-B,C-D,...]` (0-based inclusive
ranges) creates the segments; `status`/`list-pending` display a segment's range when present. The
Android ops dashboard's Registry section (its own "Create segments" dialog) is the primary
consumer — see below.

**TD-205: ops dashboard dataset management segment.** The Android app's Registry tab (previously a
thin, kind-unaware CRUD screen) is now a full planning-and-management segment: a pool-health
overview across every kind (pending/trained counts, total samples, independent of the currently
selected filter), kind filter chips within the group detail screen, full CLI parity (unassign,
delete, manual-add, upload, migrate-to-kind, create-segments — previously missing entirely), and
every mutating action (including assign and both Gutenberg/HuggingFace fetch dialogs, previously
ungated) now goes through the same `ConfirmActionDialog` biometric/PIN gate and literal HTTP-call
preview force-release already used. `migrateToKind()`/`createSegments()` in
`RegistryRepository.kt` are client-side composites of existing calls (no new server endpoints),
mirroring `dataset_manager migrate`'s/`segment`'s own exact sequencing.

## Configuration

Config files use `KEY=VALUE` format, parsed by the single `ServiceConfig`/`ConfigLoader` in
`src/Config.{hpp,cpp}`. Loading priority: **env vars → config file → hardcoded defaults**; explicit CLI
flags (where a binary has them) override all three. Only `chatbot_api_server` hot-reloads its file
via `SIGHUP` (and only some keys — see TD-251/TD-285); **every other binary, including
`incremental_trainer` and `trainer_service`, is terminated by `SIGHUP`** (no handler — TD-283).

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

**TD-196: MNS `kind` schema + encoder/decoder/world-model connection standard.** Every MNS
`ModelRecord` has a `kind`: `encoder` | `decoder` | `world_model` | `chatbot` (default `chatbot` —
every pre-TD-196 record keeps working unchanged). `encoder`/`decoder`/`world_model` are
independently-registrable entities, each using the existing 6 architecture fields per the table
below; a `chatbot` is either legacy-bundled (all 6 inline, as before) or a **named pairing** of a
linked encoder + decoder (`connection.encoder_name`/`decoder_name`, set at register time,
immutable — `mns_server` rejects a dimensionally-incompatible pairing with `409`).

| kind | uses | own layer-count field |
|---|---|---|
| `encoder` | `d_model`/`num_heads`/`d_ff`/`max_seq_length` | `num_encoder_layers` (`num_decoder_layers` must be 0) |
| `decoder` | same 4 | `num_decoder_layers` (`num_encoder_layers` must be 0) |
| `world_model` | same 4 | `num_encoder_layers` (standalone encoder; `num_decoder_layers` must be 0) |
| `chatbot` (new-style) | resolved via `connection.encoder_name`/`decoder_name` | — |
| `chatbot` (legacy) | inline, as before TD-196 | both |

A `chatbot`'s `connection` (a JSON blob, `ModelConnection`) also carries the **world-model
connection standard** — everything that used to live only in local
`config.trainer.conf`/`config.chatbot.conf`, kept in sync purely by operator discipline:
`world_model_name` (empty = none attached), `world_model_inject_every_n_layers`, and every
`HIPPOCAMPAL_*` tuning parameter. Mutable via `POST /models/{name}/link-world-model` (unlike
`encoder_name`/`decoder_name` — swapping the world model doesn't break checkpoint compatibility);
`mns_server` rejects a `d_model` mismatch between the chatbot (via its encoder, or its own inline
field for a legacy chatbot) and the target world model with `409` whenever
`world_model_inject_every_n_layers > 0`. A `world_model`-kind record's own `connection.sigreg_lambda`/
`sigreg_num_sketches` are set at register time.

`ModelNameClient::get_architecture()` needed **no signature or call-site change** — for a
new-style chatbot it transparently composes the 6-field `ModelArchitecture` from 2 additional
`GET /models/{name}` calls against the linked encoder/decoder; a legacy chatbot or a standalone
encoder/decoder/world_model record falls back to the inline fields exactly as before. Both
`ChatbotAPIServer.cpp` and `IncrementalTrainingTool.cpp` also now resolve a linked world model's
own artifact path/architecture/connection settings from MNS first (via
`ModelNameClient::get_connection()`/`resolve_model()`/`get_architecture()`), falling back to local
`WORLD_MODEL_*`/`HIPPOCAMPAL_*` config only when unlinked or MNS is unreachable — closing the gap
TD-184 originally flagged. `mns_cli register` gained `--kind`, `--num-layers`, `--encoder`/
`--decoder`, `--sigreg-lambda`/`--sigreg-num-sketches`; `mns_cli link-world-model` wraps the new
endpoint. See [TECHNICAL_DEBT_RESOLVED.md](docs/development/archive/TECHNICAL_DEBT_RESOLVED.md)'s
TD-196 entry for the full design rationale.

Other architecturally significant keys:

| Key | Notes |
|---|---|
| `BATCH_SIZE` | Samples per gradient update; increase for GPU efficiency |
| `GPU_ENABLED`, `GPU_DEVICE_ID`, `GPU_MEMORY_FRACTION`, `GPU_STRATEGY` | GPU control |
| `METRICS_SERVER_URL` | URL of `metrics_api_server`; empty = no push |
| `METRICS_STALENESS_THRESHOLD_SECONDS` | Seconds idle before dashboard drops session (default: 60) |
| `METRICS_HEARTBEAT_INTERVAL_MS` | Idle heartbeat period from trainer (default: 30 000) |
| `NAME_SERVICE_URL`, `MODEL_NAME`, `MODEL_ROLE` | MNS connection |
| `WORLD_MODEL_ARTIFACT_PATH` | Overrides the default `<session_dir>/world_model` checkpoint directory; normally set automatically from a linked world model's own MNS `artifact.path` (TD-196), not by hand |
| `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID` | Distributed dataset registry |
| `DATASET_KIND` | TD-202: per-trainable-piece sub-pool within `RUN_GROUP` (`encoder`/`decoder`/`world_model`/`chatbot`); empty = legacy shared pool. Read by `dataset_manager` (falls back to it when `--kind` isn't passed); **ignored by `incremental_trainer`** (TD-204), which always derives it from the training objective unless `--dataset-kind` overrides — see "Distributed Dataset Registry" above |
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
- **Artifacts**: trained weights go in `training_sessions/` (gitignored); feature proposals in `docs/proposals/` (listed in its `README.md`).
- **File status tag**: every in-scope file (`src/`, Android `src/main`, `tizen-metrics-app/js/`, operational `scripts/`) carries a header comment block — `@adai-status` (`experimental`/`beta`/`stable`/`deprecated`/`legacy`), `@adai-version` (per-file SemVer, `stable` requires `MAJOR >= 1`), `@adai-reviewed` (ISO date). See [file-status-standard.md](docs/development/guides/file-status-standard.md); validate with `./scripts/check_file_status.py --changed` and regenerate the dashboard with `./scripts/gen_status_report.py`.

## Active Technical Debt Tags

The codebase uses `// TD-NNN` comment tags, though most that remain inline today cite
already-resolved items kept as historical design-rationale footnotes, not pending work — check
[TECHNICAL_DEBT.md](docs/development/guides/TECHNICAL_DEBT.md) for what's actually open rather than
trusting a `grep TD-NNN` alone. Currently active items:

| Tag | Description |
|---|---|
| **TD-059** (HIGH) | Fixed September 13, 2026: `MultiHeadAttention`/`CrossAttention` now genuinely split into per-head slices on both CPU and GPU paths (previously every self- and cross-attention call was single-head attention over the full `d_model` width with a mismatched softmax scale). Retraining every existing checkpoint under the new math is still outstanding — needs the user's own training infrastructure, not available in a dev session. |
| TD-050 | Mostly resolved — CPU cache root-caused clean (no bug found) and a GPU-resident cache (`GPUKVCache`/`GPUDecoderKVCache`) designed and implemented; only on-device validation and the latency benchmark remain, blocked on real GPU hardware |
| **TD-033** | Mostly resolved — wired in and concurrency-safe; only the before/after GPU latency benchmark remains, blocked on real hardware |
| TD-014 | Missing standalone tooling (quantization, eval, data-prep binaries) |
| TD-006 | Fill-in-the-Middle (FIM) training data generation not implemented |
| TD-172 | Implemented September 14, 2026: `trainer_service` (process supervisor) replaces `incremental_trainer serve`; see "Incremental trainer admin API" above. Code implemented, unit-tested, and verified end-to-end locally (real training pass, real proxying, clean shutdown) — a live deployed host still needs its own `adai-trainer.service` file and binaries updated to actually cut over. |
| TD-173 | Fixed September 14, 2026 (same day): a review of TD-172's control pattern found pause/resume had no real service-level effect (only drained/proxied to whichever child happened to be alive, so a paused pass was immediately followed by a fresh, unpaused one) and that a wedged child could hang `trainer_service`'s own shutdown indefinitely. `TrainerServiceControlState` (new, process-lifetime) plus `ChildProcess::stop_and_wait()`'s SIGKILL escalation fix both — see "Incremental trainer admin API" above. |
| TD-211–TD-214 | Filed October 8, 2026: `BatchedInferenceEngine` (`--batched-inference`) only queues and serializes requests — it never batches the model (TD-211, blocked on TD-171 for real batching); one throwing request fails its whole group and shutdown can abandon queued requests (TD-212); its stats are misleading and unexposed (TD-213); batched mode ignores `strategy`/`beam_width` and shares one fixed-seed RNG (TD-214). See [BatchedInferenceEngine.md](docs/development/reference/source/BatchedInferenceEngine.md). Real batching planned as v2 (October 8, 2026): [real_batching_v2_plan.md](docs/proposals/real_batching_v2_plan.md). |
| TD-215, TD-216 | Filed October 8, 2026: `Activation` has two wrong header comments (`softmax_derivative`, `leaky_relu_derivative`) and two unchecked inputs (TD-215); softmax/GELU math is re-typed across CPU hot paths and both GPU backends, so a change in one place can silently cause CPU/GPU drift (TD-216). See [Activation.md](docs/development/reference/source/Activation.md). |
| TD-217, TD-218 | Filed October 8, 2026: `/chat/batch`'s public `stats` describe padding for grouping that never happens (TD-217); `BatchProcessor.hpp` helpers have unused surface, lose input order, and skip input checks (TD-218). See [BatchProcessor.md](docs/development/reference/source/BatchProcessor.md). Decided October 8, 2026: TD-217 relabels the stats as an estimate; TD-218's helpers are kept as groundwork for the v2 plan. (TD-219, `Dataset`'s batch wrappers, closed October 11, 2026 with `Dataset.hpp`'s retirement.) |
| **TD-220, TD-221** (HIGH) | Filed October 8, 2026: BPE merge learning is corrupted by its `"|||"` pair-key separator, so pipe-heavy corpora stop learning merges or repeat a wrong rule; audit existing vocabs (TD-220). `BPETokenizer::save_vocab()` fails silently, so checkpoints can be written without their `.vocab` (TD-221). |
| TD-222–TD-225 | Filed October 8, 2026: never-invalidated `apply_bpe()` thread-local cache (TD-222); `load_vocab()`/`build_vocab()` can leave the tokenizer gutted or with duplicate IDs (TD-223); lossy lowercase/whitespace normalization, owner decision pending (TD-224); tokenizer logging/API hygiene (TD-225). See [BPETokenizer.md](docs/development/reference/source/BPETokenizer.md). |
| **TD-226, TD-227** (HIGH) | Filed October 8, 2026: `/chat/batch` rejects any request whose messages contain `]` (ChatbotAPI's hand-written JSON parser; replace with the vendored nlohmann/json) (TD-226); ChatbotAPI session pointers are used outside the session lock, allowing same-session races and a possible use-after-free (TD-227). |
| TD-228–TD-239 | Filed October 8, 2026: further ChatbotAPI bugs, one TD each: JSON string parsing, invalid JSON output, RAG holding `config_mutex_`, session expiry only on `/health`, unlimited client session IDs, non-CSPRNG IDs, batch endpoints bypassing the dispatcher, no batch size limit, HTTP 200 for errors, non-atomic `running_`, leaked exception text, logging/dead code. See [ChatbotAPI.md](docs/development/reference/source/ChatbotAPI.md). |
| TD-240–TD-245 | Filed October 8, 2026: `chatbot_api_server` CLI/config. Numeric flags parsed with `atoi`/`atof` (`--port abc` → port 0; TD-240); `ConfigLoader::validate()` never runs at startup, only on `SIGHUP` reload (TD-241); reload discards CLI generation overrides (TD-242); every engine-mode flag passed gets started (TD-243); `TOP_K`/`BEAM_WIDTH` parsed but never applied (TD-244); misleading or silent argument errors (TD-245). See [ChatbotApiServerArgs.md](docs/development/reference/source/ChatbotApiServerArgs.md). |
| **TD-246, TD-247** (HIGH) | Filed October 8, 2026: `chatbot_api_server` silently serves a randomly initialized model when `MODEL_PATH` is missing or fails to load, while `/health` reports `ok` (TD-246); RAG retrieval embeds with an untrained `LLMEncoder` because its weights are never loaded (TD-247). |
| TD-248–TD-252 | Filed October 8, 2026: shutdown during server startup is lost and the process hangs (TD-248); `SIGHUP` handler isn't async-signal-safe (TD-249); hippocampal memory saved only on graceful shutdown, paths can land in the working directory (TD-250); reload silently ignores most settings (TD-251); inaccurate startup log/usage (TD-252). See [ChatbotAPIServer.md](docs/development/reference/source/ChatbotAPIServer.md). |
| **TD-253** (HIGH) | Filed October 8, 2026: the `chatbot` CLI loops forever at end of input (Ctrl+D or piped stdin), emitting ~95 MB/s of prompts and never auto-saving. |
| TD-254–TD-259 | Filed October 8, 2026: `chatbot` `/set` settings are sent but ignored by the server (TD-254); `success:false` replies shown as empty (TD-255); help banner gaps (TD-256); unescaped `session_id` (TD-257); save loss/overwrite (TD-258); code hygiene (TD-259). See [ChatbotCLI.md](docs/development/reference/source/ChatbotCLI.md). |
| **TD-260, TD-261** (HIGH) | Filed October 9, 2026: `chatbot_gui` never loads a trained checkpoint, because it checks for a file at the bare model path, which `save_model()` never writes (TD-260); `EncoderDecoderModel::generate_response_with_strategy()` forces beam search whenever `num_beams > 1`, overriding the strategy in `chatbot_gui` (default 5) and `RLHFTrainer` (header default 4) (TD-261). |
| TD-262–TD-266 | Filed October 9, 2026: `chatbot_gui` architecture from local config only / unusable on init failure (TD-262); UI freezes during generation (TD-263); launcher hardcodes Qt5/x86_64 paths (TD-264); prompt format differs from the server (TD-265); minor gaps (TD-266). See [ChatbotGUI.md](docs/development/reference/source/ChatbotGUI.md). |
| **TD-267** (HIGH) | Filed October 9, 2026: `ChatbotTrainer` keeps pairs that `preprocess_data()` skipped (empty/invalid UTF-8) as empty entries: training throws on each and drops its accumulation window, and validation counts each as loss 0, biasing validation loss low (verified). |
| TD-268–TD-276 | Filed October 9, 2026: `ChatbotTrainer` `end_epoch()` only with validation data (TD-268); optimizer state and LR schedule restart every incremental pass (TD-269); adaptive clipping resets every epoch (TD-270); cache vs. random split (TD-271); mid-window/last-window gradient loss (TD-272); LOW cleanup (TD-273–276). See [ChatbotTrainer.md](docs/development/reference/source/ChatbotTrainer.md). |
| TD-277–TD-282 | Filed October 9, 2026: `ChildProcess` (`trainer_service`) reports every signal death as 128, losing SIGSEGV vs SIGKILL (TD-277); a failed `exec` looks like a successful launch, then a silent 127 retry loop (TD-278); latent `kill(-1, SIGTERM)` under concurrent `request_stop()` (TD-279); LOW: orphaned passes, fork/fd hygiene, Windows gaps (TD-280–282). See [ChildProcess.md](docs/development/reference/source/ChildProcess.md). |
| **TD-283, TD-284** (HIGH) | Filed October 9, 2026: `SIGHUP` terminates every binary except `chatbot_api_server` (TD-283); `registry_server` ignores the FTP/FTPS security keys `config.registry.conf` ships with (`FTPS_ENABLED`, `FTP_DATA_SERVER_SECRET`…), which are CLI-only (TD-284). |
| TD-285–TD-290 | Filed October 9, 2026: `ConfigLoader::reload()` succeeds without applying untracked changes (TD-285); two boolean parsers (TD-286); negative/partial numbers accepted (TD-287); `validate()` gaps (TD-288); inline comments/enum typos (TD-289); output hygiene (TD-290). See [Config.md](docs/development/reference/source/Config.md). |
| **TD-291–TD-296** (TD-291/292 HIGH) | Filed October 9, 2026: conversation imports replace session limits and trust stored token counts, so `/chat/session/import` can make a session unbounded (TD-291); one message over 1.2× the token budget evicts the whole conversation → HTTP 500 (TD-292); malformed import wipes the session (TD-293); token budget unrelated to model `max_seq_length` (TD-294); spoofable role labels (TD-295); hygiene (TD-296). See [ConversationContext.md](docs/development/reference/source/ConversationContext.md). |
| TD-297–TD-303 | Filed October 10, 2026: layer `update_weights()` steps a shared optimizer once per layer, so `EncoderDecoderModel::train_step()` over-steps 6–10× when an optimizer is registered (shipped trainers call `step()` once and are unaffected) (TD-297); `CrossAttention::load()` loads truncated files as NaN (TD-298); LOW: SIGFPE on `num_heads == 0` (TD-299), stale cached encoder K/V (TD-300), GPU/CPU divergence (TD-301), LoRA persistence/grad norm (TD-302), decoder-only dummy cross-attention (TD-303). See [CrossAttention.md](docs/development/reference/source/CrossAttention.md). |
| TD-304–TD-308 | Filed October 11, 2026: `DaemonConfigStore` has no SQLite busy timeout and standalone (`LocalTransport`) session numbers use a non-atomic read-modify-write, so concurrent writers fail instantly and two processes can get the same session number (TD-304, verified); a rejected or failed `PUT /admin/config` still partly applies, in memory or in the DB (TD-305); LOW: no range checks on `registry_server`/`metrics_api_server` PUTs (TD-306), hidden `load_all()` read errors (TD-307), counters reuse the config table (TD-308). See [DaemonConfigStore.md](docs/development/reference/source/DaemonConfigStore.md). |
| **TD-309, TD-310** (HIGH) | Filed October 11, 2026: `DataFetcher`'s Gutenberg sentence regex segfaults on long unpunctuated text, so one `fetch/gutenberg` request can crash `registry_server` (TD-309, verified); HuggingFace `split` may contain `..`, so `fetch/huggingface` writes outside the registry data dir (TD-310, verified). |
| TD-311–TD-314 | Filed October 11, 2026: Gutenberg pairs are next-sentence non-sequiturs, owner decision on intended task pending (TD-311); downloads via shell `curl` with `HF_TOKEN` in argv and no timeouts (TD-312); partial/concurrent HF downloads cached as complete (TD-313); source coverage/hygiene (TD-314). See [DataFetcher.md](docs/development/reference/source/DataFetcher.md). |
| TD-315–TD-320 | Filed and resolved October 11, 2026: `src/Dataset.hpp` (tagged `stable`, but used only by tests/examples) was **retired** along with its tests and two examples, closing TD-315–TD-320 and TD-219 (see [TECHNICAL_DEBT_RESOLVED.md](docs/development/archive/TECHNICAL_DEBT_RESOLVED.md)). Its capabilities with no other implementation (k-fold, stratified splits, augmentation/preprocessing hooks, multi-format import, lazy indexing) are listed under TD-014 for possible recreation; [Dataset.md](docs/development/reference/source/Dataset.md) is kept as a retired reference. |
| TD-321–TD-325 | Filed October 11, 2026: `dataset_manager assign`/`migrate` treat `--count` ≤ 0 as "all", so a typo like `--count -2` reassigns or migrates the whole pending pool (TD-321, MEDIUM, verified); LOW: unknown flags become file targets (TD-322), partial/non-positive numbers accepted (TD-323), `segment --ranges` overflow/overlap (TD-324), `huggingface` positional arguments shift (TD-325). See [DatasetManagerArgs.md](docs/development/reference/source/DatasetManagerArgs.md). |
| **TD-326** (HIGH) | Filed October 11, 2026: `dataset_manager migrate` drops TD-205 segment ranges and ignores failed source deletes, so a segmented legacy file lands whole in the new pool while every segment stays in the legacy pool, which trains it twice (verified). |
| TD-327–TD-332 | Filed October 11, 2026: `dataset_manager remove` reports success (exit 0) against `registry_server` while nothing is removed (TD-327, verified); only `assign` uses the MNS `run_group` (TD-329); LOW: relative local paths (TD-328), output/flag hygiene (TD-330), no command-level tests (TD-331), operator guide covers 8 of 15 commands (TD-332). See [DatasetManagerTool.md](docs/development/reference/source/DatasetManagerTool.md). |
| **TD-333** (HIGH) | Filed October 11, 2026: the training lifecycle releases and records trained data by bare path, so a failed pass leaves its TD-205 segment claimed (stuck after a `retrain` or any standalone pass) and a trained segment marks the whole file trained while later segments' samples go unrecorded (verified). |
| TD-334–TD-336 | Filed October 11, 2026: trained sample counts are always 0 (callers pass zeros) (TD-334); `DatasetRegistry` dead code and an unused checksum (TD-335); legacy-parser response truncation and BOM detection (TD-336). See [DatasetRegistry.md](docs/development/reference/source/DatasetRegistry.md). |
| TD-337–TD-340 | Filed October 11, 2026: `DataTransport` names FTP downloads by basename, so TD-205 segments of one file download it N times concurrently into one path and same-named files collide (TD-337, verified); a same-size stale file is reused without checking the token checksum (TD-338, verified); FTPS doesn't verify the registry's certificate (TD-339); LOW robustness/header hygiene (TD-340). See [DataTransport.md](docs/development/reference/source/DataTransport.md). |
| **TD-341** (HIGH) | Filed October 11, 2026: `DecoderBlock`'s TD-180 world-model/hippocampal gates aren't optimizer parameters and `ChatbotTrainer` only calls `optimizer->step()`, so the gates stay at 0 and the gated cross-attentions never learn. World-model injection and hippocampal memory are no-ops in every trained model (verified). |
| TD-342–TD-345 | Filed October 11, 2026: KV-cached generation and the GPU path skip the gated paths (TD-342); hippocampal memory updated on every forward pass, not once per decode step (TD-343); LOW: dropout never applied (TD-344), gated-state persistence gaps incl. SafeTensors (TD-345). See [DecoderBlock.md](docs/development/reference/source/DecoderBlock.md). |
| TD-346–TD-348 | Filed October 11, 2026: KV-cached and full decoder forwards diverge past `max_seq_length`, so over-length generation depends on the strategy (TD-347, MEDIUM, verified); LOW: `LLMDecoder::update_weights()` never trains the final norm (TD-346, verified), dead code/robustness (TD-348). TD-343 amended: memory is updated once per injected layer. See [Decoder.md](docs/development/reference/source/Decoder.md). |
| TD-349–TD-351 | Filed October 11, 2026: RAG indexes whole `.txt` files but `LLMEncoder::encode()` truncates to `max_seq_length` tokens, so content past each file's opening can't be retrieved (TD-349, verified); `DocumentStore::retrieve()` isn't safe concurrently, so it must be fixed before or with TD-230 (TD-350); LOW embedding/housekeeping gaps (TD-351). See [DocumentStore.md](docs/development/reference/source/DocumentStore.md). |
| TD-352–TD-355 | Filed and resolved October 11, 2026: `src/EfficientBatching.hpp` (an unused duplicate of `BatchProcessor`, with non-reproducible OpenMP augmentation and `EOS`-inserting masking) was **retired** with its tests, example and benchmark; `BatchedInferenceConfig`'s dead `padding_strategy` field went with it (`BatchedInferenceEngine.hpp` now `@adai-version 2.0.0`). Token-budget bucketing is kept as a design note in [real_batching_v2_plan.md](docs/proposals/real_batching_v2_plan.md) §4.1; augmentation is listed under TD-014. See [EfficientBatching.md](docs/development/reference/source/EfficientBatching.md). |
| TD-356, TD-357 | Filed October 11, 2026 (LOW): `EncoderBlock` gradient norm/clipping ignore `LayerNorm` and the header still describes Post-LN (TD-356); learning-rate overwrite, no `set_learning_rate()`, unchecked loads, `std::cout` (TD-357). TD-344 amended: the encoder's `dropout` is never applied either. See [EncoderBlock.md](docs/development/reference/source/EncoderBlock.md). |
| **TD-358** (HIGH — do first) | Filed October 11, 2026: `EncoderDecoderModel::gpu_forward()` feeds the **unshifted** targets as decoder input, so GPU training (every `incremental_trainer` pass on ai-machine) teaches the model to copy its input token instead of predicting the next one. Loss looks excellent but generation fails (CPU reproduction: 12/12 vs 0/12 exact generations). Every GPU-trained checkpoint needs retraining after the fix. |
| TD-359–TD-363 | Filed October 11, 2026: `generate_response()`/`gpu_generate_response()` ignore `max_length` (TD-359, verified); CPU vs GPU generation differences (TD-362); LOW: no-optimizer path never updates the encoder (TD-360, verified), double `BOS` in training targets (TD-361, verified), clone/load/world-model-input gaps (TD-363). TD-343 amended: training-time generation writes hippocampal memory. See [EncoderDecoderModel.md](docs/development/reference/source/EncoderDecoderModel.md). |
| TD-364–TD-367 | Filed October 11, 2026: `LLMEncoder::encode()` adds `BOS`/`EOS` while the model trains its encoder without them, so pipeline inference feeds the trained encoder out-of-distribution input, and truncation drops the `EOS` (TD-364, MEDIUM, verified); chat input is never truncated to `max_seq_length`, so extra positions get no positional encoding (TD-365, MEDIUM, verified); LOW: serving encoders cache unread activations (TD-366), console output and unchecked weight I/O (TD-367). TD-360 corrected: only `LLMEncoder::update_weights()` is missing. See [LLMEncoder.md](docs/development/reference/source/LLMEncoder.md). |
| TD-368–TD-371 | Filed October 12, 2026: weight initialization is seeded from `std::random_device` with no seed setting (`FeedForward`, `Matrix::randomize()`), so training runs and bug reproductions can't be replayed (TD-368, MEDIUM, verified); `FeedForward::load_weights()` loads a truncated file silently, keeping old weights past the cut (TD-369, MEDIUM, verified); LOW: argument/shape checks, GPU null dereference, console output (TD-370); owner decision on scaling residual-branch output for depth, related to the Pre-LN residual-growth question (TD-371). See [FeedForward.md](docs/development/reference/source/FeedForward.md). |
| **TD-372** (HIGH), TD-373–TD-377 | Filed October 12, 2026: `FtpDataServer` accepts the PASV data connection from any host, so a third party that connects first receives the whole dataset file, even under FTPS `PROT P`, and the trainer's download fails with its token consumed (TD-372, verified); `--ftps` doesn't require `AUTH TLS`/`PROT P` (TD-373, verified); re-login leaks rate-limit slots (TD-374, verified); no connection cap/idle timeout/line limit, threads outlive `stop()` (TD-375); LOW: 60 s `stop()`, unencrypted `FEAT`, silent start failure (TD-376), token/setup gaps (TD-377). See [FtpDataServer.md](docs/development/reference/source/FtpDataServer.md). |
| TD-378–TD-380 | Filed October 12, 2026: `GenerationQualityMetrics` applies add-1 smoothing to every BLEU order including unigrams, so an unrelated 5-word answer scores BLEU-4 0.23 and an exactly correct 3-word answer 0.0067, and per-epoch BLEU-4 can rank outputs backwards (TD-378, MEDIUM, verified); LOW: `bleu1`/`bleu2` never reach the metrics DB (columns always `-1`) (TD-379); scores not comparable with published numbers or the lesson's 0–100 ranges, noisy 10-sample default (TD-380). See [GenerationQualityMetrics.md](docs/development/reference/source/GenerationQualityMetrics.md). |
| TD-381–TD-383 | Filed October 12, 2026 (CUDA `GPUManager`, by inspection): `chatbot_api_server` never passes `GPU_STRATEGY`, so it always uses the low-priority stream although `config.chatbot.conf` ships `full` (TD-381, MEDIUM); `GPUManager` has no locking while `Matrix::operator*` uses the shared GPU stream from any thread, including `ChatbotTrainer`'s "CPU-only" async quality thread (TD-382, MEDIUM); LOW: `set_device()`/`cleanup()`/`initialize()` lifecycle gaps (TD-383). See [GPUUtils.md](docs/development/reference/source/GPUUtils.md). |
| **TD-384** (HIGH), TD-385, TD-386 | Filed October 12, 2026 (GPU kernels, by inspection): softmax, LayerNorm forward and cross-entropy loss/grad reuse the shared `scratch` buffer for a second reduction with no barrier after reading `scratch[0]`, a data race that can silently corrupt a row (possibly `NaN`) in attention, LayerNorm and the loss, in **both** CUDA and SYCL, so ai-machine is exposed — **verified** on a oneAPI CPU device, where the barrier fix makes all four pass (TD-384; fix with TD-358 before retraining); LOW: `matrix_sum_gpu` allocates and synchronizes every training step (TD-385); unchecked kernel inputs and uncalled batch functions (TD-386). See [MatrixGPU_cu.md](docs/development/reference/source/MatrixGPU_cu.md). |
| TD-387–TD-389 | Filed October 12, 2026 (`MatrixGPU.hpp`, by inspection): GPU generation sizes its cross-attention KV cache at `max_seq_length` but encoder input isn't truncated (TD-365), so over-length chat input throws `out_of_range` on a GPU server while CPU answers (TD-387, MEDIUM); LOW: `GPUKVCache` copies the whole cache every decode step (TD-388); unchecked `GPUMatrix` shapes, uninitialized memory, no GPU KV-cache tests (TD-389). See [MatrixGPU.md](docs/development/reference/source/MatrixGPU.md). |
| TD-390–TD-392 | Filed October 12, 2026 (SYCL `GPUManager`, ai-machine's backend): every `GPUMemory` free is a `host_task` that stalls the in-order queue for a host round trip, and every upload/download drains it (TD-390, MEDIUM, verified on a CPU queue: 1.35 ms vs 0.051 ms per op; measure on ai-machine); one Arc card is listed once per SYCL platform so `GPU_DEVICE_ID` silently picks Level Zero or OpenCL by order (TD-391, MEDIUM); LOW: `GPUMemory(0)` throws on SYCL only (TD-392, verified). TD-382/TD-383 amended for SYCL. See [GPUUtils_SYCL.md](docs/development/reference/source/GPUUtils_SYCL.md). |
| TD-393, TD-394 | Filed October 12, 2026 (`MatrixGPU_SYCL.cpp`, first GPU kernels ever run, on a oneAPI CPU device): the SYCL sub-group sum reduction is correct only for sub-group size ≥ 16; at 8 (CPU device, Arc A-series) 256 ones sum to 64, breaking sums, saturation counts and losses for `seq_len > 64` (TD-393, MEDIUM, verified; probably latent on ai-machine's Xe2); no test can run GPU kernels without a GPU — add a CPU-device test mode (TD-394, MEDIUM). See [MatrixGPU_SYCL_cpp.md](docs/development/reference/source/MatrixGPU_SYCL_cpp.md). |
| TD-395 | Filed October 12, 2026 (LOW): `GPUMatrix` is hand-maintained twice (CUDA `MatrixGPU.hpp` and SYCL `MatrixGPU_SYCL.hpp`) though only `copy()`/`zero()` differ; the SYCL `copy()` drains the queue. TD-387 and TD-389 amended as verified on the oneAPI CPU device (first run of the GPU KV cache). See [MatrixGPU_SYCL.md](docs/development/reference/source/MatrixGPU_SYCL.md). |
| TD-396–TD-398 | Filed October 13, 2026: `HippocampalMemory`'s least-used eviction starts new memories at coverage 0, so with near-uniform attention (likely while TD-341 keeps it untrained) the newest memory is always evicted next and the buffer freezes on its first `capacity − 1` entries (TD-396, MEDIUM, verified by simulation; fix with TD-341); LOW: the swap file grows without bound (TD-397); per-step `read_all()` copies and unchecked saves (TD-398). See [HippocampalMemory.md](docs/development/reference/source/HippocampalMemory.md). |
| TD-399, TD-400 | Filed October 13, 2026 (LOW): `IMetricsReporter` doesn't document which methods `MetricsPushClient` buffers until `end_epoch()` or the start/end ordering rule; `AbnormalSample` texts are untruncated (TD-399); `ChatbotTrainer` treats a non-null reporter as "metrics enabled" but `incremental_trainer` always passes one (`NullMetricsReporter`), so reporting-only work still runs (TD-400). See [IMetricsReporter.md](docs/development/reference/source/IMetricsReporter.md). |
| TD-401–TD-404 | Filed October 13, 2026 (`incremental_trainer` CLI): `--admin-port abc` aborts the process (uncaught `std::stoi`), `8084x` is accepted, a mistyped `--objective=` silently trains the chatbot, and a flag missing its value becomes a positional (TD-401, MEDIUM, verified); LOW: `--dataset-kind` unvalidated, local path can leave `session_dir` (TD-402); `trainer_service` drops `--objective`/`--dataset-kind` (TD-403); `derive_run_id()` can collide across hosts, always used by LeJEPA (TD-404). See [IncrementalTrainerArgs.md](docs/development/reference/source/IncrementalTrainerArgs.md). |
| **TD-405–TD-407** (HIGH), TD-408–TD-410 | Filed October 13, 2026 (`IncrementalTrainer`, verified with the real class): every pass restarts from the lowest validation loss across all history although each pass validates on different data, so later passes are discarded (TD-405); a paused/crashed pass's auto-save is deleted and its session dropped on the next start (TD-406); checkpoint deletion removes only the top-level sidecars (0.1 of 29.6 MB), so `SESSION_DIR` grows without bound (TD-407) — fix all three before the TD-358 retraining; MEDIUM: `latest`/`best` symlinks always dangle (TD-408), tokenized cache key has no preprocessing version (TD-409); LOW: bookkeeping/hard-coded policy (TD-410). See [IncrementalTrainer.md](docs/development/reference/source/IncrementalTrainer.md). |
| **TD-411, TD-412** (HIGH), TD-413–TD-416 | Filed October 13, 2026 (`incremental_trainer` CLI): `resume` — every `trainer_service` pass — never downloads over FTP and uses registry-host paths (TD-411); a pass that loads zero samples returns success, so its files are marked trained (TD-412, verified) — together, an always-on service on a remote FTP registry marks files trained without training; MEDIUM: standalone pre-check reads the root pending list, not the `<kind>` pool (TD-413, verified); first-run vocab bootstrap unreachable, `train`/`status`/`history` abort without a vocab (TD-414, verified); LeJEPA ignores segment ranges and strands claims after a crash (TD-415); LOW: command hygiene (TD-416). See [IncrementalTrainingTool.md](docs/development/reference/source/IncrementalTrainingTool.md). |
| TD-417–TD-420 | Filed October 13, 2026 (`--integrated-inference`): ignores the request strategy (always greedy), adds special tokens to encoder input and skips masking/repetition, so answers differ from normal `/chat` (TD-417, verified); worker-thread exceptions terminate `chatbot_api_server` and shutdown drops queued requests (TD-418); drives the model without `model_mutex_` while `/chat/batch` uses it (TD-419); LOW: batches nothing despite "50–100×"/"Production Ready" claims, dead config, fictional stats (TD-420). See [IntegratedInferenceEngine.md](docs/development/reference/source/IntegratedInferenceEngine.md). |
| TD-421, TD-422 | Filed October 13, 2026 (`KVCache`): every `append()` reallocates and copies the whole cache, so CPU cached generation does O(n²·d) copying (255 ms per cache for 512 steps; about 6 s at 24 layers) (TD-422, MEDIUM, verified); LOW: `append()` has no shape checks, so mismatched input overflows the heap (ASan-verified), and layer indices are unchecked (TD-421). See [KVCache.md](docs/development/reference/source/KVCache.md). |
| TD-423–TD-425 | Filed October 14, 2026 (`LanguageModelHead`): a truncated `.lm_head` checkpoint sidecar loads silently with `NaN` weights, so every logit is `NaN` (TD-423, MEDIUM, verified); uncached generation (beam, CPU `model_fn`, `/chat/batch`) projects every position each step though only the last row is used, about 30× the work at 100 tokens (TD-424, MEDIUM, verified); LOW: hygiene (TD-425). See [LanguageModelHead.md](docs/development/reference/source/LanguageModelHead.md). |
| TD-426 | Filed October 14, 2026 (LOW, `LayerNorm`): `set_gamma()`/`set_beta()` silently ignore shape mismatches (verified); no shape checks in `forward()`/`backward()`; unchecked `load_weights()` (fix with the other legacy loaders, Tier 55). CPU backward verified correct. See [LayerNorm.md](docs/development/reference/source/LayerNorm.md). |
| TD-427–TD-429 | Filed October 14, 2026 (`LeJEPAEncoder`): `save()`/`load()` skip the `Predictor`, so every resumed world-model pass starts with a random predictor (loss 0.0017 → 2.05 after reload) whose gradients disturb the resumed encoder (TD-427, MEDIUM, verified); LOW: two-update `train_step()` workaround obsolete after TD-195 (TD-428); escaping exception type, console output, unchecked loads (TD-429). See [LeJEPAEncoder.md](docs/development/reference/source/LeJEPAEncoder.md). |
| TD-430, TD-431 | Filed October 14, 2026 (`Logger`): with `LOG_FILE_PATH` empty, `trainer_service` and every child pass fall back to the same `chatbot_server.log` and rotate it from two processes (~27% less retained history, verified), and the trainer binaries ignore `LOG_LEVEL` (TD-431, MEDIUM); LOW: unsynchronized lazy init makes concurrent first-use log calls throw `spdlog_ex` (TD-430, verified). See [Logger.md](docs/development/reference/source/Logger.md). |
| TD-432, TD-433 | Filed October 14, 2026 (LOW, `LoRA`): `LoRAAdapter::load()` silently adopts the file's dims/rank/α (TD-432, verified); unseeded global `rand()` init, `update()` doesn't zero gradients, unused `dropout`/`apply_to_ffn` (TD-433). `src/LoRA.hpp` re-tagged `experimental` (no production entry point; TD-302). See [LoRA.md](docs/development/reference/source/LoRA.md). |
| TD-434, TD-435 | Filed October 14, 2026 (`Matrix`, every CPU `operator*` path verified correct): `operator*` only parallelizes when `rows > 64`, so the ≤ 64-row products of training samples and decode steps run single-threaded (2–6× slower than a work-based gate), and `portable` builds get no SIMD multiply because AVX2 is compile-time only (7–10× slower) (TD-434, MEDIUM, verified by timing); LOW: out-of-line bounds-checked `operator()` (2.4×), `float` `sum()` drift (2.2% at 16.8M), `randomize()` static-RNG race, API hygiene (TD-435). See [Matrix.md](docs/development/reference/source/Matrix.md). |
| TD-436, TD-437 | Filed October 14, 2026 (`MatrixSIMD`): the CMake `-march`/`-mavx2 -mfma` checks match only `GNU\|Clang`, and `icpx` reports `IntelLLVM`, so the `sycl` preset (ai-machine's trainer) compiles `Matrix.cpp` with no SIMD — 6–8× slower CPU multiply (TD-436, MEDIUM, verified); LOW: the compile-time vs. runtime SIMD test never compiles (verified), no startup CPU check (`SIGILL` on older CPUs), `has_avx2()` skips the OS-support check (TD-437). See [MatrixSIMD.md](docs/development/reference/source/MatrixSIMD.md). |
| TD-438, TD-439 | Filed October 14, 2026 (`metrics_api_server` CLI): numeric flags are unchecked `atoi`/`strtoul`, so `--port abc` listens on a random port while printing `:0`, `--port 99999` binds 34463, `--max-live-sessions -1` removes the cap, `--sweep-interval-seconds 0` disables eviction; saved admin overrides have the same gaps; bad arguments exit 0 (TD-438, MEDIUM, verified with the real server; fix with TD-240); LOW: unknown `--storage-backend` accepted, admin store located from the file's `db_path` not `--db-path` (TD-439). See [MetricsApiServerArgs.md](docs/development/reference/source/MetricsApiServerArgs.md). |
| TD-440, TD-441 | Filed October 14, 2026 (`IMetricsDatabase`): nothing deletes metrics rows or rotates the JSONL, and `max_records_on_disk` (CLI/config/admin setting) is read by no persistence code, so `metrics_api_server` storage grows without bound; `gradient_variance_history` (one row per optimizer step) and `generation_quality` are written but never read (TD-440, MEDIUM, verified by search; retention policy is an owner decision); LOW: step counter restarts per service instance, `db-history?limit=N` returns the oldest N, unbounded compare/export reads, undocumented contract (TD-441). See [MetricsDatabase.md](docs/development/reference/source/MetricsDatabase.md). |
