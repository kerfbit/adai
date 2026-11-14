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

GUI widget tests (TD-037) use Qt Test, built when the GUIs are (`qtbase5-dev` provides it) and run offscreen by
`ctest`: `chatbotGuiWidgetTests`, `mnsManagerGuiWidgetTests`. Each drives the real window against an in-process stand-in
server (`tests/qt_widget_test_support.hpp`); give new controls an object name for the tests to find them. To run one
directly: `QT_QPA_PLATFORM=offscreen ./build/debug/tests/chatbotGuiWidgetTests`.

For sanitizer testing: `./scripts/run_tests.sh --asan|--ubsan|--tsan|--coverage`

## Architecture

### Executable Targets

| Binary | Purpose | Port |
|---|---|---|
| `chatbot` | Interactive CLI client | — |
| `chatbot_api_server` | REST inference API + session management | 8080 |
| `chatbot_gui` | Qt GUI client of `chatbot_api_server` (optional; TD-260) | — |
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

**TD-033** (mostly resolved September 13, 2026): `ChatbotAPI::generate_response()`'s plain path now checks `GPUManager::is_available()` and routes through a new `EncoderDecoderModel::gpu_generate_response_with_strategy()` — a persistent-residency decode path supporting every generation strategy — instead of unconditionally calling the plain CPU `forward()` path. `model_mutex_` is now held across it (and the pre-existing `gpu_generate_response()`), closing a concurrency gap this exact wiring would otherwise have reintroduced (encoder/decoder/lm_head's own `gpu_forward()` calls write through to persistent per-instance `GPUState`, shared across concurrent chat requests — the same class of bug TD-156 fixed for the CPU path). Only the before/after latency benchmark remains open: `gpu_decode_bench` in the GPU test package (TD-050) measures it, and needs a run on ai-machine.

**`GPU_STRATEGY`** (config key): `background` (low-priority queue, default) or `full` (normal priority).

**GPU training keeps the training math on the device (owner rule, October 19, 2026).** No host↔device transfers happen without a critical
reason. Since TD-468, `ChatbotTrainer` binds a `GPUOptimizer` (`src/GPUOptimizer.{hpp,cpp}`) to the layers' device mirrors for the whole of
`train()`. The norms, clipping and the SGD/Adam/AdamW update run on the device, and a step downloads one float per parameter group. The device
weights are the live copy while training. `sync_host_from_device()` copies weights and moments back only for checkpoints, the per-epoch
quality-metrics clone and the end of `train()`. Since TD-547 the token embeddings live on the device too (a gather kernel for lookup and positional encoding, scatter-add for backward), so only token IDs, target IDs, the scalar loss and the small causal mask cross per sample. A SYCL build runs its real kernels on the
oneAPI CPU device with `ADAI_SYCL_ALLOW_CPU_DEVICE=1` (TD-394), for tests on a machine with no GPU (e.g. `build/sycl-cpudev`).

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
`incremental_trainer --objective=<o> --dataset-kind <k> --foreground --admin-port <N> resume` as a single-pass child
(TD-403: its own `--objective`/`--dataset-kind`, so it trains any objective and pool), waits for it to
exit (TD-569: or stops it after `TRAINER_PASS_STALL_MINUTES` without progress), and launches another immediately if that pass did work (more may be pending) or after a poll
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

`dataset_manager status` calls `load_pending` (returns all entries). The trainer calls `acquire` (returns only untrained entries — `run_id` empty — that are also unassigned or assigned to the caller's own `model_name`; a caller with no `model_name` can only claim unassigned entries, never one assigned to a specific model). A file visible in `status` but not acquired means its `run_id` is still set from a previous run, or it's assigned to a different model (`dataset_manager assign`/`unassign`). Release with `POST /registry/{group}/release {"run_id":"","files":[...]}` (empty `run_id` bypasses the owner check). Move trained files back to pending with `dataset_manager requeue (--all | <file> ...) [--model <name>]` (`POST /registry/{group}[/{kind}]/requeue`; added October 18, 2026, TD-540), for example before retraining from scratch. **The trainer always takes its data files and configuration from the remote services (`registry_server` over FTP, MNS) first**; local paths and config are only a fallback.

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
via `SIGHUP` (and only some keys — see TD-251/TD-285); every other long-running binary
(`incremental_trainer`, `trainer_service`, `metrics_api_server`, `mns_server`, `registry_server`) ignores
`SIGHUP` and logs at startup that a restart is needed (`adai::ignore_sighup()`, `src/SignalSetup.hpp`; it
used to terminate them, TD-283).

Configuration is split into 5 service-scoped files, each read only by the binaries that need those
keys (non-architecture keys used by more than one binary, e.g. GPU/MNS-client settings, are duplicated
across files rather than shared):

| File | Read by | Covers |
|---|---|---|
| `config.chatbot.conf` | `chatbot_api_server` | server/log, architecture (fallback), generation, RAG, GPU, MNS client |
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
`HIPPOCAMPAL_*` tuning parameter. Mutable via `POST /models/{name}/link-world-model` (a body with `"merge":true` keeps the fields it omits, TD-449; unlike
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
| `METRICS_SERVER_URL` | URL of `metrics_api_server`; empty = no push. `https://` works when built with libcurl (TD-444) |
| `METRICS_SERVER_HEADERS`, `METRICS_SERVER_CA_FILE` | TD-444, `config.trainer.conf`: extra headers for every metrics push ("Name: value" pairs separated by `;`, e.g. a Cloudflare Access service token) and a CA bundle for an https:// server |
| `METRICS_STALENESS_THRESHOLD_SECONDS` | Seconds idle before dashboard drops session (default: 60) |
| `METRICS_HEARTBEAT_INTERVAL_MS` | Idle heartbeat period from trainer (default: 30 000) |
| `NAME_SERVICE_URL`, `MODEL_NAME`, `MODEL_ROLE` | MNS connection |
| `WORLD_MODEL_ARTIFACT_PATH` | Overrides the default `<session_dir>/world_model_checkpoint` checkpoint directory (`<session_dir>/world_model` before TD-548, still read if only it has one); honored by `--objective=lejepa` too since TD-548; normally set automatically from a linked world model's own MNS `artifact.path` (TD-196), not by hand |
| `REGISTRY_SERVER_URL`, `RUN_GROUP`, `RUN_ID` | Distributed dataset registry |
| `DATASET_KIND` | TD-202: per-trainable-piece sub-pool within `RUN_GROUP` (`encoder`/`decoder`/`world_model`/`chatbot`); empty = legacy shared pool. Read by `dataset_manager` (falls back to it when `--kind` isn't passed); **ignored by `incremental_trainer`** (TD-204), which always derives it from the training objective unless `--dataset-kind` overrides — see "Distributed Dataset Registry" above |
| `METRICS_API_CORS_ALLOWED_ORIGINS` | TD-526: comma-separated browser origins allowed cross-origin access to `metrics_api_server` (reads and writes; `*` = any origin may read, never write). Empty by default. Clients that aren't browsers send no `Origin` and are unaffected. `config.metrics.conf`, or `--cors-allowed-origins` |
| `FTPS_PINNED_PUBLIC_KEY`, `FTPS_CA_FILE` | TD-339: how the trainer authenticates `registry_server`'s FTPS certificate — the `sha256//…` pin `registry_server` logs at startup, or a CA bundle that signs it (hostname checked). Either one makes the trainer refuse plain FTP; with neither, FTPS is unverified and logs a warning. `config.trainer.conf` |
| `REGISTRY_CLAIM_LEASE_MINUTES`, `REGISTRY_MAX_CONCURRENT_FETCHES`, `REGISTRY_MAX_UPLOAD_MB` | TD-494, `registry_server` (`config.registry.conf` or `--claim-lease-minutes`/`--max-concurrent-fetches`/`--max-upload-mb`): a claim expires after the lease (default 30 min, 0 = never) unless renewed — trainers renew automatically (`DatasetRegistry`'s claim renewer, every third of the lease); at most N fetches at once (default 2, else 429); request bodies up to N MiB (default 2048, else 413) |
| `METRICS_MAX_RECORDS_ON_DISK`, `METRICS_RETENTION_DAYS`, `METRICS_FILE_MAX_MB`, `METRICS_FILE_BACKUPS` | TD-440, `metrics_api_server` (`config.metrics.conf`, matching flags, admin-mutable): DB rows kept per session (100,000; older rows thinned), days before an un-updated session and its archived files are deleted (90, 0 = forever), and JSONL rotation size/backups (100 MB, 3) |
| `REGISTRY_LISTEN_PORT`, `REGISTRY_DATA_DIR` | `registry_server`'s own listen port / data dir (server-side, distinct from the client-side `REGISTRY_SERVER_URL` above) |
| `HIPPOCAMPAL_STATE_PATH`, `HIPPOCAMPAL_SAVE_INTERVAL_SECONDS` | TD-250, `config.chatbot.conf`: where `chatbot_api_server` keeps hippocampal memory (default `MODEL_PATH.hippocampal`, plus `.assoc`/`.swap`; with neither set it isn't attached) and how often a changed memory is saved (300 s; 0 = only at shutdown) |
| `LOSS_QUALITY_BACKFILL`, `GENERATION_QUALITY_BACKFILL` | TD-273, `config.trainer.conf`: report each pair's `exp(-loss)` every epoch (default on) and, after training, a generated reply's BLEU-4 / 100 (default off: one generation per pair) to `metrics_api_server`'s `…/metrics/sample-quality` as a distribution plus the lowest-scoring pairs |
| `EARLY_STOPPING_ENABLED`, `EARLY_STOPPING_PATIENCE`, `RESTORE_BEST_WEIGHTS`, `LR_SCHEDULE` | TD-410, `config.trainer.conf`: early stopping (on, 5 epochs, restore best) and `linear_warmup` (default) or `constant`; no decaying schedule, since passes continue one run (TD-269) |
| `SEED` | TD-368, `config.trainer.conf`: replays weight initialization, epoch shuffles, dropout and LeJEPA's random choices (0 = a random seed, logged); `incremental_trainer --seed <n>` overrides it |
| `DROPOUT` | TD-344, `config.trainer.conf`: dropout on every transformer block's attention and feed-forward outputs while training (default 0.1, `[0, 1)`, 0 = off). `ChatbotTrainer::train()` applies it through `EncoderDecoderModel::set_dropout()` and resets it to 0 afterwards; validation, generation and serving never drop |
| `WORLD_MODEL_SIGREG_BATCH_SIZE` | Texts per LeJEPA training batch (default 32, at least 2); SIGReg is applied across their mean-pooled embeddings (`LeJEPAEncoder::train_batch()`, TD-503). `config.trainer.conf` |
| `RESUME_FROM_BEST_CHECKPOINT` | TD-405: `false` (default) starts each pass from the most recent completed session; `true` uses the lowest validation loss, only meaningful with a fixed validation set. `config.trainer.conf` |
| `AUTO_SAVE_ENABLED`, `AUTO_SAVE_EVERY_SAMPLES`, `AUTO_SAVE_EVERY_MINUTES`, `MAX_SESSIONS_TO_KEEP` | Checkpoint cadence / retention — map into `IncrementalConfig`'s matching fields via `make_incremental_config()`; live-tunable under `serve` via `PUT /admin/config`, see below |
| `TRAINER_ADMIN_ENABLED`, `TRAINER_ADMIN_PORT`, `TRAINER_ADMIN_HOST`, `TRAINER_ADMIN_DIR` | `trainer_service`'s admin HTTP API — see below |
| `TRAINER_CHILD_ADMIN_PORT` | Loopback-only port `trainer_service` assigns each single-pass child it launches — see below |
| `TRAINER_PASS_STALL_MINUTES` | TD-569, `config.trainer.conf`, read by `trainer_service`: stop a pass that has made no progress for this long (default 30, 0 = off), as a failed pass; progress goes through `<SESSION_DIR>/trainer_service_pass_<pool>.heartbeat` |

### Daemon admin config API

`metrics_api_server`, `mns_server`, and `registry_server` now also load their `config.<service>.conf`
at startup and expose `GET`/`PUT /admin/config` for a documented allow-list of settings that are safe
to change without restarting (things baked into an already-open socket, DB handle, or FTP listener —
e.g. `port`, `data_dir`, `db_path` — are excluded and file/CLI-only). `PUT` writes accepted keys to a
`daemon_config` SQLite table (`src/DaemonConfigStore.{hpp,cpp}`, one `daemon_config.db` per daemon,
isolated from that daemon's own data store) which overlays the config file on the next restart. The
endpoint is gated behind `--admin-enabled` or `REGISTRY_ADMIN_ENABLED`/`MNS_ADMIN_ENABLED` (`registry_server`, `mns_server`) or the existing
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
supervisor immediately launched a fresh, unpaused child right after — see
[TECHNICAL_DEBT_RESOLVED.md](docs/development/archive/TECHNICAL_DEBT_RESOLVED.md)'s TD-173 entry for the full incident writeup. `/admin/status` also reports several
`trainer_service`-only observability fields (`supervisor_paused`, `total_passes_launched`,
`total_passes_did_work`, `total_passes_crashed`, `total_passes_failed`, `consecutive_failures`,
`last_failure_unix` (TD-523), `total_passes_stalled`, `seconds_since_pass_progress` (TD-569), `last_exit_code`,
`service_uptime_seconds`) as
additional top-level keys alongside the pre-existing shape — additive only, so existing clients
(the Android ops dashboard's `TrainerStatusDto` already sets `ignoreUnknownKeys = true`) are
unaffected.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness — always answered by `trainer_service` itself, never proxied |
| GET | `/admin/config` | current `auto_save_enabled`/`auto_save_every_samples`/`auto_save_every_minutes`/`max_sessions_to_keep` — read from the current child if one is alive, else from the shared `TRAINER_ADMIN_DIR/daemon_config.db` directly |
| PUT | `/admin/config` | mutate the same four keys; persisted to `TRAINER_ADMIN_DIR/daemon_config.db` either way (via the live child's own overlay, or directly when idle) — either way the next child launched picks it up |
| GET | `/admin/status` | phase (`idle`/`loading_data`/`tokenizing`/`training`/`checkpointing`/`pausing`), run/session identity, epoch/sample/loss progress, real (not hardcoded) `paused`, checkpoint counters, plus `trainer_service`'s own observability fields (see above) |
| POST | `/admin/checkpoint[?wait_ms=N]` | force a checkpoint at the next optimizer-step boundary; 409 if idle (no active pass to checkpoint); `wait_ms` at most 60 000, and `completed` only once a checkpoint written for this request exists (TD-521) |
| POST | `/admin/pause` | sets `trainer_service`'s own persistent pause flag (stops it launching another pass) and, best-effort, drains the current pass (if any) via `ChatbotTrainer::set_abort_flag()`/checkpoint/release-claimed-files — `trainer_service` keeps serving regardless, it does not exit |
| POST | `/admin/resume` | clears the persistent pause flag and wakes the idle-poll sleep immediately, plus resumes a live child if one happens to be paused too |

No HTTP shutdown endpoint exists or is planned — `systemctl stop`/SIGTERM stays the sole way to end the
`trainer_service` process, which in turn forwards a graceful-stop request to whatever child is
currently running, escalating to a force kill (`ChildProcess::stop_and_wait()`) if it doesn't exit
within 25s (TD-416: the child checkpoints and releases its files on SIGTERM) — protects against a genuinely wedged child (this host's own documented GPU-driver
*hang*, not just crash) even without systemd's own external `TimeoutStopSec` as a backstop. No
companion CLI wraps this API (matches `mns_cli`/`dataset_manager` not wrapping their daemons'
`/admin/config` either) — `curl` is the documented interface, e.g.
`curl -s http://127.0.0.1:8084/admin/status`.

## Code Conventions

- **Logging**: `adai::Logger::info/warn/error/debug(fmt, args...)` — never `std::cout` in library code.
- **Headers**: `#pragma once`, no implementation in headers except templates.
- **Ownership**: `std::unique_ptr` / `std::shared_ptr`; no raw owning pointers.
- **New component**: `src/Component.{cpp,hpp}` + `tests/component_test.cpp` + register in `src/CMakeLists.txt` and `tests/CMakeLists.txt`; new files start tagged `@adai-status: experimental`, `@adai-version: 0.1.0` (see below).
- **Reference docs**: every code change updates the changed file's code-traced reference in `docs/development/reference/source/` in the same change, so the doc describes the code as it now is: its header (files, build, status tags, tests, last traced), its sections and its known-gaps table. An addendum can record what changed and why, but doesn't replace updating the earlier text (`ChatbotGUI.md` drifted that way: four fixes added addenda while §1–§6 kept describing removed code).
- **Artifacts**: trained weights go in `training_sessions/` (gitignored); feature proposals in `docs/proposals/` (listed in its `README.md`).
- **File status tag**: every in-scope file (`src/`, Android `src/main`, `tizen-metrics-app/js/`, operational `scripts/`) carries a header comment block — `@adai-status` (`experimental`/`beta`/`stable`/`deprecated`/`legacy`), `@adai-version` (per-file SemVer, `stable` requires `MAJOR >= 1`), `@adai-reviewed` (ISO date). See [file-status-standard.md](docs/development/guides/file-status-standard.md); validate with `./scripts/check_file_status.py --changed` and regenerate the dashboard with `./scripts/gen_status_report.py`.

## Active Technical Debt Tags

The codebase uses `// TD-NNN` comment tags, though most that remain inline today cite
already-resolved items kept as historical design-rationale footnotes, not pending work. Check
[TECHNICAL_DEBT.md](docs/development/guides/TECHNICAL_DEBT.md) for what's actually open rather than
trusting a `grep TD-NNN` alone; its Recommended Execution Order says what to take next. Resolved items,
with the history these rows used to carry, are in
[TECHNICAL_DEBT_RESOLVED.md](docs/development/archive/TECHNICAL_DEBT_RESOLVED.md). Currently open items
(103 on November 14, 2026, one row each; HIGH in bold; "Left:" is the first open action item, with the
reference doc that tracks it):

| Tag | Description |
|---|---|
| **TD-059** (HIGH) | Multi-Head and Cross-Attention Never Actually Split Into Heads. Code fix landed and verified (September 13, 2026); retraining not performed. Left: Retrain from scratch and validate against a from-scratch baseline before declaring any existing deployment upgraded — needs the real training… (and 1 more). |
| TD-050 (MEDIUM) | GPU-Resident KV-Cache for Autoregressive Generation. CPU cache root-caused clean, GPU-resident cache designed/implemented, and the flagged CPU beam-vs-cache gap fixed (all September 14, 2026); matches the CPU on ai-machine's Arc (November 10 and 12, 2026); `gpu_decode_bench` packaged November 13, 2026 for the direct check against GPU full recompute and the benchmark, which need a run on ai-machine. Left: Validate correctness against the existing full-recompute GPU path (identical token-for-token output for greedy decoding). On ai-machine's Arc the… (and 1 more). |
| TD-033 (MEDIUM) | chatbot_api_server Inference Never Uses Persistent GPU-Resident Decode. Wired in and verified September 13, 2026; before/after benchmark packaged November 13, 2026 (`gpu_decode_bench`, TD-050), needs a run on ai-machine. Left: Benchmark chat response latency before/after on a representative prompt/`max_length` on both CUDA and SYCL builds to confirm the per-token round-trip… |
| TD-014 (MEDIUM) | LLM Operations and Training Tooling Suite. Planned. |
| TD-006 (LOW) | Fill-in-the-Middle (FIM) Training Data Generation. Planned. Left: Add FIM data generation to `create_qa_pairs_from_text()` (and 6 more). |
| TD-038 (LOW) | Advanced Features Tested in Isolation, Never Wired Into a Shipped Binary. 7/7 non-blocked items done — Quantization deliberately deferred. Left: `Quantization`: still explicitly deferred — needs a scoping decision: a standalone checkpoint-manipulation CLI tool (lower risk) vs. actually… |
| TD-039 (MEDIUM) | Core Training/Metrics Classes Too Large and Fast-Moving to Certify Stable. Left: Let the current round of feature work (trainer admin API, session-scoped metrics) land and settle — no more structural changes planned (and 2 more). |
| TD-047 (MEDIUM) | Android Data/Repository/API Layer Has No CI or Release History. CI added; release process defined, first real release still pending. Left: Resolve the `origin` repo's GitHub billing/spending-limit issue so `android-ci.yml` actually runs there (confirmed working on the kerfbit mirror… (and 1 more). |
| TD-048 (MEDIUM) | Android UI/DI/Entry-Point Classes Are Untested and Unreleased. 12/12 ViewModels done; Compose UI testing adopted, 12/12 screens covered; admin-dialog confirm-click flows and DI containers/entry points now covered too; `wearcomplications` services and full real-device verification remain. Left: `BiometricAdminAuthGate.kt` specifically: consider an instrumented test using `BiometricPrompt`'s test/fake authenticator support instead of leaving… |
| TD-164 (LOW) | chatbot-guide.md Needs a Live-Pair Verification Pass. Left: Start a real `chatbot_api_server` + `chatbot` pair and walk every example in the Commands Reference, Generation Strategies, and Configuration… (and 1 more). |
| TD-171 (LOW) | No Batch Dimension Anywhere in the Model Stack — Real Parallel Batched Training Not Supported. Inference half planned as v2 (October 8, 2026), B0 CPU results November 9, 2026, Arc results November 10, 2026; training half not started. Left: Owner decision: is this worth pursuing at all? The current per-sample training loop already works correctly (gradient accumulation already gives a… (and 2 more). See [BatchProcessor.md](docs/development/reference/source/BatchProcessor.md). |
| TD-172 (MEDIUM) | incremental_trainer's `serve` Command Embeds the Always-On Service in the Same Binary as Its CLI Commands. Implemented, unit-tested, and verified end-to-end locally (September 14, 2026); only a live deployed host's own systemd cutover remains. Left: **Not done, not this session's to do:** cut over an actual live deployed host's systemd unit and binaries. Everything in this repo is ready for that… |
| TD-211 (MEDIUM) | BatchedInferenceEngine Queues and Serializes Requests but Never Batches the Model. Decided October 8, 2026: real batching planned as v2; v1.x honesty pass outstanding. Left: v1.x honesty pass: rewrite the file/class doc comments and `--batched-inference`'s `--help` text to describe today's queue-and-serialize behaviour… (and 3 more). See [BatchedInferenceEngine.md](docs/development/reference/source/BatchedInferenceEngine.md). |
| TD-213 (LOW) | BatchedInferenceEngine Stats Are Misleading and Unused in Production. Left: Compute real per-request latency from `submit_time` (queue wait plus generation, ideally both), and guard every derived field against a zero… (and 3 more). See [BatchedInferenceEngine.md](docs/development/reference/source/BatchedInferenceEngine.md). |
| TD-214 (LOW) | Batched-Inference Mode Generates Differently From the Inline Path. Left: Map `strategy`/`beam_width` (or reject unsupported strategies explicitly) in the batched branch of `ChatbotAPI::generate_response()` (and 2 more). See [BatchedInferenceEngine.md](docs/development/reference/source/BatchedInferenceEngine.md). |
| TD-217 (LOW) | `/chat/batch` Reports Hypothetical Batching Stats as if They Were Real. Decided October 8, 2026: relabel as an estimate. Left: Implement the relabel: emit the block as `"padding_estimate"` (same fields), and document in `rest-api.md` that it estimates padding for a… (and 3 more). See [BatchProcessor.md](docs/development/reference/source/BatchProcessor.md). |
| TD-218 (LOW) | BatchProcessor Helpers Have Dead Surface, Lost Ordering, and Unchecked Inputs. Decided October 8, 2026: keep as groundwork for v2. Left: v1.x: rewrite the file comment to describe what the helpers do today and point at the v2 plan; add bounds checks to `unbatch_outputs()` and fix its… (and 1 more). See [BatchProcessor.md](docs/development/reference/source/BatchProcessor.md). |
| **TD-220** (HIGH) | BPE Merge Learning Is Corrupted by the `"\|\|\|"` Pair-Key Separator. Code fix landed October 17, 2026 (vocab audit remains). Left: Audit existing vocab files (run `scripts/audit_vocab_merges.py` on ai-machine) (`vocab.txt`, every `<checkpoint>.vocab` on ai-machine): compare the… |
| TD-233 (MEDIUM) | ChatbotAPI Session IDs Come From a Non-Cryptographic Generator. Code fix landed October 27, 2026. Left: Consider requiring an auth token for export/import if the server is reachable beyond localhost. See [ChatbotAPI.md](docs/development/reference/source/ChatbotAPI.md). |
| **TD-247** (HIGH) | RAG Retrieval in `chatbot_api_server` Uses an Untrained Encoder. Code fix landed October 26, 2026. Left: Test: retrieval ranks a document containing the query's content above unrelated ones with a trained checkpoint. (Not done: needs a trained… See [ChatbotAPIServer.md](docs/development/reference/source/ChatbotAPIServer.md). |
| TD-263 (MEDIUM) | `chatbot_gui` Freezes While Generating. Code fix landed November 4, 2026 (with TD-260). Left: Add a cancel/stop control if the generator can support it. Still open: `chatbot_api_server` can't stop a generation, so a Stop button would only stop… See [ChatbotGUI.md](docs/development/reference/source/ChatbotGUI.md). |
| TD-269 (MEDIUM) | Optimizer State and LR Schedule Restart Every Incremental Training Pass. Code fix landed October 19, 2026. Left: Track run-level `global_step` in the MNS/session record. (The step is in the checkpoint's training state; MNS doesn't show it yet.). See [Optimizer.md](docs/development/reference/source/Optimizer.md). |
| TD-296 (LOW) | ConversationContext Hygiene. Left: Check the stream and write via temp file + rename (and 1 more). See [ConversationContext.md](docs/development/reference/source/ConversationContext.md). |
| TD-301 (LOW) | `CrossAttention` GPU Path Diverges from the CPU Path. Code fix landed November 13, 2026. Left: Run the GPU test package on ai-machine with these tests. See [CrossAttention.md](docs/development/reference/source/CrossAttention.md), [MultiHeadAttention.md](docs/development/reference/source/MultiHeadAttention.md). |
| TD-302 (LOW) | LoRA Adapters Aren't Persisted or Counted in Gradient Norms. Left: Persist adapters (e.g. extra SafeTensors entries) or document merge-before-save and enforce it (and 1 more). See [CrossAttention.md](docs/development/reference/source/CrossAttention.md). |
| TD-303 (LOW) | Decoder-Only Mode Still Runs Cross-Attention over a Dummy Input. Left: Skip the cross-attention sub-layer in `DecoderBlock` when no encoder output is given (e.g. a null pointer), and fix the comment. See [CrossAttention.md](docs/development/reference/source/CrossAttention.md). |
| TD-314 (LOW) | `DataFetcher` Source Coverage and Hygiene Gaps. Left: Add an optional HF `config` parameter (CLI, endpoint, dashboard) (and 2 more). See [DataFetcher.md](docs/development/reference/source/DataFetcher.md). |
| TD-331 (LOW) | `dataset_manager` Has No Tests for Its Commands. Left: Add a CLI-level test target that runs `dataset_manager` against a scratch `SESSION_DIR` (and a local `registry_server` for remote mode), covering… See [DatasetManagerTool.md](docs/development/reference/source/DatasetManagerTool.md). |
| TD-332 (LOW) | Operator Guide Covers Only 8 of 15 `dataset_manager` Commands. Left: Add sections for the missing commands and `--kind`, including the "omit = all" semantics (TD-321) and remote/local differences. See [DatasetManagerTool.md](docs/development/reference/source/DatasetManagerTool.md). |
| TD-335 (LOW) | `DatasetRegistry` Dead Code and Unused Integrity Data. Left: Use the checksum (warn when a queued/trained file changed) or drop it; ~~count pairs without storing them~~ (done with TD-543); delete the unused… See [DatasetRegistry.md](docs/development/reference/source/DatasetRegistry.md). |
| TD-340 (LOW) | `DataTransport` Robustness and Header Hygiene. Left: Call `curl_global_init()` once at trainer startup; keep partial files on failure; treat `COULDNT_CONNECT` as retryable with backoff; move the… See [DataTransport.md](docs/development/reference/source/DataTransport.md). |
| **TD-341** (HIGH) | World-Model and Hippocampal Gates Never Train in Production. Code fix landed October 20, 2026. Left: Push per-layer `tanh(gate)`/`tanh(gate_h)` as a training metric so a stuck gate is visible. (Partly: logged per epoch; a metrics-pipeline series… See [DecoderBlock.md](docs/development/reference/source/DecoderBlock.md). |
| TD-345 (LOW) | `DecoderBlock` Persistence Gaps for Gated State. Left: Check reads in `load()` and validate before assigning; add SafeTensors entries (and accessors) for the gates and gated sub-layers. See [DecoderBlock.md](docs/development/reference/source/DecoderBlock.md). |
| TD-348 (LOW) | `LLMDecoder` Dead Code and Robustness Gaps. Left: Remove the empty loop, the unused copy and (if still unused) `forward_with_mask()`; check header reads; validate constructor arguments; guard the… See [Decoder.md](docs/development/reference/source/Decoder.md). |
| TD-351 (LOW) | `DocumentStore` Embedding and Housekeeping Gaps. Left: Pool over content tokens only; add metadata filtering or drop the field; optionally cache embeddings on disk keyed by file checksum; treat NaN scores… See [DocumentStore.md](docs/development/reference/source/DocumentStore.md). |
| TD-356 (LOW) | `EncoderBlock` Gradient Norm and Clipping Ignore `LayerNorm`; Header Describes Post-LN. Left: Include `LayerNorm` gradients in `get_gradient_norm()`/`clip_gradients()` (add a norm accessor to `LayerNorm`), or remove the unused… (and 1 more). See [EncoderBlock.md](docs/development/reference/source/EncoderBlock.md). |
| TD-357 (LOW) | `EncoderBlock` Learning-Rate Handling, Unchecked Loads and Console Output. Left: Add `set_learning_rate()` (propagating to sub-layers, like `DecoderBlock`'s) and stop overwriting in `update_weights()`; check reads in… See [EncoderBlock.md](docs/development/reference/source/EncoderBlock.md). |
| **TD-358** (HIGH) | GPU Training Feeds Unshifted Targets, Teaching the Model to Copy Instead of Predict. Code fix landed October 17, 2026. Left: Retrain affected models; consider bumping the checkpoint format version so GPU-trained pre-fix checkpoints are flagged. See [EncoderDecoderModel.md](docs/development/reference/source/EncoderDecoderModel.md). |
| TD-367 (LOW) | `LLMEncoder` Console Output and Unchecked Weight I/O. Left: Switch to `adai::Logger` (debug level for the per-call message) (and 1 more). See [LLMEncoder.md](docs/development/reference/source/LLMEncoder.md). |
| TD-368 (MEDIUM) | Weight Initialization Can't Be Seeded or Reproduced. Code fix landed November 9, 2026. Left: GPU replays are close, not exact (see below). See [FeedForward.md](docs/development/reference/source/FeedForward.md). |
| TD-370 (LOW) | `FeedForward` Argument Checks, GPU Null Dereference and Console Output. Left: Validate constructor arguments; check shapes in the setters and invalidate (or re-upload) the GPU mirror (and 3 more). See [FeedForward.md](docs/development/reference/source/FeedForward.md). |
| TD-376 (LOW) | `FtpDataServer` Slow Shutdown, Unencrypted `FEAT` and Silent Start Failures. Left: Replace the sleep with a condition-variable wait that `stop()` notifies (and 2 more). See [FtpDataServer.md](docs/development/reference/source/FtpDataServer.md). |
| TD-377 (LOW) | `FtpDataServer` Token Semantics and Setup Gaps. Left: Re-check expiry on `RETR`; end the session after a successful `RETR` (or refuse further ones); use 64-bit usernames or refuse to overwrite (and 2 more). See [FtpDataServer.md](docs/development/reference/source/FtpDataServer.md). |
| TD-380 (LOW) | Generation-Quality Scores Aren't Comparable or Well-Sampled. Code fix landed October 25, 2026. Left: Raise the default sample size (or report a confidence interval) once TD-359 makes generation length bounded. (TD-359's fix landed October 26, 2026… See [GenerationQualityMetrics.md](docs/development/reference/source/GenerationQualityMetrics.md). |
| TD-383 (LOW) | `GPUManager` Lifecycle Gaps (`set_device`, `cleanup`, `initialize`). Left: Make `set_device()` rebuild the stream, handle and budget (or remove it); have `cleanup()` refuse or warn while allocations are live; apply or reject… See [GPUUtils.md](docs/development/reference/source/GPUUtils.md), [GPUUtils_SYCL.md](docs/development/reference/source/GPUUtils_SYCL.md). |
| **TD-384** (HIGH) | Shared-Memory Race in GPU Softmax, LayerNorm and Cross-Entropy Kernels (CUDA and SYCL). Code fix landed October 17, 2026. Left: On a GPU host, run `compute-sanitizer --tool racecheck` (CUDA) over a training step to confirm no other shared-memory races remain. See [MatrixGPU_SYCL_cpp.md](docs/development/reference/source/MatrixGPU_SYCL_cpp.md), [MatrixGPU_cu.md](docs/development/reference/source/MatrixGPU_cu.md). |
| TD-385 (LOW) | `matrix_sum_gpu` Allocates and Synchronizes on Every Call. Code fix landed November 9, 2026. Left: Keep the loss on the device until it's actually needed on the host. Not done: the trainer reads each sample's loss (`CLAUDE.md`'s one scalar per… See [MatrixGPU_cu.md](docs/development/reference/source/MatrixGPU_cu.md). |
| TD-386 (LOW) | GPU Kernel Input Checks and Dead Batch Functions. Left: Validate targets on the host before upload (or clamp/flag on the device); document or assert alignment; add a debug-build `cudaStreamSynchronize` +… (and 1 more). See [MatrixGPU_cu.md](docs/development/reference/source/MatrixGPU_cu.md). |
| TD-387 (MEDIUM) | GPU Generation Fails on Input Longer Than `max_seq_length`. Code fix landed October 26, 2026 (with TD-365). Left: Test (on a GPU host): an over-length prompt generates without throwing and matches the CPU path. (Run on the oneAPI CPU device, and on ai-machine's… See [MatrixGPU.md](docs/development/reference/source/MatrixGPU.md), [MatrixGPU_SYCL.md](docs/development/reference/source/MatrixGPU_SYCL.md). |
| TD-389 (LOW) | `GPUMatrix` Shape Checks, Uninitialized Memory and Untested GPU KV Caches. Left: Add shape checks to the unchecked operations; zero (or document) new allocations; validate before allocating; check layer indices (and 1 more). See [MatrixGPU.md](docs/development/reference/source/MatrixGPU.md), [MatrixGPU_SYCL.md](docs/development/reference/source/MatrixGPU_SYCL.md). |
| TD-390 (MEDIUM) | SYCL Deferred Frees and Blocking Copies Stall the In-Order Queue. Code fix landed October 20, 2026. Left: Measure first on ai-machine: time one training step, and count `host_task`s per step (or use `onetrace`/`unitrace`) (and 1 more). See [GPUUtils_SYCL.md](docs/development/reference/source/GPUUtils_SYCL.md). |
| TD-392 (LOW) | SYCL `GPUMemory(0)` Throws Where CUDA Succeeds. Left: Treat `count == 0` as a valid empty allocation (skip `malloc_device`, keep `device_ptr_` null) in the SYCL `GPUMemory`, matching CUDA. See [GPUUtils_SYCL.md](docs/development/reference/source/GPUUtils_SYCL.md), [MatrixGPU_SYCL.md](docs/development/reference/source/MatrixGPU_SYCL.md). |
| TD-393 (MEDIUM) | SYCL Sub-Group Reduction Is Wrong When the Sub-Group Size Is 8. Code fix landed October 17, 2026. Left: Cover it with TD-394's CPU-device test mode (sizes 255, 256, 257, 1e6). The tests exist in `tests/matrixgpu_td003_test.cpp`; only the device-less CI… See [MatrixGPU_SYCL_cpp.md](docs/development/reference/source/MatrixGPU_SYCL_cpp.md). |
| TD-394 (MEDIUM) | GPU Kernel Tests Can't Run Without a GPU. CPU-device opt-in landed October 19, 2026. Left: Add an opt-in CPU fallback to the SYCL `GPUManager` (for example `ADAI_SYCL_ALLOW_CPU=1`, test-only), and run `gpuutilsTests`/`matrixgpuTd003Tests`… (and 2 more). See [MatrixGPU_SYCL_cpp.md](docs/development/reference/source/MatrixGPU_SYCL_cpp.md). |
| TD-395 (LOW) | `GPUMatrix` Is Duplicated Between the CUDA and SYCL Headers. Left: Add a backend `matrix_zero_gpu()` (and use `matrix_copy_device_to_device_gpu()` for `copy()`), then move `GPUMatrix` below the backend split in… (and 1 more). See [MatrixGPU_SYCL.md](docs/development/reference/source/MatrixGPU_SYCL.md). |
| TD-397 (LOW) | Hippocampal Swap File Grows Without Bound. Left: Cap the swap file (a ring of N records, or drop the oldest records when over a size limit), make the cap configurable, and log its size at startup. See [HippocampalMemory.md](docs/development/reference/source/HippocampalMemory.md). |
| TD-398 (LOW) | `HippocampalMemory` Per-Step Copies and Unchecked Saves. Left: Cache the packed key/value matrices and update them incrementally on write/evict; check stream state after saves; save both files atomically together… See [HippocampalMemory.md](docs/development/reference/source/HippocampalMemory.md). |
| TD-399 (LOW) | `IMetricsReporter` Buffering and Ordering Contract Is Undocumented. Left: Document per method whether it's pushed immediately or included in the next `end_epoch()`, and the ordering rule, in the interface (and 1 more). See [IMetricsReporter.md](docs/development/reference/source/IMetricsReporter.md). |
| TD-400 (LOW) | `NullMetricsReporter` Keeps Reporting-Only Work Switched On. Left: Add `virtual bool enabled() const` to `IMetricsReporter` (false for `NullMetricsReporter`), and have `ChatbotTrainer` check `metrics_reporter_ &&… See [IMetricsReporter.md](docs/development/reference/source/IMetricsReporter.md). |
| **TD-411** (HIGH) | `resume` (Every `trainer_service` Pass) Never Downloads Over FTP. Code fix landed October 18, 2026. Left: Download already-trained files for `retrain` against a remote registry. See [IncrementalTrainingTool.md](docs/development/reference/source/IncrementalTrainingTool.md). |
| TD-425 (LOW) | `LanguageModelHead` Hygiene. Left: Switch to `adai::Logger`; add shape checks; invalidate the GPU mirror in the setters; guard `gpu_backward()`; remove or test `get_probabilities()`. See [LanguageModelHead.md](docs/development/reference/source/LanguageModelHead.md). |
| TD-429 (LOW) | `LeJEPAEncoder` Exception Escape, Console Output and Unchecked Loads. Left: Catch tokenizer exceptions per sample in the LeJEPA loop; switch to `adai::Logger`; validate `max_seq_length` and check reads on load. See [LeJEPAEncoder.md](docs/development/reference/source/LeJEPAEncoder.md). |
| TD-432 (LOW) | `LoRAAdapter::load()` Silently Adopts the File's Shape. Left: Validate the file's shape against the adapter (throw on mismatch, leaving it unchanged), reject non-positive rank/dims, and check reads and writes. See [LoRA.md](docs/development/reference/source/LoRA.md). |
| TD-433 (LOW) | `LoRA` Initialization, Update and Config Gaps. Left: Use the project RNG (TD-368) for initialization; zero gradients in `update()` (or document); implement or remove `dropout`/`apply_to_ffn`; guard the… See [LoRA.md](docs/development/reference/source/LoRA.md). |
| TD-435 (LOW) | `Matrix` Accessor Cost, `sum()` Accuracy, `randomize()` Race and API Hygiene. Left: Inline `operator()` in the header (keep the bounds check under a debug macro, or offer an unchecked accessor) and measure the attention loops (and 2 more). See [Matrix.md](docs/development/reference/source/Matrix.md). |
| TD-436 (MEDIUM) | SYCL (`icpx`) Builds Get No `-march` or SIMD Flags. Code fix landed October 19, 2026. Left: Measure a training pass's wall time on ai-machine before/after. See [MatrixSIMD.md](docs/development/reference/source/MatrixSIMD.md). |
| TD-437 (LOW) | SIMD Self-Check Test Never Compiles and No Startup CPU Check. Left: Expose what `Matrix.cpp` was compiled with (e.g. `adai::simd::compiled_with_avx2()` defined in `Matrix.cpp`) and test that against `has_avx2()` (and 2 more). See [MatrixSIMD.md](docs/development/reference/source/MatrixSIMD.md). |
| TD-447 (LOW) | `MetricsTracker` Printer State Leak, Convergence Off-by-One and Smoothing Gaps. Left: Save and restore stream flags in the printers (or route them through `Logger`) (and 2 more). See [MetricsTracker.md](docs/development/reference/source/MetricsTracker.md). |
| TD-453 (LOW) | Hand-Written JSON Duplicated Across MNS and Metrics Code. Left: Move the MNS (server, client, CLI, GUI) and metrics (server, push client) request/response code to `nlohmann/json`, one file at a time behind the… (and 1 more). |
| TD-459 (LOW) | `ModelNameClient` Validation, Unescaping and Slow-Failure Gaps. Left: Validate architecture (all fields > 0, kind compatible with the caller) and fail clearly otherwise (and 2 more). See [ModelNameClient.md](docs/development/reference/source/ModelNameClient.md). |
| TD-462 (LOW) | `ModelNameService` Dead Datasets Proxy, Admin-Settings Race and Swallowed Persistence Errors. Left: Fix or remove the datasets proxy (record `model_id` on mark-trained, use the model's `run_group` and kind) (and 3 more). See [ModelNameService.md](docs/development/reference/source/ModelNameService.md). |
| TD-464 (LOW) | `mns_server` Startup Crash, Early-SIGTERM Hang and Fixed Log Level. Left: Fail startup cleanly when the data directory can't be created (catch in `start()` or check in `main()` and exit 1) (and 2 more). See [ModelNameServiceServer.md](docs/development/reference/source/ModelNameServiceServer.md). |
| TD-466 (LOW) | `ModelSerializer` Claims T5 Compatibility It Doesn't Have and Skips Input Validation. Decision: make it T5-faithful. Left: Implement the export-side mappings above; stop labelling non-T5 models as `T5ForConditionalGeneration` (and 5 more). See [ModelSerializer.md](docs/development/reference/source/ModelSerializer.md). |
| TD-468 (MEDIUM) | GPU Training Round-Trips Every Gradient and Weight Through a 3× Slow CPU Optimizer. Device-resident optimizer landed October 19, 2026. Left: Rewrite the update, norm and clip loops over raw row pointers (vectorizable); re-measure (and 1 more). See [Optimizer.md](docs/development/reference/source/Optimizer.md). |
| TD-472 (LOW) | `ParquetReader` Contract, Memory Use and Test Coverage Gaps. Left: Re-enable the pyarrow comparison on generated fixtures (no network) (and 1 more). See [ParquetReader.md](docs/development/reference/source/ParquetReader.md). |
| TD-473 (LOW) | `Profiler` Keeps Every Timing Forever, Sorts Under Its Lock and Uses the Wall Clock. Left: Use `std::chrono::steady_clock` (and 2 more). See [PerformanceProfiler.md](docs/development/reference/source/PerformanceProfiler.md). |
| TD-477 (LOW) | Timestamp Parsing Turns Bad Input Into 1899, Ignores Offsets and Misreads Fractions. Left: Add `parse_iso8601_utc()` returning `std::optional<time_point>`: strict digits and ranges, fractional seconds by digit position, `Z`/`±HH:MM`… (and 2 more). See [PortableTime.md](docs/development/reference/source/PortableTime.md). |
| TD-478 (LOW) | `PositionalEncoding` Width Check, Warning Spam and Duplicated Formula. Left: Throw `invalid_argument` when `input.cols != d_model` (and 1 more). See [PositionalEncoding.md](docs/development/reference/source/PositionalEncoding.md). |
| TD-480 (LOW) | `PostgresMetricsDatabase` One-Time Schema Bootstrap, Reconnects Under Lock and Unsafe Retries. Left: Bootstrap the schema on the first successful connection (any time), not just at construction (and 2 more). See [PostgresMetricsDatabase.md](docs/development/reference/source/PostgresMetricsDatabase.md). |
| TD-481 (LOW) | PPO Advantages Are Normalized Within Each Response. Deferred with RLHF. Left: Normalize advantages across a batch of rollouts (or not at all, relying on the value baseline), not within one response (and 1 more). See [PPOOptimizer.md](docs/development/reference/source/PPOOptimizer.md). |
| TD-482 (LOW) | `PPOOptimizer` Never Updates the Policy and Its Ratio Is Always 1. Deferred with RLHF. Left: Apply the clipped objective to the policy (compute new log-probs with the live model each PPO epoch and backpropagate the clipped loss, with KL early… (and 1 more). See [PPOOptimizer.md](docs/development/reference/source/PPOOptimizer.md). |
| TD-483 (LOW) | `Quantization` Asymmetric Range, Mode-Less Files and Unpacked INT4. Left: Asymmetric calibration: `min = min(min, 0)`, `max = max(max, 0)` (standard), then compute `zero_point` without clamping away the range (and 2 more). See [Quantization.md](docs/development/reference/source/Quantization.md). |
| TD-498 (LOW) | `LocalTransport`/`RemoteTransport` Divergence, Dead `model_id` and Client Hygiene. Left: Make `LocalTransport::commit_trained()` match the server (and TD-493's ownership rule); make `add_pending()` return false for duplicates (and 2 more). See [RegistryTransport.md](docs/development/reference/source/RegistryTransport.md). |
| TD-499 (LOW) | `RewardModel` Unchecked Loading, Invalid Widths and Non-He Initialization. Left: Check every read in `load()`, validate counts and per-layer shapes, and reject a file whose architecture differs from the constructed one (and 2 more). See [RewardModel.md](docs/development/reference/source/RewardModel.md). |
| TD-500 (LOW) | RLHF Rollouts Never Explore and Updates Train on a Different Sequence. Left: Return the generated token IDs from generation and train on exactly those, with the same encoder input generation used (no special tokens). See [RLHFTrainer.md](docs/development/reference/source/RLHFTrainer.md). |
| TD-501 (LOW) | `RLHFTrainer` Lacks a KL Penalty, Batch Updates and Efficient Value States. Left: Add a KL penalty (per-token reward shaping against a frozen reference model's log-probs) (and 2 more). See [RLHFTrainer.md](docs/development/reference/source/RLHFTrainer.md). |
| TD-504 (LOW) | `SIGReg` Frequency Grid, NaN Inputs and Repeated Projections. Left: Use a weighted quadrature over t; check for non-finite inputs; compute loss, gradient and diagnostics in one pass. See [SIGReg.md](docs/development/reference/source/SIGReg.md). |
| TD-513 (LOW) | `startup_sweep()` Basename Matching and Unchecked Filesystem Calls. Basename matching fixed October 19, 2026 (with TD-337). Left: Use the `std::error_code` overloads and log failures; release segment-aware. See [StartupSweep.md](docs/development/reference/source/StartupSweep.md). |
| TD-518 (LOW) | `TokenEmbedding` Unchecked Loading, Full-Matrix Updates and Validation Gaps. Left: Check every read and the file size in `load_pretrained()`; validate sizes and `set_embeddings()`'s shape (and 1 more). See [TokenEmbedding.md](docs/development/reference/source/TokenEmbedding.md). |
| TD-527 (LOW) | `TrainingMetricsAPI` Per-Poll Logging, Parsing and Unbounded History Queries. Code fix landed October 25, 2026. Left: Parse bodies with `nlohmann/json` (TD-453) and return field-specific errors; factor the per-route boilerplate. (Parsing and errors done; the… See [TrainingMetricsAPI.md](docs/development/reference/source/TrainingMetricsAPI.md). |
| **TD-531** (HIGH) | JSONL Training Reader Doesn't Decode `\uXXXX`, Corrupting Non-ASCII Text. Code fix landed October 17, 2026 (dataset scan remains). Left: Scan queued datasets for `\u` before retraining, and drop tokenized caches built from affected files so they re-tokenize. A pending file needs… See [TrainingSampleMeta.md](docs/development/reference/source/TrainingSampleMeta.md). |
| **TD-538** (HIGH) | Encoder Token Embedding and Final Norm Are Overwritten in Every Checkpoint. Code fix landed October 18, 2026. Left: Retrain rather than continuing from an existing checkpoint: none has the encoder's own embedding or norm (with TD-358). |
| TD-540 (MEDIUM) | `incremental_trainer reset` Re-queues Trained Files Locally, Not on the Registry Server. Code fix landed October 18, 2026. Left: Before the TD-358 retraining: `dataset_manager requeue --all --model <name>` on the deployment (covers TD-411's audit item) (and 1 more). |
| TD-547 (MEDIUM) | GPU Training Does the Token Embeddings on the Host. Code fix landed October 19, 2026. Left: Verify on the oneAPI CPU device against the host path, as for TD-468; measure on ai-machine. (CPU device done; ai-machine open.). |
| TD-553 (LOW) | Per-File Versions Have No Build Number. Left: Get the owner's decision on BUILD's reset behavior (monotonic recommended) (and 5 more). |
| TD-554 (LOW) | No Aggregate Version per CMake Build. Left: Owner decision: the aggregation rules above (bump-from-baseline vs. another derivation), the build-metadata contents, and where the released… (and 5 more). |
| TD-556 (LOW) | Admin-API Test Binaries Abort Under a Parallel `ctest`. Code fix landed October 26, 2026. Left: `registry_ftp_confinement_test.cpp` still picks fixed HTTP/FTP ports from 44100 upwards, inside the ephemeral range. It runs `registry_server` as a… |
| TD-558 (LOW) | Android Ops Dashboard Doesn't Show Per-Sample Quality Reports. Left: `MetricsApiService.sampleQuality(key)` for `GET api/sessions/{key}/metrics/sample-quality`, with `SampleQualityResponseDto`/… (and 4 more). |
| TD-559 (LOW) | Tizen Dashboard Doesn't Show Per-Sample Quality Reports. Left: Fetch `…/metrics/sample-quality` with `fetchJSON()` for the selected session, less often than the live metrics (reports arrive once per epoch, so… (and 5 more). |
| TD-562 (LOW) | Vocabulary Sizing and Tools Assume Lowercase Text. Code fix landed November 4, 2026 (real-data measurement pending). Left: Re-measure the token cost on the real training data (the registry pool) at the chosen vocabulary size before the rebuild, and record it next to the… |
| TD-563 (MEDIUM) | Rebuild Vocabularies and Retrain With Case, and Deploy in Order. Format refusal landed November 4, 2026; rebuild, retrain and rollout pending (TD-358). Left: Deploy binaries with TD-560 everywhere (`incremental_trainer`/`trainer_service` hosts, `chatbot_api_server`, `vocab_builder`, any tool reading… (and 4 more). |
| TD-573 (LOW) | What the Android Apps Need to Reach `stable`. Left: Setup 1: make `wear-sdk` optional for `opsdashboard` (checking the watch-face push code without it) (and 7 more). |
| TD-574 (MEDIUM) | Arc Engine Resets During Device-Resident Training: Find the GPU Work They Interrupt. First bisection run November 13, 2026; second round packaged. Left: Run `run_gpu_tests.sh --bisect-only --engine-tuning` on ai-machine (and 1 more). See [ChatbotTrainer.md](docs/development/reference/source/ChatbotTrainer.md). |
