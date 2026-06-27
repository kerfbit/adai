# Proposal: Model Name Service

**Status:** Draft  
**Date:** 2026-06-20  
**Author:** rjv717

---

## Problem Statement

Every system in ADAI — trainers, inference servers, chatbot daemons, metrics dashboards — refers
to a model by its filesystem path on the machine where it happens to live.  This works when all
processes share a disk, but it fails silently in every other situation:

- A `ChatbotAPIServer` on a remote host receives a path that resolves to `ENOENT`.
- Multiple `IncrementalTrainer` instances in distributed mode each write to different checkpoint
  directories with no shared notion of which checkpoint is the *current best model*.
- Promoting a checkpoint to "production" means manually copying a file and updating a config on
  every machine that loads that model.
- The `TrainingMetricsAPI` can track loss curves, but it has no stable identifier to correlate a
  metrics session with the physical model artifact that produced it.
- There is no system-wide answer to "what is the current production model for the chatbot role?"

The root cause is that ADAI has no *name* for a model — only a *path*.

---

## Current State

| Component | How it refers to a model |
|-----------|--------------------------|
| `ServiceConfig` | `model_path` — absolute path on the local machine |
| `IncrementalTrainer` | `model_path_`, `checkpoint_dir`, `latest_checkpoint.bin` symlink |
| `CheckpointManager` | `best_checkpoint.bin` symlink in `checkpoint_dir` |
| `ChatbotAPIServer` | `model_path` from config; must be valid on the server's filesystem |
| `TrainingMetricsAPI` | `metrics_session_key` — identifies a training run, not a model |
| `RegistryServer` | Tracks dataset files; no model identity concept |

**Gap:** Every system owns a local path alias.  There is no authoritative, stable identifier that
survives file moves, host changes, or promotion events.  Systems cannot discover which model is
current without out-of-band coordination (manual config edits, shared NFS mounts, or convention).

---

## Proposed Solution: Model Name Service (MNS)

Add a lightweight HTTP daemon — the **ModelNameService** (`mns_server`) — that is the single
authoritative source of truth for model identity across all ADAI processes.

Every model has a **stable `model_id`** (UUID) assigned at registration.  Clients resolve a
human-readable name or a *role* (e.g. `chatbot`, `reward-model`) to an `ArtifactLocation` —
either a local path or a remote URL — without caring which machine the file lives on.  State
transitions (training → candidate → production → retired) are explicit API calls; no system
infers model state from file conventions.

### Design Principles

1. **Name-first, path-second.** Systems address models by name or role; the MNS resolves the
   artifact location.  Paths are an implementation detail of the artifact store, not an identity.
2. **Single writer per state transition.** Only one caller can hold the `training` lock on a
   model at a time.  Production promotion is an explicit atomic PUT.
3. **Additive.** Existing systems continue to load models by path; MNS integration is opt-in per
   component and activated by setting `name_service_url` in config.
4. **Durable.** Model records survive `mns_server` restarts; the backing store is a JSONL file
   (Phase 1) or SQLite (Phase 2).
5. **No model weights.** The MNS stores metadata and location pointers only.  It never handles
   weight files.  Transfer of weights remains out-of-scope (see dataset-transport-proposal.md
   for the analogous file transfer design).

---

## Lifecycle State Machine

Each model record moves through a linear lifecycle.  Transitions are one-way except `retired`
(which can be explicitly revived to `candidate`).

```
   ┌──────────────┐
   │ initializing │  — record created, no trained weights yet
   └──────┬───────┘
          │  PUT /models/{name}/state  body: {"state":"training","run_id":"…"}
          ▼
   ┌──────────────┐
   │   training   │  — one IncrementalTrainer holds a training lock
   └──────┬───────┘
          │  PUT /models/{name}/state  body: {"state":"candidate","artifact":…}
          ▼
   ┌──────────────┐
   │  candidate   │  — training complete; awaiting evaluation or promotion
   └──────┬───────┘
          │  PUT /roles/{role}/production  body: {"model_name":"…"}
          ▼
   ┌──────────────┐
   │  production  │  — live; GET /roles/{role}/production resolves here
   └──────┬───────┘
          │  (new production model promoted for same role)
          ▼
   ┌──────────────┐
   │   retired    │  — superseded; weights retained but no longer served
   └──────────────┘
         │  PUT /models/{name}/state  body: {"state":"candidate"}
         └───────────────────────────────────┘  (revival — rare)
```

Rules enforced by the server:
- Only one model per role may be in state `production` at any time.  Promoting model B
  automatically transitions the current production model for that role to `retired`.
- A `training` lock is held by a `run_id`; the server rejects a second caller that tries to
  transition the same model to `training` with a different `run_id`.
- `initializing` → `candidate` (skipping `training`) is permitted for imported pre-trained
  models that were not trained by ADAI.

---

## Architecture

```
  Any Client Machine                      MNS Machine
 ┌──────────────────────────────┐        ┌───────────────────────────────────────────┐
 │  IncrementalTrainer          │        │  ModelNameService  :8083                  │
 │    register_model()  ───────►│  HTTP  │    POST /models                           │
 │    set_training()    ───────►│◄──────►│    PUT  /models/{name}/state              │
 │    promote_candidate() ─────►│        │    PUT  /roles/{role}/production          │
 │                              │        │                                           │
 │  ChatbotAPIServer            │        │  Persistent store (JSONL → SQLite)        │
 │    resolve_role("chatbot") ─►│        │    models.jsonl  — one record per line    │
 │    ◄── ArtifactLocation ─────│        │    roles.json   — role→model_id map       │
 │                              │        │                                           │
 │  TrainingMetricsAPI          │        │  In-memory index                          │
 │    tag_session(model_id) ───►│        │    unordered_map<name, ModelRecord>       │
 │                              │        │    unordered_map<role, model_id>          │
 │  RegistryServer              │        │                                           │
 │    tag_dataset(model_id) ───►│        │                                           │
 └──────────────────────────────┘        └───────────────────────────────────────────┘
```

The MNS is a peer of `RegistryServer` and `TrainingMetricsAPI` — a standalone HTTP daemon in the
same binary family, configured via the existing `config.conf` key-value format.

---

## Data Model

### `ModelRecord`

```json
{
  "model_id":    "550e8400-e29b-41d4-a716-446655440000",
  "model_name":  "adai-chatbot-v3",
  "role":        "chatbot",
  "state":       "production",
  "run_id":      "trainer-host-4201",
  "created_utc": "2026-06-20T10:00:00Z",
  "updated_utc": "2026-06-20T14:22:00Z",
  "artifact": {
    "host":      "192.168.1.19",
    "path":      "/var/adai/checkpoints/chatbot-v3-best.bin",
    "checksum":  "8388608_1718890000",
    "format":    "adai-native"
  },
  "arch": {
    "d_model":            512,
    "num_heads":          8,
    "d_ff":               2048,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "max_seq_length":     1024
  },
  "training_history": [
    {
      "run_id":           "trainer-host-4201",
      "metrics_session_key": "chatbot-v3-session-7",
      "dataset_group":    "gutenberg-en",
      "epochs":           10,
      "final_loss":       1.432,
      "started_utc":      "2026-06-20T10:00:00Z",
      "finished_utc":     "2026-06-20T14:20:00Z"
    }
  ],
  "tags": {
    "vocab_path": "/var/adai/vocab.txt",
    "owner":      "rjv717"
  }
}
```

Field notes:
- `model_id` — UUID assigned at `POST /models`; immutable.
- `model_name` — unique human-readable key; used in all subsequent API calls.
- `role` — optional logical purpose (e.g. `chatbot`, `reward-model`); drives role-based
  resolution.  A model without a role can still be resolved by name.
- `artifact.host` — the hostname where the weight file lives; empty means localhost of the
  server that registered the artifact.
- `artifact.format` — `adai-native` (default), `safetensors`, or `gguf`.
- `training_history` — append-only list; each training run appends one entry.  An imported
  pre-trained model has an empty list.

---

## HTTP API

All requests and responses use `application/json`.

### Models

#### `POST /models`
Register a new model.  Returns `409 Conflict` if `model_name` is already in use.

**Request:**
```json
{
  "model_name": "adai-chatbot-v3",
  "role":       "chatbot",
  "arch": {
    "d_model": 512, "num_heads": 8, "d_ff": 2048,
    "num_encoder_layers": 6, "num_decoder_layers": 6, "max_seq_length": 1024
  },
  "tags": { "owner": "rjv717" }
}
```

**Response `201`:**
```json
{ "model_id": "550e8400-e29b-41d4-a716-446655440000", "state": "initializing" }
```

---

#### `GET /models`
List all models.  Query params: `state=`, `role=`, `limit=` (default 50).

**Response `200`:**
```json
{ "models": [ { /* ModelRecord */ }, … ] }
```

---

#### `GET /models/{name}`
Retrieve one model by `model_name`.  Returns `404` if unknown.

**Response `200`:** full `ModelRecord` JSON.

---

#### `GET /models/{name}/resolve`
Return the artifact location for `{name}` without the full record.  Returns `404` if not found or
`409` if state is `initializing` (no artifact yet).

**Response `200`:**
```json
{
  "model_id":  "550e8400-e29b-41d4-a716-446655440000",
  "model_name": "adai-chatbot-v3",
  "state":     "production",
  "artifact": {
    "host":     "192.168.1.19",
    "path":     "/var/adai/checkpoints/chatbot-v3-best.bin",
    "checksum": "8388608_1718890000",
    "format":   "adai-native"
  }
}
```

---

#### `PUT /models/{name}/state`
Transition lifecycle state.  Returns `409` if the transition is invalid (e.g. `retired` →
`training` without explicit revival flag).

**Request — start training:**
```json
{
  "state":   "training",
  "run_id":  "trainer-host-4201",
  "metrics_session_key": "chatbot-v3-session-7"
}
```

**Request — mark candidate (training complete):**
```json
{
  "state":  "candidate",
  "run_id": "trainer-host-4201",
  "artifact": {
    "host":     "192.168.1.19",
    "path":     "/var/adai/checkpoints/chatbot-v3-best.bin",
    "checksum": "8388608_1718890000",
    "format":   "adai-native"
  },
  "training_summary": {
    "dataset_group": "gutenberg-en",
    "epochs":        10,
    "final_loss":    1.432,
    "started_utc":   "2026-06-20T10:00:00Z",
    "finished_utc":  "2026-06-20T14:20:00Z"
  }
}
```

**Request — retire explicitly:**
```json
{ "state": "retired" }
```

**Response `200`:** updated `ModelRecord`.

---

#### `DELETE /models/{name}`
Hard-delete a model record.  Only permitted when state is `retired` or `initializing`.
Returns `409` if the model is `training`, `candidate`, or `production`.
Does **not** delete weight files.

**Response `200`:**
```json
{ "deleted": true, "model_name": "adai-chatbot-v3" }
```

---

### Roles

#### `GET /roles/{role}/production`
Resolve the current production model for a role.  Returns `404` if no model is in state
`production` for that role.

**Response `200`:** same shape as `GET /models/{name}/resolve`.

---

#### `PUT /roles/{role}/production`
Promote a `candidate` model to `production` for a role.  Atomically retires the previous
production model for the same role (if any).  Returns `409` if the named model is not in state
`candidate`.

**Request:**
```json
{ "model_name": "adai-chatbot-v3" }
```

**Response `200`:**
```json
{
  "promoted":  "adai-chatbot-v3",
  "retired":   "adai-chatbot-v2",
  "role":      "chatbot"
}
```

---

#### `GET /roles`
List all roles and their current production model (if any).

**Response `200`:**
```json
{
  "roles": [
    { "role": "chatbot",      "production_model": "adai-chatbot-v3" },
    { "role": "reward-model", "production_model": null }
  ]
}
```

---

### Health

#### `GET /health`
```json
{ "status": "ok", "model_count": 12, "uptime_seconds": 3600 }
```

---

## C++ Client: `ModelNameClient`

A thin HTTP client (modelled after `RegistryTransport`) wraps the MNS API.  All methods are
blocking and throw `std::runtime_error` on non-2xx responses or network failure.

```cpp
struct ArtifactLocation {
    std::string host;
    std::string path;
    std::string checksum;
    std::string format;  // "adai-native" | "safetensors" | "gguf"
};

struct ResolvedModel {
    std::string   model_id;
    std::string   model_name;
    std::string   state;
    ArtifactLocation artifact;
};

class ModelNameClient {
public:
    explicit ModelNameClient(const std::string& server_url,
                             int timeout_ms = 5000);

    // Register a new model; returns its UUID.
    std::string register_model(const std::string& model_name,
                               const std::string& role,
                               const adai::ServiceConfig& arch,
                               const std::map<std::string,std::string>& tags = {});

    // Transition state.
    void set_training(const std::string& model_name,
                      const std::string& run_id,
                      const std::string& metrics_session_key = "");

    void set_candidate(const std::string& model_name,
                       const std::string& run_id,
                       const ArtifactLocation& artifact,
                       const std::map<std::string,std::string>& training_summary = {});

    // Resolve by name or role.
    ResolvedModel resolve_model(const std::string& model_name);
    ResolvedModel resolve_role(const std::string& role);

    // Promote a candidate to production for a role.
    void promote(const std::string& role, const std::string& model_name);
};
```

---

## Integration per Subsystem

### `IncrementalTrainer`

At the start of `train_on_files()`:

```cpp
if (!mns_client_) return;  // MNS not configured — existing path unchanged

mns_client_->set_training(config.model_name, run_id_,
                          active_session_key_);
```

At the end of `run_training()`, after best checkpoint is saved:

```cpp
if (!mns_client_) return;

ArtifactLocation loc;
loc.host     = hostname();
loc.path     = best_checkpoint_path;
loc.checksum = compute_checkpoint_checksum(best_checkpoint_path);
loc.format   = "adai-native";

mns_client_->set_candidate(config.model_name, run_id_, loc,
    { {"epochs",     std::to_string(num_epochs)},
      {"final_loss", std::to_string(best_validation_loss)} });
```

`IncrementalConfig` gains two new fields:

```cpp
std::string model_name;       // MNS model name; empty = MNS disabled
std::string mns_server_url;   // URL of the ModelNameService daemon
```

---

### `ChatbotAPIServer`

At startup, if `name_service_url` and `model_role` are set in config, resolve the artifact
before loading weights:

```cpp
if (!svc_config.name_service_url.empty()) {
    ModelNameClient mns(svc_config.name_service_url);
    auto resolved = mns.resolve_role(svc_config.model_role);
    svc_config.model_path = resolved.artifact.path;
    // If resolved.artifact.host != hostname(), trigger DataTransport fetch
    // (see dataset-transport-proposal.md for the analogous FTP pattern).
}
// existing model load path unchanged below this point
```

`ServiceConfig` gains:

```cpp
std::string name_service_url;   // URL of MNS daemon; empty = local path mode
std::string model_role;         // e.g. "chatbot"; empty = resolve by model_name
std::string model_name;         // explicit name; used when model_role is empty
```

---

### `TrainingMetricsAPI`

When a session is created, the trainer optionally passes `model_id` in the session-open request.
The metrics API stores it as a session tag.  No MNS calls are made by the metrics server itself;
it is the trainer's responsibility to pass a `model_id` obtained from the MNS.

Session-open request (extension to existing endpoint):

```json
{
  "session_key": "chatbot-v3-session-7",
  "label":       "chatbot-v3 epoch 10",
  "model_id":    "550e8400-e29b-41d4-a716-446655440000"
}
```

The `model_id` field is optional; existing clients that do not set it continue to work unchanged.

---

### `RegistryServer`

When a trainer acquires a dataset file group, it may pass `model_id` in the acquire request body:

```json
{
  "run_id":   "trainer-host-4201",
  "model_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

The registry stores `model_id` in the `DataVersion` record alongside the existing `trained` flag
so that the full audit trail — which datasets trained which model — is queryable.

`DataVersion` gains one optional field:
```cpp
std::string model_id;   // empty for files trained before MNS integration
```

No other registry endpoints change.  The registry does not call the MNS; it only stores the
`model_id` that the trainer provides.

---

## Server Implementation

`ModelNameService` is implemented in C++ as a standalone HTTP daemon in the same binary family
as `RegistryServer` and `TrainingMetricsAPI`.  It uses the same HTTP server library already
present in the project.

### Storage Layer

**Phase 1 — JSONL flat file**

```
name_service_dir/
  models.jsonl    — one ModelRecord JSON per line; new records appended
  roles.json      — { "chatbot": "550e8400-…", "reward-model": null }
```

On startup the server reads `models.jsonl` into an in-memory
`unordered_map<string, ModelRecord>` keyed by `model_name`, and a parallel
`unordered_map<string, string>` (`role_id_map`) keyed by role.

On every state-changing write, the server rewrites `roles.json` atomically (write to `.tmp`,
`rename()`), and appends a new record line to `models.jsonl` (the latest line for a given
`model_name` wins on reload).

**Phase 2 — SQLite**

Replace the JSONL store with a single `models.db` SQLite database.  Provides atomic transactions,
point queries by `model_id`/`model_name`/`role`/`state`, and full training history without
O(n) log replay on startup.

### Request Routing

```
POST   /models                       → handle_register()
GET    /models                       → handle_list()
GET    /models/{name}                → handle_get()
GET    /models/{name}/resolve        → handle_resolve()
PUT    /models/{name}/state          → handle_state_transition()
DELETE /models/{name}                → handle_delete()
GET    /roles                        → handle_list_roles()
GET    /roles/{role}/production      → handle_resolve_role()
PUT    /roles/{role}/production      → handle_promote()
GET    /health                       → handle_health()
```

All handlers are called on the HTTP thread pool; state is protected by a single
`std::shared_mutex` (readers share, writers exclusive).

---

## Configuration

New keys added to `ServiceConfig` and `config.conf`:

```toml
[name_service]
name_service_url        = "http://192.168.1.19:8083"  # clients: MNS address
name_service_port       = 8083                         # server: listen port
name_service_dir        = "name_service"               # server: storage directory
name_service_timeout_ms = 5000                         # clients: HTTP timeout

[model]
model_name = "adai-chatbot-v3"   # human-readable name registered in MNS
model_role = "chatbot"           # role used for resolve_role() at startup
```

When `name_service_url` is absent or empty, all client-side MNS calls are skipped and all
existing file-path behaviour is unchanged.

---

## Security Considerations

| Concern | Mitigation |
|---------|-----------|
| Unauthorized state transition | Phase 1: trusted internal network; Phase 2: HMAC-signed `run_id` tokens identical to dataset-transport token scheme |
| Production demotion | `PUT /roles/{role}/production` requires the named model to be in state `candidate` — a training lock cannot accidentally demote a production model |
| Record injection | `model_name` is validated against `[a-z0-9][a-z0-9\-\.]{1,127}`; role names follow the same pattern |
| Path traversal | `artifact.path` is stored as an opaque string; the MNS never opens or reads weight files |
| Concurrent promotion | The `std::shared_mutex` write lock ensures only one promotion is in flight at a time; the JSONL `rename()` is atomic at the OS level |
| Stale artifact references | `GET /models/{name}/resolve` returns the checksum; callers verify the file matches before loading weights |

---

## Implementation Phases

### Phase 1 — Core name service (MVP)

- Implement `ModelNameService` HTTP daemon in C++ (`src/ModelNameService.cpp`,
  `src/ModelNameService.hpp`).
  - JSONL flat-file storage with in-memory index.
  - Full lifecycle state machine with transition validation.
  - All HTTP endpoints listed above.
- Implement `ModelNameClient` (`src/ModelNameClient.cpp`, `src/ModelNameClient.hpp`).
- Add `model_name`, `mns_server_url` to `IncrementalConfig` and `ServiceConfig`.
- Wire `IncrementalTrainer::train_on_files()` and `run_training()` to call `set_training()` and
  `set_candidate()` when `mns_server_url` is non-empty.
- Add `name_service_url`, `model_role`, `model_name` to `ServiceConfig`; wire
  `ChatbotAPIServer` startup to call `resolve_role()` when set.
- Extend `TrainingMetricsAPI` session-open request to accept optional `model_id`.
- Build target: `adai_mns_server` (analogous to `adai_registry_server`).

### Phase 2 — Persistent store and dataset correlation

- Replace JSONL store with SQLite; full history queries without log replay.
- Extend `RegistryServer` `/acquire` endpoint to accept and store `model_id`; expose
  `GET /registry/{group}/history?model_id=…` to list dataset files used to train a model.
- Add `GET /models/{name}/datasets` to MNS — proxied query to the registry.

### Phase 3 — Security and cross-host artifact access

- HMAC-signed `run_id` tokens for state transitions (same scheme as dataset-transport Phase 3).
- If `artifact.host` differs from the client's hostname, the `ChatbotAPIServer` triggers a
  DataTransport fetch (FTP, or a future HTTP object-store backend) to download weights locally
  before loading, then calls `mns_client_->resolve_model()` to verify the checksum.
- Audit log: every registration, transition, and promotion logged with `run_id`, `model_name`,
  and operator IP.

---

## Alternatives Considered

| Alternative | Reason not chosen |
|-------------|-------------------|
| MLflow Model Registry | External Python service; adds a mandatory out-of-process dependency with a large footprint; ADAI's use case is narrower and benefits from a tight C++ integration |
| Extend `RegistryServer` with model records | Dataset registry and model registry have different lifecycles and different clients; mixing them increases coordination coupling and complicates the existing stable protocol |
| Name model by checkpoint path hash | A hash is stable but not human-readable; it cannot express role assignment or promotion semantics |
| Git-tag model versions | Requires a Git repo on every participant; impractical on ad-hoc trainer machines |
| Environment variables / config convention | Already the status quo; breaks when the config on two machines disagrees; no audit trail for promotions |
| Shared NFS mount with a well-known symlink | Works in a fixed cluster but requires infrastructure outside the ADAI process; not portable; no API for querying history |
