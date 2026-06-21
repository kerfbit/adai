# Model Name Service — Operational Manual

**Daemon:** `mns_server`  
**Default port:** 8083  
**Binary:** `build/bin/mns_server`  
**Storage:** SQLite (`<data-dir>/models.db`)

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Quick Start](#2-quick-start)
3. [Command-Line Reference](#3-command-line-reference)
4. [Configuration (config.conf)](#4-configuration-configconf)
5. [Lifecycle State Machine](#5-lifecycle-state-machine)
6. [HTTP API Reference](#6-http-api-reference)
   - 6.1 [Health](#61-health)
   - 6.2 [Register a model](#62-register-a-model)
   - 6.3 [List models](#63-list-models)
   - 6.4 [Get model record](#64-get-model-record)
   - 6.5 [Resolve artifact location](#65-resolve-artifact-location)
   - 6.6 [Transition lifecycle state](#66-transition-lifecycle-state)
   - 6.7 [Delete model record](#67-delete-model-record)
   - 6.8 [List roles](#68-list-roles)
   - 6.9 [Resolve production role](#69-resolve-production-role)
   - 6.10 [Promote to production](#610-promote-to-production)
   - 6.11 [List datasets trained on a model](#611-list-datasets-trained-on-a-model)
7. [Common curl Workflows](#7-common-curl-workflows)
8. [Data Storage Layout](#8-data-storage-layout)
9. [Integration per Subsystem](#9-integration-per-subsystem)
   - 9.1 [IncrementalTrainer](#91-incrementaltrainer)
   - 9.2 [ChatbotAPIServer](#92-chatbotapiserver)
   - 9.3 [TrainingMetricsAPI](#93-trainingmetricsapi)
   - 9.4 [RegistryServer](#94-registryserver)
10. [Port Assignments](#10-port-assignments)
11. [Startup and Shutdown](#11-startup-and-shutdown)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Purpose

The **Model Name Service** (MNS) is the single authoritative source of truth for model identity across all ADAI processes. It solves the problem of every system referring to a model by a local filesystem path — a reference that breaks silently across hosts, after file moves, and when multiple trainers compete.

The MNS assigns each model a stable UUID (`model_id`) at registration and tracks it through its entire lifecycle: `initializing → training → candidate → production → retired`. Clients resolve a human-readable name or a logical *role* (e.g. `chatbot`) to an `ArtifactLocation` without caring which machine the weight file lives on.

**What the MNS stores:**
- Model identity (UUID, name, role, state)
- Artifact location (host, path, checksum, format)
- Architecture metadata (d_model, num_heads, etc.)
- Training history (one entry per training run)
- Tags (arbitrary key-value pairs)

**What the MNS never does:**
- Read, write, or transfer weight files — it stores pointers only.
- Make calls to other services — it is a passive registry that clients call.

MNS integration is **opt-in**: all existing binaries continue to operate by filesystem path when `NAME_SERVICE_URL` is not configured.

---

## 2. Quick Start

```bash
# Build
cmake --build build --target mns_server

# Start with defaults (port 8083, data directory ./name_service)
./build/bin/mns_server

# Start with explicit options
./build/bin/mns_server \
  --port 8083 \
  --data-dir /var/adai/name_service \
  --registry-url http://192.168.1.19:8082 \
  --registry-group default

# Verify
curl http://localhost:8083/health
# {"status":"ok","model_count":0,"uptime_seconds":1}
```

---

## 3. Command-Line Reference

```
Usage: mns_server [OPTIONS]

Options:
  --port PORT           Listening port (default: 8083)
  --data-dir DIR        Storage directory for SQLite DB (default: name_service)
  --registry-url URL    Registry server URL for dataset proxy
                        (e.g. http://localhost:8082); enables GET /models/{name}/datasets
  --registry-group GRP  Registry group name to query (default: default)
  --help                Show this message
```

| Flag | Required | Default | Notes |
|------|----------|---------|-------|
| `--port` | No | `8083` | Must not conflict with metrics (8081), registry (8082), or chatbot API (8080) |
| `--data-dir` | No | `name_service` | Relative to working directory; created on first run |
| `--registry-url` | No | *(empty)* | Required only if clients call `GET /models/{name}/datasets` |
| `--registry-group` | No | `default` | Only meaningful when `--registry-url` is set |

---

## 4. Configuration (config.conf)

These keys are read by all ADAI binaries that use the standard `config.conf` format. Trainer and server binaries consume client-side keys; only the `mns_server` itself uses the server-side keys.

### Server-side keys (mns_server)

| Key | Default | Description |
|-----|---------|-------------|
| `NAME_SERVICE_PORT` | `8083` | Port the daemon listens on |
| `NAME_SERVICE_DIR` | `name_service` | Directory for `models.db` |

### Client-side keys (trainer, chatbot API server, etc.)

| Key | Default | Description |
|-----|---------|-------------|
| `NAME_SERVICE_URL` | *(empty)* | Full URL of the MNS daemon, e.g. `http://192.168.1.19:8083`. Empty = MNS disabled for this process |
| `NAME_SERVICE_TIMEOUT_MS` | `5000` | HTTP timeout for MNS calls (milliseconds) |
| `MODEL_NAME` | *(empty)* | Human-readable model name registered in MNS (trainer only) |
| `MODEL_ROLE` | *(empty)* | Role to resolve at startup, e.g. `chatbot` (chatbot API server only). When set, overrides `MODEL_NAME` for resolution |

### Example config.conf — trainer node

```ini
# Training targets (normal settings)
VOCAB_PATH           = /var/adai/vocab.txt
MODEL_PATH           = /var/adai/checkpoints/chatbot.bin
METRICS_SERVER_URL   = http://192.168.1.19:8081
REGISTRY_SERVER_URL  = http://192.168.1.19:8082

# MNS integration
NAME_SERVICE_URL     = http://192.168.1.19:8083
NAME_SERVICE_TIMEOUT_MS = 5000
MODEL_NAME           = adai-chatbot-v3
```

### Example config.conf — chatbot API server

```ini
NAME_SERVICE_URL     = http://192.168.1.19:8083
MODEL_ROLE           = chatbot
NAME_SERVICE_TIMEOUT_MS = 5000
```

When `MODEL_ROLE` is set, the chatbot API server calls `GET /roles/{role}/production` at startup and loads the artifact path returned. `MODEL_PATH` in config becomes the fallback if MNS resolution fails.

---

## 5. Lifecycle State Machine

Each model record moves through a linear state machine. Transitions are enforced server-side; invalid transitions return `409 Conflict`.

```
   ┌──────────────┐
   │ initializing │  record created, no trained weights yet
   └──────┬───────┘
          │  PUT /models/{name}/state {"state":"training","run_id":"…"}
          ▼
   ┌──────────────┐
   │   training   │  one IncrementalTrainer holds the training lock
   └──────┬───────┘
          │  PUT /models/{name}/state {"state":"candidate","artifact":{…}}
          ▼
   ┌──────────────┐
   │  candidate   │  training complete; awaiting evaluation or promotion
   └──────┬───────┘
          │  PUT /roles/{role}/production {"model_name":"…"}
          ▼
   ┌──────────────┐
   │  production  │  live; GET /roles/{role}/production resolves here
   └──────┬───────┘
          │  (new model promoted for same role — automatic)
          ▼
   ┌──────────────┐
   │   retired    │  superseded; record retained, weights preserved
   └──────┬───────┘
          │  PUT /models/{name}/state {"state":"candidate"}  (rare revival)
          └──────────────────────────────────────────────────┘
```

**Rules enforced by the server:**

- Only one model per role can be in state `production` at any time. Promoting model B automatically retires the current production model for that role.
- The `training` lock is owned by a `run_id`. A second caller attempting to transition the same model to `training` with a *different* `run_id` receives `409`. The same `run_id` is idempotent.
- `initializing → candidate` (skipping `training`) is permitted for importing pre-trained models that were not trained by ADAI.
- `DELETE /models/{name}` is only permitted in states `initializing` or `retired`.

---

## 6. HTTP API Reference

All requests and responses use `Content-Type: application/json`.

---

### 6.1 Health

**`GET /health`**

```bash
curl http://localhost:8083/health
```

```json
{"status":"ok","model_count":7,"uptime_seconds":3600}
```

Returns `200` when the server is running. Use this for load-balancer liveness checks.

---

### 6.2 Register a model

**`POST /models`**

Creates a new model record and assigns a stable UUID. Returns `409 Conflict` if `model_name` is already registered.

**Model name rules:** must match `[a-z0-9][a-z0-9\-\.]{1,127}` — lowercase letters, digits, hyphens, and dots only.

**Request (minimal):**
```bash
curl -s -X POST http://localhost:8083/models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"adai-chatbot-v3"}'
```

**Request (full):**
```json
{
  "model_name": "adai-chatbot-v3",
  "role":       "chatbot",
  "arch": {
    "d_model":            512,
    "num_heads":          8,
    "d_ff":               2048,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "max_seq_length":     1024
  },
  "tags": {
    "owner":      "rjv717",
    "vocab_path": "/var/adai/vocab.txt"
  }
}
```

**Response `201 Created`:**
```json
{
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "state":    "initializing"
}
```

**Error responses:**

| Status | Meaning |
|--------|---------|
| `400` | Missing or invalid `model_name` |
| `409` | `model_name` already registered |

---

### 6.3 List models

**`GET /models`**

Returns all model records. Supports optional query parameters.

| Query param | Description |
|-------------|-------------|
| `state=`    | Filter by state: `initializing`, `training`, `candidate`, `production`, `retired` |
| `role=`     | Filter by role name |
| `limit=`    | Maximum records to return (default: 50) |

```bash
# All models
curl http://localhost:8083/models

# Only production models
curl 'http://localhost:8083/models?state=production'

# Chatbot role, limit 10
curl 'http://localhost:8083/models?role=chatbot&limit=10'
```

**Response `200`:**
```json
{
  "models": [
    { /* full ModelRecord */ },
    { /* full ModelRecord */ }
  ]
}
```

---

### 6.4 Get model record

**`GET /models/{name}`**

Returns the full record for one model.

```bash
curl http://localhost:8083/models/adai-chatbot-v3
```

**Response `200` — full ModelRecord:**
```json
{
  "model_id":    "550e8400-e29b-41d4-a716-446655440000",
  "model_name":  "adai-chatbot-v3",
  "role":        "chatbot",
  "state":       "production",
  "run_id":      "",
  "created_utc": "2026-06-20T10:00:00Z",
  "updated_utc": "2026-06-20T14:22:00Z",
  "artifact": {
    "host":      "192.168.1.19",
    "path":      "/var/adai/checkpoints/chatbot-v3-best.bin",
    "checksum":  "8388608_1718890000",
    "format":    "adai-native"
  },
  "arch": {
    "d_model": 512, "num_heads": 8, "d_ff": 2048,
    "num_encoder_layers": 6, "num_decoder_layers": 6, "max_seq_length": 1024
  },
  "training_history": [
    {
      "run_id":              "trainer-host-4201",
      "metrics_session_key": "chatbot-v3-session-7",
      "dataset_group":       "gutenberg-en",
      "epochs":              10,
      "final_loss":          1.432,
      "started_utc":         "2026-06-20T10:00:00Z",
      "finished_utc":        "2026-06-20T14:20:00Z"
    }
  ],
  "tags": {
    "owner":      "rjv717",
    "vocab_path": "/var/adai/vocab.txt"
  }
}
```

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | Model not found |

---

### 6.5 Resolve artifact location

**`GET /models/{name}/resolve`**

Returns only the artifact location without the full record. Intended for ChatbotAPIServer startup: resolve a name to a path before loading weights.

```bash
curl http://localhost:8083/models/adai-chatbot-v3/resolve
```

**Response `200`:**
```json
{
  "model_id":   "550e8400-e29b-41d4-a716-446655440000",
  "model_name": "adai-chatbot-v3",
  "state":      "production",
  "artifact": {
    "host":     "192.168.1.19",
    "path":     "/var/adai/checkpoints/chatbot-v3-best.bin",
    "checksum": "8388608_1718890000",
    "format":   "adai-native"
  }
}
```

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | Model not found |
| `409` | Model is in `initializing` state — no artifact attached yet |

---

### 6.6 Transition lifecycle state

**`PUT /models/{name}/state`**

Moves a model through its lifecycle. The set of accepted transitions depends on current state (see [Section 5](#5-lifecycle-state-machine)).

#### Start training

Acquires the training lock. The `run_id` must be unique per training run (e.g. `"trainer-hostname-<pid>"`).

```bash
curl -s -X PUT http://localhost:8083/models/adai-chatbot-v3/state \
  -H 'Content-Type: application/json' \
  -d '{
    "state":               "training",
    "run_id":              "trainer-host-4201",
    "metrics_session_key": "chatbot-v3-session-7"
  }'
```

#### Mark candidate (training complete)

Releases the training lock and attaches the artifact location. The `artifact` block and `training_summary` are both optional but strongly recommended.

```bash
curl -s -X PUT http://localhost:8083/models/adai-chatbot-v3/state \
  -H 'Content-Type: application/json' \
  -d '{
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
      "epochs":        "10",
      "final_loss":    "1.432"
    }
  }'
```

#### Retire explicitly

```bash
curl -s -X PUT http://localhost:8083/models/adai-chatbot-v3/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"retired"}'
```

#### Revive retired model to candidate

```bash
curl -s -X PUT http://localhost:8083/models/adai-chatbot-v3/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"candidate"}'
```

**Response `200`:** updated full ModelRecord.

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | Model not found |
| `409` | Transition not permitted from current state; or training lock held by a different `run_id` |

---

### 6.7 Delete model record

**`DELETE /models/{name}`**

Hard-deletes the model record from the database. Only permitted when state is `initializing` or `retired`. Does **not** delete weight files.

```bash
curl -s -X DELETE http://localhost:8083/models/adai-chatbot-old
```

**Response `200`:**
```json
{"deleted":true,"model_name":"adai-chatbot-old"}
```

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | Model not found |
| `409` | Model is in `training`, `candidate`, or `production` state — retire it first |

---

### 6.8 List roles

**`GET /roles`**

Returns all known roles and their current production model (if any).

```bash
curl http://localhost:8083/roles
```

**Response `200`:**
```json
{
  "roles": [
    {"role": "chatbot",      "production_model": "adai-chatbot-v3"},
    {"role": "reward-model", "production_model": null}
  ]
}
```

---

### 6.9 Resolve production role

**`GET /roles/{role}/production`**

Returns the artifact location for the current production model assigned to a role. This is the primary endpoint used by `ChatbotAPIServer` at startup.

```bash
curl http://localhost:8083/roles/chatbot/production
```

**Response `200`** (same shape as `GET /models/{name}/resolve`):
```json
{
  "model_id":   "550e8400-e29b-41d4-a716-446655440000",
  "model_name": "adai-chatbot-v3",
  "state":      "production",
  "artifact": {
    "host":     "192.168.1.19",
    "path":     "/var/adai/checkpoints/chatbot-v3-best.bin",
    "checksum": "8388608_1718890000",
    "format":   "adai-native"
  }
}
```

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | No model currently in `production` for this role |

---

### 6.10 Promote to production

**`PUT /roles/{role}/production`**

Atomically promotes a `candidate` model to `production` for a role. If another model is already in `production` for that role, it is automatically transitioned to `retired`.

```bash
curl -s -X PUT http://localhost:8083/roles/chatbot/production \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"adai-chatbot-v3"}'
```

**Response `200`:**
```json
{
  "promoted": "adai-chatbot-v3",
  "retired":  "adai-chatbot-v2",
  "role":     "chatbot"
}
```

The `retired` field is `null` when no previous production model existed for the role.

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | Named model not found |
| `409` | Model is not in `candidate` state |

---

### 6.11 List datasets trained on a model

**`GET /models/{name}/datasets`**

Returns dataset files from the `RegistryServer` that were used to train this model, filtered by the model's UUID. Requires the MNS server to be started with `--registry-url`.

```bash
curl http://localhost:8083/models/adai-chatbot-v3/datasets
```

The response is the raw JSON from `GET /registry/{group}/history?model_id={uuid}` on the registry server.

**Error responses:**

| Status | Meaning |
|--------|---------|
| `404` | Model not found in MNS |
| `501` | MNS was not started with `--registry-url` |

---

## 7. Common curl Workflows

### Full lifecycle from scratch

```bash
MNS=http://localhost:8083

# 1. Register the model
curl -s -X POST $MNS/models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"adai-chatbot-v3","role":"chatbot"}'

# 2. Start training (called automatically by IncrementalTrainer when NAME_SERVICE_URL is set)
curl -s -X PUT $MNS/models/adai-chatbot-v3/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"training","run_id":"run-$(hostname)-$$"}'

# 3. Training complete — attach artifact
curl -s -X PUT $MNS/models/adai-chatbot-v3/state \
  -H 'Content-Type: application/json' \
  -d '{
    "state":"candidate",
    "run_id":"run-myhost-12345",
    "artifact":{
      "path":"/var/adai/checkpoints/chatbot-v3-best.bin",
      "checksum":"8388608_1718890000",
      "format":"adai-native"
    }
  }'

# 4. Promote to production
curl -s -X PUT $MNS/roles/chatbot/production \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"adai-chatbot-v3"}'

# 5. Verify resolution
curl $MNS/roles/chatbot/production
```

### Check which model is currently serving

```bash
curl -s http://localhost:8083/roles/chatbot/production | python3 -m json.tool
```

### View training history for a model

```bash
curl -s http://localhost:8083/models/adai-chatbot-v3 \
  | python3 -c "import sys,json; h=json.load(sys.stdin)['training_history']; [print(e['run_id'],e['final_loss'],e['finished_utc']) for e in h]"
```

### List all candidate models awaiting promotion

```bash
curl -s 'http://localhost:8083/models?state=candidate' \
  | python3 -c "import sys,json; [print(m['model_name']) for m in json.load(sys.stdin)['models']]"
```

### Import a pre-trained model (skip training state)

```bash
MNS=http://localhost:8083
curl -s -X POST $MNS/models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"pretrained-base","role":"chatbot"}'

# Jump directly from initializing to candidate
curl -s -X PUT $MNS/models/pretrained-base/state \
  -H 'Content-Type: application/json' \
  -d '{
    "state":   "candidate",
    "artifact":{"path":"/var/adai/pretrained/base.bin","format":"adai-native"}
  }'
```

### Retire and clean up an old model

```bash
MNS=http://localhost:8083

# Retire first (required if currently candidate or production after demotion)
curl -s -X PUT $MNS/models/adai-chatbot-v1/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"retired"}'

# Then hard-delete the record
curl -s -X DELETE $MNS/models/adai-chatbot-v1
```

---

## 8. Data Storage Layout

The MNS stores all state in a single SQLite database. The file is created on first start.

```
<data-dir>/
  models.db          — SQLite WAL-mode database (primary store)
  models.db-wal      — WAL write-ahead log (normal during operation)
  models.db-shm      — shared memory file (normal during operation)

  # Legacy Phase 1 files (auto-migrated to models.db on first SQLite start)
  models.jsonl       — one ModelRecord JSON per line (import source only)
  roles.json         — {"role": "model_name"} map (import source only)
```

### Database schema

```sql
-- One row per registered model
CREATE TABLE models (
    model_id           TEXT PRIMARY KEY,
    model_name         TEXT UNIQUE NOT NULL,
    role               TEXT DEFAULT '',
    state              TEXT NOT NULL DEFAULT 'initializing',
    run_id             TEXT DEFAULT '',
    created_utc        TEXT NOT NULL,
    updated_utc        TEXT NOT NULL,
    artifact_host      TEXT DEFAULT '',
    artifact_path      TEXT DEFAULT '',
    artifact_checksum  TEXT DEFAULT '',
    artifact_format    TEXT DEFAULT 'adai-native',
    d_model            INTEGER DEFAULT 0,
    num_heads          INTEGER DEFAULT 0,
    d_ff               INTEGER DEFAULT 0,
    num_encoder_layers INTEGER DEFAULT 0,
    num_decoder_layers INTEGER DEFAULT 0,
    max_seq_length     INTEGER DEFAULT 0,
    tags_json          TEXT DEFAULT '{}'
);

-- Append-only training history; one row per training run
CREATE TABLE training_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT NOT NULL,
    run_id              TEXT DEFAULT '',
    metrics_session_key TEXT DEFAULT '',
    dataset_group       TEXT DEFAULT '',
    epochs              INTEGER DEFAULT 0,
    final_loss          REAL DEFAULT 0.0,
    started_utc         TEXT DEFAULT '',
    finished_utc        TEXT DEFAULT ''
);

-- Current production model per role
CREATE TABLE roles (
    role        TEXT PRIMARY KEY,
    model_name  TEXT NOT NULL
);
```

**Migration from Phase 1 JSONL:** On first startup with an empty `models` table, the server automatically reads `models.jsonl` and `roles.json` (if present) and imports them into SQLite. Subsequent starts skip migration.

**Backup:** To back up all MNS state, copy `models.db` while the server is stopped, or use `sqlite3 models.db ".backup models.db.bak"` while the server is running (WAL mode makes online backups safe).

**Inspect the database directly:**
```bash
sqlite3 name_service/models.db "SELECT model_name, state, artifact_path FROM models"
sqlite3 name_service/models.db "SELECT * FROM training_history WHERE model_name='adai-chatbot-v3'"
sqlite3 name_service/models.db "SELECT * FROM roles"
```

---

## 9. Integration per Subsystem

### 9.1 IncrementalTrainer

When `NAME_SERVICE_URL` and `MODEL_NAME` are both set in `config.conf`, the trainer wires MNS calls automatically. No code changes are required.

**What happens automatically:**

1. At the start of each training run, the trainer calls `PUT /models/{name}/state` with `"state":"training"` and its `run_id`. This acquires the training lock in the MNS.
2. On successful training completion (after `save_model()`), the trainer calls `PUT /models/{name}/state` with `"state":"candidate"` and attaches the checkpoint path as the artifact.
3. Both calls are guarded with `try/catch` — a network failure logs a warning but does not abort training.

**config.conf keys for the trainer:**

```ini
NAME_SERVICE_URL = http://192.168.1.19:8083
MODEL_NAME       = adai-chatbot-v3
```

The model must already be registered in the MNS before the first training run. Register it once with `POST /models`.

**IncrementalConfig fields (when constructing trainer programmatically):**

```cpp
IncrementalConfig cfg;
cfg.mns_server_url  = "http://192.168.1.19:8083";
cfg.mns_model_name  = "adai-chatbot-v3";
```

---

### 9.2 ChatbotAPIServer

When `NAME_SERVICE_URL` is set, the chatbot API server resolves the artifact location from MNS at startup before loading model weights.

**Resolution order:**
1. If `MODEL_ROLE` is set → calls `GET /roles/{role}/production` to get the current production artifact.
2. Else if `MODEL_NAME` is set → calls `GET /models/{name}/resolve`.
3. On any MNS failure (network error, 404, 409) → falls back to `MODEL_PATH` from config and logs a warning.

**config.conf for role-based resolution:**

```ini
NAME_SERVICE_URL     = http://192.168.1.19:8083
MODEL_ROLE           = chatbot
NAME_SERVICE_TIMEOUT_MS = 5000
# MODEL_PATH is the fallback if MNS is unreachable
MODEL_PATH           = /var/adai/checkpoints/latest.bin
```

This allows the chatbot API server to automatically pick up a newly promoted model on the next restart without any config change.

---

### 9.3 TrainingMetricsAPI

When the trainer starts a metrics session, it can pass the `model_id` (UUID from MNS) in the session-start request body. The metrics API stores it in the session's `config_snapshot`.

**Session start with model_id:**
```json
{
  "session_id":    7,
  "total_epochs":  10,
  "total_samples": 50000,
  "model_id":      "550e8400-e29b-41d4-a716-446655440000"
}
```

This correlates a metrics session with the MNS model record. The `model_id` field is optional; omitting it leaves backward compatibility intact.

The TrainingMetricsAPI does **not** make any MNS API calls itself — the trainer is responsible for obtaining the `model_id` from MNS and passing it along.

---

### 9.4 RegistryServer

When the trainer acquires dataset files for a training run, it includes `model_id` in the `POST /registry/{group}/trained` request body:

```json
{
  "run_id":   "trainer-host-4201",
  "model_id": "550e8400-e29b-41d4-a716-446655440000",
  "files":    ["/data/corpus/book1.txt"],
  "samples":  [42000]
}
```

The registry stores `model_id` in the `DataVersion` record alongside the `trained` flag. This creates a full audit trail: which datasets trained which model, queryable via `GET /registry/{group}/history?model_id={uuid}`.

The `model_id` field is optional — existing trainer code that does not set it continues to work; those registry entries simply have an empty `model_id`.

When the MNS is started with `--registry-url`, `GET /models/{name}/datasets` proxies this query transparently.

---

## 10. Port Assignments

| Service | Default port |
|---------|-------------|
| `chatbot_api_server` | 8080 |
| `metrics_api_server` | 8081 |
| `registry_server` | 8082 |
| `mns_server` | **8083** |

All four daemons can run on the same host without conflict.

---

## 11. Startup and Shutdown

### Manual

```bash
# Start in foreground (Ctrl+C to stop)
./build/bin/mns_server --port 8083 --data-dir /var/adai/name_service

# Start in background
nohup ./build/bin/mns_server \
  --port 8083 \
  --data-dir /var/adai/name_service \
  --registry-url http://localhost:8082 \
  > /var/log/adai/mns_server.log 2>&1 &

echo $! > /var/run/adai/mns_server.pid
```

### Graceful shutdown

The server handles `SIGINT` and `SIGTERM`. In-flight requests complete before the process exits; no data is lost because SQLite WAL mode persists writes before the response is sent.

```bash
# Graceful stop
kill -TERM $(cat /var/run/adai/mns_server.pid)

# Verify stopped
curl http://localhost:8083/health   # should fail with "connection refused"
```

### systemd unit (example)

```ini
[Unit]
Description=ADAI Model Name Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=adai
ExecStart=/opt/adai/bin/mns_server \
    --port 8083 \
    --data-dir /var/adai/name_service \
    --registry-url http://localhost:8082
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable adai-mns
sudo systemctl start adai-mns
sudo systemctl status adai-mns
```

---

## 12. Troubleshooting

### Server fails to start — port in use

```
Error: failed to bind to port 8083
```

Check what is using the port and either stop it or use `--port` to pick a different port:

```bash
ss -tlnp | grep 8083
./build/bin/mns_server --port 18083
```

### `GET /health` returns connection refused

The server is not running. Check the process:
```bash
pgrep -a mns_server
```

### `POST /models` returns `400 Bad Request`

Model name failed validation. Names must be lowercase letters, digits, hyphens, or dots; must not start with a hyphen or dot; and must be 2–128 characters. Example of invalid names: `MyModel`, `_chatbot`, `v1`.

### `PUT /models/{name}/state` returns `409` when starting training

Another process holds the training lock — a different `run_id` is in the `training` state for this model. Either:
- The previous training run crashed without sending the `candidate` transition. Manually advance: `PUT /models/{name}/state {"state":"candidate"}` (or `"retired"` then `"candidate"`) to release the lock.
- A legitimate concurrent trainer is running. Wait for it to finish.

### `GET /roles/{role}/production` returns `404`

No model is currently in state `production` for that role. Either promote a candidate with `PUT /roles/{role}/production`, or check the role name spelling.

### `GET /models/{name}/datasets` returns `501`

The MNS was not started with `--registry-url`. Restart it with `--registry-url http://<registry-host>:8082`.

### SQLite database corruption

If the server exits uncleanly and the WAL log is incomplete, SQLite may not be able to open the database. Run:

```bash
sqlite3 /var/adai/name_service/models.db "PRAGMA integrity_check;"
sqlite3 /var/adai/name_service/models.db "PRAGMA wal_checkpoint(FULL);"
```

If the database is unrecoverable, the server recreates an empty schema on next start. To restore from a backup:
```bash
cp models.db.bak models.db
```

### Model record shows stale artifact path after checkpoint was moved

The MNS stores the path the trainer reported; it never re-verifies file existence. Update the artifact by advancing through `training` → `candidate` again, or by sending a direct `PUT /models/{name}/state` with the corrected `artifact.path` in the candidate transition.

### Trainer sends MNS calls but they are silently ignored

Check that both `NAME_SERVICE_URL` and `MODEL_NAME` are set in the trainer's `config.conf`. Either missing causes the trainer to skip all MNS calls without error.
