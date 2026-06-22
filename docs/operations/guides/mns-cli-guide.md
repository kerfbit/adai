# mns_cli Operations Manual

**Binary:** `build/bin/mns_cli` (portable: `build/portable/bin/mns_cli`)
**Requires:** running `mns_server` daemon
**Default server:** `http://localhost:8083`

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Installation](#2-installation)
3. [Server Connection](#3-server-connection)
4. [Command Reference](#4-command-reference)
   - 4.1 [health](#41-health)
   - 4.2 [list](#42-list)
   - 4.3 [get](#43-get)
   - 4.4 [register](#44-register)
   - 4.5 [resolve](#45-resolve)
   - 4.6 [set-training](#46-set-training)
   - 4.7 [set-candidate](#47-set-candidate)
   - 4.8 [promote](#48-promote)
   - 4.9 [roles](#49-roles)
   - 4.10 [resolve-role](#410-resolve-role)
   - 4.11 [delete](#411-delete)
5. [Model Lifecycle](#5-model-lifecycle)
6. [Workflows](#6-workflows)
   - 6.1 [Register, train, and promote a new model](#61-register-train-and-promote-a-new-model)
   - 6.2 [Re-train an existing model](#62-re-train-an-existing-model)
   - 6.3 [Roll back production to a previous model](#63-roll-back-production-to-a-previous-model)
   - 6.4 [Retire and clean up old models](#64-retire-and-clean-up-old-models)
   - 6.5 [Audit what is registered](#65-audit-what-is-registered)
7. [Configuration](#7-configuration)
8. [Exit Codes](#8-exit-codes)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Purpose

`mns_cli` is the operator's command-line interface for the ADAI Model Name Service. It replaces ad-hoc `curl` commands with a discoverable, self-documenting tool for:

- Registering new models and recording their architecture.
- Driving the lifecycle state machine (initializing, training, candidate, production, retired).
- Promoting candidate models to production for a role.
- Querying what is registered and where each model's weights live.

The CLI talks to the same HTTP API used by `incremental_trainer`, `chatbot_api_server`, and `metrics_api_server`. It does not access the SQLite database directly and can run from any machine that can reach the MNS port.

---

## 2. Installation

`mns_cli` is built automatically when `adai_mns` is available (requires `cpp-httplib` and `SQLite3`):

```bash
cmake --preset portable
cmake --build --preset portable --target mns_cli
```

The server bundle install script (`scripts/install_server_bundle.sh`) copies `mns_cli` to `/opt/adai/bin/` alongside the three server daemons.

---

## 3. Server Connection

`mns_cli` determines which MNS server to contact using the following priority (highest wins):

| Priority | Source | Example |
|----------|--------|---------|
| 1 | `--url` flag | `mns_cli --url http://10.0.0.5:8083 list` |
| 2 | `NAME_SERVICE_URL` in config file or environment | `NAME_SERVICE_URL=http://10.0.0.5:8083` |
| 3 | Built-in default | `http://localhost:8083` |

Config file discovery: `--config <path>` > `./config.conf` > `/etc/adai/config.conf`.

All commands print the raw JSON response from the MNS server to stdout and set the exit code based on the HTTP status (0 for 2xx, 1 otherwise). Pipe through `jq` for formatted output:

```bash
mns_cli list | jq .
```

---

## 4. Command Reference

### 4.1 health

Check that the MNS server is reachable and running.

```bash
mns_cli health
```

**Output:**
```json
{"status":"ok","uptime_seconds":3421}
```

Use this as a smoke test after installation or service restart.

---

### 4.2 list

List registered models, optionally filtered by state, role, or count.

```
mns_cli list [--state STATE] [--role ROLE] [--limit N]
```

| Option | Description |
|--------|-------------|
| `--state` | Filter by lifecycle state: `initializing`, `training`, `candidate`, `production`, `retired` |
| `--role` | Filter by role name (e.g. `chatbot`) |
| `--limit` | Maximum number of records to return |

**Examples:**

```bash
# All models
mns_cli list

# Only models currently in production
mns_cli list --state production

# Chatbot-role models, most recent 5
mns_cli list --role chatbot --limit 5
```

---

### 4.3 get

Retrieve the full record for a single model by name.

```
mns_cli get <name>
```

**Example:**

```bash
mns_cli get adai-chatbot-v3
```

The response includes identity, state, artifact location, architecture metadata, training history, and tags.

---

### 4.4 register

Register a new model in the MNS. The model enters the `initializing` state and is assigned a UUID.

```
mns_cli register <name> <role> [architecture options] [--tag key=value ...]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--d-model N` | Model dimension | From config.conf (`D_MODEL`) |
| `--num-heads N` | Attention heads | From config.conf (`NUM_HEADS`) |
| `--d-ff N` | Feed-forward dimension | From config.conf (`D_FF`) |
| `--encoder-layers N` | Encoder layer count | From config.conf (`NUM_ENCODER_LAYERS`) |
| `--decoder-layers N` | Decoder layer count | From config.conf (`NUM_DECODER_LAYERS`) |
| `--max-seq-length N` | Maximum sequence length | From config.conf (`MAX_SEQ_LENGTH`) |
| `--tag key=value` | Arbitrary tag (repeatable) | none |

Architecture parameters default to values from `config.conf`, so a typical registration only needs the name and role:

```bash
# Uses architecture from config.conf
mns_cli register adai-chatbot-v3 chatbot

# Override specific architecture values
mns_cli register adai-chatbot-v3 chatbot --d-model 256 --num-heads 8

# Attach tags for tracking
mns_cli register adai-chatbot-v3 chatbot \
  --tag dataset=minipile-v2 \
  --tag owner=rodney
```

**Response:**
```json
{"model_id":"550e8400-e29b-41d4-a716-446655440000","state":"initializing"}
```

**Errors:**

| Situation | HTTP status | Meaning |
|-----------|-------------|---------|
| Name already taken | 409 | A model with this name already exists |
| Missing name/role | 400 | Both `<name>` and `<role>` are required |

---

### 4.5 resolve

Look up a model's artifact location and current state by name.

```
mns_cli resolve <name>
```

**Example:**

```bash
mns_cli resolve adai-chatbot-v3
```

**Response:**
```json
{
  "model_id":   "550e8400-...",
  "model_name": "adai-chatbot-v3",
  "state":      "production",
  "artifact": {
    "host":     "192.168.1.19",
    "path":     "/opt/adai/checkpoints/chatbot-v3-best.bin",
    "checksum": "8388608_1718890000",
    "format":   "adai-native"
  }
}
```

This is the same endpoint that `chatbot_api_server` calls at startup to find its weight file.

---

### 4.6 set-training

Transition a model from `initializing` or `candidate` to the `training` state. This acquires the training lock: only one `run_id` can hold a model in `training` at a time.

```
mns_cli set-training <name> <run-id> [session-key]
```

| Argument | Description |
|----------|-------------|
| `<name>` | Model name |
| `<run-id>` | Unique run identifier (e.g. `session-42`, a UUID, or `hostname-pid`) |
| `[session-key]` | Optional metrics session key for cross-referencing with `metrics_api_server` |

**Example:**

```bash
mns_cli set-training adai-chatbot-v3 run-42 metrics-key-abc
```

**Errors:**

| Situation | HTTP status | Meaning |
|-----------|-------------|---------|
| Different run_id holds the lock | 409 | Another training run already owns this model |
| Invalid source state | 409 | Model is in `production` or `retired` (must retire first or start from candidate) |

The `incremental_trainer` calls this automatically when `MNS_SERVER_URL` and `MODEL_NAME` are configured.

---

### 4.7 set-candidate

Transition a model from `training` to `candidate` state, attaching the artifact location and optional training summary. This releases the training lock.

```
mns_cli set-candidate <name> <run-id> [options...]
```

| Option | Description |
|--------|-------------|
| `--artifact-path PATH` | Absolute path to weight file |
| `--artifact-host HOST` | Hostname where the file lives (empty = localhost) |
| `--artifact-checksum CHK` | Opaque checksum string |
| `--artifact-format FMT` | `adai-native` (default), `safetensors`, or `gguf` |
| `--summary key=value` | Training summary entry (repeatable: `epochs=15`, `final_loss=0.42`, etc.) |

**Example:**

```bash
mns_cli set-candidate adai-chatbot-v3 run-42 \
  --artifact-path /opt/adai/checkpoints/session_3_best.bin \
  --summary epochs=15 \
  --summary final_loss=0.38 \
  --summary dataset_group=training_sessions
```

The `incremental_trainer` calls this automatically after a successful training session.

**Valid source states:** `training`, `initializing` (for direct imports), `retired` (for revival).

---

### 4.8 promote

Promote a `candidate` model to `production` for a role. If another model already holds `production` for that role, it is automatically transitioned to `retired`.

```
mns_cli promote <role> <model-name>
```

**Example:**

```bash
mns_cli promote chatbot adai-chatbot-v3
```

**Response:**
```json
{
  "promoted": "adai-chatbot-v3",
  "retired":  "adai-chatbot-v2",
  "role":     "chatbot"
}
```

The `retired` field is `null` when no previous production model existed.

**Errors:**

| Situation | HTTP status | Meaning |
|-----------|-------------|---------|
| Model not in `candidate` state | 409 | Only candidates can be promoted |
| Model not found | 404 | Check the name spelling with `mns_cli list` |

---

### 4.9 roles

List all known roles and their current production model.

```bash
mns_cli roles
```

**Response:**
```json
{
  "roles": [
    {"role": "chatbot",      "production_model": "adai-chatbot-v3"},
    {"role": "reward-model", "production_model": null}
  ]
}
```

A role appears once any model has been registered with that role name. The `production_model` field is `null` until a candidate is promoted.

---

### 4.10 resolve-role

Resolve the production model for a role. Returns the artifact location, just like `resolve`, but looked up by role instead of name.

```
mns_cli resolve-role <role>
```

**Example:**

```bash
mns_cli resolve-role chatbot
```

This is the endpoint the `chatbot_api_server` calls at startup when `MODEL_ROLE` is configured.

**Errors:**

| Situation | HTTP status |
|-----------|-------------|
| No production model for this role | 404 |

---

### 4.11 delete

Hard-delete a model record from the database. Only permitted when the model is in `initializing` or `retired` state. Does **not** delete weight files from disk.

```
mns_cli delete <name>
```

**Example:**

```bash
mns_cli delete adai-chatbot-old
```

**Errors:**

| Situation | HTTP status | Meaning |
|-----------|-------------|---------|
| Model is training, candidate, or production | 409 | Retire it first |
| Model not found | 404 | Already deleted or never registered |

---

## 5. Model Lifecycle

Every model follows a state machine. `mns_cli` commands drive transitions; invalid transitions are rejected with HTTP 409.

```
                  +--------------+
                  | initializing |
                  +------+-------+
                         |
              set-training / set-candidate
                    |              |
                    v              v
              +-----------+   +-----------+
         +--->| training  |-->| candidate |<---+
         |    +-----------+   +-----+-----+    |
         |     set-candidate        |          |
         |                     promote         |
         |                          v          |
         |                   +------------+    |
         |                   | production |    |
         |                   +-----+------+    |
         |                         |           |
         |          (auto-retired on promote)  |
         |                         v           |
         |                   +---------+       |
         |                   | retired |-------+
         |                   +---------+  set-candidate
         |                        |        (revival)
         +--- set-training -------+
              (re-train from candidate)
```

**Transition summary:**

| From | To | Command | Notes |
|------|----|---------|-------|
| initializing | training | `set-training` | First training run |
| initializing | candidate | `set-candidate` | Direct import (skip training) |
| training | candidate | `set-candidate` | Normal training completion |
| training | training | `set-training` | Idempotent if same run_id |
| candidate | training | `set-training` | Re-train cycle |
| candidate | production | `promote` | Via role promotion |
| production | retired | (automatic) | When another model is promoted to the same role |
| retired | candidate | `set-candidate` | Revival |
| any except retired | retired | `set-candidate` with state=retired | Manual retirement (via HTTP API) |
| initializing, retired | (deleted) | `delete` | Hard removal from database |

---

## 6. Workflows

### 6.1 Register, train, and promote a new model

This is the standard end-to-end lifecycle for a new model.

```bash
# 1. Register the model identity
mns_cli register adai-chatbot-v3 chatbot

# 2. Start a training run (acquires training lock)
mns_cli set-training adai-chatbot-v3 run-42

# 3. (Training happens via incremental_trainer...)

# 4. Mark training complete and attach the artifact
mns_cli set-candidate adai-chatbot-v3 run-42 \
  --artifact-path /opt/adai/training_sessions/session_3_best.bin \
  --summary epochs=15 \
  --summary final_loss=0.38

# 5. Verify the candidate
mns_cli resolve adai-chatbot-v3

# 6. Promote to production (auto-retires previous production model)
mns_cli promote chatbot adai-chatbot-v3

# 7. Confirm
mns_cli resolve-role chatbot
```

When `incremental_trainer` has `NAME_SERVICE_URL` and `MODEL_NAME` configured, steps 2 and 4 happen automatically. You only need to run steps 1, 6, and 7 manually.

---

### 6.2 Re-train an existing model

A model in `candidate` state can be sent back to `training` for another round:

```bash
mns_cli set-training adai-chatbot-v3 run-43
# ... training ...
mns_cli set-candidate adai-chatbot-v3 run-43 \
  --artifact-path /opt/adai/training_sessions/session_4_best.bin \
  --summary epochs=10 \
  --summary final_loss=0.32
```

Each training run is appended to the model's `training_history` array, so you have a complete audit trail.

---

### 6.3 Roll back production to a previous model

If the newly promoted model has issues, register the old weights as a new candidate and promote:

```bash
# The old model was auto-retired during promotion.
# Re-register it as a candidate to make it promotable again.
mns_cli set-candidate adai-chatbot-v2 rollback-1 \
  --artifact-path /opt/adai/checkpoints/chatbot-v2-best.bin

# Promote it back (auto-retires v3)
mns_cli promote chatbot adai-chatbot-v2
```

---

### 6.4 Retire and clean up old models

```bash
# List retired models
mns_cli list --state retired

# Delete retired models you no longer need
mns_cli delete adai-chatbot-v1

# Weight files are NOT deleted — remove them manually if desired
```

---

### 6.5 Audit what is registered

```bash
# Full inventory
mns_cli list | jq '.models[] | {name: .model_name, state: .state, role: .role}'

# Which model is in production for each role?
mns_cli roles | jq .

# Architecture of a specific model
mns_cli get adai-chatbot-v3 | jq '{d_model, num_heads, d_ff}'

# Training history
mns_cli get adai-chatbot-v3 | jq '.training_history'
```

---

## 7. Configuration

`mns_cli` reads from `config.conf` for two purposes:

1. **Server URL** — the `NAME_SERVICE_URL` key determines which MNS server to contact (overridden by `--url`).
2. **Architecture defaults** — when `register` is called without explicit `--d-model`, `--num-heads`, etc., the values from config.conf are used so registration stays consistent with what the trainer will use.

Relevant config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `NAME_SERVICE_URL` | (empty) | MNS server URL; `http://localhost:8083` when using the server bundle |
| `NAME_SERVICE_TIMEOUT_MS` | `5000` | HTTP timeout for MNS calls |
| `D_MODEL` | `512` | Model dimension (used as default for `register`) |
| `NUM_HEADS` | `8` | Attention heads (used as default for `register`) |
| `D_FF` | `2048` | Feed-forward dimension (used as default for `register`) |
| `NUM_ENCODER_LAYERS` | `6` | Encoder layers (used as default for `register`) |
| `NUM_DECODER_LAYERS` | `6` | Decoder layers (used as default for `register`) |
| `MAX_SEQ_LENGTH` | `1024` | Maximum sequence length (used as default for `register`) |

Config file discovery order: `--config <path>` > `./config.conf` > `/etc/adai/config.conf`.

---

## 8. Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (HTTP 2xx response) |
| 1 | Error (HTTP 4xx/5xx, connection failure, or invalid arguments) |

All output goes to stdout (JSON response body) except error messages, which go to stderr.

---

## 9. Troubleshooting

### Connection refused

```
Error: connection to localhost:8083 failed
```

The MNS server is not running or is on a different port.

- Check: `systemctl status adai-mns`
- Check: `ss -tlnp | grep 8083`
- Override: `mns_cli --url http://correct-host:8083 health`

### Model not found (404)

```json
{"error":"model not found"}
```

The model name is case-sensitive. Run `mns_cli list` to see all registered names.

### Invalid state transition (409)

```json
{"error":"invalid transition: production -> training"}
```

The state machine does not allow the requested transition. Refer to the [lifecycle diagram](#5-model-lifecycle) for valid paths. Common fixes:

| Want to do | Current state | Fix |
|-----------|--------------|-----|
| Re-train a production model | `production` | Promote a different model first (or register a new name) |
| Delete a candidate | `candidate` | Retire it first: set state to `retired` via the HTTP API, then `delete` |
| Promote a training model | `training` | Complete training first with `set-candidate` |

### Training lock conflict (409)

```json
{"error":"model is locked for training by a different run_id"}
```

Another process is already training this model. Only one `run_id` can hold the training lock. If the previous run crashed:

```bash
# Complete the stale run (moves to candidate, releases lock)
mns_cli set-candidate <name> <stale-run-id> \
  --artifact-path /dev/null

# Then start your new run
mns_cli set-training <name> <new-run-id>
```

### Wrong architecture registered

Architecture is recorded at registration time and cannot be updated. If you registered with wrong values, delete and re-register:

```bash
mns_cli delete bad-model-name
mns_cli register bad-model-name chatbot --d-model 256 --num-heads 8
```

This only works if the model is in `initializing` state (never trained). If it has progressed past that, you need to retire it first.

### Stale config.conf values

`mns_cli` reads `config.conf` once at startup. If you update the file, the next invocation picks up the changes automatically — there is no daemon to restart.
