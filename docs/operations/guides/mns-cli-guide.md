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

Config file discovery: `--config <path>` > `./config.mns.conf` > `/etc/adai/config.mns.conf` > `./config.conf` (legacy) > `/etc/adai/config.conf` (legacy).

`http://` URLs default to port 8083 and `https://` URLs to port 443; a path in the URL is kept as a prefix for every
request. For an `https://` server, such as MNS behind a Cloudflare tunnel, `mns_cli` checks the certificate against
the system store, or `NAME_SERVICE_CA_FILE` / `--ca-file`, and sends extra headers from `NAME_SERVICE_HEADERS` and any
`--header 'Name: value'` flags (for example a Cloudflare Access service token; keep secrets in the config file rather
than on the command line, where they show in the process list):

```bash
mns_cli --url https://mns.example.com --header 'CF-Access-Client-Id: <id>' \
  --header 'CF-Access-Client-Secret: <secret>' health
```

Global options (`--url`, `--config`, `--header`, `--ca-file`, `--help`) go **before** the command; after the command
every argument belongs to it, so `mns_cli get m --url X` is an error. `mns_cli <command> --help` prints that command's
usage and sends nothing. (Before November 1, 2026, `https://` was sent as plain HTTP to port 8083, and `--help` only
worked as the first word; TD-451.)

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
mns_cli register <name> <role> [architecture options] [--from-config <trainer config>] [--tag key=value ...]
```

| Option | Description | When not given |
|--------|-------------|---------|
| `--d-model N` | Model dimension | `--from-config` file's `D_MODEL` |
| `--num-heads N` | Attention heads | `--from-config` file's `NUM_HEADS` |
| `--d-ff N` | Feed-forward dimension | `--from-config` file's `D_FF` |
| `--encoder-layers N` | Encoder layer count | `--from-config` file's `NUM_ENCODER_LAYERS` |
| `--decoder-layers N` | Decoder layer count | `--from-config` file's `NUM_DECODER_LAYERS` |
| `--max-seq-length N` | Maximum sequence length | `--from-config` file's `MAX_SEQ_LENGTH` |
| `--from-config FILE` | The trainer's config, for the fields not given as flags | none |
| `--tag key=value` | Arbitrary tag (repeatable) | none |

The architecture can't be changed after registering, and the trainer adopts it from MNS in place of its own config, so
`register` needs every field: as flags, or from `--from-config` pointing at the trainer's config. A missing field is an
error that lists what's missing; nothing is sent. (Before November 1, 2026, missing fields came silently from
`mns_cli`'s own config, `config.mns.conf`, whose 512/8 differed from the trainer's 768/12; TD-450.) `mns_cli` prints
the architecture it's about to register, and where it came from, on stderr:

```bash
# Same architecture the trainer's config describes
mns_cli register adai-chatbot-v3 chatbot --from-config /etc/adai/config.trainer.conf

# The trainer's config, with one field changed
mns_cli register adai-chatbot-v3 chatbot --from-config /etc/adai/config.trainer.conf --max-seq-length 256

# Everything on the command line
mns_cli register adai-chatbot-v3 chatbot --d-model 256 --num-heads 8 --d-ff 1024 \
  --encoder-layers 4 --decoder-layers 4 --max-seq-length 512 \
  --tag dataset=minipile-v2 --tag owner=rodney
```

For `--kind encoder`/`decoder`/`world_model`, the layer count is `--num-layers` (or the file's `NUM_ENCODER_LAYERS`,
`NUM_DECODER_LAYERS` for a decoder), and the other layer count is 0. A chatbot linked with `--encoder`/`--decoder`
takes no architecture options.
`--sigreg-lambda`/`--sigreg-num-sketches` are only accepted with `--kind world_model`.

To attach a world model to a chatbot, or change its link, use `link-world-model`. Settings you don't pass keep their
current values, so a relink can change one of them:

```bash
mns_cli link-world-model adai-chatbot-v3 --world-model adai-wm-v1 --inject-every-n-layers 1 \
  --hippocampal-enabled --hippocampal-capacity 256
# Later: change only the injection interval; hippocampal memory stays on at 256
mns_cli link-world-model adai-chatbot-v3 --world-model adai-wm-v1 --inject-every-n-layers 2
# Turn hippocampal memory off, or detach the world model
mns_cli link-world-model adai-chatbot-v3 --world-model adai-wm-v1 --hippocampal-disabled
mns_cli link-world-model adai-chatbot-v3 --world-model ""
```

Model names and `list` filter values are URL-encoded, so any characters work.

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

Transition a model from `initializing` or `candidate` to the `training` state. MNS allocates the `run_id` and prints it in
the response (`run-01` the first time a model trains); pass it to `set-candidate` later.

```
mns_cli set-training <name> [--new-run] [session-key]
```

| Argument | Description |
|----------|-------------|
| `<name>` | Model name |
| `--new-run` | Start the next run (`run-02`, …), as `incremental_trainer retrain` does; without it the current run continues |
| `[session-key]` | Optional metrics session key for cross-referencing with `metrics_api_server` |

Anything else is a usage error (TD-448); the run ID is no longer an argument.

**Example:**

```bash
mns_cli set-training adai-chatbot-v3 metrics-key-abc
```

**Errors:**

| Situation | HTTP status | Meaning |
|-----------|-------------|---------|
| Different run_id holds the lock | 409 | Another training run already owns this model |
| Invalid source state | 409 | Model is in `production` or `retired` (must retire first or start from candidate) |

The `incremental_trainer` calls this automatically when `NAME_SERVICE_URL` and `MODEL_NAME` are configured.

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
| `--revive` | Bring a `retired` model back to `candidate` |

**Example:**

```bash
mns_cli set-candidate adai-chatbot-v3 run-01 \
  --artifact-path /opt/adai/checkpoints/session_3_best.bin \
  --summary epochs=15 \
  --summary final_loss=0.38 \
  --summary dataset_group=training_sessions
```

The `incremental_trainer` calls this automatically after a successful training session.

**Valid source states:** `training`, `initializing` (for direct imports), `retired` (only with `--revive`, so a trainer finishing its pass can't undo a retirement).

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
# 1. Register the model identity, with the trainer's architecture
mns_cli register adai-chatbot-v3 chatbot --from-config /etc/adai/config.trainer.conf

# 2. Start a training run; the response carries the allocated run_id (run-01)
mns_cli set-training adai-chatbot-v3

# 3. (Training happens via incremental_trainer...)

# 4. Mark training complete and attach the artifact
mns_cli set-candidate adai-chatbot-v3 run-01 \
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
mns_cli set-training adai-chatbot-v3 --new-run    # allocates run-02
# ... training ...
mns_cli set-candidate adai-chatbot-v3 run-02 \
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

`mns_cli` reads its own config file for these keys:

| Key | Default | Description |
|-----|---------|-------------|
| `NAME_SERVICE_URL` | (empty) | MNS server URL (overridden by `--url`); `http://localhost:8083` when empty |
| `NAME_SERVICE_HEADERS` | (empty) | Extra request headers, `Name: value` pairs separated by `;` (added to by `--header`) |
| `NAME_SERVICE_CA_FILE` | (empty) | CA bundle for an `https://` server (overridden by `--ca-file`); empty = the system store |
| `RUN_GROUP` | (empty) | Default for `register --run-group` |

It doesn't take `register`'s architecture from this file (TD-450); use `register --from-config` with the trainer's
config instead.

Config file discovery order: `--config <path>` > `./config.mns.conf` > `/etc/adai/config.mns.conf` > `./config.conf` (legacy) > `/etc/adai/config.conf` (legacy).

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
# Start a new run; MNS archives the unfinished one in training_history as "incomplete"
mns_cli set-training <name> --new-run
```

### Wrong architecture registered

Architecture is recorded at registration time and cannot be updated. If you registered with wrong values, delete and re-register:

```bash
mns_cli delete bad-model-name
mns_cli register bad-model-name chatbot --from-config /etc/adai/config.trainer.conf
```

This only works if the model is in `initializing` state (never trained). If it has progressed past that, you need to retire it first.

### Stale config values

`mns_cli` reads its config file (and `register --from-config`'s file) once per invocation. If you update the file, the next invocation picks up the changes automatically — there is no daemon to restart.
