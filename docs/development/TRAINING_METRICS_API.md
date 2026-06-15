# Training Metrics REST API

The `metrics_api_server` is a standalone HTTP daemon that receives real-time training metrics pushed by `incremental_trainer` and exposes them for dashboards, monitoring, and external integrations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Full Endpoint Reference](#full-endpoint-reference)
4. [Consumer Endpoints](#consumer-endpoints)
5. [Push Endpoints](#push-endpoints)
6. [Command-Line Options](#command-line-options)
7. [Integration Examples](#integration-examples)
8. [Prometheus Integration](#prometheus-integration)
9. [Security Notes](#security-notes)
10. [See Also](#see-also)

---

## Quick Start

### 1. Build

```bash
cd /home/rodney/Repos/adai
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

### 2. Start the Server

```bash
# Default settings (port 8081, persists to training_sessions/)
./build/bin/metrics_api_server

# Custom port and persistence frequency
./build/bin/metrics_api_server --port 9090 --persist-samples 50 --persist-seconds 15

# Disable persistence (in-memory only)
./build/bin/metrics_api_server --no-persistence
```

### 3. Enable Metric Pushing from the Trainer

In `config.conf`:

```ini
ENABLE_METRICS_SERVICE=true
METRICS_SERVER_URL=http://localhost:8081
```

Start the server **before** training. `incremental_trainer` connects at startup and pushes metrics throughout the run.

### 4. Query Metrics

```bash
# Health check
curl http://localhost:8081/health

# Current snapshot (0-default session)
curl http://localhost:8081/api/sessions/0-default/metrics/current | jq

# Session list
curl http://localhost:8081/api/sessions | jq
```

---

## Architecture

The server uses a session-keyed design. Each training run opens a named **session** before it starts and closes it when finished. Monitoring clients poll the session for live data.

**Session key format:** 1–64 characters; must start with an alphanumeric character; may contain alphanumerics, `_`, and `-`. Example keys: `0-default`, `42-finetune-gpu0`, `run_2026-06-15`.

**Default session (`0-default`):** Created automatically at startup. `incremental_trainer` pushes to this session by default when `METRICS_SERVER_URL` is set.

**Legacy paths:** All pre-multi-session endpoints (`/api/metrics/current`, `/api/session/status`, etc.) are backwards-compatible aliases for the `0-default` session. They work identically but include a `Deprecation: true` response header and a `Link:` header pointing to the canonical session-scoped URL. New integrations should use the session-scoped paths.

---

## Full Endpoint Reference

### Consumer endpoints (monitoring dashboards)

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/api/sessions` | List all sessions with status |
| **GET** | `/api/sessions/{key}/metrics/current` | Real-time snapshot |
| **GET** | `/api/sessions/{key}/metrics/summary` | Aggregated session summary |
| **GET** | `/api/sessions/{key}/metrics/history` | Historical records (`max_records`, `session_id`) |
| **GET** | `/api/sessions/{key}/metrics/abnormal` | Anomalous samples flagged by outlier detection |
| **GET** | `/api/sessions/{key}/metrics/generation-quality` | BLEU/ROUGE scores per epoch |
| **GET** | `/api/sessions/{key}/metrics/padding-efficiency` | Padding efficiency per epoch |
| **GET** | `/api/sessions/{key}/metrics/prometheus` | Prometheus text format |
| **GET** | `/api/sessions/{key}/metrics/csv` | CSV format |
| **GET** | `/api/sessions/{key}/status` | Session status and ETA |
| **GET** | `/api/sessions/{key}/epochs` | Per-epoch history |
| **GET** | `/api/metrics/aggregate` | Compact JSON of all live sessions |
| **GET** | `/api/metrics/prometheus/aggregate` | Prometheus text for all live sessions with `session=` labels |
| **GET** | `/health` | Server health and active session count |

### Push endpoints (trainer → server)

| Method | Endpoint | Description |
|--------|----------|-------------|
| **POST** | `/api/sessions/{key}/start` | Create / start a training session |
| **POST** | `/api/sessions/{key}/end` | End and finalize a session |
| **POST** | `/api/sessions/{key}/epoch/start` | Signal epoch start |
| **POST** | `/api/sessions/{key}/epoch/end` | Report epoch results |
| **POST** | `/api/sessions/{key}/metrics/sample` | Push per-sample loss/gradient metrics |
| **POST** | `/api/sessions/{key}/metrics/validation` | Push validation metrics |
| **POST** | `/api/sessions/{key}/metrics/best` | Record best epoch checkpoint |
| **POST** | `/api/sessions/{key}/metrics/advanced` | Push advanced diagnostic metrics |
| **POST** | `/api/sessions/{key}/metrics/generation-quality` | Push BLEU/ROUGE scores |
| **POST** | `/api/sessions/{key}/control/flush` | Force flush to disk |
| **POST** | `/api/sessions/{key}/control/clear` | Clear in-memory history |

### Legacy aliases (deprecated, 0-default session only)

These paths work identically to their session-scoped equivalents but include `Deprecation: true` in the response header.

| Legacy path | Canonical path |
|---|---|
| `GET /api/metrics/current` | `GET /api/sessions/0-default/metrics/current` |
| `GET /api/metrics/summary` | `GET /api/sessions/0-default/metrics/summary` |
| `GET /api/metrics/history` | `GET /api/sessions/0-default/metrics/history` |
| `GET /api/metrics/abnormal` | `GET /api/sessions/0-default/metrics/abnormal` |
| `GET /api/metrics/generation-quality` | `GET /api/sessions/0-default/metrics/generation-quality` |
| `GET /api/metrics/padding-efficiency` | `GET /api/sessions/0-default/metrics/padding-efficiency` |
| `GET /api/metrics/prometheus` | `GET /api/sessions/0-default/metrics/prometheus` |
| `GET /api/metrics/csv` | `GET /api/sessions/0-default/metrics/csv` |
| `GET /api/session/status` | `GET /api/sessions/0-default/status` |
| `GET /api/session/epochs` | `GET /api/sessions/0-default/epochs` |
| `POST /api/session/start` | `POST /api/sessions/0-default/start` |
| `POST /api/session/end` | `POST /api/sessions/0-default/end` |
| `POST /api/epoch/start` | `POST /api/sessions/0-default/epoch/start` |
| `POST /api/epoch/end` | `POST /api/sessions/0-default/epoch/end` |
| `POST /api/metrics/sample` | `POST /api/sessions/0-default/metrics/sample` |
| `POST /api/metrics/validation` | `POST /api/sessions/0-default/metrics/validation` |
| `POST /api/metrics/best` | `POST /api/sessions/0-default/metrics/best` |
| `POST /api/metrics/advanced` | `POST /api/sessions/0-default/metrics/advanced` |
| `POST /api/metrics/generation-quality` | `POST /api/sessions/0-default/metrics/generation-quality` |
| `POST /api/control/flush` | `POST /api/sessions/0-default/control/flush` |
| `POST /api/control/clear` | `POST /api/sessions/0-default/control/clear` |

---

## Consumer Endpoints

### `GET /api/sessions`

Lists all registered sessions.

Response:

```json
{
  "sessions": [
    {
      "key": "42-finetune-gpu0",
      "session_id": 42,
      "is_training": true,
      "current_epoch": 3,
      "total_epochs": 10,
      "current_loss": 2.1234,
      "best_validation_loss": 2.0501,
      "session_start_time": 1748720400,
      "last_update_time": 1748721234,
      "metrics_url": "/api/sessions/42-finetune-gpu0/metrics/current"
    }
  ],
  "total": 1,
  "live": 1
}
```

---

### `GET /api/sessions/{key}/metrics/current`

Real-time training snapshot.

Response:

```json
{
  "session_id": 1,
  "is_training": true,
  "current_epoch": 2,
  "total_epochs": 10,
  "current_sample": 450,
  "total_samples": 1000,
  "current_loss": 2.3456,
  "current_validation_loss": 2.4123,
  "current_learning_rate": 0.001,
  "current_gradient_norm": 1.234,
  "current_perplexity": 10.43,
  "running_loss": 2.4001,
  "running_validation_loss": 2.5012,
  "samples_per_second": 12.5,
  "estimated_time_remaining_seconds": 120.5,
  "best_validation_loss": 2.3001,
  "best_epoch": 1,
  "total_samples_trained": 2450,
  "total_training_time_seconds": 196.0,
  "gradient_variance": 0.0123,
  "compute_time_ratio": 0.8234,
  "weight_update_ratio": 0.000456,
  "activation_saturation_ratio": 0.1234,
  "current_validation_perplexity": 11.23,
  "current_validation_accuracy": -1.0,
  "current_bleu4": -1.0,
  "current_rouge1": -1.0,
  "current_rouge2": -1.0,
  "current_rougeL": -1.0
}
```

**Field notes:**

- `current_validation_perplexity` — `0.0` if not yet computed this epoch.
- `current_validation_accuracy` — token-level accuracy. `-1.0` if the trainer did not push `validation_accuracy`.
- `current_bleu4`, `current_rouge1`, `current_rouge2`, `current_rougeL` — `-1.0` when `ENABLE_GENERATION_QUALITY_METRICS=false` in `config.conf` or before the first scored epoch.

---

### `GET /api/sessions/{key}/metrics/summary`

Aggregated session summary.

Response:

```json
{
  "session_id": 1,
  "total_epochs_completed": 2,
  "total_samples_trained": 2000,
  "total_training_time_seconds": 160.5,
  "average_loss": 2.5,
  "best_validation_loss": 2.3,
  "best_epoch": 1,
  "average_samples_per_second": 12.5
}
```

---

### `GET /api/sessions/{key}/metrics/history`

Historical per-sample records.

Query parameters:

- `max_records` (default: 1000) — Maximum records to return.
- `session_id` — Filter by numeric session ID.

```bash
curl "http://localhost:8081/api/sessions/0-default/metrics/history"
curl "http://localhost:8081/api/sessions/0-default/metrics/history?max_records=100"
curl "http://localhost:8081/api/sessions/0-default/metrics/history?session_id=1"
```

Response:

```json
{
  "records": [
    {
      "timestamp": "1678123456",
      "session_id": 1,
      "epoch": 0,
      "sample": 100,
      "loss": 3.2,
      "validation_loss": 3.3,
      "learning_rate": 0.001,
      "gradient_norm": 1.5,
      "perplexity": 24.5
    }
  ],
  "count": 1
}
```

---

### `GET /api/sessions/{key}/status`

Session status and progress estimate.

Response:

```json
{
  "is_training": true,
  "session_id": 1,
  "current_epoch": 2,
  "total_epochs": 10,
  "current_sample": 450,
  "total_samples": 1000,
  "progress_percent": 45.00,
  "samples_per_second": 12.5,
  "estimated_time_remaining_seconds": 120.5,
  "is_stale": false,
  "seconds_since_last_update": 1.2,
  "effective_is_training": true
}
```

**Field notes:**

- `is_stale` — `true` when the session has not received a metrics push within the staleness threshold. A stale session may have completed or crashed.
- `seconds_since_last_update` — wall-clock seconds since the last push from the trainer.
- `effective_is_training` — `is_training && !is_stale`; use this field to determine whether the session is actually live.

---

### `GET /api/sessions/{key}/epochs`

Per-epoch history for the session.

Response:

```json
{
  "current_epoch": 2,
  "total_epochs": 10,
  "epoch_losses": [3.5, 2.8, 2.4],
  "epoch_validation_losses": [3.6, 2.9, 2.5],
  "epoch_learning_rates": [0.001, 0.001, 0.0009],
  "epoch_perplexities": [33.1, 16.4, 11.0],
  "epoch_durations": [45.2, 44.8, 45.1],
  "epoch_gradient_norms": [2.1, 1.8, 1.5],
  "epoch_adaptive_clip_thresholds": [-1.0, -1.0, -1.0],
  "epoch_validation_perplexities": [34.2, 17.1, 11.8],
  "epoch_validation_accuracies": [-1.0, -1.0, -1.0],
  "best_validation_loss": 2.5,
  "best_epoch": 2
}
```

**Field notes:**

- `epoch_adaptive_clip_thresholds` — average adaptive gradient clip threshold per epoch. `-1.0` entries mean fixed-clip mode was active for that epoch (no adaptive clipping). See `GRADIENT_CLIP_ADAPTIVE` in `config.conf`.
- `epoch_validation_perplexities` — validation perplexity recorded at end of each epoch.
- `epoch_validation_accuracies` — token-level validation accuracy per epoch; `-1.0` when not computed.

See also [`GET /api/sessions/{key}/metrics/generation-quality`](#get-apisessionskeymetricsgeneration-quality) for per-epoch BLEU/ROUGE arrays.

---

### `GET /api/sessions/{key}/metrics/abnormal`

Samples flagged as anomalous by the outlier-detection subsystem (e.g., loss spikes, extreme gradient norms).

Response:

```json
{
  "abnormal_samples": [
    {
      "epoch": 3,
      "sample_id": 214,
      "loss": 9.871200,
      "grad_norm": 15.320000,
      "reason": "loss_spike",
      "input_text": "Hello",
      "target_text": "Hi there",
      "timestamp": 1748721234
    }
  ],
  "count": 1
}
```

Returns an empty `abnormal_samples` array when no anomalies have been detected.

---

### `GET /api/sessions/{key}/metrics/generation-quality`

BLEU-4 and ROUGE scores for the session. All values are `-1.0` by default; opt in via `config.conf`.

**Enable in `config.conf`:**

```ini
ENABLE_GENERATION_QUALITY_METRICS=true
GENERATION_QUALITY_SAMPLE_SIZE=10    # validation pairs scored per epoch
GENERATION_QUALITY_MAX_TOKENS=50     # max tokens per generation call
```

Enabling this adds overhead per epoch proportional to `GENERATION_QUALITY_SAMPLE_SIZE`. Keep `SAMPLE_SIZE` ≤ 20 for routine training; use larger values for final evaluation passes.

Response:

```json
{
  "current_bleu4": 0.312500,
  "current_rouge1": 0.487654,
  "current_rouge2": 0.281234,
  "current_rougeL": 0.421875,
  "epoch_bleu4":   [0.210000, 0.265000, 0.312500],
  "epoch_rouge1":  [0.380000, 0.440000, 0.487654],
  "epoch_rouge2":  [0.190000, 0.235000, 0.281234],
  "epoch_rougeL":  [0.330000, 0.380000, 0.421875]
}
```

| Metric | Range | Description |
|--------|-------|-------------|
| `current_bleu4` | 0–1 | Corpus BLEU-4 with brevity penalty (Lin & Och add-1 smoothing) |
| `current_rouge1` | 0–1 | Macro-averaged ROUGE-1 F1 (unigram overlap) |
| `current_rouge2` | 0–1 | Macro-averaged ROUGE-2 F1 (bigram overlap) |
| `current_rougeL` | 0–1 | Macro-averaged ROUGE-L F1 (rolling 2-row DP LCS) |

A `-1.0` entry in a per-epoch array means that epoch was not scored (e.g., the feature was enabled mid-training).

---

### `GET /api/sessions/{key}/metrics/padding-efficiency`

Token padding efficiency per epoch. Values near `1.0` mean minimal wasted padding tokens relative to actual sequence content.

Response:

```json
{
  "current_padding_efficiency": 0.873214,
  "epoch_padding_efficiencies": [0.841234, 0.856781, 0.873214]
}
```

---

### `GET /api/sessions/{key}/metrics/prometheus`

Metrics in Prometheus text format for scraping.

Response (`Content-Type: text/plain`):

```text
# HELP training_loss Current training loss
# TYPE training_loss gauge
training_loss{session="0-default"} 2.3456
# HELP validation_loss Current validation loss
# TYPE validation_loss gauge
validation_loss{session="0-default"} 2.4123
...
```

---

### `GET /api/sessions/{key}/metrics/csv`

Current metric snapshot as a CSV row (header + current values).

Response (`Content-Type: text/csv`):

```csv
session_id,epoch,sample,loss,validation_loss,learning_rate,gradient_norm,perplexity
1,2,450,2.3456,2.4123,0.001,1.234,10.43
```

---

### `GET /api/metrics/aggregate`

Compact JSON summary of all currently **active** (training, not stale) sessions. Idle and completed sessions are omitted.

Response:

```json
{
  "live_sessions": 2,
  "sessions": [
    {"key": "42-finetune-gpu0", "epoch": 3, "loss": 2.1234, "validation_loss": 2.2010},
    {"key": "43-finetune-gpu1", "epoch": 3, "loss": 2.0981, "validation_loss": 2.1750}
  ]
}
```

---

### `GET /api/metrics/prometheus/aggregate`

Prometheus exposition for **all live sessions**, each labelled with `session=`. Use this endpoint for multi-run Prometheus scraping.

Response (`Content-Type: text/plain`):

```text
# HELP training_loss Current training loss
# TYPE training_loss gauge
training_loss{session="42-finetune-gpu0"} 2.123400 1748721234000
training_loss{session="43-finetune-gpu1"} 2.098100 1748721235000
...
```

---

### `GET /health`

Server health status.

Response:

```json
{
  "status": "ok",
  "service": "TrainingMetricsAPI",
  "is_training": true,
  "any_stale": false
}
```

**Field notes:**

- `is_training` — `true` if at least one session is active and has received a push within the last 60 seconds.
- `any_stale` — `true` if any session is in `is_training` state but has not received an update in the last 60 seconds, indicating the trainer may have hung or crashed.

---

### Control Endpoints

Enabled by default; disable with `--no-control`.

#### `POST /api/sessions/{key}/control/flush`

Forces immediate flush of all in-memory metrics to disk.

Response: `{"status":"success","message":"Metrics flushed to disk"}`

#### `POST /api/sessions/{key}/control/clear`

Clears the in-memory metrics history for the session. Does not affect persisted files.

Response: `{"status":"success","message":"Metrics history cleared"}`

---

## Push Endpoints

These endpoints are called by `incremental_trainer` (via `MetricsPushClient`) to push training state to the server. They are also useful for custom trainers or test harnesses.

### Session lifecycle

#### `POST /api/sessions/{key}/start`

Creates the session if it does not exist, then starts a training run for it. Returns `409 Conflict` if the session is already active and not stale. Returns `503 Service Unavailable` with `{"error":"metrics_server_full","max_live_sessions":N}` if `--max-live-sessions` is reached.

Request body:

```json
{
  "session_id":    42,
  "total_epochs":  10,
  "total_samples": 5000,
  "label":  "#42: wiki-finetune (gpu0, 2026-06-15)",
  "config": {"LEARNING_RATE": 0.0003, "BATCH_SIZE": 32}
}
```

| Field | Required | Description |
|---|---|---|
| `session_id` | yes | Numeric identifier |
| `total_epochs` | no | Used for progress calculation |
| `total_samples` | no | Used for ETA estimation |
| `label` | no | Human-readable label shown in `GET /api/sessions` |
| `config` | no | Training config snapshot stored with the session for audit |

Response: `{"status":"ok","message":"Session started"}`

#### `POST /api/sessions/{key}/end`

Marks the session as complete and writes its final summary to disk.

Response: `{"status":"ok","message":"Session ended"}`

---

### Epoch lifecycle

#### `POST /api/sessions/{key}/epoch/start`

Signals that an epoch has begun.

Request body:

```json
{"epoch": 2, "total_samples": 1000}
```

Response: `{"status":"ok","message":"Epoch started"}`

#### `POST /api/sessions/{key}/epoch/end`

Reports epoch results. Core fields are required; advanced fields are optional.

Request body:

```json
{
  "epoch":              2,
  "loss":               2.3456,
  "validation_loss":    2.4123,
  "learning_rate":      0.001,
  "perplexity":         10.43,
  "gradient_norm":      1.234,
  "epoch_time":         44.8,
  "gradient_variance":  0.0123,
  "compute_time_ratio": 0.823,
  "weight_update_ratio": 0.000456,
  "activation_saturation_ratio": 0.1234,
  "attention_entropy": 2.31,
  "current_padding_efficiency": 0.873
}
```

Required: `epoch`, `loss`, `validation_loss`, `learning_rate`, `perplexity`, `gradient_norm`.
Optional: all remaining fields. Omit or set to `-1.0` if not computed.

Response: `{"status":"ok","message":"Epoch ended"}`

---

### Per-sample and validation pushes

#### `POST /api/sessions/{key}/metrics/sample`

Push metrics for a single training sample.

```json
{"sample": 450, "loss": 2.3456, "gradient_norm": 1.234, "learning_rate": 0.001}
```

Response: `{"status":"ok"}`

#### `POST /api/sessions/{key}/metrics/validation`

Push validation metrics mid-epoch or at epoch end.

```json
{
  "validation_loss":       2.4123,
  "validation_accuracy":   -1.0,
  "validation_perplexity": 11.23
}
```

`validation_accuracy` — optional; pass `-1.0` if not computed.
`validation_perplexity` — optional; the server derives `exp(validation_loss)` when `0.0`.

Response: `{"status":"ok"}`

#### `POST /api/sessions/{key}/metrics/best`

Records the best checkpoint seen so far.

```json
{"validation_loss": 2.3001, "epoch": 1}
```

Response: `{"status":"ok"}`

#### `POST /api/sessions/{key}/metrics/generation-quality`

Push BLEU/ROUGE scores for the current epoch.

```json
{"bleu4": 0.3125, "rouge1": 0.4876, "rouge2": 0.2812, "rougeL": 0.4218}
```

Response: `{"status":"ok"}`

#### `POST /api/sessions/{key}/metrics/advanced`

Push advanced diagnostic metrics independently of epoch end (for trainers that compute these asynchronously).

```json
{
  "gradient_variance":   0.0123,
  "compute_time_ratio":  0.823,
  "weight_update_ratio": 0.000456
}
```

Response: `{"status":"ok"}`

---

## Command-Line Options

```text
Training Metrics REST API Server
Usage: metrics_api_server [OPTIONS]

Options:
  --port PORT                  Port number (default: 8081)
  --metrics-file FILE          Metrics JSONL file path
                               (default: training_sessions/metrics.jsonl)
  --summary-file FILE          Summary JSON file path
                               (default: training_sessions/metrics_summary.json)
  --prometheus-file FILE       Prometheus file path
                               (default: training_sessions/metrics.prom)
  --persist-samples N          Persist every N samples (default: 100)
  --persist-seconds N          Persist every N seconds (default: 30)
  --max-memory-records N       Max records in memory (default: 10000)
  --max-disk-records N         Max records on disk (default: 100000)
  --max-live-sessions N        Max concurrent live sessions (default: 16)
  --completed-ttl-seconds N    Seconds to retain completed sessions in memory
                               (default: 3600)
  --sweep-interval-seconds N   Seconds between background eviction sweeps
                               (default: 60; 0 disables)
  --no-persistence             Disable persistence to disk
  --enable-prometheus          Enable Prometheus format output
  --no-control                 Disable control endpoints (flush, clear)
  --help                       Show this help message
```

**Capacity limits:** When `--max-live-sessions` is reached, `POST /api/sessions/{key}/start` returns `503 Service Unavailable` with body `{"error":"metrics_server_full","max_live_sessions":N}`. Completed sessions are evicted after `--completed-ttl-seconds`; the sweep runs every `--sweep-interval-seconds`.

---

## Integration Examples

### Python monitoring client

```python
import requests
import time

api_url = "http://localhost:8081"
session = "0-default"

while True:
    status = requests.get(f"{api_url}/api/sessions/{session}/status").json()

    if status.get("effective_is_training"):
        metrics = requests.get(f"{api_url}/api/sessions/{session}/metrics/current").json()
        print(f"Epoch {status['current_epoch']} ({status['progress_percent']:.1f}%) "
              f"| loss={metrics['current_loss']:.4f} val_loss={metrics['current_validation_loss']:.4f}")
    elif status.get("is_stale"):
        print("Warning: session is stale — training may have completed or crashed")

    time.sleep(2)
```

### JavaScript monitoring client

```javascript
const apiUrl = 'http://localhost:8081';
const session = '0-default';

async function pollMetrics() {
    const metrics = await fetch(`${apiUrl}/api/sessions/${session}/metrics/current`)
        .then(r => r.json());
    console.log(`Loss: ${metrics.current_loss} | Epoch: ${metrics.current_epoch}/${metrics.total_epochs}`);
}

setInterval(pollMetrics, 1000);
```

### Bash monitoring script

```bash
#!/bin/bash
API_URL="http://localhost:8081"
SESSION="0-default"

while true; do
    STATUS=$(curl -s "$API_URL/api/sessions/$SESSION/status")
    if [ "$(echo "$STATUS" | jq -r '.effective_is_training')" = "true" ]; then
        METRICS=$(curl -s "$API_URL/api/sessions/$SESSION/metrics/current")
        echo "Epoch $(echo "$STATUS" | jq -r '.current_epoch') | Loss: $(echo "$METRICS" | jq -r '.current_loss')"
    fi
    sleep 2
done
```

---

## Prometheus Integration

```bash
./build/bin/metrics_api_server --enable-prometheus
```

### Single session

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'training_metrics'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/api/sessions/0-default/metrics/prometheus'
    scrape_interval: 5s
```

### All live sessions

```yaml
scrape_configs:
  - job_name: 'training_metrics_all'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/api/metrics/prometheus/aggregate'
    scrape_interval: 5s
```

---

## Security Notes

- The server binds to `0.0.0.0` (all interfaces) by default.
- No authentication is built in. Add a reverse proxy (nginx, Caddy) for production access control.
- CORS headers (`Access-Control-Allow-Origin: *`) are included on all responses to support browser-based dashboards.
- Control endpoints (`flush`, `clear`) can be disabled with `--no-control` to make the server read-only for external clients.
- Session key format is validated on all push endpoints (1–64 chars, alphanumeric start, alphanumerics/`_`/`-` only). Invalid keys return `400 Bad Request`.

---

## See Also

- [TrainingMetricsService.hpp](../../src/TrainingMetricsService.hpp) — Core metrics service and snapshot structs
- [TrainingMetricsAPI.hpp](../../src/TrainingMetricsAPI.hpp) — REST API implementation
- [MetricsPushClient.hpp](../../src/MetricsPushClient.hpp) — Client used by `incremental_trainer` to push metrics
- [MetricsSessionRegistry.hpp](../../src/MetricsSessionRegistry.hpp) — Session lifecycle and eviction
- [OPERATIONS_MANUAL.md §5.3](../operations/OPERATIONS_MANUAL.md#53-training-metrics-service) — Operational overview and `config.conf` reference
