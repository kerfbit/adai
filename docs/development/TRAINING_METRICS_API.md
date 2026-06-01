# Training Metrics REST API Documentation

## Overview

The Training Metrics REST API provides real-time access to training progress and metrics through HTTP endpoints. It's designed for monitoring training jobs, building dashboards, and integrating with external monitoring systems.

## Full API Catalog Quick Reference

|Method|Endpoint|Description|
|--------|----------|-------------|
|**GET**|`/api/metrics/current`|Current training snapshot with all real-time metrics|
|**GET**|`/api/metrics/summary`|Aggregated metrics summary|
|**GET**|`/api/metrics/history`|Historical metrics records (supports `max_records`, `session_id`)|
|**GET**|`/api/metrics/abnormal`|Samples flagged as anomalous by outlier detection|
|**GET**|`/api/metrics/generation-quality`|Current and per-epoch BLEU/ROUGE generation quality scores (TD-016)|
|**GET**|`/api/metrics/prometheus`|Metrics in Prometheus text format|
|**GET**|`/api/metrics/csv`|Current metrics in CSV format|
|**GET**|`/api/session/status`|Current training session status and progress|
|**GET**|`/api/session/epochs`|Per-epoch metrics history|
|**POST**|`/api/control/flush`|Forces immediate flush of metrics to disk|
|**POST**|`/api/control/clear`|Clears historical metrics from memory|
|**GET**|`/health`|Returns server health status|
|**GET**|`/api/sessions`|List all registered sessions with status|
|**GET**|`/api/metrics/aggregate`|Live-session summary across all active sessions|
|**GET**|`/api/metrics/prometheus/aggregate`|Prometheus text for all live sessions, each labelled with `session=` (TD-021)|
|**POST**|`/api/sessions/{key}/start`|Create and start a named training session|
|**POST**|`/api/sessions/{key}/end`|End a named training session|

## Quick Start

### 1. Build the Metrics API Server

```bash
cd build
cmake .. -DBUILD_METRICS_API_SERVER=ON
make metrics_api_server
```

### 2. Start the Server

```bash
# With default settings (port 8081)
./src/metrics_api_server

# With custom configuration
./src/metrics_api_server \
  --port 9090 \
  --persist-samples 50 \
  --persist-seconds 15 \
  --enable-prometheus
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8081/health

# Get current metrics
curl http://localhost:8081/api/metrics/current | jq

# Get session status
curl http://localhost:8081/api/session/status | jq
```

## API Endpoints

### Metrics Endpoints

#### `GET /api/metrics/current`

Returns the current training snapshot with all real-time metrics.

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

- `current_validation_perplexity` — validation perplexity for the current epoch (TD-015). `0.0` if not yet computed.
- `current_validation_accuracy` — token-level validation accuracy (TD-015). `-1.0` if not computed (requires explicit `update_validation_metrics()` call with an accuracy value).
- `current_bleu4`, `current_rouge1`, `current_rouge2`, `current_rougeL` — generation quality scores (TD-016). All default to `-1.0` when `enable_generation_quality_metrics = false` or before the first scored epoch.

---

#### `GET /api/metrics/summary`

Returns aggregated metrics summary.

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

#### `GET /api/metrics/history`

Returns historical metrics records.

Query Parameters:

- `max_records` (optional, default: 1000) - Maximum number of records to return
- `session_id` (optional) - Filter by specific session ID

Examples:

```bash
# Get last 1000 records
curl "http://localhost:8081/api/metrics/history"

# Get last 100 records
curl "http://localhost:8081/api/metrics/history?max_records=100"

# Get records for session 1
curl "http://localhost:8081/api/metrics/history?session_id=1"
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

#### `GET /api/metrics/abnormal`

Returns samples that the outlier-detection subsystem has flagged as anomalous (e.g., loss spikes, extreme gradient norms).

Response:

```json
{
  "abnormal_samples": [
    {
      "epoch": 3,
      "sample": 214,
      "loss": 9.8712,
      "gradient_norm": 15.32,
      "reason": "loss_spike"
    }
  ],
  "count": 1
}
```

Returns an empty `abnormal_samples` array when outlier detection has not been configured or no anomalies have been detected.

---

#### `GET /api/metrics/generation-quality`

Returns current and per-epoch BLEU and ROUGE generation quality scores (TD-016). All values are `-1.0` when `enable_generation_quality_metrics = false` (the default) or before the first epoch has been scored.

Scoring is opt-in because it requires calling `model->generate_response()` on a sample of the validation set, which is significantly more expensive than loss-only validation. Enable it via `TrainingConfig`:

```cpp
config.enable_generation_quality_metrics = true;  // default: false
config.generation_quality_sample_size   = 10;     // validation pairs to sample per epoch
config.generation_quality_max_tokens    = 50;     // max tokens per generation call
```

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

**Score interpretation:**

| Metric | Range | Description |
|--------|-------|-------------|
| `current_bleu4` | 0–1 | Corpus BLEU-4 with clipped modified precision and brevity penalty (Lin & Och 2004 add-1 smoothing) |
| `current_rouge1` | 0–1 | Macro-averaged ROUGE-1 F1 (unigram overlap) |
| `current_rouge2` | 0–1 | Macro-averaged ROUGE-2 F1 (bigram overlap) |
| `current_rougeL` | 0–1 | Macro-averaged ROUGE-L F1 via rolling 2-row DP LCS |

Per-epoch arrays (`epoch_bleu4`, `epoch_rouge1`, `epoch_rouge2`, `epoch_rougeL`) are appended at the end of each epoch. An entry of `-1.0` in a per-epoch array indicates that generation quality was not computed for that epoch (e.g., when the feature was enabled mid-training).

---

#### `GET /api/metrics/prometheus`

Returns metrics in Prometheus text format for scraping.

Response:

```text
# HELP training_loss Current training loss
# TYPE training_loss gauge
training_loss 2.3456

# HELP validation_loss Current validation loss
# TYPE validation_loss gauge
validation_loss 2.4123

# HELP learning_rate Current learning rate
# TYPE learning_rate gauge
learning_rate 0.001
...
```

---

#### `GET /api/metrics/csv`

Returns current metrics in CSV format (header + current row).

Response:

```csv
session_id,epoch,sample,loss,validation_loss,learning_rate,gradient_norm,perplexity
1,2,450,2.3456,2.4123,0.001,1.234,10.43
```

---

### Session Endpoints

#### `GET /api/session/status`

Returns current training session status and progress.

Response:

```json
{
  "is_training": true,
  "session_id": 1,
  "current_epoch": 2,
  "total_epochs": 10,
  "current_sample": 450,
  "total_samples": 1000,
  "progress_percent": 45.0,
  "samples_per_second": 12.5,
  "estimated_time_remaining_seconds": 120.5
}
```

---

#### `GET /api/session/epochs`

Returns per-epoch metrics history.

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
  "epoch_validation_perplexities": [34.2, 17.1, 11.8],
  "epoch_validation_accuracies": [-1.0, -1.0, -1.0],
  "best_validation_loss": 2.5,
  "best_epoch": 2
}
```

**Field notes:**

- `epoch_validation_perplexities` — validation perplexity recorded at the end of each epoch (TD-015).
- `epoch_validation_accuracies` — token-level validation accuracy per epoch (TD-015). `-1.0` entries indicate epochs where accuracy was not computed.

See also [`GET /api/metrics/generation-quality`](#get-apimetricsgeneration-quality) for per-epoch BLEU/ROUGE arrays.

---

### Control Endpoints

#### `POST /api/control/flush`

Forces immediate flush of metrics to disk.

Response:

```json
{
  "status": "success",
  "message": "Metrics flushed to disk"
}
```

---

#### `POST /api/control/clear`

Clears historical metrics from memory.

Response:

```json
{
  "status": "success",
  "message": "Metrics history cleared"
}
```

**Note:** Control endpoints can be disabled with `--no-control` flag.

---

### Health Check

#### `GET /health`

Returns server health status.

Response:

```json
{
  "status": "ok",
  "service": "TrainingMetricsAPI",
  "is_training": true
}
```

---

## Multi-Session API

All endpoints above operate on the implicit `0-default` session for backwards compatibility.
The multi-session API lets you manage and query independent named sessions identified by a
human-readable *session key* (e.g. `42-finetune-gpu0`).

### `GET /api/sessions`

Returns a list of all registered sessions with their current status.

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

### `GET /api/metrics/aggregate`

Returns a compact JSON summary of every currently **active** (training) session.
Completed or idle sessions are omitted.

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

### `GET /api/metrics/prometheus/aggregate` *(TD-021)*

Returns Prometheus text exposition for **all live sessions**, concatenated into a single
scrape payload. Each metric line carries a `session=` label so that individual sessions
are distinguishable in Grafana or any other Prometheus-compatible consumer.

Response (Content-Type: `text/plain`):

```text
# HELP training_session_id Current training session ID
# TYPE training_session_id gauge
training_session_id{session="42-finetune-gpu0"} 42 1748721234000

# HELP training_is_active Whether training is currently active
# TYPE training_is_active gauge
training_is_active{session="42-finetune-gpu0"} 1 1748721234000

# HELP training_loss Current training loss
# TYPE training_loss gauge
training_loss{session="42-finetune-gpu0"} 2.123400 1748721234000

# HELP training_session_id Current training session ID
# TYPE training_session_id gauge
training_session_id{session="43-finetune-gpu1"} 43 1748721235000
...
```

For Prometheus scrape configuration targeting this endpoint:

```yaml
scrape_configs:
  - job_name: 'training_metrics_all'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/api/metrics/prometheus/aggregate'
    scrape_interval: 5s
```

---

### `POST /api/sessions/{key}/start`

Creates the session if it does not exist, then starts a training run for it.
Returns `409 Conflict` if the session is already active.

Request body:

```json
{
  "session_id":    42,
  "total_epochs":  10,
  "total_samples": 5000,
  "label":  "#42: wiki-finetune (gpu0, 2026-05-31)",
  "config": {"lr": 0.001, "batch_size": 32, "dropout": 0.1}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | integer | yes | Numeric session identifier |
| `total_epochs` | integer | no | Total epochs planned for progress calculation |
| `total_samples` | integer | no | Total training samples for ETA estimation |
| `label` | string | no | Human-readable label shown in dashboards and `GET /api/sessions`; empty string silently accepted |
| `config` | object | no | Compact training-configuration snapshot; stored with the session for audit purposes |

Response:

```json
{"status": "ok", "message": "Session started"}
```

---

### `POST /api/sessions/{key}/end`

Marks the session as complete and writes its final summary to disk.

Response:

```json
{"status": "ok", "message": "Session ended"}
```

**Per-session variants of other endpoints** follow the pattern
`/api/sessions/{key}/<resource>` (e.g. `/api/sessions/42-finetune-gpu0/metrics/current`)
and return the same payload shapes as their legacy counterparts documented above.

---

## Command-Line Options

```text
Training Metrics REST API Server
Usage: metrics_api_server [OPTIONS]

Options:
  --port PORT                  Port number (default: 8081)
  --metrics-file FILE          Metrics JSONL file path
  --summary-file FILE          Summary JSON file path
  --prometheus-file FILE       Prometheus file path
  --persist-samples N          Persist every N samples (default: 100)
  --persist-seconds N          Persist every N seconds (default: 30)
  --max-memory-records N       Max records in memory (default: 10000)
  --max-disk-records N         Max records on disk (default: 100000)
  --no-persistence             Disable persistence to disk
  --enable-prometheus          Enable Prometheus format output
  --sweep-interval-seconds N   Seconds between registry sweep runs (default: 60; 0 disables)
  --no-control                 Disable control endpoints (flush, clear)
  --help                       Show help message
```

---

## Integration Examples

### Python Client

```python
import requests
import time

api_url = "http://localhost:8081"

while True:
    # Get current status
    status = requests.get(f"{api_url}/api/session/status").json()

    if status['is_training']:
        progress = status['progress_percent']
        epoch = status['current_epoch']

        # Get metrics
        metrics = requests.get(f"{api_url}/api/metrics/current").json()
        loss = metrics['current_loss']

        print(f"Epoch {epoch} |{progress:.1f}%| Loss: {loss:.4f}")

    time.sleep(2)
```

### JavaScript Client

```javascript
const apiUrl = 'http://localhost:8081';

async function pollMetrics() {
    const response = await fetch(`${apiUrl}/api/metrics/current`);
    const metrics = await response.json();

    console.log(`Loss: ${metrics.current_loss}`);
    console.log(`Epoch: ${metrics.current_epoch}/${metrics.total_epochs}`);
}

setInterval(pollMetrics, 1000);
```

### Bash/curl Script

```bash
#!/bin/bash
API_URL="http://localhost:8081"

while true; do
    STATUS=$(curl -s "$API_URL/api/session/status")
    IS_TRAINING=$(echo "$STATUS" | jq -r '.is_training')

    if [ "$IS_TRAINING" = "true" ]; then
        METRICS=$(curl -s "$API_URL/api/metrics/current")
        LOSS=$(echo "$METRICS" | jq -r '.current_loss')
        EPOCH=$(echo "$STATUS" | jq -r '.current_epoch')

        echo "Epoch: $EPOCH | Loss: $LOSS"
    fi

    sleep 2
done
```

---

## Prometheus Integration

Enable Prometheus format output:

```bash
./metrics_api_server --enable-prometheus
```

Configure Prometheus to scrape the metrics endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'training_metrics'
    static_configs:
      - targets: ['localhost:8081']
    metrics_path: '/api/metrics/prometheus'
    scrape_interval: 5s
```

---

## Use Cases

1. **Real-time Dashboards**: Poll metrics API to build web dashboards
2. **Mobile Monitoring**: Access training progress from mobile apps
3. **CI/CD Integration**: Monitor training jobs in automated pipelines
4. **Multi-job Coordination**: Track multiple training sessions
5. **Alert Systems**: Trigger alerts based on metrics thresholds
6. **Data Science Notebooks**: Query metrics from Jupyter notebooks
7. **Prometheus/Grafana**: Export metrics to monitoring systems

---

## Performance Considerations

- **Non-blocking**: All endpoints are non-blocking and optimized for frequent polling
- **Thread-safe**: Safe to call from multiple clients simultaneously
- **Low overhead**: Minimal performance impact on training
- **Configurable persistence**: Adjust write frequency to balance durability vs. performance
- **Memory bounded**: Automatic cleanup of old records

---

## Security Notes

- API runs on `0.0.0.0` (all interfaces) by default
- No authentication/authorization built-in (add reverse proxy for production)
- Control endpoints can be disabled with `--no-control`
- Consider firewall rules for production deployments
- Use HTTPS reverse proxy (nginx, Apache) for secure access

---

## See Also

- [TrainingMetricsService.hpp](../src/TrainingMetricsService.hpp) - Core metrics service
- [TrainingMetricsAPI.hpp](../src/TrainingMetricsAPI.hpp) - REST API implementation
- [TrainingMetricsAPIExample.cpp](../examples/TrainingMetricsAPIExample.cpp) - Integration examples
