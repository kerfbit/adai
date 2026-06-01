# Proposal: IncrementalTrainer × Metrics Service Decoupling

**Status:** Draft  
**Date:** 2026-05-31  
**Area:** Training infrastructure / IncrementalTrainer / Metrics  

---

## 1. Problem Statement

### 1.1 Trainer is directly coupled to server-side C++ metrics classes

`IncrementalTrainer` and `ChatbotTrainer` currently hold direct C++ dependencies on
`TrainingMetricsService` — a class that belongs exclusively to the metrics API server process.
`IncrementalTrainer` instantiates it:

```cpp
// src/IncrementalTrainer.cpp — lines 232–240, 329–333, 398–402  (three constructors)
if (config.enable_metrics_service) {
    metrics_service_ = std::make_unique<TrainingMetricsService>(config.metrics_config);
}
```

`ChatbotTrainer` stores and calls it directly:

```cpp
// src/ChatbotTrainer.hpp — line 423
void set_metrics_service(TrainingMetricsService* service);

// src/ChatbotTrainer.cpp — ~15 call sites including:
metrics_service_->start_epoch(epoch + 1, num_samples);
metrics_service_->update_sample_metrics(i + 1, loss, grad_norm, lr, ...);
metrics_service_->flag_abnormal_sample(ab);
metrics_service_->end_epoch(epoch + 1, epoch_loss, val_loss, ...);
// ...
```

`TrainingMetricsService` is a stateful server-side object: it owns a ring buffer of historical
records, manages file I/O (`metrics.jsonl`, `metrics_summary.json`, Prometheus output), and runs
push threads. None of this belongs in a training process. The metrics service is an independent
package, ideally running on a separate machine. The trainer's only legitimate interaction with it
is through the HTTP REST API.

### 1.2 Push mode is optional — it should be the only mode

`IncrementalConfig::metrics_push_enabled` controls whether metrics are sent to the server. When
`false`, the trainer runs a full local `TrainingMetricsService` — an in-process metrics store
with no external visibility. This "local mode" contradicts the architectural principle that the
metrics service is an independent, external system. There is no valid reason for a trainer to
maintain a local copy of what the metrics server already manages.

### 1.3 No session label or metadata

`TrainingMetricsService::start_session()` pushes a minimal body:

```json
{ "session_id": 3, "total_epochs": 15, "total_samples": 50000 }
```

The `POST /api/sessions/{key}/start` endpoint (TD-018 §4.4) was designed to accept `label` and
`config` fields for human-readable identification and a training configuration snapshot. Neither
field is populated. `MetricsSessionSummary` has no `label` field, so `GET /api/sessions`
responses contain no useful identifying metadata beyond the raw session key.

### 1.4 No 409 Conflict retry

When a trainer is restarted before the server's TTL expires for the previous run's key — or when
two trainers on the same host coincidentally derive the same key — the server returns HTTP 409.
The trainer has no recovery path; the error is silently discarded and the session never properly
registers.

### 1.5 No background sweep thread in MetricsSessionRegistry

TD-018 §4.8 specified "a background sweep thread running every 60 seconds". This was not
implemented. Eviction is lazy-only: it fires only inside `create_or_get_session()` when a new
session is requested. On a long-running server with infrequent new registrations, completed
sessions accumulate in memory past their TTL.

### 1.6 No Prometheus session-key labels

`to_prometheus_format()` emits flat metric names (e.g., `training_loss 2.34`). In a multi-session
deployment these names collide when a Prometheus scraper targets the server, making it impossible
to distinguish metrics from concurrent training runs. There is also no aggregate endpoint.

---

## 2. Goals

1. **Remove all direct C++ dependencies** between the trainer processes and
   `TrainingMetricsService` / `MetricsSessionRegistry`. The metrics service is an independent
   package; trainers communicate with it exclusively via HTTP.
2. Introduce `MetricsPushClient` — a thin HTTP-only client that provides the same reporting
   interface to `ChatbotTrainer` that `TrainingMetricsService` does today.
3. Introduce `IMetricsReporter` — a minimal abstract interface so `ChatbotTrainer` has no
   compile-time dependency on any concrete metrics class.
4. Remove "local-only" metrics mode. Metrics reporting is always external. If no server URL is
   configured, the trainer runs silently with a no-op reporter.
5. Auto-derive a human-readable session label; propagate it and a training-config snapshot in the
   session-start HTTP body; surface both in `GET /api/sessions` responses.
6. Add 409-conflict retry in `MetricsPushClient` so key conflicts are resolved transparently.
7. Implement the deferred background sweep thread in `MetricsSessionRegistry`.
8. Add per-session `session=` labels to Prometheus output and a new aggregate endpoint.

### Non-goals

- **Parallel trainer pool.** Multiple trainers operate on independent systems. The metrics API
  server is the only shared component across training machines.
- **SQL persistence backend.** Tracked separately as TD-020.
- **Dashboard session selector UI.** Separate UI proposal; the API changes here are a
  prerequisite.
- **Distributed multi-host registry federation.** One metrics API server per deployment is
  sufficient.

---

## 3. Current Architecture

```
Trainer process
┌─ IncrementalTrainer ──────────────────────────────────────────────────────┐
│  ctor() { metrics_service_ = make_unique<TrainingMetricsService>(...) }   │  ← WRONG
│                                                                            │
│  train_incremental() {                                                     │
│    derive session key and push URL                                         │
│    metrics_service_->set_config(...)          // patch push URL            │
│    trainer.set_metrics_service(metrics_service_.get())                     │
│    metrics_service_->start_session(id, epochs, n)                          │
│    trainer.train(epochs)                                                   │
│    metrics_service_->end_session()                                         │
│  }                                                                         │
└────────────────────────────────────────────────────────────────────────────┘
│  owns
▼
TrainingMetricsService     ← server-side class running inside trainer process
  in-memory ring buffer    ← unnecessary local copy of server state
  file I/O (jsonl, json)   ← conflicts with server's own persistence
  push thread              ← the only part that should exist in the trainer
│ (when push enabled)
│  POST /api/sessions/{key}/start  { session_id, total_epochs, total_samples }
│  POST .../metrics/sample         (per-sample metrics)
│  POST .../epoch/end              (per-epoch metrics)
│  ...
▼
MetricsAPIServer process (same or different machine)
  MetricsSessionRegistry
    TrainingMetricsService (server-managed, correct owner)
```

**The problem in one sentence:** the trainer runs a full copy of `TrainingMetricsService`
purely for its push side-effects, dragging all server-side storage and threading overhead into the
training process.

---

## 4. Proposed Architecture

### 4.1 Separation of Concerns

```
Trainer process                          Metrics API server (any host)
┌─ IncrementalTrainer ───────┐           ┌─ MetricsAPIServer ──────────────┐
│  Derive session key        │           │  MetricsSessionRegistry         │
│  Derive session label      │   HTTP    │    owns N TrainingMetricsService │
│  Create MetricsPushClient  │ ────────► │    background sweep thread      │
│  Pass client to Trainer    │  POST     │    per-session file persistence │
└────────────────────────────┘           │    Prometheus output            │
┌─ ChatbotTrainer ───────────┐           └─────────────────────────────────┘
│  IMetricsReporter*         │
│  calls: start_epoch()      │
│         update_sample()    │
│         end_epoch() ...    │
└────────────────────────────┘
```

The trainer side knows nothing about `TrainingMetricsService`, `MetricsSessionRegistry`, or any
other server-side C++ class. All metrics data flows outward via HTTP POST.

### 4.2 IMetricsReporter Interface

`ChatbotTrainer`'s dependency on `TrainingMetricsService*` is replaced by a dependency on
`IMetricsReporter*`, defined in a new header with no dependencies on any metrics-service
internals:

```cpp
// src/IMetricsReporter.hpp  (new file)
#pragma once
#include <string>

struct AbnormalSample;   // moved here from TrainingMetricsService.hpp

class IMetricsReporter {
   public:
    virtual ~IMetricsReporter() = default;

    virtual void start_epoch(int epoch, int total_samples) = 0;
    virtual void end_epoch(int epoch, float loss, float val_loss,
                           double duration_seconds, float lr) = 0;
    virtual void update_sample_metrics(int sample, float loss, float grad_norm,
                                       float lr, float samples_per_sec) = 0;
    virtual void update_validation_metrics(float val_loss, float accuracy,
                                           float perplexity) = 0;
    virtual void update_best_metrics(float best_val_loss, int best_epoch) = 0;
    virtual void update_advanced_epoch_metrics(float grad_variance,
                                               float compute_time_ratio,
                                               float weight_update_ratio) = 0;
    virtual void update_adaptive_clip_metrics(float threshold, int spikes) = 0;
    virtual void update_adaptive_clip_epoch(float avg_threshold, int spikes) = 0;
    virtual void update_activation_saturation(float saturation) = 0;
    virtual void update_attention_entropy(float entropy) = 0;
    virtual void update_padding_efficiency(float efficiency) = 0;
    virtual void update_generation_quality_metrics(float bleu4, float rouge1,
                                                   float rouge2, float rougeL) = 0;
    virtual void flag_abnormal_sample(const AbnormalSample& sample) = 0;
};
```

The `AbnormalSample` struct moves from `TrainingMetricsService.hpp` into `IMetricsReporter.hpp`
so that `ChatbotTrainer` can construct one without depending on the server-side header.

The outlier detection thresholds previously read from `MetricsServiceConfig` via
`metrics_service_->get_config()` (line 635 in `ChatbotTrainer.cpp`) move to `TrainingConfig`
where they belong — they describe what the trainer considers abnormal, not the metrics server's
storage policy.

### 4.3 MetricsPushClient

`MetricsPushClient` is the sole concrete production implementation of `IMetricsReporter`. It maps
every interface method to a non-blocking enqueue that is drained by a single dedicated background
thread which performs the actual HTTP POST to the appropriate session-scoped endpoint.

#### 4.3.1 Class declaration

```cpp
// src/MetricsPushClient.hpp  (new file)
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include "IMetricsReporter.hpp"

class MetricsPushClient final : public IMetricsReporter {
   public:
    // max_queue_depth: cap on buffered events; oldest per-sample entries are
    // dropped when full — epoch and session events are always preserved.
    MetricsPushClient(std::string server_url,
                      std::string session_key,
                      int push_timeout_ms  = 1000,
                      size_t max_queue_depth = 1000);

    ~MetricsPushClient();   // stops push thread; see §4.3.3

    // Session lifecycle — called by IncrementalTrainer, not part of IMetricsReporter
    int  start_session(int session_id, int total_epochs, int total_samples,
                       const std::string& label, const std::string& config_json);
    void end_session();     // enqueues session-end event, then drains and joins thread

    std::string session_key() const { return session_key_; }

    // IMetricsReporter — each method builds JSON and calls enqueue(); returns immediately
    void start_epoch(int epoch, int total_samples) override;
    void end_epoch(int epoch, float loss, float val_loss,
                   double duration_seconds, float lr) override;
    void update_sample_metrics(int sample, float loss, float grad_norm,
                               float lr, float samples_per_sec) override;
    // ... (all other IMetricsReporter methods)

   private:
    enum class EventPriority { Sample, Epoch, Session };

    struct PushEvent {
        std::string   endpoint;   // e.g. "/metrics/sample"
        std::string   body;       // JSON payload
        EventPriority priority;
    };

    void enqueue(std::string endpoint, std::string body, EventPriority priority);
    void push_loop();   // runs on push_thread_
    bool attempt_post(const std::string& endpoint, const std::string& body);

    std::string  server_url_;
    std::string  session_key_;
    int          push_timeout_ms_;
    size_t       max_queue_depth_;

    std::deque<PushEvent>   queue_;
    std::mutex              queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool>       stop_{false};
    std::thread             push_thread_;   // started in constructor, joined in end_session()
};
```

#### 4.3.2 Threading model

A **single background thread** (`push_thread_`) is started in the constructor and owns the
entire HTTP lifecycle:

```
Training loop thread           push_thread_
       │                            │
 update_sample_metrics()            │
   → enqueue(body, Sample)          │
   → queue_cv_.notify_one()         │
   → returns ~100 ns                │
       │                     wait(queue_cv_)
       │                     dequeue PushEvent
       │                     POST /api/sessions/{key}/metrics/sample
       │                       ↳ retry up to 3× with exponential backoff
       │                     if HTTP 5xx / network error: log warning
       │                     loop
```

All `IMetricsReporter` methods call `enqueue()` and return in O(1) time — the training loop is
never stalled by network I/O, server latency, or transient failures.

**Queue overflow policy:** when `queue_.size() >= max_queue_depth_`, `enqueue()` checks the
priority of the incoming event:
- `EventPriority::Sample` — the incoming event is **dropped** and a warning is logged once per
  overflow episode. Per-sample data is lossy by design under backpressure.
- `EventPriority::Epoch` or `EventPriority::Session` — the **oldest Sample-priority entry** in
  the queue is evicted to make room. Epoch and session events are never dropped.

This ensures that even during extended server unavailability, epoch summaries and the
session-end notification always reach the server when it recovers.

**Retry policy:** `attempt_post()` retries up to three times on HTTP 5xx or network error, with
back-off delays of 0 ms, 200 ms, and 1000 ms. HTTP 4xx errors (including 409) are not retried —
they are logged and discarded.

#### 4.3.3 Shutdown sequence

`end_session()` must be called by `IncrementalTrainer` after `trainer.train()` returns:

1. Enqueue the session-end POST with `EventPriority::Session`.
2. Set `stop_ = true`.
3. Call `queue_cv_.notify_one()`.
4. Call `push_thread_.join()` — blocks until the thread has drained the queue and sent the
   session-end event.

The destructor also calls `join()` if `push_thread_` is still joinable (guard against exceptions
that bypass `end_session()`). This guarantees no threads outlive the `MetricsPushClient` object.

`MetricsPushClient` has **no ring buffer, no file I/O, and no local metrics state** beyond the
bounded event queue.

### 4.4 NullMetricsReporter

A no-op implementation handles the case where no metrics server URL is configured and is used
in tests that do not exercise metrics behaviour:

```cpp
// src/IMetricsReporter.hpp  (same header, below the interface)
class NullMetricsReporter final : public IMetricsReporter {
   public:
    void start_epoch(int, int) override {}
    void end_epoch(int, float, float, double, float) override {}
    void update_sample_metrics(int, float, float, float, float) override {}
    // ... (all other methods are no-ops)
    void flag_abnormal_sample(const AbnormalSample&) override {}
};
```

When `METRICS_SERVER_URL` is empty or `enable_metrics_reporting` is false, `IncrementalTrainer`
creates a `NullMetricsReporter`. No push threads are started and no `TrainingMetricsService` is
instantiated anywhere in the trainer process.

### 4.5 Session Label Auto-Derivation

A new `std::string label` field is added to `IncrementalConfig` and mapped from a new
`METRICS_SESSION_LABEL` config-file key.

When `label` is empty at the start of a training run, the trainer auto-derives:

```
"#{session_id}: {first_data_source | 'general'} ({hostname}, {date})"
```

| Scenario | Auto-derived label |
|----------|--------------------|
| session 3, first pending file `alpaca.txt`, host `devbox` | `#3: alpaca (devbox, 2026-05-31)` |
| session 7, full retrain, first trained file `gutenberg_1234.txt`, host `gpu0` | `#7: gutenberg_1234 (gpu0, 2026-05-31)` |
| session 1, no data files, host `trainer` | `#1: general (trainer, 2026-05-31)` |

The first data source is derived using `std::filesystem::path::stem()` only — no parent directory
path is included. For `train_full_retrain()` the first file from `trained_data_files_` is used
since there are no pending files at that point.

A compact `config_snapshot` JSON string is also constructed:

```json
{ "d_model": 256, "num_heads": 4, "d_ff": 1024,
  "num_encoder_layers": 6, "num_decoder_layers": 6,
  "learning_rate": 0.0001, "num_epochs": 15, "batch_size": 32 }
```

Both `label` and `config_snapshot` are sent as fields in the `start_session()` POST body.

### 4.6 409 Conflict Retry

`MetricsPushClient::start_session()` returns the HTTP response code. `IncrementalTrainer` wraps
client creation and session start in a retry loop:

```cpp
// Pseudocode — src/IncrementalTrainer.cpp
const std::string base_key = derive_or_configure_session_key(session_id);
std::string session_key    = base_key;
std::unique_ptr<MetricsPushClient> client;

for (int attempt = 0; attempt < 3; ++attempt) {
    if (attempt > 0) {
        session_key = base_key + "-" + std::to_string(attempt + 1);
        adai::Logger::warn("Metrics key conflict (409), retrying with '{}'", session_key);
        std::this_thread::sleep_for(std::chrono::milliseconds(100 * attempt));
    }
    client = std::make_unique<MetricsPushClient>(
        config.metrics_server_url, session_key, config.metrics_push_timeout_ms);
    int rc = client->start_session(current_session_id + 1, num_epochs,
                                   static_cast<int>(training_pairs.size()),
                                   effective_label, config_snapshot_json);
    if (rc != 409) break;
}
active_session_key_ = session_key;
```

Suffixes are appended to `base_key` (not to prior suffixes) so the progression is
`1-devbox1234`, `1-devbox1234-2`, `1-devbox1234-3`. If all three attempts return 409, training
proceeds with the last key and a warning is logged — metrics may be lost for this run but the
training job is not aborted.

### 4.7 Background Sweep Thread in MetricsSessionRegistry

The TD-018 §4.8 sweep thread is implemented on the server side. New private members are added to
`MetricsSessionRegistry`:

```cpp
std::thread              sweep_thread_;
std::atomic<bool>        stop_sweep_{false};
std::condition_variable_any sweep_cv_;
```

The constructor gains a fourth parameter `sweep_interval_seconds` (default: 60). When greater
than zero, the constructor starts the sweep thread. The thread sleeps for `sweep_interval_seconds`
between each call to `evict_completed_sessions_locked(completed_ttl_seconds_)`. The destructor
sets `stop_sweep_ = true`, notifies the condition variable, and joins the thread before the
session map is destroyed.

`TrainingMetricsAPIServer` passes `Config::metrics_sweep_interval_seconds` (already a parsed
config field from TD-018) as the fourth argument.

### 4.8 Prometheus Session Labels

`TrainingMetricsService::to_prometheus_format()` gains an optional `session_key` parameter:

```cpp
std::string to_prometheus_format(const std::string& session_key = "") const;
```

When non-empty, every metric line is annotated with a `session` label:

```
# Before
training_loss 2.34

# After
training_loss{session="3-devbox1234"} 2.34
```

`TrainingMetricsAPI::handle_prometheus_metrics()` passes the current session key. A new endpoint
is added:

```
GET /api/metrics/prometheus/aggregate
```

This snapshots the session `shared_ptr` list under the registry's shared read lock, then releases
the lock before calling `to_prometheus_format(key)` on each session. The concatenated output
represents all live sessions in a single Prometheus-compatible response.

---

## 5. Trainer-Side Changes

### 5.1 New files

| File | Description |
|------|-------------|
| `src/IMetricsReporter.hpp` | Abstract interface + `AbnormalSample` struct + `NullMetricsReporter` |
| `src/MetricsPushClient.hpp` | `MetricsPushClient` declaration; `PushEvent` struct; `EventPriority` enum |
| `src/MetricsPushClient.cpp` | Background push thread (`push_loop()`), bounded queue with priority overflow policy, `attempt_post()` with 3× retry/backoff, shutdown sequence |

### 5.2 Removed items

| Item | Location | Replacement |
|------|----------|-------------|
| `std::unique_ptr<TrainingMetricsService> metrics_service_` | `IncrementalTrainer` private member | Removed entirely |
| `#include "TrainingMetricsService.hpp"` | `IncrementalTrainer.hpp` | `#include "IMetricsReporter.hpp"` |
| `#include "MetricsSessionRegistry.hpp"` | `IncrementalTrainer.hpp` | Removed entirely |
| `bool metrics_push_enabled` | `IncrementalConfig` | Replaced by: empty `metrics_server_url` → `NullMetricsReporter` |
| `MetricsServiceConfig metrics_config` | `IncrementalConfig` | Replaced by flat fields (`metrics_server_url`, `metrics_push_timeout_ms`) |
| `void set_metrics_service(TrainingMetricsService*)` | `ChatbotTrainer` | `void set_metrics_reporter(IMetricsReporter*)` |
| `adv_cfg = metrics_service_->get_config()` | `ChatbotTrainer.cpp` line 635 | Outlier thresholds read from `TrainingConfig` directly |

### 5.3 IncrementalConfig changes

```cpp
struct IncrementalConfig {
    // ... (existing fields: session_dir, checkpointing, data management, etc.) ...

    // Metrics reporting — HTTP only; no local TrainingMetricsService
    bool        enable_metrics_reporting = true;    // false → NullMetricsReporter
    std::string metrics_server_url       = "";      // empty → NullMetricsReporter
    std::string metrics_session_key      = "";      // empty → auto-derived
    std::string label                    = "";      // empty → auto-derived
    int         metrics_push_timeout_ms  = 1000;
};
```

### 5.4 IncrementalTrainer member changes

**Removed:**
```cpp
std::unique_ptr<TrainingMetricsService> metrics_service_;   // REMOVED (legacy)
```

**Added:**
```cpp
std::string active_session_key_;   // key used in most recent run; empty before first run
```

**Added public method:**
```cpp
std::string get_metrics_session_key() const;    // returns active_session_key_
```

**Constructor changes:** All three constructor overloads remove the
`make_unique<TrainingMetricsService>()` call entirely.

**train_incremental() / train_full_retrain() changes:**
1. If `metrics_server_url` is empty or `enable_metrics_reporting` is false: create `NullMetricsReporter`.
2. Otherwise: derive key and label, create `MetricsPushClient`, call `start_session()` with 409 retry.
3. Pass reporter to `ChatbotTrainer` via `set_metrics_reporter()`.
4. After training completes: call `client->end_session()` (no-op on `NullMetricsReporter`).
5. Store the final key in `active_session_key_`.

### 5.5 ChatbotTrainer changes

`set_metrics_service(TrainingMetricsService* service)` is renamed to
`set_metrics_reporter(IMetricsReporter* reporter)`. All ~15 call sites in `ChatbotTrainer.cpp`
change from `metrics_service_->` to `metrics_reporter_->`. The method names on `IMetricsReporter`
are identical to those currently called on `TrainingMetricsService`, so each call site body is
unchanged.

The `adv_cfg = metrics_service_->get_config()` call at line 635 is removed. The three outlier
detection fields it provided (`loss_outlier_z_threshold`, `grad_norm_outlier_threshold`,
`max_abnormal_samples`) are added directly to `TrainingConfig`.

### 5.6 TrainingConfig additions

```cpp
// src/ChatbotTrainer.hpp — TrainingConfig struct
float loss_outlier_z_threshold    = 3.0f;     // moved from MetricsServiceConfig
float grad_norm_outlier_threshold = 10.0f;    // moved from MetricsServiceConfig
int   max_abnormal_samples        = 1000;     // moved from MetricsServiceConfig
```

These are wired through `IncrementalTrainer::make_incremental_config()` from `ServiceConfig`.

---

## 6. Metrics Server-Side Changes

The metrics API server (`TrainingMetricsAPIServer`, `MetricsSessionRegistry`,
`TrainingMetricsService`, `TrainingMetricsAPI`) changes are limited to:

1. **`MetricsSessionRegistry`** — add background sweep thread (§4.7); add `label` and
   `config_snapshot` fields to `MetricsSessionSummary` and `SessionEntry`.
2. **`TrainingMetricsService::start_session()`** — add `label` and `config_snapshot` parameters;
   store them so `list_sessions()` can return them.
3. **`TrainingMetricsAPI::handle_session_start()`** — parse `label` and `config` from the POST
   request body; forward to the registry.
4. **`TrainingMetricsService::to_prometheus_format()`** — add optional `session_key` parameter
   (§4.8).
5. **`TrainingMetricsAPI`** — add `GET /api/metrics/prometheus/aggregate` (§4.8).
6. **`TrainingMetricsService::persist_summary()` / `restore_from_summary()`** — `label` and
   `config_snapshot` are written into `_metrics_summary.json` alongside the existing snapshot
   fields. `restore_from_summary()` reads them back and repopulates `MetricsSessionSummary::label`
   and `config_snapshot` on server restart. This ensures label persistence with no dependency on
   the SQL backend (TD-020). When TD-020 is active, `upsert_session()` additionally stores both
   fields in the `sessions` table for durable storage independent of the JSON file.

No changes to the `TrainingMetricsService` data model, ring buffer, or any other endpoint.

---

## 7. Configuration Changes

### 7.1 New ServiceConfig field

```cpp
// src/Config.hpp
std::string metrics_session_label = "";   // empty → auto-derived; maps to IncrementalConfig::label
```

Parsed from config file key `METRICS_SESSION_LABEL` and environment variable override.

### 7.2 make_incremental_config() additions

```cpp
cfg.label                         = svc.metrics_session_label;
cfg.loss_outlier_z_threshold      = svc.loss_outlier_z_threshold;
cfg.grad_norm_outlier_threshold   = svc.grad_norm_outlier_threshold;
cfg.max_abnormal_samples          = svc.max_abnormal_samples;
```

### 7.3 config.conf / config-remote.conf

New key added beneath `METRICS_SESSION_KEY`:

```ini
# Optional human-readable label surfaced in GET /api/sessions responses.
# Leave blank to auto-derive from session ID, hostname, and data source name.
# METRICS_SESSION_LABEL =
```

`METRICS_PUSH_ENABLED` is removed from both files. Push to the metrics API server is now the
only mode; whether it is active is determined solely by whether `METRICS_SERVER_URL` is set.

---

## 8. Implementation Plan

| Step | File(s) affected | Description | Depends on |
|------|-----------------|-------------|-----------|
| 1 | `src/IMetricsReporter.hpp` *(new)* | Define `IMetricsReporter`, `AbnormalSample`, `NullMetricsReporter` | — |
| 2 | `src/MetricsPushClient.hpp` / `.cpp` *(new)* | Implement `MetricsPushClient`; all methods post to session-scoped endpoints; `start_session()` returns HTTP status code | 1 |
| 3 | `src/ChatbotTrainer.hpp` / `.cpp` | Replace `TrainingMetricsService*` with `IMetricsReporter*`; rename to `set_metrics_reporter()`; add outlier fields to `TrainingConfig`; remove `get_config()` call | 1 |
| 4 | `src/Config.hpp` / `.cpp` | Add `metrics_session_label`; ensure outlier threshold fields present | — |
| 5 | `src/IncrementalTrainer.hpp` | Remove `metrics_service_` member; update `IncrementalConfig`; add `active_session_key_`; add `get_metrics_session_key()` | 1 |
| 6 | `src/IncrementalTrainer.cpp` | Remove `TrainingMetricsService` construction from all three constructors; replace training-run metrics setup with `MetricsPushClient` + 409 retry; add `derive_session_label()`; update `make_incremental_config()` | 2, 3, 4, 5 |
| 7 | `src/MetricsSessionRegistry.hpp` | Add sweep thread; add `label` + `config_snapshot` to `MetricsSessionSummary` and `SessionEntry` | — |
| 8 | `src/TrainingMetricsService.hpp` / `.cpp` | `start_session()` gains `label` + `config_snapshot` params; add `session_key` param to `to_prometheus_format()` | 7 |
| 9 | `src/TrainingMetricsAPI.hpp` / `.cpp` | Parse `label` + `config` from POST start body; add `GET /api/metrics/prometheus/aggregate` | 7, 8 |
| 10 | `src/TrainingMetricsAPIServer.cpp` | Pass `metrics_sweep_interval_seconds` as 4th registry constructor arg | 7 |
| 11 | `config.conf`, `config-remote.conf` | Add `METRICS_SESSION_LABEL`; remove `METRICS_PUSH_ENABLED` | 4 |
| 12 | `tests/incremental_trainer_registry_test.cpp` *(new)* | Push-client creation, label derivation, key accessor, 409 retry, null-reporter path | 5, 6 |
| 13 | `tests/metrics_session_registry_test.cpp` | Add sweep-thread and label/config-snapshot tests | 7 |
| 14 | `tests/CMakeLists.txt` | Register `incrementalTrainerDecouplingTests` target | 12 |
| 15 | `docs/development/TRAINING_METRICS_API.md` | Document `GET /api/metrics/prometheus/aggregate`; update POST start body schema with `label` + `config` fields | 9 |

Steps 1, 4, and 7 have no dependencies and can start in parallel.  
Steps 2, 3, and 5 depend on step 1.  
Step 6 depends on steps 2, 3, 4, and 5.  
Steps 8 and 9 depend on step 7.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| All existing call sites of `set_metrics_service()` and all tests that construct a `TrainingMetricsService` to inject into `ChatbotTrainer` fail to build | High (guaranteed until step 3) | Build error | Steps 3 and 6 are coordinated; the rename is mechanical — callers pass the same pointer, now typed as `IMetricsReporter*` |
| Tests that rely on a locally-owned `TrainingMetricsService` for in-process metric inspection can no longer do so | Medium | Test refactoring needed | Replace with a `RecordingMetricsReporter` test double (a test-only `IMetricsReporter` that captures calls for assertion), defined in `tests/TestHelpers.hpp` |
| `push_thread_` outlives the trainer if `end_session()` is not called (e.g., exception during training) | Low | Leaked thread | Destructor calls `push_thread_.join()` if joinable; `stop_` is set atomically before join so the thread exits cleanly regardless of queue state |
| Deployments using `METRICS_PUSH_ENABLED = false` as a way to collect local metrics (e.g., embedded dashboards) lose that capability | Low | Behaviour change | This mode was never visible externally; affected users should instead run the metrics API server locally (`localhost:8081`), which achieves the same result with proper session isolation |
| Sweep thread in `MetricsSessionRegistry` wakes during destructor teardown | Low | Crash | `stop_sweep_ = true` and `notify_all()` called before the sessions map is destroyed; thread joins before destructor returns |
| `to_prometheus_format(session_key)` called concurrently from the aggregate endpoint | Low | Data race | `to_prometheus_format()` holds the per-service `mutex_` for its full duration; no additional locking needed |

---

## 10. Open Questions

1. **Label persistence across server restarts — Resolved.** `label` and `config_snapshot` are
   written to `_metrics_summary.json` by `persist_summary()` and read back by
   `restore_from_summary()` on restart (see §6, item 6). Labels survive server restarts with no
   dependency on the SQL backend. When the SQL backend (TD-020) is active, `upsert_session()`
   additionally persists both fields to the `sessions` table, providing a second durable copy
   independent of the JSON file.

2. **config_snapshot depth.** The proposal stores seven scalar fields from `TrainingConfig`.
   Should adaptive-gradient-clip parameters and generation-quality settings be included, or is
   the minimal set sufficient for comparison use in `GET /api/sessions`?

3. **RecordingMetricsReporter as a shared test utility.** Should a `RecordingMetricsReporter`
   that captures all calls for assertion be shipped as a first-class test utility in
   `tests/TestHelpers.hpp`, or is it left to individual test files to implement inline?

4. **Prometheus aggregate content policy.** Should `GET /api/metrics/prometheus/aggregate`
   include completed (non-training) sessions not yet evicted, or restrict to
   `is_training == true` sessions only?
