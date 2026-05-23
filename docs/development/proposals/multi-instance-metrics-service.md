# Proposal: Multi-Instance Training Metrics Service

**Status:** Draft  
**Date:** 2026-05-21  
**Area:** Training infrastructure / Metrics  

---

## 1. Problem Statement

The current `TrainingMetricsService` is architected around a single training session at a time. Key assumptions baked into the design include:

- A single `TrainingMetricsSnapshot current_snapshot_` holds all live data.
- A single `std::atomic<int> current_session_id_` tracks the active session.
- All file paths are configured globally (`METRICS_FILE`, `METRICS_SUMMARY_FILE`, `METRICS_PROMETHEUS_FILE`). Concurrent trainers would overwrite each other's files.
- REST API endpoints (`GET /api/metrics/current`, `POST /api/session/start`, etc.) act implicitly on the one active session — calling `POST /api/session/start` from a second trainer silently replaces the first.
- The `GlobalMetricsService::instance()` singleton gives all callers the same in-process object.

Launching a second training job (e.g., a hyperparameter search, a parallel fine-tune, or a second GPU node) therefore corrupts the metrics of the first job.

---

## 2. Goals

1. Allow any number of training instances to report metrics to one `TrainingMetricsAPIServer` concurrently without data loss or cross-contamination.
2. Keep the existing API shape backwards-compatible for callers that specify a `session_id` of `0`.
3. Expose a new `/api/sessions` enumeration endpoint so dashboards can discover active and recently-completed sessions.
4. Scope all file I/O per session so files are never shared between concurrent writers.
5. Bound total memory consumption regardless of how many sessions are open simultaneously.

### Non-goals

- Distributed metrics aggregation across multiple API server processes (out of scope; one server per host is sufficient).
- Changing the Tizen dashboard front-end protocol beyond adding a session selector dropdown (a separate UI proposal).
- Moving persistence to a database engine (SQLite or similar could be a future follow-up).

---

## 3. Current Architecture Summary

```
TrainingMetricsService (one global instance)
├── TrainingMetricsSnapshot current_snapshot_   ← single session
├── std::vector<PersistentMetricsRecord> history_
├── std::vector<AbnormalSample> abnormal_samples_
├── std::mutex mutex_
└── File handles → training_sessions/metrics.jsonl
                    training_sessions/metrics_summary.json
                    training_sessions/metrics.prom      (optional)

TrainingMetricsAPI
└── Routes all HTTP verbs to the one TrainingMetricsService instance

config.conf
└── METRICS_FILE, METRICS_SUMMARY_FILE fixed paths
```

---

## 4. Proposed Architecture

### 4.1 Session Registry

Introduce a `MetricsSessionRegistry` that owns a map from `session_key` (see §4.2) to a `TrainingMetricsService` instance. The registry is the sole owner of all session objects and is injected into `TrainingMetricsAPI` at construction.

```
MetricsSessionRegistry
├── std::unordered_map<std::string, std::shared_ptr<TrainingMetricsService>> sessions_
├── std::shared_mutex registry_mutex_     ← reader–writer lock
├── size_t max_live_sessions_             ← configurable cap (default: 16)
├── create_or_get_session(key) → shared_ptr<TrainingMetricsService>
├── get_session(key)           → optional<shared_ptr<TrainingMetricsService>>
├── list_sessions()            → vector<SessionSummary>
└── evict_completed_sessions(max_age_seconds)
```

`TrainingMetricsAPI` acquires a session handle from the registry per request, then delegates to it exactly as it does today. No logic inside `TrainingMetricsService` changes.

### 4.2 Session Key

A **session key** is a short, URL-safe string that uniquely identifies one training run. Trainers supply it when starting a session via `POST /api/session/start`. It appears in every subsequent API call as a path segment or query parameter.

**Format:** `{session_id}-{instance_tag}`, e.g. `42-gpu0`, `7-finetune`, `1-default`.

For backwards compatibility, requests that omit the key are treated as targeting key `"0-default"` — the same slot that existed before this change.

The key must match `^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$`. The server rejects keys that fail this pattern with HTTP 400.

### 4.3 URL Scheme Changes

All session-scoped routes gain a `/{session_key}` prefix:

| Old route | New route | Notes |
|-----------|-----------|-------|
| `POST /api/session/start` | `POST /api/sessions/{key}/start` | Creates session slot in registry |
| `POST /api/session/end` | `POST /api/sessions/{key}/end` | Marks session completed; slot kept for `METRICS_COMPLETED_TTL_SECONDS` |
| `POST /api/epoch/start` | `POST /api/sessions/{key}/epoch/start` | — |
| `POST /api/epoch/end` | `POST /api/sessions/{key}/epoch/end` | — |
| `POST /api/metrics/sample` | `POST /api/sessions/{key}/metrics/sample` | — |
| `POST /api/metrics/validation` | `POST /api/sessions/{key}/metrics/validation` | — |
| `POST /api/metrics/best` | `POST /api/sessions/{key}/metrics/best` | — |
| `POST /api/metrics/advanced` | `POST /api/sessions/{key}/metrics/advanced` | — |
| `POST /api/metrics/generation-quality` | `POST /api/sessions/{key}/metrics/generation-quality` | — |
| `POST /api/control/flush` | `POST /api/sessions/{key}/control/flush` | — |
| `POST /api/control/clear` | `POST /api/sessions/{key}/control/clear` | — |
| `GET /api/metrics/current` | `GET /api/sessions/{key}/metrics/current` | — |
| `GET /api/metrics/summary` | `GET /api/sessions/{key}/metrics/summary` | — |
| `GET /api/metrics/history` | `GET /api/sessions/{key}/metrics/history` | — |
| `GET /api/metrics/prometheus` | `GET /api/sessions/{key}/metrics/prometheus` | — |
| `GET /api/metrics/csv` | `GET /api/sessions/{key}/metrics/csv` | — |
| `GET /api/metrics/abnormal` | `GET /api/sessions/{key}/metrics/abnormal` | — |
| `GET /api/metrics/generation-quality` | `GET /api/sessions/{key}/metrics/generation-quality` | — |
| `GET /api/metrics/padding-efficiency` | `GET /api/sessions/{key}/metrics/padding-efficiency` | — |
| `GET /api/session/status` | `GET /api/sessions/{key}/status` | — |
| `GET /api/session/epochs` | `GET /api/sessions/{key}/epochs` | — |
| *(new)* | `GET /api/sessions` | List all live + recently completed sessions |
| *(new)* | `GET /api/metrics/aggregate` | Aggregate view across all live sessions |
| `GET /health` | `GET /health` | Unchanged; reports overall server health |

**Backwards compatibility layer:** The old flat routes (`POST /api/session/start`, `GET /api/metrics/current`, etc.) are preserved as aliases that map to session key `"0-default"`. They will emit a deprecation header (`Deprecation: true`, `Link: /api/sessions/0-default/...`) and can be removed in a future release.

### 4.4 `POST /api/sessions/{key}/start` — Session Registration

Request body (JSON):
```json
{
  "session_id": 42,
  "total_epochs": 15,
  "total_samples": 50000,
  "label": "optional human-readable name",
  "config": { "d_model": 256, "num_heads": 4 }
}
```

- If the key already exists **and** `is_training == true`, respond with HTTP 409 Conflict.
- If the key exists but `is_training == false` (a completed previous session with the same key), the registry evicts the old slot and creates a fresh one.
- The `label` and `config` fields are stored in the session summary and returned by `GET /api/sessions`.

### 4.5 `GET /api/sessions`

Returns the session index:

```json
{
  "sessions": [
    {
      "key": "42-gpu0",
      "session_id": 42,
      "label": "fine-tune run A",
      "is_training": true,
      "current_epoch": 3,
      "total_epochs": 15,
      "current_loss": 2.341,
      "best_validation_loss": 2.310,
      "session_start_time": "2026-05-21T10:15:00Z",
      "last_update_time": "2026-05-21T11:02:34Z",
      "metrics_url": "/api/sessions/42-gpu0/metrics/current"
    },
    {
      "key": "7-finetune",
      "session_id": 7,
      "label": "",
      "is_training": false,
      "completed_at": "2026-05-21T09:55:12Z",
      "best_validation_loss": 1.987,
      "metrics_url": "/api/sessions/7-finetune/metrics/current"
    }
  ],
  "total": 2,
  "live": 1
}
```

### 4.6 `GET /api/metrics/aggregate`

A lightweight cross-session view, useful for dashboards that want a bird's-eye comparison:

```json
{
  "live_sessions": 2,
  "sessions": [
    { "key": "42-gpu0", "epoch": 3, "loss": 2.341, "validation_loss": 2.410 },
    { "key": "1-default", "epoch": 10, "loss": 1.892, "validation_loss": 1.901 }
  ]
}
```

This endpoint holds the registry reader lock for the minimum time needed (snapshot values only, no deep copy of history vectors).

### 4.7 Per-Session File Paths

When a session is created the registry derives per-session file paths using the session key:

```
METRICS_FILE         → training_sessions/{key}_metrics.jsonl
METRICS_SUMMARY_FILE → training_sessions/{key}_metrics_summary.json
METRICS_PROMETHEUS_FILE (if enabled) → training_sessions/{key}_metrics.prom
ABNORMAL_SAMPLES_FILE → training_sessions/{key}_abnormal_samples.json
```

For the `"0-default"` key (backwards-compat slot), the paths remain the legacy values from `config.conf` so that existing deployments continue to work without file renames.

The `TrainingMetricsService` constructor already accepts `Config` for paths; no changes to its constructor signature are required. The registry passes a per-session `Config` copy with the paths substituted.

### 4.8 Memory Management

The registry enforces `METRICS_MAX_LIVE_SESSIONS` (default: 16). On `create_or_get_session()`:

1. If the registry is at capacity and no completed session can be evicted, respond HTTP 503 with body `{"error": "metrics_server_full", "max_live_sessions": 16}`.
2. Completed sessions older than `METRICS_COMPLETED_TTL_SECONDS` (default: 3600) are evicted lazily on each `create_or_get_session()` call and by a background sweep thread running every 60 seconds.

Per-session in-memory limits (`METRICS_MAX_RECORDS_IN_MEMORY`, `METRICS_MAX_RECORDS_ON_DISK`) remain unchanged; the registry merely multiplies them by the number of live sessions.

### 4.9 Thread Safety

| Concern | Solution |
|---------|----------|
| Registry map reads | `std::shared_mutex` with shared lock for reads, exclusive lock only for create/evict |
| Per-session data | Existing `std::mutex mutex_` inside each `TrainingMetricsService` — unchanged |
| Concurrent `POST /api/sessions/{key}/start` for the same key | Registry exclusive lock + HTTP 409 guard |
| Eviction during active request | `std::shared_ptr` ownership: registry eviction drops its `shared_ptr`; any in-flight request holding its own copy continues safely until it returns |

---

## 5. Trainer-Side Changes

### 5.1 Session Key Configuration

Add `METRICS_SESSION_KEY` to `config.conf`. Trainers pick a key:

```ini
METRICS_SESSION_KEY = 1-gpu0
```

If absent, the trainer generates a key from `session_id` and hostname/process-ID:

```
{session_id}-{hostname[:8]}{pid % 10000}
```

e.g. `3-devbox1234`. This ensures uniqueness without manual configuration in the common single-machine case.

### 5.2 HTTP Client Changes

The existing fire-and-forget push thread in `ChatbotTrainer` / `IncrementalTrainer` constructs the URL from the base URL + endpoint. With this change, the base URL for a session becomes:

```
{METRICS_SERVER_URL}/api/sessions/{METRICS_SESSION_KEY}
```

All existing push calls (e.g., `.Post("/api/epoch/end", ...)`) are prefixed with this base path. No other logic changes in the trainers.

### 5.3 In-Process Singleton

For users who use `GlobalMetricsService::instance()` directly, the singleton becomes a `MetricsSessionRegistry` rather than a single `TrainingMetricsService`. The existing convenience API (`instance().start_session(id, ...)`) is preserved by having the singleton proxy calls through to the `"0-default"` session slot for backwards compatibility.

---

## 6. Configuration Changes

New `config.conf` keys:

| Key | Default | Description |
|-----|---------|-------------|
| `METRICS_SESSION_KEY` | *(derived)* | Key for this trainer's session slot |
| `METRICS_MAX_LIVE_SESSIONS` | `16` | Max concurrent sessions in registry |
| `METRICS_COMPLETED_TTL_SECONDS` | `3600` | How long to retain completed session data in memory |
| `METRICS_SWEEP_INTERVAL_SECONDS` | `60` | How often the registry background thread evicts stale sessions |

Existing keys are unchanged.

---

## 7. Dashboard (Tizen App) Impact

The Tizen metrics app currently polls `/api/metrics/current` and `/api/session/status`. With the backwards-compat aliases this continues to work for `"0-default"`.

A follow-up UI change should:

1. Poll `GET /api/sessions` on page load and on a 10-second timer.
2. Render a session selector (dropdown or tab strip) populated from the `sessions` array.
3. Substitute the selected `key` into all existing API calls (e.g., `/api/sessions/{key}/metrics/current`).
4. Show an aggregate tile from `GET /api/metrics/aggregate` at the top of the dashboard.

That UI work is outside the scope of this proposal.

---

## 8. Implementation Plan

| Step | File(s) affected | Description |
|------|-----------------|-------------|
| 1 | `src/TrainingMetricsService.hpp/.cpp` | Accept derived file paths via a per-session `Config` copy at construction. No other functional change. |
| 2 | `src/MetricsSessionRegistry.hpp` *(new)* | Implement the registry: session map, reader-writer lock, `create_or_get_session`, `list_sessions`, eviction. |
| 3 | `src/TrainingMetricsAPI.hpp/.cpp` | Replace single `TrainingMetricsService*` member with `MetricsSessionRegistry*`. Rewrite route registration to use `/{key}/` prefix. Add `/api/sessions` and `/api/metrics/aggregate` routes. Add flat-route backwards-compat aliases. |
| 4 | `src/TrainingMetricsAPIServer.cpp` | Construct `MetricsSessionRegistry` from `Config`; pass to `TrainingMetricsAPI`. Remove construction of a single `TrainingMetricsService`. |
| 5 | `src/Config.hpp/.cpp` | Add new config keys with defaults. |
| 6 | `src/ChatbotTrainer.cpp` / `src/IncrementalTrainer.cpp` | Read `METRICS_SESSION_KEY` (or derive it). Prefix HTTP push base URL with `/api/sessions/{key}`. |
| 7 | `src/GlobalMetricsService.hpp` *(or equivalent)* | Change singleton type to `MetricsSessionRegistry`; add proxy methods for `"0-default"` slot for backwards compatibility. |
| 8 | `tests/training_metrics_service_test.cpp` | Add multi-instance tests: concurrent sessions, eviction, 409 on duplicate start, aggregation endpoint. |
| 9 | `config.conf` | Add new keys with defaults. |
| 10 | `docs/development/api/TrainingMetricsAPI.md` | Update endpoint reference. |

Steps 1–2 can be done in parallel; steps 3–5 depend on step 2; steps 6–7 depend on step 3.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Key collisions when multiple trainers derive the same key automatically | Low (hostname + PID differentiates) | Log a warning and return HTTP 409; trainer retries with a PID-based suffix |
| Registry memory growth if trainers forget to call `POST .../end` | Medium | Background sweep evicts sessions with no update for `METRICS_COMPLETED_TTL_SECONDS` |
| Backwards-compat aliases masking bugs in new routes during development | Medium | Integration tests run against both the new and legacy URL forms |
| Per-session file path conflicts if two sessions share the same key across restarts | Low | Key uniqueness enforced at registration time; stale completed-session files are suffixed with a timestamp on re-use |
| Increased lock contention on the registry under high session churn | Low | `shared_mutex` reader lock for hot read path; exclusive lock only on session create/evict |

---

## 10. Open Questions

1. **Persistent registry across server restarts:** Should the registry restore all in-progress sessions from their `_metrics_summary.json` files on startup, similar to how the current single-session service calls `restore_from_summary()`? This would require scanning `training_sessions/` for files matching `*_metrics_summary.json` and inferring session keys from filenames.

2. **Authentication / isolation:** If multiple users or automated jobs share one metrics server, should session keys be scoped with a token to prevent one trainer from reading or corrupting another's session? Out of scope for now but worth noting.

3. **`GET /api/metrics/prometheus` aggregation:** The current per-session Prometheus output uses flat metric names (e.g., `training_loss`). In a multi-session world, these need a `session_key` label: `training_loss{session="42-gpu0"} 2.34`. The aggregate Prometheus endpoint should emit all sessions with this label.

4. **Cap behaviour:** Should hitting `METRICS_MAX_LIVE_SESSIONS` block the trainer (HTTP 503) or silently drop per-sample updates while still accepting epoch-level data? Blocking with a clear error is safer during development; could be made configurable later.
