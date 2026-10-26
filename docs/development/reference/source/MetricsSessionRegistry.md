# `MetricsSessionRegistry` — Source File Reference

- **File:** [`src/MetricsSessionRegistry.hpp`](../../../../src/MetricsSessionRegistry.hpp) (header-only, 515 lines)
- **Namespace:** global (`MetricsSessionRegistry`, `MetricsSessionSummary`, `SessionStartResult`)
- **Used by:** `metrics_api_server` only: created in `TrainingMetricsAPIServer.cpp`'s `main()`, shared with `TrainingMetricsAPI`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/metrics_session_registry_test.cpp`](../../../../tests/metrics_session_registry_test.cpp) (24 tests)
- **Related:** [MetricsDatabase.md](MetricsDatabase.md) (the shared DB it owns), [MetricsApiServerArgs.md](MetricsApiServerArgs.md) (where its limits come from), [TRAINING_METRICS_API.md](../../TRAINING_METRICS_API.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the
> real registry and `TrainingMetricsService`.

---

## 1. What this file is

The metrics server's map of **session key → `TrainingMetricsService`**, one entry per training run a
trainer has registered (`"12-ai-machi"`, `"lejepa-<model>"`, the legacy `"0-default"`). It enforces a
cap on entries, retires finished sessions after a TTL, owns the shared metrics database, and keeps a
retired session's history separate from a later session that reuses its key by **archiving** it.

```text
POST /api/sessions/{key}/start ──► start_session_or_conflict(key)
                                     ├─ live session holds key  → 409
                                     ├─ completed/stale holds it → archive it, create new
                                     ├─ size ≥ max_live_sessions → 503 metrics_server_full
                                     └─ else create TrainingMetricsService(config_for_session(key))
every other /api/sessions/{key}/… ─► get_session(key)            (404 if absent)
sweep thread (every sweep_interval) ► evict completed/stale sessions idle ≥ completed_ttl → archive
```

---

## 2. API

| Member | Behaviour |
|---|---|
| Constructor `(base_config, max_live = 16, completed_ttl = 3600, sweep_interval = 60, backend, db_path, db_url, pool)` | Creates the DB through `MetricsDatabaseFactory` (a failure or unknown backend is logged and the registry runs without one); starts the sweep thread **only if `sweep_interval > 0`** |
| `start_session_or_conflict(key)` | The only path that creates sessions from HTTP. Under one exclusive lock: evict expired sessions; if `key` exists, archive and replace it when completed or stale, **refuse** (`conflict`) when genuinely live, or return it unchanged when it never started; otherwise create, unless `size() ≥ max_live_sessions` (`service == nullptr`, which the API turns into 503) |
| `create_or_get_session(key)` | Same without the conflict check; used by `main()` for `"0-default"` (the API's `resolve_session_service(…, create_if_missing)` is never called with `true`) |
| `get_session(key)` | Shared lock; `nullopt` if absent, **logging a warning plus one line per live key** |
| `list_sessions()` | Live sessions from each service's snapshot, plus completed sessions from the DB that are no longer in memory (skipping `_archived_` keys) |
| `evict_completed_sessions(max_age)`, `size()`, `get_database()` | — |
| Admin setters/getters | `set_max_live_sessions`, `set_completed_ttl_seconds`, `set_sweep_interval_seconds` (wakes the sweep thread), `update_metrics_config(mutator)` (applies to the base config and every live session) |

### Session lifecycle rules

- **Started** = the snapshot has a session ID, epoch or sample count (`has_started()`).
- **Replaceable** = started and either not training (completed) or training but **stale** (no push within
  `METRICS_STALENESS_THRESHOLD_SECONDS`).
- **Evictable** (sweep, or on every start) = replaceable and idle for at least `completed_ttl` seconds.
- **Archiving** renames the session's DB rows and its per-session files (`<key>_metrics.jsonl`, …, not
  `"0-default"`'s shared legacy files) to `<key>_archived_<ms>_<n>`, then drops it from the map.

---

## 3. Problems (verified)

### Finished sessions keep their slots for an hour, so new runs get 503

`max_live_sessions` counts **every** entry in the map, including sessions that completed normally or went
stale. A completed session leaves the map only when it has been idle for `completed_ttl_seconds` (3,600 in
`config.metrics.conf`), or when its own key is reused. With a cap of 2, two sessions that started and ended
left the registry full: a third key's `start_session_or_conflict()` returned no service, and the API answers
`503 metrics_server_full`. Nothing evicts a finished session to make room.

That matters for the always-on trainer. Every chatbot pass registers a **new** key (the session number
increments), so with the shipped cap of 16, more than 16 passes within an hour fills the server with
finished sessions. `trainer_service` launches the next pass immediately whenever the last one did work,
and crashed passes' stale sessions count too. From then on, `IncrementalTrainer`'s `start_session()` gets
503 and the pass logs "training will proceed WITHOUT metrics reporting", until an hour after the oldest
finished session. Making room by evicting the oldest finished session before refusing would fix it, so
the cap applies only to live sessions as its name says.

### Setting the sweep interval at runtime misbehaves

`PUT /admin/config` can change `sweep_interval_seconds` (with no range check, TD-306):
- **To 0 or less while the thread runs**, `wait_for(0)` returns immediately and the loop spins: the
  probe used **0.51 CPU-seconds in 2 seconds** of wall time with an empty registry, taking the
  registry's exclusive lock on every iteration (each pass is an eviction scan).
- **From 0 to a positive value**, nothing happens: the thread was never started, so with a 1-second
  interval and TTL, a finished session was still there after 3 seconds, while `GET /admin/config` reported
  the new interval. (`--sweep-interval-seconds 0` at startup is TD-438.)

---

## 4. Smaller items (by inspection)

- **Log noise on misses.** `get_session()` logs a warning plus one line per live key for every unknown
  key. `handle_prometheus_aggregate()` calls it for each session from `list_sessions()`, which includes
  the DB-only completed sessions, so every Prometheus aggregate request logs `(DB sessions) × (1 + live
  sessions)` warning lines. A 404 for any completed session's key does the same.
- **I/O under the registry lock.** Eviction archives (a DB transaction plus file renames) run under the
  exclusive lock, on every start and every sweep, and `list_sessions()` queries the DB under the shared
  lock, so slow storage stalls every request.
- **Late writes after archiving.** An evicted service can still be referenced by an in-flight request; if
  it persists after its files were renamed, it recreates them under the original key, which a new session
  with that key then shares.

---

## 5. Tests

`metrics_session_registry_test.cpp` (24 tests): per-session paths and the `"0-default"` legacy paths;
`start_session_or_conflict()` refusing a live session, reclaiming a stale one and creating a fresh one;
replacement and eviction archiving in the DB and on disk; the archived-key filter; TTL eviction (manual and
by the sweep thread); `sweep_interval = 0` not starting the thread; labels and config snapshots; `get_session()`;
and concurrency (creation, reads and writes, TTL changes during a sweep).

**Not covered:** the cap with finished sessions, runtime interval changes, log volume.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `max_live_sessions` counts finished and stale sessions until their TTL (1 h); nothing evicts one to make room (§3) | More than 16 trainer passes an hour make later passes run without metrics (503) | **Verified** | [TD-445](../../guides/TECHNICAL_DEBT.md#td-445-finished-metrics-sessions-hold-max_live_sessions-slots-for-an-hour) |
| Runtime `sweep_interval` ≤ 0 spins the sweep thread; 0 → positive never starts it (§3) | A quarter of a core plus constant exclusive locking; or no eviction despite the reported setting | **Verified** | [TD-446](../../guides/TECHNICAL_DEBT.md#td-446-metricssessionregistry-sweep-interval-changes-miss-logging-and-lock-held-io) |
| `get_session()` miss logging (amplified by Prometheus aggregation); DB and file I/O under the registry lock; late writes after archiving (§4) | Log spam; request stalls on slow storage; rare file sharing between runs | By inspection | [TD-446](../../guides/TECHNICAL_DEBT.md#td-446-metricssessionregistry-sweep-interval-changes-miss-logging-and-lock-held-io) |
| No range checks on admin settings; `--sweep-interval-seconds 0`; unknown backend | Already tracked | — | TD-306, TD-438, TD-439 |

---

## Addendum (October 25, 2026): finished sessions make room (TD-445)

`make_room_locked()`, called by `create_or_get_session()` and `start_session_or_conflict()`, archives finished or stale sessions, least
recently updated first, while the registry is at `max_live_sessions`. The history is kept in the DB, as on key reuse. Live sessions and
never-started placeholders are never evicted. A new key is refused (`nullptr` / 503) only when every slot is live. Tests:
`FinishedSessionsDontBlockANewKey`, `LiveSessionsStillFillTheRegistry`, `NeverStartedPlaceholderIsNotEvicted`.


---

## Addendum (October 25, 2026): age retention (TD-440)

`apply_retention(now)` deletes DB sessions not updated for `retention_days` (via `IMetricsDatabase::delete_sessions_updated_before()`),
and archived files (names carrying `_archived_`, rotated backups included) in the metrics directories last written before the same
cutoff. Files that aren't archived are left alone. The sweep thread runs it at most once an hour. `retention_days` lives in the base
`MetricsServiceConfig`, so `PUT /admin/config` changes it with `update_metrics_config()`. Archiving a session now renames its rotated
JSONL backups (`<file>.1` … `.N`) with it. Tests: `RetentionDeletesOldSessionsAndArchivedFiles`,
`EvictionArchivesRotatedMetricsBackups`.

## Addendum (October 25, 2026): sweep interval, archiving outside the lock (TD-446)

- **Sweep thread.** It's started by every constructor. An interval ≤ 0 makes it idle (it waits until the interval changes), and any
  change wakes it to start a fresh wait with the new value.
- **Archiving.** A leaving session is detached under the exclusive lock (`detach_locked()`, key marked in `archiving_keys_`) and
  archived afterwards without the lock (`archive_detached()`): `TrainingMetricsService::retire()`, `archive_session()`, file renames.
  Creating or starting that key waits on `archive_cv_` until it's done. `create_or_get_session()`/`start_session_or_conflict()` loop
  once when they had to archive the key's own previous session.
- **`get_session()`.** A miss logs at debug. `list_sessions()` queries the DB after releasing the shared lock.

Tests: `DisabledSweepIntervalDoesNotSpin`, `EnablingSweepLaterStartsSweeping`, `ShorterSweepIntervalAppliesImmediately`,
`EvictedServiceStopsWriting`, `ConcurrentKeyReuseWithArchivingOutsideTheLock`; TSan-clean.

## Addendum (October 25, 2026): starts claim their session

`start_session_or_conflict()` marks the session it hands out as claimed (`SessionEntry::claimed`). Another start for the key gets
the conflict until the first has started, at which point the session is live. Before, two concurrent starts could share one
never-started placeholder. A placeholder made by `create_or_get_session()` (e.g. by a GET) is still handed to the first start.
Test: `SecondStartBeforeTheFirstHasRunIsRefused`.
