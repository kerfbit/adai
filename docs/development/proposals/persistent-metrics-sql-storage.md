# Proposal: Persistent Metrics Storage via SQL Database

**Status:** Draft  
**Date:** 2026-05-31  
**Area:** Training infrastructure / Metrics  

---

## 1. Problem Statement

The current `TrainingMetricsService` persists data to flat files:

- **`{key}_metrics.jsonl`** — append-only JSON Lines, one `PersistentMetricsRecord` per line.
- **`{key}_metrics_summary.json`** — latest `TrainingMetricsSnapshot` for fast state restore.
- **`{key}_abnormal_samples.json`** — flagged outlier samples.

This approach has several limitations:

1. **No query capability.** Retrieving records in a time range, or any subset of the history, requires reading the entire JSONL file and filtering in memory.
2. **Hard in-memory cap.** `METRICS_MAX_RECORDS_IN_MEMORY` (default: 10,000) limits how much history `GET /api/sessions/{key}/metrics/history` can return. Records beyond the cap are only on disk and cannot be served via the API without reading the full file.
3. **No cross-session analytics.** Comparing two sessions' loss curves requires the caller to read two separate files and join them externally.
4. **Concurrent read / write fragility.** Appending to a JSONL file while another process reads it can produce partial-line reads. SQLite's WAL mode eliminates this class of problem.
5. **Long-term retention is file-system dependent.** As training_sessions/ accumulates JSONL files there is no built-in pruning, indexing, or searchable history.

---

## 2. Goals

1. Introduce a SQL persistence backend (SQLite by default; PostgreSQL as a compile-time option) that replaces JSONL/JSON as the primary durable store.
2. Provide an `IMetricsDatabase` abstraction so both backends share the same interface and callers are database-agnostic.
3. Keep file output as a configurable secondary export (`sqlite+file` dual-write mode) so existing deployments can migrate safely.
4. Enable time-range history queries, cross-session metric comparison, and full unbounded history export via new REST endpoints.
5. Preserve all existing API routes and behaviour.

### Non-goals

- Replacing the in-memory `history_` ring buffer (it is kept for O(1) low-latency reads by the dashboard).
- Distributed SQL across multiple API server processes (one server per host remains the model).
- Changing the Tizen dashboard front-end beyond consuming the new query endpoints.
- Authentication or per-user session isolation.

---

## 3. Current Architecture Summary

```
TrainingMetricsService
├── TrainingMetricsSnapshot current_snapshot_
├── std::vector<PersistentMetricsRecord> history_   ← capped at 10,000 records
├── std::vector<AbnormalSample> abnormal_samples_
└── File handles
      {key}_metrics.jsonl          ← append-only history
      {key}_metrics_summary.json   ← latest snapshot for restart restore
      {key}_abnormal_samples.json

MetricsSessionRegistry
└── owns N × TrainingMetricsService instances
    (one per concurrent training session)
```

---

## 4. Proposed Architecture

### 4.1 Database Abstraction Interface

A new header `src/MetricsDatabase.hpp` defines:

```cpp
struct SessionRecord {
    std::string key;
    int         session_id     = 0;
    std::string label;
    std::string config_json;   // arbitrary JSON blob from POST .../start
    bool        is_training    = false;
    std::chrono::system_clock::time_point created_at;
    std::optional<std::chrono::system_clock::time_point> ended_at;
    std::chrono::system_clock::time_point last_update_at;
    int   total_epochs  = 0;
    int   total_samples = 0;
    float best_validation_loss = std::numeric_limits<float>::max();
    int   best_epoch           = 0;
};

class IMetricsDatabase {
public:
    virtual ~IMetricsDatabase() = default;

    virtual void upsert_session(const SessionRecord& rec) = 0;
    virtual void mark_session_ended(const std::string& key) = 0;

    virtual void insert_metrics_record(
        const std::string& session_key,
        const PersistentMetricsRecord& rec) = 0;

    virtual void insert_abnormal_sample(
        const std::string& session_key,
        const AbnormalSample& sample) = 0;

    virtual void insert_generation_quality(
        const std::string& session_key,
        int epoch,
        const GenerationQualityScore& score) = 0;

    virtual std::vector<PersistentMetricsRecord> query_history(
        const std::string& session_key,
        std::optional<std::chrono::system_clock::time_point> from,
        std::optional<std::chrono::system_clock::time_point> to,
        int limit = 0) = 0;

    virtual std::vector<SessionRecord> list_sessions(
        std::optional<bool> is_training_filter = std::nullopt) = 0;

    virtual std::optional<SessionRecord> get_session(
        const std::string& key) = 0;
};
```

`MetricsDatabaseFactory::create(const Config&)` reads `METRICS_STORAGE_BACKEND` and
`METRICS_DB_PATH` / `METRICS_DB_URL` and returns a `std::unique_ptr<IMetricsDatabase>`.

### 4.2 SQLite Backend

`SQLiteMetricsDatabase : IMetricsDatabase` in `src/SQLiteMetricsDatabase.hpp/.cpp`:

- Uses the SQLite3 C API directly (no heavy C++ wrapper required).
- Opens the database with `PRAGMA journal_mode = WAL` so concurrent dashboard reads do not block training writes.
- Bootstraps the schema on first open (see §4.4); applies incremental migrations via the `schema_version` table.
- All INSERT/UPDATE statements use **prepared statements** bound at open time — avoids runtime SQL injection and improves write throughput.
- Thread model: one `sqlite3*` handle per `SQLiteMetricsDatabase` instance, protected by a `std::mutex`. The registry creates one instance shared across all sessions; prepared statements are re-bound per call under the lock.

### 4.3 PostgreSQL Backend (Optional)

`PostgresMetricsDatabase : IMetricsDatabase` in `src/PostgresMetricsDatabase.hpp/.cpp`:

- Uses libpq (PostgreSQL C client library).
- Connection pool of size `METRICS_DB_POOL_SIZE` (default: 4).
- Schema equivalent to SQLite but uses PostgreSQL-native types (`SERIAL`, `TIMESTAMPTZ`, `BOOLEAN`).
- Connection-loss is non-fatal: the backend retries up to three times with exponential backoff (100 ms, 400 ms, 1600 ms), then logs the error and continues. In-memory data and the file export (if dual-write is enabled) are unaffected.
- Enabled at compile time by the CMake option `ENABLE_POSTGRES_METRICS=ON` (default `OFF`); all PostgreSQL-specific code is guarded by `#ifdef ADAI_ENABLE_POSTGRES`.

### 4.4 Database Schema

Four tables. `generation_quality` is populated only for epochs where BLEU/ROUGE evaluation runs; all other tables are always written.

**`schema_version`** — migration tracking:

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL   -- ISO-8601 UTC
);
```

**`sessions`** — one row per training session:

```sql
CREATE TABLE IF NOT EXISTS sessions (
    key                  TEXT    PRIMARY KEY,
    session_id           INTEGER NOT NULL,
    label                TEXT    NOT NULL DEFAULT '',
    config_json          TEXT,
    is_training          INTEGER NOT NULL DEFAULT 1,  -- 0 = completed
    created_at           TEXT    NOT NULL,
    ended_at             TEXT,
    last_update_at       TEXT    NOT NULL,
    total_epochs         INTEGER NOT NULL DEFAULT 0,
    total_samples        INTEGER NOT NULL DEFAULT 0,
    best_validation_loss REAL,
    best_epoch           INTEGER
);
```

**`metrics_history`** — one row per `PersistentMetricsRecord`:

```sql
CREATE TABLE IF NOT EXISTS metrics_history (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key                 TEXT    NOT NULL REFERENCES sessions(key),
    recorded_at                 TEXT    NOT NULL,
    epoch                       INTEGER NOT NULL,
    sample                      INTEGER NOT NULL,
    loss                        REAL,
    validation_loss             REAL,
    learning_rate               REAL,
    gradient_norm               REAL,
    perplexity                  REAL,
    validation_perplexity       REAL,
    validation_accuracy         REAL,
    padding_efficiency          REAL,
    gradient_variance           REAL,
    compute_time_ratio          REAL,
    weight_update_ratio         REAL,
    activation_saturation_ratio REAL,
    attention_entropy           REAL
);

CREATE INDEX IF NOT EXISTS idx_metrics_history_session_time
    ON metrics_history (session_key, recorded_at);
```

**`generation_quality`** — BLEU/ROUGE scores, only written when generation quality evaluation is enabled. Keyed to the same `(session_key, epoch)` as the corresponding `metrics_history` row:

```sql
CREATE TABLE IF NOT EXISTS generation_quality (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT    NOT NULL REFERENCES sessions(key),
    recorded_at TEXT    NOT NULL,
    epoch       INTEGER NOT NULL,
    bleu1       REAL,
    bleu2       REAL,
    bleu4       REAL,
    rouge1      REAL,
    rouge2      REAL,
    rougeL      REAL
);

CREATE INDEX IF NOT EXISTS idx_generation_quality_session_epoch
    ON generation_quality (session_key, epoch);
```

Rows are inserted only when a `POST /api/sessions/{key}/metrics/generation-quality` call is received. Epochs with no generation quality evaluation produce no row, avoiding NULL-heavy columns in `metrics_history`.

**`abnormal_samples`** — one row per `AbnormalSample`:

```sql
CREATE TABLE IF NOT EXISTS abnormal_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT    NOT NULL REFERENCES sessions(key),
    epoch       INTEGER NOT NULL,
    sample_id   INTEGER NOT NULL,
    loss        REAL,
    grad_norm   REAL,
    reason      TEXT,
    input_text  TEXT,
    target_text TEXT,
    recorded_at TEXT    NOT NULL
);
```

### 4.5 Dual-Write Mode

When `METRICS_STORAGE_BACKEND` includes `+file` (e.g., `sqlite+file`), `TrainingMetricsService`:

1. Writes to the SQL database first.
2. Appends to the JSONL file second (existing path logic unchanged).
3. If the DB write fails, the file write still proceeds and the error is logged.

This lets operators validate the SQL output against the known-good JSONL files before switching to `sqlite`-only mode.

### 4.6 Changes to `TrainingMetricsService`

`TrainingMetricsService` gains an optional `IMetricsDatabase* db_` member (non-owning; null if no DB is configured):

| Method | Change |
|--------|--------|
| `persist_metrics()` | After appending to `history_` (and optionally JSONL), calls `db_->insert_metrics_record(key_, record)` |
| `persist_summary()` | Calls `db_->upsert_session(...)` with current snapshot fields; file write still occurs in dual-write mode |
| `restore_from_summary()` | Queries `db_->get_session(key_)` first; falls back to reading `_metrics_summary.json` if the row is absent — ensures existing deployments with only JSONL files migrate safely on first upgrade |
| Session end | Calls `db_->mark_session_ended(key_)` |

No changes to the `TrainingMetricsService` constructor signature; the `MetricsSessionRegistry` injects `db_` via a setter after construction.

### 4.7 Changes to `MetricsSessionRegistry`

- Owns a `std::unique_ptr<IMetricsDatabase> db_` created via `MetricsDatabaseFactory::create(config_)` at construction time.
- After calling `create_or_get_session(key)`, passes `db_.get()` to the new service via `service->set_database(db_.get())`.
- `list_sessions()` supplements the in-memory registry map with `db_->list_sessions(false)` (completed sessions) so recently evicted sessions are still visible via `GET /api/sessions`.

### 4.8 New REST Endpoints

All new endpoints are added to `TrainingMetricsAPI`:

| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions/{key}/metrics/history?from=<ISO8601>&to=<ISO8601>&limit=<n>` | Time-range filtered history served from DB; not limited to the 10,000-record in-memory cap |
| `GET /api/metrics/compare?keys=key1,key2&metric=<field>` | Returns parallel epoch arrays for the named metric from each session; useful for loss-curve overlays |
| `GET /api/sessions?status=completed&from=<ISO8601>` | Extends the existing sessions list with date-range and status filters backed by the DB |
| `GET /api/sessions/{key}/metrics/export?format=csv\|json` | Full unbounded history export from DB; streams the response to avoid buffering the entire result set in memory |

Existing routes are unchanged.

---

## 5. CMake Changes

```cmake
# SQLite3 — prefer system library; fall back to bundled amalgamation for Windows/MinGW
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND)
    message(STATUS "SQLite3 system library not found; using bundled amalgamation")
    add_library(sqlite3_amalgamation OBJECT external/sqlite3/sqlite3.c)
    target_compile_definitions(sqlite3_amalgamation PRIVATE SQLITE_THREADSAFE=1)
    add_library(SQLite::SQLite3 ALIAS sqlite3_amalgamation)
endif()

# PostgreSQL — optional
option(ENABLE_POSTGRES_METRICS
    "Build PostgreSQL metrics backend (requires libpq)" OFF)
if(ENABLE_POSTGRES_METRICS)
    find_package(PostgreSQL REQUIRED)
    add_compile_definitions(ADAI_ENABLE_POSTGRES)
endif()
```

The SQLite amalgamation (`external/sqlite3/sqlite3.c` + `sqlite3.h`) is copied from the official SQLite3 distribution and committed to `external/sqlite3/` for deterministic Windows builds.

---

## 6. Configuration Changes

New `config.conf` keys:

| Key | Default | Description |
|-----|---------|-------------|
| `METRICS_STORAGE_BACKEND` | `sqlite+file` | `file`, `sqlite`, `postgres`, `sqlite+file`, `postgres+file` |
| `METRICS_DB_PATH` | `training_sessions/metrics.db` | SQLite database file path |
| `METRICS_DB_URL` | *(empty)* | PostgreSQL connection string (used when backend includes `postgres`) |
| `METRICS_DB_POOL_SIZE` | `4` | PostgreSQL connection pool size |

Existing keys are unchanged.

---

## 7. Docker / Systemd Impact

**docker-compose.yml:** The `training_sessions/` bind-mount already covers `metrics.db` since the DB file defaults to that directory — no volume changes needed.

**Systemd:** `metrics-api-server.service` requires no changes. SQLite creates the DB file automatically on first start.

**PostgreSQL (optional):** If `METRICS_STORAGE_BACKEND=postgres`, operators must supply a running PostgreSQL instance and set `METRICS_DB_URL`. A companion `docker-compose.override.yml` example should be documented in `docs/operations/`.

---

## 8. Implementation Plan

| Step | File(s) affected | Description | Dependencies |
|------|-----------------|-------------|--------------|
| 1 | `src/MetricsDatabase.hpp` *(new)* | Define `SessionRecord`, `IMetricsDatabase`, `MetricsDatabaseFactory` | — |
| 2 | `src/SQLiteMetricsDatabase.hpp/.cpp` *(new)* | SQLite3 implementation: schema bootstrap, WAL mode, prepared statements | Step 1 |
| 3 | `src/PostgresMetricsDatabase.hpp/.cpp` *(new)* | libpq implementation with connection pool and retry logic | Step 1 |
| 4 | `src/TrainingMetricsService.hpp/.cpp` | Add `IMetricsDatabase* db_` member; modify `persist_metrics`, `persist_summary`, `restore_from_summary`, session end | Steps 1–2 |
| 5 | `src/MetricsSessionRegistry.hpp/.cpp` | Construct DB via factory; inject into sessions; supplement `list_sessions()` with DB query | Steps 1–4 |
| 6 | `src/TrainingMetricsAPI.hpp/.cpp` | Register four new endpoints (§4.8) | Step 5 |
| 7 | `src/TrainingMetricsAPIServer.cpp` | Pass DB-aware registry to API (no new logic — registry init covers this) | Step 5 |
| 8 | `adai/CMakeLists.txt` + `adai/src/CMakeLists.txt` | Add SQLite3/PostgreSQL dependency; add new source files | — |
| 9 | `external/sqlite3/` *(new)* | Commit SQLite amalgamation for Windows builds | — |
| 10 | `config.conf`, `config-remote.conf` | Add new keys with defaults | — |
| 11 | `tests/MetricsDatabaseTest.cpp` *(new)* | Unit tests for DB layer (see §9) | Steps 1–2 |
| 12 | `docs/development/TRAINING_METRICS_API.md` | Document new endpoints and `METRICS_STORAGE_BACKEND` semantics | Steps 6 |

Steps 1 and 8–9 can proceed in parallel. Step 3 (PostgreSQL) is independent of Steps 4–7 and can be deferred to a second iteration.

---

## 9. Testing

New test file: `tests/MetricsDatabaseTest.cpp`

| Test | What it verifies |
|------|-----------------|
| `SchemaBootstrap` | Opening a new DB creates all four tables and inserts `schema_version` row 1 |
| `WalModeEnabled` | `PRAGMA journal_mode` returns `wal` after open |
| `InsertAndQueryHistory` | Insert 50 records; `query_history` with no filter returns all 50 in timestamp order |
| `TimeRangeFilter` | Insert records spanning 10 minutes; `query_history` with `from`/`to` returns only the expected subset |
| `LimitClause` | `query_history` with `limit=5` returns exactly 5 most-recent records |
| `UpsertSession` | Calling `upsert_session` twice for the same key updates the row, does not duplicate it |
| `MarkSessionEnded` | `mark_session_ended` sets `is_training = 0` and `ended_at` without deleting the row |
| `AbnormalSampleRoundTrip` | Insert an `AbnormalSample`; read back via raw SQL and confirm all fields match |
| `DualWritePath` | With `METRICS_STORAGE_BACKEND=sqlite+file`, confirm both `metrics.db` and JSONL file are written after `persist_metrics()` |
| `RestoreFromDB` | Populate DB; create a new `TrainingMetricsService` pointing at the same DB; confirm `restore_from_summary()` loads the session |
| `RestoreFromFileFallback` | No DB row present; confirm `restore_from_summary()` falls back to reading the JSON summary file |
| `PostgresRoundTrip` *(skipped unless `ENABLE_POSTGRES_METRICS=ON`)* | Equivalent insert/query test against a live PostgreSQL instance |

---

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SQLite write lock contention under high sample-rate training | Low | WAL mode allows concurrent reads; single-writer model is fine for one training process per DB |
| PostgreSQL connection loss during training | Medium | Non-fatal retry with backoff; dual-write mode keeps file as fallback |
| Windows MinGW build failing to link system SQLite3 | Medium | Bundled amalgamation at `external/sqlite3/` is the explicit fallback path |
| Existing deployments losing history on upgrade | Low | `restore_from_summary()` reads JSON file if DB row is absent; operators can also run the optional JSONL import script (see §11.2) |
| DB file growing unbounded over long runs | Low | The schema supports future `DELETE FROM metrics_history WHERE recorded_at < ?` pruning; a `METRICS_DB_RETENTION_DAYS` key can be added without schema changes |
| SQL injection via session keys or labels | Low | All user-supplied values are bound via prepared statement parameters, never string-interpolated into SQL |

---

## 11. Open Questions

1. **JSONL import utility.** Existing deployments have months of JSONL history. A migration script (`scripts/import_metrics_jsonl.py` or a `--import-jsonl` flag on the server) would let operators backfill the new DB without losing that history. Worth scoping as a companion task.

2. **Prometheus label changes.** The existing `GET /metrics/prometheus` emits flat metric names (e.g., `training_loss`). Cross-session DB queries suggest adding a `session_key` label: `training_loss{session="42-gpu0"} 2.34`. This is consistent with the open question in the multi-instance proposal and could be addressed in the same pass.

3. **DB-backed session restore on server restart.** `MetricsSessionRegistry` currently only restores sessions from in-memory state. With the DB in place it becomes straightforward to re-populate the registry from `sessions WHERE is_training = 1` on startup. This would be a natural follow-on step after Phase 2 is stable.

4. **`METRICS_DB_RETENTION_DAYS` config key.** For very long training runs, `metrics_history` can grow large. A configurable pruning window (e.g., keep last 30 days) controlled by a background task in the registry would be a clean addition that requires no schema changes.
