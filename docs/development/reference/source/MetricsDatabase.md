# `MetricsDatabase` — Source File Reference

- **File:** [`src/MetricsDatabase.hpp`](../../../../src/MetricsDatabase.hpp) (header-only interface, 92 lines)
- **Namespace:** global (`SessionRecord`, `IMetricsDatabase`, `MetricsDatabaseFactory`)
- **Implementations:** `SQLiteMetricsDatabase` ([`src/SQLiteMetricsDatabase.{hpp,cpp}`](../../../../src/SQLiteMetricsDatabase.cpp), which also defines `MetricsDatabaseFactory::create()`), `PostgresMetricsDatabase` ([`src/PostgresMetricsDatabase.{hpp,cpp}`](../../../../src/PostgresMetricsDatabase.cpp), only with `ADAI_ENABLE_POSTGRES`)
- **Used by:** `metrics_api_server` only: `MetricsSessionRegistry` owns the one instance and shares it with every `TrainingMetricsService` and with `TrainingMetricsAPI`'s DB-backed endpoints
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/MetricsDatabaseTest.cpp`](../../../../tests/MetricsDatabaseTest.cpp) (30 tests, SQLite), [`tests/postgres_metrics_database_test.cpp`](../../../../tests/postgres_metrics_database_test.cpp) (18, needs a live server)
- **User docs:** [TRAINING_METRICS_API.md](../../TRAINING_METRICS_API.md) (`db-history`, `--storage-backend`)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. The interface's contract is traced through both implementations and every
> caller; the backends themselves get their own references.

---

## 1. What this file is

The storage interface behind the metrics server's durable history (TD-020). In-memory state lives in
each `TrainingMetricsService`; this database is the copy that survives restarts and the stale-session
sweep.

```text
trainer ──MetricsPushClient──► metrics_api_server
                                 ├─ MetricsSessionRegistry  ── owns ──► IMetricsDatabase (one, shared)
                                 │     └─ TrainingMetricsService per session ── writes ──┘
                                 └─ TrainingMetricsAPI  ── reads (db-history, compare, export, sessions)
```

### `SessionRecord`

One row per session key: `key`, `session_id`, `label`, `config_json`, `is_training`, `created_at`,
`ended_at` (optional), `last_update_at`, `total_epochs`, `total_samples`, `best_validation_loss` (starts
at `FLT_MAX`, meaning "none yet"), `best_epoch`, and `final_loss`/`final_validation_loss` (the last
persisted values, so archived sessions still show a real final loss).

---

## 2. The interface

| Method | Caller | Contract (as implemented by both backends) |
|---|---|---|
| `upsert_session(rec)` | `TrainingMetricsService` on each persist | Insert or replace by `key` |
| `mark_session_ended(key)` | `TrainingMetricsService` on `end_session` | Sets `ended_at`, `is_training = false` |
| `archive_session(key, archived_key)` | `MetricsSessionRegistry`'s sweep, before a stale key is reused | Renames the session and all its rows in one transaction; no-op if `key` doesn't exist |
| `insert_metrics_record(key, rec)` | `TrainingMetricsService`, every `persist_every_samples` | One `metrics_history` row |
| `insert_gradient_variance_sample(key, step, epoch, value)` | `TrainingMetricsService::update_advanced_epoch_metrics()`, unthrottled | One row per call (the trainer reports every optimizer step). `step` is a per-`TrainingMetricsService` counter |
| `insert_abnormal_sample(key, sample)` | `TrainingMetricsService` | One `abnormal_samples` row |
| `insert_generation_quality(key, epoch, score)` | `TrainingMetricsService` | One `generation_quality` row |
| `query_history(key, from, to, limit)` | `TrainingMetricsAPI` (`db-history`, compare, export) | `metrics_history` rows in `recorded_at` order, **ascending**; `limit = 0` means all |
| `list_sessions(is_training)` | Registry startup recovery (`false`), `GET /api/sessions?...` | Optional filter |
| `get_session(key)` | `TrainingMetricsService` (restoring best loss) | `nullopt` if absent |

Both backends serialize every call with one mutex (SQLite) or a mutex-guarded connection pool (Postgres),
and report errors by **throwing**. Every caller catches: `TrainingMetricsService` and the registry wrap
each call and log, and `TrainingMetricsAPI`'s route handlers catch at the route level, so a database
failure never reaches a trainer.

### `MetricsDatabaseFactory::create(backend, db_path, db_url, pool_size)`

Strips a `+file` suffix (the dual write to JSONL is the caller's decision), then: `sqlite` →
`SQLiteMetricsDatabase(db_path or training_sessions/metrics.db)`; `postgres` → the Postgres backend **only
when compiled with `ADAI_ENABLE_POSTGRES`**; `file` or empty → no database. Anything else, including
`postgres` in a build without it, logs a warning and returns `nullptr`, so the server runs file-only
(TD-439). It's declared here but **defined in `SQLiteMetricsDatabase.cpp`**.

---

## 3. Problems

### Nothing limits the database's growth (verified by search)

The interface has no delete, prune or retention operation, and neither backend deletes rows except while
renaming them in `archive_session()`. `max_records_on_disk` (default 100,000; the `--max-disk-records` flag,
`config.metrics.conf`, and admin-mutable through `PUT /admin/config`) is **read by no persistence code at
all**: a search of `src/` finds it only in the config structs, the CLI/admin plumbing and the
`GET /admin/config` output. Only `max_records_in_memory` is enforced, on the in-memory ring buffer. So
the metrics DB (and the JSONL file) grow for as long as the server runs. The fastest-growing table is
`gradient_variance_history`, at one row per optimizer step per training session. The setting's
documentation in TRAINING_METRICS_API.md ("Max records on disk") describes a limit that doesn't exist.

### Two tables are written but never read (verified by search)

`gradient_variance_history` and `generation_quality` have insert methods and no query method. No C++
code reads them, and neither do the repository's tools; `tools/abnormal_review/` reads only
`abnormal_samples`, directly through SQLite. The live dashboards get these values from the in-memory
snapshot, so the per-step rows are pure write cost and disk growth until something queries them.

### Smaller items (by inspection)

- **`step` restarts at 0** for a session key whenever its `TrainingMetricsService` is recreated (server
  restart, or a session restored after a sweep), so `gradient_variance_history` gets duplicate
  `(session_key, step)` pairs, and its own comment calls the counter "monotonic".
- **`limit` keeps the oldest rows.** `query_history()` sorts ascending, then applies `LIMIT`, so
  `db-history?limit=500` returns the first 500 records of a run, not the latest. TRAINING_METRICS_API.md
  just says "maximum records".
- **Unbounded reads.** Compare and export call `query_history()` with no limit, materializing a
  session's whole history in memory per request; there's no paging in the interface.
- **The contract is undocumented**: thread safety, throwing on error, ascending order, `limit = 0`,
  and the `FLT_MAX` sentinel are only discoverable from the implementations.
- **Factory placement.** `MetricsDatabaseFactory::create()` lives in `SQLiteMetricsDatabase.cpp` and
  reaches the Postgres backend through a block-scope `extern` declaration.

---

## 4. Tests

`MetricsDatabaseTest.cpp` (30 tests, SQLite): schema bootstrap, WAL mode, the four column migrations,
upsert/list with filters and the `FLT_MAX` sentinel, `mark_session_ended`, archiving (including the no-op
case and child rows), every insert (the abnormal-sample and generation-quality round trips read back with
raw SQL), `query_history` ranges and limits, the dual-write and restore paths through
`TrainingMetricsService`, and the factory. `postgres_metrics_database_test.cpp` (18) runs the same contract against
a live Postgres and skips without one.

**Not covered:** retention (there is none), step-counter restarts, `limit` returning the oldest rows.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No retention anywhere; `max_records_on_disk` is configurable but read by nothing (§3) | The metrics DB and JSONL grow without bound; a documented limit doesn't exist | **Verified** (search) | [TD-440](../../guides/TECHNICAL_DEBT.md#td-440-metrics-storage-has-no-retention-and-max_records_on_disk-is-never-applied) |
| `gradient_variance_history` and `generation_quality` are write-only; per-step inserts (§3) | One write per optimizer step per session for data nothing reads | **Verified** (search) | [TD-440](../../guides/TECHNICAL_DEBT.md#td-440-metrics-storage-has-no-retention-and-max_records_on_disk-is-never-applied) |
| `step` restarts per service instance; `limit` keeps the oldest rows; unbounded compare/export reads; undocumented contract; factory in the SQLite file (§3) | Duplicate step numbers; a "latest N" query returns the earliest N | By inspection | [TD-441](../../guides/TECHNICAL_DEBT.md#td-441-imetricsdatabase-contract-gaps-step-counter-limit-order-unbounded-reads) |
| Unknown backend name falls back to file-only | Already tracked | — | TD-439 |

---

## Addendum (October 25, 2026): retention and readers (TD-440)

`IMetricsDatabase` (1.1.0) has four new methods, implemented by both backends:
- `query_gradient_variance(key, limit)` and `query_generation_quality(key)` read the two tables that used to be write-only. Rows come
  oldest first; `limit` keeps the newest N. They're served as `GET …/metrics/db-gradient-variance` and `…/db-generation-quality`.
- `cap_session_rows(key, max_rows)` keeps a session's `metrics_history` and `gradient_variance_history` at or below `max_rows`, using
  `plan_row_cap()`. Nothing happens until the cap is exceeded. Then each table is trimmed to 3/4 of the cap by deleting every second
  row among the oldest, or by deleting the oldest rows when more than half must go. Repeated trims make old history sparser while the
  start of the run survives. `TrainingMetricsService` calls it after every DB persist with `max_records_on_disk`.
- `delete_sessions_updated_before(cutoff)` deletes sessions whose `last_update_at` is older than the cutoff, and their rows in all four
  child tables, in one transaction with children first (Postgres enforces the foreign keys). The registry sweep calls it hourly with
  `retention_days`.

Tests: `CapSessionRowsThinsOldestAndKeepsNewest`, `DeleteSessionsUpdatedBeforeRemovesAllTheirRows`,
`GradientVarianceAndGenerationQualityReadBack`, `MetricsRowCapPlan.ThinsOldestOrDeletesOldest` (SQLite);
`RetentionAndReadersWork` (PostgreSQL 16).
