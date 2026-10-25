# `SQLiteMetricsDatabase` — Source File Reference

- **Files:** [`src/SQLiteMetricsDatabase.hpp`](../../../../src/SQLiteMetricsDatabase.hpp) (64 lines), [`src/SQLiteMetricsDatabase.cpp`](../../../../src/SQLiteMetricsDatabase.cpp) (875 lines; also defines `MetricsDatabaseFactory::create()`)
- **Namespace:** global
- **Built into:** `adai_core`; the default store of `metrics_api_server` (`--storage-backend sqlite`, `training_sessions/metrics.db` unless `--db-path`)
- **Implements:** [IMetricsDatabase](MetricsDatabase.md) (the shared contract, retention and query gaps are documented there)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0` (hpp) / `1.0.2` (cpp), `@adai-reviewed: 2026-09-10` / `2026-09-23`
- **Tests:** [`tests/MetricsDatabaseTest.cpp`](../../../../tests/MetricsDatabaseTest.cpp) (30 tests, `metricsDatabaseTests`; all pass)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real class on a scratch database.

---

## 1. What this file is

One SQLite connection behind one mutex, with prepared statements for the hot writes.

| Part | Behaviour |
|---|---|
| Constructor `(db_path)` | `sqlite3_open()` (throws if it fails, e.g. a missing parent directory); `journal_mode = WAL`, `synchronous = NORMAL` (errors ignored); schema bootstrap; statement preparation |
| Schema | `sessions` (key primary key), `metrics_history`, `gradient_variance_history`, `generation_quality`, `abnormal_samples` (child tables `REFERENCES sessions(key)`, not enforced: SQLite's `foreign_keys` pragma is off), `schema_version` (row 1 written, never read) |
| Migrations | Four independent `PRAGMA table_info` checks add `final_loss`/`final_validation_loss`, the TD-013 diagnostics, the TD-178 LeJEPA losses and the LeJEPA "advanced" columns to older databases; `ALTER TABLE` errors are ignored |
| Writes | `upsert_session()` (`INSERT … ON CONFLICT DO UPDATE`), `mark_session_ended()`, `insert_metrics_record()`, `insert_gradient_variance_sample()`, `insert_abnormal_sample()`, `insert_generation_quality()`; each is its own autocommit transaction; any error is logged and **thrown** (`check_sqlite()`) |
| `archive_session(key, archived_key)` | One transaction: copy the session row under the new key, repoint the four child tables, delete the old row; rolls back and rethrows on error |
| Reads | `query_history(key, from, to, limit)` (built SQL, ascending; TD-441), `list_sessions(filter)` (newest first), `get_session(key)` (prepared). `best_validation_loss` reads NULL as `FLT_MAX` (the TD-065 fix) |
| Timestamps | Written `YYYY-MM-DDTHH:MM:SS.mmmZ`; parsed with TD-477's helper |
| `MetricsDatabaseFactory::create(backend, path, url, pool)` | Strips `+file`; `sqlite` → this class; `postgres` → the Postgres backend when built; `file`/empty → none; anything else warns and returns none (TD-439) |

---

## 2. Problems

### A found `get_session()` keeps a read transaction open, so the WAL never checkpoints (verified)

`get_session()` steps its prepared statement once and returns on `SQLITE_ROW` **without resetting it**. The statement stays active, holding a read
snapshot until the next `get_session()` call resets it. In WAL mode an open reader stops checkpoints from moving past its snapshot, so every later
write accumulates in the `-wal` file. 20,000 `insert_metrics_record()` calls produced a **254 MB WAL with an empty 4 KB main file** after one found
`get_session()`, against a 4 MB WAL and a 2.1 MB database without it. All the data then lives only in the WAL, and reads have to scan it.

In production, `TrainingMetricsService::restore_from_summary()` calls `get_session()` when a service is constructed. When the key already exists (the
server's `0-default` session after a restart, or a trainer re-registering its key after `metrics_api_server` restarts mid-pass) the statement stays
active until some later service is constructed with a new key, and every row written meanwhile (including the one-per-optimizer-step gradient-variance
rows) grows the WAL. Resetting the statement after reading the row fixes it.

### A NaN loss is stored as NULL and reads back as 0 (verified)

SQLite stores a bound NaN as NULL, and `sqlite3_column_double()` returns 0.0 for NULL. A history row with NaN `loss` and `validation_loss` read back as
**0 and 0**, and a session's NaN `final_loss` as 0, so a diverged run looks like a perfect one in `db-history`, comparisons and exports. (Infinity
round-trips correctly; `best_validation_loss` is guarded and reads back as the `FLT_MAX` "no data" sentinel.) The other float columns lack the guard
too, and `final_loss`/`final_validation_loss` NULLs in pre-migration rows also read as 0.

### No busy timeout (verified)

No `sqlite3_busy_timeout()` is set. With a second connection holding a write transaction (an operator's `sqlite3` shell, a backup, a second server
on the same file), an insert threw `database is locked` after **0.3 ms** instead of waiting. The callers catch and log it, so the row is simply lost
(the same gap TD-304 covers for `DaemonConfigStore`).

### Smaller items (by inspection)

- **Every write is its own transaction**, fsynced per `synchronous = NORMAL` on WAL checkpoints; a burst of per-step gradient-variance rows is many tiny
  transactions.
- **PRAGMA and migration errors are ignored**, so a failed `ALTER TABLE` surfaces later as a prepare error with a less obvious cause.
- **`schema_version` is written but never read**; migrations rely on column probes.
- **The default path's directory isn't created**: `training_sessions/metrics.db` fails to open if `training_sessions/` doesn't exist (verified with a
  missing directory: `unable to open database file`).

---

## 3. Tests

`MetricsDatabaseTest.cpp` (30, SQLite): schema bootstrap, WAL mode, the four column migrations, upserts, final-loss and LeJEPA column round trips, the
`FLT_MAX` sentinel for NULL `best_validation_loss`, time-range and limit queries, list filters, gradient-variance, abnormal-sample and
generation-quality inserts, archiving, and the factory.

**Not covered:** NaN values, WAL growth after `get_session()`, lock contention.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `get_session()` leaves its statement active after a found row, blocking WAL checkpoints (§2) | Unbounded WAL (254 MB vs 4 MB for 20k rows), empty main file, slower reads | **Verified** (production trigger by inspection) | [TD-509](../../guides/TECHNICAL_DEBT.md#td-509-sqlitemetricsdatabaseget_session-blocks-wal-checkpoints) |
| NaN stored as NULL and read as 0 for every unguarded float column (§2) | Diverged runs show loss 0 | **Verified** | [TD-510](../../guides/TECHNICAL_DEBT.md#td-510-sqlite-metrics-store-reads-nan-losses-back-as-0) |
| No busy timeout (§2) | Lock contention drops rows immediately | **Verified** | [TD-511](../../guides/TECHNICAL_DEBT.md#td-511-sqlitemetricsdatabase-busy-timeout-transactions-and-setup-gaps) |
| Per-row transactions; ignored PRAGMA/migration errors; unused `schema_version`; default directory not created (§2) | Robustness | By inspection (directory verified) | [TD-511](../../guides/TECHNICAL_DEBT.md#td-511-sqlitemetricsdatabase-busy-timeout-transactions-and-setup-gaps) |
| No retention; limit keeps oldest rows; timestamp parsing; unknown backend accepted | Already tracked | — | TD-440, TD-441, TD-477, TD-439 |

---

## Addendum (October 25, 2026): WAL checkpoints and NaN reads (TD-509, TD-510)

- **TD-509:** `get_session()` resets `stmt_get_session_` on every return path (a scope guard). A found row used to leave a read
  snapshot open that stopped WAL checkpoints: 86 MB of WAL after 4,000 rows in the new test, 254 MB after 20,000 when the bug was filed.
- **TD-510:** every nullable float column reads through `sqlite_nan_float()`. NULL, which is how SQLite stores a bound NaN and what
  pre-migration rows hold, comes back as NaN, not 0.

Tests: `FoundGetSessionDoesNotBlockWalCheckpoints`, `NanLossesReadBackAsNanNotZero`.

