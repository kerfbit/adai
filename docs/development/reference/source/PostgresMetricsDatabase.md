# `PostgresMetricsDatabase` — Source File Reference

- **Files:** [`src/PostgresMetricsDatabase.hpp`](../../../../src/PostgresMetricsDatabase.hpp), [`src/PostgresMetricsDatabase.cpp`](../../../../src/PostgresMetricsDatabase.cpp) (1,015 lines together)
- **Namespace:** global (`PostgresMetricsDatabase`, `create_postgres_metrics_database()`)
- **Built into:** `adai_core`, only with `-DENABLE_POSTGRES_METRICS=ON` (defines `ADAI_ENABLE_POSTGRES`, links libpq); **not built by default**. Selected by `metrics_api_server --storage-backend postgres`
- **Implements:** [IMetricsDatabase](MetricsDatabase.md)
- **Status tag:** `@adai-status: beta` ("TD-042 resolved — real test coverage added; still not built by default"), `@adai-version: 0.4.2`, `@adai-reviewed: 2026-09-23`
- **Tests:** [`tests/postgres_metrics_database_test.cpp`](../../../../tests/postgres_metrics_database_test.cpp) (18 tests against a scratch `initdb`/`pg_ctl` cluster; **all 18 pass** here with PostgreSQL 16)
- **Schema:** also in [`scripts/setup_postgres.sql`](../../../../scripts/setup_postgres.sql)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed against a real PostgreSQL 16 cluster, through the real
> `TrainingMetricsService` write path.

---

## 1. What this file is

The PostgreSQL implementation of the metrics store, with the same tables as the SQLite backend plus **foreign keys**: `metrics_history`,
`gradient_variance_history`, `generation_quality` and `abnormal_samples` each have `session_key REFERENCES sessions(key)`.

| Part | Behaviour |
|---|---|
| Constructor `(url, pool_size = 4)` | Opens `pool_size` connections with `PQconnectdb()`; each runs `SET TIME ZONE 'UTC'` (so `TIMESTAMPTZ` text round-trips, an earlier fix). Bootstraps the schema on connection 0 (`CREATE TABLE IF NOT EXISTS`, idempotent `ADD COLUMN IF NOT EXISTS` migrations) |
| Pool | `acquire_connection()` waits for a free slot and reconnects a dead one; a failed reconnect frees the slot (an earlier deadlock fix, tested) |
| `execute_with_retry(context, op)` | Up to 3 attempts, sleeping 100 and 400 ms between them; reconnects a connection that died during the attempt |
| Operations | Parameterized `PQexecParams()` for every statement. `upsert_session()` is `INSERT … ON CONFLICT (key) DO UPDATE`; `archive_session()` copies the session row under the new key, repoints the four child tables and deletes the old row in one transaction (so foreign keys hold throughout); floats are written with `max_digits10` (an earlier precision fix); nullable floats are read with `PQgetisnull()` (TD-065) |
| Timestamps | `format_timestamp()` writes `YYYY-MM-DDTHH:MM:SS.mmm+00`; `parse_timestamp()` reads Postgres's text form (TD-477's fraction bug applies: Postgres drops trailing zeros) |

---

## 2. Problems

### The first batch of every session's history is rejected by the foreign keys (verified)

`TrainingMetricsService` writes a session's **row** only in `persist_summary()`, which runs after `persist_metrics()` at each periodic persist
(every `persist_every_samples` samples or `persist_every_seconds`) and at `end_session()`. Child rows are written earlier: history records in
`persist_metrics()` (just before the first summary), and **gradient-variance samples on every optimizer step** from the start. Until the session
row exists, every one of those inserts **violates its foreign key**, is retried three times, and is then dropped.

Through the real service on PostgreSQL 16, with a fresh session key (persisting every 5 samples): the gradient-variance insert and the first persist's
history inserts all failed with `violates foreign key constraint … Key (session_key)=(fresh-key) is not present in table "sessions"` and were dropped after
three attempts. At the end the session row existed but **0 history rows and 0 gradient-variance rows** did. Each rejected insert also blocked its caller
(the metrics server's ingest handler) for about **0.5 s** of retry sleeps. In a normal run that means every session loses its first persist batch and
every per-step gradient-variance row until the first persist (100 steps or 30 s at the defaults), and the trainer's `/metrics/advanced` pushes stall
for half a second each meanwhile. A short session can lose its entire history. The SQLite backend doesn't enforce foreign keys, so it's unaffected, and
the backend's own tests always upsert the session first, so they don't exercise this.

Upserting the session row at `start_session()` (or writing it before any child row) fixes it; retrying a constraint violation is also pointless, since
it can't succeed.

### Smaller items (by inspection)

- **Schema bootstrap happens once.** It runs only if connection 0 succeeded at construction. If PostgreSQL is down when `metrics_api_server` starts,
  later reconnects work but the tables are never created on a fresh database, so every statement fails until a restart.
- **Network work under the pool mutex.** `acquire_connection()` reconnects (a blocking `PQconnectdb()`) while holding `pool_mutex_`, and
  `execute_with_retry()` does the same on failure, so one slow reconnect stalls every thread waiting for a connection. There's no `connect_timeout`
  or `statement_timeout` unless the URL supplies one.
- **Retried writes aren't idempotent.** An insert whose result was lost with the connection (committed but unacknowledged) is executed again, duplicating
  the row.
- **Non-retryable errors are retried** (constraint violations, syntax errors), each costing the full back-off.
- **`archive_session()` copies the row under `archived_key` with plain `INSERT`**; a pre-existing row with that key aborts the whole transaction (keys
  include a millisecond timestamp and counter, so collisions are unlikely).

---

## 3. Tests

`postgres_metrics_database_test.cpp` (18 tests, real cluster, skipped without `initdb`/`pg_ctl`): schema bootstrap, insert/query history, time ranges,
limits, upserts, final-loss round trips, the `FLT_MAX` sentinel for NULL, TD-013 and LeJEPA columns, gradient-variance inserts, `mark_session_ended()`,
archiving (row and child rows, missing key), abnormal samples, generation quality, list filters, the factory, and the pool not deadlocking on sustained
connection failure.

**Not covered:** child rows written before their session row (every test upserts first), schema creation after a late connection.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Child rows written before the session row violate foreign keys; retried 3× (~0.5 s) and dropped (§2) | Every session loses its first history batch and early per-step gradient rows; ingest stalls 0.5 s per rejected insert | **Verified** (real PostgreSQL 16) | [TD-479](../../guides/TECHNICAL_DEBT.md#td-479-postgres-backend-drops-each-sessions-first-history-batch-on-foreign-key-violations) |
| Schema bootstrap only on a successful first connection; reconnects under the pool mutex; no timeouts; non-idempotent and non-retryable retries; archive key collision (§2) | Broken store after a late DB start; stalls; duplicate rows | By inspection | [TD-480](../../guides/TECHNICAL_DEBT.md#td-480-postgresmetricsdatabase-one-time-schema-bootstrap-reconnects-under-lock-and-unsafe-retries) |
| Timestamp fraction parsing; no retention | Already tracked | — | TD-477, TD-440 |
