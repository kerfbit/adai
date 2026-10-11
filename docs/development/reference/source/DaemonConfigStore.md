# `DaemonConfigStore` — Source File Reference

- **Files:** [`src/DaemonConfigStore.hpp`](../../../../src/DaemonConfigStore.hpp), [`src/DaemonConfigStore.cpp`](../../../../src/DaemonConfigStore.cpp)
- **Namespace:** `adai`
- **Built into:** `adai_core` (links `sqlite3`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0` (`.hpp`) / `1.0.1` (`.cpp`), `@adai-reviewed: 2026-09-10` / `2026-09-12`
- **Tests:** [`tests/daemonconfigstore_test.cpp`](../../../../tests/daemonconfigstore_test.cpp) (4 tests)
- **Origin:** the "Daemon admin config API" (CLAUDE.md) — persisted `PUT /admin/config` overrides
- **Last traced against the code:** 2026-10-10

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> against real SQLite files, including two processes writing to the same database.

---

## 1. What this file is

A ~100-line SQLite-backed **string → string map** with one table:

```sql
CREATE TABLE IF NOT EXISTS daemon_config (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_utc TEXT NOT NULL);
```

Its intended job is to persist settings an operator changed at runtime through a daemon's
`PUT /admin/config`, so they survive a restart. At startup a daemon loads its
`config.<service>.conf`, then overlays `load_all()` on top: **file < persisted admin overrides <
CLI flags**. Each daemon gets its own `daemon_config.db`, separate from its main database (MNS's
`models.db`, metrics' `metrics.db`). The paths those settings live under (`data_dir`, `db_path`,
`port`) are never stored here, which avoids a "need the DB to find the DB" loop.

### Why it matters

This is what makes runtime tuning durable: auto-save cadence on the trainer, session limits on
the metrics server, the registry target on MNS, FTP limits on the registry. It's also, less
obviously, the **session-number allocator** for training runs (§3), so its concurrency behaviour
decides whether two training passes can be handed the same session ID.

---

## 2. API

| Member | Behaviour |
|---|---|
| `DaemonConfigStore(db_path)` | `sqlite3_open()` (creates the file if missing; the **directory** must exist), sets `PRAGMA journal_mode=WAL` (result unchecked), creates the table. Throws `std::runtime_error` if open or schema creation fails, closing the handle first |
| `~DaemonConfigStore()` | Closes the handle |
| `load_all()` | `SELECT key, value` into a `std::map`. A prepare failure logs a warning and returns an empty map. Stepping stops at the first non-`SQLITE_ROW` result **without checking whether it was an error**, so a mid-scan failure returns a partial map silently |
| `set(key, value)` | One upsert (`INSERT … ON CONFLICT(key) DO UPDATE`) stamping `updated_utc` as ISO-8601 UTC (`PortableTime`'s `gmtime_utc`). Throws on prepare or step failure, with SQLite's message (verified: `"database is locked"` is reported correctly) |

Copying is deleted. There is no `get(key)`, `erase(key)`, or batch/transaction API, and **no busy
timeout** (§4). The class isn't internally synchronized; the header says callers serialize access.

---

## 3. Where it's used

| Caller | File | What it stores | Load / save |
|---|---|---|---|
| `mns_server` | `ModelNameServiceServer.cpp`, `ModelNameService.cpp` | `registry_url`, `registry_group` | Overlay read at startup through a short-lived instance; a long-lived `config_store_` written on `PUT /admin/config` |
| `metrics_api_server` | `TrainingMetricsAPI.cpp`, `MetricsApiServerArgs.cpp` | 9 session-registry and persistence tunables | Startup overlay; all 9 rewritten on every PUT |
| `registry_server` | `RegistryServer.cpp` | `ftp_token_ttl_minutes`, `ftp_max_sessions_per_run` | Startup overlay (`g_config_store`) and PUT |
| `registry_server` | `RegistryServer.cpp` (`handle_session_next`) | **Session counters**: key `model\|run_id` → last session number, in `<data_dir>/session_counters.db` | `load_all()` then `set(n + 1)`, under `g_session_store_mtx` |
| `incremental_trainer` child | `TrainerAdminAPI.cpp` | The four auto-save/retention keys, in `TRAINER_ADMIN_DIR/daemon_config.db` | Overlay in `start()`; all four rewritten on PUT |
| `trainer_service` | `TrainerServiceProxy.cpp` | The **same** file as the child | Opened per request while no child is alive (idle GET/PUT) |
| standalone trainer | `RegistryTransport.cpp` (`LocalTransport::next_session`) | Session counters in `<session_dir>/session_counters.db` | `load_all()` then `set(n + 1)`, **no lock** |

The counter users reuse the class as a generic key-value store, so their file contains a table
named `daemon_config` holding session counters. `load_all()` reads **every** counter ever
allocated just to look up one, and the table only grows (one row per run). That's cheap at today's
scale but O(runs) per allocation.

---

## 4. Concurrency (verified)

> **No busy timeout.** SQLite's default is to fail immediately when another connection holds the
> write lock. Running it: with a second connection holding a write transaction, `set()` threw
> `database is locked` after **0.1 ms**. In WAL mode `load_all()` still worked, returning the last
> committed value.

Two processes do share files:

- `trainer_service` and its `incremental_trainer` child share `TRAINER_ADMIN_DIR/daemon_config.db`.
  The proxy only opens it when it believes no child is alive, so overlap needs a race around a
  child's start or exit.
- Two standalone trainers can share a `session_dir`.

> **Session counters lose updates across processes.** `LocalTransport::next_session()` does
> `load_all()` → `+1` → `set()` with no transaction and no lock. Two processes each doing 200
> such increments against one file produced:
> - 284 of 400 calls **threw** (`database is locked`);
> - a final counter of **113**, although 116 calls reported success, so 3 increments were
>   silently lost (two processes handed the same session number).
>
> In `begin_run()` a throw is caught and logged, leaving the session ID **empty**.
> `registry_server`'s handler is safe within one process (it holds `g_session_store_mtx`) but
> would race the same way if two daemons shared a `data_dir`.

The fix in this class is small:
- `sqlite3_busy_timeout()` in the constructor;
- an atomic `increment(key)`, either `UPDATE … SET value = value + 1 RETURNING value` or a
  `BEGIN IMMEDIATE` read-modify-write;
- optionally a `set_many()` that writes several keys in one transaction (§5).

---

## 5. Caller-side gotchas found while tracing

These are in the callers, not this class, but they're about how persisted config is written:

- **Multi-key PUTs aren't atomic.** Every daemon's PUT calls `set()` once per key with no
  transaction. If one `set()` throws (e.g. locked), earlier keys are persisted and later ones
  aren't, while the in-memory state already has all the new values. The exception escapes the
  handler, so the client gets an error for a change that partly applied.
- **Trainer PUTs apply keys before validating later ones.** Both
  `TrainerAdminAPI::handle_put_config()` and `TrainerServiceProxy::handle_put_config_idle()` walk
  the body key by key. `{"auto_save_enabled":false,"auto_save_every_samples":-5}` returns **400**,
  but `auto_save_enabled=false` has already been applied:
  - in the child, to the live state only (the 400 returns before anything is persisted, so live
    state and DB now disagree);
  - in the idle proxy, already written to the DB, so the next pass starts with auto-save off.
- **`registry_server` and `metrics_api_server` don't range-check their PUT values**, so negatives
  are accepted and persisted. A negative `max_live_sessions` becomes a huge `size_t`. The trainer
  PUTs do validate.

---

## 6. Other notes

- **The WAL pragma's result is ignored.** On filesystems without shared memory (some network
  mounts) SQLite falls back to rollback journalling, where readers *are* blocked by writers. Then
  `load_all()`'s unchecked step loop can return a partial or empty map silently at startup, dropping
  persisted overrides without a warning.
- The constructor needs the parent directory to exist (verified: `unable to open database file`).
  Every caller runs `create_directories()` first.
- A read-only database file opens successfully (SQLite falls back to read-only, and
  `CREATE TABLE IF NOT EXISTS` is a no-op), so the failure only appears at the first `set()`.
- Values are untyped strings. Each caller parses its own (`std::stoi` with `catch (...)` →
  ignore, or `== "true" || == "1"` for booleans), so a corrupted value is skipped silently rather
  than reported.

---

## 7. Tests

`daemonconfigstore_test.cpp` has 4 tests: an empty fresh store, set/load round trip, upsert without
duplication, and persistence across reopen.

**Not covered:**
- open failure;
- a locked database or concurrent writers;
- the session-counter read-modify-write;
- error propagation from `load_all()`.

---

## 8. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No busy timeout; session counters are a non-atomic read-modify-write with no lock in `LocalTransport` (§4) | Concurrent writers fail instantly; two standalone trainers (or a trainer and its supervisor) can get "database is locked" errors or duplicate session numbers | **Verified** | [TD-304](../../guides/TECHNICAL_DEBT.md#td-304-session-counters-can-be-duplicated-and-daemonconfigstore-has-no-busy-timeout) |
| Multi-key PUTs are non-atomic; trainer PUTs apply keys before validating later ones (§5) | A rejected or failed PUT still partly applies; live state and DB can disagree | By inspection | [TD-305](../../guides/TECHNICAL_DEBT.md#td-305-a-rejected-or-failed-put-adminconfig-still-partly-applies) |
| `registry_server`/`metrics_api_server` PUTs don't range-check values (§5) | Negative limits accepted and persisted | By inspection | [TD-306](../../guides/TECHNICAL_DEBT.md#td-306-registry_server-and-metrics_api_server-admin-puts-dont-range-check-values) |
| `load_all()` treats a step error as end of data; WAL pragma unchecked (§2, §6) | Persisted overrides can be silently dropped at startup | By inspection | [TD-307](../../guides/TECHNICAL_DEBT.md#td-307-daemonconfigstoreload_all-hides-read-errors) |
| Counter store reuses the config table and scans it per allocation (§3) | Naming confusion; O(runs) per allocation, unbounded growth | By inspection | [TD-308](../../guides/TECHNICAL_DEBT.md#td-308-session-counters-reuse-the-config-stores-table-and-scan-it-per-allocation) |
