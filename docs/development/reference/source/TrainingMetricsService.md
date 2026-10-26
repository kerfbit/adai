# `TrainingMetricsService` — Source File Reference

- **Files:** [`src/TrainingMetricsService.hpp`](../../../../src/TrainingMetricsService.hpp) (508 lines), [`src/TrainingMetricsService.cpp`](../../../../src/TrainingMetricsService.cpp) (1,856 lines)
- **Namespace:** global (`TrainingMetricsSnapshot`, `PersistentMetricsRecord`, `MetricsServiceConfig`, `TrainingMetricsService`, `GlobalMetricsService`)
- **Built into:** `adai_core`
- **Used by:** `metrics_api_server` only: one instance per session in [`MetricsSessionRegistry`](MetricsSessionRegistry.md), driven by [`TrainingMetricsAPI`](TrainingMetricsAPI.md)'s push routes. Trainers don't use it (they push through [`MetricsPushClient`](MetricsPushClient.md)); `MetricsServiceConfig` is also read by `ChatbotTrainer` for its outlier thresholds and generation-quality settings
- **Status tag:** `@adai-status: beta` ("capped by TD-039 — large, actively evolving"), `@adai-version: 0.9.2`, `@adai-reviewed: 2026-09-23` (both files)
- **Tests:** [`training_metrics_service_resume_test.cpp`](../../../../tests/training_metrics_service_resume_test.cpp) (5), [`stale_detection_test.cpp`](../../../../tests/stale_detection_test.cpp) (16), [`lejepa_metrics_test.cpp`](../../../../tests/lejepa_metrics_test.cpp) (18), [`padding_efficiency_test.cpp`](../../../../tests/padding_efficiency_test.cpp) (16), [`adaptive_clipping_test.cpp`](../../../../tests/adaptive_clipping_test.cpp) (12), plus the database suites; all pass
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linking `libadai_core.a` (and the real SQLite backend).

---

## 1. What this file is

The per-session state machine of the metrics daemon: it receives pushes, keeps a live snapshot for dashboards, keeps a sampled history, and persists
to JSONL/summary/Prometheus files and the metrics database.

### Data

| Type | Contents |
|---|---|
| `TrainingMetricsSnapshot` | Live state: session identity/label/config, epoch and sample progress, current and running losses, perplexities, per-epoch histories (losses, LR, durations, gradient norms, validation accuracy/perplexity, BLEU/ROUGE, padding efficiency, LeJEPA losses and diagnostics, adaptive-clip thresholds), throughput and ETA, the TD-013 diagnostics, staleness (`is_stale`, `seconds_since_last_update`, `effective_is_training`) |
| `PersistentMetricsRecord` | One history row: timestamp, session, epoch, sample, losses, LR, gradient norm, perplexity, diagnostics, LeJEPA fields, per-layer gradient JSON |
| `MetricsServiceConfig` | Persistence files and cadence (`persist_every_samples` 100, `persist_every_seconds` 30), `max_records_in_memory` 10,000, `max_records_on_disk` 100,000 (**unused**, TD-440), `compress_old_records` (**unused**), Prometheus, a push section (`enable_push`, URL, key, timeout; see §2), trainer-side outlier thresholds and generation-quality settings, `staleness_threshold_seconds` 60 |

### Behaviour

| Area | Behaviour |
|---|---|
| Sessions | `start_session()` resets progress (keeping best-validation tracking unless `reset_best`), stores label/config, writes the summary; `end_session()` marks the end (and `mark_session_ended()` in the DB); `heartbeat()` refreshes staleness |
| Samples | `update_sample_metrics()` updates the snapshot, the running loss (EMA 0.1) and throughput/ETA (measured from the first sample, not session start). **History is sampled:** one `PersistentMetricsRecord` is added every `persist_every_samples` samples (when `persist_metrics()` runs) and one per epoch from `end_epoch()` (which persists on the `persist_every_seconds` trigger) |
| Epochs and validation | `start_epoch()`, `end_epoch()` (appends per-epoch histories, recomputes perplexity), `update_validation_metrics()`, `update_best_metrics()`, plus the diagnostic updaters (advanced, adaptive clip, generation quality, padding efficiency, activation saturation, attention entropy, LeJEPA, layer gradient norms) |
| Staleness | `is_stale` when training and no ingest for `staleness_threshold_seconds`, extended to epoch duration + threshold while validation is expected (after the epoch's last sample) |
| Outliers | `flag_abnormal_sample()` keeps up to `max_abnormal_samples` and persists them to `abnormal_samples_file` |
| Export | `to_json()`, `to_json_summary()`, `to_prometheus(key)`, `to_csv_header()`/`to_csv_row()` |
| History access | `get_history(n)` (latest n), `get_session_history(id)`, `get_epoch_losses()`, `get_epoch_validation_losses()` |
| Persistence | `persist_metrics()` writes the records added since `persisted_up_to_`: DB inserts first, then appends JSONL; `persist_summary()` upserts the DB session row and rewrites the summary JSON; `flush_to_disk()` does all of them; `trim_history()` caps memory and shifts the marker |
| Restore | At construction, `restore_from_summary()` loads the session row from the DB (TD-020) or else the summary file |
| `clear_history()` | Clears history and the best-validation tracking, then persists the summary |
| Threading | One mutex per instance; file writes in `flush_to_disk()` happen outside the lock where possible |

`GlobalMetricsService` (`instance()`/`initialize()`/`shutdown()`) wraps a registry's `0-default` slot; only tests use it.

---

## 2. Problems

### A failing JSONL write duplicates every unpersisted row in the database (verified)

`persist_metrics()` inserts the pending records into the database, then opens the JSONL file; if the open fails it logs a warning and **returns before
advancing `persisted_up_to_`**. The next persist inserts the same records again, plus the new ones. With the JSONL path in a read-only directory and a
real SQLite store, 30 samples at `persist_every_samples = 10` (3 history records) left **9 rows** in `metrics_history` (1 + 2 + 3 + 3): the row count
grows quadratically while the file can't be written, and every duplicate is served by `db-history`, compare and export. (A DB exception mid-loop has the
opposite effect: the marker advances and the remaining rows never reach the DB, TD-479.) The DB and file progress should be tracked separately, or the
marker advanced once the DB write succeeded.

### Smaller items (by inspection)

- **Dead push path:** `enable_push` is never set anywhere (trainers use `MetricsPushClient`), so `push_to_api()` and the push config fields are unused;
  if enabled it would spawn a detached thread per sample.
- **JSONL rows carry only the basic fields** (loss, validation loss, LR, gradient norm, perplexity); the diagnostics and LeJEPA fields reach only the
  database.
- **`compress_old_records`** is never read, and `max_records_on_disk` isn't enforced (TD-440).
- **`GlobalMetricsService`'s comment** says it is kept because `instance()` "is used everywhere"; nothing outside the tests calls it.
- **The sampled history isn't documented as sampled**: `/history` and the DB contain one row per `persist_every_samples` samples (plus one per epoch), which matters for anyone
  computing per-sample statistics from them.

### Already tracked

| Item | Tracked as |
|---|---|
| NaN/inf in JSON output | TD-443 |
| No retention; `max_records_on_disk` unused | TD-440 |
| Session row written after child rows (Postgres FK drops) | TD-479 |
| NaN read back as 0 from SQLite | TD-510 |
| Finished sessions fill the registry cap | TD-445 |
| Step counter restarts per service instance | TD-441 |

---

## 3. Tests

The service is exercised by: `training_metrics_service_resume_test.cpp` (5: DB-first restore, best-validation carry-over, the `GlobalMetricsService`
proxy), `stale_detection_test.cpp` (16: staleness and the validation extension), `lejepa_metrics_test.cpp` (18), `padding_efficiency_test.cpp` (16),
`adaptive_clipping_test.cpp` (12), and the SQLite/Postgres database suites (writes through the service).

**Not covered:** persistence when the JSONL file can't be opened; the history's sampling rate.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| JSONL open failure returns before the persisted marker moves, so DB rows are re-inserted every persist (§2) | Quadratic duplicate rows (9 for 3 records) in every DB query | **Verified** | [TD-529](../../guides/TECHNICAL_DEBT.md#td-529-metrics-service-re-inserts-db-rows-when-the-jsonl-file-cant-be-written) |
| Dead push path; JSONL missing extended fields; unused `compress_old_records`; stale `GlobalMetricsService` comment; undocumented sampling (§2) | Dead code and misleading expectations | By inspection | [TD-530](../../guides/TECHNICAL_DEBT.md#td-530-trainingmetricsservice-dead-push-path-partial-jsonl-rows-and-undocumented-sampling) |
| NaN JSON, retention, FK ordering, NaN reads, session cap, step counter | Already tracked | — | TD-443, TD-440, TD-479, TD-510, TD-445, TD-441 |

---

## 5. Merge notes

This file replaces `docs/development/TRAINING_METRICS_SERVICE.md` (409 lines). Where they disagree, this file wins:
- **"Integration with IncrementalTrainer"** (a `TrainingMetricsService` member wired to trainer callbacks) describes an older design; trainers now push
  through `MetricsPushClient`, and this class runs only inside `metrics_api_server`.
- **"Global Singleton Access"** presented `GlobalMetricsService` as the normal entry point; it is a test-only compatibility shim.
- **"Max records on disk"** and **`compress_old_records`** aren't implemented.
- **"Efficient persistence… minimal overhead"** omitted that history is sampled every `persist_every_samples`.

Kept: the snapshot fields, configuration, export formats and threading model. Dropped: the example program, monitoring script and use-case list.

---

## Addendum (October 25, 2026): persistence markers and session row order (TD-529, TD-479)

- **TD-529:** `persist_metrics()` keeps two markers, `db_persisted_up_to_` for the database and `persisted_up_to_` for the JSONL file.
  Each advances only past rows its own write accepted. An unopenable JSONL file no longer makes the next persist re-insert DB rows (10
  samples now give 10 rows; the old code left 75), and a DB exception leaves the failed row for the next persist instead of skipping it.
  `clear_history()` and `trim_history()` adjust both markers.
- **TD-479:** `start_session()` upserts the session row as soon as a database is attached, so child rows (history, gradient variance)
  always have their parent. This matters for Postgres, which enforces the foreign keys.

Tests: `UnwritableMetricsFileDoesNotDuplicateDbRows`, `FailingDbRowsAreInsertedOnceWhenItRecovers` (`MetricsDatabaseTest.cpp`),
`FreshServiceSessionKeepsFirstRowsWithoutRetryStalls` (`postgres_metrics_database_test.cpp`).

---

## Addendum (October 25, 2026): row cap and JSONL rotation (TD-440)

- `max_records_on_disk` is enforced: after each DB persist, `persist_metrics()` calls `IMetricsDatabase::cap_session_rows()` for the
  session (see MetricsDatabase.md). A 200-sample run with a cap of 40 keeps 30–40 rows, including the newest.
- The JSONL file rotates: `rotate_metrics_file_if_needed()` moves it to `<file>.1` at `metrics_file_max_mb` (backups shift up to
  `<file>.<metrics_file_backups>`, the oldest is deleted) before the next append.
- `retention_days` is read by the registry sweep, not by this class.
- `compress_old_records` was removed. It was never read.

Tests: `ServiceKeepsSessionRowsUnderMaxRecordsOnDisk`, `MetricsFileRotatesAtSizeAndKeepsBackups`.

## Addendum (October 25, 2026): batched DB writes (TD-511)

`persist_metrics()` writes the rows since the DB marker as one `insert_metrics_records()` batch, and the marker moves only when the
batch is stored. `update_advanced_epoch_metrics()` no longer inserts a row per optimizer step. It buffers the row, and
`flush_gradient_variance_locked()` writes the buffer as one batch at every persist (even with file persistence off) or every 256 rows.
On failure the buffer is kept, capped at 100,000 rows with the oldest dropped. `set_database()` and the registry's archiving flush
first.

## Addendum (October 25, 2026): push path removed, full JSONL records (TD-530)

- **Removed.** The push path (`push_to_api()`, `build_push_url()`, the `enable_push`/`push_url`/`session_key`/`push_timeout_ms`
  fields) and `GlobalMetricsService` are gone. Trainers push through `MetricsPushClient`, and `MetricsSessionRegistry` owns the
  services.
- **JSONL.** `persist_metrics()` writes every `PersistentMetricsRecord` field through `JsonlFloat` (null for non-finite, shortest
  round-trip otherwise). The layer-gradient-norms JSON uses it too.
- **Sampling.** History sampling (one record per `persist_every_samples` samples plus one per epoch) is documented in
  TRAINING_METRICS_API.md.

Test: `TrainingMetricsServiceJsonl.LinesAreValidJsonWithEveryField`.
