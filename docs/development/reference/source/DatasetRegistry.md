# `DatasetRegistry` — Source File Reference

- **Files:** [`src/DatasetRegistry.hpp`](../../../../src/DatasetRegistry.hpp) (504 lines), [`src/DatasetRegistry.cpp`](../../../../src/DatasetRegistry.cpp) (947 lines)
- **Namespace:** global (no namespace)
- **Built into:** `adai_core`; used by `incremental_trainer` (including `trainer_service`'s passes), `dataset_manager`, `vocab_builder`'s loader and their tests
- **Status tags:** `@adai-status: stable`; `DatasetRegistry.cpp` and `.hpp` `@adai-version: 2.0.0`, `@adai-reviewed: 2026-11-12` (MAJOR, TD-552: `remote_upload()` returns an `UploadResult`)
- **Tests:** [`tests/DatasetRegistryTests.cpp`](../../../../tests/DatasetRegistryTests.cpp) → `datasetRegistryTests` (85), [`tests/dataset_registry_live_test.cpp`](../../../../tests/dataset_registry_live_test.cpp) → `datasetRegistryLiveTests` (106, against a real `registry_server`), [`tests/incremental_trainer_registry_test.cpp`](../../../../tests/incremental_trainer_registry_test.cpp) (14, in `incrementalTrainerDecouplingTests`)
- **Design:** [dataset_manager_separation.md](../../archive/dataset_manager_separation.md) (TD-028, archived); TD-202 (kinds), TD-205 (segments)
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class against a scratch `LocalTransport` pool, or a
> real `registry_server`.

---

## 1. What this file is

The **training-data ledger**: the in-memory view of, and operations on, two lists.

| List | Contents | Local file (`LocalTransport`) |
|---|---|---|
| **Pending queue** (`pending_`, `PendingEntry`) | Files or row-range segments waiting to be trained, with an optional `model_name` assignment and a `run_id` claim | `<session_dir>[/<kind>]/pending_files.txt` |
| **Trained registry** (`registry_` + `trained_set_`, `DataVersion`) | Trained units: path, segment range, checksum, sample count, timestamp | `<session_dir>[/<kind>]/data_registry.txt` |

All storage goes through a `RegistryTransport` chosen at construction: `registry_server_url` set → `RemoteTransport` (HTTP to `registry_server`;
group = `run_group`, else the session directory's basename; a `/<kind>` segment for a kind); otherwise → `LocalTransport` (JSON Lines files, an
advisory `flock`, atomic saves). A binary built without cpp-httplib warns and uses local files. A `dataset_kind` other than the four kinds throws
`std::invalid_argument` (TD-402); the empty legacy pool is allowed.

It also owns `load_conversation_pairs()`, the training-file parser every training path uses (§5).

Every training pass starts with `acquire_pending()` and ends with `mark_trained()` or `release_pending()`. That sequence decides what each model
trains on, keeps two trainers off the same unit, and records what's done.

---

## 2. Configuration

`DatasetConfig`: `session_dir`, `data_registry_file`; `registry_server_url`, `run_group`, `dataset_kind`, `run_id`, `registry_timeout_ms`;
`model_name`, `max_files_per_run`; `download_dir`, `max_parallel_downloads`, `large_file_warn_threshold_mb`; the FTPS verification (TD-339); and
the tokenized-cache settings. `make_config(ServiceConfig)` copies the matching fields, but **neither `max_files_per_run`** (so it is always 0 =
all) **nor `data_registry_file`** (TD-335).

---

## 3. API

### Pending queue

| Method | Behaviour |
|---|---|
| `add_file(path, seg_start = -1, seg_count = -1)` | The path must exist locally, the unit must not be trained (`is_trained()` of that unit) or already pending (`segment_matches()` on `(path, range)`); persisted through `transport_->add_pending()`. Stored as given (callers make it absolute, TD-328) |
| `add_pending_path_unchecked(…)` | The same without the existence check (`migrate`/`segment`, where the path lives on the server) |
| `add_files(paths)` | Loops `add_file()`; no caller outside the class |
| `clear_pending()` | Memory only; the caller must save, which is a no-op on `RemoteTransport` |
| `remove_pending(path)` | Erases the first entry with that path, saves, returns the save result (TD-327) |
| `assign_model(…)` / `unassign_model(…)` / `delete_entries(…)` | Delegate to the transport (both modes), then mirror the result into `pending_` with segment-exact matching. `count` 0 with no paths assigns **all**; a negative count is refused (TD-321) |
| `pending_files()`, `pending_entries()`, `trained_files()` (each trained path once), `trained_units()` (whole files and segments; a whole-file record covers its segments, TD-546) | Snapshots |
| `is_trained(path, seg_start = -1, seg_count = -1)` | A whole-file query is true when any record of the path exists; a segment query when the whole file or that exact segment was trained (TD-333) |
| `count_acquirable()` | Loaded pending entries this registry's `model_name` may acquire (unassigned, or assigned to it), the rule `acquire()` uses; `incremental_trainer`'s pre-check (TD-413) |
| `requeue_trained(paths, all, model)` | Moves trained entries back to pending through the transport (server-side against `registry_server`), then reloads both lists (TD-540) |
| `local_pool_dir(cfg)` (static) | `session_dir`, or `session_dir/<kind>`; shared by the transport factory and `IncrementalTrainer::reset_all()` (TD-545) |

### Training lifecycle

| Method | Behaviour |
|---|---|
| `acquire_pending(run_id, max_files)` | `transport_->acquire(run_id, limit, model_name)`: claims unclaimed (or this run's own) entries that are unassigned or assigned to this model, and returns `FileToken`s with segment ranges. With a claim lease, starts a `ClaimRenewer` (TD-494). New units are added to `pending_`; entries already there don't get their `run_id` updated in memory (TD-335) |
| `ClaimRenewer` (private) | A thread calling `transport_->renew_claims(run_id)` every third of the lease (at least every 5 s), logging once if the claims are gone; stopped by `mark_trained()`, `release_pending()`, the next acquire and the destructor |
| `next_session(model, run_id)` | The registry's per-run session counter ([DaemonConfigStore.md](DaemonConfigStore.md), TD-304) |
| `mark_trained(run_id, units, counts)` | The production overload, with an acquired batch's `PendingFileRange`s (`resp.file_ranges()`). `build_new_versions()` adds a `DataVersion` per unit, keyed by (path, segment), with its pair count (TD-334); with a run id a unit already recorded is sent again (TD-493) and the transport records only units this run claimed. `commit_trained()` removes exactly those units from pending |
| `mark_trained(run_id, paths, counts)` | Whole-file overload |
| `mark_trained(paths, counts)` | Run-less overload: saves and drops the paths from `pending_`; test-only, skips units already recorded |
| `release_pending(run_id, units)` / `release_pending(run_id, paths, segments)` | `transport_->release()` of exactly the given units (TD-333) |
| `print_run_assignments()` | Prints claims; no caller |

### Remote-only helpers

`remote_fetch_gutenberg()` and `remote_fetch_huggingface()` call the server and add the returned path to `pending_`.
`remote_upload(local_path, overwrite = false)` returns the transport's `UploadResult` (TD-552) and adds the path only for a new entry: a
file that `overwrite` replaced in the queue is already there. In local mode they fail; `dataset_manager` uses `DataFetcher` + `add_file()`
instead.

### Persistence and reporting

- `load_registry()`/`save_registry()`, `load_pending_list()`/`save_pending_list()` through the transport.
- `print_registry()` (an ANSI table to stdout), `total_samples_trained()` (the sum of `num_samples`).
- `compute_checksum(path)` is `"<size>_<mtime>"`. Each `DataVersion` stores it, but nothing compares a stored one against the file
  (TD-335); `IncrementalTrainer` uses the function in its tokenized-cache key.
- The private `registry_file_path()`/`pending_file_path()` are unused and ignore the `<kind>` directory (TD-335).

---

## 4. TD-205 segments in the training lifecycle

`acquire_pending()` hands out segments with their ranges, and every production caller passes the batch's exact units (`resp.file_ranges()`) to
`mark_trained()` and `release_pending()`: the chatbot `train`/`retrain`, `IncrementalTrainer::resume_last_session()`, and the LeJEPA pass. Each
trained segment gets its own record and count, a failed pass frees exactly its units, and both `registry_server` and `RemoteTransport` carry the
segment fields (TD-333, verified on a 30-pair file in three segments).

---

## 5. `load_conversation_pairs()` and `count_pairs()`

The format is detected from the first non-empty line, after skipping a UTF-8 byte-order mark (TD-336): `{` means JSONL, anything else the
legacy `INPUT:`/`RESPONSE:` form.

- **JSONL:** each line through `parse_jsonl_sample()`, keeping its `SampleMeta`. With a segment range, only successfully parsed lines count
  toward the index (the same index space as `count_pairs()`); the read stops once the range is filled, and a range past the end yields fewer
  pairs.
- **Legacy:** a pair ends at a blank line or the next `INPUT:`; lines after `RESPONSE:` are part of the response, joined with `\n` and keeping
  their indentation (TD-336). A segment range on a legacy file logs a warning and loads the whole file.
- Each line read calls `adai::note_pass_progress()` (TD-569), so loading a large file counts as progress for `trainer_service`'s stall check.
- `count_pairs()` parses every pair to count them, materializing all the text.

---

## 6. Sample counts

The chatbot passes report `IncrementalTrainer::last_pass_pair_counts()` (pairs loaded per unit; a segment counts only its range) and the LeJEPA
pass the pairs it loads (TD-334). A 20-pair file records 20.

---

## 7. Tests

- **`DatasetRegistryTests.cpp` (85):** queue operations, assignment, delete, re-queue (TD-540), the upload pass-through (a replaced file
  isn't queued twice, a refusal's reason kept, TD-552), segments (TD-205/TD-333: an acquired segmented
  batch released exactly, separately trained segments each recorded), the registry file's segment columns and legacy lines,
  `count_acquirable()` (TD-413), unknown kinds (TD-402), claim renewal (`DatasetRegistryClaimLeaseTest`, TD-494), and the parser (segment ranges,
  a BOM, multi-line legacy responses).
- **`dataset_registry_live_test.cpp` (106):** the same surface against a real `registry_server`, including an exact-segment commit and release.
- **`incremental_trainer_registry_test.cpp` (14):** the trainer integration, including non-zero pair counts.

**Not covered:** a segment trained under different boundaries than it was queued with (only the exact range or a whole-file record matches).

---

## 8. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Stored checksums never verified; `count_pairs()` materializes every pair; unused private path helpers, `add_files()`, `print_run_assignments()`; `make_config()` drops `max_files_per_run`/`data_registry_file`; in-memory `run_id` not updated on acquire (§2–3, §5) | Dead code, an unused integrity field, memory spent counting | By inspection | [TD-335](../../guides/TECHNICAL_DEBT.md#td-335-datasetregistry-dead-code-and-unused-integrity-data) (LOW) |
| Segments released and recorded by bare path | — | **Fixed October 19, 2026** | [TD-333](../../guides/TECHNICAL_DEBT.md#td-333-segment-claims-arent-released-on-failure-and-trained-records-ignore-segments) |
| All-zero sample counts | — | **Fixed October 19, 2026** | [TD-334](../../guides/TECHNICAL_DEBT.md#td-334-trained-sample-counts-are-always-zero) |
| Legacy responses truncated to one line; a BOM breaking JSONL detection | — | **Fixed November 8, 2026** | [TD-336](../../guides/TECHNICAL_DEBT.md#td-336-legacy-training-file-parser-truncates-responses-bom-breaks-jsonl-detection) |
| `remove_pending()` ignoring its save; relative paths; `count <= 0`; unknown kinds; re-trained units; claim leases | — | **Fixed** | TD-327, TD-328, TD-321, TD-402, TD-493, TD-494 |

`stable` stands: TD-335's items are dead code and an unused integrity field, not a gap a caller would trip over.

## 9. History

- **October 18–19, 2026:** `requeue_trained()` (TD-540); segment-aware trained records and releases (TD-333); pair counts (TD-334);
  `count_acquirable()` (TD-413).
- **October 24–25, 2026:** re-trained units sent with their counts (TD-493); claim renewal (TD-494).
- **November 1, 2026:** unknown kinds refused (TD-402). **November 7–8, 2026:** TD-327, TD-328, TD-336.
- **November 11, 2026:** progress per line read (TD-569). This document was re-traced against the code, folding in its addenda.
- **November 12, 2026:** `remote_upload()` takes `overwrite` and returns an `UploadResult` (TD-552).
