# `DatasetRegistry` — Source File Reference

- **Files:** [`src/DatasetRegistry.hpp`](../../../../src/DatasetRegistry.hpp), [`src/DatasetRegistry.cpp`](../../../../src/DatasetRegistry.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_core`; used by `incremental_trainer` (including `trainer_service`'s child passes), `dataset_manager`, and their tests
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.5.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** [`tests/DatasetRegistryTests.cpp`](../../../../tests/DatasetRegistryTests.cpp) (74), [`tests/dataset_registry_live_test.cpp`](../../../../tests/dataset_registry_live_test.cpp) (87, against a live `registry_server`), [`tests/incremental_trainer_registry_test.cpp`](../../../../tests/incremental_trainer_registry_test.cpp) (14)
- **Design:** [dataset_manager_separation.md](../../archive/dataset_manager_separation.md) (TD-028, archived); TD-202 (kinds), TD-205 (segments)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (linked against the debug libraries) against a scratch `LocalTransport` pool.

---

## 1. What this file is

The **training-data ledger**. It's the in-memory view and operations over two lists:

| List | Contents | Local file (`LocalTransport`) |
|---|---|---|
| **Pending queue** (`pending_`, `PendingEntry`) | Files or row-range segments waiting to be trained, with an optional `model_name` assignment and a `run_id` claim | `<session_dir>[/<kind>]/pending_files.txt` |
| **Trained registry** (`registry_` + `trained_set_`, `DataVersion`) | Files already trained: path, checksum, sample count, timestamp | `<session_dir>[/<kind>]/data_registry.txt` |

All storage goes through a `RegistryTransport` chosen at construction (`build_transport()`):
- `registry_server_url` set → `RemoteTransport` (HTTP to `registry_server`, group = `run_group`
  or the session directory's basename, plus a `/<kind>` segment);
- otherwise → `LocalTransport` (flat files with an advisory `flock`).

If the binary was built without cpp-httplib, a configured `REGISTRY_SERVER_URL` logs a warning
and **falls back to local files**.

It also owns `load_conversation_pairs()`, **the** training-file parser the trainer uses (§5).

### Why it matters

Every training pass starts with `acquire_pending()` and ends with `mark_trained()` or
`release_pending()`. That sequence decides what each model trains on, guarantees two trainers
don't train the same file, and records what's done. Bugs here mean data trained twice, never
trained, or stuck.

---

## 2. Configuration

`DatasetConfig` holds:
- `session_dir`, `data_registry_file`;
- `registry_server_url`, `run_group`, `dataset_kind`, `run_id`, `registry_timeout_ms`;
- `model_name`, `max_files_per_run`;
- FTP download settings (`download_dir`, `max_parallel_downloads`, `large_file_warn_threshold_mb`);
- tokenized-cache settings.

`make_config(ServiceConfig)` copies the matching `ServiceConfig` fields. It copies **neither
`max_files_per_run`** (always 0 = all) **nor `data_registry_file`**.

---

## 3. API

### Pending queue

| Method | Behaviour |
|---|---|
| `add_file(path, seg_start = -1, seg_count = -1)` | Requires the path to exist **locally**, not be trained, and not already be pending (`segment_matches` on `(path, range)`). Persists through `transport_->add_pending()`. Stores the path **as given** (TD-328) |
| `add_pending_path_unchecked(…)` | Same, minus the existence check (for `migrate`/`segment`, where the path may live on the server) |
| `add_files(paths)` | Loops `add_file()`. No caller outside this class |
| `clear_pending()` | Clears memory only; the caller must `save_pending_list()`, a no-op on `RemoteTransport` |
| `remove_pending(path)` | Erases the first entry with that path, saves, **ignores the save result** (TD-327) |
| `assign_model(model, paths, count, segments)` / `unassign_model(…)` / `delete_entries(…)` | Delegate to the transport (work in both modes), then mirror the result into `pending_` with segment-exact matching (TD-205). `count <= 0` with no paths assigns **all** (TD-321) |
| `pending_files()`, `pending_entries()`, `trained_files()` (each trained path once) | Snapshots |
| `is_trained(path, seg_start = -1, seg_count = -1)` | TD-333: a whole-file query is true when any record of the path exists; a segment query when the whole file or that exact segment was trained. `add_file()` checks the unit being added |
| `count_acquirable()` | TD-413: loaded pending entries this registry's `model_name` may acquire (unassigned, or assigned to it), the rule `acquire()` uses; `incremental_trainer`'s pre-check |
| `requeue_trained(paths, all, model)` | TD-540: moves trained entries back to pending through the transport (server-side against `registry_server`), then reloads both lists |
| `local_pool_dir(cfg)` (static) | `session_dir`, or `session_dir/<kind>`: where `LocalTransport` keeps the pool (TD-202). Shared by the transport factory and `IncrementalTrainer::reset_all()` (TD-545) |

### Training lifecycle

| Method | Behaviour |
|---|---|
| `acquire_pending(run_id, max_files)` | `transport_->acquire(run_id, limit, model_name)`. Claims unclaimed (or already-own-`run_id`) entries that are unassigned or assigned to this model; returns `FileToken`s with segment ranges. Entries already in `pending_` don't get their `run_id` updated in memory |
| `next_session(model, run_id)` | Thin wrapper ([DaemonConfigStore.md](DaemonConfigStore.md), TD-304) |
| `mark_trained(run_id, units, counts)` | TD-333: the production overload, taking an acquired batch's `PendingFileRange`s (`resp.file_ranges()`). `build_new_versions()` adds a `DataVersion` per unit not already recorded, keyed by (path, segment), with that unit's pair count (TD-334). `transport_->commit_trained()` then removes exactly those units from pending |
| `mark_trained(run_id, paths, counts)` | Whole-file overload: one record per new path, path-wide pending removal of entries claimed by `run_id` |
| `mark_trained(paths, counts)` | Older single-run overload: saves the registry and drops those paths from `pending_` (test-only) |
| `release_pending(run_id, units)` / `release_pending(run_id, paths, segments)` | `transport_->release()` of exactly the given whole files and segments (TD-333; a bare path releases only a whole-file entry, so production passes the batch's units) |
| `print_run_assignments()` | Prints claims; no caller |

### Remote-only helpers (Phase 11)

`remote_fetch_gutenberg()`, `remote_fetch_huggingface()` and `remote_upload()` call the server and
append the returned path to `pending_`. In local mode they fail; callers use `DataFetcher` +
`add_file()` instead.

### Persistence and reporting

- `load_registry()`/`save_registry()` and `load_pending_list()`/`save_pending_list()` go through
  the transport.
- `print_registry()` (ANSI table to stdout) and `total_samples_trained()` (sum of `num_samples`).
- `compute_checksum(path)` is `"<size>_<mtime>"`. It's stored in each `DataVersion` but **never
  compared** anywhere.
- The private `registry_file_path()`/`pending_file_path()` are **unused**, and ignore the TD-202
  `<kind>` subdirectory.

---

## 4. TD-205 segments in the training lifecycle

`acquire_pending()` hands out segments with their ranges. Since TD-333 (October 19, 2026), every
production caller passes the batch's exact units (`resp.file_ranges()`) to `mark_trained()` and
`release_pending()`:
- `IncrementalTrainingTool.cpp` for train, resume, retrain and LeJEPA;
- `IncrementalTrainer::resume_last_session()`.

> **Fixed (TD-333): segments used to be released and recorded by bare path.** Verified on a 30-pair
> file split into three segments:
> - **Failed pass:** a failing `run-A` called `release_pending("run-A", {path})`, which releases only
>   whole-file entries, so its segment **stayed claimed** for good after a `retrain` or any
>   standalone pass (a new run id).
> - **Successful pass:** `run-B` trained two segments and got **one** trained record with 10 samples,
>   and the whole path then counted as trained, so re-queuing any segment was rejected.
>
> Now:
> - releases free exactly the batch's units (`ReleasingAnAcquiredBatchFreesItsSegments`);
> - each trained segment gets its own record and count (`SegmentsTrainedSeparatelyAreEachRecorded`);
> - `registry_server` and `RemoteTransport` carry the segment fields
>   (`RemoteTransportCommitsAndReleasesExactSegments`);
> - the local registry file stores them in trailing columns, and legacy lines still read.

---

## 5. `load_conversation_pairs()` and `count_pairs()`

The format is detected from the first non-empty line: `{` means JSONL, anything else means legacy
`INPUT:`/`RESPONSE:`.

- **JSONL** is parsed per line with `parse_jsonl_sample()`, keeping `SampleMeta`. With a segment
  range, only successfully parsed lines count toward the index (the same index space as
  `count_pairs()`). The read stops once the range is filled; a range past the end just yields
  fewer pairs.
- **Legacy** format: lines starting `INPUT:`/`RESPONSE:`; a pair ends at a blank line or the next
  `INPUT:` (unlike `Dataset.hpp`'s parser, unseparated pairs are kept). Only the **first line** of a
  response is kept; continuation lines are dropped. A segment range on a legacy file logs a warning
  and loads the whole file.
- `count_pairs()` **materializes every pair** to count them (a full parse, including copies of
  all text).

By inspection:
- A UTF-8 byte-order mark before the first `{` makes a JSONL file parse as legacy (0 pairs).
- A JSONL file with blank or comment lines first works, since detection skips blank lines.

---

## 6. Sample counts in production

Fixed October 19, 2026 (TD-334). Every production `mark_trained()` call used to pass
`std::vector<int> counts(n, 0)`, so `DataVersion::num_samples` was always 0 and `dataset_manager
status` reported "Samples: 0 total trained". Now:
- the chatbot passes hand over `IncrementalTrainer::last_pass_pair_counts()`, the pairs loaded per file
  (a segment counts only its range);
- the LeJEPA pass counts the pairs it loads.

A 20-pair file now records 20.

---

## 7. Tests

- **`DatasetRegistryTests.cpp` (74 tests):** queue operations, assignment, delete, the parser
  (including segment ranges and `NegativeSegmentStartBehavesLikeTheTwoArgOverload`), and the
  TD-205 segment-targeted assign/unassign/delete cases.
- **`dataset_registry_live_test.cpp` (87 tests):** the same surface against a real `registry_server`.
- **`incremental_trainer_registry_test.cpp` (14 tests):** the trainer integration.

Added October 18–19, 2026:
- re-queue, including edge cases (TD-540);
- releasing an acquired segmented batch, and separately trained segments (TD-333);
- the registry file's segment columns with legacy lines;
- a live exact-segment commit/release;
- `count_acquirable()` (TD-413).

Non-zero sample counts are covered from the trainer side (`TrainedRecordsCarryThePairsEachFileContributed`).

**Not covered:** segments trained under different boundaries than they were queued with (only the
exact range or a whole-file record matches).

---

## 8. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Segment claims aren't released on failure (bare-path release); trained records are per path, so a trained segment marks the whole file trained and later segments' samples aren't recorded (§4) | Segments stuck claimed after a failed standalone/retrain pass; re-queuing rejected; trained accounting wrong | **Fixed October 19, 2026** (was verified) | [TD-333](../../guides/TECHNICAL_DEBT.md#td-333-segment-claims-arent-released-on-failure-and-trained-records-ignore-segments) |
| Production passes all-zero sample counts to `mark_trained()` (§6) | `num_samples`/`total_samples_trained()` were always 0 | **Fixed October 19, 2026** | [TD-334](../../guides/TECHNICAL_DEBT.md#td-334-trained-sample-counts-are-always-zero) |
| `compute_checksum()` stored but never verified; `count_pairs()` materializes all pairs; private path helpers unused and kind-unaware; `add_files()`/`print_run_assignments()` unused; `make_config()` drops `max_files_per_run`/`data_registry_file`; in-memory `run_id` not updated on acquire (§2–3, §5) | Dead code, missed integrity checks, wasted memory | By inspection | [TD-335](../../guides/TECHNICAL_DEBT.md#td-335-datasetregistry-dead-code-and-unused-integrity-data) |
| Legacy parser keeps only the first response line; a BOM before `{` breaks JSONL detection (§5) | Silent truncation or 0 pairs for affected files | By inspection | [TD-336](../../guides/TECHNICAL_DEBT.md#td-336-legacy-training-file-parser-truncates-responses-bom-breaks-jsonl-detection) |
| `remove_pending()` ignores the save result | Already tracked | — | TD-327 |
| Relative paths stored as given | Already tracked | — | TD-328 |
| `count <= 0` assigns all | Already tracked | — | TD-321 |

## Addendum (October 18, 2026): `requeue_trained()`

`DatasetRegistry::requeue_trained(paths, all, model_name)` delegates to the transport and then reloads the in-memory registry and pending lists. `IncrementalTrainer::reset_all()`'s keep-data branch now uses it (TD-540). It used to delete a local `data_registry.txt` and re-add files by local path, which re-queued nothing on a remote registry.

## Addendum (October 19, 2026): segment-aware trained records (TD-333)

`DataVersion` carries `segment_start`/`segment_count`. `mark_trained(run_id, units, counts)` and `release_pending(run_id, units)` take an acquired batch's `PendingFileRange`s, and every production caller uses them. Trained records are kept per unit, `is_trained(path, segment)` is segment-aware, and delete and re-queue handle segment records. See TD-333 in the tracker for the full list.

## Addendum (October 19, 2026): `count_acquirable()` (TD-413)

`count_acquirable()` returns how many loaded pending entries (whole files and segments) this registry's `model_name` may acquire: unassigned ones, plus ones assigned to it. This is the same rule as `RegistryTransport::acquire()`. `incremental_trainer` uses it for its pre-check, so that check reads the same kind sub-pool and model filter as the pass.

---

## Addendum (October 24, 2026): re-trained units are sent with their counts (TD-493)

`build_new_versions()` takes `record_again`. The run-scoped `mark_trained()` overloads pass `true`, so a unit already in the registry
(re-queued and trained again) is still committed with its real sample count, and the transport decides by claim whether to record it.
The run-less overload, used only by tests, still skips recorded units.

---

## Addendum (October 25, 2026): claim renewal (TD-494)

When `acquire_pending()` gets `claim_lease_seconds > 0`, a private `ClaimRenewer` thread calls `transport_->renew_claims(run_id)` every
third of the lease (at least every 5 s) and logs once if the claims are gone. `mark_trained()`, `release_pending()` and the destructor
stop it. The class gained an out-of-line destructor and move operations. Tests: `DatasetRegistryClaimLeaseTest.*`.
