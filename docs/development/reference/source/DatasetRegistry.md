# `DatasetRegistry` — Source File Reference

- **Files:** [`src/DatasetRegistry.hpp`](../../../../src/DatasetRegistry.hpp), [`src/DatasetRegistry.cpp`](../../../../src/DatasetRegistry.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_core`; used by `incremental_trainer` (including `trainer_service`'s child passes), `dataset_manager`, and their tests
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.1`, `@adai-reviewed: 2026-09-20`
- **Tests:** [`tests/DatasetRegistryTests.cpp`](../../../../tests/DatasetRegistryTests.cpp) (67), [`tests/dataset_registry_live_test.cpp`](../../../../tests/dataset_registry_live_test.cpp) (86, against a live `registry_server`), [`tests/incremental_trainer_registry_test.cpp`](../../../../tests/incremental_trainer_registry_test.cpp) (14)
- **Design:** [dataset_manager_separation.md](../../archive/dataset_manager_separation.md) (TD-028, archived); TD-202 (kinds), TD-205 (segments)
- **Last traced against the code:** 2026-10-11

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
| `pending_files()`, `pending_entries()`, `trained_files()`, `is_trained(path)` | Snapshots; `is_trained` is **path-only** |

### Training lifecycle

| Method | Behaviour |
|---|---|
| `acquire_pending(run_id, max_files)` | `transport_->acquire(run_id, limit, model_name)`. Claims unclaimed (or already-own-`run_id`) entries that are unassigned or assigned to this model; returns `FileToken`s with segment ranges. Entries already in `pending_` don't get their `run_id` updated in memory |
| `next_session(model, run_id)` | Thin wrapper ([DaemonConfigStore.md](DaemonConfigStore.md), TD-304) |
| `mark_trained(run_id, paths, counts)` | `build_new_versions()` adds a `DataVersion` per **new path** to the registry, then `transport_->commit_trained()` removes pending entries with those paths claimed by `run_id` |
| `mark_trained(paths, counts)` | Older single-run overload: saves the registry and drops those paths from `pending_` (test-only) |
| `release_pending(run_id, paths, segments)` | `transport_->release()`. A bare path releases only a **whole-file** entry; segments need `segments` |
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

## 4. TD-205 segments in the training lifecycle (verified)

`acquire_pending()` hands out segments correctly. Every production caller
(`IncrementalTrainingTool.cpp` for train/resume/retrain/LeJEPA, `IncrementalTrainer::resume_last_session()`)
then calls `mark_trained()`/`release_pending()` with **bare paths**
(`resp.registry_paths()`), as `IncrementalTrainer.cpp`'s own comment notes. Running the real class
on one 30-pair file split into three segments:

> **1. A failed pass doesn't release its segment.** `run-A` acquired one segment, and its failure
> path called `release_pending("run-A", {path})`. Since a bare path only releases whole-file
> entries, the segment **stayed claimed by `run-A`**. A pass with a different `run_id` acquired
> only the other two. Acquire lets the *same* `run_id` reclaim its own claims, so under MNS
> (`resume` keeps the run ID) the next pass of that run recovers it. But a `retrain` (new MNS run)
> or any standalone pass (run ID derived from hostname + **PID**, new every pass) leaves it stuck
> until an operator releases it by hand.
>
> **2. Success records the file once, by path.** `run-B` trained two segments of the same file.
> `mark_trained` produced **1 trained file with 10 samples** (`build_new_versions()` skips a path
> already in `trained_set_`, even within one call), and the whole path now counts as trained:
> re-queuing any segment of it is **rejected** ("already trained"). Segments of a file trained in
> separate passes can't all be recorded either.

The fix is to key trained records and releases by `(path, segment_start, segment_count)`, passing
`FileToken` segments through as `SegmentTarget`s.

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

Every production `mark_trained()` call passes `std::vector<int> counts(n, 0)`: the trainer, the
retrain path, LeJEPA passes, and `resume_last_session()`. So `DataVersion::num_samples` is **always
0** for trained files, and `total_samples_trained()` (shown by `dataset_manager status` as
"Samples: N total trained") always reports 0. That's by inspection of all call sites; the pair
counts are available to the trainer (`train_on_files()` loads them), they're just not passed back.

---

## 7. Tests

- **`DatasetRegistryTests.cpp` (67 tests):** queue operations, assignment, delete, the parser
  (including segment ranges and `NegativeSegmentStartBehavesLikeTheTwoArgOverload`), and the
  TD-205 segment-targeted assign/unassign/delete cases.
- **`dataset_registry_live_test.cpp` (86 tests):** the same surface against a real `registry_server`.
- **`incremental_trainer_registry_test.cpp` (14 tests):** the trainer integration.

**Not covered:** the acquire → fail → release and acquire → train → `mark_trained` lifecycle for
segmented entries (§4), multi-segment trained accounting, and non-zero sample counts.

---

## 8. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Segment claims aren't released on failure (bare-path release); trained records are per path, so a trained segment marks the whole file trained and later segments' samples aren't recorded (§4) | Segments stuck claimed after a failed standalone/retrain pass; re-queuing rejected; trained accounting wrong | **Verified** | [TD-333](../../guides/TECHNICAL_DEBT.md#td-333-segment-claims-arent-released-on-failure-and-trained-records-ignore-segments) |
| Production passes all-zero sample counts to `mark_trained()` (§6) | `num_samples`/`total_samples_trained()` are always 0 | By inspection | [TD-334](../../guides/TECHNICAL_DEBT.md#td-334-trained-sample-counts-are-always-zero) |
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
