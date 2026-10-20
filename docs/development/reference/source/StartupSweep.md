# `StartupSweep` — Source File Reference

- **File:** [`src/StartupSweep.hpp`](../../../../src/StartupSweep.hpp) (header-only, 147 lines)
- **Namespace:** global (`startup_sweep()`)
- **Used by:** `incremental_trainer` ([`IncrementalTrainingTool.cpp`](../../../../src/IncrementalTrainingTool.cpp)): before the first acquire in `train`, `retrain` and `--objective=lejepa`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.3.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** [`tests/StartupSweepTests.cpp`](../../../../tests/StartupSweepTests.cpp) (10 tests; all pass)
- **Design:** [dataset-transport-proposal.md](../../archive/dataset-transport-proposal.md), "Stale Download Cleanup"
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real `DatasetRegistry` and `startup_sweep()`.

---

## 1. What this file is

`startup_sweep(reg, run_id, download_dir)` cleans the FTP download directory (`DOWNLOAD_DIR`, `downloads` in `config.trainer.conf`) before a pass
acquires files. It returns at once if the directory is unset or missing. It loads the pending list itself, and **skips the whole sweep if that
fails** (TD-512). It then loads the trained registry, treating a failure as "nothing trained". Files are matched to the registry by
`download_file_name(registry_path)`, the name `DataTransport` gives a download: a full-path hash, `_`, the basename (TD-337). It used to
match by basename:

| Condition | Test | Action |
|---|---|---|
| — | A `<download>.checksum` sidecar (TD-338) | Skipped; removed only if its download is gone. Every deletion below also removes the file's sidecar |
| D | Name is a trained file's and **not** also a pending one's (a file can be trained in part, TD-333) | Delete the local copy |
| G | Name isn't any pending entry's | "Orphaned": delete |
| A | Pending, zero bytes | Delete and `release_pending(run_id, {path})` |
| B/C | Pending, non-empty | Keep: `DataTransport::fetch()` reuses a complete file or resumes a partial one |

Subdirectories are skipped.

---

## 2. Problems

### Fixed: production never loaded the pending list, so the sweep deleted every download (TD-512, October 18, 2026)

`pending_files()` is filled only by `load_pending_list()`, and every call site loaded only the trained registry. With two queued files
(`a.jsonl` partly downloaded, `b.jsonl` complete) and an operator's `notes.txt` in `downloads/`, the production sequence saw 0 pending files and
deleted all three as orphans. So conditions A, B and C never ran: every pass re-downloaded everything in full, resume never happened, and
anything else in `DOWNLOAD_DIR` was deleted. The tests missed it because each added files to the instance it then swept. A registry failure
had the same effect. Now the sweep loads the pending list itself and skips deleting when it can't
(`FreshRegistryKeepsQueuedDownloads`, `UnreadablePendingListDeletesNothing`).

### Fixed: basename matching (TD-513's first part, October 19, 2026, with TD-337)

Condition D used to delete a pending file's download whenever any trained file anywhere shared its basename (`train.jsonl`), and two pending
files with one basename collapsed to one map entry. Downloads are now matched by their full-path names
(`TrainedFileWithSameBasenameDoesNotDeleteAPendingDownload`). **Upgrade note:** downloads still under old basename names match nothing, so
the first sweep after the upgrade deletes them once (condition G), and they are fetched again.

### Smaller items (by inspection)

- **Throwing filesystem calls:** `fs::file_size()` and `fs::remove()` (the throwing overloads) and `directory_iterator` can throw on a permission or
  race error, which `incremental_trainer` doesn't catch around the sweep.
- **Condition A releases with `release_pending(run_id, {path})`**: whole-file only, so a zero-byte download of a TD-205 segment isn't released (TD-333),
  and the release is a no-op when another run holds the claim.

---

## 3. Tests

`StartupSweepTests.cpp` (10): conditions D, G, A (deleted and released) and B/C (kept), an empty directory, a missing directory and an empty path.
Added October 18–19, 2026:
- the production sequence and an unreadable pending list (TD-512);
- a same-basename trained file next to a pending download, with sidecar cleanup (TD-337/338).

**Not covered:** filesystem errors, segment release in condition A.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Callers never call `load_pending_list()`; every file looks orphaned (§2) | Every pass deleted all downloads (and anything else in `DOWNLOAD_DIR`); resume and reuse never happened | **Fixed October 18, 2026** (was verified) | [TD-512](../../guides/TECHNICAL_DEBT.md#td-512-startup-sweep-deletes-every-download-because-the-pending-list-is-never-loaded) |
| Registry failure also wipes the directory (§2) | Data loss on a network blip | **Fixed October 18, 2026** | [TD-512](../../guides/TECHNICAL_DEBT.md#td-512-startup-sweep-deletes-every-download-because-the-pending-list-is-never-loaded) |
| Basename matching: trained-name collisions delete pending downloads; same-name pending files collapse (§2) | Spurious deletions and wrong releases | **Fixed October 19, 2026** (with TD-337) | [TD-513](../../guides/TECHNICAL_DEBT.md#td-513-startup_sweep-basename-matching-and-unchecked-filesystem-calls) |
| Throwing filesystem calls; whole-file release (§2) | Crash on permission errors; stranded segment claims | By inspection | [TD-513](../../guides/TECHNICAL_DEBT.md#td-513-startup_sweep-basename-matching-and-unchecked-filesystem-calls) |
| Download naming by basename (fixed October 19, 2026); segment release | Already tracked | — | TD-337, TD-333 |

## Addendum (October 18, 2026): TD-512 fixed

`startup_sweep()` now calls `reg.load_pending_list()` and `reg.load_registry()` itself.
- If the pending list can't be loaded, it returns without deleting anything.
- If only the trained registry fails to load, condition D is disabled (empty trained set) and the rest runs.

Callers no longer need to load anything first. The `resume` command (every `trainer_service` pass) now sweeps before its pass too. New tests:
- `FreshRegistryKeepsQueuedDownloads` reproduces the production sequence above and keeps both queued downloads; the old code deleted them.
- `UnreadablePendingListDeletesNothing`.

TD-513 (basename matching) was left for TD-337 and fixed with it (next addendum).

## Addendum (October 19, 2026): download names and checksum sidecars (TD-337, TD-338)

The sweep now matches downloads to pending and trained entries by `download_file_name(registry_path)`, the name `DataTransport` gives them, instead of by basename. A trained file that shares a pending file's basename therefore no longer deletes that file's download. Each deleted download takes its `.checksum` sidecar with it. A sidecar whose download is gone is removed on its own. After an upgrade, downloads still under their old basename names look orphaned and are deleted once (condition G), so they are fetched again on the next pass.
