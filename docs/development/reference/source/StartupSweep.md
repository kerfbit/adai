# `StartupSweep` — Source File Reference

- **File:** [`src/StartupSweep.hpp`](../../../../src/StartupSweep.hpp) (header-only, 107 lines)
- **Namespace:** global (`startup_sweep()`)
- **Used by:** `incremental_trainer` ([`IncrementalTrainingTool.cpp`](../../../../src/IncrementalTrainingTool.cpp)): before the first acquire in `train`, `retrain` and `--objective=lejepa`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/StartupSweepTests.cpp`](../../../../tests/StartupSweepTests.cpp) (7 tests; all pass)
- **Design:** [dataset-transport-proposal.md](../../archive/dataset-transport-proposal.md), "Stale Download Cleanup"
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real `DatasetRegistry` and `startup_sweep()`.

---

## 1. What this file is

`startup_sweep(reg, run_id, download_dir)` cleans the FTP download directory (`DOWNLOAD_DIR`, `downloads` in `config.trainer.conf`) before a pass
acquires files. It returns at once if the directory is unset or missing. Files are matched to the registry **by basename**:

| Condition | Test | Action |
|---|---|---|
| D | Basename is in `reg.trained_files()` | Delete the local copy |
| G | Basename isn't in `reg.pending_files()` | "Orphaned": delete |
| A | Pending, zero bytes | Delete and `release_pending(run_id, {path})` |
| B/C | Pending, non-empty | Keep: `DataTransport::fetch()` reuses a complete file or resumes a partial one |

Subdirectories are skipped.

---

## 2. Problems

### Production never loads the pending list, so the sweep deletes every download (verified)

`pending_files()` returns `DatasetRegistry`'s in-memory `pending_`, which is filled only by `load_pending_list()`. All three call sites construct a
fresh `DatasetRegistry`, call **`load_registry()` only**, and then call `startup_sweep()`. With two queued files (`a.jsonl` partly downloaded, `b.jsonl`
complete) and an operator's `notes.txt` in `downloads/`, the production sequence saw **0 pending files** and **deleted all three** as orphans
(condition G). After `load_pending_list()`, both downloads were kept.

So conditions A, B and C never run in production: every pass deletes every completed and partial download (they are re-downloaded in full, and resume
never happens), and anything else placed in `DOWNLOAD_DIR` is deleted too. The tests don't catch this because each one adds files to the same registry
instance it then sweeps, so `pending_` is already populated.

### The same deletion follows any registry failure (by inspection)

If the registry can't be reached, `load_registry()` returns false with the trained set empty, and the pending list is empty anyway, so every file again
matches condition G. A cleanup step that deletes data should skip deletion when it couldn't load the state it relies on.

### Basename matching (by inspection)

- **Condition D deletes a pending file's download** when any trained file anywhere (another group, model or directory) shares its basename (`train.jsonl`,
  `data.jsonl`): D is checked before the pending lookup. This is the sweep side of TD-337's basename naming of downloads.
- Two pending files with the same basename collapse to one map entry, so condition A can release the wrong one.

### Smaller items (by inspection)

- **Throwing filesystem calls:** `fs::file_size()` and `fs::remove()` (the throwing overloads) and `directory_iterator` can throw on a permission or
  race error, which `incremental_trainer` doesn't catch around the sweep.
- **Condition A releases with `release_pending(run_id, {path})`**: whole-file only, so a zero-byte download of a TD-205 segment isn't released (TD-333),
  and the release is a no-op when another run holds the claim.

---

## 3. Tests

`StartupSweepTests.cpp` (7): conditions D, G, A (deleted and released) and B/C (kept), an empty directory, a missing directory and an empty path. Every
test populates the pending list on the instance it sweeps.

**Not covered:** the production sequence (`load_registry()` only), registry failure, basename collisions, filesystem errors.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Callers never call `load_pending_list()`; every file looks orphaned (§2) | Every pass deletes all downloads (and anything else in `DOWNLOAD_DIR`); resume and reuse never happen | **Verified** | [TD-512](../../guides/TECHNICAL_DEBT.md#td-512-startup-sweep-deletes-every-download-because-the-pending-list-is-never-loaded) |
| Registry failure also wipes the directory (§2) | Data loss on a network blip | By inspection | [TD-512](../../guides/TECHNICAL_DEBT.md#td-512-startup-sweep-deletes-every-download-because-the-pending-list-is-never-loaded) |
| Basename matching: trained-name collisions delete pending downloads; same-name pending files collapse (§2) | Spurious deletions and wrong releases | By inspection | [TD-513](../../guides/TECHNICAL_DEBT.md#td-513-startup_sweep-basename-matching-and-unchecked-filesystem-calls) |
| Throwing filesystem calls; whole-file release (§2) | Crash on permission errors; stranded segment claims | By inspection | [TD-513](../../guides/TECHNICAL_DEBT.md#td-513-startup_sweep-basename-matching-and-unchecked-filesystem-calls) |
| Download naming by basename; segment release | Already tracked | — | TD-337, TD-333 |
