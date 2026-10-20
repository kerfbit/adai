# `IncrementalTrainer` — Source File Reference

- **Files:** [`src/IncrementalTrainer.hpp`](../../../../src/IncrementalTrainer.hpp) (526 lines), [`src/IncrementalTrainer.cpp`](../../../../src/IncrementalTrainer.cpp) (2,392 lines)
- **Namespace:** global (`IncrementalConfig`, `TrainingSession`, `IncrementalTrainer`)
- **Built into:** the `incremental_trainer` executable (compiled with `ChatbotTrainer.cpp`, `Config.cpp`, `TrainingMetricsService.cpp`), not a library
- **Status tag:** `@adai-status: beta` ("capped by TD-039"), `IncrementalTrainer.cpp` `@adai-version: 0.17.1`, `IncrementalTrainer.hpp` `0.17.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** `incrementaltrainer_test.cpp` (58), `incremental_trainer_registry_test.cpp` (14), `incremental_trainer_background_test.cpp` (8), `incremental_trainer_control_test.cpp` (7)
- **Operator guide:** [incremental-trainer-quick-reference.md](../../guides/quick-reference/incremental-trainer-quick-reference.md)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed by constructing the real class
> (compiled from these sources) against scratch session directories holding real checkpoints written by
> `EncoderDecoderModel::save_model()`.

---

## 1. What this file is

The **session and checkpoint manager around `ChatbotTrainer`**. `incremental_trainer`'s
`train`/`retrain`/`resume`/`reset` commands, and through them every `trainer_service` pass, construct
one of these. It:
- maps `ServiceConfig` to `IncrementalConfig` (`make_incremental_config()`);
- owns the vocabulary bootstrap and builds the model (the single `EncoderDecoderModel` construction
  point), attaching a frozen LeJEPA world model and hippocampal memory when configured;
- keeps `session_history.txt` and the checkpoint files in `SESSION_DIR`;
- decides **which checkpoint each pass starts from**;
- runs a pass: loads the claimed files, builds a fresh `ChatbotTrainer`, wires the metrics push
  session, MNS run/progress/candidate calls, auto-save, admin pause/checkpoint, and finalizes.

### Why it matters

It decides what the model actually learns from across passes. Every incremental pass's starting
weights, and whether its results are kept, come from here.

---

## 2. Lifecycle of a pass

```text
IncrementalTrainer(vocab, model, cfg)          constructor
  ├─ build_model()  (+ maybe_attach_world_model())
  ├─ load_session_history()                     current_session_id = max id + 1
  ├─ init_from_session_history()
  │    ├─ take_interrupted_session()            TD-406: last entry still an auto_save_session_N
  │    │                                        → newest of its auto-save / session_N_best; id back to N
  │    ├─ best tracking: lowest final_validation_loss among sane sessions with a real loss
  │    └─ load select_resume_checkpoint()       TD-405: most recent completed session
  │                                             (RESUME_FROM_BEST_CHECKPOINT=true: lowest loss);
  │                                             the interrupted snapshot outranks both
  └─ cleanup_dead_sessions()                    §3
begin_run(is_retrain)                           MNS set_training → run_id; registry next_session
train_on_files / retrain_on_files / resume_last_session
  ├─ load_training_pairs()                      per-file counts kept for mark_trained (TD-334);
  │                                             nothing loaded → fail, files released (TD-412)
  ├─ vocab bootstrap on first run
  ├─ initialize_session()                       append in-progress entry (val loss 0)
  ├─ tokenized-cache key = hash(file checksums + ranges, vocab checksum, tokenizer mode, max_seq,
  │                              ChatbotTrainer::kPreprocessFormatVersion)   (TD-409)
  ├─ fresh ChatbotTrainer(config.base_config); set_run_id(run_id)
  ├─ train_on_files only: load_training_state(<loaded checkpoint>.optimizer)   (TD-269)
  └─ run_training()
       ├─ metrics push session (/session/start; 409 → retry with a suffix ×3; otherwise NullMetricsReporter)
       ├─ callbacks: per epoch (history, MNS push_progress), per sample (auto-save, forced checkpoint),
       │             best model (session_<id>_best.bin + its .optimizer)
       ├─ trainer.train(); model = trainer.release_model()
       ├─ paused (/admin/pause) → perform_auto_save(); return false (files released by the caller)
       └─ success → save session_<id>_checkpoint.bin; a failed save fails the pass (TD-221)
                    → save_training_state(<checkpoint>.optimizer), .metrics.csv, finalize_session(),
                      save history, MNS set_candidate
```

`resume_last_session()` returns `ResumeResult::Trained`, `NothingPending` or `Failed` (TD-523). It
acquires from the registry (an unreachable registry is `Failed`), hands the batch to the caller's
`PassFileResolver` (the FTP download step, TD-411; with no resolver, an FTP registry fails the pass),
loads the interrupted snapshot, the in-progress best or the selected checkpoint, trains, then
marks the exact units trained with their pair counts, or releases them (TD-333/TD-334).

`make_incremental_config()` copies the architecture, optimizer, clipping, batch, metrics, MNS,
generation-quality, auto-save/retention, tokenizer, dataset and world-model/hippocampal fields, and
`RESUME_FROM_BEST_CHECKPOINT` (TD-405). It **hard-codes** `enable_early_stopping = true`, `patience = 5`
and `restore_best_weights = true` (not configurable), and the config-file constructor uses
`LRSchedule::LINEAR_WARMUP`. That is the TD-269 owner decision: warm up once per run, then hold the
base LR. It replaced `WARMUP_COSINE`. `IncrementalConfig::save_incremental_checkpoints` and
`checkpoint_dir` are never read.

`retrain` calls `reset_model_for_config()`, which rebuilds fresh weights, forgets the loaded
checkpoint's training state, and clears and saves an empty `session_history`. Every earlier
checkpoint stays on disk but is no longer referenced; nothing deletes it until a `reset`.

---

## 3. Checkpoint selection and cleanup (verified)

All three problems below were fixed October 18, 2026 and are kept here as history.

### Fixed: each pass started from the best-ever checkpoint, not the latest (TD-405)

The constructors and `resume_last_session()` loaded the session with the **lowest
`final_validation_loss` across the whole history**. But each pass validates on a split of its own
files, so those numbers aren't comparable, and a later pass whose new data validated worse sent the
next pass back to an older checkpoint. Verified: with session 0 (val 1.20) followed by session 1 (val
2.00, newer), the old code loaded session 0. Now `select_resume_checkpoint()` continues from the most
recent completed session with a loadable checkpoint. `RESUME_FROM_BEST_CHECKPOINT=true` restores the
old rule, which only makes sense with a fixed validation set. Best tracking and the `best` link still
use the lowest loss. `ResumesFromMostRecentSessionNotLowestValidationLoss` covers both modes.

TD-539, found alongside: the constructors tested for the bare checkpoint path, which `save_model()`
never writes, so they never loaded any checkpoint, and `train`/`retrain` always started from fresh
weights. `select_resume_checkpoint()` only returns checkpoints whose five sidecars exist.

### Fixed: a paused or crashed pass's work was deleted on the next start (TD-406)

`perform_auto_save()` (sample-cadence auto-save, `/admin/checkpoint`, and the pause path) saves
`auto_save_session_N.bin` and records session N in the history with validation loss 0. The next start
used to:
1. look for `session_<N+1>_best.bin`;
2. call session N's files orphaned;
3. drop session N and delete its auto-save.

So a paused pass's checkpoint was thrown away, and its released files were retrained from older
weights. Now `take_interrupted_session()` recognizes the interim entry by its auto-save checkpoint
path. It resumes from the newer of the auto-save and `session_N_best`, and sets `current_session_id`
back to N, so cleanup keeps those snapshots. `InterruptedSessionResumesFromItsNewestSnapshot` covers
it.

### Fixed: checkpoint deletion removed 0.2% of the bytes (TD-407)

`save_model()` writes five top-level sidecars plus the per-layer files that hold almost all the
weights. `remove_model_files()` deleted only the sidecars and `.metrics.csv`: on a d_model 256,
4 + 4 layer checkpoint, 0.1 of 29.6 MB. So retention, superseded best snapshots and broken-session
cleanup each left nearly a full checkpoint behind. `cleanup_dead_sessions()` and `reset_all()`
scanned for bare `*.bin` names that don't exist.

Now:
- **`checkpoint_files(base)`** lists every file named `<base>.…` or `<base>_…` (the bare `<base>` too,
  if present). That covers the sidecars, per-layer files, `.metrics.csv` and TD-269's `.optimizer`,
  whatever components a checkpoint has. `remove_model_files()` deletes them all.
- **Grouping:** `session_checkpoint_base_of()` groups a directory's files by checkpoint.
  `cleanup_dead_sessions()` then removes, per base:
  - a remnant without a `.config` (which also reclaims what the old deletion left behind);
  - an orphaned best/auto-save snapshot;
  - an unreferenced checkpoint.

  `reset_all()` removes every file that belongs to a session checkpoint.
- **Tests:** `RemoveModelFilesDeletesTheWholeCheckpointOnly` and
  `CleanupDeadSessionsReclaimsCheckpointRemnants`.

---

## 4. Other members

| Member | Behaviour |
|---|---|
| `bootstrap_vocab(pairs)` | First run only: builds the BPE vocab (auto-sized via `recommend_vocab_size()` unless `VOCAB_BUILD_SIZE`), saves it (a failed save returns false, TD-221), and logs fertility warnings. Still unreachable from the CLI (TD-414) |
| `compute_tokenized_cache_key()` | `std::hash` of the files' content checksums and segment ranges, the vocab checksum, the tokenizer mode, `max_seq_length`, and `ChatbotTrainer::kPreprocessFormatVersion`. That last one has been in the key since TD-409, so a change to `ChatbotTrainer`'s tokenization (bumped for TD-531/532/275 and TD-361) invalidates old caches |
| `begin_run(is_retrain)` | MNS `set_training()` → `run_id`, registry `next_session()` → session id; both best-effort |
| `resume_last_session(resolver)` | §2; returns `ResumeResult` (TD-523) |
| `last_pass_pair_counts()` | TD-334: pairs loaded per file of the last `train_on_files()`/`retrain_on_files()` call, for `mark_trained()` |
| `should_auto_save()` / `perform_auto_save()` | By samples or minutes (live-tunable under `trainer_service`); saves mid-epoch through the live `ChatbotTrainer`, including its training state. An interrupted pass resumes from these (TD-406) |
| `finalize_session()` | Records results with **`epochs_completed = num_epochs` requested** (early stopping, which is forced on, can stop sooner), removes the session's best snapshot (all its files since TD-407), updates the links, and runs retention. A pass with no validation data records `TrainingSession::kNoValidationLoss` (-1) instead of 0, so its checkpoint stays resumable (TD-541) |
| `is_sane_checkpoint_candidate()` / `TrainingSession::has_validation_loss()` | Sane: samples trained, a real validation loss or `kNoValidationLoss`, all five sidecars present. Only sessions with a real loss take part in lowest-loss selection (TD-541) |
| `update_best_checkpoint()` | **Returns immediately when symlinks are disabled**, so `best_validation_loss`/`best_checkpoint_path` aren't updated in-process in that mode (they're re-derived on the next construction). A pass without a measured loss becomes "best" only when there is no best yet, and never records -1 as the loss to beat |
| `cleanup_old_sessions()` | Keeps the newest `MAX_SESSIONS_TO_KEEP` history entries; re-picks the best and the link when the best is removed |
| `reset_all(keep_data)` | Backs up every `MODEL_PATH` file to `<model>.bak…`, removes the history, the links and every session checkpoint file. Against a **local** registry it works on the trainer's own pool, `DatasetRegistry::local_pool_dir()` = `SESSION_DIR/<kind>/` (TD-545, October 19, 2026; it used the legacy root pool). Without `--keep-data` it removes that pool's queue and trained records. With `--keep-data` it keeps the queue and re-queues every trained entry (`requeue_trained()`, server-side against `registry_server`, TD-540). Then it rebuilds fresh |
| `print_training_summary()` | Console summary with loss sparklines; a session without validation shows `val=n/a` |
| Links | `latest_checkpoint.bin` / `best_checkpoint.bin` in the **working directory**. Since TD-408 (October 18, 2026) each is a set of links, one per checkpoint file (`latest_checkpoint.bin.config` → `…/session_N_checkpoint.bin.config`, and so on), so the link name loads like the checkpoint itself and can be used as `MODEL_PATH`. They used to be one link to a bare path that never exists. On Windows they are copies |

---

## 5. Tests

The 87 tests across four files cover:
- config mapping;
- session history v2 parsing (including paths with spaces);
- sanity checks;
- retention and best re-selection;
- symlinks;
- auto-save cadence and mid-epoch saves;
- registry integration (run ids, release/mark);
- background-pass and control/pause behaviour.

Added October 17–19, 2026:
- most-recent vs lowest-loss resume (TD-405);
- the interrupted-session resume (TD-406);
- whole-checkpoint deletion and remnant reclaim (TD-407);
- loadable links (TD-408);
- empty passes failing (TD-412);
- training state continuing into the next pass (TD-269);
- resolver use and release (TD-411);
- `ResumeResult` (TD-523);
- unvalidated passes kept (TD-541);
- per-file pair counts (TD-334);
- `reset_all()` on the kind pool (TD-545).

**Not covered:** tokenized-cache invalidation end to end (the key includes the version; no test
rebuilds a cache across a version bump), `reset_all()` against `registry_server`.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Every pass resumes from the lowest validation loss across all history, though each pass validates on different data (§3) | Later passes' learning is discarded whenever new data validates worse; the model can stay pinned to an old checkpoint | **Fixed October 18, 2026** (most recent session by default; `RESUME_FROM_BEST_CHECKPOINT` keeps the old rule) | [TD-405](../../guides/TECHNICAL_DEBT.md#td-405-each-training-pass-restarts-from-the-best-ever-checkpoint-not-the-latest) |
| A paused or crashed pass's auto-save is deleted and its session dropped on the next start; its in-progress best snapshot is never found (§3) | `/admin/pause` and crash-recovery checkpoints are lost; work is redone from older weights | **Fixed October 18, 2026** (resumes from the interrupted session's newest snapshot) | [TD-406](../../guides/TECHNICAL_DEBT.md#td-406-a-paused-or-crashed-passs-checkpoint-is-deleted-on-the-next-start) |
| `remove_model_files()` deletes only the top-level sidecars (0.2% of a checkpoint); cleanup/reset scan for bare `.bin` names that don't exist (§3) | `SESSION_DIR` grows by roughly a full checkpoint per rotation, snapshot removal and retrain; retention doesn't bound disk use | **Fixed October 18, 2026** (was verified: 29.5 of 29.6 MB left) | [TD-407](../../guides/TECHNICAL_DEBT.md#td-407-checkpoint-deletion-leaks-the-per-layer-weight-files) |
| Tokenized-data cache key has no preprocessing version (§4) | Tokenization fixes silently reuse stale cached IDs | **Fixed October 17, 2026** (`ChatbotTrainer::kPreprocessFormatVersion` in the key) | [TD-409](../../guides/TECHNICAL_DEBT.md#td-409-tokenized-data-cache-key-has-no-preprocessing-version) |
| Constructors only loaded a checkpoint if the bare path existed, which `save_model()` never writes, so they never loaded one; `train`/`retrain` started from fresh weights | Every CLI pass discarded all earlier training | **Fixed October 18, 2026** (was verified) | [TD-539](../../guides/TECHNICAL_DEBT.md#td-539-incrementaltrainer-constructors-never-load-a-checkpoint) |
| `latest`/`best` checkpoint symlinks always dangle and can't be loaded (§4) | The advertised "best checkpoint" link was unusable | **Fixed October 18, 2026** (one link per file) | [TD-408](../../guides/TECHNICAL_DEBT.md#td-408-latestbest-checkpoint-symlinks-always-dangle) |
| `finalize_session()` records requested, not completed, epochs; early stopping/patience/restore-best and the LR schedule (now `LINEAR_WARMUP`) hard-coded; `update_best_checkpoint()` skips tracking without symlinks; two config fields unused; `retrain` orphans every earlier checkpoint (§2, §4) | Misleading history; untunable training policy | By inspection | [TD-410](../../guides/TECHNICAL_DEBT.md#td-410-incrementaltrainer-session-bookkeeping-and-hard-coded-training-policy) |
| `reset_all()` removed the legacy root pool's queue and registry instead of the trainer's `SESSION_DIR/<kind>/` pool, so a local `reset` left files marked trained; once targeted, `--keep-data` would also have dropped the untrained queue (§4) | `reset` on a local registry didn't reset training data, and deleted another pool's queue | **Fixed October 19, 2026** (was verified: legacy queue deleted, kind pool untouched) | [TD-545](../../guides/TECHNICAL_DEBT.md#td-545-incremental_trainer-reset-works-on-the-legacy-pool-not-the-trainers-own) |
| Fresh optimizer each pass; bare-path release/mark; zero sample counts; unchecked vocab save | Fixed (TD-269 October 19; TD-333 October 19; TD-334 October 19; TD-221 October 17) | — | TD-269, TD-333, TD-334, TD-221 |

---

## 7. Merge notes

This file replaces `docs/development/guides/internals/incremental-trainer-internals.md` (1,267 lines),
which described an earlier API: `add_new_data()`, `add_gutenberg_book(s)()`, `train_incremental()`,
`train_on_new_data_only()`, `print_data_registry()` and a `DataVersion` structure. **None of these
exist any more.** Data now comes from `DatasetRegistry` claims (`train_on_files()`), and Gutenberg
fetching lives in `DataFetcher`/`registry_server`. The old doc's claim that `retrain` starts from
random weights is still true (`reset_model_for_config()`). Its file-layout diagram (session history,
checkpoints, auto-saves, symlinks) is kept and corrected in §2–3. The operator-facing
[quick reference](../../guides/quick-reference/incremental-trainer-quick-reference.md) is unchanged.

## Addendum (October 19, 2026): passes without validation data (TD-541)

A completed pass with no validation data now records `final_validation_loss = TrainingSession::kNoValidationLoss` (-1) instead of 0. `is_sane_checkpoint_candidate()` accepts it, so the next pass resumes from that checkpoint and startup cleanup keeps it. `TrainingSession::has_validation_loss()` excludes it from every lowest-loss choice: `RESUME_FROM_BEST_CHECKPOINT`, the best-checkpoint tracking, and `update_best_checkpoint()`. 0 still marks an in-progress session, and NaN a diverged one.

## Addendum (October 19, 2026): per-file pair counts (TD-334)

`load_training_pairs()` records how many pairs each file contributed, in the order the files were passed; a TD-205 segment counts only its own range. `last_pass_pair_counts()` exposes these counts. `resume_last_session()` and `incremental_trainer`'s `train`, `retrain` and LeJEPA passes hand them to `DatasetRegistry::mark_trained()`, so trained records and `dataset_manager status`'s sample total are no longer always 0.
