# `IncrementalTrainer` — Source File Reference

- **Files:** [`src/IncrementalTrainer.hpp`](../../../../src/IncrementalTrainer.hpp) (541 lines), [`src/IncrementalTrainer.cpp`](../../../../src/IncrementalTrainer.cpp) (2,507 lines)
- **Namespace:** global (`IncrementalConfig`, `TrainingSession`, `IncrementalTrainer`)
- **Built into:** the `incremental_trainer` executable (with `ChatbotTrainer.cpp` and the core libraries), not a library
- **Status tags:** `@adai-status: beta` ("capped by TD-039"); `IncrementalTrainer.cpp` `@adai-version: 0.24.1` (reviewed 2026-11-11), `.hpp` `0.21.0` (2026-11-10)
- **Tests:** `incrementaltrainerTests` (66), `incrementalTrainerDecouplingTests` (14, `incremental_trainer_registry_test.cpp`), `incrementalTrainerBackgroundTests` (8), `incrementalTrainerControlTests` (7)
- **Operator guide:** [incremental-trainer-quick-reference.md](../../guides/quick-reference/incremental-trainer-quick-reference.md)
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **verified** was confirmed by constructing the real class against scratch session directories holding
> real checkpoints, or by running `incremental_trainer`.

---

## 1. What this file is

The **session and checkpoint manager around `ChatbotTrainer`**. `incremental_trainer`'s `train`/`retrain`/`resume`/`reset` commands, and through
them every `trainer_service` pass, build one. It:

- maps `ServiceConfig` to `IncrementalConfig` (`make_incremental_config()`);
- builds the vocabulary on the first run and the model (the one `EncoderDecoderModel` construction point), attaching a frozen LeJEPA world model
  and hippocampal memory when configured;
- keeps `session_history.txt` and the checkpoints in `SESSION_DIR`, and decides which checkpoint each pass starts from;
- runs a pass: loads the claimed files, drives a fresh `ChatbotTrainer`, and wires the metrics push session, MNS run, progress and candidate
  calls, auto-save, the admin API's pause and checkpoint requests, and the finish.

It decides what the model keeps across passes: every pass's starting weights, and whether its results are kept, come from here.

---

## 2. Lifecycle of a pass

```text
IncrementalTrainer(vocab, model, cfg)
  ├─ no vocabulary yet → defer the model until the first pass builds one (TD-414); else build_model() (+ maybe_attach_world_model())
  ├─ load_session_history()                     current_session_id = max id + 1
  ├─ init_from_session_history()
  │    ├─ take_interrupted_session()            an interim auto_save_session_N entry → its newest snapshot; id back to N (TD-406)
  │    ├─ best tracking: lowest real validation loss among sane sessions
  │    └─ load select_resume_checkpoint()       most recent completed session (TD-405); RESUME_FROM_BEST_CHECKPOINT: lowest loss
  └─ cleanup_dead_sessions()                    remnants, orphaned snapshots, unreferenced checkpoints (TD-407)
begin_run(is_retrain)                           MNS set_training() → run_id (new run on retrain); registry next_session()
train_on_files / retrain_on_files / resume_last_session
  ├─ load_training_pairs()                      per unit with its segment range; per-unit pair counts (TD-334); nothing loaded → fail (TD-412)
  ├─ first run: bootstrap_vocab() then build_model()
  ├─ initialize_session()                       append an in-progress entry
  ├─ tokenized-cache key (file checksums + ranges, vocab checksum, mode, max_seq, kPreprocessFormatVersion, validation_split)
  ├─ fresh ChatbotTrainer(config.base_config); set_run_id(run_id); add the pairs with their metadata (TD-273)
  ├─ train_on_files: load_training_state(<loaded checkpoint>.optimizer) (TD-269)
  └─ run_training()
       ├─ metrics push session (409 → retry with a suffix ×3; otherwise NullMetricsReporter)
       ├─ callbacks: per epoch (session history, MNS push_progress, admin epoch), per sample (admin progress, forced
       │             checkpoint, auto-save), best model (session_<id>_best.bin + .optimizer)
       ├─ trainer.train(); model = trainer.release_model()
       ├─ paused (/admin/pause or SIGTERM, TD-416) → perform_auto_save(); return false (the caller releases the files)
       └─ success → save session_<id>_checkpoint.bin (a failed save fails the pass, TD-221) and its .optimizer, .metrics.csv;
                    finalize_session(); save history; MNS set_candidate; end the metrics session last (TD-442)
```

`resume_last_session(resolver)` returns `Trained`, `NothingPending` or `Failed` (TD-523). It acquires from the registry (unreachable → `Failed`),
hands the batch to the caller's `PassFileResolver` (the FTP download step, TD-411), trains from the interrupted snapshot or the selected
checkpoint, then marks the exact units trained with their counts, or releases them (TD-333/TD-334).

`retrain` (`reset_model_for_config()`) archives the previous checkpoints and history to `<SESSION_DIR>/archive/<UTC time>/` (TD-410), then
rebuilds fresh weights and forgets the loaded training state.

`make_incremental_config()` copies the architecture, optimizer, clipping, dropout, batch, metrics, MNS, generation-quality and backfill,
auto-save and retention, tokenizer, dataset, world-model and hippocampal fields, `RESUME_FROM_BEST_CHECKPOINT`, and the TD-410 policy (`SEED` isn't mapped: `incremental_trainer` applies it with
`set_init_seed()` before anything is built):
`EARLY_STOPPING_ENABLED` (true), `EARLY_STOPPING_PATIENCE` (5), `RESTORE_BEST_WEIGHTS` (true), `LR_SCHEDULE` (`linear_warmup`, else `constant`;
no decaying schedule, since passes continue one run). `best_model_dir` is the session directory (TD-274).

---

## 3. Checkpoint selection and clean-up

- **`select_resume_checkpoint()`** (TD-405): the most recent completed session with a loadable checkpoint (all its sidecars); with
  `RESUME_FROM_BEST_CHECKPOINT=true`, the lowest real validation loss (only meaningful with a fixed validation set). An interrupted pass's
  newest snapshot outranks both (TD-406). Before TD-539 the constructors tested the bare checkpoint path, which `save_model()` never writes, so
  no CLI pass ever loaded a checkpoint.
- **Sessions without validation** record `TrainingSession::kNoValidationLoss` (−1): resumable, but never chosen as the lowest loss (TD-541). 0
  marks an in-progress session, NaN a diverged one.
- **Whole-checkpoint deletion** (TD-407): `checkpoint_files(base)` lists every file named `<base>.…`/`<base>_…` (sidecars, per-layer files,
  `.metrics.csv`, `.optimizer`); `remove_model_files()` deletes them all. `cleanup_dead_sessions()` groups a directory's files by checkpoint and
  removes remnants without a `.config`, orphaned best/auto-save snapshots and unreferenced checkpoints.
- **Links** (TD-408): `latest_checkpoint.bin` / `best_checkpoint.bin` in the working directory are one link per checkpoint file, so the link name
  loads like the checkpoint (usable as `MODEL_PATH`); copies on Windows.

---

## 4. Other members

| Member | Behaviour |
|---|---|
| `bootstrap_vocab(pairs)` | First run: builds a vocabulary from the pairs' texts and history (sized by `recommend_vocab_size()` unless `VOCAB_BUILD_SIZE`), saves it (a failed save returns false), logs the letter case, whitespace and fertility. The model is then built from the **saved file** |
| `compute_tokenized_cache_key()` | `std::hash` of the units' content checksums and ranges, the vocabulary checksum, mode, `max_seq_length`, `kPreprocessFormatVersion` (TD-409) and `validation_split` (TD-271) |
| `begin_run(is_retrain)` | MNS `set_training()` → run id, registry `next_session()`; both best-effort |
| `last_pass_pair_counts()` | Pairs per unit of the last pass, for `mark_trained()` (TD-334) |
| `should_auto_save()` / `perform_auto_save()` | By samples or minutes (live under `trainer_service`); saves mid-epoch through the live `ChatbotTrainer`, training state included; notes progress after each attempt (TD-569) |
| `finalize_session()` | Records the epochs actually run (`ChatbotTrainer::get_epochs_run()`, TD-410), the losses, removes the session's best snapshot, updates the links and runs retention |
| `is_sane_checkpoint_candidate()` / `has_validation_loss()` | Sane: samples trained, a real loss or `kNoValidationLoss`, all sidecars present |
| `update_best_checkpoint()` | Best tracking, separate from the links since TD-410 |
| `cleanup_old_sessions()` | Keeps the newest `MAX_SESSIONS_TO_KEEP` entries; re-picks the best when it's removed |
| `reset_all(keep_data)` | Backs up `MODEL_PATH`'s files to `.bak`, removes history, links and every session checkpoint; on a local registry works on `local_pool_dir()` (`SESSION_DIR/<kind>/`, TD-545); without `--keep-data` clears that pool's queue and records, with it re-queues the trained entries (`requeue_trained()`, server-side against `registry_server`, TD-540); then rebuilds fresh |
| `read_session_history(dir)` (static) | Parses `session_history.txt` without a trainer; `status`/`history` use it (TD-416) |
| `print_training_summary()` | Console summary with loss sparklines; `val=n/a` without validation |

---

## 5. Problems

### Fixed November 11, 2026: two problems found by the re-trace

- **The vocabulary was reloaded from a file the tokenizer misread** (TD-570). `bootstrap_vocab()` saves the vocabulary and `build_model()` loads
  it back, as every later pass and the server do, and the reader took `#` comments and header keywords inside the token and merge sections
  ([BPETokenizer.md](BPETokenizer.md) §8): `#` tokens never reached the model, and a vocabulary with a token starting with a header keyword
  would have failed every pass.
- **The epoch early stopping ended on was missing** from the session's per-epoch lists, MNS progress and the admin API's epoch, because
  `ChatbotTrainer::train()` broke out before its epoch callback (TD-571, [ChatbotTrainer.md](ChatbotTrainer.md) §10).

---

## 6. Tests

The four suites (95) cover:

- config mapping (TD-410's settings included); session history parsing (paths with spaces); sanity checks; retention and best re-selection;
- most-recent vs lowest-loss resume (TD-405); the interrupted-session resume (TD-406); whole-checkpoint deletion and remnant reclaim (TD-407);
  loadable links (TD-408); unvalidated passes kept (TD-541);
- auto-save cadence and mid-epoch saves; training state continuing into the next pass (TD-269); epochs run recorded; `retrain` archiving
  (TD-410);
- registry integration: run ids, exact release and marking, resolver use (TD-411), `ResumeResult` (TD-523), per-unit pair counts (TD-334),
  `reset_all()` on the kind pool (TD-545), empty passes failing (TD-412);
- the first-run vocabulary (TD-414); background passes; the control state and pause.

**Not covered:** tokenized-cache invalidation across a version bump end to end; `reset_all()` against `registry_server`; a vocabulary with `#` or
keyword tokens end to end (the tokenizer's own tests cover the reader).

---

## 7. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| The vocabulary reloaded from a file the tokenizer misread; the early-stopping epoch missing from the per-epoch data (§5) | — | **Fixed November 11, 2026** | [TD-570](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-570-bpetokenizerload_vocab-takes-header-syntax-inside-the-token-and-merge-sections), [TD-571](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-571-findings-from-the-november-11-2026-trace-of-the-files-td-569-touched) |
| Resume from the best-ever checkpoint; interrupted passes lost; leaky deletion; dangling links; the cache key; checkpoints never loaded; the wrong reset pool | — | **Fixed** October 17–19, 2026 | TD-405–TD-409, TD-539, TD-545 |
| Bookkeeping and hard-coded policy | — | **Fixed November 9, 2026** | TD-410 |
| Fresh optimizer per pass; bare-path release and marking; zero counts; unchecked vocabulary save; the vocabulary bootstrap | — | **Fixed** | TD-269, TD-333, TD-334, TD-221, TD-414 |
| The whole class below `stable` | Still changing shape (it changed again today) | — | TD-039 |

## 8. History

- **October 17–19, 2026:** TD-405–TD-409, TD-539, TD-541, TD-545, TD-269, TD-333, TD-334, TD-411, TD-412, TD-523.
- **October 25, 2026:** the metrics session ends last (TD-442).
- **November 2, 2026:** `read_session_history()` and SIGTERM drains (TD-416); the first-run vocabulary (TD-414).
- **November 4, 2026:** pairs keep their metadata, quality switches mapped (TD-273); the best-weights directory (TD-274).
- **November 9, 2026:** epochs run, policy settings, `retrain` archive (TD-410).
- **November 11, 2026:** progress per checkpoint attempt (TD-569). This document was re-traced against the code, replacing its October sections,
  its merge notes and its addenda; its two findings were fixed the same day (TD-570, TD-571).
