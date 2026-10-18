# `IncrementalTrainer` — Source File Reference

- **Files:** [`src/IncrementalTrainer.hpp`](../../../../src/IncrementalTrainer.hpp) (440 lines), [`src/IncrementalTrainer.cpp`](../../../../src/IncrementalTrainer.cpp) (2,128 lines)
- **Namespace:** global (`IncrementalConfig`, `TrainingSession`, `IncrementalTrainer`)
- **Built into:** the `incremental_trainer` executable (compiled with `ChatbotTrainer.cpp`, `Config.cpp`, `TrainingMetricsService.cpp`), not a library
- **Status tag:** `@adai-status: beta` ("capped by TD-039"), `@adai-version: 0.10.1`, `@adai-reviewed: 2026-09-21`
- **Tests:** `incrementaltrainer_test.cpp` (46), `incremental_trainer_registry_test.cpp` (14), `incremental_trainer_background_test.cpp` (8), `incremental_trainer_control_test.cpp` (7)
- **Operator guide:** [incremental-trainer-quick-reference.md](../../guides/quick-reference/incremental-trainer-quick-reference.md)
- **Last traced against the code:** 2026-10-13

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
  ├─ pick best_checkpoint_path = lowest final_validation_loss among sane sessions → load it
  ├─ if session_<current_session_id>_best.bin exists and is newer → load that instead
  └─ cleanup_dead_sessions()
begin_run(is_retrain)                           MNS set_training → run_id; registry next_session
train_on_files / retrain_on_files / resume_last_session
  ├─ initialize_session()                       append in-progress entry (val loss 0)
  ├─ load pairs (OpenMP across files), vocab bootstrap on first run
  ├─ tokenized-cache key = hash(file checksums + ranges, vocab checksum, tokenizer mode, max_seq)
  ├─ fresh ChatbotTrainer(config.base_config)   (fresh optimizer every pass, TD-269)
  └─ run_training()
       ├─ metrics push session (/session/start; 409 → retry with a suffix ×3; otherwise NullMetricsReporter)
       ├─ callbacks: per epoch (history, MNS push_progress), per sample (auto-save, forced checkpoint),
       │             best model (session_<id>_best.bin)
       ├─ trainer.train(); model = trainer.release_model()
       ├─ paused (/admin/pause) → perform_auto_save(); return false (files released by the caller)
       └─ success → save session_<id>_checkpoint.bin (+ .metrics.csv), finalize_session(),
                    save history, MNS set_candidate
```

`make_incremental_config()` copies the architecture, optimizer, clipping, batch, metrics, MNS,
generation-quality, auto-save/retention, tokenizer, dataset and world-model/hippocampal fields. It
**hard-codes** `enable_early_stopping = true`, `patience = 5` and `restore_best_weights = true` (not
configurable), and the config-file constructor forces `LRSchedule::WARMUP_COSINE`.
`IncrementalConfig::save_incremental_checkpoints` and `checkpoint_dir` are never read.

`retrain` calls `reset_model_for_config()`, which rebuilds fresh weights **and clears and saves an
empty `session_history`**. Every earlier checkpoint stays on disk but is no longer referenced.

---

## 3. Checkpoint selection and cleanup (verified)

### Each pass starts from the best-ever checkpoint, not the latest

The constructor (all three constructors, and `resume_last_session()` too) loads the session with the
**lowest `final_validation_loss` across the whole history**. But each pass's validation loss is
measured on a validation split of **that pass's own files**, so the numbers from different passes
aren't comparable. When a later pass trains on new data whose validation loss comes out higher, the
next pass starts from the older checkpoint again, and the later pass's learning is discarded.

Verified: a history with session 0 (val 1.20) followed by session 1 (val 2.00, newer) logged `Best
checkpoint: …/session_0_checkpoint.bin (val loss: 1.2)` and loaded session 0. In an incremental setting
that streams new datasets (daily_dialog, then simple_wikipedia, …), the model can keep returning to the
same old checkpoint while the registry marks the newer files as trained.

### A paused or crashed pass's work is deleted on the next start

`perform_auto_save()` (sample-cadence auto-save, forced `/admin/checkpoint`, and the pause path) saves
`auto_save_session_N.bin` and persists session N to the history with `final_validation_loss = 0`. On
the next construction:
1. `current_session_id` becomes N+1, so the in-progress snapshot check looks for
   `session_<N+1>_best.bin`, not session N's.
2. `cleanup_dead_sessions()` treats session N's files as "orphaned … from a crashed/superseded
   session".
3. It drops session N from the history (val loss 0 isn't "sane") and deletes its checkpoint.

Verified with a history holding session 0 (complete) and session 1 (paused after an auto-save, with
`auto_save_session_1` and `session_1_best` on disk): the restart loaded session 0, logged `Removing
broken session 1`, and deleted the auto-save. `session_1_best` survived but **is never loaded**. So
TD-173's "pause drains the pass and checkpoints it" saves a checkpoint that the next pass deletes,
and the claimed files the pause released are retrained from the older weights.

### Checkpoint deletion removes 0.2% of the bytes

`EncoderDecoderModel::save_model()` writes the top-level sidecars (`.config`, `.vocab`, `.encoder`,
`.decoder`, `.lm_head`) **plus per-layer files** (`_token_emb.bin`, `_encoder_block_<i>.bin`,
`_attention.bin`/`_feedforward.bin`, `_decoder_block_<i>.bin` + `.self_attn`/`.cross_attn`/`.ff`,
`_final_norm.bin`), and those per-layer files hold almost all the weights. `remove_model_files()`
deletes only the five top-level sidecars plus `.metrics.csv`. Measured on a d_model 256, 4 + 4 layer
checkpoint: **29.6 MB total, 0.1 MB (0.2%) deleted, 29.5 MB left behind** each time a checkpoint is
"removed" by:
- retention (`cleanup_old_sessions()`, `MAX_SESSIONS_TO_KEEP`);
- superseded best snapshots (`finalize_session()`);
- broken-session cleanup.

`cleanup_dead_sessions()` and `reset_all()` scan for `*.bin` files expecting a bare base file that
`save_model()` never writes. They match per-layer files by accident (deleting some layers of an
auto-save and keeping its `.config`), and leave the `.self_attn`/`.cross_attn`/`.ff` files. Disk use
in `SESSION_DIR` therefore grows with every pass (checkpoint + best snapshot + auto-saves), and
`retrain`/`reset` leave everything from before.

---

## 4. Other members

| Member | Behaviour |
|---|---|
| `bootstrap_vocab(pairs)` | First run only: builds the BPE vocab (auto-sized via `recommend_vocab_size()` unless `VOCAB_BUILD_SIZE`), saves it (unchecked, TD-221), and logs fertility warnings |
| `compute_tokenized_cache_key()` | `std::hash` of the files' content checksums and segment ranges, the vocab checksum, the tokenizer mode and `max_seq_length`. **It doesn't include a preprocessing-format version**, so a code change to how `ChatbotTrainer` tokenizes (for example fixing TD-361's double `BOS` or TD-267's skipped pairs) would silently reuse stale cached token IDs. The cache would have to be cleared by hand after such fixes |
| `begin_run(is_retrain)` | MNS `set_training()` → `run_id`, registry `next_session()` → session id; both best-effort |
| `resume_last_session()` | Acquires pending files, loads the in-progress best or the best-ever checkpoint, trains, then marks trained or releases (by bare path, TD-333) |
| `should_auto_save()` / `perform_auto_save()` | By samples or minutes (live-tunable under `trainer_service`); saves mid-epoch through the live `ChatbotTrainer`. The saves are never used for resuming (§3) |
| `finalize_session()` | Records results with **`epochs_completed = num_epochs` requested** (early stopping, which is forced on, can stop sooner), removes the session's best snapshot (§3: only its top-level files), updates symlinks, and runs retention |
| `update_best_checkpoint()` | **Returns immediately when symlinks are disabled**, so `best_validation_loss`/`best_checkpoint_path` aren't updated in-process in that mode (they're re-derived on the next construction) |
| `cleanup_old_sessions()` | Keeps the newest `MAX_SESSIONS_TO_KEEP` history entries; re-picks the best and the symlink when the best is removed |
| `reset_all(keep_data)` | Backs up `MODEL_PATH`, removes history, pending file, symlinks and `session_*`/`auto_save_*` `.bin` files (§3: not the top-level sidecars or attention/FF files), optionally re-queues trained files (local registry file only), and rebuilds fresh |
| `print_training_summary()` | Console summary with loss sparklines |
| Symlinks | `latest_checkpoint.bin` / `best_checkpoint.bin` in the **working directory**, pointing at the bare checkpoint path, which `save_model()` never creates. **They always dangle**, and loading through one looks for `best_checkpoint.bin.config` (not a link), so neither can be used as a `MODEL_PATH`. On Windows the `copy_file()` fallback copies the same nonexistent bare file and always fails |

---

## 5. Tests

The 75 tests across four files cover:
- config mapping;
- session history v2 parsing (including paths with spaces);
- sanity checks;
- retention and best re-selection;
- symlinks;
- auto-save cadence and mid-epoch saves;
- registry integration (run ids, release/mark);
- background-pass and control/pause behaviour.

**Not covered:**
- resume selection across passes trained on different data;
- the pause → restart → cleanup sequence (§3);
- the bytes actually removed by `remove_model_files()`;
- tokenized-cache invalidation after code changes.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Every pass resumes from the lowest validation loss across all history, though each pass validates on different data (§3) | Later passes' learning is discarded whenever new data validates worse; the model can stay pinned to an old checkpoint | **Verified** | [TD-405](../../guides/TECHNICAL_DEBT.md#td-405-each-training-pass-restarts-from-the-best-ever-checkpoint-not-the-latest) |
| A paused or crashed pass's auto-save is deleted and its session dropped on the next start; its in-progress best snapshot is never found (§3) | `/admin/pause` and crash-recovery checkpoints are lost; work is redone from older weights | **Verified** | [TD-406](../../guides/TECHNICAL_DEBT.md#td-406-a-paused-or-crashed-passs-checkpoint-is-deleted-on-the-next-start) |
| `remove_model_files()` deletes only the top-level sidecars (0.2% of a checkpoint); cleanup/reset scan for bare `.bin` names that don't exist (§3) | `SESSION_DIR` grows by roughly a full checkpoint per rotation, snapshot removal and retrain; retention doesn't bound disk use | **Verified** (29.5 of 29.6 MB left) | [TD-407](../../guides/TECHNICAL_DEBT.md#td-407-checkpoint-deletion-leaks-the-per-layer-weight-files) |
| Tokenized-data cache key has no preprocessing version (§4) | Tokenization fixes silently reuse stale cached IDs | By inspection; **fixed in code October 17, 2026** (`ChatbotTrainer::kPreprocessFormatVersion` in the key) | [TD-409](../../guides/TECHNICAL_DEBT.md#td-409-tokenized-data-cache-key-has-no-preprocessing-version) |
| `latest`/`best` checkpoint symlinks always dangle and can't be loaded (§4) | The advertised "best checkpoint" link is unusable | By inspection | [TD-408](../../guides/TECHNICAL_DEBT.md#td-408-latestbest-checkpoint-symlinks-always-dangle) |
| `finalize_session()` records requested, not completed, epochs; early stopping/patience/restore-best and WARMUP_COSINE hard-coded; `update_best_checkpoint()` skips tracking without symlinks; two config fields unused; `retrain` orphans every earlier checkpoint (§2, §4) | Misleading history; untunable training policy | By inspection | [TD-410](../../guides/TECHNICAL_DEBT.md#td-410-incrementaltrainer-session-bookkeeping-and-hard-coded-training-policy) |
| Fresh optimizer each pass; bare-path release/mark; zero sample counts; unchecked vocab save | Already tracked | — | TD-269, TD-333, TD-334, TD-221 |

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
