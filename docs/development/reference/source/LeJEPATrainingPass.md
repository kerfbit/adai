# `LeJEPATrainingPass` and `TrainingPassFiles` — Source File Reference

- **Files:** [`src/LeJEPATrainingPass.hpp`](../../../../src/LeJEPATrainingPass.hpp) (42 lines), [`src/LeJEPATrainingPass.cpp`](../../../../src/LeJEPATrainingPass.cpp) (678), [`src/TrainingPassFiles.hpp`](../../../../src/TrainingPassFiles.hpp) (32), [`src/TrainingPassFiles.cpp`](../../../../src/TrainingPassFiles.cpp) (83)
- **Namespace:** `adai`
- **Built into:** `incremental_trainer` (with [IncrementalTrainingTool.cpp](IncrementalTrainingTool.md), which calls them) and `lejepaTrainingPassTests`
- **Status tags:** all four `beta` 0.1.0, reviewed 2026-11-12 (moved out of `IncrementalTrainingTool.cpp` that day for TD-566, and from `experimental` to `beta` the same day; what `stable` needs is [TD-572](../../guides/TECHNICAL_DEBT.md#td-572-lejepatrainingpass-and-trainingpassfiles-what-stands-between-them-and-stable))
- **Tests:** [`tests/lejepa_training_pass_test.cpp`](../../../../tests/lejepa_training_pass_test.cpp) → `lejepaTrainingPassTests` (4: a whole pass against a local registry and a tiny world model; a failed batch on a working GPU; a lost GPU; an epoch that trains nothing)
- **Last traced against the code:** 2026-11-12

> Code-traced reference. The code was moved unchanged from `IncrementalTrainingTool.cpp` on November 12, 2026, then given TD-566's GPU-loss
> handling; the history of the LeJEPA pass before that is in [IncrementalTrainingTool.md](IncrementalTrainingTool.md) §5.

---

## 1. `TrainingPassFiles`: claimed files to local files

- **`resolve_pass_files(resp, dcfg, ranges, downloaded)`** (TD-411), used by every pass (chatbot `train`/`retrain`/`resume` and LeJEPA): when the
  registry advertises an FTP host, downloads the claimed files with `DataTransport::fetch_all()` into `DOWNLOAD_DIR` (or `<SESSION_DIR>/downloads`,
  with a warning); otherwise uses the registry's own paths. `ranges` gets one entry per claimed unit, in order, with its TD-205 segment range;
  `downloaded` lists the files to delete. A failed download returns false, and the caller releases the files.
- **`cleanup_downloads(paths)`:** removes each download and its TD-338 checksum sidecar; a path listed twice (segments of one file) is fine.

---

## 2. `run_lejepa_training_pass(svc, vocab, world_model_dir, epochs, control = nullptr, hooks = {})`

One `--objective=lejepa` pass (`train` runs `epochs`, `resume` `NUM_EPOCHS`). Returns 0 trained, 3 nothing pending, 1 failed.

1. Load the vocabulary (`TOKENIZER_MODE`, TD-416) to size a `LeJEPAEncoder`. A missing vocabulary is an exception (the caller exits 1): this
   pass doesn't build one.
2. Resume from the checkpoint `LeJEPAEncoder::resolve_checkpoint_dir()` finds (the pre-TD-548 `<SESSION_DIR>/world_model` too); a failed load
   returns 1 before claiming anything.
3. A fresh Adam optimizer; the run id is `persistent_run_id(RUN_ID, world_model_dir)` (TD-415), so a restarted pass reclaims its files.
4. Sweep the download directory and acquire. An unreachable registry returns 1; an empty pool returns 3.
5. Download, and load each unit's segment rows (TD-415). Both sides of every pair become unpaired texts; no texts → release, return 1.
6. Push metrics to the stable key `lejepa-<model>`, retrying a 409 twice.
7. Walk the texts in batches of `WORLD_MODEL_SIGREG_BATCH_SIZE` through `train_batch()` (TD-503), noting progress per batch (TD-569):
   - **Checkpoints:** `AUTO_SAVE_*` (live from the admin API) and `/admin/checkpoint` write one; a failed periodic save is logged and skipped.
   - **Pause:** `/admin/pause` or a SIGTERM checkpoints, releases the files and returns 1.
   - **A batch that throws** (TD-566): the device is checked (`hooks.device_usable`, else `GPUManager::device_usable()` when a GPU is in use;
     always usable in a CPU build). A working device: the batch is skipped, as before. A lost one (a driver engine reset): the pass stops at
     once.
   - **An epoch in which no batch trained** (TD-566) stops the pass.
8. A lost device or an empty epoch: no checkpoint (after a lost device the weights can't be trusted), the files released, the metrics session
   ended, return 1, so `trainer_service` starts a fresh process.
9. Otherwise the final `world_model.save()`, then `mark_trained()` with the pair counts; a failed save releases the files and returns 1
   (TD-571).

`LeJEPAPassHooks` exists for the tests: `device_usable` answers the device check, `before_batch(i)` runs before each batch and can throw to make
it fail as a GPU error would. Both are empty in production.

---

## 3. Problems

### Fixed: a lost GPU failed every remaining batch (TD-566, November 12, 2026)

The chatbot pass has stopped on a lost GPU since November 10, but this pass caught every batch's exception, logged "Skipping batch" and went
on. After a driver engine reset each later batch failed the same way. The pass then "completed" on the batches before the reset: it saved,
recorded the files as trained and exited 0. Every one of those quick failures also counted as progress for TD-569's stall check. Now the first
failure on an unusable device ends the pass with nothing saved and the files back in the pool, and an epoch that trained nothing fails it too.
`ALostGpuStopsThePassAndReleasesItsFiles` and `AnEpochThatTrainsNothingFailsThePass` fail with either check removed.

### Earlier fixes to the pass (while it was in `IncrementalTrainingTool.cpp`)

The final save before marking the files trained (TD-571), batches with SIGReg across them (TD-503), the checkpoint directory (TD-548), segment
ranges and a persistent run id (TD-415), the per-pass admin API under `trainer_service` (TD-403), `TOKENIZER_MODE` (TD-416), and progress per
batch (TD-569). A batch that throws, `TokenizerEncodingError` included, has been skipped rather than ending the pass since TD-503. See
[IncrementalTrainingTool.md](IncrementalTrainingTool.md) §3 and §5.

---

## 4. Known gaps

| Item | Impact | Tracked as |
|---|---|---|
| The device-loss path is tested with an injected check; a real loss can't be produced on demand, and the pass hasn't run on ai-machine since the move | The real `GPUManager::device_usable()` probe is the chatbot pass's, which ai-machine's soak exercised (TD-566) | [TD-572](../../guides/TECHNICAL_DEBT.md#td-572-lejepatrainingpass-and-trainingpassfiles-what-stands-between-them-and-stable) |
| `TrainingPassFiles` has no direct test; only its local-file branch runs, through `lejepaTrainingPassTests` | The FTP branch and `cleanup_downloads()` are untested here (`DataTransport` has its own tests) | TD-572 |
| `lejepaTrainingPassTests` doesn't cover nothing pending, pause, a checkpoint request, resuming or a failed load | Those paths rest on reading and the earlier real-binary checks | TD-572 |
| The pass has no validation split | Its loss is the training predictor loss only | By design (TD-183) |
