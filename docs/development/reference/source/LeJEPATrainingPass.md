# `LeJEPATrainingPass` and `TrainingPassFiles` — Source File Reference

- **Files:** [`src/LeJEPATrainingPass.hpp`](../../../../src/LeJEPATrainingPass.hpp) (45 lines), [`src/LeJEPATrainingPass.cpp`](../../../../src/LeJEPATrainingPass.cpp) (721), [`src/TrainingPassFiles.hpp`](../../../../src/TrainingPassFiles.hpp) (32), [`src/TrainingPassFiles.cpp`](../../../../src/TrainingPassFiles.cpp) (83)
- **Namespace:** `adai`
- **Built into:** `incremental_trainer` (with [IncrementalTrainingTool.cpp](IncrementalTrainingTool.md), which calls them) and `lejepaTrainingPassTests`
- **Status tags:** all four `stable` 1.0.0, reviewed 2026-11-12, except `LeJEPATrainingPass.cpp` 1.1.1, reviewed 2026-11-22 (training on the device, TD-588; a TD-589 tag; its traffic line at info level, TD-590) (moved out of `IncrementalTrainingTool.cpp` that day for TD-566, to `beta` the same day, and to `stable` once their tests landed and the LeJEPA pass ran on ai-machine; [TD-572](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-572-lejepatrainingpass-and-trainingpassfiles-what-stands-between-them-and-stable))
- **Tests:** [`tests/lejepa_training_pass_test.cpp`](../../../../tests/lejepa_training_pass_test.cpp) → `lejepaTrainingPassTests` (10, plus a device pass in GPU builds, TD-588/TD-589; against a local registry and a tiny world model: a whole pass, a failed batch on a working GPU, a lost GPU, an epoch that trains nothing, a second pass resuming and finding nothing pending, a re-queued file training again, a checkpoint that can't load, a pause, a checkpoint request); [`tests/training_pass_files_test.cpp`](../../../../tests/training_pass_files_test.cpp) → `trainingPassFilesTests` (5, `TrainingPassFiles` against a loopback `FtpDataServer`)
- **Last traced against the code:** 2026-11-12; device training 2026-11-22 (TD-588, TD-589)

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
| A real device loss has never reached this pass: 220 passes on ai-machine's Arc B60 caused no engine reset (the chatbot pass's workload does, about once in five passes) | The loss branch is tested with an injected check, and its probe, `GPUManager::device_usable()`, is the chatbot pass's, proved on every one of its losses | Accepted at promotion ([TD-572](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-572-lejepatrainingpass-and-trainingpassfiles-what-stands-between-them-and-stable)) |
| The pass has no validation split | Its loss is the training predictor loss only | By design (TD-183) |
| The pass trained on the host, re-tokenized every epoch, and read its diagnostics through host-matrix hooks | — | **Fixed November 22, 2026**: TD-588, [TD-589](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-589-the-lejepa-pass-re-tokenizes-every-epoch-and-reads-diagnostics-on-the-host) |

## Addendum (November 22, 2026): the pass trains on the device (TD-588)

After registering the Adam optimizer, a GPU build with the GPU in use (`GPUManager::is_available()`) calls
`world_model.gpu_begin_training()`, so every batch trains on the device ([LeJEPAEncoder.md](LeJEPAEncoder.md)). Both saves, the
periodic/requested/pause checkpoint and the end-of-pass one, call `sync_host_from_device()` first (a no-op on the host path), so the
checkpoint holds the trained weights and Adam moments. TD-566's device-loss handling is unchanged. Until TD-589, the activation
saturation and attention entropy are reported as -1 on the device path: their hooks take host matrices, and the device layers don't
call them. Test: `WithAGpuThePassTrainsOnTheDeviceAndCheckpoints` (the log shows device training and no host groups; the pass
checkpoints and records its file), on the oneAPI CPU device.

## Addendum (November 22, 2026): tokenized once, diagnostics on the device (TD-589)

The pass calls `world_model.prepare_texts(texts)` once, then `train_prepared(first, last)` per batch, so each text is tokenized once
per pass (and, on the device, its IDs uploaded once). On the device path, when something reports them (a metrics client, or the new
`LeJEPAPassHooks::epoch_diagnostics`), the encoder blocks accumulate activation saturation and attention entropy into four device
floats (`set_gpu_diagnostics()`), zeroed each epoch and downloaded once at its end; they replace the host hooks' sums there, so the
values are measured instead of -1. The entropy is the per-head average, as chatbot GPU training reports it. `epoch_diagnostics` is
called after every epoch with the values pushed to the metrics server. Tests: `AnEpochReportsItsDiagnostics` (host) and the device pass
test, now checking its diagnostics too.

## Addendum (November 22, 2026): the pass logs its host↔device traffic (TD-590)

In a GPU build the pass logs, at info level once it ends, what it moved between host and device (`adai::gpu::transfer_stats()` since its
start): uploads and their bytes, downloads and their bytes, and host round trips. It was at debug level at first, so the first ai-machine
run of the GPU test package didn't record it; at info level `run_gpu_tests.sh`'s LeJEPA passes carry it in their logs.
