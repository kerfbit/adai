# `IncrementalTrainingTool` — Source File Reference

- **File:** [`src/IncrementalTrainingTool.cpp`](../../../../src/IncrementalTrainingTool.cpp) (1,145 lines): the `main()` of `incremental_trainer`
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.37.0`, `@adai-reviewed: 2026-11-12`
- **Built into:** the `incremental_trainer` executable, with `IncrementalTrainer.cpp`, `IncrementalTrainerArgs.cpp`, `LeJEPATrainingPass.cpp` and `TrainingPassFiles.cpp` (the LeJEPA pass and the shared download step, moved out of this file November 12, 2026) and the core libraries; the per-pass admin API (`TrainerAdminAPI.cpp`, `BUILD_TRAINER_ADMIN`) and metrics push when cpp-httplib is found; FTP downloads (`BUILD_FTP_TRANSPORT`) with libcurl
- **Related:** [IncrementalTrainer.md](IncrementalTrainer.md) (the class the chatbot commands drive), [LeJEPATrainingPass.md](LeJEPATrainingPass.md) (the LeJEPA pass and the download step), [IncrementalTrainerArgs.md](IncrementalTrainerArgs.md) (flag parsing and shared helpers), [TrainerServiceMain.md](TrainerServiceMain.md) (which runs `resume` as each pass), [PassHeartbeat.md](PassHeartbeat.md), [incremental-trainer-quick-reference.md](../../guides/quick-reference/incremental-trainer-quick-reference.md) (operator guide)
- **Tests:** none for this file itself. Its parser and helpers (`incrementalTrainerArgsTests`), `IncrementalTrainer` (four suites), the heartbeat (`passHeartbeatTests`), the LeJEPA pass (`lejepaTrainingPassTests`) and the real binary's exit on a bad flag and on `SIGHUP` (`sighupTests`) have their own
- **Last traced against the code:** 2026-11-12

> Code-traced reference. Behaviour marked **verified** was confirmed by running the built `build/debug/bin/incremental_trainer` (and
> `dataset_manager`, `vocab_builder`) against a scratch session directory.

---

## 1. What this file is

```text
incremental_trainer [--config F] [--model N] [--gpu-strategy S] [--foreground] [--admin-port P] [--seed N]
                    [--objective=chatbot|lejepa] [--dataset-kind K] [--log-file F] [--heartbeat-file F]  <command> [args]
  init [vocab] [model]           build the trainer once and print the architecture
  train [epochs]                 acquire pending files and train (the LeJEPA world model with --objective=lejepa)
  retrain [epochs]               fresh weights; train on every trained unit plus newly acquired pending files
  reset [--yes] [--keep-data]    wipe sessions and checkpoints (asks unless --yes)
  resume                         one pass over what is pending; what trainer_service runs as each pass
  status / history               print the session history and the registry (read-only)
```

**Start-up, every command:**

1. Parse the flags ([IncrementalTrainerArgs.md](IncrementalTrainerArgs.md)); an error prints the usage and exits 1. With `--heartbeat-file`, start
   a `PassHeartbeatWriter` at once (TD-569), so a hang even in GPU initialization shows as no progress.
2. Load `config.trainer.conf` (`--config` > `./config.trainer.conf` > `/etc/adai/…` > the legacy `config.conf`s) and run `validate_or_exit()`
   (TD-288). `--seed`/`SEED` seeds initialization (TD-368).
3. Set `DATASET_KIND` from the objective, or `--dataset-kind` (TD-202/TD-204), and apply `--gpu-strategy`.
4. Initialize the GPU now, unless the command is `train`/`retrain`/`resume`, which initialize it in the worker after any fork, once
   the pass's `PassAdminApi` exists (TD-571).
5. With `NAME_SERVICE_URL` set: resolve the model (`--model` > `MODEL_NAME` > an interactive pick, refused without a terminal, TD-416); take
   its artifact path and `run_group` from MNS; take the 6 architecture fields from MNS; and, when a world model is linked, its artifact path,
   architecture, injection and `HIPPOCAMPAL_*` settings and SIGReg sketch count (TD-196). Each lookup that fails is logged with its
   reason and falls back to local config (TD-571). These lines go through `Logger::deferred()`: on the console at once, and into the
   pass's log file when the worker opens it.
6. `chatbot_run_id`: without `RUN_ID`, `train`/`retrain`/`resume` (chatbot) take a run id kept in `SESSION_DIR/chatbot_run_id` (TD-404), so a
   restarted pass reclaims its files; an MNS run id still comes first in the passes.

**`run_training_pipeline()`** runs `train`/`retrain`/`resume`'s work. Without `--foreground` it detaches (`fork()` + `setsid()`, stdio to
`/dev/null`; Windows: re-launch with `DETACHED_PROCESS`), and the parent prints a banner and exits 0. The worker initializes the logger with
`LOG_LEVEL` and `--log-file`/`LOG_FILE_PATH` (a detached run with neither logs to `~/.local/state/adai/log/incremental_trainer.log`, TD-431),
ignores `SIGHUP` (TD-283), installs the SIGTERM/SIGINT handler and runs the work. Each pass's work builds its `PassAdminApi` first, then
initializes the GPU, then builds the trainer (TD-571), so the control state and admin API exist through GPU initialization.

**Signals** (TD-416): the handler only stores atomics. The first SIGTERM/SIGINT asks the registered pass (its `TrainerControlState`, through
`PassAdminApi`) to drain as `/admin/pause` does: stop at the next sample or batch, checkpoint, release its claimed files, exit 1. With no
pass registered yet, or on a second signal, it `_exit(1)`s.

**`PassAdminApi`** (TD-403/TD-416): every pass builds one. It always provides a `TrainerControlState` (seeded from `AUTO_SAVE_*` and
`MAX_SESSIONS_TO_KEEP`) for the signal handler; with `--admin-port` (only `trainer_service` passes it), it also runs the per-pass
`TrainerAdminAPI` on `127.0.0.1:<port>`, waiting up to 2 s for it to listen.

**`resolve_pass_files()`/`cleanup_downloads()`** (TD-411), shared by every pass, download the claimed files when the registry serves them over
FTP and remove them afterwards; they live in `TrainingPassFiles.cpp` since November 12, 2026 ([LeJEPATrainingPass.md](LeJEPATrainingPass.md)
§1).

`main()` turns an exception from any command into `❌ <message>` and exit 1 (TD-414).

---

## 2. Commands

| Command | What it does |
|---|---|
| `init` | Builds an `IncrementalTrainer(vocab, model, cfg)` (paths from the arguments, else config) and prints the session directory and architecture. Extra arguments are ignored |
| `train [epochs]` (chatbot) | Epochs via `parse_epochs_arg()`. Pre-fork: `count_acquirable()` on this model's pool (TD-413); none → exit 1. Worker: `PassAdminApi`, GPU initialization, the trainer (TD-571), `begin_run(false)` (MNS run id, else the local one), `startup_sweep()` (TD-512), `acquire_pending()`, `resolve_pass_files()`, `train_on_files()`, then `mark_trained()` with the per-file pair counts (TD-334) or `release_pending()` of the exact units (TD-333) |
| `retrain [epochs]` | Pre-fork count of trained files plus acquirable ones. Worker: `PassAdminApi`, GPU initialization, the trainer, `reset_model_for_config()`, `begin_run(true)` (a new MNS run), the sweep, `trained_units()` (TD-546) plus newly acquired pending files, `retrain_on_files()`; marks only the pending units trained (their counts are the trailing ones) or releases them. Trained units are read by their registry path, never downloaded (TD-411's open item) |
| `reset [--yes] [--keep-data]` | An unknown argument is an error, exit 1 (TD-571). Prints what it will do, asks unless `--yes`, then `reset_all(keep_data)` on this binary's pool (TD-545); `--keep-data` re-queues the trained entries, server-side against `registry_server` (TD-540) |
| `resume` (chatbot) | Worker: `PassAdminApi` (with the per-pass admin API under `trainer_service`), GPU initialization, the trainer (TD-571), `begin_run(false)`, the sweep, `resume_last_session(resolve_pass_files)`. Exits **0** trained, **3** nothing pending, **1** failed (TD-523) |
| `train`/`resume --objective=lejepa` | Needs `WORLD_MODEL_ENABLED`. `run_lejepa_training_pass()` (below); `train` runs `epochs`, `resume` `NUM_EPOCHS`, with the same exit codes |
| `status`, `history` | Read `session_history.txt` with `IncrementalTrainer::read_session_history()` without building a trainer (TD-416), and print it with the registry |

**`run_lejepa_training_pass()`** is in `LeJEPATrainingPass.cpp` since November 12, 2026 (moved to be testable, TD-566); it is described in
[LeJEPATrainingPass.md](LeJEPATrainingPass.md). The tool builds the pass's `PassAdminApi`, initializes the GPU and calls it.

---

## 3. Problems

### Fixed: the LeJEPA pass recorded its files as trained before its final save (TD-571, November 11, 2026; was verified)

(The pass has lived in `LeJEPATrainingPass.cpp` since November 12, 2026.)

`run_lejepa_training_pass()` called `mark_trained()` and only then `world_model.save()`, outside any `try`, so a failed save reached `main()`
(exit 1) with the files recorded as trained and the pass's training unwritten. With `WORLD_MODEL_ARTIFACT_PATH` under a read-only directory
the log said "Pass complete — 40 steps", the world-model pool recorded the file as trained, and no checkpoint existed. **Now** it saves first;
a failed save is logged ("Could not save the world-model checkpoint…; releasing the pass's files") and the files go back to the pool. The same
run with the fixed binary left no trained record and the claim cleared.

### Fixed in the same review (TD-571)

- **`reset` ignored a mistyped flag** and deleted the data registry ([IncrementalTrainerArgs.md](IncrementalTrainerArgs.md) §3); it now exits
  1 with `Unknown reset argument`.
- **The per-pass admin API started late**, after GPU initialization and, for the chatbot passes, the trainer's construction; until then
  `trainer_service`'s `/admin/status` answered a bare 503 and a SIGTERM ended the process at once. It now starts first, and the proxy adds
  its own fields to a 503 as well ([TrainerServiceProxy.md](TrainerServiceProxy.md)).
- **A failed MNS model lookup was silent** (`catch (...)`); it is now logged with its reason. The `[MNS]` start-up lines went to stdout before
  the logger existed; they now go through `Logger::deferred()`, which also writes them to the pass's log file.

---

## 4. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| The LeJEPA pass marked its files trained before its final save; `reset` ignored unknown tokens; the per-pass admin API started late; a silent MNS lookup failure; start-up lines on stdout (§3) | — | **Fixed November 11, 2026** | [TD-571](../../guides/TECHNICAL_DEBT.md#td-571-findings-from-the-november-11-2026-trace-of-the-files-td-569-touched) |
| `retrain` reads trained units by registry path | A retrain against a remote FTP registry can't read its trained files | By inspection | [TD-411](../../guides/TECHNICAL_DEBT.md#td-411-resume-every-trainer_service-pass-never-downloads-over-ftp) (open item) |
| The LeJEPA pass had no device-lost check | A lost GPU failed every remaining batch, and the pass then recorded its files as trained | **Fixed November 12, 2026** ([LeJEPATrainingPass.md](LeJEPATrainingPass.md)) | TD-566 |
| `resume` without FTP, zero-sample success, the wrong pending pool, the vocabulary bootstrap, LeJEPA segments and claims, signals, auto-save overrides, read-only `status`, log files | — | **Fixed** | TD-411, TD-412, TD-413, TD-414, TD-415, TD-416, TD-431 |

## 5. History

- **October 18–19, 2026:** FTP for every pass (TD-411), zero-sample passes fail (TD-412), the kind pool's pre-check (TD-413).
- **October 21, 2026:** batched LeJEPA pass (TD-503); the world-model checkpoint directory (TD-548).
- **November 1–2, 2026:** objective and kind under `trainer_service` and the LeJEPA pass's admin API (TD-403), LeJEPA segments and run id
  (TD-415), command hygiene and signals (TD-416), the first-run vocabulary and exception exits (TD-414), run ids (TD-404), log files (TD-431).
- **November 9, 2026:** `--seed` (TD-368).
- **November 12, 2026:** the LeJEPA pass and the download helpers moved to their own files, and the LeJEPA pass stops on a lost GPU (TD-566).
- **November 11, 2026:** `--heartbeat-file` and progress per LeJEPA batch (TD-569). This document was re-traced against the code, replacing its
  October sections and folding in its addenda; the findings were fixed the same day (TD-571).
