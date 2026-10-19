# `IncrementalTrainingTool` — Source File Reference

- **File:** [`src/IncrementalTrainingTool.cpp`](../../../../src/IncrementalTrainingTool.cpp) (1,477 lines): the `main()` of `incremental_trainer`
- **Built into:** the `incremental_trainer` executable (with `IncrementalTrainer.cpp`, `ChatbotTrainer.cpp`, `Config.cpp`, `TrainingMetricsService.cpp`, `IncrementalTrainerArgs.cpp`; the admin API when `BUILD_TRAINER_ADMIN`)
- **Related:** [IncrementalTrainer.md](IncrementalTrainer.md) (the class every command drives), [IncrementalTrainerArgs.md](IncrementalTrainerArgs.md) (flag parsing), [incremental-trainer-quick-reference.md](../../guides/quick-reference/incremental-trainer-quick-reference.md) (operator guide)
- **Tests:** none for this file itself (the parser and `IncrementalTrainer` have their own)
- **Last traced against the code:** 2026-10-13

> Code-traced reference. Behaviour marked **verified** was confirmed by running the built
> `build/debug/bin/incremental_trainer` and `dataset_manager`, or by driving `IncrementalTrainer`
> from a scratch program.

---

## 1. What this file is

The **`incremental_trainer` command-line program**:

```text
incremental_trainer [--config F] [--model N] [--gpu-strategy S] [--foreground] [--admin-port P]
                    [--objective=chatbot|lejepa] [--dataset-kind K]  <command> [args]
  init [vocab] [model]   build the trainer once (prints the architecture)
  train [epochs]         acquire pending files → train (chatbot, or LeJEPA world model with --objective=lejepa)
  retrain [epochs]       fresh weights; train on every trained file + newly acquired pending files
  reset [--yes] [--keep-data]   wipe sessions/checkpoints (confirmation prompt)
  resume                 resume_last_session(); the command trainer_service runs every pass
  status / history       print the summary/history and the registry
```

Startup:
1. Parse flags and load `config.trainer.conf`.
2. Derive `DATASET_KIND` from the objective (TD-204).
3. Apply `--gpu-strategy`.
4. Initialize the GPU now, unless the command forks (`train`/`retrain`/`resume`, where the child
   initializes it).
5. With MNS configured, resolve the model name (`--model` > `MODEL_NAME` > an **interactive pick on
   stdin**), then take `model_path`, `run_group`, the architecture and the world-model link from MNS,
   falling back to local config.

`train`/`retrain`/`resume` go through `run_training_pipeline()`: `fork()` + `setsid()` with stdio to
`/dev/null` (Windows: re-launch detached), the parent prints a banner and exits, and the child
initializes the logger, installs SIGTERM/SIGINT handlers, initializes the GPU and runs the work.
`--foreground` (used by `trainer_service`/systemd) skips the fork.

---

## 2. Commands

| Command | What it does |
|---|---|
| `train` | Pre-fork pending check, then (child) `IncrementalTrainer(vocab, model, cfg)`, `begin_run(false)`, `startup_sweep()`, `acquire_pending(run_id)`, FTP `fetch_all()` when the registry advertises an FTP host, `train_on_files()` with segment ranges, then `mark_trained()` or `release_pending()`, and remove downloads |
| `train --objective=lejepa` | `run_lejepa_training_pass()`: probes the vocab size, builds a `LeJEPAEncoder`, resumes from `<SESSION_DIR>/world_model`, uses a fresh Adam optimizer, `derive_run_id()` (never MNS), acquires and downloads like `train`, feeds both sides of every pair as unpaired texts to `train_step()`, auto-saves on the `AUTO_SAVE_*` cadence, pushes metrics to a **stable** key `lejepa-<model>`, then marks trained or releases |
| `retrain` | Pre-fork count, then `reset_model_for_config()` (fresh weights, history cleared), `begin_run(true)`, and trains on **all `trained_files()` + newly acquired pending files** (`retrain_on_files()`); marks only the new pending ones trained |
| `resume` | `IncrementalTrainer` + optional per-pass `TrainerAdminAPI` on `127.0.0.1:<admin-port>` (TD-172), `begin_run(false)`, `resume_last_session()`; returns 1 when there was nothing to do |
| `reset` | Prints what it will do, asks `Continue? [y/N]` unless `--yes`, then `reset_all(keep_data)` |
| `status`, `history` | Construct an `IncrementalTrainer` (loading the best checkpoint) just to print history and the registry |
| `init` | Constructs an `IncrementalTrainer` and prints the config; it **overrides** auto-save to 30 min / 1,000 samples (as `retrain` also does), ignoring the config file |

---

## 3. Verified problems

### `resume` never downloads over FTP, so `trainer_service` can't train from a remote FTP registry

`trainer_service` runs `incremental_trainer --foreground --admin-port N resume` for every pass, and
`resume` goes to `IncrementalTrainer::resume_last_session()`. That function acquires pending files but
uses `resp.registry_paths()` directly, with the comment *"resume_last_session runs on the same machine
as the registry … ftp_server_host is always empty here"*. That's only true for a local registry. With
`REGISTRY_SERVER_URL` pointing at a remote `registry_server` that hands out FTP tokens, the paths are
the **registry host's** paths, and `train`'s `DataTransport::fetch_all()` step is skipped. The files
can't be opened on the trainer (by inspection), which leads straight into the next problem.

### A pass that loads nothing "succeeds", so the caller marks its files trained

`ChatbotTrainer::train()` has no empty-data check. Verified with `IncrementalTrainer::train_on_files()`
on a nonexistent path: it logged `Cannot open file`, `Total samples loaded: 0`, `Training for 2
epochs...`, `Incremental training session completed successfully`, and **returned `true`**. Every
caller (`train`, `retrain`, `resume`) then calls `mark_trained()`. Together with the FTP gap above,
every file the always-on service claims from a remote FTP registry would be marked trained without
ever being trained on. The same happens for any file that's missing, unreadable, or contains no valid
pairs.

### Standalone (local-registry) `train` can't see the kind sub-pool it trains from

The pre-fork check (and `retrain`'s count) reads `SESSION_DIR/pending_files.txt` when
`REGISTRY_SERVER_URL` is empty. But `incremental_trainer` always sets `DATASET_KIND`
(`chatbot`/`world_model`, TD-204), so `LocalTransport` keeps the real pool in
`SESSION_DIR/<kind>/pending_files.txt` (TD-202). Verified: after `dataset_manager --kind chatbot add
d.jsonl` (`sess/chatbot/pending_files.txt` has 1 file), `incremental_trainer train 1` printed `No
pending data. Use DatasetManager to queue training data.` Queuing without `--kind` passes the check
but puts the file where the child's acquire won't look.

### The first-ever `train`, and `status`/`history`, abort without a vocabulary

Every command constructs `IncrementalTrainer(vocab, model, cfg)`, the **three-argument** constructor,
which calls `build_model()` → `load_vocab()` unconditionally. The first-run vocabulary bootstrap
(`pending_vocab_build_`) only exists in the *config-file* constructor, which `main()` never uses.
Verified on a fresh `SESSION_DIR` with data queued and no `vocab.txt`: `train 1` aborted with
`terminate called … VocabularyFileError: Could not open vocabulary file: vocab.txt` (exit 134), and so
did `status` and `history`. In practice a vocabulary must be built separately (`vocab_builder`) before
the first run, which the usage text doesn't say. That fits the "missing `VOCAB_PATH`" crash recorded
during ai-machine's first deployment.

### Non-numeric epoch counts abort

`train abc` / `retrain abc`: `std::stoi(args[1])` throws outside any `try`, giving
`terminate called … what(): stoi`, exit 134 (verified; same class as TD-401).

---

## 4. Other issues (by inspection)

- **LeJEPA ignores segment ranges.** `run_lejepa_training_pass()` calls `load_conversation_pairs(path,
  pairs)` without the TD-205 range (the chatbot path passes `file_ranges()`), so a file queued as N
  segments is loaded **in full** N times when its segments are claimed together, and each segment's
  "share" is the whole file.
- **LeJEPA claims can't be reclaimed after a crash.** Its run id is always `derive_run_id()` (hostname
  prefix + PID tail, TD-404), and claimed files are only reclaimable by the same run id. After a crash,
  kill, or SIGTERM mid-pass, the files stay claimed by a run id no future process will have, until
  someone releases them by hand.
- **SIGTERM/SIGINT handler** logs through spdlog and calls `std::exit(0)` from the signal handler
  (neither is async-signal-safe; the same pattern as TD-249). It runs static destructors while the
  training thread is still running, doesn't checkpoint or release claimed files, and exits **0**.
- **`init`/`retrain` hard-code auto-save** (30 min / 1,000 samples) over `AUTO_SAVE_*`.
- **LeJEPA's tokenizer** is always ASCII (`TOKENIZER_MODE` isn't threaded through).
- **Wasted work.** `status`/`history` construct a full `IncrementalTrainer`, which builds the model,
  loads the best checkpoint and runs `cleanup_dead_sessions()`, a deleting cleanup (TD-406/TD-407),
  just to print history. A read-only command can therefore delete files.
- **The interactive MNS model pick** reads stdin; under systemd it gets EOF and the command exits with
  "No model selected", which is the intended failure but undocumented.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `resume` (every `trainer_service` pass) never downloads over FTP; it uses registry-host paths (§3) | The always-on service can't train from a remote FTP registry | By inspection (definite); **fixed in code October 18, 2026** (shared `resolve_pass_files()` download step; `retrain`'s already-trained files still use registry paths) | [TD-411](../../guides/TECHNICAL_DEBT.md#td-411-resume-every-trainer_service-pass-never-downloads-over-ftp) |
| A pass with zero loaded samples returns success, so its files are marked trained (§3) | Missing or unreadable files, and every remote-FTP `resume` pass, are recorded as trained with no training | **Verified**; **fixed in code October 18, 2026** (zero loaded or tokenized samples fail the pass) | [TD-412](../../guides/TECHNICAL_DEBT.md#td-412-a-training-pass-that-loads-zero-samples-marks-its-files-trained) |
| Local-registry `train`/`retrain` pre-check reads the root `pending_files.txt`, not `SESSION_DIR/<kind>/` (§3) | Standalone training refuses to start ("No pending data") although the pool has data | **Verified** | [TD-413](../../guides/TECHNICAL_DEBT.md#td-413-standalone-trainretrain-pre-check-reads-the-wrong-pending-pool) |
| First-run vocab bootstrap unreachable from the CLI; `train`/`status`/`history` abort without a vocab (§3) | A fresh install crashes until a vocab is built by hand; undocumented | **Verified** | [TD-414](../../guides/TECHNICAL_DEBT.md#td-414-first-run-vocabulary-bootstrap-is-unreachable-from-the-cli) |
| `train`/`retrain` epoch argument parsed with an uncaught `std::stoi` (§3) | A typo aborts the process | **Verified** | [TD-416](../../guides/TECHNICAL_DEBT.md#td-416-incremental_trainer-command-hygiene) |
| LeJEPA ignores segment ranges; LeJEPA claims are stuck after a crash or SIGTERM (§4) | Segmented files trained N× in full; claimed files stuck until released by hand | By inspection | [TD-415](../../guides/TECHNICAL_DEBT.md#td-415-lejepa-pass-ignores-segment-ranges-and-strands-claims-after-a-crash) |
| Signal handler calls `std::exit(0)` (no release/checkpoint, exit 0); `init`/`retrain` override `AUTO_SAVE_*`; LeJEPA tokenizer ASCII-only; read-only commands run destructive cleanup (§4) | Unclean stops; ignored config; unexpected deletions from `status` | By inspection | [TD-416](../../guides/TECHNICAL_DEBT.md#td-416-incremental_trainer-command-hygiene) |
| Bare-path release/mark, zero sample counts, no SIGHUP handler, PID-based run ids | Already tracked | — | TD-333, TD-334, TD-283, TD-404 |
