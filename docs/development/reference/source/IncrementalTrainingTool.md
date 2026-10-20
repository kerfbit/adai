# `IncrementalTrainingTool` — Source File Reference

- **File:** [`src/IncrementalTrainingTool.cpp`](../../../../src/IncrementalTrainingTool.cpp) (1,508 lines): the `main()` of `incremental_trainer`; `@adai-version: 0.20.1`, `@adai-reviewed: 2026-10-19`
- **Built into:** the `incremental_trainer` executable (with `IncrementalTrainer.cpp`, `ChatbotTrainer.cpp`, `Config.cpp`, `TrainingMetricsService.cpp`, `IncrementalTrainerArgs.cpp`; the admin API when `BUILD_TRAINER_ADMIN`)
- **Related:** [IncrementalTrainer.md](IncrementalTrainer.md) (the class every command drives), [IncrementalTrainerArgs.md](IncrementalTrainerArgs.md) (flag parsing), [incremental-trainer-quick-reference.md](../../guides/quick-reference/incremental-trainer-quick-reference.md) (operator guide)
- **Tests:** none for this file itself (the parser and `IncrementalTrainer` have their own)
- **Last traced against the code:** 2026-10-19

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
| `train` | Pre-fork pending check (`count_acquirable()` on this model's kind pool, TD-413), then (child) `IncrementalTrainer(vocab, model, cfg)`, `begin_run(false)`, `startup_sweep()`, `acquire_pending(run_id)`, `resolve_pass_files()` (FTP `fetch_all()` whenever the registry advertises an FTP host, into `DOWNLOAD_DIR` or `<SESSION_DIR>/downloads`, TD-411), `train_on_files()` with segment ranges, then `mark_trained()` with the per-file pair counts (TD-334) or `release_pending()` of the exact units (TD-333), and remove the downloads and their checksum sidecars |
| `train --objective=lejepa` | `run_lejepa_training_pass()`: probes the vocab size, builds a `LeJEPAEncoder`, resumes from `<SESSION_DIR>/world_model`, uses a fresh Adam optimizer, `derive_run_id()` (never MNS), acquires and downloads like `train` (through `resolve_pass_files()` since October 19, 2026; it used to download only when `DOWNLOAD_DIR` was set), feeds both sides of every pair as unpaired texts to `train_step()`, auto-saves on the `AUTO_SAVE_*` cadence, pushes metrics to a **stable** key `lejepa-<model>`, then marks trained (with pair counts) or releases |
| `retrain` | Pre-fork count, then `reset_model_for_config()` (fresh weights, history cleared), `begin_run(true)`, and trains on **all `trained_files()` + newly acquired pending files** (`retrain_on_files()`); marks only the new pending ones trained |
| `resume` | `IncrementalTrainer` + optional per-pass `TrainerAdminAPI` on `127.0.0.1:<admin-port>` (TD-172), `begin_run(false)`, `startup_sweep()`, `resume_last_session(resolve_pass_files)`. Exits **0** trained, **3** nothing pending, **1** failed (TD-523; it used to exit 1 for both idle and failed) |
| `reset` | Prints what it will do, asks `Continue? [y/N]` unless `--yes`, then `reset_all(keep_data)`: on a local registry it works on this binary's kind pool (TD-545); `--keep-data` keeps the queue and re-queues the trained entries, server-side against `registry_server` (TD-540) |
| `status`, `history` | Construct an `IncrementalTrainer` (loading the best checkpoint) just to print history and the registry |
| `init` | Constructs an `IncrementalTrainer` and prints the config; it **overrides** auto-save to 30 min / 1,000 samples (as `retrain` also does), ignoring the config file |

---

## 3. Verified problems

### Fixed: `resume` never downloaded over FTP (TD-411, October 18, 2026)

`trainer_service` runs `resume` every pass. `resume_last_session()` used the registry's own paths
(`resp.registry_paths()`), assuming a local registry. With a remote `registry_server` handing out FTP
tokens, those are the **registry host's** paths, so the always-on service couldn't read its files.
Now `train`, `retrain`, `resume` and the LeJEPA pass all use `resolve_pass_files()`, which downloads
whenever the registry advertises an FTP host. That follows the owner's rule: remote services first,
local paths only as a fallback. **Still open:** `retrain` reads its already-trained files by registry
path (`TODO: See TD-411`), so a retrain against a remote FTP registry can't read them.

### Fixed: a pass that loaded nothing "succeeded" (TD-412, October 18, 2026)

`train_on_files()` on a nonexistent path used to log `Total samples loaded: 0`, "train" for the
requested epochs and return `true`, and the caller marked the file trained. Combined with the FTP gap,
every file the always-on service claimed would have been marked trained untrained. Now a pass that
loads (or tokenizes) zero samples fails, and its files are released.

### Fixed: standalone `train` couldn't see the kind sub-pool it trains from (TD-413, October 19, 2026)

The pre-fork check (and `retrain`'s count) read `SESSION_DIR/pending_files.txt`. But this binary
always sets `DATASET_KIND` (TD-204), so `LocalTransport` keeps the pool in
`SESSION_DIR/<kind>/pending_files.txt` (TD-202). After `dataset_manager --kind chatbot add d.jsonl`,
`train 1` printed "No pending data". Both counts now go through `DatasetRegistry::count_acquirable()`
in either mode, and that sequence trains the file.

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
  "share" is the whole file. (Since TD-337 the file is at least downloaded only once.)
- **`retrain` retrains a partly trained segmented file whole** (TD-546): `trained_files()` lists each
  trained path once, and its still-pending segments are acquired and trained again in the same pass.
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
| `resume` (every `trainer_service` pass) never downloads over FTP; it uses registry-host paths (§3) | The always-on service can't train from a remote FTP registry | **Fixed October 18, 2026** (shared `resolve_pass_files()`; the LeJEPA pass joined it October 19); `retrain`'s already-trained files still use registry paths | [TD-411](../../guides/TECHNICAL_DEBT.md#td-411-resume-every-trainer_service-pass-never-downloads-over-ftp) |
| A pass with zero loaded samples returns success, so its files are marked trained (§3) | Missing or unreadable files, and every remote-FTP `resume` pass, are recorded as trained with no training | **Fixed October 18, 2026** (zero loaded or tokenized samples fail the pass) | [TD-412](../../guides/TECHNICAL_DEBT.md#td-412-a-training-pass-that-loads-zero-samples-marks-its-files-trained) |
| Local-registry `train`/`retrain` pre-check reads the root `pending_files.txt`, not `SESSION_DIR/<kind>/` (§3) | Standalone training refused to start ("No pending data") although the pool had data | **Fixed October 19, 2026** (`count_acquirable()`) | [TD-413](../../guides/TECHNICAL_DEBT.md#td-413-standalone-trainretrain-pre-check-reads-the-wrong-pending-pool) |
| First-run vocab bootstrap unreachable from the CLI; `train`/`status`/`history` abort without a vocab (§3) | A fresh install crashes until a vocab is built by hand; undocumented | **Verified** | [TD-414](../../guides/TECHNICAL_DEBT.md#td-414-first-run-vocabulary-bootstrap-is-unreachable-from-the-cli) |
| `train`/`retrain` epoch argument parsed with an uncaught `std::stoi` (§3) | A typo aborts the process | **Verified** | [TD-416](../../guides/TECHNICAL_DEBT.md#td-416-incremental_trainer-command-hygiene) |
| LeJEPA ignores segment ranges; LeJEPA claims are stuck after a crash or SIGTERM (§4) | Segmented files trained N× in full; claimed files stuck until released by hand | By inspection | [TD-415](../../guides/TECHNICAL_DEBT.md#td-415-lejepa-pass-ignores-segment-ranges-and-strands-claims-after-a-crash) |
| Signal handler calls `std::exit(0)` (no release/checkpoint, exit 0); `init`/`retrain` override `AUTO_SAVE_*`; LeJEPA tokenizer ASCII-only; read-only commands run destructive cleanup (§4) | Unclean stops; ignored config; unexpected deletions from `status` | By inspection | [TD-416](../../guides/TECHNICAL_DEBT.md#td-416-incremental_trainer-command-hygiene) |
| `retrain` retrains a partly trained segmented file whole plus its pending segments (§4) | Some pairs trained twice in one retrain | By inspection | [TD-546](../../guides/TECHNICAL_DEBT.md#td-546-retrain-trains-a-partly-trained-segmented-file-twice) |
| No SIGHUP handler, PID-based run ids | Already tracked | — | TD-283, TD-404 (bare-path release/mark, TD-333, and zero sample counts, TD-334, fixed October 19, 2026) |

## Addendum (October 19, 2026): pre-check through the registry (TD-413)

`train`'s pre-fork pending check and `retrain`'s file count now use `DatasetRegistry` in both local and remote mode, so they read the kind sub-pool the pass acquires from. They count with `DatasetRegistry::count_acquirable()`: unassigned entries plus those assigned to `MODEL_NAME`. Checked with the real binaries: a file queued with `dataset_manager --kind chatbot add` is now trained by `incremental_trainer train` instead of being reported as "No pending data".
