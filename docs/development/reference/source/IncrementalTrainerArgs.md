# `IncrementalTrainerArgs` — Source File Reference

- **Files:** [`src/IncrementalTrainerArgs.hpp`](../../../../src/IncrementalTrainerArgs.hpp), [`src/IncrementalTrainerArgs.cpp`](../../../../src/IncrementalTrainerArgs.cpp)
- **Namespace:** `adai`
- **Built into:** `incremental_trainer` (`IncrementalTrainingTool.cpp`) and `trainer_service` (`TrainerServiceMain.cpp`)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.6.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** [`tests/incremental_trainer_args_test.cpp`](../../../../tests/incremental_trainer_args_test.cpp) (25 tests)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed by running the built
> `build/debug/bin/incremental_trainer`.

---

## 1. What this file is

The **global command-line parser** for `incremental_trainer`, plus four small helpers that were split
out of `main()` so they can be tested:

| Function | Purpose |
|---|---|
| `parse_incremental_trainer_global_args(argc, argv)` | Pulls the global flags out of `argv` (anywhere on the line) and returns the rest as `args` (`args[0]` = the command) |
| `incremental_trainer_command_defers_gpu_init(cmd)` | True for `train`/`retrain`/`resume`, which fork, so GPU init must happen in the child (CUDA contexts aren't fork-safe) |
| `derive_run_id(configured)` | The configured `RUN_ID`, else `<first 8 chars of hostname>_<pid % 10000>` |
| `resolve_dataset_kind(objective, override)` | The TD-202/TD-204 sub-pool: `--dataset-kind` if given, else `world_model` for `lejepa` and `chatbot` for everything else (config `DATASET_KIND` deliberately ignored) |
| `parse_reset_command_args(args)` | `--yes`, `--keep-data` for `reset` |
| `kResumeExitTrained` (0), `kResumeExitNothingPending` (3), `kResumeExitFailed` (1) | TD-523 (October 19, 2026): `incremental_trainer resume`'s exit codes, shared with `trainer_service`. `resume` used to exit 1 both when idle and when it failed |
| `classify_pass_exit(code)` → `PassOutcome` | `trainer_service`'s reading of a child pass: 0 `DidWork`, 3 `Idle`, ≥ 128 `Crashed` (a signal death, see ChildProcess), anything else `Failed`, including 127 (exec failure) |
| `failure_backoff_seconds(n, base, max)` | Wait after *n* ≥ 1 consecutive failed or crashed passes: `base` doubled per extra failure, capped at `max` (the supervisor uses 45 s and 900 s) |

`trainer_service` reuses the same parser for its own command line, but only forwards `--config`,
`--model` and `--gpu-strategy` to the children it launches (plus its own `--foreground --admin-port N
resume`). So `--objective`/`--dataset-kind` given to `trainer_service` are **dropped**: the always-on
service only ever runs chatbot training.

---

## 2. The global flags

| Flag | Value | Handling |
|---|---|---|
| `--config <path>` | Config file (discovery otherwise) | Stored |
| `--gpu-strategy <s>` | `background` / `full` | Stored; an unknown value warns and falls back to `background` (in `gpu_strategy_from_string()`) |
| `--model <name>` | MNS model name | Stored |
| `--foreground` | — | Sets a flag (no fork) |
| `--admin-port <N>` | Child admin port (set by `trainer_service`) | **`std::stoi`**; see below |
| `--objective=<o>` | `chatbot` (default) / `lejepa` | **Only the `=` form**; the value is free text |
| `--dataset-kind <k>` | Sub-pool override | Stored, **not validated** |

Anything else, including a flag whose value is missing at the end of the line, is appended to `args`.

### Problems (verified)

- **`--admin-port abc` aborts the process.** `std::stoi` throws `std::invalid_argument`, and `main()`
  calls the parser outside any `try`: `terminate called … what(): stoi`, exit 134 (verified). A value
  out of `int` range throws `out_of_range` the same way. **`--admin-port 8084x` is accepted as 8084**
  (verified), and nothing range-checks the port.
- **A mistyped objective trains the chatbot.** `main()` only checks `cli.objective == "lejepa"`, and
  `resolve_dataset_kind()` maps any other value to `chatbot`. So `--objective=lejpa` was accepted
  silently (verified), and with `train` it would run chatbot training on the chatbot pool instead of
  failing. The test `UnrecognizedObjectiveFallsBackToChatbot` locks this in. The space form
  `--objective lejepa` at least fails clearly, with `Unknown command: --objective` (verified), because
  `--objective` becomes the command.
- **A flag missing its value is silently re-used as a positional.** `status --config` put `--config`
  into `args` and carried on (verified). For `train`, a trailing `--dataset-kind` would be read by the
  command as a file argument. The tests deliberately specify this (`FlagMissingItsValueFallsThroughAsPositional`
  and friends).
- **`--dataset-kind` isn't validated.** For `RemoteTransport` the value becomes a URL segment, which
  `registry_server`'s route regex rejects (404) unless it's one of the four kinds. For `LocalTransport`
  it's joined as `session_dir + "/" + kind`, so a typo silently starts a new, empty sub-pool ("no
  pending data"), and a value containing `..` points the local registry files outside `session_dir`.

### `derive_run_id()`

It's still used in two places. The LeJEPA world-model path (`--objective=lejepa`) **always** uses it,
since it never calls `begin_run()`. `train`/`retrain` use it only when `begin_run()` returns no
MNS-allocated run id. Its 8-character hostname prefix plus a 4-digit PID tail can collide between
hosts with similar names, which matters for a shared remote registry, where run ids decide who owns
claimed files.

---

## 3. Tests

`incremental_trainer_args_test.cpp` (25 tests) covers:
- flag stripping in any order;
- the `=`-only objective syntax;
- missing values falling through as positionals;
- the GPU-defer command set;
- `derive_run_id()` (configured value, fallback, stability);
- `resolve_dataset_kind()` (default, `lejepa`, unknown → chatbot, override wins);
- `reset` flags (including unknown tokens ignored);
- TD-523's exit-code classification and failure backoff (2 tests, October 19, 2026).

**Not covered:** a non-numeric `--admin-port` (which aborts), or validation of the objective and kind
(none exists; the tests assert the permissive behaviour).

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `--admin-port` parsed with `std::stoi` outside any `try`: non-numeric aborts the process, partial (`8084x`) is accepted, no range check (§2) | A typo crashes `incremental_trainer` with a core dump; odd values pass silently | **Verified** | [TD-401](../../guides/TECHNICAL_DEBT.md#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values) |
| Any `--objective` other than `lejepa` silently means chatbot training (§2) | `--objective=lejpa train` trains the chatbot instead of the world model | **Verified** (accepted silently) | [TD-401](../../guides/TECHNICAL_DEBT.md#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values) |
| A flag missing its value becomes a positional argument (by design, per the tests) (§2) | Typos are reinterpreted as commands or file arguments instead of rejected | **Verified** | [TD-401](../../guides/TECHNICAL_DEBT.md#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values) |
| `--dataset-kind` unvalidated; a `LocalTransport` path can escape `session_dir` (§2) | Typos select an empty pool silently; `..` writes registry files elsewhere | By inspection | [TD-402](../../guides/TECHNICAL_DEBT.md#td-402---dataset-kind-is-not-validated) |
| `trainer_service` drops `--objective`/`--dataset-kind` (§1) | The always-on service can't run LeJEPA pretraining; the flags are silently ignored | By inspection | [TD-403](../../guides/TECHNICAL_DEBT.md#td-403-trainer_service-drops---objective-and---dataset-kind) |
| `derive_run_id()` can collide across hosts; the LeJEPA path always uses it (§2) | Claimed-file ownership in a shared registry can collide for LeJEPA runs, or for chatbot runs without MNS | By inspection | [TD-404](../../guides/TECHNICAL_DEBT.md#td-404-derive_run_id-can-collide-across-hosts) |
