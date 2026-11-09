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

---

## Addendum (November 1, 2026): strict global flags (TD-401)

`parse_incremental_trainer_global_args()` now reports errors in `error`/`error_message`, and `main()`
prints the message with the usage and exits 1:
- **Missing value:** `--config`, `--gpu-strategy`, `--model`, `--admin-port` or `--dataset-kind` given no
  value (last in argv, or followed by another `--` flag) says `Missing value for <flag>`. It used to
  become a positional, so `status --config` ran with normal config discovery.
- **`--admin-port`:** must be a whole number from 1 to 65535 (the shared `parse_strict_int()` from
  `Config.hpp`). `abc` used to abort the process with SIGABRT (an uncaught `std::stoi`), and `8084x` was
  accepted as 8084.
- **`--objective=`:** must be `chatbot` or `lejepa`. `--objective=lejpa train` used to run chatbot
  training silently. A bare `--objective` (no `=`) says how to write the flag.
- **Command flags** (`reset`'s `--yes`/`--keep-data`) still pass through to the command.

Tests: `incremental_trainer_args_test.cpp` (26 tests) replaces the four that asserted the permissive
behaviour, and `SighupTest.IncrementalTrainerExitsOneOnABadFlag` runs the real binary. The §4 rows for
TD-401 are fixed in code; `--dataset-kind` validation remains TD-402.


## Addendum (November 1, 2026): `parse_trainer_service_args()` (TD-557)

`parse_trainer_service_args(argc, argv)` returns a `TrainerServiceArgs`: `help` for `--help`/`-h` anywhere, else the
`IncrementalTrainerGlobalArgs` from `parse_incremental_trainer_global_args()` with three more errors: `--foreground`,
`--admin-port` (what `trainer_service` passes to each pass) and any positional argument (it has no commands).

## Addendum (November 1, 2026): `--dataset-kind` is validated (TD-402)

`--dataset-kind` must be `encoder`, `decoder`, `world_model` or `chatbot` (`adai::is_dataset_kind()` in `Config.hpp`, the
list `dataset_manager --kind` now shares); anything else is an error and `incremental_trainer` exits 1. It used to be
stored as given, so a typo selected a new, empty local pool and `../x` put the registry files outside `session_dir`.

## Addendum (November 1, 2026): `trainer_service` forwards `--objective` and `--dataset-kind` (TD-403)

Every pass `trainer_service` launches now gets `--objective=<objective>` and `--dataset-kind <kind>` (built by
`trainer_service_child_argv()`; the kind is `resolve_dataset_kind()`'s: an explicit `--dataset-kind`, else `world_model`
for `lejepa`, else `chatbot`). Both used to be parsed and dropped. `incremental_trainer resume` now runs the LeJEPA
world-model pass for `--objective=lejepa` (it ran chatbot training whatever the objective), for `NUM_EPOCHS` epochs, and
that pass exits 3 when nothing is pending and 1 when the registry can't be asked, as the chatbot resume does (TD-523).
The LeJEPA pass has no per-pass admin API yet: during it, `trainer_service` answers `/admin/status` from its own state
and a pause takes effect between passes.

## Addendum (November 1, 2026): LeJEPA segments and a persistent run id (TD-415)

`run_lejepa_training_pass()` loads each claimed unit with its TD-205 segment range (it loaded the whole file for every
segment), and takes its run id from `persistent_run_id(RUN_ID, world_model_dir)`: `RUN_ID` when set, else the id in
`<world_model_dir>/lejepa_run_id`, created once as `lejepa-<derive_run_id()>-<8 random hex digits>`. Registries hand a
claim back to the run id that made it, so a pass restarted after a crash, kill or SIGTERM reclaims the last pass's
files; with the old PID-based id they stayed claimed.

## Addendum (November 2, 2026): unique fallback run ids (TD-404)

`derive_run_id("")` is now `<full hostname>_<pid>_<8 random hex digits>` (it was the first 8 hostname characters and
`pid % 10000`, which two hosts sharing a prefix could both derive). `persistent_run_id(configured, dir, name)` keeps an
id in `<dir>/<name>_run_id`: `lejepa` in the world-model directory for the LeJEPA pass, and, when no `RUN_ID` is set,
`chatbot` in `SESSION_DIR` for `train`/`retrain`/`resume`, which `incremental_trainer` copies into `RUN_ID`. An
MNS-allocated run id still comes first. A restarted pass therefore reclaims its own claimed files.

## Addendum (November 2, 2026): optional child admin port, `sibling_executable_path()` (TD-524)

`trainer_service_child_argv()` takes `std::optional<int> admin_port` and adds `--admin-port` only when it's set.
`sibling_executable_path(argv0, name)` returns `name` in the running executable's directory, falling back to argv[0]'s
directory and then to the bare name.

## TD-368 addendum (November 9, 2026)

`--seed <n>` (a whole number of at least 1) overrides `SEED`; anything else is an error, exit 1.
