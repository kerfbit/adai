# `IncrementalTrainerArgs` — Source File Reference

- **Files:** [`src/IncrementalTrainerArgs.hpp`](../../../../src/IncrementalTrainerArgs.hpp) (175 lines), [`src/IncrementalTrainerArgs.cpp`](../../../../src/IncrementalTrainerArgs.cpp) (323 lines)
- **Namespace:** `adai`
- **Built into:** `incremental_trainer` (`IncrementalTrainingTool.cpp`) and `trainer_service` (`TrainerServiceMain.cpp`)
- **Status tags:** `@adai-status: stable` 1.0.0 (both, promoted November 11, 2026); `@adai-reviewed: 2026-11-11`
- **Tests:** [`tests/incremental_trainer_args_test.cpp`](../../../../tests/incremental_trainer_args_test.cpp) → `incrementalTrainerArgsTests` (42); `SighupTest.IncrementalTrainerExitsOneOnABadFlag` runs the real binary
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the built `build/debug/bin/incremental_trainer`.

---

## 1. What this file is

`incremental_trainer`'s global command-line parser and the small pure helpers `incremental_trainer` and `trainer_service` share, split out of
`main()` so they can be tested without a real `IncrementalTrainer`.

| Function / constant | Purpose |
|---|---|
| `parse_incremental_trainer_global_args(argc, argv)` | Takes the global flags (§2) from anywhere on the line; the rest is `args` (`args[0]` the command). Errors set `error`/`error_message` and `main()` exits 1 with the usage |
| `parse_trainer_service_args(argc, argv)` | `trainer_service`'s line (TD-557): `help` for `--help`/`-h` anywhere; else the same flags, with `--log-file`, `--heartbeat-file`, `--foreground`, `--admin-port` (set by `trainer_service` on its passes) and any positional argument as errors |
| `trainer_service_child_argv(path, cli, admin_port, pass_log_file, heartbeat_file)` | Each pass's command: `path`, `--config`/`--model`/`--gpu-strategy` when given, always `--objective=<o>` and `--dataset-kind <resolve_dataset_kind()>` (TD-403), `--log-file` and `--heartbeat-file` when non-empty (TD-431, TD-569), `--foreground`, `--admin-port N` when set (TD-524), `resume` |
| `pass_log_path(log_path)` | TD-431: `.pass` before the extension (`trainer.log` → `trainer.pass.log`; no extension → `trainer.pass`), so each process rotates its own file |
| `default_background_log_path()` | TD-431: a detached run's log when `LOG_FILE_PATH` is empty: `$XDG_STATE_HOME` (else `~/.local/state`)`/adai/log/incremental_trainer.log`, or `%LOCALAPPDATA%\adai\Logs` on Windows; creates the directory; the working directory with no home |
| `incremental_trainer_command_defers_gpu_init(cmd)` | True for `train`/`retrain`/`resume`, which may fork, so GPU initialization happens after the fork (CUDA contexts aren't fork-safe) |
| `derive_run_id(configured)` | `configured` when set, else `<full hostname, sanitized>_<pid>_<8 random hex digits>` (TD-404) |
| `persistent_run_id(configured, dir, name)` | `configured` when set, else the id in `<dir>/<name>_run_id`, created once as `<name>-<derive_run_id("")>` (temporary file + rename; a failed write still returns the id, with a warning). Registries hand a claim back to the run id that made it, so a restarted pass reclaims its files (TD-415/TD-404) |
| `parse_epochs_arg(args, default, error)` | `train`/`retrain`'s optional epoch count: a whole number of at least 1, else `error` (TD-416); more than one argument is an error |
| `sibling_executable_path(argv0, name)` | `name` beside the running executable (`/proc/self/exe`; the module path on Windows), else beside argv[0], else the bare name with a warning (TD-524) |
| `resolve_dataset_kind(objective, override)` | `--dataset-kind` if given, else `world_model` for `lejepa`, `chatbot` otherwise; config `DATASET_KIND` deliberately can't reach it (TD-204) |
| `parse_reset_command_args(args)` | `--yes`, `--keep-data` for `reset`; any other argument sets `error`, and `reset` exits 1 before touching anything (TD-571, §3) |
| `kResumeExitTrained` 0, `kResumeExitFailed` 1, `kResumeExitNothingPending` 3 | `resume`'s exit codes (TD-523), shared with `trainer_service` |
| `classify_pass_exit(code)` → `PassOutcome` | 0 `DidWork`, 3 `Idle`, 128 and up `Crashed`, anything else `Failed` (127 included) |
| `failure_backoff_seconds(n, base, max)` | After *n* ≥ 1 failures in a row: `base` doubled per extra failure, capped at `max` (`trainer_service`: 45 s, 900 s) |

---

## 2. The global flags

| Flag | Value | Checks |
|---|---|---|
| `--config <path>` | Config file | — |
| `--gpu-strategy <s>` | `background` / `full` | An unknown value warns and means `background` (in `gpu_strategy_from_string()`) |
| `--model <name>` | MNS model name | — |
| `--foreground` | — | Don't fork |
| `--admin-port <N>` | The pass's admin port (set by `trainer_service`) | A whole number 1–65535 (`parse_strict_int()`), TD-401 |
| `--seed <n>` | Overrides `SEED` | A whole number of at least 1 (TD-368) |
| `--objective=<o>` | `chatbot` (default) / `lejepa` | Only the `=` form; any other value is an error, and a bare `--objective` says how to write it (TD-401) |
| `--dataset-kind <k>` | Sub-pool | `encoder`/`decoder`/`world_model`/`chatbot` (`is_dataset_kind()`, TD-402) |
| `--log-file <path>` | Overrides `LOG_FILE_PATH` | — (TD-431) |
| `--heartbeat-file <path>` | Where the run writes its progress for `trainer_service` | Needs `--foreground`: a background run forks after the writer starts (TD-569) |

A value flag given no value (last on the line, or followed by another `--` word) is `Missing value for <flag>` (TD-401). Anything else goes to
`args`, so command flags such as `reset`'s reach the command.

---

## 3. Problems

### Fixed: `reset` ignored a mistyped flag and deleted the data registry (TD-571, November 11, 2026; was verified)

`parse_reset_command_args()` takes `--yes` and `--keep-data` and drops every other token. `reset` is destructive: without `--keep-data` it deletes
the data registry, and with `--yes` it doesn't ask. So `incremental_trainer reset --yes --keep-dat` (a one-letter typo) reset everything and
deleted the registry, exiting 0: run against a scratch session, the log said "Data registry removed" and `data_registry.txt` was gone. The
confirmation text does say "Delete the data registry", but `--yes` skips reading it. Other commands' stray arguments are errors since TD-416
(`parse_epochs_arg()`); `reset`'s weren't, and a test asserted the permissive behaviour. Now any other argument is an error
(`Unknown reset argument '--keep-dat'`), exit 1, nothing reset: checked with the real binary; `UnrecognizedTokensAreAnError`.

---

## 4. Tests

`incremental_trainer_args_test.cpp` (42):

- the global flags in any order; every error (missing values, `--admin-port`, `--seed`, `--objective`, `--dataset-kind`, `--heartbeat-file`
  without `--foreground`);
- `parse_trainer_service_args()`: `--help`, and the flags it refuses;
- `trainer_service_child_argv()`: objective and kind, the optional admin port, the log file, the heartbeat file;
- `derive_run_id()`, `persistent_run_id()` (kept, reused, configured value wins);
- `resolve_dataset_kind()`, `parse_reset_command_args()`, `classify_pass_exit()`, `failure_backoff_seconds()`, `parse_epochs_arg()`,
  `sibling_executable_path()`, `pass_log_path()`, `default_background_log_path()`, the GPU-defer command set.

**Not covered:** nothing known.

---

## 5. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `reset` ignored unknown tokens, so a mistyped `--keep-data` with `--yes` deleted the data registry (§3) | — | **Fixed November 11, 2026** (was verified) | [TD-571](../../guides/TECHNICAL_DEBT.md#td-571-findings-from-the-november-11-2026-trace-of-the-files-td-569-touched) |
| `--admin-port` abort, a mistyped objective training the chatbot, missing values as positionals | — | **Fixed November 1, 2026** | [TD-401](../../guides/TECHNICAL_DEBT.md#td-401-incremental_trainer-flag-parsing-crashes-or-silently-misreads-bad-values) |
| `--dataset-kind` unvalidated; flags dropped by `trainer_service`; colliding run ids; child port; log files; `--help` | — | **Fixed** | TD-402, TD-403, TD-404, TD-524, TD-431, TD-557 |

## 6. History

- **October 19, 2026 (TD-523):** `resume` exit codes, `classify_pass_exit()`, `failure_backoff_seconds()`.
- **November 1–2, 2026:** strict flags (TD-401), `parse_trainer_service_args()` (TD-557), `--dataset-kind` checked (TD-402), objective and kind
  forwarded (TD-403), `persistent_run_id()` (TD-415/TD-404), unique fallback run ids (TD-404), optional child port and `sibling_executable_path()`
  (TD-524), `--log-file` and the log paths (TD-431), `parse_epochs_arg()` (TD-416).
- **November 9, 2026 (TD-368):** `--seed`.
- **November 11, 2026:** `--heartbeat-file` (TD-569); `reset` refuses unknown arguments (TD-571). Promoted to `stable` 1.0.0 the same day: 42
  tests and checks with the real binary, no open gap, used by `incremental_trainer` and `trainer_service`. This document was re-traced against the code, replacing its October sections and folding
  in its addenda.
