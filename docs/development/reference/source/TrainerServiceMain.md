# `TrainerServiceMain` — Source File Reference

- **File:** [`src/TrainerServiceMain.cpp`](../../../../src/TrainerServiceMain.cpp) (315 lines)
- **Binary:** `trainer_service` (admin API proxy on 8084, opt-in); deployed by [`scripts/adai-trainer.service`](../../../../scripts/adai-trainer.service)
- **Operator guide:** [trainer-service-guide.md](../../../operations/guides/trainer-service-guide.md)
- **Uses:** [`ChildProcess`](ChildProcess.md), [`PassSupervision`/`PassHeartbeat`](PassHeartbeat.md) (TD-569), [`TrainerServiceControlState`](TrainerServiceControlState.md), [`TrainerServiceProxy`](TrainerServiceProxy.md), [`IncrementalTrainerArgs`](IncrementalTrainerArgs.md), `Config`, `Logger`
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.13.0`, `@adai-reviewed: 2026-11-11`
- **Tests:** `SmokeTests_trainer_service_main` ([`tests/smoke/trainer_service_main_test.sh`](../../../../tests/smoke/trainer_service_main_test.sh), 38 checks) runs the real binary against a stub pass (§3). Its pieces have their own tests too: the wait for a pass, with the stall check, against real child processes (`passHeartbeatTests`, TD-569); the proxy (`trainerServiceProxyTests`); `ChildProcess` (`childProcessTests`); the argument parser, the pass command line, exit-code classification and backoff (`incrementalTrainerArgsTests`); `SIGHUP` handling of the real binary (`sighupTests`)
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `trainer_service` binary next to a stub `incremental_trainer`
> script.

---

## 1. What this file is

The TD-172 process supervisor that replaced `incremental_trainer serve`. It never links the model or GPU libraries; every training pass runs in a
child process, so a GPU-driver crash ends only that pass.

1. **Arguments** (`parse_trainer_service_args()`, TD-557): `--config`, `--model`, `--gpu-strategy`, `--objective=…`, `--dataset-kind`. `--help`/`-h`
   prints the usage and exits 0. A flag error, a positional argument, or a flag `trainer_service` sets on its passes (`--foreground`, `--admin-port`,
   `--log-file`, `--heartbeat-file`) prints the error and the usage and exits 1.
2. **Start-up:** load `config.trainer.conf`, initialize the logger with `LOG_LEVEL` and `LOG_FILE_PATH` (TD-431), run `validate_or_exit()` (TD-288),
   create the control state, ignore `SIGHUP` (TD-283), and install SIGTERM/SIGINT handlers that only set `stop_requested`.
3. **The pass command,** built once by `trainer_service_child_argv()`: the `incremental_trainer` beside this executable (`sibling_executable_path()`,
   TD-524), its `--config`/`--model`/`--gpu-strategy` when given, always `--objective=<o>` and `--dataset-kind <kind>` (TD-403), `--log-file
   <LOG_FILE_PATH with .pass>` when `LOG_FILE_PATH` is set (TD-431), `--heartbeat-file <SESSION_DIR>/trainer_service_pass_<pool>.heartbeat` when
   `TRAINER_PASS_STALL_MINUTES` is above 0 (TD-569), then `--foreground [--admin-port TRAINER_CHILD_ADMIN_PORT] resume` (the port only with
   `TRAINER_ADMIN_ENABLED`, TD-524). It logs the stall limit, or a warning when it is 0.
4. **Admin proxy:** with `TRAINER_ADMIN_ENABLED`, `TrainerServiceProxy` runs on `TRAINER_ADMIN_HOST:TRAINER_ADMIN_PORT` in a thread (a bind failure is
   logged and the service carries on without it). Idle `GET /admin/config` reports the config file's tunables (TD-525).
5. **The loop,** until `stop_requested`:
   - paused (`/admin/pause`) → `interruptible_sleep(45)`;
   - launch the pass. A launch failure (TD-278) counts as a failure and backs off (below) without counting a pass;
   - record the launch time and PID, point the proxy at the pass's port, and wait with `wait_for_pass()` (`PassSupervision.cpp`). It returns
     when the pass exits; on a stop request; or, TD-569, when the pass's heartbeat count hasn't changed for `TRAINER_PASS_STALL_MINUTES`. The
     last two stop the pass with `ChildProcess::stop_and_wait(25 s)`: SIGTERM, then a kill (TD-416: a pass stopped by SIGTERM checkpoints and
     releases its files first). While waiting it keeps `seconds_since_pass_progress` current for `/admin/status`;
   - record the exit time and code; on a stop request, leave the loop;
   - a **stalled** pass (TD-569) counts in `total_passes_stalled` and `consecutive_failures`, whatever its exit code, is logged with its time
     without progress, and backs off;
   - otherwise `classify_pass_exit()` (TD-523):
     - **0, trained** → relaunch at once (more may be pending);
     - **3, nothing pending** → sleep 45 s;
     - **128 and up, crashed** (128 + the signal, TD-277) or **any other code, failed** → log an error with the code, count it, and back off.

     The backoff is `failure_backoff_seconds()`: 45 s, doubled per consecutive failure, up to 900 s. A pass that trains or finds nothing pending
     resets the count. Every sleep is wakeable by `/admin/resume` and ends on a stop request.
6. Stop the proxy and return 0.

---

## 2. Problems

### Fixed: a pass that hangs stalled the service (TD-569, November 11, 2026)

The loop waited for a pass with no time limit: it polled `poll_exit()` every 200 ms and ended a pass early only on a stop request. A pass whose
GPU call never returns after a driver engine reset fails no sample, so TD-566's device check never runs, and the pass neither progresses nor
exits. The ai-machine soaks saw this three times on the v1 Level Zero adapter (one pass hung after a checkpoint save, 0.7 s after a
`GT0: Engine reset`). Under `trainer_service` the always-on service would sit on it until someone restarted it: nothing trained, its claimed
files waited for their lease, and `/admin/status` showed `training` at a stale epoch.

Now each pass writes a heartbeat file (`PassHeartbeat.hpp`): the count of its progress calls, made from the working code itself, per training
and validation sample, tokenized pair, loaded line, downloaded chunk, BPE merge, LeJEPA batch and checkpoint. `wait_for_pass()` stops a pass
whose count hasn't changed for `TRAINER_PASS_STALL_MINUTES` (default 30, 0 = off). The time is measured on `trainer_service`'s own steady
clock, and a file left by an earlier pass is ignored by its pid. The stalled pass is counted, logged and backed off like a failure, and the next
pass is a fresh process.

**Verified** with the real binary, `TRAINER_PASS_STALL_MINUTES=1`, and a stub pass that wrote one heartbeat and hung. After 60 s the log
said the pass had made no progress for 60 s, and the pass was stopped by SIGTERM (exit 143) with a 45 s backoff. `/admin/status` then showed
`total_passes_stalled: 1`, `consecutive_failures: 1`, `total_passes_failed: 0`, `total_passes_crashed: 0` and
`seconds_since_pass_progress: -1`. A real CPU `incremental_trainer` pass given `--heartbeat-file` wrote its own pid and a count rising by
about 115 every 2 s, and removed the file at exit.

### Fixed: a child that fails every pass looked idle (TD-523, October 19, 2026)

`incremental_trainer resume` returned 1 both for "nothing pending" and for real failures, so a failing trainer showed as idle and retried every
45 s, claiming and releasing files, with nothing in `trainer_service`'s log. `resume` now exits 0/3/1 and the supervisor counts, logs and backs off
failures (§1 step 5). Verified with a stub child: exit 1 showed `total_passes_failed: 1`, `consecutive_failures: 1` and an error in the log; exit 3
stayed idle.

### Fixed: TD-403, TD-416, TD-431, TD-524, TD-557

The pass command forwards `--objective`/`--dataset-kind` (TD-403); the graceful stop allows 25 s (TD-416); each pass logs to its own file and
`LOG_LEVEL` applies (TD-431); the child port opens only with the admin API on, the child binary is the one beside `trainer_service`, a launch
failure backs off, and the shipped unit can write the download directory (TD-524); `--help` and argument errors (TD-557). §1 describes each.

---

## 3. Tests

- `passHeartbeatTests` drives `wait_for_pass()` with real `/bin/sh` children that write the heartbeat a pass would: a pass that stops making
  progress is stopped after the limit (SIGTERM, exit 143); one that keeps making progress runs past the limit and exits 0; a hung one that
  ignores SIGTERM is killed (exit −2); an earlier pass's file doesn't keep a silent pass alive; no heartbeat file means no stall check; a stop
  request stops the pass; and `seconds_since_pass_progress` is reported while waiting. Removing the stall check or the pid check makes these fail.
- `incrementalTrainerArgsTests`: the pass command line (including `--heartbeat-file`), `trainer_service`'s argument errors, exit-code
  classification and the backoff.
- `trainerServiceProxyTests`: the proxy, the status fields, pause/resume/wake.

- `SmokeTests_trainer_service_main` copies the real `trainer_service` next to a stub `incremental_trainer` that follows a plan per pass, with the
  admin API on, and checks the loop end to end: a pass that trained is relaunched at once and an idle one counted as neither; each pass's command
  line (objective, pool, heartbeat file, admin port, `resume`); a failed pass (exit 1) and a killed one (137) counted and backed off; pause and
  resume; a hung pass stopped after a 1-minute stall limit and counted as stalled (exit 143), with `seconds_since_pass_progress` reported while it
  ran; SIGTERM during a pass ending both; and, with the stall check and the admin API off, neither flag passed. About 70 s. Disabling the stall
  branch makes it fail.

**Not covered by a test:** a launch failure (a missing `incremental_trainer`), and the backoff's doubling over several failures.

---

## 4. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| A pass that hangs without failing a sample was waited for forever (§2) | The always-on service stalled until a manual restart | **Fixed November 11, 2026** (was verified on ai-machine, v1 adapter) | [TD-569](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-569-trainer_service-has-no-time-limit-for-a-pass-that-stops-making-progress) |
| A long stretch of work with no progress call is stopped as a stall | A pass doing such work for more than `TRAINER_PASS_STALL_MINUTES` would be stopped and retried | By design; every long loop found reports progress, and the 30-minute default is far above any gap seen | TD-569 |
| Exit code 1 meant both "idle" and "failed" (§2) | A failing trainer looked idle | **Fixed October 19, 2026** | [TD-523](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-523-trainer_service-cant-tell-a-failing-pass-from-an-idle-one) |
| Child port, binary resolution, launch backoff, unit write paths | — | **Fixed November 2, 2026** | [TD-524](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-524-trainer_service-child-port-binary-resolution-backoff-and-unit-write-paths) |
| Forwarded flags, log files, SIGHUP, exit codes, FTP resume, `--help` | — | Fixed | TD-403, TD-431, TD-283, TD-277, TD-278, TD-411, TD-412, TD-557 |

## 5. History

- **October 19, 2026 (TD-523):** failing passes counted, logged and backed off; `resume` exits 0/3/1.
- **November 1–2, 2026:** TD-557 (`--help`, argument errors), TD-403 (objective and pool forwarded; the LeJEPA pass hosts the per-pass admin
  API), TD-416 (25 s graceful stop), TD-524, TD-431.
- **November 11, 2026:** the smoke test above; the heartbeat file named after the pass's pool (`trainer_service_pass_<pool>.heartbeat`), so two
  services with different objectives sharing `SESSION_DIR` don't overwrite each other's file and stop each other's passes; the
  [operator guide](../../../operations/guides/trainer-service-guide.md).
- **November 11, 2026 (TD-569):** the wait for a pass moved to `wait_for_pass()` with the heartbeat stall check, `TRAINER_PASS_STALL_MINUTES`,
  `total_passes_stalled`/`seconds_since_pass_progress`. This document was re-traced at the same time, replacing its October §1–§3 and folding
  its addenda into them.
