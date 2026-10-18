# `TrainerServiceMain` — Source File Reference

- **File:** [`src/TrainerServiceMain.cpp`](../../../../src/TrainerServiceMain.cpp) (237 lines)
- **Binary:** `trainer_service` (admin API proxy on 8084, opt-in); deployed by [`scripts/adai-trainer.service`](../../../../scripts/adai-trainer.service)
- **Uses:** [`ChildProcess`](ChildProcess.md), [`TrainerServiceControlState`](TrainerServiceControlState.md), `TrainerServiceProxy`, `IncrementalTrainerArgs` ([IncrementalTrainerArgs.md](IncrementalTrainerArgs.md)), `Config`, `Logger`
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.2.0`, `@adai-reviewed: 2026-09-14`
- **Tests:** none for `main()` itself; the proxy (`trainerServiceProxyTests`, 17) and `ChildProcess` have their own. TD-172 records an end-to-end local verification
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `trainer_service` binary next to a stub `incremental_trainer`
> script, with the admin proxy enabled.

---

## 1. What this file is

The TD-172 process supervisor that replaced `incremental_trainer serve`. It never links the model or GPU libraries; every training pass runs in a
child process.

1. Parse `incremental_trainer`'s global flags (`--config`, `--model`, `--gpu-strategy`; others ignored, TD-403), load `config.trainer.conf`, initialize
   the logger at INFO (TD-431), create the control state, install SIGTERM/SIGINT handlers that only set `stop_requested` (no SIGHUP handler, TD-283).
2. Resolve the child binary: `incremental_trainer` next to `argv[0]` if it has a directory, else a `PATH` lookup.
3. Build the child command once: `incremental_trainer [--config …] [--model …] [--gpu-strategy …] --foreground --admin-port TRAINER_CHILD_ADMIN_PORT resume`.
4. If `TRAINER_ADMIN_ENABLED`, start `TrainerServiceProxy` on `TRAINER_ADMIN_HOST:TRAINER_ADMIN_PORT` in a thread (a bind failure is logged and ignored).
5. Loop until `stop_requested`:
   - paused → `interruptible_sleep(45)`;
   - launch the child (a launch failure sleeps 45 s and retries); record launch time, PID; point the proxy at the child port;
   - poll every 200 ms; on a stop request, `ChildProcess::stop_and_wait(10 s)` (SIGTERM, then a force kill), well inside the unit's `TimeoutStopSec=30`;
   - record exit time and code; exit codes other than 0 and 1 count as crashes;
   - **exit 0 → relaunch immediately** (more work may be pending); **anything else → sleep 45 s** (wakeable by `/admin/resume`).
6. Stop the proxy and return 0.

---

## 2. Problems

### A child that fails every pass looks idle (verified)

The loop's whole model of the child is its exit code, and `incremental_trainer resume` returns **1** both for "nothing pending" and for real failures:
`resume_last_session()` returns false for a missing checkpoint, a failed `load_model()` and a failed `train_on_files()`, as well as for an empty queue.
With a stub child that printed an error and exited 1, `/admin/status` reported `"phase":"idle"`, `"total_passes_crashed":0`, `"last_exit_code":1`, the
same as a healthy idle service, and nothing reached `trainer_service`'s own log. In production a corrupt checkpoint, an unreadable dataset or a training
exception therefore turns into a silent retry every 45 s: each attempt claims the pending files, fails and releases them (segments stay stuck, TD-333),
indefinitely, while the dashboard shows an idle trainer. The child should use a distinct exit code for "nothing to do" (and for errors), and the
supervisor should count and surface failures (and back off).

### Smaller items (by inspection)

- **The child admin port opens even when the admin API is disabled.** `--admin-port` is always passed, so every child binds
  `127.0.0.1:TRAINER_CHILD_ADMIN_PORT` (8085 by default) regardless of `TRAINER_ADMIN_ENABLED=false`, which CLAUDE.md describes as keeping the host free of
  new ports.
- **`PATH` fallback** can launch a different `incremental_trainer` version than the one installed beside `trainer_service` when it is started by a bare
  name.
- **No backoff** for repeated crashes or launch failures (a fixed 45 s), and none at all for a child that exits 0 without consuming work.
- **The shipped unit may not let the child write its downloads:** `adai-trainer.service` uses `ProtectSystem=strict` with `ReadWritePaths` covering logs,
  models, sessions, the tokenized cache and the admin dir, but not `/opt/adai/downloads` (the default relative `DOWNLOAD_DIR=downloads` under
  `WorkingDirectory=/opt/adai`), so FTP downloads in `train`/`retrain` there would fail with a read-only filesystem (`resume` doesn't download, TD-411).
  The empty-`LOG_FILE_PATH` fallback to `chatbot_server.log` in the working directory has the same problem (TD-431).

### Already tracked

| Item | Tracked as |
|---|---|
| `--objective`/`--dataset-kind` not forwarded to children | TD-403 |
| Children and supervisor share the fallback log file; `LOG_LEVEL` ignored | TD-431 |
| SIGHUP terminates the process | TD-283 |
| Signal deaths report 128; failed `exec` reports 127 | TD-277, TD-278 |
| `resume` never downloads over FTP; zero-sample passes return success | TD-411, TD-412 |

---

## 3. Tests

No test runs `main()`. `trainer_service_proxy_test.cpp` covers the proxy and the control state's pause/resume/wake behaviour; `ChildProcess` has its
own tests.

**Not covered:** the supervisory loop's exit-code handling, stop escalation through `main()`, the child command line.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Exit code 1 means both "idle" and "failed"; failures aren't counted or logged by the supervisor (§2) | A permanently failing trainer looks idle and retries every 45 s, claiming and releasing files | **Verified** | [TD-523](../../guides/TECHNICAL_DEBT.md#td-523-trainer_service-cant-tell-a-failing-pass-from-an-idle-one) |
| Child admin port opened with the admin API disabled; `PATH` fallback; no backoff; unit's `ReadWritePaths` lacks the download dir (§2) | Unexpected port; wrong binary; download failures under the shipped unit | By inspection | [TD-524](../../guides/TECHNICAL_DEBT.md#td-524-trainer_service-child-port-binary-resolution-backoff-and-unit-write-paths) |
| Unforwarded flags, shared log file, SIGHUP, exit-code reporting, FTP resume | Already tracked | — | TD-403, TD-431, TD-283, TD-277, TD-278, TD-411, TD-412 |
