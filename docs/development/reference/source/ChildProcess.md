# `ChildProcess` — Source File Reference

- **Files:** [`src/ChildProcess.hpp`](../../../../src/ChildProcess.hpp), [`src/ChildProcess.cpp`](../../../../src/ChildProcess.cpp)
- **Namespace:** `adai`
- **Built into:** `trainer_service` and its test binary
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.2.0`, `@adai-reviewed: 2026-09-14`
- **Tests:** [`tests/child_process_test.cpp`](../../../../tests/child_process_test.cpp) (14 tests, against real short-lived children)
- **Origin:** TD-172 (process-supervisor `trainer_service`) and TD-173 (bounded, escalating stop)
- **Last traced against the code:** 2026-10-09

> Code-traced reference. The exit-code behaviour in §3 was confirmed by running the real class
> against children killed by SIGSEGV/SIGKILL and against a missing binary.

---

## 1. What this file is

A small, cross-platform "launch and babysit **one** child process" helper. `trainer_service` uses
it to run each training pass as a separate `incremental_trainer --foreground --admin-port N resume`
process (see CLAUDE.md, "Incremental trainer admin API"). If a GPU-driver crash or hang takes down a
pass, only that child dies; the supervisor and its admin API keep running and launch a fresh child.

```text
trainer_service main loop
  └─ child.start(argv)            fork()+execvp()   |  CreateProcessA(CREATE_NEW_PROCESS_GROUP)
     loop every 200 ms:
       child.poll_exit(&code)     waitpid(WNOHANG)  |  GetExitCodeProcess
       on stop request: child.stop_and_wait(10 s)   SIGTERM → wait → SIGKILL → wait
```

### Why it matters

This is the piece that makes "a driver crash takes down one pass, not the service" true. Its
exit-code reporting is also what `trainer_service`'s `/admin/status` counters
(`total_passes_crashed`, `last_exit_code`) are built on, and it's the only diagnostic left when a
pass dies on ai-machine.

---

## 2. API

| Member | Behaviour |
|---|---|
| `start(argv)` | Launches `argv[0]` with the rest as arguments. Returns false if a child is already running or `argv` is empty, or if `fork()`/`CreateProcessA` fails. **POSIX:** `fork()`, then in the child `execvp()`; on exec failure the child `_exit(127)`s. **Windows:** builds a quoted command line (minimal escaping, by its own admission) and creates the process in a new process group so console-control events can target it |
| `poll_exit(&code)` | Non-blocking. Returns true **once** when the child has exited and records the exit code (see §3); false while running or when no child is tracked. POSIX `waitpid` failure (e.g. `ECHILD`) is reported as exited with code `-1` |
| `request_stop()` | Fire-and-forget graceful stop: `kill(pid, SIGTERM)`, or Windows `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT)` falling back to `TerminateProcess` if the send fails. Doesn't wait or escalate |
| `stop_and_wait(timeout_ms, &code)` | `request_stop()`, polls every 50 ms up to `timeout_ms`, then `SIGKILL`/`TerminateProcess` and polls up to 5 s more. A forced kill reports code `-2`. If the child still isn't reaped after the kill, it logs an error and **forgets** the child anyway (returns true) so the caller can move on. Returns false only if no child was running |
| `~ChildProcess()` | If a child is still tracked, `stop_and_wait(5000)` (TD-173: never hang the destructor on a wedged child) |
| `is_running()`, `pid()` | Tracking state; `pid()` is 0 when nothing is running |

Copying is deleted. The class isn't thread-safe and is meant to be driven from one loop thread.

**Used by:** `src/TrainerServiceMain.cpp` only: `start()` per pass, `poll_exit()` every 200 ms,
`stop_and_wait(10000)` on `SIGTERM`/`SIGINT`, and `pid()` for status. Its signal handler only sets a
flag, so in practice all calls come from the main loop thread.

---

## 3. Exit codes (verified)

| Child outcome | `poll_exit()` / `stop_and_wait()` code |
|---|---|
| Normal exit with *n* | *n* |
| Killed by **any** signal (SIGSEGV, SIGKILL, SIGTERM, SIGABRT…) | **128** |
| `execvp()` failed (missing or non-executable binary) | **127**, and `start()` returned **true** |
| `waitpid()` error | −1 |
| Force-killed by `stop_and_wait()` | −2 |

> **The signal number is discarded.** The header calls 128 "POSIX convention", but the shell
> convention is **128 + signal number** (139 for SIGSEGV, 137 for SIGKILL). Running it confirmed both
> SIGSEGV and SIGKILL deaths come back as 128. On ai-machine, the known failure mode is a
> **SIGSEGV** inside the Intel graphics compiler (see [ONEAPI_SYCL_DRIVER_SEGFAULT.md](../../../operations/guides/troubleshooting/ONEAPI_SYCL_DRIVER_SEGFAULT.md)), and an OOM kill
> would be SIGKILL. `/admin/status`'s `last_exit_code` can't tell them apart.

> **A missing or broken binary looks like a successful launch.** `start()` returns true after
> `fork()`. The exec failure only surfaces later as exit code 127, which `trainer_service` counts as a
> crashed pass and retries after every poll interval, indefinitely, with no message saying the
> binary couldn't be executed. (The standard fix is a close-on-exec pipe through which the child
> reports the `exec` error back before `start()` returns, or `posix_spawn()`.)

---

## 4. Process-model notes

- **Only the direct child is signalled.** There's no process group on POSIX, and no
  `PR_SET_PDEATHSIG`. If `trainer_service` itself is killed hard (outside systemd, whose default
  `KillMode=control-group` kills the whole cgroup), a running pass is orphaned and keeps going,
  while a restarted supervisor launches a second one.
- **`fork()` from a multithreaded process.** `trainer_service` runs its httplib admin proxy on other
  threads, so `start()` forks a multithreaded process. The child calls only `execvp()` (argv is built
  before the fork), but `execvp` isn't on POSIX's async-signal-safe list (`execve` is), because its
  PATH search may allocate. `posix_spawn()` would avoid the question entirely.
- **File descriptors are inherited.** httplib marks its *listening* socket close-on-exec, but
  *accepted* connection sockets aren't (`accept()`, not `accept4(SOCK_CLOEXEC)`). A child forked while
  an admin request is in flight inherits that connection's fd until the pass ends. It's minor, since
  httplib shuts sockets down explicitly before closing them.
- **`request_stop()`'s signal-safety claim is fragile.** The header says it's safe to call from a
  signal handler's thread. On POSIX it checks the plain `bool running_`, then calls
  `kill(pid_, SIGTERM)`. `stop_and_wait()`'s give-up path sets `pid_ = -1` *before* clearing
  `running_`, and nothing is atomic, so a concurrent `request_stop()` could read `running_ == true`
  and `pid_ == -1`. `kill(-1, SIGTERM)` signals **every process the user can signal**. It isn't
  reachable today, since `trainer_service`'s handler only sets a flag, but a `pid_ > 0` guard is a
  one-line safety net.
- **Windows specifics** (not built for any shipped Windows target today):
  - command-line quoting doesn't handle backslashes before quotes;
  - a child that genuinely exits with code 259 (`STILL_ACTIVE`) would be treated as running forever;
  - `CTRL_BREAK_EVENT` only works if the child shares a console.

---

## 5. Tests

`child_process_test.cpp` (14): launch and running state, polling while running, real exit codes,
`request_stop()` termination, double-start and empty-argv rejection, destructor reaping
(including a SIGTERM-ignoring child), sequential reuse, `pid()`, and `stop_and_wait()`'s graceful
path, no-child path and force-kill escalation. Not covered: signal-death codes, exec failure, and
concurrent `request_stop()`.

---

## 6. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Every signal death reported as 128; signal number lost (§3) | Can't distinguish GPU-driver SIGSEGV from an OOM SIGKILL in status/metrics | **Verified** | [TD-277](../../guides/TECHNICAL_DEBT.md#td-277-childprocess-reports-every-signal-death-as-exit-code-128) |
| Exec failure reported as a successful `start()` then exit 127 (§3) | Missing binary → endless silent retry loop | **Verified** | [TD-278](../../guides/TECHNICAL_DEBT.md#td-278-childprocess-reports-a-failed-exec-as-a-successful-launch) |
| `request_stop()` could `kill(-1, SIGTERM)` if called concurrently, as the header permits (§4) | Would signal every user process; not reachable today | By inspection | [TD-279](../../guides/TECHNICAL_DEBT.md#td-279-childprocessrequest_stop-can-signal-every-process-if-called-concurrently) |
| No process group or parent-death signal (§4) | Orphaned passes if the supervisor is hard-killed outside systemd | By inspection | [TD-280](../../guides/TECHNICAL_DEBT.md#td-280-trainer_service-passes-can-be-orphaned-if-the-supervisor-is-hard-killed) |
| `fork()` + `execvp()` in a multithreaded process; accepted sockets inherited (§4) | Theoretical deadlock; fd leak during a pass | By inspection | [TD-281](../../guides/TECHNICAL_DEBT.md#td-281-childprocess-forks-a-multithreaded-process-and-leaks-file-descriptors) |
| Windows quoting / `STILL_ACTIVE` / console-event limits (§4) | Windows-only | By inspection | [TD-282](../../guides/TECHNICAL_DEBT.md#td-282-childprocess-windows-path-has-known-gaps) |
