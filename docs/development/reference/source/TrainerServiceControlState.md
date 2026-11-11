# `TrainerServiceControlState` — Source File Reference

- **File:** [`src/TrainerServiceControlState.hpp`](../../../../src/TrainerServiceControlState.hpp) (header-only, 118 lines)
- **Namespace:** `adai`
- **Used by:** `trainer_service`: [`TrainerServiceMain.cpp`](TrainerServiceMain.md) owns it for the process lifetime; [`PassSupervision.cpp`](PassHeartbeat.md) updates it while waiting for a pass; [`TrainerServiceProxy`](TrainerServiceProxy.md) reads and sets it from its HTTP handlers
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.4.0`, `@adai-reviewed: 2026-11-11`
- **Tests:** no dedicated file. `trainerServiceProxyTests` (26) covers pause, resume, the wake and the status fields; `passHeartbeatTests` (16) covers `seconds_since_pass_progress`
- **Last traced against the code:** 2026-11-11

> Code-traced reference.

---

## 1. What this file is

`trainer_service`'s own process-lifetime control state (the TD-173 fix). [`TrainerControlState`](TrainerControlState.md) lives for one pass; this
one decides whether the service launches passes at all, and holds the supervisor's own figures for `GET /admin/status`.

| Member | Writer → reader | Meaning |
|---|---|---|
| `paused` | proxy (`/admin/pause`, `/admin/resume`) → loop | Checked before every launch; deliberately **not persisted**, so a restart comes back unpaused |
| `stop_requested` | SIGTERM/SIGINT handler (a plain atomic store) → loop, `wait_for_pass()`, `interruptible_sleep()` | Exit request |
| `service_started_unix`, `last_launch_unix`, `last_exit_unix`, `last_exit_code` (128 + the signal for a signal death, TD-277), `current_child_pid` (0 = none) | loop → proxy | Reported as extra top-level keys of `/admin/status` |
| `total_passes_launched`, `total_passes_did_work`, `total_passes_crashed` (exit 128 and up), `total_passes_failed` (any other code but 0/3, TD-523), `total_passes_stalled` (stopped for no progress, TD-569; not also counted as crashed or failed) | loop → proxy | Counters for `/admin/status` |
| `consecutive_failures`, `last_failure_unix` | loop → proxy | Failed, crashed, stalled passes and launch failures in a row (TD-523/TD-278/TD-569), reset by a pass that trained or found nothing pending; drives the backoff |
| `seconds_since_pass_progress` | `wait_for_pass()` → proxy | The running pass's seconds since its last progress (TD-569); −1 with no pass running, 0 throughout with the stall check off |
| `wake()` | proxy (`/admin/resume`) | Sets a sticky `wake_requested_` flag and notifies, so a wake before the sleep starts isn't lost. **Not** async-signal-safe (it locks a mutex), so the signal handler never calls it |
| `interruptible_sleep(seconds)` | loop | Waits in one-second slices, returning early on a wake or `stop_requested` (so a signal is noticed within about a second without the handler touching the mutex), and clears the wake flag |

Every shared field is atomic or guarded by `wake_mutex_`, and the signal handler only stores to `stop_requested`.

---

## 2. Gaps

None. The items this document used to list are fixed: signal deaths report 128 + the signal (TD-277), a failed `exec` is a launch failure
rather than a pass with exit 127 (TD-278), the per-pass state lost its unused wake/sleep copy (TD-522), and `trainer_service` tells a failed pass
from an idle one by `resume`'s exit codes 0/3/1 (TD-523). How the loop uses these fields is in [TrainerServiceMain.md](TrainerServiceMain.md).

---

## 3. Tests

`trainer_service_proxy_test.cpp`: pause sets the flag and survives across launches, resume clears it and wakes a sleeping loop, a wake before the
sleep returns at once (a 60-second sleep returning immediately), and every status field, TD-569's two included, appears in `/admin/status`.
`pass_heartbeat_test.cpp`: `wait_for_pass()` keeps `seconds_since_pass_progress` current while a pass runs and sets it to −1 when the wait ends, and
a stop request set from another thread ends the wait.
