# `TrainerServiceControlState` — Source File Reference

- **File:** [`src/TrainerServiceControlState.hpp`](../../../../src/TrainerServiceControlState.hpp) (header-only, 111 lines)
- **Namespace:** `adai`
- **Used by:** `trainer_service` ([`TrainerServiceMain.cpp`](../../../../src/TrainerServiceMain.cpp), which owns it for the process lifetime) and `TrainerServiceProxy` (its HTTP handlers)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.2.0`, `@adai-reviewed: 2026-10-19`
- **Tests:** no dedicated file; exercised by [`tests/trainer_service_proxy_test.cpp`](../../../../tests/trainer_service_proxy_test.cpp) (18 tests, `trainerServiceProxyTests`; all pass), including the pause-while-sleeping and wake-before-sleep cases
- **Last traced against the code:** 2026-10-19

> Code-traced reference.

---

## 1. What this file is

`trainer_service`'s own process-lifetime control state (the TD-173 fix). [`TrainerControlState`](TrainerControlState.md) lives for one child pass;
this one decides whether the service launches passes at all and records supervisor-level observability for `GET /admin/status`.

| Member | Writer → reader | Meaning |
|---|---|---|
| `paused` | proxy (`/admin/pause`, `/admin/resume`) → loop | Checked before every child launch; deliberately **not persisted**, so a restart comes back unpaused |
| `stop_requested` | SIGTERM/SIGINT handler (plain atomic store) → loop | Exit request |
| `service_started_unix`, `last_launch_unix`, `last_exit_unix`, `last_exit_code`, `current_child_pid` (0 = none), `total_passes_launched`, `total_passes_did_work`, `total_passes_crashed` (signal deaths, exit ≥ 128), and since TD-523 (October 19, 2026) `total_passes_failed` (any other code but 0/3), `consecutive_failures` (reset by a pass that trained or found nothing), `last_failure_unix` | loop → proxy | Observability, reported as extra top-level keys of `/admin/status`; `consecutive_failures` also drives the supervisor's backoff |
| `wake()` | proxy (`/admin/resume`) | Sets a sticky `wake_requested_` flag and notifies, so a wake before the sleep starts isn't lost. Documented as **not** async-signal-safe and not called from the signal handler |
| `interruptible_sleep(seconds)` | loop | Waits in one-second slices, returning early on a wake or `stop_requested` (so a signal is noticed within about a second without the handler touching the mutex) |

The threading contract holds: every shared field is atomic or guarded by `wake_mutex_`, and the signal handler only stores to `stop_requested`.

---

## 2. Gaps

None of its own. Related items tracked elsewhere:

| Item | Tracked as |
|---|---|
| `total_passes_crashed`/`last_exit_code` can't tell signals apart (every signal death reports 128) | TD-277 (TODO in this file) |
| A failed `exec` looks like a launch and reports 127 with no clear log | TD-278 (TODO in this file) |
| The per-pass `TrainerControlState` duplicates the wake/sleep mechanism, unused there | [TD-522](../../guides/TECHNICAL_DEBT.md#td-522-trainercontrolstate-stale-design-text-unused-pausing-phase-and-dead-wake-logic) |

How `TrainerServiceMain.cpp` interprets exit codes (0 = did work, anything else treated as "nothing to do" unless it counts as a crash) is covered
with that file.

---

## 3. Tests

Through `trainer_service_proxy_test.cpp`: pause sets the flag and survives across launches, resume clears it and wakes a sleeping loop, a wake
before the sleep returns immediately (a 60-second sleep returning at once), and the status fields appear in `/admin/status`.
