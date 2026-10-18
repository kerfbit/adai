# `TrainerServiceProxy` — Source File Reference

- **Files:** [`src/TrainerServiceProxy.hpp`](../../../../src/TrainerServiceProxy.hpp) (94 lines), [`src/TrainerServiceProxy.cpp`](../../../../src/TrainerServiceProxy.cpp) (417 lines)
- **Namespace:** `adai`
- **Used by:** `trainer_service` ([TrainerServiceMain.md](TrainerServiceMain.md)): its public admin listener on `TRAINER_ADMIN_HOST:TRAINER_ADMIN_PORT` (127.0.0.1:8084 shipped), started only with `TRAINER_ADMIN_ENABLED=true`. Clients: `curl`, the Android ops dashboard's Trainer tab (via the `trainer.kerfbit.dev` tunnel)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.2.0`, `@adai-reviewed: 2026-09-14`
- **Tests:** [`tests/trainer_service_proxy_test.cpp`](../../../../tests/trainer_service_proxy_test.cpp) (17 tests, `trainerServiceProxyTests`; all pass)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program running the real proxy in front of a real
> [`TrainerAdminAPI`](TrainerAdminAPI.md) child instance, called with `curl`.

---

## 1. What this file is

The always-on admin endpoint of `trainer_service`. While a child pass is alive (`set_child_port(N)`), requests are forwarded to the child's private
`TrainerAdminAPI` on `127.0.0.1:N`; while none is (`set_child_port(0)`), it answers itself.

| Endpoint | Child alive | No child |
|---|---|---|
| `GET /health` | Answered here (never proxied) | Same |
| `GET`/`PUT /admin/config` | Proxied | Read/write `TRAINER_ADMIN_DIR/daemon_config.db` directly (same validation as the child, TD-305's partial-apply included); GET overlays the file onto **hardcoded** defaults |
| `GET /admin/status` | Proxied; on 200, supervisor fields appended | Idle-shaped body with the supervisor's `paused`; supervisor fields appended |
| `GET /admin/logs` | Proxied | `{"entries":[]}` |
| `POST /admin/checkpoint[?wait_ms=N]` | Proxied (`wait_ms` copied into the path) | 409 |
| `POST /admin/pause` / `resume` | Always set/clear `TrainerServiceControlState::paused` and `wake()` the supervisor (TD-173); then best-effort proxy to the child (response ignored); 202 | Same, without the proxy |

`with_supervisor_fields()` appends `supervisor_paused`, `total_passes_launched`, `total_passes_did_work`, `total_passes_crashed`, `last_exit_code` and
`service_uptime_seconds` before the final `}` (only if the body ends with one). Proxy requests use a new `httplib::Client` with a 1 s connection and 5 s
read timeout; a connection failure becomes **503 "training pass is starting or exiting; try again shortly"**.

---

## 2. Problems

### A long checkpoint wait is reported as "starting or exiting" (verified)

The child honours any `wait_ms`, but the proxy's read timeout is 5 s. `POST /admin/checkpoint?wait_ms=8000` through the proxy returned **503 "training
pass is starting or exiting; try again shortly"** (after 10 s, httplib retrying the read once) while the child had in fact accepted the request
(`checkpoint_requested` was set). The caller is told the request failed for the wrong reason, and retrying requests yet another checkpoint. The same
503 text is returned for every proxied request when the child's admin API failed to bind, for the whole pass. (The Android dashboard doesn't send
`wait_ms`, so it isn't affected.)

### Idle config doesn't match what the next pass will use (verified)

Idle `GET /admin/config` starts from hardcoded defaults (1000 / 30 / 50 / true) and overlays `daemon_config.db`; it never sees `config.trainer.conf`,
which the child seeds its tunables from before applying the same overlay. With a child seeded `auto_save_every_samples = 250` (as from a config file
with `AUTO_SAVE_EVERY_SAMPLES=250`), the proxy reported **1000 while idle and 250 while the child ran**, so an operator checking settings between
passes sees values that won't apply. The shipped config uses the same numbers as the hardcoded defaults, which hides it.

### Smaller items (by inspection)

- **`wait_ms` is re-inserted into the proxied path unencoded** (the decoded query value is concatenated), so a crafted value can alter the request line
  sent to the child (loopback only).
- **`paused` means two things** in `/admin/status`: the child's per-pass flag while a pass runs, the supervisor's flag when idle (`supervisor_paused` is
  always the supervisor's).
- **A new client and connection per proxied request**; pause/resume proxy failures are silently ignored.

### Already tracked

| Item | Tracked as |
|---|---|
| Spaced `"auto_save_enabled": true` reads as false (idle handler copy) | TD-519 |
| NaN losses in the child's status relayed as invalid JSON | TD-520 |
| Idle PUT persists earlier keys before rejecting a later one | TD-305 |
| Duplicated hand-written JSON helpers | TD-453 |
| A failing child looks idle in the supervisor fields | TD-523 |

---

## 3. Tests

`trainer_service_proxy_test.cpp` (17): health never proxied; idle status/logs/checkpoint/config; idle PUT persisting and rejecting immutable keys;
status and PUT proxied verbatim to a live child; 503 for an unreachable child port; returning to idle; pause persisting, waking a sleeping supervisor
and also reaching a live child; resume; idle status showing the real pause state; supervisor fields in status.

**Not covered:** `wait_ms` longer than the read timeout, idle config vs a non-default `config.trainer.conf`.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| 5 s proxy read timeout vs unbounded `wait_ms`; generic "starting or exiting" 503 (§2) | An accepted checkpoint reported as a failure; misleading errors when the child API is down | **Verified** | [TD-525](../../guides/TECHNICAL_DEBT.md#td-525-trainerserviceproxy-misreports-long-checkpoint-waits-and-idle-config) |
| Idle config ignores `config.trainer.conf` (§2) | Between passes, reported settings differ from what the next pass uses | **Verified** | [TD-525](../../guides/TECHNICAL_DEBT.md#td-525-trainerserviceproxy-misreports-long-checkpoint-waits-and-idle-config) |
| Unencoded `wait_ms` in the proxied path; two meanings of `paused`; per-request clients; ignored pause/resume proxy failures (§2) | Hygiene | By inspection | [TD-525](../../guides/TECHNICAL_DEBT.md#td-525-trainerserviceproxy-misreports-long-checkpoint-waits-and-idle-config) |
| Boolean parsing, NaN status, partial PUT, duplicated JSON, failing child shown idle | Already tracked | — | TD-519, TD-520, TD-305, TD-453, TD-523 |
