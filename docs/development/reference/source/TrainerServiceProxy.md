# `TrainerServiceProxy` — Source File Reference

- **Files:** [`src/TrainerServiceProxy.hpp`](../../../../src/TrainerServiceProxy.hpp) (118 lines), [`src/TrainerServiceProxy.cpp`](../../../../src/TrainerServiceProxy.cpp) (379 lines)
- **Namespace:** `adai`
- **Used by:** `trainer_service` ([TrainerServiceMain.md](TrainerServiceMain.md)): its public admin listener on `TRAINER_ADMIN_HOST:TRAINER_ADMIN_PORT` (127.0.0.1:8084 shipped), started only with `TRAINER_ADMIN_ENABLED=true`. Clients: `curl`, the Android ops dashboard's Trainer tab (via the `trainer.kerfbit.dev` tunnel)
- **Status tags:** `@adai-status: beta`; `TrainerServiceProxy.cpp` `@adai-version: 0.8.0`, `.hpp` `0.4.1`; `@adai-reviewed: 2026-11-11`
- **Tests:** [`tests/trainer_service_proxy_test.cpp`](../../../../tests/trainer_service_proxy_test.cpp) → `trainerServiceProxyTests` (26, all pass)
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **verified** was confirmed with the real proxy, in front of a real [`TrainerAdminAPI`](TrainerAdminAPI.md)
> child or, on November 11, 2026, inside the real `trainer_service` with a stub pass.

---

## 1. What this file is

The always-on admin endpoint of `trainer_service` (TD-172). While a pass is alive (`set_child_port(N)`), requests go to the pass's own
`TrainerAdminAPI` on `127.0.0.1:N`; while none is (`set_child_port(0)`), the proxy answers itself. The HTTP server is hidden behind a pimpl
(`ServerImpl`). It installs the browser write guard (TD-550: a write with an `Origin` gets 403, a preflight-free `Content-Type` 415) and binds
without `SO_REUSEPORT` (TD-551).

| Endpoint | Pass alive | No pass |
|---|---|---|
| `GET /health` | `{"status":"ok"}`, never proxied | Same |
| `GET /admin/config` | Proxied | The constructor's `TrainerAdminDefaults` (the `config.trainer.conf` values, TD-525), overlaid with `TRAINER_ADMIN_DIR/daemon_config.db` through `parse_trainer_admin_overrides()`; a saved value `PUT` would reject is skipped with a warning (TD-521); an unreadable store is a warning and the defaults |
| `PUT /admin/config` | Proxied | Parsed by the shared `parse_trainer_admin_config()` (TD-519; the same rules as the pass), 400 with the reason; else every key saved with one `DaemonConfigStore::set_many()` under `idle_config_mutex_` (TD-305), 500 "nothing was changed" if that fails; then answers like the idle `GET` |
| `GET /admin/status` | Proxied; every answer that is a JSON object, a 503/504 included, gets the supervisor fields (TD-571) | The idle body (below) with the supervisor fields |
| `GET /admin/logs[?limit=N]` | Proxied, the limit rebuilt from the number | `{"entries":[]}`; `limit` must be 0–200 (`kMaxAdminLogEntries`), else 400 even with no pass (TD-522) |
| `POST /admin/checkpoint[?wait_ms=N]` | Proxied with a read timeout of `wait_ms` + 5 s; `wait_ms` 0–60,000 (`kMaxCheckpointWaitMs`, the pass's own limit, TD-521), else 400 | 409 "no active training pass" |
| `POST /admin/pause` | Sets `TrainerServiceControlState::paused` and `wake()`s the loop (TD-173), then asks the pass to pause; a refusal is logged; 202 `{"paused":true}` | Same, without the pass |
| `POST /admin/resume` | Clears `paused`, `wake()`, asks the pass to resume (a refusal is logged); 202 `{"paused":false}` | Same, without the pass |

**Proxying** (`proxy(method, path, body, read_timeout_ms)`): a new `httplib::Client` per request to `127.0.0.1:<child port>`, 1 s to connect,
5 s to read by default. A refused or timed-out connection is **503** "training pass is starting or exiting; try again shortly"; any other failure
(the pass took the request but didn't answer in time) is logged and is **504** "did not answer within N s; the request may still have been
applied" (TD-525). Any answer, whatever its status, is relayed unchanged, except that a status answer gets the supervisor fields (below).

**The idle status body** has the per-pass shape with a fresh pass's values: `phase "idle"`, empty ids, zero counters, `last_loss`/`best_loss`
`null` and `loss_diverged false` (TD-520). Its `paused` is the supervisor's flag; during a pass, `paused` is the pass's own.

**The supervisor fields** (`with_supervisor_fields()`, appended before the closing `}` of a body that ends with one): `supervisor_paused`,
`total_passes_launched`, `total_passes_did_work`, `total_passes_crashed`, `total_passes_failed`, `consecutive_failures`, `last_failure_unix`,
`total_passes_stalled`, `seconds_since_pass_progress` (TD-569), `last_exit_code`, `service_uptime_seconds`. The meanings are in
[TrainerServiceControlState.md](TrainerServiceControlState.md). The idle `PUT`'s 400 and 500 bodies carry variable text (the parser's reason,
an exception message) and are built with nlohmann/json (`error_json()`, TD-571); the other error bodies are fixed text.

---

## 2. Problems

### Fixed: the service's own status was hidden while a pass's admin API wasn't up (TD-571, November 11, 2026; was verified)

`/admin/status` added the supervisor fields only to a 200, and a pass started its `TrainerAdminAPI` only after GPU initialization. So a pass
stuck there (which TD-569 stops after `TRAINER_PASS_STALL_MINUTES`) showed only `{"error":"training pass is starting or exiting; try again
shortly"}` with 503, without `seconds_since_pass_progress` or the pass counters; seen with the real `trainer_service` and a stub pass. **Now**
the fields go on every status answer that is a JSON object, a 503/504 included, and the pass starts its admin API before GPU initialization
([IncrementalTrainingTool.md](IncrementalTrainingTool.md)). The same review moved the idle `PUT`'s error bodies from a five-character
`json_escape()` to nlohmann/json.

### Notes

- **A new client and connection per proxied request.** It is cheap on loopback, and a request is rare.

---

## 3. Tests

`trainer_service_proxy_test.cpp` (26), against the real proxy and, for the proxied cases, a real `TrainerAdminAPI`:

- `/health` never proxied;
- idle status, logs, checkpoint (409) and config (the config-file defaults, TD-525);
- idle `PUT`: persists and reads back, parses spaced JSON and rejects a body as a whole, stores nothing when the save fails, rejects an
  immutable key;
- status and `PUT` proxied verbatim to a live pass; `?limit=` checked and passed on;
- an unreachable pass port → 503; a pass that never answers → 504; port 0 → idle again;
- pause persisting, waking a sleeping supervisor and reaching a live pass; resume; the idle status showing the real pause state;
- valid JSON idle and with a diverged pass (`null` losses);
- every supervisor field in `/admin/status`, TD-569's included;
- a checkpoint wait longer than 5 s relayed; a malformed or oversized `wait_ms` → 400;
- an unreachable pass's 503 carrying the supervisor fields (TD-571).

---

## 4. Known gaps and gotchas (summary)

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Supervisor fields only on a 200; a pass's admin API started after GPU initialization; a five-character `json_escape()` (§2) | — | **Fixed November 11, 2026** | [TD-571](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-571-findings-from-the-november-11-2026-trace-of-the-files-td-569-touched) |
| Long checkpoint waits misreported, idle config ignoring the config file, unencoded `wait_ms`, ignored pause/resume failures | — | **Fixed November 2, 2026** | [TD-525](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-525-trainerserviceproxy-misreports-long-checkpoint-waits-and-idle-config) |
| Spaced JSON booleans, partial idle saves, `null` losses, `?limit=`, the shared `wait_ms` limit, a failing pass looking idle | — | **Fixed** | TD-519, TD-305, TD-520, TD-522, TD-521, TD-523 |
| Browser writes, a second instance on the same port, a test harness abort | — | **Fixed** | TD-550, TD-551, TD-556 |

## 5. History

- **October 19, 2026:** shared `PUT` parser (TD-519); failure fields (TD-523).
- **October 24–26, 2026:** browser write guard (TD-550); exclusive port (TD-551); test harness (TD-556).
- **November 2–3, 2026:** checkpoint waits, 503 vs 504, idle defaults, logged pause/resume refusals (TD-525); shared `wait_ms` limit and checked
  saved settings (TD-521); `null` losses (TD-520); `?limit=` (TD-522); idle saves in one transaction (TD-305).
- **November 11, 2026:** `total_passes_stalled` and `seconds_since_pass_progress` (TD-569); the fields on every status answer and nlohmann
  error bodies (TD-571). This document was re-traced against the code, replacing its October sections and folding in its addenda.
