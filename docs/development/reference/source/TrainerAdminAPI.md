# `TrainerAdminAPI` — Source File Reference

- **Files:** [`src/TrainerAdminAPI.hpp`](../../../../src/TrainerAdminAPI.hpp) (101 lines), [`src/TrainerAdminAPI.cpp`](../../../../src/TrainerAdminAPI.cpp) (337 lines)
- **Namespace:** `adai`
- **Used by:** `incremental_trainer --admin-port <N> resume` (the single-pass child `trainer_service` launches; bound to `127.0.0.1:TRAINER_CHILD_ADMIN_PORT`). `trainer_service`'s own public listener, `TrainerServiceProxy`, proxies `/admin/*` to it (see CLAUDE.md, "Incremental trainer admin API")
- **State:** an in-process [`TrainerControlState`](../../../../src/TrainerControlState.hpp) shared with that pass's `IncrementalTrainer`
- **Status tag:** `@adai-status: stable`, `TrainerAdminAPI.cpp` `@adai-version: 1.1.0`, `@adai-reviewed: 2026-10-19`; `.hpp` `1.0.0`, `2026-09-10`
- **Tests:** [`tests/trainer_admin_api_test.cpp`](../../../../tests/trainer_admin_api_test.cpp) (16 tests, `trainerAdminApiTests`; all pass; the shared parser also has `trainerAdminConfigTests`, see [TrainerAdminConfig.md](TrainerAdminConfig.md))
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed by starting the real class on a local port and calling it with `curl`.

---

## 1. What this file is

A small httplib server exposing one training pass's control state.

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Always `{"status":"ok"}` |
| `GET /admin/config` | `auto_save_enabled`, `auto_save_every_samples`, `auto_save_every_minutes`, `max_sessions_to_keep` from the control state |
| `PUT /admin/config` | Rejects `port`/`host`/`dir`/`enabled` (400); validates samples/minutes ≥ 0 and sessions ≥ 1; applies each key as it goes (TD-305); persists all four to `daemon_config.db`; logs the change |
| `GET /admin/status` | Phase, paused, run/session/model, epoch progress, samples this pass, last and best loss, checkpoint count/path/time |
| `GET /admin/logs` | The control state's log ring buffer |
| `POST /admin/checkpoint[?wait_ms=N]` | 409 when idle; else sets `checkpoint_requested` and, with `wait_ms`, polls every 50 ms until `checkpoints_written` increases; 202 with `completed` |
| `POST /admin/pause`, `POST /admin/resume` | Set/clear `paused` (resume also wakes the loop); 202 |

`start()` opens `<config_store_dir>/daemon_config.db`, overlays the persisted tunables onto the control state (a failure only logs a warning), then
blocks in `listen()` and returns false if the bind fails. `stop()` is idempotent. `PUT /admin/config` bodies are parsed as JSON by the shared `parse_trainer_admin_config()` (TD-519); responses are still hand-written JSON (TD-453).

---

## 2. Problems

### Fixed: `{"auto_save_enabled": true}` turned auto-save **off** (TD-519, October 19, 2026)

The old `json_bool()` compared the four characters right after `"auto_save_enabled":` with `true`. With a space after the colon, the default
output of Python's `json.dumps`, it saw ` tru` and returned false. `PUT {"auto_save_enabled": true}` answered `"auto_save_enabled":false` and
**persisted it to `daemon_config.db`**, so every later pass ran without auto-save. `trainer_service`'s idle handler had a copy of the same
parser. A body rejected on a later key had also already applied its earlier keys (TD-305).

Both handlers now use `parse_trainer_admin_config()` ([TrainerAdminConfig.md](TrainerAdminConfig.md)), which:
- reads any valid JSON spacing;
- rejects a non-boolean `auto_save_enabled`, non-integer or out-of-range counts, and malformed JSON with 400;
- validates the whole body before applying anything.

`PutConfigAcceptsSpacedAndPrettyPrintedJson` and `PutConfigRejectedBodyChangesNothing` both failed on the old code.

### A NaN loss makes `/admin/status` invalid JSON (verified)

`last_loss` (and `best_loss`) are streamed as raw doubles. With `last_loss = NaN` (a diverged pass) the response contained `"last_loss":nan` and
Python's `json.load` failed at that character, so the Android dashboard and any other client lose the whole status exactly when it matters (the same
class as TD-443/TD-458). `best_loss` is only written for finite values, but its initial `0.0` reads as a perfect loss before validation runs.

### Smaller items (by inspection)

- **Stale header comment:** it describes the class as the "always-on HTTP admin daemon for `incremental_trainer serve`"; `serve` was removed by TD-172,
  and the class now lives only for one child pass behind `trainer_service`'s proxy.
- **`wait_ms` is unbounded**: each waiting request holds an httplib worker thread for as long as the caller asks, so a few long waits can exhaust the
  pool and make the admin API unresponsive.
- **Checkpoint completion is judged by the counter**, so an auto-save that happens to land during the wait is reported as the requested checkpoint.
- **Persisted overrides aren't range-checked on load** (only `PUT` validates), and a failed bind is logged by the caller, not here.

---

## 3. Tests

`trainer_admin_api_test.cpp` (14): health, config defaults, update and persistence across restart, immutable-key and invalid-value rejection, status
mirroring the control state, logs (startup entry, direct entries, admin actions), checkpoint (409 when idle, flag set, `wait_ms` observing completion),
pause and resume.

**Not covered:** whitespace in request JSON, non-finite losses, long `wait_ms`.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `json_bool()` reads `"key": true` (with a space) as false; persisted (§2) | A normal JSON request silently disabled auto-save for every later pass | **Fixed October 19, 2026** (was verified) | [TD-519](../../guides/TECHNICAL_DEBT.md#td-519-trainer-admin-put-reads-a-spaced-true-as-false-and-disables-auto-save) |
| NaN `last_loss` streamed as `nan` (§2) | Invalid status JSON during divergence | **Verified** | [TD-520](../../guides/TECHNICAL_DEBT.md#td-520-trainer-admin-status-emits-nan-losses-as-invalid-json) |
| Stale "serve" header; unbounded `wait_ms`; counter-based completion; unchecked persisted values (§2) | Misleading docs; admin API can be tied up | By inspection | [TD-521](../../guides/TECHNICAL_DEBT.md#td-521-traineradminapi-stale-header-unbounded-wait_ms-and-checkpoint-accounting) |
| Hand-written JSON responses; non-atomic multi-key persistence | Already tracked | — | TD-453, TD-305 (its validate-first half fixed here October 19, 2026) |

## Addendum (October 19, 2026): JSON request parsing (TD-519)

`handle_put_config()` now parses the body with the shared `parse_trainer_admin_config()` (`src/TrainerAdminConfig.{hpp,cpp}`), built on the vendored `nlohmann/json`. `TrainerServiceProxy`'s idle handler uses the same parser. A spaced `"auto_save_enabled": true` now reads as true. These are rejected with 400: a non-boolean `auto_save_enabled`, non-integer or out-of-range counts, and malformed JSON. The whole body is validated before anything is applied, so a rejected PUT changes nothing (TD-305's validate-first half). Persistence is still one `set()` per key (TD-305).

---

## Addendum (October 24, 2026): browser write guard (TD-550)

This server installs `install_browser_write_guard()` ([BrowserWriteGuard.md](BrowserWriteGuard.md)) as its pre-routing handler. A write
(POST/PUT/PATCH/DELETE) carrying any `Origin` gets 403, and a write with a `text/plain`, form-encoded or multipart body gets 415. A web
page open in an operator's browser can no longer make it execute writes. Clients that aren't browsers send no `Origin` and are unaffected
as long as they send JSON (or, for `registry_server`'s `upload`, `application/octet-stream`).

## Addendum (October 26, 2026): test harness no longer aborts under load (TD-556)

`tests/trainer_admin_api_test.cpp` runs each server through an RAII thread holder (stop and join on destruction) on kernel-assigned ports, with a 10 s readiness
wait that stops when `start()` fails. Its fixed ports in the ephemeral range used to be taken under a parallel `ctest`, and the failed
`ASSERT` then left a joinable `std::thread` that aborted the whole binary.
