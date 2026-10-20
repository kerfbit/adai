# `TrainerAdminConfig` — Source File Reference

- **Files:** [`src/TrainerAdminConfig.hpp`](../../../../src/TrainerAdminConfig.hpp) (43 lines), [`src/TrainerAdminConfig.cpp`](../../../../src/TrainerAdminConfig.cpp) (83 lines)
- **Namespace:** `adai` (`TrainerAdminConfigUpdate`, `TrainerAdminConfigParse`, `parse_trainer_admin_config()`)
- **Built into:** `adai_core` (with a private include of the vendored `external/nlohmann/json.hpp`)
- **Used by:** [`TrainerAdminAPI::handle_put_config()`](TrainerAdminAPI.md) (a live pass) and [`TrainerServiceProxy::handle_put_config_idle()`](TrainerServiceProxy.md) (no child running)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-10-19` (both files)
- **Tests:** [`tests/trainer_admin_config_test.cpp`](../../../../tests/trainer_admin_config_test.cpp) (3 tests, `trainerAdminConfigTests`), plus the HTTP tests of both callers
- **Last traced against the code:** 2026-10-19

> Code-traced reference, written with the file (TD-519, October 19, 2026).

---

## 1. What this file is

One parser for the body of the trainer's `PUT /admin/config`, shared by the live child's handler and `trainer_service`'s idle handler. That
way both accept and reject exactly the same bodies. It replaced two copies of substring helpers (`json_bool()`, `json_int()`,
`json_has_key()`). Those read `"auto_save_enabled": true`, with a space, as false, and applied keys one by one before validating the rest
(TD-519, TD-305).

| Member | Behaviour |
|---|---|
| `TrainerAdminConfigUpdate` | `std::optional` fields for the four mutable keys: `auto_save_enabled` (bool), `auto_save_every_samples`, `auto_save_every_minutes`, `max_sessions_to_keep` (int). Set only when the body sent them |
| `TrainerAdminConfigParse` | `update` plus `error`: empty when the body is valid, otherwise the reason for a 400 |
| `parse_trainer_admin_config(body)` | Parses with `nlohmann::json::parse(…, allow_exceptions = false)` and validates everything before returning, so a caller that applies the result never applies part of a rejected request |

Rejected, with the message the 400 carries:
- a body that isn't a JSON object (malformed, empty, an array);
- any of the immutable keys `port`, `host`, `dir`, `enabled` ("immutable at runtime; set it via config.trainer.conf");
- `auto_save_enabled` that isn't a JSON boolean (`"true"`, `1` and `null` are all rejected);
- a count that isn't a JSON integer (`"500"`, `1.5`), or is out of range: `auto_save_every_samples`/`auto_save_every_minutes` must be ≥ 0
  (0 disables that trigger), `max_sessions_to_keep` ≥ 1, all within `int`;
- a body with none of the four mutable keys.

Other keys are ignored. Any valid JSON spacing, including pretty-printed bodies, is accepted.

---

## 2. Gaps

None known. Persisting several keys is still one `DaemonConfigStore::set()` per key with no transaction, in the callers (TD-305's other
half). Responses are still hand-written JSON (TD-453).

---

## 3. Tests

`trainer_admin_config_test.cpp` (3): compact, spaced and pretty-printed booleans; reading only the keys sent; and 18 rejected bodies
covering every rule above. The callers' tests check the HTTP behaviour:
- **live API:** `PutConfigAcceptsSpacedAndPrettyPrintedJson`, `PutConfigRejectedBodyChangesNothing`;
- **idle proxy:** `PutConfigWhileIdleParsesSpacedJsonAndRejectsAtomically`.

All three failed on the code before the fix.
