# `MnsServerArgs` — Source File Reference

- **Files:** [`src/MnsServerArgs.hpp`](../../../../src/MnsServerArgs.hpp), [`src/MnsServerArgs.cpp`](../../../../src/MnsServerArgs.cpp) (145 lines together)
- **Namespace:** `adai`
- **Built into:** `mns_server` (with [`src/ModelNameServiceServer.cpp`](../../../../src/ModelNameServiceServer.cpp), its `main()`)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-11` (both files)
- **Tests:** [`tests/mns_server_args_test.cpp`](../../../../tests/mns_server_args_test.cpp) (16 tests; the header calls it `MnsServerArgs_test.cpp`)
- **Siblings:** [MetricsApiServerArgs.md](MetricsApiServerArgs.md) (TD-438), [ChatbotApiServerArgs.md](ChatbotApiServerArgs.md) (TD-240), the same pattern for the other daemons
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real `mns_server`.

---

## 1. What this file is

`mns_server`'s command line and configuration layering, pulled out of `main()` for testing (TD-035).

| Function | Behaviour |
|---|---|
| `parse_mns_server_args(argc, argv)` | `--config`, `--port`/`-p`, `--data-dir`, `--registry-url`, `--registry-group`, `--admin-enabled <bool>`, `--help`/`-h`. Each value is an `optional`, so "not given" is distinguishable. A non-numeric port or any unrecognized argument sets `error` (and `main()` exits **1**); a value-taking flag with no value is reported as "Unknown argument" |
| `parse_bool_flag(value, default)` | `true/1/yes/on` and `false/0/no/off`, case-insensitive; **anything else returns `default`** |
| `resolve_mns_server_config(args, file, overrides)` | `port` and `data_dir`: CLI, else `NAME_SERVICE_PORT`/`NAME_SERVICE_DIR`. `registry_url`/`registry_group`: file (`REGISTRY_SERVER_URL`, `RUN_GROUP` or `"default"`), then saved admin overrides, then CLI. `admin_enabled`: CLI, else `true` |

`main()` resolves `data_dir` from the CLI before opening `<data_dir>/daemon_config.db` for the saved overrides, so,
unlike `metrics_api_server` (TD-439), the admin store follows `--data-dir`. `registry_url` and `registry_group` are the
only admin-mutable settings (`PUT /admin/config`); MNS uses them to proxy dataset-registry requests.

---

## 2. Problems (verified)

### A mistyped `--admin-enabled` leaves the admin endpoint on

`parse_bool_flag()` falls back to the default for unrecognized text, and the default for `--admin-enabled` is `true`.
Started with `--admin-enabled flase`, the real server **kept `PUT /admin/config` enabled**: an unauthenticated request
changed `registry_group` to `"hijacked"` and the server logged `admin config updated`. An operator who meant to lock
the endpoint (it can repoint MNS's registry proxy to another server) gets no warning. (`mns_server` has no
authentication on any endpoint, so this is about honouring an explicit choice, not the only exposure.) An explicit
`--admin-enabled` value that isn't recognized should be an error.

### The port isn't range-checked

`std::stoi` accepts trailing junk and any `int`. `--port 99999` was accepted: the server logged `listening on port
99999` and **listened on 34463** (99999 mod 65536); `8083x` becomes 8083; `0` gives a random port; negative values pass.
Clients configured with `NAME_SERVICE_URL` then can't find MNS, so the trainer and `chatbot_api_server` fall back to
their local architecture. That's the same class as TD-438 and TD-240.

---

## 3. Smaller items (by inspection)

- **"Unknown argument" for a missing value.** `mns_server --port` reports `Unknown argument: --port` (tested as
  intended in `FlagMissingItsValueIsTreatedAsUnknown`, but misleading).
- **`--admin-enabled` has no config key**, unlike `metrics_api_server`'s `METRICS_API_ALLOW_CONTROL`; it can only be set
  on the command line.
- **The header names the test `MnsServerArgs_test.cpp`**; it's `mns_server_args_test.cpp`.

---

## 4. Tests

`mns_server_args_test.cpp` (16 tests): boolean variants and fallback, empty arguments, every flag, `-p`, a non-numeric
port, `--help`, unknown arguments, a missing value, and the layering (file baseline, `RUN_GROUP`, overrides beating the
file, a group override without a URL, CLI beating both, CLI port and data directory).

**Not covered:** out-of-range or partial port numbers; a mistyped `--admin-enabled` (the fallback is tested as intended
behaviour).

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Unrecognized `--admin-enabled` values fall back to `true` (§2) | A typo meant to disable the admin endpoint leaves it open | **Verified** (real server) | [TD-456](../../guides/TECHNICAL_DEBT.md#td-456-mns_server-treats-a-mistyped---admin-enabled-as-true-and-doesnt-range-check---port) |
| Port not range-checked: `99999` → 34463, `8083x` → 8083, `0` → random (§2) | MNS starts where clients can't find it | **Verified** (real server) | [TD-456](../../guides/TECHNICAL_DEBT.md#td-456-mns_server-treats-a-mistyped---admin-enabled-as-true-and-doesnt-range-check---port) |
| Missing value reported as "Unknown argument"; no config key for `--admin-enabled`; test-file name in header (§3) | Minor | By inspection | [TD-456](../../guides/TECHNICAL_DEBT.md#td-456-mns_server-treats-a-mistyped---admin-enabled-as-true-and-doesnt-range-check---port) |
| Same parsing pattern in the other daemons | Already tracked | — | TD-438, TD-240 |
