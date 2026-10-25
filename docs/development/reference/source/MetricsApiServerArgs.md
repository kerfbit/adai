# `MetricsApiServerArgs` — Source File Reference

- **Files:** [`src/MetricsApiServerArgs.hpp`](../../../../src/MetricsApiServerArgs.hpp), [`src/MetricsApiServerArgs.cpp`](../../../../src/MetricsApiServerArgs.cpp) (198 lines together)
- **Namespace:** `adai`
- **Built into:** `metrics_api_server` (compiled into the executable, not a library); used only by its `main()` in [`src/TrainingMetricsAPIServer.cpp`](../../../../src/TrainingMetricsAPIServer.cpp)
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-09-11` (both files)
- **Tests:** [`tests/metrics_api_server_args_test.cpp`](../../../../tests/metrics_api_server_args_test.cpp) (13 tests; the header calls it `MetricsApiServerArgs_test.cpp`)
- **Sibling:** [ChatbotApiServerArgs.md](ChatbotApiServerArgs.md) (same pattern, same `atoi` problems: TD-240, TD-245)
- **User docs:** [TRAINING_METRICS_API.md](../../TRAINING_METRICS_API.md) (CLI options), [OPERATIONS_MANUAL.md](../../../operations/OPERATIONS_MANUAL.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linked
> against the real functions, and by running the real `metrics_api_server`.

---

## 1. What this file is

`metrics_api_server`'s configuration resolution, pulled out of `main()` for testing (TD-035). Settings
come from three layers, lowest to highest precedence:

```text
config.metrics.conf  →  seed_metrics_server_config_from_file()
daemon_config.db     →  apply_metrics_admin_overrides()      (values saved by PUT /admin/config)
argv                 →  parse_metrics_api_server_args()
```

`main()` runs them in that order, after a first pass over `argv` that extracts `--config PATH` (used
for `ConfigLoader::discover_config_path()`).

### `MetricsApiServerConfig`

| Field | Default | Flag | Admin-overridable |
|---|---|---|---|
| `port` | 8081 | `--port` | — |
| `enable_persistence` | true | `--no-persistence` | — |
| `enable_prometheus` | false | `--enable-prometheus` | yes |
| `allow_control` | true | `--no-control` | — |
| `metrics_file`, `summary_file`, `prometheus_file` | `training_sessions/…` | `--metrics-file`, `--summary-file`, `--prometheus-file` | — |
| `persist_every_samples` / `persist_every_seconds` | 100 / 30 | `--persist-samples` / `--persist-seconds` | yes |
| `max_records_in_memory` / `max_records_on_disk` | 10,000 / 100,000 | `--max-memory-records` / `--max-disk-records` | yes |
| `max_live_sessions` (`size_t`) | 16 | `--max-live-sessions` | yes |
| `completed_ttl_seconds`, `sweep_interval_seconds`, `staleness_threshold_seconds` | 3600, 60, 60 | `--completed-ttl-seconds`, `--sweep-interval-seconds`, `--staleness-threshold-seconds` | yes |
| `name_service_url` | `http://localhost:8083` | `--name-service-url`, `--no-name-service` | — |
| `storage_backend`, `db_path`, `db_url`, `db_pool_size` | `sqlite+file`, `training_sessions/metrics.db`, empty, 4 | `--storage-backend`, `--db-path`, `--db-url`, `--db-pool-size` | — |

The admin-overridable column matches `TrainingMetricsAPI`'s `PUT /admin/config` allow-list.

---

## 2. Functions

| Function | Behaviour |
|---|---|
| `seed_metrics_server_config_from_file(ServiceConfig)` | Copies the 20 `metrics_*` (and `name_service_url`) fields. No validation |
| `apply_metrics_admin_overrides(config, map)` | For the nine admin keys: `std::stoi` (or `std::stoull` for `max_live_sessions`), **silently skipping** values it can't parse at all. `enable_prometheus` is `true` only for the literal `"true"` |
| `parse_metrics_api_server_args(argc, argv, config)` | Flags in place; `--help`/`-h` short-circuits; anything unrecognized, **including a value-taking flag with no value**, returns `error` with `"Unknown option: <flag>"`. `--config` and its value are skipped. Numbers use `std::atoi`/`std::strtoul` |

`main()` prints usage and **exits 0** on a parse error ("matches original parse_args() behavior").

---

## 3. Problems (verified)

### Numeric flags accept anything, and a bad port silently picks a random one

`std::atoi` returns 0 for non-numbers and stops at the first non-digit, and nothing checks ranges
afterwards. Probed with the real parser:

| Input | Result |
|---|---|
| `--port abc` | `port = 0`. **The real server then listened on a random ephemeral port** (39915 in the run), while printing `Server starting on http://0.0.0.0:0`. The dashboard, trainers (`METRICS_SERVER_URL`) and MNS can't find it, and nothing reports an error |
| `--port 99999` | Accepted; the server printed `:99999` and **listened on 34463** (99999 mod 65536) |
| `--port 8081x`, `--persist-seconds 1e3` | 8081 and **1** (truncated at the first non-digit) |
| `--port -5`, `--db-pool-size -3`, `--persist-samples 0`, `--sweep-interval-seconds 0` | Accepted as given. A zero sweep interval **disables the eviction sweep** (`MetricsSessionRegistry` only starts it when `> 0`), so completed sessions are never evicted |
| `--max-live-sessions -1` | `strtoul` wraps it to **18,446,744,073,709,551,615**: no session cap |
| `--storage-backend postgress` | Accepted; the registry logs a warning and falls back to file-only storage, while startup prints `Storage backend: postgress` |

The admin overrides have the same gaps: `"-1"` for `max_live_sessions` also wraps (`stoull`), `"50abc"`
becomes 50, and a negative `completed_ttl_seconds` is applied. (`PUT /admin/config`'s own lack of range
checks is TD-306; these are the values it saved being re-applied at the next start.)

### Errors exit 0

`metrics_api_server --bogus` prints usage and **exits 0** (verified). Under systemd that's a successful
exit: `Restart=on-failure` (the installer's unit) won't retry, and the unit shows as inactive rather
than failed, so a mistyped flag in `ExecStart` looks like a clean stop.

---

## 4. Gotchas (by inspection)

- **CLI flags beat saved admin settings.** The installer's unit (`scripts/install_metrics_service.sh`)
  passes `--port`, `--metrics-file`, `--summary-file` and `--db-path`, none of them admin-mutable, so
  that's fine today. But any admin-mutable flag added to `ExecStart` would silently undo a `PUT
  /admin/config` on every restart, though `GET /admin/config` showed the new value until then.
- **The admin store's location ignores `--db-path`.** `main()` puts `daemon_config.db` in the directory
  of the **file-configured** `db_path`, before CLI flags are parsed. With the installer's defaults that's
  the same directory (`<install>/training_sessions`). With a custom `--metrics-dir`, the store lands in
  `<install>/training_sessions`, which `ProtectSystem=strict` with `ReadWritePaths=<metrics dir>` makes
  read-only. The store then fails to open (startup prints a warning) and admin settings can't persist.
- **Two boolean conventions**: `enable_prometheus` overrides accept only `"true"`, unlike `ConfigLoader`'s
  parser (TD-286).
- **Header comment** names the test file `MetricsApiServerArgs_test.cpp`; it's
  `metrics_api_server_args_test.cpp`.

---

## 5. Tests

`metrics_api_server_args_test.cpp` (13 tests): seeding copies every field; overrides (empty, every key,
unparseable value skipped, `enable_prometheus` only on `"true"`); parsing (no arguments, `--help`, unknown
option, `--config` skipped, booleans, storage/DB flags, `--max-live-sessions` unsigned, CLI beating earlier
layers).

**Not covered:** non-numeric, partial, negative or out-of-range values; a flag missing its value.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `atoi`/`strtoul` with no validation: `--port abc` → random port, `99999` → 34463, `-1` sessions → unlimited, `0` sweep → no eviction, partial numbers truncated; same gaps in the saved admin overrides (§3) | A typo starts the server somewhere clients can't find, or silently removes its limits | **Verified** (real server) | [TD-438](../../guides/TECHNICAL_DEBT.md#td-438-metrics_api_server-accepts-invalid-numbers-and-exits-0-on-bad-arguments) |
| Parse errors exit 0; a missing flag value reads as "Unknown option" (§3) | systemd treats a bad `ExecStart` as a clean stop and doesn't restart | **Verified** | [TD-438](../../guides/TECHNICAL_DEBT.md#td-438-metrics_api_server-accepts-invalid-numbers-and-exits-0-on-bad-arguments) |
| Unknown `--storage-backend` accepted (falls back to file-only with a log warning) (§3) | Metrics DB silently absent | **Verified** (parser); by inspection (fallback) | [TD-439](../../guides/TECHNICAL_DEBT.md#td-439-metrics_api_server-backend-name-and-admin-store-location-gaps) |
| Admin store located from the file's `db_path`, not `--db-path`; CLI flags silently undo saved admin settings; test-file name in header (§4) | Admin settings can't persist with a custom installer `--metrics-dir`; confusing precedence | By inspection | [TD-439](../../guides/TECHNICAL_DEBT.md#td-439-metrics_api_server-backend-name-and-admin-store-location-gaps) |
| No range checks on `PUT /admin/config`; two boolean parsers | Already tracked | — | TD-306, TD-286 |

---

## Addendum (October 24, 2026): `--cors-allowed-origins` (TD-526)

`MetricsApiServerConfig::cors_allowed_origins` is seeded from `METRICS_API_CORS_ALLOWED_ORIGINS` and overridden by
`--cors-allowed-origins LIST`; it isn't admin-mutable. See TrainingMetricsAPI.md.

