# `TrainingMetricsAPIServer` — Source File Reference

- **File:** [`src/TrainingMetricsAPIServer.cpp`](../../../../src/TrainingMetricsAPIServer.cpp) (333 lines; `main()` of `metrics_api_server`)
- **Binary:** `metrics_api_server` (port 8081); flags in [MetricsApiServerArgs.md](MetricsApiServerArgs.md), HTTP layer in [TrainingMetricsAPI.md](TrainingMetricsAPI.md), user-facing catalogue in [TRAINING_METRICS_API.md](../../TRAINING_METRICS_API.md)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-11`
- **Tests:** none for `main()`; its pieces are tested (`metricsApiServerArgsTests`, `metricsSessionRegistryTests`, the TrainingMetricsAPI route and live tests)
- **Last traced against the code:** 2026-10-17

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real binary.

---

## 1. What this file is

The startup and shutdown sequence of the metrics daemon.

1. **Config precedence:** find `--config` first, load `config.metrics.conf`, seed `MetricsApiServerConfig` from it
   (`seed_metrics_server_config_from_file()`), overlay persisted admin overrides from `daemon_config.db` in the directory of the **file's** `db_path`
   (TD-439), then apply CLI flags (`parse_metrics_api_server_args()`). `--help` prints usage and exits 0; a bad argument prints usage and also exits 0
   (TD-438).
2. **Services:** build `MetricsServiceConfig`, create the `MetricsSessionRegistry` (storage backend, DB path/URL, pool, live-session cap, TTL, sweep
   interval) and the `0-default` session (exit 1 if that fails), and print a configuration banner to stdout.
3. **API:** construct `TrainingMetricsAPI(registry, port, allow_control, name_service_url, admin_dir)`, install SIGINT/SIGTERM handlers that only set an
   atomic flag (no SIGHUP handler, TD-283), and run `api->start()` on a thread.
4. **Main loop:** sleep in 100 ms steps until the flag is set or the server thread reports a bind failure.
5. **Shutdown:** stop the API, join the thread, flush `0-default` and every other live session to disk; return 1 if the bind failed, else 0.

Logging uses `adai::Logger`'s lazy console default (the binary never calls `Logger::init()`, and `config.metrics.conf` has no log keys), which under
systemd goes to the journal.

**Verified:** 40 starts each followed by a SIGTERM 0–90 ms later all exited (most were killed by the default action before the handlers were installed;
none hung). A bind failure exits 1, unlike `registry_server` (TD-494).

---

## 2. Gaps (by inspection)

No defects of its own beyond those already tracked, plus small items, tracked as
[TD-528](../../guides/TECHNICAL_DEBT.md#td-528-metrics_api_server-stopstart-race-stale-help-text-and-default-session-slot):

- **Stop/start race:** `TrainingMetricsAPI::stop()` returns early unless `running_` is set, and `start()` sets it on the server thread. A SIGTERM
  between the handlers being installed and `start()` running would make `stop()` a no-op and `join()` wait for a server that then runs forever (the
  `mns_server` TD-464 race). Not reproduced in 40 trials; the window is very small.
- **Help and banner text drift:** `--max-disk-records` is listed though nothing enforces it (TD-440); the endpoint lists (printed twice, in `--help`
  and the startup banner) cover the legacy flat routes and omit most session-scoped, compare, export, db-history, LeJEPA and admin routes; the banner
  lists `/admin/config` only in `--help`.
- **The `0-default` session is created at startup and counts toward `max_live_sessions`**, so the effective cap for trainers is one less than
  configured (relevant to TD-445).

| Item | Tracked as |
|---|---|
| Bad arguments exit 0; unchecked numbers | TD-438 |
| Admin store located from the file's `db_path`, not `--db-path`; unknown backend accepted | TD-439 |
| SIGHUP terminates the process | TD-283 |
| Retention / `max_records_on_disk` not enforced | TD-440 |
| Session cap filled by finished sessions | TD-445 |

---

## 3. Tests

None run `main()`. The pieces it wires together have their own suites (argument parsing, registry, API routes and live tests).

---

## Addendum (October 25, 2026): retention flags (TD-440)

`--max-disk-records` is enforced now, and there are three new flags: `--retention-days`, `--metrics-file-max-mb` and
`--metrics-file-backups` (config keys `METRICS_RETENTION_DAYS`, `METRICS_FILE_MAX_MB`, `METRICS_FILE_BACKUPS`). The startup banner
prints the retention settings. TD-528's "unenforced flag" item no longer applies.

## Addendum (October 25, 2026): stop before listen, endpoint list, default session (TD-528)

- `TrainingMetricsAPI::start()`/`stop()` hand off through `starting_`/`stop_requested_`, so a SIGTERM during startup ends the
  server instead of being lost. `start()` binds, then calls `listen_after_bind()`.
- `print_endpoint_summary()` is the one endpoint list for `--help` and the banner; the full list is TRAINING_METRICS_API.md.
- `0-default` doesn't take a `max_live_sessions` slot.

Tests: `StopBeforeStartIsNotLost`, `StopRacingStartAlwaysEndsIt`, `DefaultSessionDoesNotTakeALiveSlot`.
