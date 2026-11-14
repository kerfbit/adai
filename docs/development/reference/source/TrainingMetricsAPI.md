# `TrainingMetricsAPI` — Source File Reference

- **Files:** [`src/TrainingMetricsAPI.hpp`](../../../../src/TrainingMetricsAPI.hpp) (222 lines), [`src/TrainingMetricsAPI.cpp`](../../../../src/TrainingMetricsAPI.cpp) (2,884 lines)
- **Namespace:** global
- **Built into:** `adai_metrics_api`; served by `metrics_api_server` (port 8081, `0.0.0.0`; see [MetricsApiServerArgs.md](MetricsApiServerArgs.md)). Public through `metrics.kerfbit.dev` behind Cloudflare Access
- **Clients:** [`MetricsPushClient`](MetricsPushClient.md) (trainer pushes), the Android ops dashboard, the Tizen watch app, Prometheus scrapes
- **Endpoint reference (user-facing, kept):** [TRAINING_METRICS_API.md](../../TRAINING_METRICS_API.md)
- **Status tag:** `@adai-status: beta` ("capped by TD-039 — large, actively evolving"), `@adai-version: 0.18.0` (hpp) / `0.19.0` (cpp), `@adai-reviewed: 2026-11-14`
- **Tests:** [`tests/training_metrics_api_routes_test.cpp`](../../../../tests/training_metrics_api_routes_test.cpp) (47, `trainingMetricsApiRoutesTests`), [`tests/training_metrics_api_live_test.cpp`](../../../../tests/training_metrics_api_live_test.cpp) (44, `trainingMetricsApiLiveTests`, against a running server); all pass
- **Last traced against the code:** 2026-10-17, with the addenda below

> Code-traced reference. Behaviour marked **verified** was confirmed against the real `metrics_api_server` binary with `curl`.

---

## 1. What this file is

The HTTP layer over [`MetricsSessionRegistry`](MetricsSessionRegistry.md) and its per-session `TrainingMetricsService`s: trainers push metrics into a
session, dashboards read them, and a few admin/control routes manage the store. The full request/response catalogue lives in
TRAINING_METRICS_API.md; this file describes how the code implements it.

### Structure

| Part | Behaviour |
|---|---|
| Session keys | `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`, enforced by the route regex and `is_valid_session_key()` |
| `resolve_session_service(key, create)` | Invalid key → 400; `create` → `create_or_get_session()` (null at the live-session cap → **503 `metrics_server_full`**, TD-445); otherwise unknown key → **404** |
| Session-scoped routes | `/api/sessions/{key}/…`: reads (`metrics/current`, `summary`, `history`, `status`, `epochs`, `abnormal`, `generation-quality`, `padding-efficiency`, `lejepa`, `prometheus`, `csv`, `db-history`, `export`), pushes (`start`, `end`, `heartbeat`, `epoch/start`, `epoch/end`, `metrics/sample`, `validation`, `best`, `advanced`, `layer-gradient-norms`, `generation-quality`) and control (`control/flush`, `control/clear`) |
| Legacy aliases | The flat `/api/metrics/*`, `/api/session/*`, `/api/epoch/*`, `/api/control/*` routes act on `0-default`, with deprecation headers |
| Cross-session routes | `GET /api/sessions[?status=&from=]`, `/api/metrics/aggregate`, `/api/metrics/prometheus/aggregate` (TD-021), `/api/metrics/compare` (TD-020), `/api/models` (MNS proxy; 502 on failure), `/health` (TD-019 staleness-aware) |
| Control and admin | `control/flush`, `control/clear` and `GET`/`PUT /admin/config` exist only when `allow_control` (`METRICS_API_ALLOW_CONTROL`, default true); admin settings persist to `daemon_config.db` |
| Error mapping | `ApiRequestError` → its status; other exceptions → 500 for reads, 400 for pushes (`stoi`/`stof` failures); `create_error_response()` passes pre-built JSON through |
| Parsing / output | Pushes parse fields by substring + `stoi`/`stof`; responses are built with `ostringstream` and `escape_json()` |
| CORS | A post-routing handler adds `Access-Control-Allow-Origin: *` (methods `GET, POST, OPTIONS`) to every response; `OPTIONS .*` answers 204 |

---

## 2. Problems

### Any web page can write to (and read) the metrics server through an operator's browser (verified)

Every response carries `Access-Control-Allow-Origin: *`, there is no authentication, and push and control routes accept any body regardless of
`Content-Type`. A `POST` with `Content-Type: text/plain` is a CORS "simple request", sent by browsers without a preflight. From a fake origin
(`Origin: http://evil.example`, `text/plain`):

- `POST /api/sessions/{key}/control/clear` → **200 "Metrics history cleared"**.
- `POST /api/sessions/{key}/metrics/sample` with `{"sample":5,"loss":99.0}` → **200**, recorded.

So any site visited from a browser on the LAN (or by someone signed in to Cloudflare Access) can clear histories, inject fake losses or end sessions,
and because the wildcard origin also allows reading responses, read everything, including `metrics/abnormal`'s `input_text`/`target_text` (raw
training data). TRAINING_METRICS_API.md notes the wildcard CORS "to support browser-based dashboards" and recommends a reverse proxy for access
control, but a proxy doesn't stop requests made by a browser that is already inside it. Restricting CORS to the dashboard origins and requiring
`application/json` (which forces a preflight) on writes would close the browser path.

### Every dashboard poll writes an INFO log line (verified)

`GET /api/sessions/{key}/metrics/current` logs `[metrics/current] matched key='…' len=…` at INFO on every request; five polls added five lines. The
dashboards poll this route continuously, so the log fills with it (and rotation, TD-430/TD-431, discards real history sooner).

### Smaller items (by inspection)

- **Parsing:** pushes find keys by substring and parse with `stoi`/`stof` (`nan`/`inf` accepted and stored, TD-443); a malformed number returns 400
  with the bare exception text (`stof`).
- **`history` and `max_records`** aren't capped (`?max_records=100000000` builds the whole history in one response); history timestamps are whole
  seconds.
- **`handle_session_status()`** switches the stream to fixed precision part-way through, so `samples_per_second` is rounded to 2 decimals.
- **Size:** 2,760 lines of hand-built JSON and per-route boilerplate (each route repeats the same try/catch), the reason for its `beta` cap (TD-039).

### Already tracked

| Item | Tracked as |
|---|---|
| NaN/inf emitted as invalid JSON; LR rounding in pushes | TD-443 |
| Finished sessions fill the cap → 503 for new runs | TD-445 |
| `db-history` returns the oldest rows; unbounded compare/export | TD-441 |
| No retention; `max_records_on_disk` ignored | TD-440 |
| Admin PUT partial apply and unchecked ranges | TD-305, TD-306 |
| Hand-written JSON duplicated across daemons | TD-453 |
| Server CLI parsing | TD-438, TD-439 |

---

## 3. Tests

`training_metrics_api_routes_test.cpp` (18, in-process): the sessions list and aggregate, 404 for unknown keys on reads and pushes, 409 for a duplicate active start, 503 when the registry is full, layer-gradient and LeJEPA round trips, `model_id` injection into the config snapshot.
`training_metrics_api_live_test.cpp` (44, against a running server; skipped without one): the push lifecycle, every read route's shape, filters,
aggregate and Prometheus output, control routes, admin config.

**Not covered:** cross-origin requests, `text/plain` bodies, large `max_records`.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Wildcard CORS + no auth + writes accepting `text/plain` (§2) | Any web page in an operator's browser can clear/inject metrics and read training text | **Verified** | [TD-526](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-526-metrics-api-accepts-cross-origin-writes-and-reads-from-any-web-page) |
| INFO log line per `metrics/current` poll (§2) | Log noise; faster rotation of real history | **Verified** | [TD-527](../../guides/TECHNICAL_DEBT.md#td-527-trainingmetricsapi-per-poll-logging-parsing-and-unbounded-history-queries) |
| Substring parsing; uncapped `max_records`; second-resolution history timestamps; precision switch; size/boilerplate (§2) | Hygiene and resource use | By inspection | [TD-527](../../guides/TECHNICAL_DEBT.md#td-527-trainingmetricsapi-per-poll-logging-parsing-and-unbounded-history-queries) |
| NaN JSON, session cap, query order, retention, admin PUT, duplicated JSON, CLI | Already tracked | — | TD-443, TD-445, TD-441, TD-440, TD-305/306, TD-453, TD-438/439 |

---

## Addendum (October 20, 2026): negative `sigreg_loss` (TD-503)

The epoch-end handler recorded LeJEPA metrics only when both `predictor_loss` and `sigreg_loss` were ≥ 0, with −1 meaning "not
sent". Since TD-503, `sigreg_loss` is an unbiased estimate that can be slightly negative, so the handler now gates on
`predictor_loss` alone; the two always arrive together. Test: `NegativeSigregLossIsStillRecorded`.

---

## Addendum (October 24, 2026): CORS allow-list and write guard (TD-526)

The wildcard CORS policy is replaced by an allow-list (constructor argument `cors_allowed_origins`, from
`METRICS_API_CORS_ALLOWED_ORIGINS`/`--cors-allowed-origins`, empty by default):
- **Pre-routing handler:** refuses writes (POST/PUT/PATCH/DELETE) with an unlisted `Origin` (403), and with a preflight-free
  `Content-Type` (415).
- **Post-routing handler:** adds `Access-Control-Allow-Origin` only for a listed origin (or `*` for reads when the list has `*`), plus
  `Vary: Origin`.
- **`OPTIONS`:** answers a preflight only for a listed origin.

Requests without `Origin` (trainer, Android, `curl`) behave as before. Tests: the six TD-526 cases in
`training_metrics_api_routes_test.cpp`. The other daemons' exposure to cross-origin writes is TD-550.

---

## Addendum (October 24, 2026): shared write guard (TD-550)

The TD-526 write checks moved into [BrowserWriteGuard](BrowserWriteGuard.md), now shared with every HTTP daemon. `TrainingMetricsAPI` installs it with its CORS allow-list; behavior is unchanged except the 415 message, which now names the refused content type.

---

## Addendum (October 25, 2026): POST …/metrics/abnormal (TD-555)

Session-scoped `POST /api/sessions/<key>/metrics/abnormal`
(`{"epoch","sample_id","loss","grad_norm","reason","input_text","target_text"}`) records a pushed abnormal sample through
`TrainingMetricsService::flag_abnormal_sample()`, which stores it in memory, the DB and the JSON file. Texts are capped at 4 KiB (on a
UTF-8 boundary, marked "…") and `reason` at 128 characters. The path used to be GET-only, so remote trainers' samples were never stored.

---

## Addendum (October 25, 2026): no nan/inf in or out (TD-443)

- **Input:** field reads use `parse_json_float()`, so `null` becomes NaN; `std::stof` threw on it and the event failed. Array elements
  that are `null` stay in place as NaN.
- **`/epoch/end`:** applies `adaptive_clip_threshold`/`adaptive_clip_spikes`.
- **Output:** every JSON response passes through `sanitize_non_finite_json()` in the post-routing handler, which turns bare `nan`/`inf`
  values into `null` and resets `Content-Length`.

---

## Addendum (October 25, 2026): DB readers and retention settings (TD-440)

New routes: `GET /api/sessions/{key}/metrics/db-gradient-variance[?limit=N]` and `GET …/metrics/db-generation-quality`. Small values
keep their digits. `GET`/`PUT /admin/config` also expose `retention_days`, `metrics_file_max_mb` and `metrics_file_backups`, persisted
like the other keys. Tests: `GradientVarianceAndGenerationQualityAreReadable`, `RetentionSettingsAreAdminMutable`.

## Addendum (October 25, 2026): bounded compare/export reads (TD-441)

`GET …/metrics/export` reads the newest `kMaxReadRows` (200,000) rows, and its JSON form reports `"truncated"`. `GET /api/metrics/compare`
refuses more than `kMaxCompareSessions` (16) keys and reads each session's newest 200,000 rows. `db-history?limit=N` returns the newest
N. Tests: `CompareRefusesTooManySessions`, `ExportReportsTruncationAndDbHistoryLimitKeepsNewest`.

## Addendum (October 25, 2026): JSON body parsing, poll logging, history (TD-527)

Push handlers read bodies through `PushBody` (nlohmann/json): defaults for absent fields, NaN for `null`, a 400 naming any
wrongly typed field, and a 400 for invalid JSON. `parse_json_float()`/`parse_json_float_array()` are gone. `metrics/current` logs at
debug. `metrics/history` caps `max_records` at `kMaxHistoryRecords` (10,000) and adds `timestamp_ms`. Tests:
`PushBodiesParseAsJsonWithFieldErrors`, `HistoryHasMillisecondTimestampsAndCappedMaxRecords`.

## Addendum (November 3, 2026): admin PUT range checks (TD-306)

`handle_admin_put_config()` parses the body with nlohmann/json and checks every key against `metrics_admin_int_settings()`
(`Config.hpp`; the same ranges `metrics_api_server`'s flags and saved overrides use) before applying any. A value out of
range or not a whole number, a non-boolean `enable_prometheus`, an immutable key, a body that isn't a JSON object, or no
recognized key gets 400 naming the problem (`'max_live_sessions' must be a whole number >= 1`), and nothing changes. A
negative `max_live_sessions` used to become a huge `size_t`, and a key with a space before its colon was skipped silently.

## Addendum (November 3, 2026): admin PUT saves before applying (TD-305)

Once a body passes the TD-306 checks, `handle_admin_put_config()` saves all 12 resulting settings with one
`DaemonConfigStore::set_many()` transaction, holding `admin_config_mutex_`, and applies them to the session registry only
after the save worked. A failed save logs the reason and answers 500 (`could not save the admin config (<reason>); nothing
was changed`), and the live settings stay as they were.

## Addendum (November 4, 2026): per-sample quality routes (TD-273)

- `POST …/metrics/sample-quality` parses the report with `PushBody` (new `get_array()`), requires `source` `loss`/`generation` and
  `split` `train`/`validation`, refuses a negative `epoch`/`count` or a `lowest` that isn't an array of objects (400 naming the
  field), keeps at most 20 lowest pairs with texts cut by `truncate_sample_text()`, and hands it to
  `TrainingMetricsService::record_sample_quality()`.
- `GET …/metrics/sample-quality` returns the service's reports; `GET …/metrics/db-sample-quality[?limit=N]` the database's. Both are
  written with nlohmann/json, so a non-finite statistic is `null`.
- `EveryPushClientEndpointHasAPostRoute` now drives `report_sample_quality()` too.

## Addendum (November 14, 2026): POST …/metrics/gate-values (TD-341)

`handle_post_gate_values()` parses `world_model_gates` and `memory_gates` with `PushBody::get_float_array()` (`null` elements become
NaN, anything else that isn't a number is a 400 naming the field) and passes them to `TrainingMetricsService::update_gate_values()`.
Tests: `GateValuesRoundTripThroughPostAndGet`, and `EveryPushClientEndpointHasAPostRoute` now pushes gate values too.
