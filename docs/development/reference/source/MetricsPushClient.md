# `MetricsPushClient` — Source File Reference

- **Files:** [`src/MetricsPushClient.hpp`](../../../../src/MetricsPushClient.hpp), [`src/MetricsPushClient.cpp`](../../../../src/MetricsPushClient.cpp) (895 lines together)
- **Namespace:** global
- **Built into:** `adai_core`; HTTP only when `BUILD_METRICS_API_SERVER` is defined (otherwise every POST returns 0)
- **Used by:** `IncrementalTrainer` (chatbot passes) and `run_lejepa_training_pass()` in `IncrementalTrainingTool.cpp`, both inside `incremental_trainer`
- **Implements:** [IMetricsReporter](IMetricsReporter.md)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.1`, `@adai-reviewed: 2026-09-23` (both files)
- **Tests:** [`tests/metrics_push_client_test.cpp`](../../../../tests/metrics_push_client_test.cpp) (12 tests)
- **Server side:** [TRAINING_METRICS_API.md](../../TRAINING_METRICS_API.md) (push endpoints), [MetricsDatabase.md](MetricsDatabase.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linked against
> the real class, pushing to a real `metrics_api_server` (or to a closed port).

---

## 1. What this file is

The trainer's side of live metrics (TD-021). Every `IMetricsReporter` call becomes a JSON POST to
`{METRICS_SERVER_URL}/api/sessions/{key}{endpoint}`, queued and sent by one background thread so the
training loop never waits on the network. Session start is synchronous (the caller needs the status
code for its 409 retry); session end drains the queue and joins the thread.

The header also holds the session-key helpers both training loops share:

| Helper | Behaviour |
|---|---|
| `derive_metrics_session_key(id)` | `"{id}-{host}"`, where `host` is the hostname sanitized to `[A-Za-z0-9_-]` and **cut to 8 characters** (`ai-machine` → `ai-machi`) |
| `derive_metrics_session_label(id, model_path)` | `"#{id}: {model stem} ({host}, YYYY-MM-DD)"` |
| `build_metrics_session_push_base(url, key)` | `url` without trailing slashes + `/api/sessions/{key}` |
| `sanitize_session_key()`, `trim_trailing_slashes()`, `detect_hostname_fragment()` | Building blocks |

Two hosts whose names share the first 8 characters get the same key for the same session ID. The
server's 409 and `IncrementalTrainer`'s retry with a `-2`/`-3` suffix handle it, but the dashboards then
show near-identical keys.

---

## 2. How it works

### Lifecycle (`IncrementalTrainer::train_*`)

```text
for attempt in 0..2:                      key, key-2, key-3
    pc = MetricsPushClient(base_url, METRICS_PUSH_TIMEOUT_MS, 1024, METRICS_HEARTBEAT_INTERVAL_MS)
    rc = pc.start_session(...)            synchronous POST /start
    if rc != 409: break
2xx → use pc as the reporter;  otherwise log "WITHOUT metrics" and use NullMetricsReporter
... ChatbotTrainer::train() ...
pc.end_session()                          enqueue /end, drain the whole queue, join
save final checkpoint, mark files trained
```

### Events and priorities

| Priority | Methods | Under backpressure (queue ≥ 1024) |
|---|---|---|
| `Sample` | `update_sample_metrics()` (`/metrics/sample`), `update_advanced_epoch_metrics()` (`/metrics/advanced`, every optimizer step) | **Dropped**, with one warning per overflow episode |
| `Epoch` | `start_epoch()`, `end_epoch()`, `update_validation_metrics()`, `update_best_metrics()`, `flag_abnormal_sample()`, `update_layer_gradient_norms()`, `update_generation_quality_metrics()` | Evicts the oldest `Sample` event; never dropped (the queue grows past its limit if there's nothing to evict) |
| `Session` | `/end` | Same as `Epoch` |

**Buffered, not sent on their own:** activation saturation, attention entropy, padding efficiency, the
LeJEPA losses and diagnostics, and the advanced metrics' last values. `end_epoch()` folds them into its
payload; `start_epoch()` resets them. Adaptive clipping is buffered too, but **never sent** (§3).

### The push thread

`push_loop()` waits up to the heartbeat interval for an event, POSTs it, and on a **404** (the server
restarted or evicted the session) re-registers with `reconnect_body_` and retries the event once.
`reconnect_body_` is the `/start` body without `reset_best`, so a mid-run reconnect never wipes the best
loss (tested). Independently, it POSTs `/heartbeat` whenever the interval has passed, so a backlog can't
make the server mark the session stale.

`attempt_post()` builds a **new `httplib::Client` (a new TCP connection) for every attempt**, with
`timeout_ms` for connect, read and write; it retries network errors and 5xx up to 3 times (sleeping 0,
200 and 1,000 ms before the attempts), and returns 2xx/4xx immediately. Every result other than a 404 is
ignored by `push_loop()`; failures are logged only at debug level.

---

## 3. Problems

### A metrics outage blocks the end of a training pass (verified)

`end_session()` processes **every** queued event before returning, and each failing event costs up to 3
attempts plus 1.2 s of back-off. With the server down (connection refused), draining 10 queued epoch
events plus `/end` took **13.2 s**, which is 1.2 s per event. If the server goes away mid-pass, the
queue fills to its 1,024 limit with step events, so the drain takes about **20 minutes** when
connections are refused. When connections time out instead (a host that's down rather than a stopped
daemon), each attempt also waits `timeout_ms` (1 s by default), so it's about **70 minutes**.

`IncrementalTrainer` calls `end_session()` **before** saving the final checkpoint and marking the files
trained (the LeJEPA pass saves first, so it's only delayed). So a metrics outage near the end of a chatbot
pass delays the pass's real results by that long, and if
the process is stopped in the meantime (`systemctl stop`, `trainer_service`'s 10-second SIGKILL
escalation, a GPU-driver crash), the final save and `mark_trained()` never happen. Metrics are optional;
they shouldn't be able to hold training results hostage. A bounded drain (a deadline, or dropping
`Sample` events once stopping) and saving before `end_session()` would fix it.

### Non-finite values produce invalid JSON (verified end to end)

Numbers are written with `operator<<`, so a NaN or infinite loss is sent as `nan`/`inf`, which isn't
JSON. The server's hand-written parser (`std::stof`) accepts them, and its own output then contains them
too. Pushing an epoch with a NaN loss and an infinite validation loss to the real server made
`GET /api/sessions/{key}/epochs` return `"epoch_losses":[nan,2.5],"epoch_validation_losses":[inf,2.6]`,
which **Python's `json` module rejects**. So the session's epoch view breaks for every dashboard exactly
when training diverges, the moment it's most needed. `end_epoch()` also derives perplexity as `exp(loss)`,
which is `inf` for any loss above about 88.

### `end_epoch()` rounds small values to zero (verified)

`end_epoch()` sets `std::fixed`, which prints 6 decimal places. A learning rate of `1e-7` reached the
server as **0** (`"epoch_learning_rates":[0,0]`), and so would `gradient_variance` and
`weight_update_ratio` below `5e-7` (typical update ratios are around 1e-4 to 1e-6). Those are the
epoch-level vanishing-gradient diagnostics. The per-step `/metrics/advanced` push doesn't use `std::fixed`,
so the live view is fine; the stored epoch history isn't.

### Adaptive-clipping metrics never reach the server (verified)

`update_adaptive_clip_metrics()` and `update_adaptive_clip_epoch()` store into `buf_adaptive_clip_*`, but
`end_epoch()` **never writes those members**, and the server has no endpoint that accepts them. The
probe's session showed `"epoch_adaptive_clip_thresholds":[]`, and that field is always empty for pushed
sessions, though `TrainingMetricsService` and the `/epochs` output support it. ([IMetricsReporter.md](IMetricsReporter.md)
said end-of-epoch pushes include them; corrected.)

### Smaller items (by inspection)

- **`https://` URLs can't work.** `ParsedUrl` recognizes `https://` and switches to port 443, but always
  uses a plain `httplib::Client` (no OpenSSL support is compiled in for it), so it sends plain HTTP to a TLS
  port: `start_session()` returns 0 and the run trains without metrics. There's also no way to add
  headers such as Cloudflare Access tokens. Today's `config.trainer.conf` uses a LAN `http://` URL.
- **A connection per event.** One TCP connection per POST, including one per optimizer step for
  `/metrics/sample` and `/metrics/advanced`; a persistent client would avoid it.
- **Silent failures.** `push_loop()` ignores every non-404 status (a 400 for a malformed body, a 503
  "server full"), logging only at debug level.
- **`reconnect_body_`** is written by `start_session()` on the caller's thread while the push thread is
  already running. It's benign in practice (read only after a 404), but it's an unsynchronized shared
  string.
- The heartbeat also fires before `start_session()` if construction and start are more than one interval apart.

---

## 4. Tests

`metrics_push_client_test.cpp` (12 tests, against a local test server) covers: dropping `Sample` events
when full, `Epoch` events evicting samples and never being dropped, 5xx retries, no retries on 4xx/409,
`end_session()` draining before joining, label/config in the `/start` body, 404 reconnect without
`reset_best`, layer-norm arrays, empty URL returning 0, and the destructor not deadlocking.

**Not covered:** drain time with the server down, non-finite or very small values, adaptive-clip fields,
`https://` URLs.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `end_session()` drains everything at 1.2 s+ per failed event, and runs before the final checkpoint and `mark_trained()` (§3) | A metrics outage delays a pass's results by ~20–70 minutes; a stop in that window loses the final save | **Verified** (13.2 s for 11 events) | [TD-442](../../guides/TECHNICAL_DEBT.md#td-442-metricspushclientend_session-drain-can-delay-a-passs-final-save-by-an-hour) |
| NaN/inf sent as `nan`/`inf`; the server stores and re-emits them (§3) | A diverging run makes the session's `/epochs` JSON unparseable | **Verified** (real server) | [TD-443](../../guides/TECHNICAL_DEBT.md#td-443-metrics-push-payloads-send-naninf-round-small-values-to-0-and-omit-adaptive-clipping) |
| `end_epoch()`'s `std::fixed` rounds values below 5e-7 to 0 (§3) | Small learning rates and vanishing-gradient diagnostics stored as 0 | **Verified** | [TD-443](../../guides/TECHNICAL_DEBT.md#td-443-metrics-push-payloads-send-naninf-round-small-values-to-0-and-omit-adaptive-clipping) |
| Adaptive-clip values buffered but never sent (§3) | The dashboard's adaptive-clip history is always empty | **Verified** | [TD-443](../../guides/TECHNICAL_DEBT.md#td-443-metrics-push-payloads-send-naninf-round-small-values-to-0-and-omit-adaptive-clipping) |
| `https://` sends plain HTTP; no custom headers; a connection per event; non-404 failures silent; `reconnect_body_` race; 8-character host fragment (§1, §3) | Metrics through TLS or a tunnel impossible; overhead; hidden failures | By inspection | [TD-444](../../guides/TECHNICAL_DEBT.md#td-444-metricspushclient-transport-gaps-no-https-a-connection-per-event-silent-failures) |
| Abnormal-sample texts untruncated | Already tracked | — | TD-399 |

---

## Addendum (October 25, 2026): re-registration only for an unknown session (TD-555)

`attempt_post(endpoint, body, &response_body)` returns the final response body too. `push_loop()` treats a 404 as a lost session only
when the body says `Unknown session key`. Then it re-registers (logged once per outage, reset by the next accepted event) and retries the
event. Any other 404 means the server lacks the route: it's logged once per endpoint and the event is dropped. Every flagged abnormal
sample used to log "session gone, re-registering", because the server had no POST route for it.
