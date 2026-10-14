# `ModelNameClient` — Source File Reference

- **Files:** [`src/ModelNameClient.hpp`](../../../../src/ModelNameClient.hpp), [`src/ModelNameClient.cpp`](../../../../src/ModelNameClient.cpp) (780 lines together)
- **Namespace:** `adai` (`ModelNameClient`, `ResolvedModel`, `ModelSummary`, `ClientModelConnection`)
- **Built into:** `adai_mns`; HTTP only with `BUILD_MNS_SERVER` (otherwise every call returns status 0)
- **Used by:** `IncrementalTrainer` (run lifecycle), `IncrementalTrainingTool` (model and architecture resolution), `ChatbotAPIServer` (model, architecture and world-model resolution), `DatasetManagerTool` (`run_group` for `assign`)
- **Server:** `ModelNameService` (`mns_server`); companion CLI [MnsCliCommands](MnsCliCommands.md)
- **Status tag:** `@adai-status: stable` ("TD-196 — encoder/decoder-linked chatbot arch resolution + connection/world-model-link API added"), `@adai-version: 1.1.0`, `@adai-reviewed: 2026-09-18` (both files)
- **Tests:** [`tests/modelnameclient_test.cpp`](../../../../tests/modelnameclient_test.cpp) (21 tests, against a local mock server)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program linked against the real
> client, talking to a mock server or to a real `mns_server`.

---

## 1. What this file is

The C++ client for the Model Name Service: how the trainer and the chat server learn a model's identity, architecture,
artifact and world-model connection, and how the trainer reports its run lifecycle. It's what makes MNS
"authoritative" in practice (CLAUDE.md "Configuration" and "MNS/registry-authoritative run and session numbering").

| Method | Request | Caller and use |
|---|---|---|
| `register_model(name, role, cfg, tags)` | `POST /models` | No production caller (registration is `mns_cli`'s job) |
| `set_training(name, new_run, session_key)` | `PUT /models/{n}/state` `{"state":"training","new_run":…}` | `IncrementalTrainer::begin_run()`: returns the **server-allocated** `run_id` used for MNS and the dataset registry |
| `push_progress(name, run_id, session, epoch, loss, best)` | `PUT /models/{n}/progress` | `IncrementalTrainer` after every epoch (failures logged, non-fatal) |
| `set_candidate(name, run_id, artifact, summary)` | `PUT /models/{n}/state` `{"state":"candidate",…}` | `IncrementalTrainer` after a successful pass |
| `resolve_model(name)`, `resolve_role(role)` | `GET /models/{n}/resolve`, `GET /roles/{r}/production` | `ChatbotAPIServer` (artifact path), `IncrementalTrainingTool`, `dataset_manager` |
| `get_architecture(name)` | `GET /models/{n}` (+ the linked encoder and decoder for a TD-196 linked chatbot) | Both binaries overwrite their six architecture fields with it; `nullopt` on 404 |
| `get_connection(name)` | `GET /models/{n}` | World-model and hippocampal settings (TD-196) |
| `link_world_model(…)` | `POST /models/{n}/link-world-model` | Returns `false` on a non-2xx (a 409 is expected for incompatible pairs) |
| `list_models(state, role, limit)`, `promote(role, name)`, `update_run_group(name, group)` | — | Tools and tests |

**Transport.** Every call goes through `http_post/get/put()`: a new `httplib::Client` per attempt, connect/read/write
timeouts of `timeout_ms` (5 s default), and **up to 3 attempts** (sleeping 0, 200 and 1,000 ms) on a network error,
timeout or 5xx. A 2xx/4xx returns immediately. `check_status()` throws `runtime_error` on status 0 (connection failed) or
a non-2xx. The URL may carry a base path (for a reverse proxy).

**Parsing.** Hand-written string search (TD-453): `json_string_client()` (escape-aware end, **no unescaping**),
`json_int_client()` (`stoull`, 0 if missing), `json_double_client()`, `json_bool_client()`; `list_models()` splits records
by string-aware brace counting (TD-070). `mns_server` writes compact JSON with top-level fields first, which these rely on.

---

## 2. Problems (verified)

### Retries resend non-idempotent requests after a timeout

A request that times out is retried, even if the server already processed it. With a mock server that answered the first
`PUT` after 1.5 s and a 1 s client timeout, one `set_training(…, new_run = true)` call **reached the server twice** and returned
the second reply. Against `mns_server`:
- `set_training(new_run = true)` (a `retrain`) increments the run number on **each** delivery, so a slow MNS turns one retrain
  into two allocations: the first `run-NN` is skipped, with nothing recorded under it.
- `register_model()`'s retry gets `409` (the name now exists) and **throws although registration succeeded** (no production
  caller today).
- `set_training(new_run = false)`, `set_candidate()` and `push_progress()` are safe to repeat.

The retry policy should apply only to idempotent requests, or the requests should carry an idempotency key the server
deduplicates.

### A NaN loss makes the MNS record unparseable (verified end to end)

`push_progress()` writes the loss with `operator<<`, so a diverged epoch's NaN goes out as `nan`. `mns_server` accepts it
(`stod`) and writes it back verbatim: after pushing a NaN loss to a real `mns_server`, `GET /models/{name}` returned
`"progress":{…,"loss":nan,…}`, and **both that record and the whole `GET /models` list were rejected by Python's `json`
module**. It stays that way until the next progress push for that model, and when the run is later superseded the snapshot
is archived into `training_history` with `final_loss` NaN, permanently. Any strict JSON consumer of the MNS API breaks for
every model, not just the diverged one, exactly when a run diverges. It's the same class of defect as TD-443 in the metrics
path.

---

## 3. Smaller items (by inspection)

- **Unchecked architecture.** Callers copy `get_architecture()`'s six fields straight into their config with no
  validation. A missing field parses as 0, and pointing `MODEL_NAME` at a standalone `encoder`/`world_model` record yields
  `num_decoder_layers = 0`, so the chat server or trainer builds a model with no decoder rather than failing with a clear
  error.
- **No unescaping.** Artifact paths and names come back with JSON escapes intact (a `\` in a path stays `\\`). Rare on Linux
  paths, but `resolve_model().artifact.path` is used to load weights.
- **Slow failure when MNS is unreachable.** Each call can take 3 × `timeout_ms` plus 1.2 s (~16 s at the default 5 s with a
  host that doesn't answer); startup makes several calls in sequence (resolve, architecture, connection, the world model's
  resolve and architecture), so a down-but-routable MNS can delay `chatbot_api_server`/trainer startup by about a minute
  before falling back to local config.
- **No URL encoding** of names, roles or query values.
- **`https://`** sets port 443 but connects with a plain `httplib::Client` (same as TD-444/TD-451).
- **Duplicated helpers** (TD-453); one `httplib::Client` per attempt.

---

## 4. Tests

`modelnameclient_test.cpp` (21 tests, mock server): every method's request shape and response parsing, TD-196 composition
of a linked chatbot's architecture, the legacy fallback, 404 → `nullopt`, `link_world_model()`'s `false` on 409, brace-like
text inside a field (TD-070), non-2xx and connection failures throwing; one test skips without `BUILD_MNS_SERVER`.

**Not covered:** retries after a timeout, non-finite numbers, escaped strings, zero or missing architecture fields.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Timeouts retried for non-idempotent requests (§2) | A slow MNS skips a run number on `retrain`; `register_model()` throws after succeeding | **Verified** (mock server) | [TD-457](../../guides/TECHNICAL_DEBT.md#td-457-modelnameclient-retries-non-idempotent-requests-after-a-timeout) |
| NaN progress loss sent as `nan` and stored/re-emitted by `mns_server` (§2) | A diverged run makes `GET /models` invalid JSON for strict clients; persists in `training_history` | **Verified** (real `mns_server`) | [TD-458](../../guides/TECHNICAL_DEBT.md#td-458-a-nan-training-loss-makes-mnss-json-unparseable) |
| Architecture not validated; no unescaping; ~16 s per call when MNS doesn't answer; no URL encoding; `https://` (§3) | Wrong-shaped models instead of clear errors; slow fallback | By inspection | [TD-459](../../guides/TECHNICAL_DEBT.md#td-459-modelnameclient-validation-unescaping-and-slow-failure-gaps) |
| Duplicated JSON helpers | Already tracked | — | TD-453 |
