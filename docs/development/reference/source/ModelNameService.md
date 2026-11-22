# `ModelNameService` — Source File Reference

- **Files:** [`src/ModelNameService.hpp`](../../../../src/ModelNameService.hpp), [`src/ModelNameService.cpp`](../../../../src/ModelNameService.cpp) (2,476 lines together)
- **Namespace:** `adai` (`ModelNameService`, `ModelRecord`, `ModelConnection`, `ArtifactLocation`, `TrainingHistoryEntry`)
- **Built into:** `adai_mns`; hosted by `mns_server` ([`ModelNameServiceServer.cpp`](../../../../src/ModelNameServiceServer.cpp), args in [MnsServerArgs.md](MnsServerArgs.md))
- **Status tags:** `.hpp` `stable` 1.5.0, `.cpp` `stable` 1.7.0, both reviewed 2026-11-22
- **Tests:** [`tests/model_name_service_live_test.cpp`](../../../../tests/model_name_service_live_test.cpp) (70 tests against a running server; **all 70 pass** against a local `mns_server`, November 21, 2026, and all skip in a plain `ctest` run without one); [`tests/mns_progress_test.cpp`](../../../../tests/mns_progress_test.cpp) (4 tests, in-process: the run's step in progress and history, a restart, an older `models.db`, a pre-SQLite `models.jsonl`/`roles.json` import); [`tests/model_name_service_start_test.cpp`](../../../../tests/model_name_service_start_test.cpp) (4: start-up refusals, and a `stop()` before or during `start()`, TD-464)
- **Clients:** [ModelNameClient](ModelNameClient.md) (trainer, chat server, `dataset_manager`), [MnsCliCommands](MnsCliCommands.md) (`mns_cli`), [MnsManagerGUI](MnsManagerGUI.md), the Android ops dashboard
- **Last traced against the code:** 2026-10-14; progress, history and the number readers re-traced 2026-11-21 (TD-269); the JSONL import and `persist_model()` 2026-11-22 (TD-581); `start()`/`stop()` 2026-11-22 (TD-464)

> Code-traced reference. Behaviour marked **verified** was confirmed against a real `mns_server` built from this tree.

---

## 1. What this file is

The **Model Name Service**: the registry of model identities and the authority for each model's architecture, kind,
world-model connection, lifecycle state, run numbering, training progress and role → production mapping (CLAUDE.md
"Configuration", TD-196). It's an `httplib` server over an in-memory map guarded by one `shared_mutex`, persisted to SQLite
(`<data_dir>/models.db`, WAL; a one-time migration from the old JSONL files).

### Endpoints

| Method and path | Handler | Notes |
|---|---|---|
| `POST /models` | `handle_register` | `valid_name()` (`[a-z0-9][a-z0-9.-]{1,127}`), kind, per-kind validation (§3), 201 with `model_id` |
| `GET /models[?state=&role=&kind=&limit=]` | `handle_list` | Full records |
| `GET /models/{n}`, `GET /models/{n}/resolve` | `handle_get`, `handle_resolve` | `resolve` is compact; 409 while `initializing` |
| `PUT /models/{n}/state` | `handle_state_transition` | §2 |
| `PUT /models/{n}/progress` | `handle_progress_update` | Only while `training` and only for the active `run_id` (else 409). Stores `session_id`, `epoch`, `loss`, `best_loss` and `global_step` (the run's optimizer step; an absent one keeps the last value) |
| `PUT /models/{n}/run_group` | `handle_update_run_group` | Any state |
| `POST /models/{n}/link-world-model` | `handle_link_world_model` | Chatbot only; replaces the whole world-model/hippocampal connection; d_model check when injecting |
| `DELETE /models/{n}` | `handle_delete` | Only `initializing`/`retired`; removes history and role mappings |
| `GET /roles`, `GET /roles/{r}/production`, `PUT /roles/{r}/production` | `handle_list_roles`, `handle_resolve_role`, `handle_promote` | Promote requires `candidate`; retires the role's previous `production` model |
| `GET /models/{n}/datasets` | `handle_datasets` | Proxies `registry_server`'s `/registry/{group}/history?model_id=` (§4) |
| `GET`/`PUT /admin/config` | `handle_admin_*` | `registry_url`/`registry_group` only; PUT gated by `--admin-enabled` |
| `GET /health` | `handle_health` | Model count, uptime |

No endpoint is authenticated (beyond the admin gate).

---

## 2. Lifecycle

```text
initializing ──training──► training ──candidate──► candidate ──promote──► production
     │  ▲                     │ ▲ (re-training: archive the abandoned run as incomplete)   │
     │  └──────── candidate (import) ─────────────────────────────┘                        │
     └─────────────── retired ◄──── retire (from any state) ◄──────────────────────────────┘
                         └──candidate (revival)
```

- **`training`** (from `initializing`, `candidate`, `training`): the server **allocates** `run-NN`: `run-01` the first time,
  incrementing only for `new_run: true`. A client `run_id` is ignored. Re-entering `training` archives the previous run's
  progress snapshot (epoch, loss, `global_step`) as an `incomplete` history entry, then zeroes the progress.
- **`candidate`** (from `training`, `initializing`, `retired`): from `training` it requires the active `run_id` (else
  409 "superseded"); attaches the artifact and appends a history entry from `training_summary` (`epochs`,
  `final_loss`, `global_step`, falling back to the last progress report's step); clears progress.
- **`retired`** from any state; **`production`** only via promote. A `production` model can't re-enter `training`, so the
  always-on trainer's `set_training()` gets 409 for a promoted model and carries on without MNS run tracking (it logs a
  warning and skips `set_candidate()`).

---

## 3. Problems (verified)

**Fixed November 9, 2026 (TD-460):** the two problems below; retiring clears a model's roles and reviving takes `"revive":true`.

### Retiring a role's production model leaves the role pointing at it

`retired` is reachable from `production` through `PUT …/state`, but only `handle_promote()` and `handle_delete()` touch
`roles_`. After promoting `prod-a` to `chatbot` and then retiring it, **`GET /roles/chatbot/production` still returned
`prod-a`** (with `"state":"retired"` and its artifact), and `handle_resolve_role()` doesn't check state. A
`chatbot_api_server` configured with `MODEL_ROLE=chatbot` keeps loading the retired model, and nothing indicates the role
has no live production model. Retiring should clear the role mapping (or be refused for a production model), and
`resolve_role` should reject non-production records.

### Retiring a training model is undone by the trainer

`candidate` may be entered from `retired` ("revival"), and the `run_id` ownership check applies only when leaving
`training`. So when an operator retires a model while a pass is running (to stop it being used), the pass's normal
`set_candidate()` at the end **revives it**: retire during `run-01`, then the trainer's `PUT …/state
{"state":"candidate","run_id":"run-01",…}` returned 200 and state `candidate`. The operator's retirement is silently undone.

### Registration accepts architectures that can never work, permanently

**Fixed November 9, 2026 (TD-461):** registration now refuses these with a 400 naming the field.

Architecture is immutable, but a legacy (inline) `chatbot` is registered with **no architecture validation at all**:
- With no `arch` object: 201, stored as `d_model 0, num_heads 0, …` (the client then hands those zeros to the trainer and
  chat server, TD-459).
- `d_model 100, num_heads 8`: 201, although `MultiHeadAttention`'s constructor rejects any `d_model` not divisible by
  `num_heads`, so the model can never be built.
Standalone kinds check for positive values but not divisibility either. The only remedy is delete and re-register.

### Only compact JSON is understood

The request parsers search for `"key":"` and `"key":` with no whitespace. A standard, spaced body such as
`{"model_name": "spaced", "role": "chatbot"}` (the form Python's `json.dumps()` and many tools emit) returned **400
"model_name required"**. All in-repo clients send compact JSON, so they're unaffected, but hand-written `curl` or script
calls fail confusingly (TD-453 is the umbrella fix).

---

## 4. Smaller items (by inspection, except where noted)

- **The datasets proxy is dead.** `handle_datasets()` filters `registry_server`'s history by `model_id`, but no client ever
  sends `model_id` when marking files trained (`RemoteTransport::mark_trained()`'s body has only `run_id`, `files`,
  `samples`), so the result is always empty. It also queries MNS's global `registry_group_` instead of the model's own
  `run_group` (which MNS is canonical for), and has no TD-202 kind segment. No client calls it.
- **Data race on the registry settings.** `PUT /admin/config` writes `registry_url_`/`registry_group_` (`std::string`s)
  without a lock while `handle_datasets()` reads them on another httplib thread: undefined behaviour, in practice a rare
  crash or garbled URL.
- **Persistence errors are swallowed.** `persist_model()` (which returns false since TD-581, but only the JSONL import checks it),
  `persist_roles()` and `handle_delete()` log (or ignore) SQLite failures and the handlers still return success, so memory and disk can diverge until restart.
- **`request_id`** (TD-457): on a registration or `PUT …/state` to `training`, a repeat of the model's last id returns the original result without allocating anything; up to 128 bytes, kept in the `request_ids` table.
- **Non-finite numbers** (TD-458, fixed November 9, 2026): a loss with no finite value is `null` in every record and NULL in SQLite; other numbers ignore non-finite input.
- **Promote can leave a model in two roles**: promoting it to a new role rewrites its `role` but leaves the old role's
  mapping.
- **Unicode escapes**: `json_string()` turns `é` into `u00e9` (affects clients that ASCII-escape).

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Retiring a production model leaves its role mapping; `resolve_role` returns retired models (§3) | The chat server keeps loading a retired model via `MODEL_ROLE` | **Verified** | [TD-460](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers) |
| `retired → candidate` without an ownership check lets a running trainer undo a retirement (§3) | An operator's retire is silently reversed at the end of the pass | **Verified** | [TD-460](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers) |
| No architecture validation for legacy chatbots; no divisibility check for any kind (§3) | Permanently unusable registrations (all zeros, or `d_model % num_heads ≠ 0`) | **Verified** | [TD-461](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-461-mns-accepts-unbuildable-architectures-at-registration) |
| Only compact JSON accepted (§3) | Standard JSON from scripts/tools rejected with misleading errors | **Verified** | [TD-462](../../guides/TECHNICAL_DEBT.md#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) |
| Dead datasets proxy (no `model_id` ever recorded; wrong group; no kind); admin-settings data race; swallowed persistence errors; promote leaves stale role; `\u` escapes (§4) | Dead endpoint; rare crash; silent memory/disk divergence | By inspection | [TD-462](../../guides/TECHNICAL_DEBT.md#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) |
| The legacy `models.jsonl` import drops every model (positional insert with 26 values for 29 columns) and still imports `roles.json` (addenda below) | An old pre-SQLite data directory came up with no models and roles pointing at nothing | **Fixed November 22, 2026** (was verified) | [TD-581](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-581-mnss-legacy-jsonl-import-drops-every-model) |
| Non-finite numbers; duplicated JSON code | Already tracked | — | TD-458, TD-453 |

---

## Addendum (October 24, 2026): browser write guard (TD-550)

This server installs `install_browser_write_guard()` ([BrowserWriteGuard.md](BrowserWriteGuard.md)) as its pre-routing handler. A write
(POST/PUT/PATCH/DELETE) carrying any `Origin` gets 403, and a write with a `text/plain`, form-encoded or multipart body gets 415. A web
page open in an operator's browser can no longer make it execute writes. Clients that aren't browsers send no `Origin` and are unaffected
as long as they send JSON (or, for `registry_server`'s `upload`, `application/octet-stream`).

## Addendum (November 1, 2026): `link-world-model` merge (TD-449)

`handle_link_world_model()` reads an optional `"merge"` boolean. With `"merge":true` every connection field the body
leaves out keeps its current value (`mns_cli link-world-model` sends it). Without it the body still replaces the whole
link with defaults for omitted fields, as before; the Android dashboard depends on that, since kotlinx serialization
omits default values from its body.

## Addendum (November 3, 2026): admin PUT saves before applying (TD-305)

`handle_admin_put_config()` works out the new `registry_url`/`registry_group`, saves both with one
`DaemonConfigStore::set_many()` transaction while holding `admin_config_mutex_`, and assigns the members only after the save
worked. A failed save answers 500 (`could not save the admin config (<reason>); nothing was changed`) and keeps the old
values. The unlocked read in `handle_datasets()` is still TD-462.

## Addendum (November 21, 2026): the run's step, and summary numbers sent as strings (TD-269)

- **`global_step`.** `ModelRecord::progress_global_step` and `TrainingHistoryEntry::global_step` (`int64`) hold the run-level
  optimizer step `ChatbotTrainer` keeps across a run's passes. The record's `progress` object and every `training_history` entry
  have a `"global_step"`. `PUT …/progress` reads it with `json_int64()` and keeps the last value when a trainer sends none; a new run
  and the candidate transition zero it (the history entry keeps it). SQLite: `models.progress_global_step` is the last column (added
  last by `add_column_if_missing()`, so a migrated database has the same layout as a new one; `persist_model()` names its columns since TD-581) and
  `training_history.global_step` (named inserts).
- **Numbers sent as strings.** `ModelNameClient::set_candidate()` (and so `incremental_trainer` and `mns_cli candidate
  --summary`) sends every `training_summary` value as a JSON string (`"epochs":"3"`). `json_int()`, `json_int64()`,
  `json_double()` and `json_loss()` read only a bare number, so each such history entry was stored with **0 epochs and loss 0**
  (seen on 192.168.1.16's `chatbot-v1` history). They now skip an opening quote (`number_start()`), so a quoted number reads
  as the number. Entries already stored keep their zeros.
- **Not changed:** the one-time JSONL migration's `INSERT` into `models` still has 26 placeholders for a table that now has 29
  columns (it had 28), so that import fails for every model; it only matters for a data directory that still has the pre-SQLite
  JSONL files. Fixed by [TD-581](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-581-mnss-legacy-jsonl-import-drops-every-model) (next addendum).

## Addendum (November 22, 2026): the pre-SQLite import works (TD-581)

`init_db()` imports `models.jsonl` and `roles.json` once, when `models` has no rows (`migrate_from_jsonl()`). Its own positional
`INSERT` had 26 values for the 29-column table, so every prepare failed and every model was skipped without a log line, while
`roles.json` still came in. Now:
- **Models** go through `persist_model()`, the insert every update uses, with its history. `parse_record()` also reads `kind` and
  `connection` when a line has them (as `serialize_record()` writes them); a record from before TD-196 keeps `chatbot` and no
  connection.
- **`persist_model()`** names its 29 columns instead of relying on their order, so a column added later keeps its default rather than
  breaking the statement, and returns false (with the model's name in the warning) when the row isn't written; it then writes no history.
- **Failures are logged** per line: a parse error, a missing `model_name` or a failed write names the line (and model), and the
  summary counts models and roles actually imported plus the records that weren't.
- **Roles** are imported only when they name an imported model; each skipped role is logged. (A role resolves only to a model in
  `production`, TD-460.)

Test: `MnsLegacyImportTest.ImportsEveryModelWithItsHistoryAndOnlyRolesForImportedModels` in `mns_progress_test.cpp`. A Phase 1
record (no kind or connection) and a current one (kind, connection, run group, `progress.global_step`) both come through with their
fields and history, an unparseable line is skipped, the role naming an imported model resolves and the one naming a missing model
doesn't exist, and a second start imports nothing more. On the old code it imported 0 records.

## Addendum (November 22, 2026): a `stop()` before `listen()` is no longer lost (TD-464)

httplib's `Server::stop()` does nothing until `listen()` is under way, so a `stop()` that came during `start()`'s database work was
lost and `start()` served forever: `mns_server` hung after an early `SIGTERM`, and under a loaded parallel `ctest`
`ModelNameServiceStartTests` (whose `start_refuses()` stops a start that hasn't finished in 3 s) and `MnsProgressTests` hung to
ctest's 1200 s timeout. `start()` and `stop()` now use the handshake of `TrainingMetricsAPI` (TD-528) and `TrainerAdminAPI` (TD-582):
`start()` sets `starting_` before it checks `stop_requested_`, and checks it again just before `bind_to_port()`/`listen_after_bind()`;
`stop()` sets `stop_requested_`, waits at most 5 s while a start is under way and not yet listening, then calls httplib's `stop()`. A
stopped service can't be started again (no caller does). Test: `ModelNameServiceStartTest.StopBeforeOrDuringStartAlwaysEndsStart` (a
stop before `start()` and 100 at varying points in it); on the old code a `start()` never returned.
