# `ModelNameService` — Source File Reference

- **Files:** [`src/ModelNameService.hpp`](../../../../src/ModelNameService.hpp), [`src/ModelNameService.cpp`](../../../../src/ModelNameService.cpp) (2,233 lines together)
- **Namespace:** `adai` (`ModelNameService`, `ModelRecord`, `ModelConnection`, `ArtifactLocation`, `TrainingHistoryEntry`)
- **Built into:** `adai_mns`; hosted by `mns_server` ([`ModelNameServiceServer.cpp`](../../../../src/ModelNameServiceServer.cpp), args in [MnsServerArgs.md](MnsServerArgs.md))
- **Status tags:** `.hpp` `stable` 1.1.1, `.cpp` `stable` 1.0.2, both reviewed 2026-09-19
- **Tests:** [`tests/model_name_service_live_test.cpp`](../../../../tests/model_name_service_live_test.cpp) (62 tests against a running server; **all 62 pass** against a local `mns_server` started for this trace, and all skip in a plain `ctest` run without one)
- **Clients:** [ModelNameClient](ModelNameClient.md) (trainer, chat server, `dataset_manager`), [MnsCliCommands](MnsCliCommands.md) (`mns_cli`), [MnsManagerGUI](MnsManagerGUI.md), the Android ops dashboard
- **Last traced against the code:** 2026-10-14

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
| `PUT /models/{n}/progress` | `handle_progress_update` | Only while `training` and only for the active `run_id` (else 409) |
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
  progress snapshot as an `incomplete` history entry.
- **`candidate`** (from `training`, `initializing`, `retired`): from `training` it requires the active `run_id` (else
  409 "superseded"); attaches the artifact and appends a history entry; clears progress.
- **`retired`** from any state; **`production`** only via promote. A `production` model can't re-enter `training`, so the
  always-on trainer's `set_training()` gets 409 for a promoted model and carries on without MNS run tracking (it logs a
  warning and skips `set_candidate()`).

---

## 3. Problems (verified)

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
- **Persistence errors are swallowed.** `persist_model()`, `persist_roles()` and `handle_delete()` log (or ignore) SQLite
  failures and still return success, so memory and disk can diverge until restart.
- **Non-finite numbers** are accepted and re-emitted (TD-458).
- **Promote can leave a model in two roles**: promoting it to a new role rewrites its `role` but leaves the old role's
  mapping.
- **Unicode escapes**: `json_string()` turns `é` into `u00e9` (affects clients that ASCII-escape).

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Retiring a production model leaves its role mapping; `resolve_role` returns retired models (§3) | The chat server keeps loading a retired model via `MODEL_ROLE` | **Verified** | [TD-460](../../guides/TECHNICAL_DEBT.md#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers) |
| `retired → candidate` without an ownership check lets a running trainer undo a retirement (§3) | An operator's retire is silently reversed at the end of the pass | **Verified** | [TD-460](../../guides/TECHNICAL_DEBT.md#td-460-mns-retirement-leaves-role-mappings-and-is-undone-by-running-trainers) |
| No architecture validation for legacy chatbots; no divisibility check for any kind (§3) | Permanently unusable registrations (all zeros, or `d_model % num_heads ≠ 0`) | **Verified** | [TD-461](../../guides/TECHNICAL_DEBT.md#td-461-mns-accepts-unbuildable-architectures-at-registration) |
| Only compact JSON accepted (§3) | Standard JSON from scripts/tools rejected with misleading errors | **Verified** | [TD-462](../../guides/TECHNICAL_DEBT.md#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) |
| Dead datasets proxy (no `model_id` ever recorded; wrong group; no kind); admin-settings data race; swallowed persistence errors; promote leaves stale role; `\u` escapes (§4) | Dead endpoint; rare crash; silent memory/disk divergence | By inspection | [TD-462](../../guides/TECHNICAL_DEBT.md#td-462-modelnameservice-dead-datasets-proxy-admin-settings-race-and-swallowed-persistence-errors) |
| Non-finite numbers; duplicated JSON code | Already tracked | — | TD-458, TD-453 |
