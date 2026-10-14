# `MnsManagerGUI` — Source File Reference

- **Files:** [`src/MnsManagerGUI.hpp`](../../../../src/MnsManagerGUI.hpp), [`src/MnsManagerGUI.cpp`](../../../../src/MnsManagerGUI.cpp), [`src/MnsManagerGUI_main.cpp`](../../../../src/MnsManagerGUI_main.cpp) (931 lines together)
- **Built into:** `mns_manager_gui`, when `BUILD_GUI` (default `ON`) and Qt 5 or 6 are available, with `adai_mns` and httplib; built in the local `debug` build
- **Status tags:** `MnsManagerGUI.{hpp,cpp}` `@adai-status: beta` ("capped by TD-037 — no Qt Test infrastructure"), `0.7.0`/`0.8.0`; `MnsManagerGUI_main.cpp` `stable`, `1.0.0`
- **Tests:** none for the widget; its request bodies and JSON handling are tested through [MnsJsonHelpers](MnsJsonHelpers.md) (`mnsjsonhelpers_test.cpp`, `mns_manager_gui_test.cpp`)
- **User guide:** [mns-manager-gui-guide.md](../../../operations/guides/mns-manager-gui-guide.md)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed by sending the GUI's exact request bodies to a
> real `mns_server`.

---

## 1. What this file is

A Qt desktop front end to `mns_server`, a point-and-click alternative to `mns_cli`.

| Area | Contents | Requests |
|---|---|---|
| Toolbar | Server URL, **Connect** (re-parses the URL, runs a health check), **Refresh** | `GET /health`, then both refreshes |
| Models tab | State filter, editable role filter, table (name, role, coloured state, short ID, updated) | `GET /models[?state=&role=]` |
| Roles tab | Role → production model | `GET /roles` |
| Register tab | Name, role, six architecture spin boxes, tags | `POST /models` (`build_register_model_body()`) |
| Detail panel | Pretty-printed record of the selected model | `GET /models/{name}` |
| Actions | **Set Training** (run ID), **Set Candidate** (run ID, artifact path), **Promote** (role, model), **Retire**, **Delete** (both confirmed by a dialog) | `PUT /models/{n}/state`, `PUT /roles/{r}/production`, `DELETE /models/{n}` |

`main()` takes `--url`/`--config`; without `--url` it loads `config.mns.conf` for `NAME_SERVICE_URL`, defaulting to
`http://localhost:8083`. Every request is a synchronous `httplib::Client` call (connect 5 s, read 10 s) on the UI thread.

---

## 2. Problems

### The training → candidate workflow doesn't work (verified)

**Set Training** requires the user to type a run ID and sends `{"state":"training","run_id":"…"}`. Since run IDs became
server-allocated, `mns_server` ignores a client `run_id` for the training transition: it allocates `run-NN` itself and
reads `new_run` to decide whether to start a fresh run. Replaying the GUI's bodies against a real server:

```text
Set Training,  run ID "my-run"   → state training, run_id "run-01"
Set Candidate, run ID "my-run"   → 409 "run_id does not match the active run; this trainer's run has been superseded"
Set Training,  run ID "my-run-2" → run_id still "run-01" (the GUI never sends new_run)
```

So the user's run ID is silently discarded, **Set Candidate fails with a misleading "superseded" error** unless the user
copies the server's `run-NN` out of the detail panel, and the GUI can't start a new run at all. The user guide's
walkthrough ("enter a Run ID (e.g. `run-42`)… click Set Training… click Set Candidate") follows exactly this path, and its
`mns_cli` equivalence table still shows `set-training <name> <run-id>`, a form `mns_cli` no longer has.

### Errors are detected by searching for `"error"`

Every action treats an empty body as "Connection failed" and a body containing the text `"error"` as failure, ignoring
the HTTP status. A non-2xx reply without that key reads as success, and a successful reply containing that text (in a tag
or label, say) reads as failure. (By inspection.)

### Smaller items (by inspection)

- **The UI freezes during requests.** All HTTP runs on the UI thread, up to 15 seconds per call when the server is slow
  or unreachable; selecting a model or changing a filter blocks the window (the same class as TD-263 for `chatbot_gui`).
- **The Register tab's defaults are arbitrary.** The spin boxes start at d_model 128, 4 heads, d_ff 512, 6/6 layers, max
  sequence 1024, matching neither `config.mns.conf` (512/8) nor `config.trainer.conf` (768/12), for an immutable
  architecture. The layer boxes' minimum is 1, so standalone `encoder`/`decoder` records (which need the other count at
  0) can't be expressed, on top of the missing `kind`/`run_group`/`connection` (TD-452).
- **The role filter is never populated** with the server's roles (only "(all)"); it's editable, and typed values
  aren't URL-encoded.
- **`https://` URLs** connect with plain HTTP to port 8083 (as TD-451).
- **Hygiene:** `#ifndef` guard rather than `#pragma once`; four duplicated `http*()` helpers.

---

## 3. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Set Training sends a client `run_id` the server ignores and never sends `new_run`; Set Candidate with the typed ID gets 409; the guide documents this flow (§2) | The GUI's lifecycle actions can't take a model from training to candidate as documented, and can't start a new run | **Verified** (real server) | [TD-454](../../guides/TECHNICAL_DEBT.md#td-454-mns-manager-guis-training--candidate-workflow-fails-against-server-allocated-run-ids) |
| Success/failure judged by the substring `"error"`, not the HTTP status (§2) | Misreported results | By inspection | [TD-455](../../guides/TECHNICAL_DEBT.md#td-455-mns-manager-gui-error-detection-ui-thread-http-and-register-defaults) |
| UI-thread HTTP; arbitrary Register defaults with minimum layer counts of 1; unpopulated role filter; no URL encoding; hygiene (§2) | Freezes; easy wrong registrations | By inspection | [TD-455](../../guides/TECHNICAL_DEBT.md#td-455-mns-manager-gui-error-detection-ui-thread-http-and-register-defaults) |
| Register body lacks TD-196 fields; `https://` gap | Already tracked | — | TD-452, TD-451 |
