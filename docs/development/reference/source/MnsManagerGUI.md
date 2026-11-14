# `MnsManagerGUI` — Source File Reference

- **Files:** [`src/MnsManagerGUI.hpp`](../../../../src/MnsManagerGUI.hpp) (181 lines), [`src/MnsManagerGUI.cpp`](../../../../src/MnsManagerGUI.cpp) (1,108), [`src/MnsManagerGUI_main.cpp`](../../../../src/MnsManagerGUI_main.cpp) (74)
- **Built into:** `mns_manager_gui` (with `MnsCliCommands.cpp`, `Config.cpp`, `adai_mns`, `adai_core`, httplib and nlohmann/json), when `BUILD_GUI` (default `ON`) and Qt 5 or 6 are available; built in the local `debug` build
- **Status tags:** `MnsManagerGUI.{hpp,cpp}` `stable` 1.0.0 (promoted November 11, 2026); `MnsManagerGUI_main.cpp` `stable` 1.1.0
- **Tests:** [`tests/mns_manager_gui_widget_test.cpp`](../../../../tests/mns_manager_gui_widget_test.cpp) → `mnsManagerGuiWidgetTests` (Qt Test, 36 tests, the real window against an in-process server); its JSON helpers in [MnsJsonHelpers](MnsJsonHelpers.md) (`mnsjsonhelpersTests`, `mnsManagerGuiTests`); `SmokeTests_mns_manager_gui_main` (the arguments)
- **User guide:** [mns-manager-gui-guide.md](../../../operations/guides/mns-manager-gui-guide.md)
- **Last traced against the code:** 2026-11-11

> Code-traced reference. Behaviour marked **tested** is covered by `mnsManagerGuiWidgetTests`, which drives the real
> window offscreen; the register and lifecycle requests were also run through the real window against a real
> `mns_server` (TD-452/TD-455).

---

## 1. What this file is

A Qt desktop front end to `mns_server`, a point-and-click alternative to `mns_cli`.

| Area | Contents | Requests |
|---|---|---|
| Toolbar | Server URL, **Connect**, **Refresh**, status line | `GET /health`, then the models and roles; Refresh also the selected record |
| Models tab | State filter, role filter (the roles `GET /roles` lists, which are those with a production model, or a typed one with Enter), table (name, role, coloured state, short ID, updated) | `GET /models[?state=&role=]` |
| Roles tab | Role → production model | `GET /roles` |
| Register tab | Name, role, kind, run group, architecture, linked encoder/decoder, SIGReg, tags | `POST /models` |
| Detail panel | The selected model's record, pretty-printed in the server's field order | `GET /models/{name}` |
| Actions | **Set Training** (Start a new run), **Set Candidate** (run ID, artifact path), **Promote** (role, model), **Retire**, **Delete** (both confirmed) | `PUT /models/{n}/state`, `PUT /roles/{r}/production`, `DELETE /models/{n}` |

`main()` reads its arguments before creating `QApplication` (TD-568), so `--help`/`-h`, anywhere, prints the usage and
exits 0 even without a display. `--url` and `--config` each take a value; a missing value (`Missing value for --url`) or
any other argument (`Unknown argument: …`) prints the usage to stderr and exits 1 (**tested** by the smoke test). Without
`--url` it loads `config.mns.conf` (found by `ConfigLoader::discover_config_path()`) for `NAME_SERVICE_URL`, defaulting to
`http://localhost:8083`. The window's Server field shows the URL it starts with (**tested**).

---

## 2. Connection and requests (TD-455)

**Server URL.** `setServer()` parses the URL with `mns_cli`'s `adai::parse_url()` (host, port, and a path kept as a
prefix for every request). An unparseable URL, or an `https://` one, isn't used: httplib is built without TLS here, so
the status line says `https://` isn't supported and to use the `http://` address or `mns_cli` (**tested**). It used to
send plain HTTP to port 8083.

**One request at a time, off the GUI thread.** `runRequest(busyText, work)` runs `work` on a `std::thread` and posts the
function it returns back to the GUI thread (`QMetaObject::invokeMethod`, queued), as `ChatbotGUI` does. Until then
`setBusy()` disables the tabs, the detail and action panel (`content`), the URL field, Connect and Refresh, and the
status line shows the busy text ("Loading m1...", "Registering..."); afterwards it shows the previous status again before
the result runs. A request and the refreshes it needs (an action's `PUT` and the models list after it) run in one
`work`, so the window never interleaves them. Paths are built on the GUI thread before the request starts. **Tested:** a
click returns at once while the server holds the request, the window's controls stay disabled meanwhile, and closing it
returns in under 5 s while the server holds the request for 30 s.

**Closing.** Each request uses its own `httplib::Client`, recorded in `active_client_`; the destructor sets `closing_`,
calls `stop()` on it and joins the worker. A later request in the same `work` returns at once.

**Success by HTTP status.** `http()` returns `HttpReply{reached, status, body, error}`; `ok()` is a 2xx status. A
failure (`reportFailure()`) is "Connection failed (<reason>)" in the status line when nothing came back, otherwise a
warning dialog with the server's `error` and the status (`run_id does not match the active run (HTTP 409)`), or
`HTTP 500: <body>` when the reply has no `error` field (**tested**, as is a successful reply that contains the word
`error`). It used to treat any body containing `"error"` as a failure and anything else, any status, as success.

**URL encoding.** Model names, the role filter and the promote role are URL-encoded with `adai::url_encode()`
(**tested**: `a b/c` is requested as `/models/a%20b%2Fc`, a role `x&y` as `role=x%26y`).

---

## 3. Register tab (TD-452, TD-455)

Fields: name and role (required), **Kind** (`chatbot`, `encoder`, `decoder`, `world_model`), **Run group**, the
architecture (`d_model`, `num_heads`, `d_ff`, `max_seq_length`, and encoder/decoder layers for a chatbot or one
**Layers** count for the other kinds), **Encoder**/**Decoder** names to link a chatbot to registered models instead of
giving an architecture, **SIGReg** lambda and sketches for a world model, and tags. Choosing a kind shows only its
fields (**tested**). The architecture starts blank: it can't be changed once registered, and the old spin boxes'
defaults (128/4/512/6/6/1024) matched no config. **Load from config...** fills it from a trainer config file
(`adai::ConfigLoader::load()`, like `mns_cli register --from-config`), the layer count by kind (**tested**).

**Register Model** turns the visible fields into `mns_cli register` arguments and calls `adai::build_register_request()`
(`MnsCliCommands.cpp`), so the GUI and the CLI share one set of rules and one body: a missing architecture field, an
architecture combined with linked models, or SIGReg on another kind is refused in an "Invalid Registration" dialog with
the builder's message, before anything is sent (**tested**). Otherwise a confirmation shows the architecture to be
registered ("Registering … with d_model=…", or the linked encoder and decoder) and sends only on Yes (**tested**).
Bodies the GUI built for an encoder, a decoder, a world model and a linked chatbot were all accepted by a real
`mns_server`, which stored the kind, the standalone layer count and the SIGReg settings as sent.

---

## 4. Actions and selection

- **Selecting a model row** fills Promote's model and role, then loads the record into the detail view and its `run_id`
  into Set Candidate's Run ID. **Selecting a role row** fills Promote's role and, when the role has a production model,
  loads that model the same way (**tested**). Both tables act on `currentCellChanged` when the row changes, so a click
  and the arrow keys behave alike (TD-568, **tested**); selecting the current row again does nothing, and **Refresh**
  reloads the selected record instead (**tested**).
- **Refilling a table** (`populateModelsTable()`/`populateRolesTable()`, under a `QSignalBlocker`) never selects a row or
  loads anything (**tested**). The models table highlights the selected model's row wherever it now is, or nothing when it
  isn't listed; the roles table clears its highlight (TD-568, **tested**).
- **Set Training** sends `mns_gui::build_set_training_body(new_run)`, shows the run MNS allocated and fills the Run ID;
  the checkbox clears afterwards (TD-454, **tested**).
- **Set Candidate** sends the Run ID and artifact path (`build_set_candidate_body()`, **tested**).
- **Promote**, **Retire** and **Delete** (**tested**): Retire and Delete ask first; Delete clears the selection, the
  detail and the Run ID.
- Every action without a selected model warns "No Model Selected" and sends nothing (**tested**).
- `populateRolesTable()` also refills the role filter with the server's roles, without triggering a refresh. `GET /roles`
  lists only roles with a production model, so the current choice is kept even when it isn't one of them (a typed role
  for a model still in training); otherwise the filter would show `(all)` over a still-filtered table (**tested**).
- The server field shows the URL the window started with (**tested**).

---

## 5. Known gaps and gotchas

None open. For reference, fixed:

| Item | Fixed | Tracked as |
|---|---|---|
| Set Training sent a run ID the server ignores and never `new_run`; Set Candidate with the typed ID got 409 | November 10, 2026 | [TD-454](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-454-mns-manager-guis-training--candidate-workflow-fails-against-server-allocated-run-ids) |
| `--help` aborted without a display; a missing value or unknown argument was ignored; records loaded only on a mouse click; a refill highlighted the wrong model; an unused `QSpinBox` style | November 11, 2026 | [TD-568](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-568-mns-manager-gui-argument-handling-keyboard-selection-and-a-stale-highlight) |
| The widget class was untested | November 10, 2026: Qt Test suite | [TD-037](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-037-no-qt-test-infrastructure-for-gui-classes) |
| Success judged by the text `"error"`; HTTP on the GUI thread; arbitrary Register defaults; unpopulated role filter; no URL encoding; `#ifndef` guard and four copied HTTP helpers | November 10, 2026 | [TD-455](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-455-mns-manager-gui-error-detection-ui-thread-http-and-register-defaults) |
| Register body without kind, run group or connection; JSON helpers' parsing gaps; `https://` sent as plain HTTP | November 10, 2026 | [TD-452](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-452-mnsjsonhelpers-parsing-gaps-and-a-gui-register-body-without-td-196-fields) |

---

## 6. History

- **October 14, 2026:** first traced; filed TD-454 and TD-455 (verified against a real `mns_server`).
- **November 10, 2026 (TD-454):** Set Training's run ID replaced by **Start a new run**; Set Candidate's Run ID filled
  from the record and the reply.
- **November 10, 2026 (TD-037):** `mnsManagerGuiWidgetTests` and object names on the controls.
- **November 10, 2026 (TD-452, TD-455):** requests on a worker thread with the controls disabled, success by HTTP
  status with the server's error shown, URL encoding, the role filter filled from `/roles`, `https://` refused, one
  `http()` helper and `#pragma once`; the Register tab rebuilt on `adai::build_register_request()` with every kind, run
  group, linked models, SIGReg, a blank required architecture and Load from config. The JSON helpers moved to
  nlohmann/json ([MnsJsonHelpers](MnsJsonHelpers.md)). This document was re-traced at the same time.
- **November 11, 2026 (stable review):** the review found the Server field empty at startup and a typed role filter reset
  by a refill, both fixed in the TD-452/TD-455 change, and filed TD-568: arguments read before `QApplication` with errors
  for missing values and unknown arguments, records loaded on any change of row, refills keeping the selected model
  highlighted, Refresh reloading the selected record, the unused `QSpinBox` style removed. This document was re-traced at
  the same time.
- **November 11, 2026:** `MnsManagerGUI.{hpp,cpp}` promoted to `stable` 1.0.0 after the review against the file-status
  standard: widget tests (36) and the main-program smoke test, no open gap once TD-568 was fixed, user and reference docs
  current, shipped as `mns_manager_gui`.
