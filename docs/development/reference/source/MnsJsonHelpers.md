# `MnsJsonHelpers` — Source File Reference

- **File:** [`src/MnsJsonHelpers.hpp`](../../../../src/MnsJsonHelpers.hpp) (header-only, 291 lines)
- **Namespace:** `mns_gui`
- **Used by:** `MnsManagerGUI` only (the Qt `mns_manager_gui`, built whenever `BUILD_GUI` (default `ON`) and Qt are available). `ModelNameClient.cpp` mentions it in comments but carries its own copies (`find_string_end_client()`, `json_escape_client()`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.0`, `@adai-reviewed: 2026-09-13`
- **Tests:** [`tests/mnsjsonhelpers_test.cpp`](../../../../tests/mnsjsonhelpers_test.cpp) (37 tests); also exercised by [`tests/mns_manager_gui_test.cpp`](../../../../tests/mns_manager_gui_test.cpp) (44)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program using the real
> header.

---

## 1. What this file is

A minimal JSON reader and writer for the MNS Manager GUI, pulled out of `MnsManagerGUI.cpp` so it can be
tested without Qt (TD-037). It reads `mns_server`'s compact replies by string search rather than parsing them,
and builds the two request bodies the GUI sends.

| Function | Behaviour | GUI use |
|---|---|---|
| `find_string_end(body, start)` | Index of the closing quote, honouring backslash escapes (TD-068) | Internal |
| `json_value(body, key)` | First occurrence of `"key":`; a string value is returned **without unescaping**; any other value up to the next `, } ] space \n` | Table cells (`model_name`, `role`, `state`, `model_id`, `updated_utc`, `production_model`) |
| `json_array_objects(body, key)` | The `{…}` objects in `"key":[…]`, counting only structural braces (TD-068). **If `key` is absent, it uses the first `[` in the body** | Models and roles tables |
| `json_escape(s)` | Escapes `"`, `\`, `\n` only | Request bodies |
| `json_pretty(s)` | Re-indents compact JSON, skipping string contents (TD-102) | Detail view |
| `parse_tags(str)` | `k=v, k2=v2` → ordered pairs; entries without `=` skipped; trims spaces (not tabs) | Register tab |
| `build_register_model_body(name, role, arch, tags)` | `POST /models` body: name, role, the six architecture ints, tags | Register tab |
| `build_set_candidate_body(run_id, path)` | `PUT …/state` body for `candidate`; artifact with empty host and checksum, format `adai-native` | Candidate action |
| `ParsedUrl::from(url)` | Host and port (default 8083), `https://` stripped | Connection |

`mns_server` writes each record's top-level fields first (`model_id`, `model_name`, `role`, …, `state`,
`updated_utc`) and nested objects (`progress`, `training_history`) after them, so `json_value()`'s first match is the
top-level one for every key the GUI reads.

---

## 2. Gaps (verified unless noted)

None of these affects training or serving; the GUI is optional and display-only apart from the register and
candidate actions.

- **String values aren't unescaped.** `json_value()` returned `C:\\models\\v3.bin` for a path and `caf\u00e9 \"x\"`
  for a label: what's displayed is the escaped JSON text, not the value.
- **Only the compact form is understood.** `"state" : "training"` (spaces around the colon) returned empty;
  an object value returns a fragment (`{"d_model":512`). Fine for `mns_server`'s output, fragile for anything else.
- **`json_array_objects()` falls back to any array.** With `"models"` absent (e.g. an error body), it parsed objects
  from the first other array in the body.
- **`json_escape()` is incomplete.** A tab, carriage return or `\x01` in a name, role or tag value is written raw,
  which isn't valid JSON (the same gap as `mns_cli`'s, TD-449).
- **`json_pretty()`** turns `{}` and `[]` into three-line blocks with a blank indented line. Cosmetic.
- **The register body can't express TD-196 models (by inspection).** It has no `kind`, `run_group` or `connection`,
  so the GUI can register only legacy bundled `chatbot` records; it can't register an encoder, decoder or world
  model, link an encoder/decoder pair, or set a run group.
- **`ParsedUrl` accepts `https://`** but keeps port 8083, the same gap as TD-451.
- **Duplication (by inspection).** The same hand-written JSON escaping and string scanning exists in at least
  `MnsJsonHelpers.hpp`, `MnsCliCommands.cpp`, `ModelNameClient.cpp` (explicitly duplicated), `ModelNameService.cpp`,
  `MetricsPushClient.cpp` and `TrainingMetricsAPI.cpp`, with different escaping coverage in each. The repository
  already vendors `external/nlohmann` (used by `ModelSerializer`, `ChatbotAPI`, `DataFetcher`), the replacement
  TD-226 already recommends for `ChatbotAPI`.

---

## 3. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No unescaping; compact-only key matching; array fallback; incomplete `json_escape()`; empty-container pretty-printing (§2) | GUI shows escaped text; invalid JSON for unusual input | **Verified** | [TD-452](../../guides/TECHNICAL_DEBT.md#td-452-mnsjsonhelpers-parsing-gaps-and-a-gui-register-body-without-td-196-fields) |
| Register body lacks `kind`/`run_group`/`connection` (§2) | The GUI can't manage TD-196 encoder/decoder/world-model records | By inspection | [TD-452](../../guides/TECHNICAL_DEBT.md#td-452-mnsjsonhelpers-parsing-gaps-and-a-gui-register-body-without-td-196-fields) |
| Hand-written JSON duplicated across the MNS and metrics code while `nlohmann/json` is vendored (§2) | Each copy has its own escaping gaps (TD-443, TD-449) | By inspection | [TD-453](../../guides/TECHNICAL_DEBT.md#td-453-hand-written-json-duplicated-across-mns-and-metrics-code) |
| `https://` accepted with plain HTTP | Already tracked | — | TD-451 |
