# `MnsJsonHelpers` — Source File Reference

- **File:** [`src/MnsJsonHelpers.hpp`](../../../../src/MnsJsonHelpers.hpp) (header-only, 150 lines)
- **Namespace:** `mns_gui`
- **Used by:** `MnsManagerGUI` only (the Qt `mns_manager_gui`). `ModelNameClient.cpp` mentions it in comments but carries its own code
- **Depends on:** the vendored `external/nlohmann/json.hpp` (TD-452)
- **Status tag:** `@adai-status: stable`, `@adai-version: 2.0.0`, `@adai-reviewed: 2026-11-10`
- **Tests:** [`tests/mnsjsonhelpers_test.cpp`](../../../../tests/mnsjsonhelpers_test.cpp) (32 tests); also [`tests/mns_manager_gui_test.cpp`](../../../../tests/mns_manager_gui_test.cpp) (37, of which the live ones run against a real `mns_server` when one is reachable) and the GUI's widget tests
- **Last traced against the code:** 2026-11-10

> Code-traced reference.

---

## 1. What this file is

The MNS Manager GUI's JSON reading and writing, kept out of the widget so it can be tested without Qt (TD-037). Since
TD-452 it parses with nlohmann/json: an `ordered_json`, so a pretty-printed record keeps the server's field order. Text
that isn't JSON gives empty results, never an exception.

| Function | Behaviour | GUI use |
|---|---|---|
| `json_value(body, key)` | The value of `key` nearest the top of the document (breadth-first, so a record's own `run_id` wins over one in its `training_history`): a string unescaped, anything else as compact JSON (`512`, `true`, `null`, `{"host":"h"}`). Empty when missing or not JSON | Table cells, the record's `run_id`, a failed reply's `error` |
| `json_array_objects(body, key)` | The elements of the array under `key`, each as compact JSON; an empty `key` reads a top-level array. Empty when there is no such array | Models and roles tables |
| `json_escape(s)` | `s` escaped for a JSON string literal, without the quotes: quotes, backslashes and every control character; invalid UTF-8 becomes U+FFFD | Request bodies |
| `json_pretty(s)` | Two-space indentation in the server's field order (`{}` and `[]` stay on one line); text that isn't JSON comes back unchanged | Detail view |
| `parse_tags(str)` | `k=v, k2=v2` → ordered pairs; entries without `=` skipped; trims spaces | Register tab (passed to `adai::build_register_request()` as `--tag k=v`) |
| `build_set_training_body(new_run)` | `{"state":"training"}`, plus `"new_run":true` (TD-454) | Set Training |
| `build_set_candidate_body(run_id, path)` | `PUT …/state` body for `candidate`; artifact with empty host and checksum, format `adai-native` | Set Candidate |

The register body is no longer built here: the GUI uses `mns_cli`'s `adai::build_register_request()`, and the server URL
is parsed with `adai::parse_url()` (both `MnsCliCommands`), so the GUI and the CLI share them (TD-452).

---

## 2. Known gaps

None open. TD-453 (hand-written JSON duplicated across the MNS and metrics code) no longer applies to this file.

---

## 3. History

- **September 13, 2026 (TD-037):** extracted from `MnsManagerGUI.cpp` with its first tests; TD-068 and TD-102 fixed
  string-aware brace counting and pretty-printing in the hand-written scanner.
- **October 14, 2026:** traced; filed TD-452 (no unescaping, compact-only key matching, object values returned as
  fragments, a fallback to whichever array came first, incomplete escaping, three-line empty containers, a register
  body without `kind`/`run_group`/`connection`, `https://` kept on port 8083).
- **November 10, 2026 (TD-454):** `build_set_training_body()`.
- **November 10, 2026 (TD-452), 2.0.0:** the helpers parse with nlohmann/json, fixing each parsing and escaping gap (nine
  new tests, each failing on the old header); `find_string_end()`, `build_register_model_body()`/`RegisterModelArch`
  and `ParsedUrl` removed in favour of nlohmann and `MnsCliCommands`. This document was re-traced at the same time.
