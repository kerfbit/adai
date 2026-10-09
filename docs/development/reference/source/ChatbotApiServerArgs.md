# `ChatbotApiServerArgs` — Source File Reference

- **Files:** [`src/ChatbotApiServerArgs.hpp`](../../../../src/ChatbotApiServerArgs.hpp), [`src/ChatbotApiServerArgs.cpp`](../../../../src/ChatbotApiServerArgs.cpp)
- **Built into:** the `chatbot_api_server` executable (listed in its sources in `src/CMakeLists.txt`), and compiled directly into its test binary
- **Namespace:** `adai`
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.5.0` (hpp) / `0.6.0` (cpp), `@adai-reviewed: 2026-09-12`
- **Tests:** [`tests/chatbot_api_server_args_test.cpp`](../../../../tests/chatbot_api_server_args_test.cpp) → `chatbotApiServerArgsTests` (20 tests, ctest `ChatbotApiServerArgsTests`)
- **Last traced against the code:** 2026-10-08

> Code-traced reference. Every edge case below was confirmed by running the real parser (a
> scratch program compiled against `ChatbotApiServerArgs.cpp` and `adai_core`).

---

## 1. What this file is

The command-line layer of `chatbot_api_server`: three free functions and one result struct that
turn `argv` into settings. TD-035 split them out of `ChatbotAPIServer.cpp`'s `main()` so they can be
unit-tested without loading a tokenizer or model, or opening a port. The same pattern is used for
the other daemons' `*_args` files (`mns_server`, `registry_server`, `metrics_api_server`,
`dataset_manager`, `incremental_trainer`), each with its own `*_args_test.cpp`.

### How `main()` uses it (two passes)

```text
argv ──► extract_config_path_arg()            first pass: only --config
         ConfigLoader::discover_config_path() --config > ./config.chatbot.conf > /etc/adai/… > legacy config.conf
         ConfigLoader::load()                 file + env vars
argv ──► apply_chatbot_api_server_args()      second pass: every other flag overrides file/env
         help → print_usage, exit 0;  error → message + usage, exit 1
         validate_chatbot_api_server_config() → error → exit 1
```

Two passes are needed because `--config` must be known *before* the file loads, but every other
flag must be applied *after* it so the CLI wins (CLI > env > file > defaults, per CLAUDE.md).

### Why it matters

It's the only input validation `chatbot_api_server` does at startup (see §4), and it decides which
inference mode the server runs in. Values it lets through go straight into the port binding,
session expiry and generation settings.

---

## 2. API

### `std::optional<std::string> extract_config_path_arg(int argc, char* argv[])`

Returns the value after the first `--config`, or `nullopt`. A trailing `--config` with no value
returns `nullopt` (so normal discovery applies). The test `MissingValueIsIgnoredNotCrashed` covers
this.

**Used by:** `ChatbotAPIServer.cpp` `main()` (≈ line 153), feeding `discover_config_path()`.

### `ChatbotApiServerArgsResult apply_chatbot_api_server_args(int argc, char* argv[], ServiceConfig& config)`

Walks `argv` once, applying flags to `config` **in place**. Returns early with `help = true` on
`--help`/`-h` (config untouched from that point on), or with `error = true` and
`"Unknown argument: <arg>"` on anything unrecognized. `--config` and its value are skipped.

| Flag | Sets | Parsed with |
|---|---|---|
| `--model <path>` | `config.model_path` | string |
| `--vocab <path>` | `config.vocab_path` | string |
| `--draft-model <path>` | `config.draft_model_path` (enables speculative decoding) | string |
| `--speculative-candidates <n>` | `config.speculative_num_candidates` | `atoi` |
| `--port <n>` | `config.port` | `atoi` |
| `--timeout <min>` | `config.session_timeout` | `atoi` |
| `--log-level <lvl>` | `config.log_level` | string |
| `--d-model`, `--num-heads`, `--d-ff`, `--enc-layers`, `--dec-layers`, `--max-seq-len` `<n>` | architecture fallbacks (overwritten from MNS when it resolves the model; see CLAUDE.md) | `atoi` |
| `--max-gen-len <n>` | `config.max_gen_length` | `atoi` |
| `--temperature <f>`, `--top-p <f>` | `config.temperature`, `config.top_p` | `atof` |
| `--strategy <s>` | `config.strategy` | string |
| `--profile` | `result.profile` | flag |
| `--batched-inference` | `result.batched_inference` | flag |
| `--batch-timeout-ms <n>` | `result.batch_timeout_ms` (default 50) | `atoi` |
| `--pipeline-inference` | `result.pipeline_inference` | flag |
| `--integrated-inference` | `result.integrated_inference` | flag |

The four mode flags live on the **result**, not on `ServiceConfig`: they're per-process runtime
toggles, not persisted or hot-reloadable settings (TD-038). The flag list matches
`ChatbotAPIServer.cpp`'s `print_usage()` exactly.

There are no flags for `top_k` or `beam_width`. `ServiceConfig` has both, but
`ChatbotAPIServer.cpp` never copies them into `ChatbotAPI::GenerationConfig`, at startup or on
reload. That will be covered in the `ChatbotAPIServer.cpp` reference.

**Used by:** `main()` (≈ line 161); `cli.profile`/`batched_inference`/`pipeline_inference`/
`integrated_inference` drive `ChatbotAPI::enable_*()` (≈ lines 517–560).

### `std::optional<std::string> validate_chatbot_api_server_config(const ServiceConfig& config)`

Returns an error message if `config.vocab_path` is empty, otherwise `nullopt`. That's the **only**
check.

**Used by:** `main()` (≈ line 173).

---

## 3. Verified behaviour of edge cases

| Input | Result | Consequence |
|---|---|---|
| `--port` (no value) | `error`, message `Unknown argument: --port` | Misleading: the flag is known; its value is missing |
| `--port abc` | accepted, `port = 0` | Port 0 asks the OS for any free port, so the server starts somewhere unannounced |
| `--port 8080x` | accepted, `port = 8080` | Trailing garbage silently dropped |
| `--temperature hot` | accepted, `temperature = 0` | Temperature 0 means greedy decoding |
| `--timeout -5` | accepted, `session_timeout = -5` | Every session counts as expired at the next `/health` |
| `--strategy bogus` | accepted | `ChatbotAPI` silently treats unknown strategies as `nucleus` |
| `--batched-inference --pipeline-inference --integrated-inference` | all three `true`, no error | See below |
| `--config` (no value) | ignored, no error | Discovery falls back to default paths silently |

`atoi`/`atof` return 0 for non-numbers and stop at the first non-digit, and nothing range-checks
the results.

**The engine flags aren't mutually exclusive.** The header says at most one should be set and
that `ChatbotAPIServer.cpp` "only ever enables one". In fact `main()` checks each flag with an
independent `if`, so passing several enables **all** of them: three sets of worker threads, plus the
pipeline mode's vocab reload. `ChatbotAPI::generate_response()` then silently uses only the first
in its precedence order (batched, pipeline, integrated).

---

## 4. Validation gap: startup vs. reload

`ConfigLoader::validate()` (`src/Config.cpp` ≈ line 1070) checks port 1–65535,
`session_timeout ≥ 1`, log level and size limits, `d_model` range, and more. But it's only called
from `ConfigLoader::reload()`, the `SIGHUP` path. At startup, `ConfigLoader::load()` doesn't call it,
and this file's validator checks only `vocab_path`. So:

- invalid values from the **CLI, env or file** are accepted at startup (§3);
- the same file would be **rejected** on the next `SIGHUP`, with "keeping current configuration".

**`SIGHUP` reload also drops CLI overrides.** `reload()` rebuilds config from file and env only,
and `main()` then pushes `max_gen_length`, `temperature`, `top_p` and `strategy` from it into
`ChatbotAPI`. A server started with `--temperature 0.7` reverts to the file's temperature on the
first reload. (The mode flags on the result aren't affected, since they're never reloaded.)

---

## 5. Tests

`chatbotApiServerArgsTests` (20) covers: `--config` extraction (present, absent, missing value);
no-args; `--help`/`-h` short-circuit; unknown-argument error; `--config` skipping; every string,
int and float flag being applied; draft-model flags; each mode flag defaulting false; CLI
overriding pre-set config; and the vocab-path check. None of the §3 edge cases is tested.

The header comment refers to the test file as `ChatbotApiServerArgs_test.cpp`; the real name is
`chatbot_api_server_args_test.cpp`.

---

## 6. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Verified | Tracked as |
|---|---|---|---|
| Non-numeric/partial numbers become 0 or truncated via `atoi`/`atof` (§3) | Port 0, temperature 0 and similar applied silently | Yes | [TD-240](../../guides/TECHNICAL_DEBT.md#td-240-chatbot_api_server-silently-accepts-non-numeric-and-partial-numbers) |
| No range checks at startup; `ConfigLoader::validate()` only runs on reload (§4) | Negative timeouts, bad ports accepted, then rejected on reload | Yes (parser), by inspection (reload) | [TD-241](../../guides/TECHNICAL_DEBT.md#td-241-chatbot_api_server-never-validates-its-configuration-at-startup) |
| `SIGHUP` reload discards CLI generation overrides (§4) | Settings silently revert on reload | By inspection | [TD-242](../../guides/TECHNICAL_DEBT.md#td-242-sighup-reload-discards-chatbot_api_servers-cli-generation-overrides) |
| Engine-mode flags not mutually exclusive; header's "only ever enables one" is false (§3) | Unused engines and threads started; silent precedence | Yes (parser), by inspection (server) | [TD-243](../../guides/TECHNICAL_DEBT.md#td-243-chatbot_api_server-starts-every-engine-mode-flag-passed-not-just-one) |
| Missing value reported as "Unknown argument" (§3) | Confusing error | Yes | [TD-245](../../guides/TECHNICAL_DEBT.md#td-245-chatbot_api_server-argument-errors-are-misleading-or-silent) |
| `--strategy` not validated (§3) | Typos silently become nucleus sampling | Yes | [TD-241](../../guides/TECHNICAL_DEBT.md#td-241-chatbot_api_server-never-validates-its-configuration-at-startup) |
| `--batch-timeout-ms` without `--batched-inference` silently ignored | Minor | By inspection | [TD-243](../../guides/TECHNICAL_DEBT.md#td-243-chatbot_api_server-starts-every-engine-mode-flag-passed-not-just-one) |
| No CLI flags for `top_k`/`beam_width`; the server never applies them even from config (§2) | Config keys have no effect | By inspection | [TD-244](../../guides/TECHNICAL_DEBT.md#td-244-chatbot_api_server-ignores-the-top_k-and-beam_width-settings) |
| Header names the wrong test file (§5) | Stale comment | By inspection | [TD-245](../../guides/TECHNICAL_DEBT.md#td-245-chatbot_api_server-argument-errors-are-misleading-or-silent) |
