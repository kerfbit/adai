# `ChatbotAPI` — Source File Reference

- **Files:** [`src/ChatbotAPI.hpp`](../../../../src/ChatbotAPI.hpp), [`src/ChatbotAPI.cpp`](../../../../src/ChatbotAPI.cpp)
- **Library:** `adai_api` (links `adai_models`, `adai_nlp`; uses vendored cpp-httplib)
- **Status tag:** `@adai-status: beta` (capped by TD-033), `@adai-version: 0.10.0` (hpp) / `0.11.0` (cpp), `@adai-reviewed: 2026-09-14`
- **Tests:** [`tests/chatbotapi_test.cpp`](../../../../tests/chatbotapi_test.cpp) → `chatbotapiTests` (72 tests; `friend class ChatbotAPITest` gives them private access)
- **HTTP contract for clients:** [rest-api.md](../../api/rest-api.md) (request/response shapes); this page documents the class behind it
- **Last traced against the code:** 2026-10-08

> Code-traced reference. Behaviour flagged as surprising in the JSON helpers was confirmed by
> running the real static functions (a scratch program linked against the built `adai_api`
> library); concurrency findings are from reading the code paths.

---

## Contents

1. [What this file is](#1-what-this-file-is)
2. [Where it's used](#2-where-its-used)
3. [Types](#3-types)
4. [Construction, routes and lifecycle](#4-construction-routes-and-lifecycle)
5. [`generate_response()`: the inference dispatcher](#5-generate_response-the-inference-dispatcher)
6. [Endpoint handlers](#6-endpoint-handlers)
7. [Sessions](#7-sessions)
8. [Batch generation](#8-batch-generation)
9. [JSON helpers](#9-json-helpers)
10. [Mode switches and profiling](#10-mode-switches-and-profiling)
11. [Threading model](#11-threading-model)
12. [Known gaps and gotchas (summary)](#12-known-gaps-and-gotchas-summary)

---

## 1. What this file is

`ChatbotAPI` is the **HTTP front end of `chatbot_api_server`** (port 8080 by default). It owns a
cpp-httplib server, maps nine routes to handlers, keeps multi-turn conversation sessions in memory,
and routes every generation request through one dispatcher, `generate_response()`. That dispatcher
picks one of seven inference paths: RAG, speculative decoding, the three TD-038 engines, the
GPU-resident decoder, or the plain CPU `TextGenerator` path.

It doesn't own the model or tokenizer (raw non-owning pointers; `ChatbotAPIServer.cpp` owns them).
It does own the optional inference engines (`unique_ptr`), so their worker threads stop when it's
destroyed.

### Why it matters

Every chat a user has with a trained model over the network goes through this class: the
`chatbot` CLI client, the Android chat app, and anything else hitting port 8080. (The Qt GUI,
`chatbot_gui`, runs the model in-process and doesn't use this class.) Its parsing, session handling and mode
selection decide what text reaches the model and what the client gets back. It's also the
integration point TD-033 (GPU decode), TD-038 (speculative, batched, pipeline, integrated engines,
profiling), TD-053 (session export/import), TD-063 (JSON escaping) and TD-095 (session ID echo)
all landed in.

---

## 2. Where it's used

| Location | Uses |
|---|---|
| `src/ChatbotAPIServer.cpp` (the `chatbot_api_server` binary) | Constructs it with the loaded model and tokenizer, optional `--draft-model`, port and session timeout; applies `set_generation_config()` from CLI/config, and again on `SIGHUP` config reload while serving; calls `enable_profiling()`, `enable_batched_inference()`, `enable_pipeline_inference()` (inside a try/catch that disables the mode on failure), `enable_integrated_inference()`, `enableRAG()`; runs `start()` on a dedicated thread and `stop()` from the main thread on `SIGINT`/`SIGTERM` |
| `tests/chatbotapi_test.cpp` | Unit and integration tests, including the TD-038 mode wiring and JSON helpers |
| Clients (over HTTP) | `src/ChatbotCLI.*` (`chatbot`), the Android chat app; see [rest-api.md](../../api/rest-api.md) |

---

## 3. Types

```cpp
struct Session {                                // one conversation
    std::unique_ptr<ConversationContext> context;   // defaults: 10 messages, 2048 tokens
    std::chrono::steady_clock::time_point last_access;
};

struct ChatbotAPI::GenerationConfig {           // server-wide defaults (set_generation_config)
    size_t max_length = 100;  float temperature = 1.0f;  float top_p = 0.9f;
    size_t top_k = 50;        std::string strategy = "nucleus";  size_t beam_width = 4;
};

struct ChatbotAPI::BatchRequest  { messages, session_ids, config };   // declared, unused
struct ChatbotAPI::BatchResponse { responses, session_ids, success, error, BatchStats stats };
```

`strategy` is one of `greedy`, `beam`, `temperature`, `top_k`, `nucleus`; anything else is treated
as `nucleus`. Clients can't override `GenerationConfig` per request: every handler copies
`default_config_`. That matches [rest-api.md](../../api/rest-api.md), which documents generation
settings as server flags only, although `handle_chat()`'s comment says "from request or use
defaults". `BatchRequest` is declared but never used.

---

## 4. Construction, routes and lifecycle

```cpp
ChatbotAPI(EncoderDecoderModel* model, BPETokenizer* tokenizer, int port = 8080,
           int session_timeout_minutes = 30, EncoderDecoderModel* draft_model = nullptr,
           int speculative_num_candidates = 4);
```

The constructor registers every route. Copying and moving are deleted, because the route lambdas
capture `this`.

| Route | Handler | Status on thrown exception | Notes |
|---|---|---|---|
| `POST /chat` | `handle_chat()` | 500 | Stateless single turn |
| `POST /chat/session` | `handle_chat_session()` | 500 | Multi-turn; returns `session_id` |
| `POST /clear-session` | `handle_clear_session()` | 400 | Empties a session's history (keeps the session) |
| `POST /chat/session/export` | `handle_export_session()` | 400 | TD-053 |
| `POST /chat/session/import` | `handle_import_session()` | 400 | TD-053 |
| `GET /health` | `handle_health()` | — | Also runs session expiry (§7) |
| `GET /admin/profile` | `handle_profile()` | — | Opt-in timing stats (§10) |
| `POST /chat/batch` | `handle_batch_chat()` | 500 | §8 |
| `POST /chat/batch-session` | `handle_batch_chat_session()` | 500 | §8 |

> **Validation errors return HTTP 200.** A handler that *returns* an error, such as "Missing
> 'message' field", "Session not found" or a failed import, sends `{"success":false,...}` with
> status **200**. Only *thrown* exceptions map to 400 or 500. Clients must check `success`, not
> the status code.

- **`start()`** — blocking `listen("0.0.0.0", port_)`. It always binds to all interfaces; there's
  no host setting. Returns `false` if it was already running or the bind failed.
- **`stop()`** — calls `server.stop()` if running. The destructor calls it.
- **`is_running()`** — reads `running_`.

`running_` is a plain `bool`. `chatbot_api_server` calls `start()` on a server thread and `stop()`
from the main thread, so both threads write it with no synchronization, which is a data race
(in practice benign on x86, but undefined behaviour). It should be `std::atomic<bool>`.

---

## 5. `generate_response()`: the inference dispatcher

```cpp
std::string generate_response(const std::string& input, const GenerationConfig& config);
```

Wrapped in `PROFILE_SCOPE(profiler_, "generate_response")`. It tokenizes `input` once
(`encode(input, false)`; the user text is the *encoder* input), then takes the **first** matching
path:

| # | Path | When | Honours `config` | Notes |
|---|---|---|---|---|
| 1 | **RAG** (`rag_engine_->generate(input)`) | `enableRAG()` called | No | Checked **while holding `config_mutex_`** (see below) |
| 2 | **Speculative decoding** | constructed with a draft model | `temperature`, `max_length`; greedy iff `strategy == "greedy"` | Draft and target `TextGenerator`s + `SpeculativeDecoder` |
| 3 | **Batched engine** | `enable_batched_inference()` | `max_length`, `temperature`, `top_p`, `top_k`; **not** `strategy`/`beam_width` | Blocks on a future; see [BatchedInferenceEngine.md](BatchedInferenceEngine.md), TD-211–214 |
| 4 | **Pipeline engine** | `enable_pipeline_inference()` | `max_length` only; always greedy | Encoder uses the model's *private* tokenizer |
| 5 | **Integrated engine** | `enable_integrated_inference()` | `max_length` only; always greedy | |
| 6 | **GPU-resident decode** | built with GPU support and `GPUManager::is_available()` | all fields | `gpu_generate_response_with_strategy()` (TD-033) |
| 7 | **CPU `TextGenerator`** | otherwise | all fields | Fresh `TextGenerator` per request (seed 0 means random), `model_->forward()` full recompute per step; strategy switch picks `generate_greedy/beam_search/sampling/top_k/nucleus` |

Any exception is rethrown as `std::runtime_error("Generation failed: " + what)`. The route
handlers turn that into a 500 response containing the message, so internal error text reaches
clients.

> **RAG holds `config_mutex_` for the entire generation.** Path 1 is written as
> `{ lock_guard(config_mutex_); if (rag_engine_) return rag_engine_->generate(input); }`, so the
> lock is held until generation returns. Every other request takes `config_mutex_` briefly to copy
> `default_config_`, and `set_generation_config()` needs it too, so with RAG enabled **all requests
> are serialized behind whichever RAG generation is running**. The fix is to copy the
> `shared_ptr` under the lock and generate after releasing it.

Model access itself is serialized by `EncoderDecoderModel`'s internal `model_mutex_` (TD-156,
TD-033), so concurrent inline requests interleave token steps rather than racing on layer state.

---

## 6. Endpoint handlers

| Handler | Request fields | Behaviour |
|---|---|---|
| `handle_chat(body)` | `message` | Copies default config, `generate_response(message)`, returns `{"success":true,"response":...}` |
| `handle_chat_session(body)` | `message`, optional `session_id` | `get_or_create_session()`; if the ID was empty, recovers the new ID by reverse pointer lookup (TD-095 fix); appends the user message, formats the whole conversation with `ConversationContext::format_for_model()`, generates, appends the reply, returns the response plus `session_id` |
| `handle_clear_session(body)` | `session_id` | Clears that session's history under `sessions_mutex_`; error response if not found |
| `handle_export_session(body)` | `session_id` | Returns `ConversationContext::serialize()` JSON-escaped in `data` |
| `handle_import_session(body)` | `data`, optional `session_id` | Creates or reuses the session (same reverse lookup), `deserialize(data)`, returns `message_count` |
| `handle_health()` | — | Runs `cleanup_expired_sessions()`, returns `{"status":"ok","active_sessions":N}` |
| `handle_profile()` | — | §10 |
| `handle_batch_chat(body)` / `handle_batch_chat_session(body)` | `messages[]` (and `session_ids[]`) | §8. A mismatched or missing `session_ids` array is replaced by all-new sessions |

---

## 7. Sessions

Stored in `std::unordered_map<std::string, std::unique_ptr<Session>> sessions_`, guarded by
`sessions_mutex_`.

- **`create_session_id()`** (static) — 32 hex digits from a `static std::mt19937` seeded once from
  `std::random_device`. It's only called under `sessions_mutex_`, so its shared generator is safe.
  `mt19937` isn't a cryptographic generator, though, and a session ID is the only thing protecting
  a conversation (export returns its full history to anyone who presents the ID), so IDs should
  come from a CSPRNG.
- **`get_or_create_session(id)`** — under the lock: returns the existing session (refreshing
  `last_access`), or creates one. **Client-chosen IDs are accepted as-is**: an unknown non-empty
  ID creates a session under that exact string, with no length, format or count limit. (The
  ChatbotCLI relies on this, per the project note that clients should send their own non-empty
  IDs.)
- **`cleanup_expired_sessions()`** / **`is_session_expired()`** — erase sessions idle for
  `session_timeout_` or longer (default 30 min). **This only runs from `GET /health`.** `start()`
  has a comment saying a background thread should do it, but none exists, so on a server nobody
  health-checks, sessions never expire.

Together, those last two points mean any client can grow `sessions_` without bound (one request
per new ID), and nothing reclaims it unless `/health` is being polled.

> **Session pointers outlive the lock.** `get_or_create_session()` returns a raw `Session*` after
> releasing `sessions_mutex_`. `handle_chat_session()` and `handle_import_session()` then mutate
> `session->context` with no lock held, possibly across a multi-second generation. That allows:
> - two concurrent requests on the **same** session racing on its `ConversationContext`, which
>   has no internal locking;
> - `/clear-session` clearing the context mid-request;
> - `/health`'s cleanup **erasing** the session mid-request, a use-after-free. This one is unlikely
>   with a 30-minute timeout, since `last_access` is refreshed at the start of each request, but
>   it isn't prevented.
>
> A per-session mutex plus `shared_ptr<Session>` ownership would close all three.

---

## 8. Batch generation

- **`generate_batch_responses(inputs, config)`** — encodes every input, computes `BatchStats` from
  `create_dynamic_batches(…, 32, 10, PAD)` (stats only; TD-217), then generates **each input
  sequentially in input order** via a fresh `TextGenerator` and `model_->forward()`.

  It does **not** call `generate_response()`, so it bypasses RAG, speculative decoding, all three
  engines **and the GPU path**: batch requests always run on the plain CPU path even on a GPU host.
  It also has no limit on the number of messages; one request can occupy an HTTP thread for as
  long as its inputs take.
- **`generate_batch_session_responses(inputs, session_ids, config)`** — for each input, gets or
  creates a session (reverse lookup for new IDs), appends the user message, formats the context,
  then calls `generate_batch_responses()` and appends each reply to its session. It has the same
  unlocked-`Session*` exposure as §7, and partial failure leaves user messages appended without
  replies.

Both return `BatchResponse` (`success=false` with an error on exception), serialized by
`create_batch_json_response()`.

---

## 9. JSON helpers

All are public static members (public so the tests can call them). They're a hand-rolled,
substring-based JSON reader and writer, even though **nlohmann/json is already vendored** (used by
`ModelSerializer.cpp`).

### `parse_json_string(json, key)`

Finds the first occurrence of `"key"`, the next `:`, the next `"`, then scans to an unescaped
`"`, and unescapes `\n \r \t \" \\`. Verified behaviour:

| Input | Returned | Correct |
|---|---|---|
| `{"session_id":123,"message":"hi"}` → `session_id` | `"message"` (the next key's name) | `""` / error |
| `{"message":"path C:\\","session_id":"abc"}` → `message` | `path C:\",` (runs past the closing quote) | `path C:\` |
| `{"message":"caf\u00e9"}` | `caf\u00e9` (escape left literal) | `café` |
| `{"note":"the \"message\": field","message":"real"}` | `real` | correct here, but only because the embedded key text was escaped |

A missing key returns `""`, which handlers treat as "missing", so `{"message":""}` and a missing
`message` look the same.

### `parse_json_array(json, key)`

Finds `[`, then the **first `]` anywhere after it**, and parses strings in between.

> **Verified: any message containing `]` breaks the whole array.**
> `{"messages":["see [1] here","second"]}` returns **0 items**, so `/chat/batch` and
> `/chat/batch-session` answer "Missing or empty 'messages' array" for a perfectly valid request.
> Brackets in chat text (citations, code, lists) are common.

### `escape_json_string(s)`

Escapes `"`, `\`, `\n`, `\r`, `\t`. **Other control characters (U+0000–U+001F) pass through raw**
(verified: `\x01` and `\b` emitted as bytes 01 and 08), which makes the output invalid JSON that
strict clients reject. This is the single escaping function since TD-063, used for every string
field.

### `create_json_response()`, `create_error_response()`, `create_batch_json_response()`

These build `{"success":true,"response":…}`, `{"success":false,"error":…}`, and the batch shape
(`responses`, optional `session_ids`, and the `stats` object TD-217 relabels).

---

## 10. Mode switches and profiling

| Method | Effect |
|---|---|
| `set_generation_config(cfg)` | Replaces `default_config_` under `config_mutex_` |
| `enableRAG(shared_ptr<RAGInference>)` | Takes over every `generate_response()` (path 1) |
| `enable_profiling(bool)` | Gates `GET /admin/profile`. Timing is always collected; this only controls exposure |
| `enable_batched_inference(cfg)` | Creates the `BatchedInferenceEngine` with a placeholder default `model_fn` that throws if ever used, and a non-owning `shared_ptr` to `tokenizer_` |
| `enable_pipeline_inference(vocab_path, cfg)` | Loads `vocab_path` into the **encoder's private tokenizer**, then creates the `StandardPipelineEngine`. Throws on a bad vocab; the server catches that and disables the mode, but a mid-parse failure can leave the encoder's tokenizer gutted (TD-223) |
| `enable_integrated_inference(cfg)` | Creates the `IntegratedInferenceEngine` with `tokenizer_` |

At most one of the three engine modes should be enabled; if several are, path order (§5) decides
silently. `chatbot_api_server` only ever enables one.

**`handle_profile()`** returns `{"enabled":false,...}` unless enabled. Otherwise it returns
`generate_response`'s call count and total/mean/median/min/max/p95/p99 times (ms) from
`PerformanceProfiler`, which is thread-safe. The endpoint has no authentication; it's opt-in via
`--profile`.

---

## 11. Threading model

cpp-httplib serves each request on a pool thread, so all handlers can run concurrently.

| Shared state | Protection | Gaps |
|---|---|---|
| `sessions_` map | `sessions_mutex_` | `Session*` used after unlock (§7) |
| `Session::context` | **none** | Concurrent same-session requests race (§7) |
| `default_config_`, `rag_engine_` | `config_mutex_` | RAG generation runs while holding it (§5) |
| model | `EncoderDecoderModel::model_mutex_` | — |
| `profiler_` | internal mutex | — |
| `running_` | **none** (plain `bool`) | Written from both the server and main threads (§4) |
| optional engines | set during startup before `start()`; their own internal locking | Enabling one after `start()` would race; nothing prevents it |

---

## 12. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Verified | Tracked as |
|---|---|---|---|
| `parse_json_array()` stops at the first `]` anywhere (§9) | Batch requests with `]` in any message fail entirely | Yes: 0 items | [TD-226](../../guides/TECHNICAL_DEBT.md#td-226-chatbatch-rejects-any-request-whose-messages-contain-) |
| `parse_json_string()` mis-parses non-string values, values ending in `\\`, and `\uXXXX` (§9) | Wrong session IDs and messages; garbled text | Yes | [TD-228](../../guides/TECHNICAL_DEBT.md#td-228-chatbotapiparse_json_string-mis-parses-common-json) |
| `escape_json_string()` emits raw control characters (§9) | Invalid JSON responses | Yes | [TD-229](../../guides/TECHNICAL_DEBT.md#td-229-chatbotapi-responses-can-be-invalid-json-unescaped-control-characters) |
| RAG generation holds `config_mutex_` (§5) | With RAG on, all requests serialize | By inspection | [TD-230](../../guides/TECHNICAL_DEBT.md#td-230-rag-mode-serializes-every-chatbotapi-request-behind-config_mutex_) |
| `Session*` used after unlock; `ConversationContext` unsynchronized (§7) | Same-session races; use-after-free possible with cleanup | By inspection | [TD-227](../../guides/TECHNICAL_DEBT.md#td-227-chatbotapi-session-pointers-are-used-outside-the-session-lock) |
| Session expiry only runs on `GET /health` (§7) | Unbounded session growth if `/health` isn't polled | By inspection | [TD-231](../../guides/TECHNICAL_DEBT.md#td-231-chatbotapi-sessions-only-expire-when-get-health-is-called) |
| Client-chosen session IDs accepted unbounded; IDs from `mt19937` (§7) | Memory DoS; non-cryptographic IDs guard full history export | By inspection | [TD-232](../../guides/TECHNICAL_DEBT.md#td-232-chatbotapi-accepts-unlimited-client-chosen-session-ids), [TD-233](../../guides/TECHNICAL_DEBT.md#td-233-chatbotapi-session-ids-come-from-a-non-cryptographic-generator) |
| Batch endpoints bypass `generate_response()`: no GPU, RAG, speculative or engine paths; no size limit (§8) | Slow CPU-only batches on GPU hosts; one request can tie up a thread | By inspection | [TD-234](../../guides/TECHNICAL_DEBT.md#td-234-chatbotapi-batch-endpoints-bypass-the-inference-dispatcher), [TD-235](../../guides/TECHNICAL_DEBT.md#td-235-chatbotapi-batch-endpoints-have-no-size-limit) |
| Returned validation errors use HTTP 200 (§4) | Clients must check `success` | By inspection | [TD-236](../../guides/TECHNICAL_DEBT.md#td-236-chatbotapi-returns-http-200-for-validation-errors) |
| `running_` is a non-atomic `bool` written from two threads (§4) | Data race (UB) | By inspection | [TD-237](../../guides/TECHNICAL_DEBT.md#td-237-chatbotapirunning_-is-a-non-atomic-flag-written-from-two-threads) |
| `std::cout`/`std::cerr` in library code (constructor, `start`/`stop`, cleanup) | Breaks the logging convention | By inspection | [TD-239](../../guides/TECHNICAL_DEBT.md#td-239-chatbotapi-logging-and-dead-code-cleanup) |
| Internal exception text returned to clients ("Generation failed: …") (§5) | Minor information leak | By inspection | [TD-238](../../guides/TECHNICAL_DEBT.md#td-238-chatbotapi-returns-internal-exception-text-to-clients) |
| `BatchRequest` unused; `handle_chat()` comment claims per-request config (§3) | Dead code / misleading comment | By inspection | [TD-239](../../guides/TECHNICAL_DEBT.md#td-239-chatbotapi-logging-and-dead-code-cleanup) |
| `/chat/batch` `stats` are hypothetical | Already tracked | — | TD-217 |
| Batched mode ignores `strategy`/`beam_width` | Already tracked | — | TD-214 |
