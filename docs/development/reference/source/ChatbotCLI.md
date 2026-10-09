# `ChatbotCLI` — Source File Reference

- **Files:** [`src/ChatbotCLI.hpp`](../../../../src/ChatbotCLI.hpp), [`src/ChatbotCLI.cpp`](../../../../src/ChatbotCLI.cpp), [`src/ChatbotCLI_main.cpp`](../../../../src/ChatbotCLI_main.cpp)
- **Built into:** the `chatbot` executable (`add_executable(chatbot ChatbotCLI_main.cpp ChatbotCLI.cpp)`; needs cpp-httplib, TD-159)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.1.0` (class), `1.0.0` (main), `@adai-reviewed: 2026-09-13` / `2026-09-11`
- **Tests:** [`tests/chatbotcli_test.cpp`](../../../../tests/chatbotcli_test.cpp) → `chatbotcliTests` (83 tests, ctest `ChatbotCLITests`); [`tests/chatbotcli_improved_test.cpp`](../../../../tests/chatbotcli_improved_test.cpp) (28 tests, including real `/save`/`/load` round-trips against an in-process HTTP server); test plan in [chatbot-cli-tests.md](../../testing/chatbot-cli-tests.md)
- **User guide:** [chatbot-guide.md](../../../operations/guides/chatbot-guide.md)
- **Last traced against the code:** 2026-10-08

> Code-traced reference. The end-of-input, error-reply and `/set` findings were confirmed by
> running the real `chatbot` binary against a stub HTTP server. This page supersedes the older
> `docs/development/guides/internals/chatbot-cli-internals.md` ("ChatbotCLI Context Documentation",
> now deleted). That doc described an earlier architecture in which the CLI loaded the vocab and
> model itself; its still-accurate content is merged here, see [§9](#9-merge-notes).

---

## 1. What this file is

`chatbot` is a terminal chat client for `chatbot_api_server`. It's a **thin HTTP client**: it holds
no model, tokenizer or conversation history. Each message is POSTed to `/chat/session`, and the
conversation lives server-side in `ChatbotAPI`'s session store (see [ChatbotAPI.md](ChatbotAPI.md)),
keyed by a `session_id` the CLI adopts from the first reply.

```text
You: hello ─► POST /chat/session {"message":"hello", ...}  ─► chatbot_api_server
Bot: …     ◄─ {"success":true,"response":"…","session_id":"…"}  (session_id remembered)
/clear     ─► POST /clear-session
/save      ─► POST /chat/session/export ─► write "data" to the local file
/load      ─► read local file ─► POST /chat/session/import ─► adopt returned session_id
/exit      ─► auto /save (if a session exists) ─► quit
```

### Why it matters

It's the reference interactive client and the quickest way for an operator to talk to a freshly
trained model. It's also the client the server's session API (TD-053 export/import, TD-095
session-ID echo) was shaped around.

---

## 2. `ChatbotCLI_main.cpp`

```text
chatbot [server_url] [conversation_save_file]
  defaults: http://localhost:8080   conversation_history.txt
  -h/--help (first argument only) prints usage
```

Positional arguments only. Builds a `ChatbotCLI` and calls `run()`; an exception escaping `run()`
prints "Fatal error" and exits 1.

---

## 3. The class

### State

| Member | Purpose |
|---|---|
| `server_url`, `conversation_save_path` | From the command line |
| `session_id` | Empty until the server returns one; then sent with every request |
| `client` | `std::unique_ptr<httplib::Client>`, created in `initialize()` |
| `max_response_length` (100), `temperature` (1.0), `top_p` (0.9), `top_k` (50), `beam_width` (5), `generation_strategy` (`"nucleus"`) | Generation settings changed by `/set` and sent with every message (but see §5) |

Copying is deleted; moving is defaulted. Test-only getters and setters expose every setting.

### Methods

| Method | Behaviour |
|---|---|
| `initialize()` | Creates the client (timeouts: connect 10 s, read **300 s** because generation is slow, write 30 s), then `GET /health`; true on HTTP 200 |
| `run()` | `initialize()`, prints the welcome banner, then the REPL (§4) |
| `print_welcome()` (static) | Banner and command list: `/help`, `/clear`, `/settings`, `/set`, `/exit`/`/quit`. **Omits `/save`, `/load` and `/stats`**, which exist |
| `print_stats()` (static) | Prints "Stats not available in API mode." |
| `print_settings()` | Prints the six settings |
| `handle_command(cmd)` | Dispatches `/help`, `/clear`, `/stats`, `/settings`, `/set …`, `/save`, `/load`; otherwise "Unknown command" |
| `handle_setting(text)` | `/set <param> <value>`; see §5 |
| `generate_response(input)` | Builds the request JSON, POSTs `/chat/session`, extracts `response` and adopts `session_id` from the reply; non-200 replies become `"Error: <status> - <body>"` |
| `save_conversation()` | Needs a session; `POST /chat/session/export`, checks `"success"`, writes `data` to the save file (overwriting it) |
| `load_conversation()` | Reads the save file, `POST /chat/session/import` (with the current session ID if any), adopts the returned ID, reports `message_count` |

### File-local helpers (`ChatbotCLI.cpp`)

- `escape_json_string()`: escapes quotes, backslashes, and **all** control characters (`\b \f \n
  \r \t`, others as `\u00XX`). That's more complete than the server's version (TD-229).
- `unescape_json_string()` and `parse_json_value(json, key)`: a hand-written reader that finds the
  first `"key"`, then returns either a string value (honouring backslash escapes) or the raw token
  up to `, }` or space. `\uXXXX` isn't decoded. It's adequate for the server's fixed response shapes.

These three have **external linkage** at global scope, so any other translation unit defining the
same global names would collide. `MetricsPushClient.cpp`'s `escape_json_string` is safe only
because it sits in an anonymous namespace.

**Colour macros:** `COLOR_RESET`, `COLOR_USER` (cyan), `COLOR_BOT` (green), `COLOR_SYSTEM`
(yellow) and `COLOR_ERROR` (red) are `#define`d in the header, so every file that includes
`ChatbotCLI.hpp` gets them, along with all of `httplib.h`.

---

## 4. The REPL (`run()`)

Each iteration prints `You: `, reads a line with `std::getline`, trims whitespace, and skips empty
lines. `/exit` and `/quit` auto-save if a session exists, then stop. Other `/…` input goes to
`handle_command()`; anything else goes to `generate_response()`, printed after `Bot: `.

> **End of input never exits (verified).** The loop doesn't check `std::getline`'s result. At EOF
> (Ctrl+D, or stdin from a file or pipe running out) `getline` fails and leaves the line empty,
> the empty-line check `continue`s, and the loop spins forever printing `You: `. Run against a
> stub server with stdin at EOF, it printed **18.6 million prompts (284 MB) in 3 seconds** before
> being killed. In a terminal that pins a CPU core; with output redirected, it fills the disk.
> It also never auto-saves. The old internals doc listed Ctrl+D as an exit path.

There's no `SIGINT` handling either: Ctrl+C kills the process immediately, without the auto-save.

---

## 5. `/set` and generation settings

`handle_setting()` accepts `strategy` (one of `greedy`, `beam`, `sampling`, `top-k`, `nucleus`),
`length`/`max_length`, `temperature`/`temp`, `top_p`/`top-p`, `top_k`/`top-k` and
`beam_width`/`beam-width`. Numbers are parsed with `std::stoi`/`std::stof` inside a `try`
(TD-072, so a typo no longer kills the session). Partial numbers (`10abc` → 10) and out-of-range
values (negative length, `top_p` 5) are still accepted.

> **`/set` has no effect (verified).** `generate_response()` sends every setting in each request
> body (confirmed: after `/set temp 0.2` the next request carried `"temperature":0.2`). But
> `ChatbotAPI` never reads them: every handler uses the server's own defaults (see
> [ChatbotAPI.md](ChatbotAPI.md) §3). The CLI confirms each `/set` ("✅ Temperature set to…") and
> `/settings` reports the new values, but generation never changes. Even if the server honoured
> them, the strategy names don't match: the CLI's `sampling` and `top-k` are not the server's
> `temperature` and `top_k`, so they would fall back to `nucleus`.

For reference, here is what the server's strategies mean. They're set server-side with
`--strategy` or `STRATEGY`, see [ChatbotApiServerArgs.md](ChatbotApiServerArgs.md):

| Server strategy | Behaviour | Uses |
|---|---|---|
| `greedy` | Always the most likely token; deterministic, can repeat itself | — |
| `beam` | Keeps the best N partial sequences | beam width |
| `temperature` | Samples from the temperature-scaled distribution; low = focused, high = creative | `temperature` |
| `top_k` | Samples among the k most likely tokens | `top_k`, `temperature` |
| `nucleus` (default) | Samples from the smallest set whose probability reaches p | `top_p`, `temperature` |

---

## 6. Server responses and errors

`generate_response()` treats any **HTTP 200** reply as success and prints its `response` field.
`ChatbotAPI` returns HTTP 200 with `{"success":false,"error":…}` for errors it returns rather than
throws (TD-236), and the CLI doesn't check `success`. So such an error is shown as an **empty**
`Bot:` line. Verified against a stub returning `{"success":false,"error":"Missing 'message'
field…"}`: two messages produced two empty `Bot:` lines. Thrown server errors (HTTP 500) are
printed as `Error: 500 - <raw JSON body>`.

`/clear` and `generate_response()` put `session_id` into the request JSON **without escaping**,
unlike `/save` and `/load`, which escape it. Server-generated IDs are hex, so this is safe today,
but any ID containing `"` or `\` (for example, one restored from an edited save file) would break
the request.

If the server restarts, the CLI keeps sending its old `session_id`. `ChatbotAPI` silently creates a
fresh, empty session under that ID (TD-232), so the conversation history disappears with no
message.

`/exit` auto-save writes to the same file every time, so starting a new conversation and exiting
**silently overwrites** a previously saved one. Writes aren't atomic (truncate, then write).

---

## 7. Tests

- `chatbotcliTests` (83): command parsing, helper functions, setting validation.
- `chatbotcli_improved_test.cpp` (28): construction, getters and setters, command handling, strategy
  validation, colour codes, and real `/save`/`/load` against an in-process server.

Not covered: EOF handling, `success:false` replies, the escaping difference, and anything showing
that settings reach generation (they don't).

---

## 8. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Verified | Tracked as |
|---|---|---|---|
| EOF (Ctrl+D, piped input) loops forever printing prompts (§4) | CPU pinned; ~95 MB/s of output when redirected; no auto-save | Yes | [TD-253](../../guides/TECHNICAL_DEBT.md#td-253-chatbot-spins-forever-at-end-of-input) |
| `/set` settings are sent but ignored by the server; strategy names don't match the server's (§5) | Users believe settings changed when they didn't | Yes (sent); server side by inspection | [TD-254](../../guides/TECHNICAL_DEBT.md#td-254-chatbots-set-generation-settings-have-no-effect) |
| `success:false` replies shown as empty bot responses (§6) | Server errors hidden | Yes | [TD-255](../../guides/TECHNICAL_DEBT.md#td-255-chatbot-shows-server-error-replies-as-empty-responses) |
| `print_welcome()` omits `/save`, `/load`, `/stats` (§3) | Undiscoverable commands | By inspection | [TD-256](../../guides/TECHNICAL_DEBT.md#td-256-chatbots-help-banner-omits-save-load-and-stats) |
| `session_id` unescaped in `/clear` and chat requests (§6) | Breaks on IDs with quotes or backslashes | By inspection | [TD-257](../../guides/TECHNICAL_DEBT.md#td-257-chatbot-sends-session_id-unescaped-in-two-requests) |
| No Ctrl+C handling; auto-save silently overwrites the previous save; non-atomic write (§4, §6) | Lost conversations | By inspection | [TD-258](../../guides/TECHNICAL_DEBT.md#td-258-chatbot-can-lose-or-overwrite-saved-conversations) |
| Numeric `/set` values accept partial and out-of-range input (§5) | Moot while settings are ignored | By inspection | [TD-259](../../guides/TECHNICAL_DEBT.md#td-259-chatbot-code-hygiene-unchecked-set-values-global-helpers-heavy-header) |
| JSON helpers have external linkage; header exports colour macros and all of httplib (§3) | ODR/collision risk; compile-time cost | By inspection | [TD-259](../../guides/TECHNICAL_DEBT.md#td-259-chatbot-code-hygiene-unchecked-set-values-global-helpers-heavy-header) |
| Server restart silently loses history under the reused session ID (§6) | Confusing | By inspection (TD-232 server side) | [TD-258](../../guides/TECHNICAL_DEBT.md#td-258-chatbot-can-lose-or-overwrite-saved-conversations) |

---

## 9. Merge notes

This file absorbed `docs/development/guides/internals/chatbot-cli-internals.md` (1,314 lines).

**Dropped as obsolete:** that doc described the CLI before it became an HTTP client. The local
`BPETokenizer`, `EncoderDecoderModel` and `ConversationContext` members, the
`[vocab_file] [model_file] [save_file]` command line, the hardcoded 512/8/2048/6/6/1024
architecture, local `save_to_file`/`load_from_file`, the `/system` command (removed), real `/stats`
output, and the "Integration with Other Components" and file-format sections no longer apply.

**Corrected:**
- "Ctrl+D exits": it loops forever (§4).
- `/set` settings and strategies "take effect": they're ignored (§5).
- The welcome banner shown included `/save`, `/load`, `/stats` and `/system`; the real banner
  lists none of them (§3).
- "23 tests": there are 83 + 28.

**Kept and merged:** colour codes, the command reference (updated), `/set` parameter names and
aliases, generation-strategy descriptions (now framed as server-side, with the server's names),
the `string_view` parsing and move-only design notes, and the file layout
(`ChatbotCLI.hpp` / `ChatbotCLI.cpp` / `ChatbotCLI_main.cpp`).
