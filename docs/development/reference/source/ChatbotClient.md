# `ChatbotClient` — Source File Reference

- **Files:** [`src/ChatbotClient.hpp`](../../../../src/ChatbotClient.hpp), [`src/ChatbotClient.cpp`](../../../../src/ChatbotClient.cpp)
- **Built into:** `chatbot` (through `ChatbotCLI`) and `chatbot_gui_binary`; needs cpp-httplib and the vendored nlohmann/json
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.1.0`, `@adai-reviewed: 2026-11-04`
- **Tests:** [`tests/chatbotclient_test.cpp`](../../../../tests/chatbotclient_test.cpp) → `chatbotclientTests` (11 tests against a stub server); also covered through `ChatbotCLI` by `chatbotcliImprovedTests`
- **Added:** November 4, 2026, with [TD-260](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-260-chatbot_gui-never-loads-a-trained-checkpoint)

---

## 1. What this file is

The HTTP session with `chatbot_api_server` that both clients share. It was taken out of `ChatbotCLI` when
`chatbot_gui` stopped running its own copy of the model (TD-260), so the CLI and the GUI send the same requests
and read replies the same way. It prints nothing: each call returns what happened, and the caller decides what to
tell the user (the CLI prints coloured lines, the GUI adds chat-pane messages and dialogs).

```text
connect()         GET  /health                 → Health {reachable, status, model_loaded, error}
send_message()    POST /chat/session           → Reply {ok, text, session_lost}
clear_session()   POST /clear-session          → Cleared | NoSession | SessionLost | Failed
export_session()  POST /chat/session/export    → Export {status, data, error}
import_session()  POST /chat/session/import    → Import {ok, error, message_count}
```

## 2. Session

The server issues session ids (TD-232). The client sends none with its first message and adopts the one in the
reply. When the server answers 404 (the session expired, was evicted, or the server restarted), `send_message()`
and `import_session()` forget the id and send again in a new session; `send_message()` reports this as
`session_lost` so the caller can say that earlier messages are gone. `clear_session()` and `export_session()`
report `SessionLost` and forget the id. `forget_session()` drops it by hand.

## 3. Settings

`GenerationOverrides` holds strategy, max length, temperature, top-p, top-k and beam width, and which of them were
changed. Only changed ones go into the request (TD-254); the server applies its own defaults to the rest and checks
the sent ones against `adai::GenerationLimits`. `reset(key)`/`reset_all()` go back to the server's default. Floats
are sent in their shortest decimal form (0.7, not 0.699999988079071).

## 4. Replies and errors

Bodies are built with `nlohmann::ordered_json`, invalid UTF-8 replaced by U+FFFD so the request stays valid JSON,
and replies are read field by field at the top level (TD-259). A failed turn's `text` is
`Error: <status> - <error> (request id N)` (TD-255); an unreachable server gives `Error: Connection failed`, and a
call before `connect()`/`open()` gives `Error: Client not initialized`.

## 5. Threads and cancelling

One request at a time. `cancel()` is the only call safe from another thread: it stops a request in flight
(`httplib::Client::stop()`), which then fails as unreachable. `chatbot_gui` calls it when its window closes, so it
doesn't wait up to the 300 s read timeout. `open()` creates the connection without contacting the server; the GUI
calls it on its own thread before starting a worker, so `cancel()` never reads the connection while it's created.

Timeouts: 10 s to connect, 300 s to read (generation can be slow), 30 s to write.

## 6. `chatbot_client::write_file_atomically()`

Writes to a temporary file beside the target and renames it over it, so a failed or interrupted save never leaves a
partial file (TD-258). Used by the CLI's `/save` and the GUI's Save Conversation.
