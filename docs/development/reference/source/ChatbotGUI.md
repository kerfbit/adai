# `ChatbotGUI` — Source File Reference

- **Files:** [`src/ChatbotGUI.hpp`](../../../../src/ChatbotGUI.hpp) (181 lines), [`src/ChatbotGUI.cpp`](../../../../src/ChatbotGUI.cpp) (608), [`src/ChatbotGuiLogic.hpp`](../../../../src/ChatbotGuiLogic.hpp) (137), [`src/ChatbotGUI_main.cpp`](../../../../src/ChatbotGUI_main.cpp) (57), [`src/ChatbotGUI_wrapper.cpp`](../../../../src/ChatbotGUI_wrapper.cpp) (106), [`src/ChatbotGuiLaunchEnv.{hpp,cpp}`](../../../../src/ChatbotGuiLaunchEnv.hpp) (51 + 88)
- **Built into:** `chatbot_gui_binary` (main, the class, [`ChatbotClient`](ChatbotClient.md), `ConversationFormat.cpp`, `GenerationSettings.cpp`; links only Qt Widgets and threads, no model library) and `chatbot_gui` (the launcher wrapper, with `ChatbotGuiLaunchEnv.cpp`). Only when `BUILD_GUI=ON` (the default) and Qt is found; CMake prefers **Qt6**, falling back to Qt5
- **Status tags:** `ChatbotGUI.{hpp,cpp}` and `ChatbotGuiLogic.hpp` `stable` 1.0.0 (promoted November 10, 2026), `ChatbotGuiLaunchEnv.{hpp,cpp}` `beta` 0.1.0; `ChatbotGUI_main.cpp` `stable` 2.0.0, `ChatbotGUI_wrapper.cpp` `stable` 1.1.0
- **Tests:** [`tests/chatbot_gui_widget_test.cpp`](../../../../tests/chatbot_gui_widget_test.cpp) → `chatbotGuiWidgetTests` (Qt Test, 22 tests, the real window against an in-process server); [`tests/chatbotguilogic_test.cpp`](../../../../tests/chatbotguilogic_test.cpp) → `chatbotguilogicTests` (11); `chatbotguilaunchenvTests` (7); smoke tests `SmokeTests_chatbot_gui_binary_main` and `SmokeTests_chatbot_gui_wrapper` (`tests/smoke/`)
- **User docs:** [chatbot-gui-guide.md](../../../operations/guides/chatbot-gui-guide.md), [GUI_QUICK_REFERENCE.md](../../../operations/guides/quick-reference/GUI_QUICK_REFERENCE.md), [CHATBOT_GUI_TROUBLESHOOTING.md](../../../operations/guides/troubleshooting/CHATBOT_GUI_TROUBLESHOOTING.md), [CPP_WRAPPER_SOLUTION.md](../../../operations/guides/troubleshooting/CPP_WRAPPER_SOLUTION.md) (historical)
- **Last traced against the code:** 2026-11-10

> Code-traced reference. Behaviour marked **tested** is covered by `chatbotGuiWidgetTests`, which drives the real
> window offscreen; the server-facing behaviour was also checked against a real `chatbot_api_server` (TD-260, TD-266).

---

## 1. What this file is

`chatbot_gui` is a Qt desktop client of `chatbot_api_server`, like the `chatbot` CLI ([ChatbotCLI.md](ChatbotCLI.md)):
the server holds the model, the tokenizer and the conversation, and the GUI sends messages to it over HTTP through
[`ChatbotClient`](ChatbotClient.md), the session class the CLI uses too. The window has a chat pane with an input
field and Clear Chat / Save / Load buttons, and a settings panel (strategy, temperature, top-p, top-k, max length,
beam width).

```text
chatbot_gui [server_url]                     (default http://localhost:8080)
   └─ wrapper: remove snap entries from the environment ─► exec chatbot_gui_binary
        └─ main: --help before QApplication ─► ChatbotGUI(server_url)
              ├─ setupUI() + applyStylesheet() + welcome message
              ├─ client.open(), then connect (GET /health) on the worker thread ─► connection notice
              └─ Send / Clear / Save / Load ─► runRequest(): the HTTP call on the worker, the result on the GUI thread
```

### Why it matters

It's the desktop way to chat with the served model. Because it is a client, it always talks to the same trained
model, architecture and settings as every other client; it used to load its own copy of the model and couldn't
load a real checkpoint (TD-260).

---

## 2. `ChatbotGUI_wrapper.cpp` (the `chatbot_gui` executable)

Works around snap-packaged library conflicts (see CPP_WRAPPER_SOLUTION.md for the original problem):

1. finds its own directory via `/proc/self/exe` (TD-103; `argv[0]` isn't reliable for `PATH` launches);
2. applies `adai::gui_launch::launch_environment()` (`ChatbotGuiLaunchEnv.{hpp,cpp}`, no Qt dependency, TD-264):
   - always unsets `GTK_PATH`, `SNAP`, `SNAP_COMMON`, `SNAP_DATA` and sets `GTK_MODULES` empty;
   - removes only `/snap/` entries from `LD_LIBRARY_PATH` and `QT_PLUGIN_PATH`, keeping the rest in order (a list
     left empty is unset); adds no system directory; unsets `GIO_MODULE_DIR` only when it points into a snap;
   - in a snap environment only, and only if the user hasn't set it outside a snap, sets
     `QT_QPA_PLATFORM_PLUGIN_PATH` to `ADAI_QT_PLATFORM_PLUGIN_DIR`, the platform plugin directory of the Qt the GUI
     is built with (from CMake's `Qt<5|6>::QOffscreenIntegrationPlugin` target, then `QXcbIntegrationPlugin`,
     `QMinimalIntegrationPlugin`), if it still exists;
3. `execvp`s `<dir>/chatbot_gui_binary` with the same arguments.

`scripts/run_chatbot_gui.sh` and `scripts/chatbot_gui_fixed.sh` leave the environment to the wrapper.

---

## 3. `ChatbotGUI_main.cpp`

Handles `--help`/`-h` **before** creating the `QApplication`, so help works without a display. More than one argument
(the old `[vocab_file] [model_file]` form) exits 1 with an explanation. Otherwise it takes `argv[1]` as the server
URL (default `http://localhost:8080`), creates the `QApplication`, shows a `ChatbotGUI` and runs the event loop.

---

## 4. `ChatbotGUI` class

### Construction

The constructor builds the UI (`setupUI()`, `applyStylesheet()`), shows a two-line welcome message, then calls
`client.open()`. An invalid URL (one `httplib` rejects, such as `ftp://…`) is reported in the chat pane and leaves the
controls disabled with Send reading "No server" (**tested**); the GUI must be restarted with a valid URL. Otherwise it
connects on the worker thread (`GET /health`) and shows `chatbot_gui::connection_notice()`: connected, unreachable
(with how to pass a URL), a server running an untrained model (`/health` 503, TD-246), or something else answering
(**tested**). An unreachable server doesn't disable sending: messages go through once the server is up.

### One request at a time, off the GUI thread (TD-263)

`runRequest(busyText, work)` joins the previous worker (which has finished: its result already re-enabled the
controls), calls `setBusy(true, busyText)`, and runs `work` on a `std::thread`. `work` makes the HTTP call and returns
a function, which is posted back to the GUI thread with `QMetaObject::invokeMethod(…, Qt::QueuedConnection)`; it runs
`setBusy(false)` and then shows the result. An exception in `work` becomes a System message.

`setBusy()` disables the input field, Send (whose text shows the busy text, e.g. "Generating..."), both Clear
buttons, Save, Load and the whole settings panel (**tested**). So only the worker touches `client` while a request
runs, and the settings never change under it.

The destructor calls `client.cancel()` and joins the worker, so closing the window during a request returns at once
instead of waiting for the server (**tested**: under 5 s while the server holds the request for 30 s). The posted
result is dropped with the destroyed window.

### UI

`setupUI()` builds a 70/30 splitter: the chat area (read-only `QTextEdit`, Clear Chat / Save Conversation / Load
Conversation), the input area (line edit, a second Clear, Send; Enter also sends) and the settings panel. Every
control the tests use has an object name (`chatDisplay`, `inputField`, `sendButton`, `clearChatButton`,
`clearInputButton`, `saveButton`, `loadButton`, `settingsPanel`, `strategyCombo`, `temperatureSpin`, `topPSpin`,
`topKSpin`, `maxLengthSpin`, `beamWidthSpin`).

Settings (TD-254, TD-260): the strategy combo starts with "Server default", then Nucleus, Top-k, Greedy, Beam Search
and Sampling (the server's names `nucleus`, `top_k`, `greedy`, `beam`, `temperature`). Each spin box covers the
server's range (`adai::GenerationLimits`) plus one step below it, shown as "server default". A setting is sent with the
next message only when moved off "server default" (**tested**); the server decides when beam width applies.

`addMessage(sender, message, isUser)` appends a timestamped, colour-coded HTML bubble (blue for the user, green for
the bot, orange for System) and scrolls down. Both `sender` and `message` are HTML-escaped, and the message span has
`white-space: pre-wrap`, so line breaks and runs of spaces show as sent (TD-567, **tested**); a loaded conversation's
role names are shown as text, not markup (**tested**).

### Slots

| Slot | Behaviour |
|---|---|
| `onSendMessage()` | Ignores blank input. Shows the message, clears the input, then sends it on the worker (`ChatbotClient::send_message()`, `POST /chat/session`). Shows the reply as "Bot", or the server's error as a System message (`Error: <status> - <error> (request id N)`, TD-255), after a note when the server had lost the session and the message started a new one (TD-232). The first reply's session id is kept for later messages (**tested**) |
| `onClearConversation()` | Asks first (No changes nothing). Clears the server's session (`POST /clear-session`), then the pane (**tested**); a refusal is shown as a System message |
| `onSaveConversation()` | Before any message: "Nothing to save yet", no dialog. Otherwise a file dialog, then on the worker `export_session()` and an atomic write of the server's copy (`chatbot_client::write_file_atomically()`, TD-258) (**tested**); a failure is shown in an error dialog |
| `onLoadConversation()` | A file dialog; an unreadable or empty file is an error dialog. Otherwise, on the worker, `import_session()` and then `export_session()`, and the pane shows the server's copy of the conversation, which it continues from (TD-266), with the server's message count (**tested**) |
| `onStrategyChanged()` and the five setting slots | `chatbot_gui::apply_strategy()`, `apply_int_setting()`, `apply_real_setting()`: record the value in the client's overrides, or reset it at "server default" |

---

## 5. `ChatbotGuiLogic.hpp`

Qt-free helpers, tested by `chatbotguilogicTests` (TD-037's logic extraction):

- `generation_strategy_for_index()`: the combo index to the server's strategy name ("" for "Server default" or an
  index out of range).
- `apply_strategy()`, `apply_int_setting()`, `apply_real_setting()`: a control's value into `GenerationOverrides`
  (sent) or reset (not sent) at or below its "server default" value.
- `connection_notice()`: the chat-pane text for a `ChatbotClient::Health`.
- `session_lost_notice()`: the note shown when the server had forgotten the session.
- `conversation_messages()`: exported conversation data (`adai::conversation_format`, `src/ConversationFormat.{hpp,cpp}`)
  as chat-pane messages, in order: the system message as "System", then "user" as "You", "assistant" as "Bot",
  "system" (a summary) as "System", any other role by its own name.

---

## 6. Known gaps and gotchas

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| No Stop control: a request runs until the server answers. `chatbot_api_server` can't stop a generation, so a Stop button would only stop waiting while the server still adds the turn | Long generations can only be waited out (or the window closed) | By design of the server API | [TD-263](../../guides/TECHNICAL_DEBT.md#td-263-chatbot_gui-freezes-while-generating)'s remaining item |
| An invalid server URL at startup leaves the window unusable until restarted with a valid one | The URL can't be changed in the window | Tested | — (by design: the URL is the only argument) |

Fixed, for reference:

| Item | Fixed | Tracked as |
|---|---|---|
| The GUI ran its own model and never loaded a trained checkpoint | November 4, 2026: a client of `chatbot_api_server` | [TD-260](../../guides/TECHNICAL_DEBT.md#td-260-chatbot_gui-never-loads-a-trained-checkpoint) |
| `num_beams > 1` forced beam search for every strategy | October 25, 2026 | [TD-261](../../guides/TECHNICAL_DEBT.md#td-261-generate_response_with_strategy-forces-beam-search-whenever-num_beams--1) |
| Local-config architecture; unusable after an init failure | Moot after TD-260 | [TD-262](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-262-chatbot_gui-ignores-mns-architecture-and-becomes-unusable-on-init-failure) |
| Generation froze the window | November 4, 2026: the worker thread | [TD-263](../../guides/TECHNICAL_DEBT.md#td-263-chatbot_gui-freezes-while-generating) |
| The wrapper forced Qt5 x86_64 paths | November 4, 2026 | [TD-264](../../guides/TECHNICAL_DEBT.md#td-264-chatbot_gui-launcher-hardcodes-qt5-and-x86_64-paths) |
| A prompt format different from the server's | Moot after TD-260 | [TD-265](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-265-chatbot_gui-and-chatbot_api_server-format-conversation-context-differently) |
| `--help` needed a display; a loaded conversation wasn't shown | November 4, 2026 | [TD-266](../../guides/TECHNICAL_DEBT.md#td-266-chatbot_gui-minor-gaps) |
| The widget class was untested | November 10, 2026: Qt Test suite | [TD-037](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-037-no-qt-test-infrastructure-for-gui-classes) |
| Multi-line messages were shown on one line; role names were inserted as HTML | November 10, 2026 | [TD-567](../../guides/TECHNICAL_DEBT.md#td-567-chatbot_gui-shows-multi-line-messages-on-one-line) |

---

## 7. History

- **October 9, 2026:** first traced. The GUI was a standalone app that loaded a vocabulary and model in-process,
  read the architecture from the local `config.chatbot.conf`, generated on the GUI thread, and built prompts in its
  own format; filed TD-260–TD-266.
- **October 25, 2026 (TD-261):** Beam Width was passed only for beam search, so the other strategies ran as chosen.
- **November 4, 2026 (TD-260, TD-263, TD-264, TD-266):** owner decision, the GUI became a client of
  `chatbot_api_server` through `ChatbotClient`; requests moved to a worker thread; the wrapper stopped forcing
  x86_64 Qt5 paths and now removes only what a snap adds (the old plugin path pointed one level above `platforms/` and
  never worked); `--help` moved before `QApplication`; a loaded conversation is shown as the server holds it. Checked by
  driving the real class headless against a real `chatbot_api_server`.
- **November 10, 2026 (TD-037):** `chatbotGuiWidgetTests` (Qt Test, offscreen) and object names on the controls.
- **November 10, 2026 (TD-567):** `addMessage()` keeps line breaks and runs of spaces (`white-space: pre-wrap`) and
  escapes the sender. Before, a reply of `line one\nline two` showed as `line one line two`, the welcome message as one
  line, and a loaded role such as `<b>tool</b>` was rendered as bold markup. This document was re-traced against the
  code at the same time, replacing its pre-TD-260 sections and addenda.
- **November 10, 2026:** `ChatbotGUI.{hpp,cpp}` and `ChatbotGuiLogic.hpp` promoted to `stable` 1.0.0 after a review
  against the file-status standard: widget and logic tests, no open correctness gap (TD-263's Stop control is a
  missing feature the server API can't support), user and reference docs current, shipped as `chatbot_gui`.
