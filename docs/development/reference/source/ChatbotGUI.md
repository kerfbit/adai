# `ChatbotGUI` — Source File Reference

- **Files:** [`src/ChatbotGUI.hpp`](../../../../src/ChatbotGUI.hpp), [`src/ChatbotGUI.cpp`](../../../../src/ChatbotGUI.cpp), [`src/ChatbotGuiLogic.hpp`](../../../../src/ChatbotGuiLogic.hpp), [`src/ChatbotGUI_main.cpp`](../../../../src/ChatbotGUI_main.cpp), [`src/ChatbotGUI_wrapper.cpp`](../../../../src/ChatbotGUI_wrapper.cpp)
- **Built into:** `chatbot_gui_binary` (main + class + `Config.cpp`; links `adai_models`, `adai_nlp`, `adai_attention`, `adai_core`, Qt6 or Qt5 Widgets) and `chatbot_gui` (the launcher wrapper). Only when `BUILD_GUI=ON` (the default) and Qt is found; CMake prefers **Qt6**, falling back to Qt5
- **Status tags:** class `beta` 0.7.0/0.8.0 (capped by TD-037, no Qt Test infrastructure); `ChatbotGuiLogic.hpp` `beta` 0.1.0; main and wrapper `stable` 1.0.0
- **Tests:** [`tests/chatbotguilogic_test.cpp`](../../../../tests/chatbotguilogic_test.cpp) → `chatbotguilogicTests` (2 tests; strategy mapping only). The widget class is untested (TD-037)
- **User docs:** [chatbot-gui-guide.md](../../../operations/guides/chatbot-gui-guide.md), [GUI_QUICK_REFERENCE.md](../../../operations/guides/quick-reference/GUI_QUICK_REFERENCE.md), [CHATBOT_GUI_TROUBLESHOOTING.md](../../../operations/guides/troubleshooting/CHATBOT_GUI_TROUBLESHOOTING.md), [CPP_WRAPPER_SOLUTION.md](../../../operations/guides/troubleshooting/CPP_WRAPPER_SOLUTION.md)
- **Last traced against the code:** 2026-10-09

> Code-traced reference. Findings come from reading the GUI and the `EncoderDecoderModel`,
> `ConversationContext` and `IncrementalTrainer` code it depends on. The GUI itself wasn't run
> (no display in this environment), so each finding below cites the code that proves it.

---

## 1. What this file is

`chatbot_gui` is a **standalone** Qt desktop chat app. Unlike the `chatbot` CLI (an HTTP client,
see [ChatbotCLI.md](ChatbotCLI.md)), it loads the vocabulary and model **in-process** and generates
locally, with no server involved. It has a chat pane with an input field and
Clear/Save/Load buttons, and a settings panel (strategy, temperature, top-p, top-k, max length,
beam width).

```text
chatbot_gui [vocab_file] [model_file]          (defaults: vocab.txt, chatbot_model.bin)
   └─ wrapper: clean snap env, set LD_LIBRARY_PATH / Qt plugin path ─► exec chatbot_gui_binary
        └─ QApplication ─► ChatbotGUI(vocab, model)
              ├─ initializeChatbot(): tokenizer ─► config.chatbot.conf architecture ─► model ─► weights? ─► context
              ├─ setupUI() + applyStylesheet()
              └─ Send ─► generateResponse(): context ─► model->generate_response_with_strategy(...)
```

### Why it matters

It's the only way to chat with a model without running `chatbot_api_server`, which is useful for
quickly trying a checkpoint on a desktop. Its value depends entirely on actually loading that
checkpoint, which it currently can't do (§4).

---

## 2. `ChatbotGUI_wrapper.cpp` (the `chatbot_gui` executable)

Works around snap-packaged library conflicts (see CPP_WRAPPER_SOLUTION.md):

1. finds its own directory via `/proc/self/exe` (TD-103 fix; argv[0] isn't reliable for PATH
   launches);
2. unsets `GTK_PATH`, `SNAP`, `SNAP_COMMON`, `SNAP_DATA`;
3. sets `LD_LIBRARY_PATH` to `/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu`, appending the old
   value only if it contains no `/snap/`;
4. sets `QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins` and empties
   `GTK_MODULES`;
5. `execvp`s `<dir>/chatbot_gui_binary` with the same arguments.

Gotchas:

- **The plugin path is always Qt5**, even though CMake builds against **Qt6 when it's available**.
  A Qt6 build launched through the wrapper is pointed at Qt5 platform plugins.
- **x86_64 Debian/Ubuntu paths are hardcoded**, so other architectures or distributions get wrong
  paths.
- Library order is forced: system paths come first, so a user's own `LD_LIBRARY_PATH` entries
  can't override system libraries, and any `LD_LIBRARY_PATH` containing `/snap/` is dropped
  entirely.

---

## 3. `ChatbotGUI_main.cpp`

Creates the `QApplication`, sets the app metadata, takes `argv[1]`/`argv[2]` as vocab/model paths,
handles `--help`/`-h`, then shows the window and runs the event loop. The `QApplication` is created
**before** the help check, so even `chatbot_gui --help` needs a working display connection.

---

## 4. `ChatbotGUI` class

### Construction and `initializeChatbot()`

The constructor sets defaults (strategy `nucleus`, max length 100, temperature 1.0, top-p 0.9,
top-k 50, beam width 5), calls `initializeChatbot()`, builds the UI, and posts a welcome
message. `initializeChatbot()`:

1. loads the tokenizer from the vocab path;
2. reads the architecture from **`config.chatbot.conf`** (discovered from the working directory,
   then `/etc/adai`, then legacy paths), not from MNS and not from the checkpoint;
3. constructs `EncoderDecoderModel` with that architecture and hands it the tokenizer
   (`set_tokenizer()` takes ownership, so the `tokenizer` member is null afterwards, by design);
4. **loads weights only if `std::ifstream(model_path).good()`**;
5. creates a `ConversationContext(20 messages, 480 tokens)`.

> **The GUI can't load a trained checkpoint.** `EncoderDecoderModel::save_model(path)` writes only
> sidecar files, `path.config`, `.vocab`, `.encoder`, `.decoder` and `.lm_head`, and **never a file
> at `path` itself**. Step 4 checks for a file at exactly `path`, which never exists for a real
> checkpoint, so `load_model()` is never called and the GUI **silently** runs on random weights
> (no message at all). `IncrementalTrainer` has a comment describing this exact trap ("checking
> for that bare path here would reject every legitimately-saved checkpoint unconditionally") and
> checks the sidecars instead. Passing a path that *does* exist, such as `model.config`, doesn't
> help: `load_model()` then looks for `model.config.config` and throws (see the next note).

> **Architecture mismatch makes it unusable, with misleading messages.** The architecture comes
> only from the local config file. The server and trainer take it from MNS (CLAUDE.md, "Model
> architecture is MNS-authoritative"), so a checkpoint trained under an MNS architecture that
> differs from the local file fails `load_model()`. Any exception in steps 1–5 shows a warning
> saying "Using default/random initialization", returns `false` (so the constructor shows a
> *second*, critical dialog), and leaves `context` null. Every message then answers
> "[Error: Chatbot not initialized]". The warning's "random initialization" claim is wrong: the
> GUI is unusable.

The 480-token context limit is hardcoded with a comment about a 512-token model, regardless of
the configured `MAX_SEQ_LENGTH`.

### UI construction

`setupUI()` builds a 70/30 splitter. `createChatArea()` holds the read-only `QTextEdit` and the
Clear Chat / Save / Load buttons. `createInputArea()` has the line edit, a second Clear button (a
local, after a past fix where it overwrote the `clearButton` member) and Send; Enter also sends.
`createSettingsPanel()` has a strategy combo (`Nucleus (Top-p)`, `Top-k Sampling`, `Greedy`,
`Beam Search`, `Sampling`), temperature 0.1–2.0, top-p 0.1–1.0, top-k 1–200, max length 10–500,
and beam width 1–10. `applyStylesheet()` sets the look. `addMessage()` appends HTML-escaped,
timestamped, colour-coded bubbles and scrolls down.

### Slots

| Slot | Behaviour |
|---|---|
| `onSendMessage()` | Shows the user message, disables input, calls `generateResponse()` **on the GUI thread**, shows the reply, re-enables input |
| `onClearConversation()` | Confirms, then clears the display and `context` |
| `onSaveConversation()` / `onLoadConversation()` | File dialog, then `ConversationContext::save_to_file()` / `load_from_file()`. Both use `serialize()`/`deserialize()`, the **same format** as the CLI's `/save`/`/load` (server export/import), so files are interchangeable. Loading doesn't redisplay the loaded messages |
| `onStrategyChanged(index)` | `chatbot_gui::generation_strategy_for_index()` (`ChatbotGuiLogic.hpp`): 0 nucleus, 1 top-k, 2 greedy, 3 beam, 4 sampling (out of range falls back to nucleus) |
| `onTemperatureChanged`, `onTopPChanged`, `onTopKChanged`, `onMaxLengthChanged`, `onBeamWidthChanged` | Store the value |

> **The UI freezes during generation.** `generateResponse()` runs synchronously in the click
> handler. `processEvents()` is called once before it, so the "Generating..." label paints, but
> then the event loop is blocked for the whole generation (seconds to minutes on CPU): the window
> can't repaint, scroll or close, and the desktop may flag it as "not responding".

### `generateResponse(user_input)`

Appends the user message, formats the context with
`ConversationContext::format_with_special_tokens()` (`<bos> [USER] … <sep> …`), calls
`model->generate_response_with_strategy(context, max_length, strategy, temperature, top_k, top_p,
beam_width)`, and appends the reply. Exceptions become `[Error: …]` replies.

> **Every strategy runs beam search unless Beam Width is 1.** In
> `EncoderDecoderModel::generate_response_with_strategy()`, the beam branch is taken when
> `strategy == "beam" || num_beams > 1`. It returns before the greedy/sampling/top-k/nucleus
> branches. The GUI always passes the Beam Width spin box (default **5**), so choosing Nucleus,
> Top-k, Greedy or Sampling still gets beam search. A comment in that function says the explicit
> strategies "are unaffected", but the code order contradicts it. The same root cause hits
> `RLHFTrainer`, which omits `num_beams` and so gets the header default of **4**: its sampling
> configuration is overridden by beam search too.

> **The GUI and the server format multi-turn context differently.** The GUI uses
> `format_with_special_tokens()` (`<bos> [USER] … <sep>` tags, which the tokenizer reads as plain
> characters, not special tokens). `ChatbotAPI` sessions use `format_for_model()` (`User: …` /
> `Assistant: …`). The same model gets differently shaped prompts depending on the client.

GPU acceleration is never initialized: there's no `GPU_ENABLED` handling, so the GUI always runs
on CPU.

---

## 5. `ChatbotGuiLogic.hpp`

`chatbot_gui::generation_strategy_for_index(int)`: the combo-box-index-to-strategy mapping,
extracted from the widget (TD-037's "logic extraction" step) so it can be tested without Qt.
`chatbotguilogicTests` covers it. Its names (`top-k`, `sampling`) match what
`generate_response_with_strategy()` accepts (`top-k` is normalized to `topk`).

---

## 6. Known gaps and gotchas (summary)

Each item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Weights load only if a file exists at the bare model path, which `save_model()` never writes (§4) | GUI never loads a trained checkpoint; silently random | `save_model()`'s file list; `IncrementalTrainer`'s own note | [TD-260](../../guides/TECHNICAL_DEBT.md#td-260-chatbot_gui-never-loads-a-trained-checkpoint) |
| `num_beams > 1` forces beam search for every strategy; GUI default 5; `RLHFTrainer` gets default 4 (§4) | Strategy choice ignored; RLHF sampling replaced by beam search | Code order in `generate_response_with_strategy()` | [TD-261](../../guides/TECHNICAL_DEBT.md#td-261-generate_response_with_strategy-forces-beam-search-whenever-num_beams--1) |
| Architecture from local config only, not MNS; init failure leaves the GUI unusable with two contradictory dialogs (§4) | Mismatched checkpoints can't load; confusing errors | By inspection | [TD-262](../../guides/TECHNICAL_DEBT.md#td-262-chatbot_gui-ignores-mns-architecture-and-becomes-unusable-on-init-failure) |
| Generation runs on the GUI thread (§4) | Window freezes for the whole generation | By inspection | [TD-263](../../guides/TECHNICAL_DEBT.md#td-263-chatbot_gui-freezes-while-generating) |
| Wrapper forces Qt5 plugin path though CMake prefers Qt6; hardcoded x86_64 paths; overrides `LD_LIBRARY_PATH` order (§2) | Qt6 builds pointed at Qt5 plugins; non-x86_64 broken | By inspection | [TD-264](../../guides/TECHNICAL_DEBT.md#td-264-chatbot_gui-launcher-hardcodes-qt5-and-x86_64-paths) |
| Different multi-turn prompt format from the server (§4) | Inconsistent model behaviour across clients | By inspection | [TD-265](../../guides/TECHNICAL_DEBT.md#td-265-chatbot_gui-and-chatbot_api_server-format-conversation-context-differently) |
| Hardcoded 480-token context; load doesn't redisplay history; `--help` needs a display; no GPU (§3–4) | Minor | By inspection | [TD-266](../../guides/TECHNICAL_DEBT.md#td-266-chatbot_gui-minor-gaps) |
| Widget class untested | Already TD-037 | — | TD-037 |
