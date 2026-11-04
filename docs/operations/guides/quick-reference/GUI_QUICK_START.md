# Chatbot GUI - Quick Start

`chatbot_gui` is a desktop client for `chatbot_api_server`, like the `chatbot` CLI: your messages go to the
model the server serves, and the server keeps the conversation. Since November 4, 2026 (TD-260) the GUI no
longer loads a vocabulary or model of its own, so it always talks to the same trained model as every other
client, with the server's architecture (MNS), checkpoint, GPU and prompt format.

---

## BUILD

```bash
cmake --preset=debug -DBUILD_GUI=ON
cmake --build --preset=debug --target chatbot_gui
```

Check the build:

```bash
./scripts/verify_gui_parallel.sh
```

---

## RUN

Start `chatbot_api_server` first (see the [chatbot guide](../chatbot-guide.md)), then give the GUI its URL
(default `http://localhost:8080`):

```bash
./build/debug/src/chatbot_gui http://localhost:8080
```

With the convenience script:

```bash
./scripts/run_chatbot_gui.sh http://ai-machine:8080
```

The chat pane says whether the GUI reached the server. If the server is running an untrained model
(`ALLOW_UNTRAINED_MODEL=true`), the pane warns that its replies are random. If the server isn't reachable,
the pane says so, and messages are sent once it is up.

---

## SETTINGS

Each setting starts at **server default**: it isn't sent, so `chatbot_api_server`'s own configuration
applies. Change one and it's sent with every message after that; set it back to server default to stop
sending it. The ranges are the server's own.

| Setting | Range |
|---|---|
| Strategy | Nucleus (Top-p), Top-k Sampling, Greedy, Beam Search, Sampling (Temperature) |
| Temperature | 0 to 2 (0 is greedy) |
| Top-p | 0 to 1 |
| Top-k | 1 to 1000 |
| Max Response Length | 1 to 4096 new tokens |
| Beam Width | 1 to 16, used by beam search |

---

## CONVERSATIONS

- **Clear Chat** clears the conversation on the server.
- **Save Conversation** asks the server for the conversation and writes it to the file you choose (the
  same format the `chatbot` CLI saves).
- **Load Conversation** sends a saved file to the server, which continues that conversation. The window then
  shows its messages as the server holds them (TD-266).
- If the server restarts or the conversation expires, the next message says so and starts a new one.

The window stays responsive while the server generates: input, buttons and settings are disabled until the
reply arrives. Closing the window stops waiting for a reply.

---

## TROUBLESHOOTING

- **"Could not reach chatbot_api_server"**: start the server, or pass its URL.
- **Replies are gibberish**: the server's model is untrained or undertrained; train it with
  `incremental_trainer` and restart the server on the new checkpoint.
- **Slow replies**: generation speed is the server's (its GPU and threads); lower Max Response Length or use
  Greedy.
- **No display**: use `ssh -X user@host` or VNC. `chatbot_gui --help` works without a display.

---

## FILES

- `build/<preset>/src/chatbot_gui_binary` - the GUI
- `build/<preset>/src/chatbot_gui` - wrapper that fixes library paths
- `scripts/run_chatbot_gui.sh`, `scripts/chatbot_gui_fixed.sh` - launchers
- `scripts/verify_gui_parallel.sh` - build check
- [chatbot-gui-guide.md](../chatbot-gui-guide.md) - full guide
