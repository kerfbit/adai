# Chatbot GUI - Quick Reference Card

---

## RUNNING THE GUI

✅ **RECOMMENDED** (handles all environment issues):

```bash
./scripts/chatbot_gui_fixed.sh
```

Also works:

```bash
./scripts/run_chatbot_gui.sh
```

❌ **AVOID** (may have thread/library errors):

```bash
./build/src/chatbot_gui
```

---

## COMMAND LINE OPTIONS

Default (uses vocab.txt and chatbot_model.bin):

```bash
./scripts/chatbot_gui_fixed.sh
```

Custom vocab:

```bash
./scripts/chatbot_gui_fixed.sh my_vocab.txt
```

Custom vocab and model:

```bash
./scripts/chatbot_gui_fixed.sh my_vocab.txt my_model.bin
```

Show help:

```bash
./scripts/chatbot_gui_fixed.sh --help
```

---

## COMMON ISSUES & FIXES

**Issue:** pthread symbol error
**Fix:** Use `./scripts/chatbot_gui_fixed.sh` ✅

**Issue:** No display found
**Fix:** `ssh -X user@host` OR use VNC

**Issue:** Slow responses
**Fix:** Reduce max length, use Greedy strategy

**Issue:** Model not loading
**Fix:** Run from `/home/rodney/Repos/adai` directory

---

## WARNINGS YOU CAN IGNORE

⚠️  **"QSocketNotifier: Can only be used with threads..."**
→ Harmless Qt warning, GUI works fine

⚠️  **"Failed to load module canberra-gtk-module"**
→ Optional sound module, GUI works without it

---

## FILES & LOCATIONS

**Executable:** `build/src/chatbot_gui`
**Launcher:** `scripts/chatbot_gui_fixed.sh` ⭐
**Alt launcher:** `scripts/run_chatbot_gui.sh`

**Required:** `vocab.txt` (in current directory)
**Optional:** `chatbot_model.bin*` (trained model)

Documentation:

- [chatbot-gui-guide.md](../chatbot-gui-guide.md) - Full guide

---

## QUICK TEST

Verify it works:

```bash
./scripts/chatbot_gui_fixed.sh --help
```

Expected output:

```text
Usage: ./build/src/chatbot_gui [vocab_file] [model_file]
...
```

---

## GENERATION STRATEGIES

- **Nucleus (Top-p)** ⭐ Recommended - balanced creativity
- **Top-k Sampling** → Good quality, controlled randomness
- **Greedy** → Fastest, deterministic
- **Beam Search** → High quality, slower
- **Sampling** → Most creative, unpredictable

---

## RECOMMENDED SETTINGS

Creative Writing:

- Strategy: Nucleus, Temp: 1.2, Top-p: 0.9

Technical Q&A:

- Strategy: Greedy, Temp: 0.7, Top-p: 0.95

General Chat:

- Strategy: Nucleus, Temp: 1.0, Top-p: 0.9

---

## KEYBOARD SHORTCUTS

- **Enter** → Send message
- **Ctrl+Q** → Quit (standard Qt)

---

## GETTING HELP

- **Read:** [chatbot-gui-guide.md](../chatbot-gui-guide.md)
- **Test:** `./scripts/test_chatbot_gui_comprehensive.sh`
- **Check:** [chatbot-gui-guide.md](../chatbot-gui-guide.md)

---

> **💡 TIP:** Always use `./scripts/chatbot_gui_fixed.sh` for best results!
