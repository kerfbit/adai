# Chatbot GUI Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: "symbol lookup error: __libc_pthread_init"

Full Error:

```text
./build/src/chatbot_gui: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
```

Cause:
Conflict between snap-provided libraries and system Qt libraries.

Solution 1 - Use the Fixed Launcher (Recommended):

```bash
./chatbot_gui_fixed.sh
# or
./run_chatbot_gui.sh
```

Solution 2 - Run Directly with Environment Fix:

```bash
env -u LD_LIBRARY_PATH \
    -u GTK_PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    ./build/src/chatbot_gui
```

Solution 3 - Unset Variables in Current Shell:

```bash
unset LD_LIBRARY_PATH
unset GTK_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
./build/src/chatbot_gui
```

---

### Issue 2: "QSocketNotifier: Can only be used with threads started with QThread"

Cause:
Warning related to Qt threading initialization (usually harmless).

Status:
This is a warning, not an error. The application will still work correctly.

To Suppress (if desired):

```bash
./build/src/chatbot_gui 2>&1 | grep -v "QSocketNotifier"
```

---

### Issue 3: "Failed to load module canberra-gtk-module"

Full Warning:

```text
Gtk-Message: Failed to load module "canberra-gtk-module"
```

Cause:
Optional GTK sound module not installed.

Status:
This is a harmless warning. The GUI works perfectly without it.

To Fix (optional):

```bash
sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```

To Suppress:

```bash
export GTK_MODULES=""
./build/src/chatbot_gui
```

---

### Issue 4: No Display Found / Cannot Open Display

Error:

```text
qt.qpa.xcb: could not connect to display
```

Cause:
No graphical environment available.

Solutions:

1. Running Locally:

Make sure you're in a graphical session (not SSH without X forwarding).

2. SSH with X11 Forwarding:

```bash
ssh -X user@host
# or
ssh -Y user@host  # trusted X11 forwarding
```

3. VNC/Remote Desktop:

Set up VNC server and connect with a VNC client.

4. Virtual Display (for testing):

```bash
# Install xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run ./build/src/chatbot_gui
```

---

### Issue 5: Qt Platform Plugin Not Found

Error:

```text
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

Solution:

```bash
# Install required Qt plugins
sudo apt-get install libqt5gui5 libqt5widgets5 qt5-qmake

# Set plugin path explicitly
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
./build/src/chatbot_gui
```

---

### Issue 6: Server Not Reachable

Since November 4, 2026 (TD-260) the GUI is a client of `chatbot_api_server`: it loads no vocabulary or model
itself. The chat pane says when it can't reach the server:

```text
Could not reach chatbot_api_server at http://localhost:8080 (Could not establish connection). ...
```

Solutions:

1. Start `chatbot_api_server`, and check it answers: `curl -s http://localhost:8080/health`
2. Pass the right URL: `./build/src/chatbot_gui http://<host>:8080`

The old `chatbot_gui vocab.txt model.bin` form now exits with an error saying so.

---

### Issue 7: Slow Responses

Generation runs in the server, so the GUI stays responsive while it waits (TD-263); input and settings are
disabled until the reply arrives.

Solutions:

- Lower max response length (50-100 tokens)
- Use "Greedy" strategy instead of "Beam Search"
- Check the server: it should be using its GPU if it has one

---

### Issue 8: Gibberish Replies / "untrained model" Warning

The server's model is untrained or undertrained. A server started with `ALLOW_UNTRAINED_MODEL=true` makes the
GUI warn when it connects. Train the model with `incremental_trainer` and restart the server on the new
checkpoint.

---

## Quick Diagnostic Commands

```bash
# Check if GUI executable exists and is valid
file build/src/chatbot_gui

# Check Qt libraries are linked
ldd build/src/chatbot_gui | grep Qt5

# Verify the server is up (the GUI's default URL)
curl -s http://localhost:8080/health

# Test with help flag (minimal initialization)
./chatbot_gui_fixed.sh --help

# Check display is available
echo $DISPLAY
xdpyinfo | head -5

# Verify Qt installation
qmake -v
```

---

## Getting Help

If issues persist:

1. **Check build logs:**

   ```bash
   cd build
   cmake .. -DBUILD_GUI=ON 2>&1 | tee cmake.log
   make chatbot_gui 2>&1 | tee build.log
   ```

2. **Run comprehensive tests:**

   ```bash
   ./test_chatbot_gui_comprehensive.sh
   ```

3. **Check system requirements:**
   - Qt5 or Qt6 installed
   - Graphical environment available
   - Sufficient RAM (>2GB recommended)

4. **Review documentation:**
   - [Full GUI Guide](docs/guides/chatbot-gui-guide.md)
   - [Quick Reference](CHATBOT_GUI_README.md)
   - [Build Summary](CHATBOT_GUI_BUILD_SUMMARY.md)

---

## Summary of Solutions

|Issue|Quick Fix|
|-------|-----------|
|pthread symbol error|Use `./chatbot_gui_fixed.sh`|
|No display|SSH with `-X` or use VNC|
|Qt plugin missing|`sudo apt install libqt5gui5`|
|GTK warnings|`export GTK_MODULES=""`|
|Server not reachable|Start `chatbot_api_server` or pass its URL|
|Slow generation|Reduce max length, use greedy|
|Gibberish replies|Train the server's model|

**Remember:** Always use the provided launcher scripts for the smoothest experience!
