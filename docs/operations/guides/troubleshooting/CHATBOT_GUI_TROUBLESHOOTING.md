# Chatbot GUI Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: "symbol lookup error: __libc_pthread_init"

**Full Error:**

```text
./build/src/chatbot_gui: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0: undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
```

**Cause:**
Conflict between snap-provided libraries and system Qt libraries.

**Solution 1 - Use the Fixed Launcher (Recommended):**

```bash
./chatbot_gui_fixed.sh
# or
./run_chatbot_gui.sh
```

**Solution 2 - Run Directly with Environment Fix:**

```bash
env -u LD_LIBRARY_PATH \
    -u GTK_PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
    ./build/src/chatbot_gui
```

**Solution 3 - Unset Variables in Current Shell:**

```bash
unset LD_LIBRARY_PATH
unset GTK_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
./build/src/chatbot_gui
```

---

### Issue 2: "QSocketNotifier: Can only be used with threads started with QThread"

**Cause:**
Warning related to Qt threading initialization (usually harmless).

**Status:**
This is a warning, not an error. The application will still work correctly.

**To Suppress (if desired):**

```bash
./build/src/chatbot_gui 2>&1 | grep -v "QSocketNotifier"
```

---

### Issue 3: "Failed to load module canberra-gtk-module"

**Full Warning:**

```text
Gtk-Message: Failed to load module "canberra-gtk-module"
```

**Cause:**
Optional GTK sound module not installed.

**Status:**
This is a harmless warning. The GUI works perfectly without it.

**To Fix (optional):**

```bash
sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```

**To Suppress:**

```bash
export GTK_MODULES=""
./build/src/chatbot_gui
```

---

### Issue 4: No Display Found / Cannot Open Display

**Error:**

```text
qt.qpa.xcb: could not connect to display
```

**Cause:**
No graphical environment available.

**Solutions:**

**1. Running Locally:**
Make sure you're in a graphical session (not SSH without X forwarding).

**2. SSH with X11 Forwarding:**

```bash
ssh -X user@host
# or
ssh -Y user@host  # trusted X11 forwarding
```

**3. VNC/Remote Desktop:**
Set up VNC server and connect with a VNC client.

**4. Virtual Display (for testing):**

```bash
# Install xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run ./build/src/chatbot_gui
```

---

### Issue 5: Qt Platform Plugin Not Found

**Error:**

```text
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

**Solution:**

```bash
# Install required Qt plugins
sudo apt-get install libqt5gui5 libqt5widgets5 qt5-qmake

# Set plugin path explicitly
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
./build/src/chatbot_gui
```

---

### Issue 6: Model Loading Fails

**Error in GUI:**

```text
Failed to initialize chatbot components
```

**Solutions:**

**1. Check Files Exist:**

```bash
ls -lh vocab.txt chatbot_model.bin*
```

**2. Use Correct Paths:**

```bash
# Run from project root
cd /home/rodney/Repos/adai
./build/src/chatbot_gui

# Or use absolute paths
./build/src/chatbot_gui /absolute/path/to/vocab.txt /absolute/path/to/model.bin
```

**3. Check Model Format:**
Ensure model files were created with the same version of the code.

---

### Issue 7: Slow Response Generation / GUI Freezes

**Cause:**
Large model or long max_length setting.

**Solutions:**

**1. Reduce Settings:**

- Lower max response length (50-100 tokens)
- Use "Greedy" strategy instead of "Beam Search"
- Reduce beam width to 3-5

**2. Use Smaller Model:**
Train with fewer layers/smaller dimensions.

**3. Future Enhancement:**
This is a known limitation. Future versions will support async generation.

---

### Issue 8: Segmentation Fault / Crashes

**Possible Causes:**

**1. Model/Vocab Mismatch:**

```bash
# Ensure vocab size matches model
head -1 chatbot_model.bin.config  # check vocab_size
wc -l vocab.txt                   # should match
```

**2. Corrupted Model Files:**
Re-train or use a different checkpoint.

**3. Memory Issues:**
Check available RAM:

```bash
free -h
```

---

## Best Practices

### Running the GUI

**Always use the launcher scripts:**

```bash
# Best option - handles all environment issues
./chatbot_gui_fixed.sh

# Alternative
./run_chatbot_gui.sh
```

**Don't run directly unless you know what you're doing:**

```bash
# Avoid this (may have library conflicts)
./build/src/chatbot_gui
```

### Development/Debugging

**Enable Qt Debug Output:**

```bash
export QT_DEBUG_PLUGINS=1
export QT_LOGGING_RULES="*.debug=true"
./build/src/chatbot_gui
```

**Check Library Dependencies:**

```bash
ldd build/src/chatbot_gui | grep -i qt
```

**Verify Symbols:**

```bash
nm build/src/chatbot_gui | grep ChatbotGUI
```

---

## Environment-Specific Issues

### Ubuntu with Snap

**Problem:** Library conflicts with snap packages.

**Solution:** Use the provided launcher scripts which handle this automatically.

### WSL (Windows Subsystem for Linux)

**Problem:** No native display server.

**Solutions:**

1. Install VcXsrv or Xming on Windows
2. Set DISPLAY environment variable
3. Use WSLg (Windows 11 22H2+)

```bash
# WSL 1/2 with X server on Windows
export DISPLAY=:0

# WSLg (Windows 11)
# Should work automatically
```

### Docker/Container

**Problem:** No display access.

**Solutions:**

```bash
# Share X11 socket
docker run -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           your-container \
           ./build/src/chatbot_gui

# Allow X11 connections first
xhost +local:docker
```

---

## Quick Diagnostic Commands

```bash
# Check if GUI executable exists and is valid
file build/src/chatbot_gui

# Check Qt libraries are linked
ldd build/src/chatbot_gui | grep Qt5

# Verify required files
ls -lh vocab.txt chatbot_model.bin.config

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

| Issue | Quick Fix |
| ------- | ----------- |
| pthread symbol error | Use `./chatbot_gui_fixed.sh` |
| No display | SSH with `-X` or use VNC |
| Qt plugin missing | `sudo apt install libqt5gui5` |
| GTK warnings | `export GTK_MODULES=""` |
| Model not found | Run from project root |
| Slow generation | Reduce max length, use greedy |
| Segfault | Check vocab/model match |

**Remember:** Always use the provided launcher scripts for the smoothest experience!
