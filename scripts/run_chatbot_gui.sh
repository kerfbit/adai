#!/bin/bash

# @adai-status: beta        (TD-044 resolved — real test suite added, see tests/scripts/run_chatbot_gui_test.sh)
# @adai-version: 0.8.1
# @adai-reviewed: 2026-09-11


# Quick start script for chatbot_gui
# This script helps users launch the GUI with the correct settings

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         ADAI Chatbot GUI - Quick Launcher             ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# TD-044: this script used bare relative paths ("build/src/chatbot_gui",
# "vocab.txt") with no anchor to the script's own location, so it only
# worked when the caller's CWD happened to already be the repo root — cd
# scripts/ && ./run_chatbot_gui.sh looked for scripts/build/src/chatbot_gui
# instead. Resolved via SCRIPT_DIR, matching every sibling launcher script,
# then cd to REPO_ROOT so every relative path below (vocab.txt, the model
# config, build/) resolves the same way regardless of caller CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}" || exit 1

# Check if running in graphical environment
if [ -z "$DISPLAY" ]; then
    echo "❌ ERROR: No graphical display found"
    echo ""
    echo "This application requires a graphical environment (X11/Wayland)."
    echo "You cannot run it in a headless terminal."
    echo ""
    echo "Options:"
    echo "  1. Run on a system with a display"
    echo "  2. Use X11 forwarding: ssh -X user@host"
    echo "  3. Use VNC or similar remote desktop"
    echo ""
    exit 1
fi

# TD-044: "build/src/chatbot_gui" (even once anchored to REPO_ROOT) was
# still wrong for any standard, documented build — every preset in
# CMakePresets.json builds into "build/<preset>/", never bare "build/".
# Auto-detects the first preset build directory that actually has the
# binary, falling back to bare "build/" for a non-preset configure.
find_build_dir() {
    local marker="$1"
    local dir
    for dir in "${REPO_ROOT}/build/debug" "${REPO_ROOT}/build/release" \
               "${REPO_ROOT}/build/portable" "${REPO_ROOT}/build/relwithdebinfo" \
               "${REPO_ROOT}/build/gpu" "${REPO_ROOT}/build/sycl" \
               "${REPO_ROOT}/build"; do
        if [ -f "${dir}/${marker}" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

BUILD_DIR="$(find_build_dir src/chatbot_gui)"
if [ -z "$BUILD_DIR" ]; then
    echo "❌ ERROR: chatbot_gui not found"
    echo ""
    echo "Please build it first:"
    echo "  cmake --preset=debug -DBUILD_GUI=ON"
    echo "  cmake --build --preset=debug --target chatbot_gui"
    echo ""
    exit 1
fi
GUI_BIN="${BUILD_DIR}/src/chatbot_gui"

# Check for vocab file
if [ ! -f "vocab.txt" ]; then
    echo "⚠️  WARNING: vocab.txt not found in current directory"
    echo ""
    echo "The chatbot needs a vocabulary file to run."
    echo "Please ensure vocab.txt is in: $(pwd)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for model
if [ ! -f "chatbot_model.bin.config" ]; then
    echo "ℹ️  INFO: No trained model found"
    echo ""
    echo "The chatbot will use random initialization."
    echo "For better results, train a model first with chatbot_trainer."
    echo ""
fi

echo "🚀 Launching Chatbot GUI..."
echo ""
echo "Executable: ${GUI_BIN}"
echo "Vocab file: vocab.txt"
echo "Model: chatbot_model.bin (if available)"
echo ""

# Fix for snap/system library conflicts
# Unset problematic environment variables that snap sets
unset GTK_PATH
unset LD_LIBRARY_PATH

# Use system libraries, not snap libraries
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
export GTK_MODULES=""

# Launch the GUI
exec "${GUI_BIN}" "$@"
