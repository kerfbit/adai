#!/bin/bash

# @adai-status: beta        (capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.8.0
# @adai-reviewed: 2026-09-07


# Quick start script for chatbot_gui
# This script helps users launch the GUI with the correct settings

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         ADAI Chatbot GUI - Quick Launcher             ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

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

# Check if executable exists
if [ ! -f "build/src/chatbot_gui" ]; then
    echo "❌ ERROR: chatbot_gui not found"
    echo ""
    echo "Please build it first:"
    echo "  cd build"
    echo "  cmake .. -DBUILD_GUI=ON"
    echo "  make chatbot_gui -j\$(nproc)"
    echo ""
    exit 1
fi

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
echo "Executable: build/src/chatbot_gui"
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
exec ./build/src/chatbot_gui "$@"
