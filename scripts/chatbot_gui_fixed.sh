#!/bin/bash

# @adai-status: beta        (workaround for a specific Qt threading bug — see THREAD_ERROR_FIX.md; capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.0
# @adai-reviewed: 2026-09-10


# Wrapper script to run chatbot_gui with correct library paths
# Fixes snap/system library conflicts

# Save current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# chatbot_gui has no RUNTIME_OUTPUT_DIRECTORY override in src/CMakeLists.txt,
# so it genuinely builds under <repo-root>/build/src/ (confirmed against
# TD-114/116/117/118's identical check). But this script used to `cd
# "$SCRIPT_DIR"` (scripts/ itself) and then exec the relative path
# `./build/src/chatbot_gui` — looking for scripts/build/src/chatbot_gui,
# one directory too deep, since the real binary lives at
# <repo-root>/build/src/chatbot_gui, not scripts/build/src/chatbot_gui.
# The exec below never found a binary; reproduced directly. Resolve and cd
# to REPO_ROOT (one level up from scripts/) instead, matching every sibling
# launcher script.
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# Unset snap-related environment variables that cause conflicts
unset GTK_PATH
unset LD_LIBRARY_PATH
unset SNAP
unset SNAP_COMMON
unset SNAP_DATA

# Set Qt plugin path to system location
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins

# Ensure we're using system libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu

# Suppress harmless GTK warnings
export GTK_MODULES=""

# Run the GUI
cd "$REPO_ROOT"
exec ./build/src/chatbot_gui "$@"
