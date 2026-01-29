#!/bin/bash

# Wrapper script to run chatbot_gui with correct library paths
# Fixes snap/system library conflicts

# Save current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
cd "$SCRIPT_DIR"
exec ./build/src/chatbot_gui "$@"
