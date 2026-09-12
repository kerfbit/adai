#!/bin/bash

# @adai-status: beta        (TD-044 resolved — workaround for a specific Qt threading bug, see THREAD_ERROR_FIX.md; real test suite added)
# @adai-version: 0.7.1
# @adai-reviewed: 2026-09-11


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

# TD-044: even after the fix above, "<repo-root>/build/src/" itself was
# still wrong for any standard, documented build — every preset in
# CMakePresets.json builds into "build/<preset>/" (build/debug,
# build/release, ...), never bare "build/". Reproduced directly: a real
# build/debug/src/chatbot_gui existed while this script still looked for
# (and failed to find) build/src/chatbot_gui. Auto-detects the first
# preset build directory that actually has the binary, falling back to
# bare "build/" for anyone who configured without presets.
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

BUILD_DIR="$(find_build_dir src/chatbot_gui)" || {
    echo "ERROR: chatbot_gui not found in any build/<preset>/src/ directory." >&2
    echo "Build it first, e.g.: cmake --preset=debug -DBUILD_GUI=ON && cmake --build --preset=debug --target chatbot_gui" >&2
    exit 1
}

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

# Run the GUI. cd to REPO_ROOT first (not just resolve BUILD_DIR to an
# absolute path) since the GUI itself may do its own relative-path lookups
# (vocab.txt, model files) that are meant to resolve from the repo root,
# matching every sibling launcher script's convention.
cd "$REPO_ROOT"
exec "${BUILD_DIR}/src/chatbot_gui" "$@"
