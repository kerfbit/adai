#!/bin/bash

# @adai-status: beta        (TD-044 resolved — workaround for a specific Qt threading bug, see THREAD_ERROR_FIX.md; real test suite added)
# @adai-version: 0.8.0
# @adai-reviewed: 2026-11-04


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

# The chatbot_gui wrapper itself removes snap library/GTK/Qt paths from the environment and, under snap,
# points Qt at the platform plugins of the Qt it was built with (TD-264). This script used to set
# x86_64 Qt5 paths here, which also replaced the user's LD_LIBRARY_PATH.

# Run the GUI from REPO_ROOT, matching every sibling launcher script's convention.
cd "$REPO_ROOT"
exec "${BUILD_DIR}/src/chatbot_gui" "$@"
