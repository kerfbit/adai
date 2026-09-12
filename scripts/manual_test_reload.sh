#!/bin/bash

# @adai-status: beta        (TD-044 resolved — real test suite added, see tests/scripts/manual_test_reload_test.sh)
# @adai-version: 0.6.2
# @adai-reviewed: 2026-09-11


# Manual test for configuration hot-reload
# This script allows you to manually test the SIGHUP handling

echo "=========================================="
echo "Manual Configuration Hot-Reload Test"
echo "=========================================="
echo ""

# Create test config
TEST_CONFIG="/tmp/manual_test_config.conf"
cat > "$TEST_CONFIG" << 'EOF'
VOCAB_PATH=vocab.txt
LOG_LEVEL=INFO
PORT=8080
SESSION_TIMEOUT=30
D_MODEL=512
NUM_HEADS=8
MAX_GEN_LENGTH=100
TEMPERATURE=1.0
TOP_P=0.9
STRATEGY=nucleus
EOF

echo "Test configuration created: $TEST_CONFIG"
echo ""
echo "Starting server..."
echo "After server starts, you can:"
echo "  1. Edit $TEST_CONFIG"
echo "  2. Send SIGHUP to reload: kill -HUP <PID>"
echo "  3. Watch the logs for reload messages"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
# chatbot_api_server's CMake target sets RUNTIME_OUTPUT_DIRECTORY to
# ${CMAKE_BINARY_DIR}/bin, not .../src/ — this previously pointed at a path
# the build has never actually produced. Also replaced the hardcoded
# /home/rodney/Repos/adai path (broken for any other checkout) with the
# same SCRIPT_DIR-relative resolution every other script in this repo uses.
#
# TD-044: "${REPO_ROOT}/build/bin/..." was STILL wrong even after that fix —
# every preset in CMakePresets.json builds into "build/<preset>/", never
# bare "build/". Auto-detects the first preset build directory that
# actually has the binary, falling back to bare "build/" for a non-preset
# configure.
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

BUILD_DIR="$(find_build_dir bin/chatbot_api_server)" || {
    echo "ERROR: chatbot_api_server not found in any build/<preset>/bin/ directory." >&2
    echo "Build it first, e.g.: cmake --preset=debug && cmake --build --preset=debug --target chatbot_api_server" >&2
    exit 1
}

cd "${REPO_ROOT}" || exit 1
exec "${BUILD_DIR}/bin/chatbot_api_server" --config "$TEST_CONFIG"
