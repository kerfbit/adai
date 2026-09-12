#!/bin/bash

# @adai-status: beta        (TD-044 resolved — real test suite added, see tests/scripts/test_signal_handling_test.sh)
# @adai-version: 0.7.2
# @adai-reviewed: 2026-09-11

# Test script for signal handling verification

echo "==================================================================="
echo "Signal Handling Test Script"
echo "==================================================================="
echo ""

# Resolve paths relative to this script instead of hardcoding one developer's
# checkout location (TD-118 — same class as TD-116/TD-117). Also corrects
# build/src/ to build/bin/: chatbot_api_server's CMake target sets
# RUNTIME_OUTPUT_DIRECTORY to ${CMAKE_BINARY_DIR}/bin, never .../src/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

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
    echo "ERROR: chatbot_api_server not found in any build/<preset>/bin/ directory."
    echo "Build it first, e.g.: cmake --preset=debug && cmake --build --preset=debug --target chatbot_api_server"
    exit 1
}

# Set up configuration
export VOCAB_PATH="${REPO_ROOT}/vocab.txt"
export PORT=18080  # Use a non-standard port to avoid conflicts
export LOG_LEVEL=INFO  # Enable INFO level logging for testing

# Clean up any previous instances
pkill -9 chatbot_api_server 2>/dev/null
sleep 1

echo "Starting chatbot_api_server on port $PORT..."
"${BUILD_DIR}/bin/chatbot_api_server" > /tmp/signal_test.log 2>&1 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo ""

# Wait for server to initialize
echo "Waiting for server to initialize (3 seconds)..."
sleep 3

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start or crashed during initialization"
    echo ""
    echo "Server output:"
    cat /tmp/signal_test.log
    exit 1
fi

echo "Server is running. Checking initialization output..."
echo ""
head -30 /tmp/signal_test.log
echo ""

# Send SIGTERM signal
echo "==================================================================="
echo "Sending SIGTERM to server (PID: $SERVER_PID)..."
echo "==================================================================="
echo ""
kill -TERM $SERVER_PID

# Wait for graceful shutdown
echo "Waiting for graceful shutdown (5 seconds)..."
sleep 5

# Check shutdown output
echo ""
echo "==================================================================="
echo "Graceful Shutdown Output:"
echo "==================================================================="
tail -20 /tmp/signal_test.log
echo ""

# Verify process is no longer running
#
# TD-044: this check only ever printed WARNING vs. SUCCESS — nothing set a
# failing exit code, so a server that never shuts down gracefully on
# SIGTERM (the actual thing this script exists to verify) was reported as
# a pass to any caller checking $?. Track it explicitly and exit 1.
GRACEFUL_SHUTDOWN=true
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "WARNING: Server is still running after SIGTERM"
    kill -9 $SERVER_PID
    echo "Force killed the server"
    GRACEFUL_SHUTDOWN=false
else
    echo "SUCCESS: Server shut down gracefully"
fi

echo ""
echo "==================================================================="
echo "Full server log saved to: /tmp/signal_test.log"
echo "==================================================================="

if [ "$GRACEFUL_SHUTDOWN" = true ]; then
    exit 0
else
    exit 1
fi
