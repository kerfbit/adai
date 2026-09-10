#!/bin/bash

# @adai-status: beta        (capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.1
# @adai-reviewed: 2026-09-10

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

# Set up configuration
export VOCAB_PATH="${REPO_ROOT}/vocab.txt"
export PORT=18080  # Use a non-standard port to avoid conflicts
export LOG_LEVEL=INFO  # Enable INFO level logging for testing

# Clean up any previous instances
pkill -9 chatbot_api_server 2>/dev/null
sleep 1

echo "Starting chatbot_api_server on port $PORT..."
"${REPO_ROOT}/build/bin/chatbot_api_server" > /tmp/signal_test.log 2>&1 &
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
sleep 2

# Check shutdown output
echo ""
echo "==================================================================="
echo "Graceful Shutdown Output:"
echo "==================================================================="
tail -20 /tmp/signal_test.log
echo ""

# Verify process is no longer running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "WARNING: Server is still running after SIGTERM"
    kill -9 $SERVER_PID
    echo "Force killed the server"
else
    echo "SUCCESS: Server shut down gracefully"
fi

echo ""
echo "==================================================================="
echo "Full server log saved to: /tmp/signal_test.log"
echo "==================================================================="
