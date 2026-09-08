#!/bin/bash

# @adai-status: experimental        (near-duplicate of test_signal_handling.sh, superseded by it; capped by TD-046 — see TECHNICAL_DEBT.md)
# @adai-version: 0.3.0
# @adai-reviewed: 2026-09-07

# Test script for SIGINT signal handling

echo "==================================================================="
echo "SIGINT Signal Handling Test"
echo "==================================================================="
echo ""

# Set up configuration
export VOCAB_PATH=/home/rodney/Repos/adai/vocab.txt
export PORT=18081  # Use different port
export LOG_LEVEL=INFO  # Enable INFO level logging for testing

# Clean up any previous instances
pkill -9 chatbot_api_server 2>/dev/null
sleep 1

echo "Starting chatbot_api_server on port $PORT..."
/home/rodney/Repos/adai/build/src/chatbot_api_server > /tmp/sigint_test.log 2>&1 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo ""

# Wait for server to initialize
echo "Waiting for server to initialize (3 seconds)..."
sleep 3

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/sigint_test.log
    exit 1
fi

echo "Server is running."
echo ""

# Send SIGINT signal (Ctrl+C equivalent)
echo "==================================================================="
echo "Sending SIGINT (Ctrl+C) to server (PID: $SERVER_PID)..."
echo "==================================================================="
echo ""
kill -INT $SERVER_PID

# Wait for graceful shutdown
echo "Waiting for graceful shutdown (3 seconds)..."
sleep 3

# Check shutdown output
echo ""
echo "==================================================================="
echo "Graceful Shutdown Output:"
echo "==================================================================="
tail -20 /tmp/sigint_test.log
echo ""

# Verify process is no longer running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "WARNING: Server is still running after SIGINT"
    kill -9 $SERVER_PID
    echo "Force killed the server"
else
    echo "SUCCESS: Server shut down gracefully after SIGINT"
fi

echo ""
echo "Full server log saved to: /tmp/sigint_test.log"
