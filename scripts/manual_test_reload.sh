#!/bin/bash

# @adai-status: beta        (capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.6.1
# @adai-reviewed: 2026-09-10


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
cd "${REPO_ROOT}" || exit 1
exec ./build/bin/chatbot_api_server --config "$TEST_CONFIG"
