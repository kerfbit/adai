#!/bin/bash

# @adai-status: beta
# @adai-version: 0.6.0
# @adai-reviewed: 2026-09-07


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

cd /home/rodney/Repos/adai
exec ./build/src/chatbot_api_server --config "$TEST_CONFIG"
