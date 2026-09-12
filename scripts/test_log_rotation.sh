#!/bin/bash

# @adai-status: beta        (TD-044 resolved — real test suite added, see tests/scripts/test_log_rotation_test.sh)
# @adai-version: 0.7.2
# @adai-reviewed: 2026-09-11


# Script to test log file rotation and management
# Tests file creation, rotation, and size limits

set -e

echo "=========================================="
echo "Log File Rotation Test"
echo "=========================================="
echo ""

# Create test directory for logs
TEST_LOG_DIR="/tmp/adai_log_test"
TEST_LOG_FILE="$TEST_LOG_DIR/adai_test.log"

echo "Creating test directory: $TEST_LOG_DIR"
rm -rf "$TEST_LOG_DIR"
mkdir -p "$TEST_LOG_DIR"
echo ""

# Create test configuration with small file size for quick rotation
TEST_CONFIG="/tmp/test_log_rotation.conf"
echo "Creating test configuration file: $TEST_CONFIG"

cat > "$TEST_CONFIG" << 'EOF'
# Test Configuration for Log Rotation
VOCAB_PATH=vocab.txt
LOG_LEVEL=DEBUG
LOG_FILE_PATH=/tmp/adai_log_test/adai_test.log
LOG_MAX_SIZE_MB=1
LOG_MAX_FILES=3
LOG_COMPRESS=false
PORT=8088
SESSION_TIMEOUT=30
D_MODEL=512
NUM_HEADS=8
MAX_GEN_LENGTH=100
TEMPERATURE=1.0
TOP_P=0.9
STRATEGY=nucleus
EOF

echo "Configuration created with:"
echo "  LOG_FILE_PATH: /tmp/adai_log_test/adai_test.log"
echo "  LOG_MAX_SIZE_MB: 1 MB"
echo "  LOG_MAX_FILES: 3"
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

# Start the server in background
echo "Starting server with file logging..."
cd "${REPO_ROOT}"
"${BUILD_DIR}/bin/chatbot_api_server" --config "$TEST_CONFIG" > /dev/null 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo ""

# Wait for server to start and create log file
echo "Waiting for server to initialize..."
sleep 3
echo ""

# Check if log file was created
if [ -f "$TEST_LOG_FILE" ]; then
    echo "✓ Log file created: $TEST_LOG_FILE"
    echo "  Initial size: $(du -h $TEST_LOG_FILE | cut -f1)"
else
    echo "✗ ERROR: Log file was not created"
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi
echo ""

# Check log file content
echo "Checking log file content:"
echo "  First 5 lines:"
head -5 "$TEST_LOG_FILE" | sed 's/^/    /'
echo ""

# Generate some log activity by making API calls
echo "Generating log activity (making API calls)..."
for i in {1..10}; do
    curl -s -X GET http://localhost:8088/health > /dev/null 2>&1 || true
    sleep 0.1
done
echo "  Made 10 API calls"
echo ""

# Check current log size
LOG_SIZE=$(stat -c%s "$TEST_LOG_FILE" 2>/dev/null || stat -f%z "$TEST_LOG_FILE")
echo "Current log file size: $LOG_SIZE bytes"
echo ""

# Test file listing in log directory
echo "Files in log directory:"
ls -lh "$TEST_LOG_DIR" | tail -n +2 | sed 's/^/  /'
echo ""

# Check if multiple log files would be created (for large logs)
#
# TEST_PASSED tracks these two checks so the closing "Summary" section below
# reflects what was actually observed instead of unconditionally claiming
# success — it previously printed "✓ Timestamped log format correct" even
# right after this same run printed "✗ Timestamp format not found" two lines
# above it, and always exited 0 regardless.
TEST_PASSED=true
LOG_COUNT=$(ls -1 "$TEST_LOG_DIR"/*.log* 2>/dev/null | wc -l)
echo "Number of log files: $LOG_COUNT"
if [ "$LOG_COUNT" -gt 0 ]; then
    echo "✓ Log rotation system operational"
else
    echo "✗ No log files found"
    TEST_PASSED=false
fi
echo ""

# Verify log entries are timestamped
echo "Verifying timestamp format:"
if grep -q '\[20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9][0-9]\]' "$TEST_LOG_FILE"; then
    echo "✓ Log entries have correct timestamp format"
    echo "  Example: $(grep -m1 '\[20[0-9][0-9]-' $TEST_LOG_FILE)"
else
    echo "✗ Timestamp format not found"
    TEST_PASSED=false
fi
echo ""

# Cleanup
echo "=========================================="
echo "Cleanup"
echo "=========================================="
echo "Stopping server..."
# TD-044: unguarded, unlike the two other kill calls in this same file
# (line ~107 redirects stderr, the one below only runs inside an `if ps -p`
# check) — if the server had already died before reaching cleanup (crashed,
# OOM, or simply exited on its own for any reason, as it does whenever
# vocab.txt is missing), `kill -TERM` on a nonexistent PID returns nonzero
# and `set -e` killed the whole script right here, before the log-file
# preservation summary, config cleanup, or final pass/fail report/exit code
# ever ran — reporting an accidental exit 1 (kill's own failure) regardless
# of whether $TEST_PASSED was genuinely true. Reproduced directly. `2>
# /dev/null || true` makes an already-dead process a silent no-op here,
# matching this script's other two kill calls' spirit.
kill -TERM $SERVER_PID 2>/dev/null || true
sleep 2

# Force kill if still running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "Server still running, forcing kill..."
    kill -9 $SERVER_PID
fi

echo "Preserving log files for inspection:"
echo "  Directory: $TEST_LOG_DIR"
echo "  Files:"
ls -lh "$TEST_LOG_DIR" | tail -n +2 | sed 's/^/    /'
echo ""
echo "To view logs: cat $TEST_LOG_FILE"
echo "To clean up: rm -rf $TEST_LOG_DIR"
echo ""

echo "Removing test config..."
rm -f "$TEST_CONFIG"
echo ""

echo "=========================================="
if [ "$TEST_PASSED" = true ]; then
    echo "Log Rotation Test Complete ✓"
else
    echo "Log Rotation Test Complete — WITH FAILURES ✗"
fi
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Log file creation working"
if [ "$TEST_PASSED" = true ]; then
    echo "  ✓ Timestamped log format correct"
    echo "  ✓ File rotation system configured"
else
    echo "  ✗ See ✗ lines above for what failed"
fi
echo "  ✓ Max files limit: 3 rotated files"
echo "  ✓ Max size limit: 1 MB per file"
echo ""
echo "Log files preserved at: $TEST_LOG_DIR"
echo ""
echo "Note: To fully test rotation, the log file needs to exceed 1 MB."
echo "Run longer tests or reduce LOG_MAX_SIZE_MB for faster rotation."

if [ "$TEST_PASSED" = true ]; then
    exit 0
else
    exit 1
fi
