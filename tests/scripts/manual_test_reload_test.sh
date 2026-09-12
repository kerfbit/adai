#!/bin/bash
#
# Tests for scripts/manual_test_reload.sh (TD-044).
#
# This script's whole purpose is to start a real server and hand control to
# a human for manual SIGHUP testing — it never exits on its own. Run under
# `timeout` and torn down unconditionally; what's actually verified is that
# it gets PAST the old "binary not found" bug into real server startup
# (tolerant of vocab.txt itself being absent in this checkout, a separate,
# environment-specific fact unrelated to the path-resolution bug this test
# targets).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/manual_test_reload.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "bin/chatbot_api_server"

# ---------------------------------------------------------------------------
t "TD-044 regression: gets past 'binary not found' into real server startup"
if [[ -x "${REPO_ROOT}/build/debug/bin/chatbot_api_server" ]]; then
    raw="$(timeout 4 bash "$TARGET" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$raw"
    assert_not_contains "No such file or directory" "must exec the real binary, not the old bare-build/bin/ path"
    assert_contains "Starting server..."
    pkill -9 -f "chatbot_api_server.*manual_test_config" 2>/dev/null
    rm -f /tmp/manual_test_config.conf
else
    t "(skipped: build/debug/bin/chatbot_api_server not built in this checkout)"
fi

harness_report
exit $?
