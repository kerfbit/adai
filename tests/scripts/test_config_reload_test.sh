#!/bin/bash
#
# Tests for scripts/test_config_reload.sh (TD-044).
#
# The real script is already a genuine, fairly complete end-to-end test
# (starts the real server, sends real SIGHUP reloads, checks it survives).
# What's verified here is the TD-044 fix specifically: it must get past the
# old "binary not found" bug into real server startup, tolerant of
# vocab.txt itself being absent in this checkout (an environment fact
# unrelated to the path-resolution bug this test targets) — if vocab.txt IS
# present, the real script's own full SIGHUP-reload sequence is allowed to
# run to completion and its exit code is checked for real.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/test_config_reload.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "bin/chatbot_api_server"

# ---------------------------------------------------------------------------
t "TD-044 regression: gets past 'binary not found', reaches real server startup"
if [[ -x "${REPO_ROOT}/build/debug/bin/chatbot_api_server" ]]; then
    raw="$(timeout 30 bash "$TARGET" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$raw"
    assert_not_contains "No such file or directory" "must exec the real binary, not the old bare-build/bin/ path"
    assert_contains "Server started with PID:"
    if [[ -f "${REPO_ROOT}/vocab.txt" ]]; then
        t "full SIGHUP reload sequence passes against a real vocab.txt"
        assert_exit 0
        assert_contains "All Tests PASSED"
    else
        t "(vocab.txt absent in this checkout — full reload sequence not exercised, only path resolution)"
    fi
    pkill -9 -f "chatbot_api_server.*test_config_reload" 2>/dev/null
    rm -f /tmp/test_config_reload.conf
else
    t "(skipped: build/debug/bin/chatbot_api_server not built in this checkout)"
fi

harness_report
exit $?
