#!/bin/bash
#
# Tests for scripts/test_log_rotation.sh (TD-044).
#
# Same approach as test_config_reload_test.sh: verify the TD-044 fix gets
# past the old "binary not found" bug, tolerant of vocab.txt being absent
# in this checkout; if present, let the real script's full sequence run
# and check its real exit code.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/test_log_rotation.sh"
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
    # Unlike test_config_reload.sh, this script's checks only depend on the
    # logging subsystem (initialized before vocab.txt is ever loaded), not
    # on the server fully starting — so its full sequence genuinely passes
    # even in a checkout with no vocab.txt.
    t "full log-rotation sequence passes"
    assert_exit 0
    assert_contains "Log Rotation Test Complete"
    pkill -9 -f "chatbot_api_server.*test_log_rotation" 2>/dev/null
    rm -f /tmp/test_log_rotation.conf
    rm -rf /tmp/adai_log_test
else
    t "(skipped: build/debug/bin/chatbot_api_server not built in this checkout)"
fi

harness_report
exit $?
