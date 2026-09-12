#!/bin/bash
#
# Tests for scripts/verify_gui_parallel.sh (TD-044).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/verify_gui_parallel.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "src/chatbot_gui_binary"

# ---------------------------------------------------------------------------
t "TD-044 regression: runs correctly regardless of caller's CWD"
run_in "${REPO_ROOT}/scripts" bash "$TARGET"
assert_not_contains "chatbot_gui_binary not found" "must find the real build/debug/src/chatbot_gui_binary, not fail with the old bare-build/ bug"
assert_contains "chatbot_gui_binary found"

t "reports the real, resolved binary paths (not stale bare build/src/...)"
if [[ -x "${REPO_ROOT}/build/debug/src/chatbot_gui_binary" ]]; then
    assert_contains "${REPO_ROOT}/build/debug/src/chatbot_gui_binary"
    assert_exit 0
else
    t "(skipped: build/debug/src/chatbot_gui_binary not built in this checkout)"
fi

harness_report
exit $?
