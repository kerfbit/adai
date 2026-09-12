#!/bin/bash
#
# Tests for scripts/test_chatbot_gui.sh (TD-044).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/test_chatbot_gui.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "src/chatbot_gui"

# ---------------------------------------------------------------------------
t "TD-044 regression: runs correctly regardless of caller's CWD"
run_in "${REPO_ROOT}/scripts" bash "$TARGET"
assert_not_contains "chatbot_gui executable not found" "must find the real build/debug/src/chatbot_gui, not fail with the old bare-build/ bug"

t "reports SUCCESS against a real, complete build (or a specific, real symbol/link gap — never the old false 'not found')"
if [[ -x "${REPO_ROOT}/build/debug/src/chatbot_gui_binary" ]]; then
    # Before the TD-044 fix, this always failed at "Executable exists"
    # regardless of build correctness. After the fix, a genuinely correct
    # build must report SUCCESS.
    assert_contains "Build Verification: SUCCESS"
    assert_exit 0
else
    t "(skipped: build/debug/src/chatbot_gui_binary not built in this checkout)"
fi

harness_report
exit $?
