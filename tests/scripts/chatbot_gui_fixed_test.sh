#!/bin/bash
#
# Tests for scripts/chatbot_gui_fixed.sh (TD-044).
#
# Never lets the real Qt GUI actually launch to completion — it execs, so a
# successful find would replace this test process. Runs it under `timeout`
# with a real (but presumably invalid/no-op in this environment) DISPLAY so
# Qt fails fast on its own, which is enough to distinguish "binary not
# found" (the bug) from "binary found, Qt/display problem" (unrelated,
# environment-dependent, and not what this test is checking).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/chatbot_gui_fixed.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "src/chatbot_gui"

# ---------------------------------------------------------------------------
t "TD-044 regression: the binary is genuinely found, not the old 'No such file or directory'"
if [[ -x "${REPO_ROOT}/build/debug/src/chatbot_gui" ]]; then
    RUN_OUT="$(cd "${REPO_ROOT}/scripts" && DISPLAY=:99999 timeout 3 bash "$(basename "$TARGET")" 2>&1)"
    RUN_CODE=$?
    assert_not_contains "No such file or directory" "TD-044 regression: exec must find the real binary, not the old bare-build/src/ path"
    pgrep -f "chatbot_gui" >/dev/null 2>&1 && pkill -9 -f "chatbot_gui_binary" 2>/dev/null
else
    t "(skipped: build/debug/src/chatbot_gui not built in this checkout)"
fi

harness_report
exit $?
