#!/bin/bash
#
# Tests for scripts/run_chatbot_gui.sh (TD-044).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/run_chatbot_gui.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "src/chatbot_gui"

# ---------------------------------------------------------------------------
t "no DISPLAY -> clean, documented error before ever reaching the binary check"
raw="$(env -u DISPLAY bash "$TARGET" 2>&1)"
RUN_CODE=$?
RUN_OUT="$raw"
assert_exit 1
assert_contains "No graphical display found"

# ---------------------------------------------------------------------------
t "TD-044 regression: with DISPLAY set, runs correctly regardless of caller's CWD"
if [[ -x "${REPO_ROOT}/build/debug/src/chatbot_gui" ]]; then
    # vocab.txt genuinely doesn't exist in every checkout (it's a generated
    # artifact, not tracked) — the script's own "Continue anyway? (y/n)"
    # prompt would otherwise block waiting on stdin; answering "y" lets
    # execution continue far enough to print the resolved executable path
    # this test actually checks.
    raw="$(cd "${REPO_ROOT}/scripts" && echo y | DISPLAY=:99999 bash "$(basename "$TARGET")" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$raw"
    assert_not_contains "chatbot_gui not found" "must find the real build/debug/src/chatbot_gui, not fail with the old bare-build/ or CWD-dependent bug"
    assert_contains "${REPO_ROOT}/build/debug/src/chatbot_gui" "should report the real, resolved executable path"
    pkill -9 -f "chatbot_gui_binary" 2>/dev/null
else
    t "(skipped: build/debug/src/chatbot_gui not built in this checkout)"
fi

harness_report
exit $?
