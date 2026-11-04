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
    # The script used to ask "Continue anyway? (y/n)" when vocab.txt was missing; since TD-260 the GUI
    # is a server client and the script asks nothing. The "y" on stdin is harmless.
    # The GUI doesn't exit on the bogus display, so bound it: timeout signals its whole process
    # group, the exec'd chatbot_gui_binary included. (A `pkill -f chatbot_gui_binary` afterward
    # also killed other tests' `nm`/`file` runs on that binary, and the GUI otherwise ran until
    # someone else's pkill ended it.)
    raw="$(cd "${REPO_ROOT}/scripts" && echo y | DISPLAY=:99999 timeout 10 bash "$(basename "$TARGET")" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$raw"
    assert_not_contains "chatbot_gui not found" "must find the real build/debug/src/chatbot_gui, not fail with the old bare-build/ or CWD-dependent bug"
    assert_contains "${REPO_ROOT}/build/debug/src/chatbot_gui" "should report the real, resolved executable path"
    assert_contains "Server:     http://localhost:8080" "TD-260: the GUI connects to chatbot_api_server"
    assert_not_contains "vocab.txt" "TD-260: the GUI no longer loads a local vocabulary"
else
    t "(skipped: build/debug/src/chatbot_gui not built in this checkout)"
fi

harness_report
exit $?
