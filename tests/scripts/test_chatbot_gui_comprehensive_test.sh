#!/bin/bash
#
# Tests for scripts/test_chatbot_gui_comprehensive.sh (TD-044).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/test_chatbot_gui_comprehensive.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "src/chatbot_gui_binary"

# ---------------------------------------------------------------------------
t "TD-044 regression: runs correctly regardless of caller's CWD"
run_in "${REPO_ROOT}/scripts" bash "$TARGET"
# Before the fix this failed 11 of the binary/linkage/symbol/artifact tests
# purely because it never found the real build/debug binaries at all.
assert_not_contains "❌ FAIL: Executable not found" "must find the real build, not fail with the old bare-build/ bug"

t "TD-044 regression: docs/guides/ path corrected to docs/operations/guides/"
run_in "${REPO_ROOT}/scripts" bash "$TARGET"
if [[ -f "${REPO_ROOT}/docs/operations/guides/chatbot-gui-guide.md" ]]; then
    assert_contains "GUI guide documentation exists"
    assert_not_contains "GUI guide documentation not found"
else
    t "(skipped: docs/operations/guides/chatbot-gui-guide.md doesn't exist in this checkout)"
fi

t "with a real, complete build, only genuinely-missing content (not stale paths) is reported failed"
if [[ -x "${REPO_ROOT}/build/debug/src/chatbot_gui_binary" ]]; then
    run_in "${REPO_ROOT}/scripts" bash "$TARGET"
    fail_count="$(printf '%s\n' "$RUN_OUT" | grep -oP 'Failed:\s*\K\d+')"
    # Not asserting 0 here: CHATBOT_GUI_README.md is a real, currently-
    # missing file (confirmed via git log — deleted, never renamed), a
    # genuine content gap outside the scope of this build-path fix. Only
    # asserting the count is *small* (down from the pre-fix 11) rather
    # than exactly a moving target.
    if [[ -n "$fail_count" ]] && (( fail_count <= 2 )); then
        pass
    else
        fail "expected at most 2 genuine failures (the known-missing README), got ${fail_count:-<unparsed>}: $RUN_OUT"
    fi
else
    t "(skipped: build/debug/src/chatbot_gui_binary not built in this checkout)"
fi

harness_report
exit $?
