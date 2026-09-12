#!/bin/bash
#
# Tests for scripts/verify_cli_parallel.sh (TD-044).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/verify_cli_parallel.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "bin/chatbot"

# ---------------------------------------------------------------------------
t "TD-044 regression: runs correctly regardless of caller's CWD"
run_in "${REPO_ROOT}/scripts" bash "$TARGET"
assert_not_contains "not found at" "must find the real build/debug/bin/chatbot, not fail with the old bare-build/ bug"
assert_contains "Binary found:"

t "reports the real, resolved binary path (not a stale bare build/bin/chatbot)"
if [[ -x "${REPO_ROOT}/build/debug/bin/chatbot" ]]; then
    assert_contains "${REPO_ROOT}/build/debug/bin/chatbot"
else
    t "(skipped: build/debug/bin/chatbot not built in this checkout)"
fi

harness_report
exit $?
