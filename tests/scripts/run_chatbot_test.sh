#!/bin/bash
#
# Tests for scripts/run_chatbot.sh (TD-044).
#
# This script always ends by exec'ing/running the CLI client against
# whatever server it started or found — never returns control on its own
# in the success path, and needs a real server to talk to. What's verified
# here, safely: it resolves both binaries to the real build/debug output
# (not the old bare build/ path), and its own pre-flight "binary not
# found" checks behave correctly against a genuinely empty build tree.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/run_chatbot.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "bin/chatbot_api_server"

# ---------------------------------------------------------------------------
t "TD-044 regression: CLIENT_BIN/SERVER_BIN resolve to the real build/debug output, not bare build/"
if [[ -x "${REPO_ROOT}/build/debug/bin/chatbot" && -x "${REPO_ROOT}/build/debug/bin/chatbot_api_server" ]]; then
    # Extracts just the resolution logic (defaults through BUILD_DIR/
    # CLIENT_BIN/SERVER_BIN assignment) rather than running the whole
    # script, which would otherwise try to actually start a server and
    # launch an interactive client — far beyond what a path-resolution
    # regression test needs. Written alongside the real script (not to an
    # unrelated scratch dir) since SCRIPT_DIR/REPO_ROOT are computed from
    # the running file's own location — anywhere else would resolve a
    # different (wrong) REPO_ROOT. Removed immediately after.
    end_line="$(grep -n '^SERVER_BIN=' "$TARGET" | head -1 | cut -d: -f1)"
    snippet="${REPO_ROOT}/scripts/.td044_resolve_test_run_chatbot.sh"
    head -n "$end_line" "$TARGET" > "$snippet"
    echo 'echo "CLIENT_BIN=$CLIENT_BIN"; echo "SERVER_BIN=$SERVER_BIN"' >> "$snippet"
    run bash "$snippet"
    rm -f "$snippet"
    assert_contains "CLIENT_BIN=${REPO_ROOT}/build/debug/bin/chatbot"
    assert_contains "SERVER_BIN=${REPO_ROOT}/build/debug/bin/chatbot_api_server"
else
    t "(skipped: build/debug binaries not built in this checkout)"
fi

t "clean, informative error when the client binary genuinely doesn't exist anywhere"
d="$(mktemp_dir)"
mkdir -p "$d/scripts"
cp "$TARGET" "$d/scripts/run_chatbot.sh"
run bash "$d/scripts/run_chatbot.sh"
assert_exit 1
assert_contains "Chatbot client binary not found at:"

harness_report
exit $?
