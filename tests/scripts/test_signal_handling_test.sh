#!/bin/bash
#
# Tests for scripts/test_signal_handling.sh (TD-044).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/test_signal_handling.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

assert_find_build_dir_correct "$TARGET" "bin/chatbot_api_server"

# ---------------------------------------------------------------------------
t "TD-044 regression: gets past 'binary not found' into real server startup"
if [[ -x "${REPO_ROOT}/build/debug/bin/chatbot_api_server" ]]; then
    raw="$(timeout 15 bash "$TARGET" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$raw"
    assert_not_contains "No such file or directory" "must exec the real binary, not the old bare-build/bin/ path"
    pkill -9 -f "chatbot_api_server" 2>/dev/null
    rm -f /tmp/signal_test.log
else
    t "(skipped: build/debug/bin/chatbot_api_server not built in this checkout)"
fi

# ---------------------------------------------------------------------------
# TD-044 regression: the exit-code fix for GRACEFUL_SHUTDOWN. Exercised
# directly against a controllable substitute process rather than the real
# server (which this environment can't fully start without vocab.txt) —
# extracts just the verification logic (from "Verify process is no longer
# running" through the final exit) into an isolated snippet, run against a
# real, harmless `sleep` process this test spawns and owns itself. The
# script only ever inspects SERVER_PID via `kill -0`/`kill -9`, so it can't
# tell this isn't a real service.
extract_verification_snippet() {
    local out="$1"
    sed -n '/^# Verify process is no longer running/,$p' "$TARGET" > "$out"
}

t "graceful shutdown (process already gone) -> exit 0"
snippet="$(mktemp_dir)/verify.sh"
extract_verification_snippet "$snippet"
sleep 100 &
real_pid=$!
kill -9 "$real_pid" 2>/dev/null
wait "$real_pid" 2>/dev/null
run bash -c "SERVER_PID=$real_pid; source '$snippet'"
assert_exit 0
assert_contains "SUCCESS: Server shut down gracefully"

t "non-graceful shutdown (process still running, must be force-killed) -> exit 1"
snippet="$(mktemp_dir)/verify.sh"
extract_verification_snippet "$snippet"
sleep 100 &
real_pid=$!
run bash -c "SERVER_PID=$real_pid; source '$snippet'"
assert_exit 1
assert_contains "WARNING: Server is still running after SIGTERM"
if kill -0 "$real_pid" 2>/dev/null; then
    fail "the extracted snippet's own force-kill (kill -9) should have terminated $real_pid"
    kill -9 "$real_pid" 2>/dev/null
else
    pass
fi

harness_report
exit $?
