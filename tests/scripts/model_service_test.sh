#!/bin/bash
#
# Tests for scripts/model_service.sh (TD-043).
#
# --pidfile/--logfile are always pointed at a scratch path so this suite
# never touches the real default paths (/tmp/adai_model_service.{pid,log})
# a real developer might actually be using. `start` is deliberately never
# invoked for real — it would launch a genuine, long-running
# chatbot_api_server bound to a real port. `stop`/`status` are instead
# exercised against a harmless `sleep` process this suite spawns and owns
# itself, which the script only ever inspects via `kill -0 <pid>` /
# `kill -TERM <pid>` — it has no way to tell that isn't a real service.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/model_service.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "help / -h / --help all exit 0"
run bash "$TARGET" help
assert_exit 0
assert_contains "ADAI Model Service Manager"
run bash "$TARGET" -h
assert_exit 0
run bash "$TARGET" --help
assert_exit 0

t "no command -> defaults to help (documented default), exit 0"
run bash "$TARGET"
assert_exit 0

t "unknown command -> exit 1"
run bash "$TARGET" totally-bogus-command
assert_exit 1
assert_contains "Unknown command: 'totally-bogus-command'"

t "unknown option -> exit 1"
run bash "$TARGET" status --totally-bogus-option
assert_exit 1
assert_contains "Unknown option: --totally-bogus-option"

# ---------------------------------------------------------------------------
d="$(mktemp_dir)"
pidfile="$d/service.pid"
logfile="$d/service.log"

t "status: no PID file -> STOPPED, exit 1"
run bash "$TARGET" status --pidfile "$pidfile" --logfile "$logfile"
assert_exit 1
assert_contains "STOPPED"
assert_contains "no PID file"

t "status: stale PID file (dead process) -> DEAD, exit 1, removes the stale file"
echo 999999 > "$pidfile"
run bash "$TARGET" status --pidfile "$pidfile" --logfile "$logfile"
assert_exit 1
assert_contains "DEAD"
assert_file_not_exists "$pidfile"

t "status: PID file pointing to a real (harmless, self-owned) process -> RUNNING, exit 0"
sleep 100 &
real_pid=$!
echo "$real_pid" > "$pidfile"
run bash "$TARGET" status --pidfile "$pidfile" --logfile "$logfile" --port 18099
assert_exit 0
assert_contains "RUNNING"
assert_contains "PID     : ${real_pid}"
kill -9 "$real_pid" 2>/dev/null
wait "$real_pid" 2>/dev/null
rm -f "$pidfile"

# ---------------------------------------------------------------------------
t "stop: no PID file -> safe no-op, exit 0"
run bash "$TARGET" stop --pidfile "$pidfile" --logfile "$logfile"
assert_exit 0
assert_contains "may not be running"

t "stop: stale PID file (dead process) -> removes it, exit 0"
echo 999999 > "$pidfile"
run bash "$TARGET" stop --pidfile "$pidfile" --logfile "$logfile"
assert_exit 0
assert_contains "stale PID file"
assert_file_not_exists "$pidfile"

t "stop: real (self-owned) process -> sends SIGTERM, waits, confirms exit, removes PID file"
sleep 100 &
real_pid=$!
echo "$real_pid" > "$pidfile"
run bash "$TARGET" stop --pidfile "$pidfile" --logfile "$logfile"
assert_exit 0
assert_contains "Sending SIGTERM to PID ${real_pid}"
assert_contains "Service stopped (was PID ${real_pid})"
assert_file_not_exists "$pidfile"
if kill -0 "$real_pid" 2>/dev/null; then
    fail "process $real_pid should have been terminated by 'stop'"
    kill -9 "$real_pid" 2>/dev/null
else
    pass
fi

# ---------------------------------------------------------------------------
t "logs: missing log file -> exit 1"
run bash "$TARGET" logs --pidfile "$pidfile" --logfile "$d/does-not-exist.log"
assert_exit 1
assert_contains "Log file not found"

# ---------------------------------------------------------------------------
t "health: nothing listening on the given port -> exit 1, clean message"
run bash "$TARGET" health --port 18098 --pidfile "$pidfile" --logfile "$logfile"
assert_exit 1
assert_contains "did not respond"

# ---------------------------------------------------------------------------
# TD-043 regression: get_binary()/get_build_dir()'s "debug" case pointed at
# bare "${REPO_ROOT}/build" instead of "${REPO_ROOT}/build/debug" — every
# preset in this repo (per CLAUDE.md) builds into "build/<preset>/", the
# same convention "release" already correctly followed here. This made
# `--build-type debug` completely non-functional: even a fully-configured,
# already-built build/debug was reported as "not configured". Found writing
# this test suite; only run for real if this checkout actually has a
# configured build/debug to build against (keeps this suite runnable
# without requiring a live cmake build first).
t "build --build-type debug finds the real build/debug directory (TD-043 regression)"
if [[ -f "${REPO_ROOT}/build/debug/CMakeCache.txt" ]]; then
    run bash "$TARGET" build --build-type debug --pidfile "$pidfile" --logfile "$logfile"
    assert_exit 0
    assert_contains "Binary ready: ${REPO_ROOT}/build/debug/bin/chatbot_api_server"
    assert_not_contains "is not configured"
else
    t "(skipped: build/debug not configured in this checkout)"
fi

harness_report
exit $?
