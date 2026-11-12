#!/bin/bash
#
# Smoke test for src/TrainerServiceMain.cpp: the real trainer_service binary supervising a stub
# incremental_trainer.
#
# trainer_service launches the incremental_trainer beside its own executable (TD-524), so the
# binary is copied into a temporary directory next to a stub script. The stub records each pass's
# command line and then does what the next line of the `plan` file says: a number is its exit code,
# "kill" kills itself with SIGKILL, "hang" writes one heartbeat and waits (a pass whose GPU call
# never returns). Each case runs one service instance with the admin API on and checks its log,
# /admin/status and its exit. Processes are stopped only by the PIDs this script started.
#
# Usage: trainer_service_main_test.sh <path to the built trainer_service>

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/harness.sh"

SVC_BIN="${1:-}"
if [[ -z "$SVC_BIN" || ! -x "$SVC_BIN" ]]; then
    t "(skipped: trainer_service not built)"
    harness_report
    exit $?
fi
for tool in curl python3; do
    if ! command -v "$tool" >/dev/null; then
        t "(skipped: $tool not installed)"
        harness_report
        exit $?
    fi
done

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/adai_trainer_service_smoke.XXXXXX")"
SVC_PID=""

stop_service() {
    if [[ -n "$SVC_PID" ]] && kill -0 "$SVC_PID" 2>/dev/null; then
        kill -TERM "$SVC_PID"
        for _ in $(seq 1 120); do
            kill -0 "$SVC_PID" 2>/dev/null || break
            sleep 0.25
        done
    fi
    wait "$SVC_PID" 2>/dev/null
    SVC_EXIT=$?
    SVC_PID=""
}
cleanup() {
    stop_service
    rm -rf "$ROOT"
}
trap cleanup EXIT

free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# setup NAME PLAN STALL_MINUTES [ADMIN_ENABLED] — a fresh directory with the binary, the stub, the
# plan (one action per pass, newline-separated) and a config; sets DIR and PORT.
setup() {
    DIR="$ROOT/$1"
    mkdir -p "$DIR/bin" "$DIR/sessions"
    cp "$SVC_BIN" "$DIR/bin/trainer_service"
    printf '%b\n' "$2" >"$DIR/plan"
    cat >"$DIR/bin/incremental_trainer" <<'STUB'
#!/bin/bash
# Stub training pass: record the command line, then follow the next line of ../plan.
base="$(cd "$(dirname "$0")/.." && pwd)"
echo "$$ $*" >>"$base/passes.log"
n=$(wc -l <"$base/passes.log")
action=$(sed -n "${n}p" "$base/plan")
case "${action:-3}" in
    hang)
        hb=""
        while [[ $# -gt 0 ]]; do [[ "$1" == "--heartbeat-file" ]] && hb="$2"; shift; done
        [[ -n "$hb" ]] && printf '%s 1\n' $$ >"$hb.tmp" && mv "$hb.tmp" "$hb"
        exec sleep 600
        ;;
    kill) kill -KILL $$ ;;
    *) exit "${action:-3}" ;;
esac
STUB
    chmod +x "$DIR/bin/incremental_trainer"
    PORT=$(free_port)
    local child_port
    child_port=$(free_port)
    cat >"$DIR/config.trainer.conf" <<CONF
NAME_SERVICE_URL=
REGISTRY_SERVER_URL=
METRICS_SERVER_URL=
LOG_FILE_PATH=
SESSION_DIR=$DIR/sessions
TRAINER_ADMIN_ENABLED=${4:-true}
TRAINER_ADMIN_HOST=127.0.0.1
TRAINER_ADMIN_PORT=$PORT
TRAINER_CHILD_ADMIN_PORT=$child_port
TRAINER_ADMIN_DIR=$DIR/admin
TRAINER_PASS_STALL_MINUTES=$3
CONF
}

start_service() {
    cd "$DIR" || return 1
    "$DIR/bin/trainer_service" --config "$DIR/config.trainer.conf" >"$DIR/service.log" 2>&1 &
    SVC_PID=$!
    cd - >/dev/null || return 1
}

# status FIELD — one field of /admin/status, or "" if the service doesn't answer.
status() {
    curl -s --max-time 2 "http://127.0.0.1:$PORT/admin/status" |
        python3 -c 'import json, sys
try:
    v = json.load(sys.stdin).get(sys.argv[1], "")
    print(str(v).lower() if isinstance(v, bool) else v)
except Exception:
    print("")' "$1"
}

# wait_for FIELD VALUE SECONDS — waits until /admin/status reports FIELD = VALUE.
wait_for() {
    local deadline=$((SECONDS + $3))
    while ((SECONDS < deadline)); do
        [[ "$(status "$1")" == "$2" ]] && return 0
        sleep 0.25
    done
    return 1
}

post() {
    curl -s -o /dev/null -w '%{http_code}' --max-time 5 -X POST -H 'Content-Type: application/json' -d '{}' \
        "http://127.0.0.1:$PORT$1"
}

# ── A pass that trained, then one that found nothing ─────────────────────────────────────────────
setup worked_then_idle '0\n3' 30
start_service
t "a pass that trained is relaunched at once, and an idle one is counted as neither failure"
if wait_for total_passes_launched 2 15; then pass; else fail "two passes weren't launched: $(cat "$DIR/service.log")"; fi
wait_for last_exit_code 3 5
assert_eq 1 "$(status total_passes_did_work)" "did work"
assert_eq 0 "$(status total_passes_failed)" "failed"
assert_eq 0 "$(status consecutive_failures)" "consecutive failures"
assert_eq -1 "$(status seconds_since_pass_progress)" "no pass running"

t "each pass gets the objective, pool, heartbeat file, admin port and resume"
RUN_OUT="$(head -1 "$DIR/passes.log")"
assert_contains "--objective=chatbot"
assert_contains "--dataset-kind chatbot"
assert_contains "--heartbeat-file $DIR/sessions/trainer_service_pass_chatbot.heartbeat"
assert_contains "--foreground --admin-port"
assert_contains " resume"

t "SIGTERM while idle stops the service with exit 0"
stop_service
assert_eq 0 "$SVC_EXIT" "trainer_service exit code"
RUN_OUT="$(cat "$DIR/service.log")"
assert_contains "stop requested"

# ── A failing pass ───────────────────────────────────────────────────────────────────────────────
setup failing '1' 30
start_service
t "a pass that exits 1 is counted as failed and backs off"
if wait_for total_passes_failed 1 15; then pass; else fail "no failure counted: $(cat "$DIR/service.log")"; fi
assert_eq 1 "$(status consecutive_failures)" "consecutive failures"
assert_eq 1 "$(status last_exit_code)" "last exit code"
assert_ne 0 "$(status last_failure_unix)" "last failure time"
RUN_OUT="$(cat "$DIR/service.log")"
assert_contains "training pass failed with exit code 1"
stop_service

# ── A pass killed by a signal ────────────────────────────────────────────────────────────────────
setup killed 'kill' 30
start_service
t "a pass killed by a signal is counted as crashed with 128 + the signal"
if wait_for total_passes_crashed 1 15; then pass; else fail "no crash counted: $(cat "$DIR/service.log")"; fi
assert_eq 137 "$(status last_exit_code)" "SIGKILL"
assert_eq 0 "$(status total_passes_failed)" "not also counted as failed"
stop_service

# ── Pause and resume ─────────────────────────────────────────────────────────────────────────────
setup paused '3' 30
start_service
t "pause stops further passes; resume starts one at once"
wait_for last_exit_code 3 15
assert_eq 202 "$(post /admin/pause)" "pause"
assert_eq true "$(status supervisor_paused)" "paused"
launched="$(status total_passes_launched)"
assert_eq 202 "$(post /admin/resume)" "resume"
if wait_for total_passes_launched $((launched + 1)) 10; then pass; else fail "resume didn't start a pass"; fi
assert_eq false "$(status supervisor_paused)" "resumed"
stop_service

# ── A pass that stops making progress (TD-569) ───────────────────────────────────────────────────
setup stalled 'hang' 1
start_service
t "while a pass runs, the time since its progress is reported"
sleep 4
since="$(status seconds_since_pass_progress)"
if [[ "$since" =~ ^[0-9]+$ ]] && ((since >= 1)); then pass; else fail "seconds_since_pass_progress was '$since'"; fi
t "a pass with no progress for TRAINER_PASS_STALL_MINUTES is stopped and counted as stalled"
if wait_for total_passes_stalled 1 90; then pass; else fail "no stall counted: $(cat "$DIR/service.log")"; fi
assert_eq 1 "$(status consecutive_failures)" "a stall counts as a failure"
assert_eq 0 "$(status total_passes_failed)" "not also counted as failed"
assert_eq 0 "$(status total_passes_crashed)" "not also counted as crashed"
assert_eq 143 "$(status last_exit_code)" "stopped with SIGTERM"
RUN_OUT="$(cat "$DIR/service.log")"
assert_contains "has made no progress for"
stop_service

# ── SIGTERM during a pass ────────────────────────────────────────────────────────────────────────
setup stop_mid_pass 'hang' 30
start_service
t "SIGTERM during a pass stops the pass and the service"
for _ in $(seq 1 40); do [[ -s "$DIR/passes.log" ]] && break; sleep 0.25; done
pass_pid="$(cut -d' ' -f1 "$DIR/passes.log" | head -1)"
stop_service
assert_eq 0 "$SVC_EXIT" "trainer_service exit code"
if [[ -n "$pass_pid" ]] && ! kill -0 "$pass_pid" 2>/dev/null; then pass; else fail "the pass ($pass_pid) is still running"; fi

# ── The stall check off, the admin API off ───────────────────────────────────────────────────────
setup options_off '3' 0 false
start_service
t "with TRAINER_PASS_STALL_MINUTES=0 and the admin API off, neither flag is passed"
for _ in $(seq 1 40); do [[ -s "$DIR/passes.log" ]] && break; sleep 0.25; done
RUN_OUT="$(cat "$DIR/passes.log")"
assert_not_contains "--heartbeat-file"
assert_not_contains "--admin-port"
assert_contains "--foreground resume"
stop_service
RUN_OUT="$(cat "$DIR/service.log")"
assert_contains "TRAINER_PASS_STALL_MINUTES=0"

harness_report
exit $?
