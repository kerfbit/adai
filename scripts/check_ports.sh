#!/usr/bin/env bash

# @adai-status: beta        (capped by TD-045 — see TECHNICAL_DEBT.md)
# @adai-version: 0.5.1
# @adai-reviewed: 2026-09-10


PORTS=(8080 8081 8082 8083 8084)

for port in "${PORTS[@]}"; do
    echo "=== Port $port ==="
    # Reset from the previous iteration — if neither ss nor netstat is
    # available below, result must read as empty for *this* port, not carry
    # over the last port that actually got a value.
    result=""
    # Try ss first, fall back to netstat
    #
    # ss -tlnp always prints its header row ("State Recv-Q Send-Q ...") even
    # when the sport filter matches nothing, so result was never actually
    # empty when ss was available — the "No service listening" branch and
    # the lsof fallback below were unreachable dead code on any port with no
    # listener, on essentially every modern Linux host (ss ships with
    # iproute2). Filter to LISTEN rows only, mirroring the netstat branch's
    # own grep-filtering, so an unmatched port genuinely yields empty result.
    if command -v ss &>/dev/null; then
        result=$(ss -tlnp "sport = :$port" 2>/dev/null | grep "LISTEN")
    elif command -v netstat &>/dev/null; then
        result=$(netstat -tlnp 2>/dev/null | grep ":$port ")
    fi

    if [[ -n "$result" ]]; then
        echo "$result"
        # Attempt to identify service by process name
        pid=$(echo "$result" | grep -oP 'pid=\K[0-9]+' || true)
        if [[ -n "$pid" ]]; then
            echo "Process: $(ps -p "$pid" -o comm= 2>/dev/null)"
        fi
    else
        # Also check with lsof as fallback
        if command -v lsof &>/dev/null; then
            lsof_result=$(lsof -iTCP:"$port" -sTCP:LISTEN 2>/dev/null)
            if [[ -n "$lsof_result" ]]; then
                echo "$lsof_result"
            else
                echo "No service listening on port $port"
            fi
        else
            echo "No service listening on port $port"
        fi
    fi
    echo
done
