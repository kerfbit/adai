#!/usr/bin/env bash

# @adai-status: beta        (hardcoded port list omits mns_server (8083) and trainer admin API (8084))
# @adai-version: 0.5.0
# @adai-reviewed: 2026-09-07


PORTS=(8080 8081 8082)

for port in "${PORTS[@]}"; do
    echo "=== Port $port ==="
    # Try ss first, fall back to netstat
    if command -v ss &>/dev/null; then
        result=$(ss -tlnp "sport = :$port" 2>/dev/null)
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
