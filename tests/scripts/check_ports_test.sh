#!/bin/bash
#
# Tests for scripts/check_ports.sh (TD-045).
#
# Purely read-only (queries ss/netstat/lsof, never modifies anything), so
# the real script is invoked directly with no isolation needed.
#
# The port-list gap TD-045 originally described (missing 8083/8084) was
# already fixed separately under TD-121 (commit 2729625) — confirmed by
# reading the current file, which already lists all 5 ports. This test
# pins that down as a permanent regression check rather than re-fixing
# something already fixed.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/check_ports.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "TD-121 regression: port list includes all 5 ADAI service ports, not just the original 3"
run bash "$TARGET"
assert_exit 0
for port in 8080 8081 8082 8083 8084; do
    assert_contains "=== Port ${port} ===" "port ${port} should be checked"
done

t "runs to completion without crashing regardless of what's actually listening"
# Already proven by the exit-0 check above, but assert explicitly on the
# per-port structure too: exactly 5 "=== Port" headers, one per port, no
# more and no fewer (catches an accidental duplicate/dropped port in the
# array without hardcoding assumptions about which tool — ss/netstat/lsof —
# is available on the machine running this test).
header_count=$(printf '%s\n' "$RUN_OUT" | grep -c "^=== Port ")
assert_eq "5" "$header_count" "expected exactly 5 '=== Port N ===' headers"

harness_report
exit $?
