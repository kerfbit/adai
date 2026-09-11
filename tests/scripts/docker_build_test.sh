#!/bin/bash
#
# Tests for scripts/docker_build.sh (TD-043).
#
# Does not actually invoke `docker build` (no assumption docker is
# available/usable in the test environment, and a real build is slow and
# network-dependent) — covers argument parsing and error handling only,
# which is exactly the surface TD-043 asks for.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/docker_build.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in -t --tag -n --name --no-cache --platform -h --help; do
    assert_contains "$flag"
done

# ---------------------------------------------------------------------------
# TD-043 regression: usage() used to hardcode `exit 0` internally, so the
# unknown-option branch (which also calls it, after printing an error)
# reported success for a run that never built anything — e.g. a typo'd flag
# silently did nothing while looking like it passed to any caller (CI job,
# wrapper script) that only checked $?. Found writing this test suite.
t "unknown flag -> exit 1 (TD-043 regression: used to wrongly exit 0)"
run bash "$TARGET" --this-flag-does-not-exist
assert_exit 1
assert_contains "Unknown option: --this-flag-does-not-exist"

t "-h and --help both still exit 0 (the fix must not have broken the real help path)"
run bash "$TARGET" -h
assert_exit 0
run bash "$TARGET" --help
assert_exit 0

harness_report
exit $?
