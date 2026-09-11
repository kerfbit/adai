#!/bin/bash
#
# Tests for scripts/package-sycl.sh (TD-043).
#
# This script takes no CLI arguments at all (confirmed by reading it — no
# `case "$1"`/getopts anywhere) and hardcodes BUILD_DIR="$PROJECT_ROOT/build/
# sycl" with no override. What's safely testable without an actual SYCL
# build is exactly its preflight guard: it refuses cleanly, before doing
# anything else (rm -rf/mkdir/tar), when build/sycl/bin/{incremental_trainer,
# chatbot_api_server} aren't both present — true in any checkout that hasn't
# built the sycl preset, verified below before relying on it.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/package-sycl.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "preflight refuses cleanly when the sycl build doesn't exist, makes no changes"
if [[ -x "${REPO_ROOT}/build/sycl/bin/incremental_trainer" && \
      -x "${REPO_ROOT}/build/sycl/bin/chatbot_api_server" ]]; then
    t "(skipped: build/sycl is actually built in this checkout — preflight would pass)"
else
    run bash "$TARGET"
    assert_exit 1
    assert_contains "Build the sycl preset first"
    assert_contains "cmake --preset sycl"
    # Confirms it exits at the very first preflight check, before ever
    # reaching "rm -rf \"$STAGE_DIR\"" further down.
    assert_not_contains "=== Packaging"
fi

harness_report
exit $?
