#!/bin/bash
#
# Tests for scripts/package_windows.sh (TD-043).
#
# No CLI arguments (confirmed by reading it). BUILD_DIR="$PROJECT_DIR/
# build-windows" is hardcoded with no override. Safely testable: its first
# preflight check refuses cleanly when build-windows/ doesn't exist, before
# touching anything (the "Clean previous package" rm -rf on PACKAGE_DIR is
# further down, never reached when this check fails).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/package_windows.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "preflight refuses cleanly when build-windows/ doesn't exist, makes no changes"
if [[ -d "${REPO_ROOT}/build-windows" ]]; then
    t "(skipped: build-windows/ exists in this checkout — preflight would pass this check)"
else
    run bash "$TARGET"
    assert_exit 1
    assert_contains "Build directory not found"
    assert_contains "Run ./scripts/build_windows.sh first"
    # Confirms it exits before ever reaching "Cleaning previous package..."
    assert_not_contains "Cleaning previous package"
fi

harness_report
exit $?
