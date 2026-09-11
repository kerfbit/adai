#!/bin/bash
#
# Tests for scripts/run_tests.sh (TD-043).
#
# Deliberately does NOT invoke this script with no flags, or with
# --asan/--ubsan/--tsan/--coverage: BUILD_DIR is hardcoded to
# "$PROJECT_ROOT/build" with no override (confirmed by reading the script),
# and any of those paths does `rm -rf "$BUILD_DIR"` — running that for real
# here would delete this checkout's actual build directory. What IS safe and
# useful to test is exactly what TD-043 asks for: argument parsing and error
# handling — the unknown-option branch runs entirely before that rm -rf, so
# it can't reach it.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/run_tests.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "unknown option rejected before any BUILD_DIR mutation, exit 1"
run bash "$TARGET" --totally-bogus-flag
assert_exit 1
assert_contains "Unknown option: --totally-bogus-flag"
assert_contains "Usage:"

# ---------------------------------------------------------------------------
t "the real build directory must be untouched by the unknown-option run above"
# BUILD_DIR="$PROJECT_ROOT/build" per the script's own hardcoded computation
# (SCRIPT_DIR two levels up from tests/scripts/, matching this test file's
# own REPO_ROOT). If the unknown-option guard were ever moved after the
# rm -rf (a regression), this would catch it by simply still finding the
# real build tree intact.
assert_file_exists "${REPO_ROOT}/build"

# ---------------------------------------------------------------------------
# Verifying each flag sets the right internal variable (SANITIZER/COVERAGE/
# VERBOSE) genuinely needs to exercise the real parsing loop — but the lines
# immediately after it are the dangerous rm -rf "$BUILD_DIR" block, so we
# can't run the real file past that point. Instead: copy the script verbatim
# up through the end of its argument-parsing `while` loop (line 55 — "done"
# — found via grep, not hardcoded blind; re-checked below) into a scratch
# file with one harmless diagnostic line appended, and run THAT. This is the
# real, unmodified parsing code, not a reimplementation — it just never
# reaches the dangerous block because that block was never copied in.
t "extracting run_tests.sh's argument-parsing block for isolated testing"
parse_end_line="$(grep -n '^done$' "$TARGET" | head -1 | cut -d: -f1)"
if [[ -z "$parse_end_line" ]]; then
    fail "could not find the arg-parsing loop's closing 'done' in $TARGET — script structure changed?"
else
    pass
fi

d="$(mktemp_dir)"
harness_extract="$d/run_tests_parse_only.sh"
head -n "$parse_end_line" "$TARGET" > "$harness_extract"
echo 'echo "PARSED: SANITIZER=${SANITIZER} COVERAGE=${COVERAGE} VERBOSE=${VERBOSE}"' >> "$harness_extract"

t "no flags -> all defaults empty/false"
run bash "$harness_extract"
assert_contains "PARSED: SANITIZER= COVERAGE=false VERBOSE=false"

t "--asan sets SANITIZER=asan"
run bash "$harness_extract" --asan
assert_contains "PARSED: SANITIZER=asan COVERAGE=false VERBOSE=false"

t "--ubsan sets SANITIZER=ubsan"
run bash "$harness_extract" --ubsan
assert_contains "PARSED: SANITIZER=ubsan COVERAGE=false VERBOSE=false"

t "--tsan sets SANITIZER=tsan"
run bash "$harness_extract" --tsan
assert_contains "PARSED: SANITIZER=tsan COVERAGE=false VERBOSE=false"

t "--coverage sets COVERAGE=true"
run bash "$harness_extract" --coverage
assert_contains "PARSED: SANITIZER= COVERAGE=true VERBOSE=false"

t "--verbose sets VERBOSE=true"
run bash "$harness_extract" --verbose
assert_contains "PARSED: SANITIZER= COVERAGE=false VERBOSE=true"

t "flags combine: --asan --coverage --verbose all apply together"
run bash "$harness_extract" --asan --coverage --verbose
assert_contains "PARSED: SANITIZER=asan COVERAGE=true VERBOSE=true"

harness_report
exit $?
