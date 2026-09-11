#!/bin/bash
#
# Shared assertion harness for the scripts/ test suite (TD-043).
#
# Each <name>_test.sh file sources this, calls t()/run()/assert_*() as many
# times as needed, then finishes with `harness_report` (its exit code is the
# test file's own exit code, so ctest sees pass/fail correctly).
#
# Deliberately plain bash (no bats-core, no external test framework): none of
# that tooling is installed or otherwise part of this project's toolchain,
# and every other test in this repo is either C++/GTest or a script invoked
# directly — a dependency-free harness matches that and needs no CI changes
# to install anything new.

HARNESS_PASS=0
HARNESS_FAIL=0
HARNESS_CURRENT="(unnamed test)"

# t DESCRIPTION — label the test about to run, used in failure messages.
t() {
    HARNESS_CURRENT="$1"
}

# run CMD [ARGS...] — runs CMD, capturing combined stdout+stderr into RUN_OUT
# and its exit code into RUN_CODE. Never itself fails the harness (a command
# that exits non-zero is frequently the exact thing under test).
# Strips ANSI SGR escape codes (the \033[...m color sequences nearly every
# script here uses for RED/GREEN/YELLOW/BLUE output) so assert_contains can
# match plain substrings like "TODO markers:  0" without the exact color
# codes the script happens to interleave in between getting in the way.
_strip_ansi() {
    sed -E 's/\x1b\[[0-9;]*m//g'
}

run() {
    # Capture output and exit code together FIRST, with no pipe in between —
    # `"$@" 2>&1 | _strip_ansi` would make $? reflect _strip_ansi's (sed's)
    # exit status, not the command under test's, silently always reporting 0
    # regardless of what "$@" actually did. Stripping ANSI codes is a
    # separate step afterward, on the already-captured string, so it can
    # never affect RUN_CODE.
    local raw
    raw="$("$@" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$(printf '%s' "$raw" | _strip_ansi)"
}

# run_in DIR CMD [ARGS...] — like run(), but executes CMD with its CWD set to
# DIR. NOT `(cd "$DIR" && run ...)` — that runs run() itself inside a
# subshell, so RUN_OUT/RUN_CODE never escape back to the caller. Here the
# subshell only wraps the `cd && "$@"`, while the assignment and `$?` happen
# in the current shell, same as plain run().
run_in() {
    local dir="$1"; shift
    local raw
    raw="$(cd "$dir" && "$@" 2>&1)"
    RUN_CODE=$?
    RUN_OUT="$(printf '%s' "$raw" | _strip_ansi)"
}

# run_stdin CMD [ARGS...] — like run(), but feeds /dev/null on stdin so a
# script that would otherwise block on an interactive prompt (e.g. a
# confirm() helper reading from stdin) fails fast instead of hanging the
# test suite.
run_stdin_devnull() {
    RUN_OUT="$("$@" < /dev/null 2>&1)"
    RUN_CODE=$?
}

pass() {
    HARNESS_PASS=$((HARNESS_PASS + 1))
}

fail() {
    HARNESS_FAIL=$((HARNESS_FAIL + 1))
    echo "FAIL [${HARNESS_CURRENT}]: $1" >&2
}

assert_eq() {
    local expected="$1" actual="$2" msg="${3:-}"
    if [[ "$expected" == "$actual" ]]; then
        pass
    else
        fail "expected '${expected}', got '${actual}'${msg:+ ($msg)}"
    fi
}

assert_ne() {
    local not_expected="$1" actual="$2" msg="${3:-}"
    if [[ "$not_expected" != "$actual" ]]; then
        pass
    else
        fail "expected value other than '${not_expected}'${msg:+ ($msg)}"
    fi
}

# assert_exit EXPECTED — checks RUN_CODE from the most recent run()/run_stdin_devnull().
assert_exit() {
    assert_eq "$1" "$RUN_CODE" "exit code; output was: ${RUN_OUT}"
}

assert_contains() {
    local needle="$1" msg="${2:-}"
    if [[ "$RUN_OUT" == *"$needle"* ]]; then
        pass
    else
        fail "output did not contain '${needle}'${msg:+ ($msg)} — got: ${RUN_OUT}"
    fi
}

assert_not_contains() {
    local needle="$1" msg="${2:-}"
    if [[ "$RUN_OUT" != *"$needle"* ]]; then
        pass
    else
        fail "output unexpectedly contained '${needle}'${msg:+ ($msg)} — got: ${RUN_OUT}"
    fi
}

assert_file_exists() {
    if [[ -e "$1" ]]; then
        pass
    else
        fail "expected path to exist: $1"
    fi
}

assert_file_not_exists() {
    if [[ ! -e "$1" ]]; then
        pass
    else
        fail "expected path NOT to exist: $1"
    fi
}

assert_true() {
    if [[ "$1" == true ]]; then pass; else fail "${2:-expected true, got $1}"; fi
}

assert_false() {
    if [[ "$1" == false ]]; then pass; else fail "${2:-expected false, got $1}"; fi
}

# harness_report — prints the summary line and returns 0 iff every assertion
# in this test file passed. The caller does `harness_report; exit $?` (or
# just `exec` this as the file's last statement) so ctest reads the correct
# pass/fail status.
harness_report() {
    echo ""
    if [[ "$HARNESS_FAIL" -eq 0 ]]; then
        echo "==== $(basename "$0"): ${HARNESS_PASS} passed ===="
        return 0
    else
        echo "==== $(basename "$0"): ${HARNESS_PASS} passed, ${HARNESS_FAIL} FAILED ===="
        return 1
    fi
}

# mktemp_dir — a scratch directory under the harness's own control, always
# under the system temp dir, auto-registered for cleanup via a trap so a
# test file doesn't have to remember to clean up on every exit path
# (including an assertion failure that doesn't itself exit).
HARNESS_SCRATCH_DIRS=()
mktemp_dir() {
    local d
    d="$(mktemp -d "${TMPDIR:-/tmp}/adai_scripts_test.XXXXXX")"
    HARNESS_SCRATCH_DIRS+=("$d")
    echo "$d"
}

harness_cleanup() {
    local d
    for d in "${HARNESS_SCRATCH_DIRS[@]:-}"; do
        [[ -n "$d" && -d "$d" ]] && rm -rf "$d"
    done
}
trap harness_cleanup EXIT
