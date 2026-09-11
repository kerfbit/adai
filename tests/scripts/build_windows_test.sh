#!/bin/bash
#
# Tests for scripts/build_windows.sh (TD-043).
#
# No CLI arguments. Its very first real check is `command -v
# x86_64-w64-mingw32-g++` — but mingw-w64 IS installed on machines that run
# this suite (confirmed: it's genuinely present here), so the "toolchain
# missing" branch can't be exercised by just invoking the script normally.
# Instead: a minimal stub PATH containing only the handful of external
# commands the script needs BEFORE that check (dirname, nproc — cd/pwd/echo/
# command are bash builtins, need no PATH entry) and deliberately omitting
# mingw's own directory, so `command -v` genuinely fails to find it — this
# exercises the real, unmodified script, just with a PATH that makes the
# "not installed" case reproducible on demand.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/build_windows.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "missing MinGW-w64 toolchain -> clean error, exit 1, install hints for 3 distros"
d="$(mktemp_dir)"
stub_bin="$d/stub_bin"
mkdir -p "$stub_bin"
for tool in dirname nproc; do
    real="$(command -v "$tool")"
    ln -s "$real" "$stub_bin/$tool"
done
bash_bin="$(command -v bash)"
raw="$(PATH="$stub_bin" "$bash_bin" "$TARGET" 2>&1)"
RUN_CODE=$?
RUN_OUT="$(printf '%s' "$raw" | sed -E 's/\x1b\[[0-9;]*m//g')"
assert_exit 1
assert_contains "MinGW-w64 toolchain not found"
assert_contains "apt-get install mingw-w64"
assert_contains "dnf install mingw64-gcc-c++"
assert_contains "pacman -S mingw-w64-gcc"

t "stub PATH sanity check: mingw really is unreachable in that PATH (test isn't a false pass)"
RUN_OUT="$(PATH="$stub_bin" command -v x86_64-w64-mingw32-g++ 2>&1)"
RUN_CODE=$?
assert_exit 1

harness_report
exit $?
