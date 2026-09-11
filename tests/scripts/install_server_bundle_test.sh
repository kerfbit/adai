#!/bin/bash
#
# Tests for scripts/install_server_bundle.sh (TD-043 priority item).
#
# Everything below runs as a non-root user against the real, unmodified
# script — safe because every case tested here (bad flag value, unknown
# flag, --help) exits during argument parsing or the EUID check, both of
# which run *before* anything in preflight_checks() that would touch the
# filesystem or require root (binary presence checks, --storage-backend
# validation, systemd unit installation). Confirmed by reading the script:
# preflight_checks()'s root check is its first statement.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/install_server_bundle.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --install-path --user --group --build-dir --mns-port --registry-port \
            --metrics-port --metrics-dir --registry-data-dir --mns-data-dir \
            --storage-backend --db-url --setup-postgres --pg-db-name --pg-db-user \
            --wipe-data --yes --help; do
    assert_contains "$flag" "help text should document $flag"
done

# ---------------------------------------------------------------------------
t "unknown flag -> exit 1, shows help"
run bash "$TARGET" --this-flag-does-not-exist
assert_exit 1
assert_contains "Unknown option: --this-flag-does-not-exist"

# ---------------------------------------------------------------------------
# validate_abs_path-backed flags: relative path must be rejected before root
# is ever checked, and — the TD-043 regression this pins down — a genuinely
# ordinary absolute path must NOT be rejected. Before this fix, EVERY one of
# these flags rejected every absolute path with a false "illegal characters"
# error: validate_abs_path's `[[ "${val}" =~ $'\n' || "${val}" =~ $'\0' ]]`
# check relied on `$'\0'`, which bash evaluates to the empty string (a shell
# variable can never hold an actual NUL byte) — matching an empty pattern is
# always true, so that whole condition fired unconditionally. Found writing
# this test suite; confirmed identically broken in 4 sibling install
# scripts, all now fixed alongside this one.
for flag in --install-path --metrics-dir --registry-data-dir --mns-data-dir; do
    t "$flag rejects a relative path"
    run bash "$TARGET" "$flag" "relative/path" --yes
    assert_exit 1
    assert_contains "${flag}: 'relative/path' must be an absolute path"

    t "$flag regression: an ordinary absolute path must be accepted, not rejected as 'illegal characters'"
    run bash "$TARGET" "$flag" "/tmp/adai-td043-regress" --yes
    assert_not_contains "illegal characters" "regression: a normal absolute path must never hit this branch"
    # Should proceed all the way to the (safe, non-root) EUID guard, not
    # bail out during argument validation.
    assert_contains "Installation requires root privileges"

    t "$flag rejects a path containing an embedded newline (the real, correct half of the check)"
    run bash "$TARGET" "$flag" "$(printf '/tmp/foo\nbar')" --yes
    assert_exit 1
    assert_contains "illegal characters"
done

# ---------------------------------------------------------------------------
# validate_identifier-backed flags: reject shell-metacharacter-bearing values.
for flag in --user --group --pg-db-name --pg-db-user; do
    t "$flag rejects a value with invalid characters"
    run bash "$TARGET" "$flag" 'bad;value' --yes
    assert_exit 1
    assert_contains "contains invalid characters"
done

# ---------------------------------------------------------------------------
t "--build-dir rejects a path containing '..'"
run bash "$TARGET" --build-dir "../../etc" --yes
assert_exit 1
assert_contains "must be a relative path with no '..'"

# TD-043 regression: validate_build_dir's character class (a-zA-Z0-9._/-)
# allows '/', so an absolute path like "/etc/passwd" satisfied it and was
# never actually rejected, despite the error message's own claim ("must be
# a relative path"). Found writing this test suite; confirmed identically
# broken in 4 sibling install scripts, all now fixed alongside this one —
# an explicit leading-'/' check was missing entirely.
t "--build-dir rejects an absolute path (must be relative)"
run bash "$TARGET" --build-dir "/etc/passwd" --yes
assert_exit 1
assert_contains "must be a relative path with no '..'"

t "--build-dir still accepts an ordinary relative path"
run bash "$TARGET" --build-dir "build/portable" --yes
assert_not_contains "must be a relative path"
assert_contains "Installation requires root privileges"

# ---------------------------------------------------------------------------
# validate_port-backed flags: each must reject an out-of-range value AND
# name itself correctly in the error (TD-043 fix — previously all three
# reported the error as "--port:" regardless of which flag was invalid,
# since they shared one validator that never received the flag name).
for flag in --mns-port --registry-port --metrics-port; do
    t "$flag rejects an out-of-range port and names itself correctly"
    run bash "$TARGET" "$flag" 99999 --yes
    assert_exit 1
    assert_contains "${flag}: '99999' is not a valid port number"
    assert_not_contains "--port: '99999'" "regression: should name '${flag}', not the generic --port"

    t "$flag rejects a non-numeric value"
    run bash "$TARGET" "$flag" abc --yes
    assert_exit 1
    assert_contains "${flag}: 'abc' is not a valid port number"
done

# ---------------------------------------------------------------------------
t "valid arguments, but not root -> refuses with a clear message, exit 1"
run bash "$TARGET" --install-path /tmp/adai-test-install --yes
assert_exit 1
assert_contains "Installation requires root privileges"

# ---------------------------------------------------------------------------
t "no destructive action happened during any of the above -- install path never created"
assert_file_not_exists "/tmp/adai-test-install"

harness_report
exit $?
