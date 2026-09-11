#!/bin/bash
#
# Tests for scripts/install_incremental_trainer.sh (TD-043).
#
# See install_server_bundle_test.sh's header comment for why running the
# real script as a non-root user is safe here. --remote is deliberately
# exercised only with an INVALID host value (rejected by validate_remote_host
# before anything network-related happens) — a valid-looking host is never
# passed, since that path would attempt a real ssh/rsync connection.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/install_incremental_trainer.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --install-path --user --group --build-dir --config-src --vocab-src \
            --with-registry-server --with-systemd --coordinator --remote \
            --sync-sessions --ssh-key --yes --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
for flag in --install-path --config-src --vocab-src --ssh-key; do
    t "$flag rejects a relative path"
    run bash "$TARGET" "$flag" "relative/path" --yes
    assert_exit 1
    assert_contains "must be an absolute path"

    t "$flag regression: an ordinary absolute path must be accepted (TD-043 \$'\\0' bug)"
    run bash "$TARGET" "$flag" "/tmp/adai-td043-regress" --yes
    assert_not_contains "illegal characters"

    t "$flag rejects an embedded newline"
    run bash "$TARGET" "$flag" "$(printf '/tmp/foo\nbar')" --yes
    assert_exit 1
    assert_contains "illegal characters"
done

for flag in --user --group; do
    t "$flag rejects invalid characters"
    run bash "$TARGET" "$flag" 'bad;value' --yes
    assert_exit 1
    assert_contains "contains invalid characters"
done

t "--build-dir rejects '..' traversal"
run bash "$TARGET" --build-dir "../../etc" --yes
assert_exit 1
assert_contains "must be a relative path with no '..'"

t "--build-dir regression: rejects an absolute path (TD-043 missing leading-/ check)"
run bash "$TARGET" --build-dir "/etc/passwd" --yes
assert_exit 1
assert_contains "must be a relative path with no '..'"

# ---------------------------------------------------------------------------
t "--remote rejects a host value with shell metacharacters, before any network attempt"
run bash "$TARGET" --remote 'evil; rm -rf /' --yes
assert_exit 1
assert_contains "--remote: 'evil; rm -rf /' contains invalid characters"

t "--remote rejects an empty host"
run bash "$TARGET" --remote "" --yes
assert_exit 1
assert_contains "--remote: value must not be empty"

# ---------------------------------------------------------------------------
t "--coordinator implies --with-registry-server (no crash, still reaches EUID guard)"
run bash "$TARGET" --coordinator --yes
assert_contains "requires root privileges"

t "valid local arguments, no root -> refuses cleanly (no --remote)"
run bash "$TARGET" --install-path /tmp/adai-test --yes
assert_exit 1
assert_contains "requires root privileges"

harness_report
exit $?
