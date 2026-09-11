#!/bin/bash
#
# Tests for scripts/install_metrics_service.sh (TD-043).
#
# See install_server_bundle_test.sh's header comment for why running the
# real script as a non-root user is safe here: every case below exits during
# argument parsing or the EUID check, both of which run before anything that
# touches the filesystem or needs root.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/install_metrics_service.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --install-path --user --group --build-dir --port --metrics-dir --wipe-data --yes --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
for flag in --install-path --metrics-dir; do
    t "$flag rejects a relative path"
    run bash "$TARGET" "$flag" "relative/path" --yes
    assert_exit 1
    assert_contains "must be an absolute path"

    t "$flag regression: an ordinary absolute path must be accepted (TD-043 \$'\\0' bug)"
    run bash "$TARGET" "$flag" "/tmp/adai-td043-regress" --yes
    assert_not_contains "illegal characters"
    assert_contains "Installation requires root privileges"

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

t "--build-dir accepts an ordinary relative path"
run bash "$TARGET" --build-dir "build/portable" --yes
assert_not_contains "must be a relative path"
assert_contains "Installation requires root privileges"

t "--port rejects an out-of-range value"
run bash "$TARGET" --port 99999 --yes
assert_exit 1
assert_contains "--port: '99999' is not a valid port number"

t "--port rejects a non-numeric value"
run bash "$TARGET" --port abc --yes
assert_exit 1
assert_contains "is not a valid port number"

t "valid arguments, no root -> refuses cleanly"
run bash "$TARGET" --port 8081 --yes
assert_exit 1
assert_contains "Installation requires root privileges"

harness_report
exit $?
