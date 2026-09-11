#!/bin/bash
#
# Tests for scripts/install_mns_server.sh (TD-043).
#
# See install_server_bundle_test.sh's header comment for why running the
# real script as a non-root user is safe here.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/install_mns_server.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --install-path --user --group --build-dir --port --wipe-data --yes --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
t "--install-path rejects a relative path"
run bash "$TARGET" --install-path "relative/path" --yes
assert_exit 1
assert_contains "must be an absolute path"

t "--install-path regression: an ordinary absolute path must be accepted (TD-043 \$'\\0' bug)"
run bash "$TARGET" --install-path "/tmp/adai-td043-regress" --yes
assert_not_contains "illegal characters"
assert_contains "Installation requires root privileges"

t "--install-path rejects an embedded newline"
run bash "$TARGET" --install-path "$(printf '/tmp/foo\nbar')" --yes
assert_exit 1
assert_contains "illegal characters"

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
assert_contains "is not a valid port number"

t "--port rejects a non-numeric value"
run bash "$TARGET" --port abc --yes
assert_exit 1
assert_contains "is not a valid port number"

t "valid arguments, no root -> refuses cleanly"
run bash "$TARGET" --port 8083 --yes
assert_exit 1
assert_contains "Installation requires root privileges"

harness_report
exit $?
