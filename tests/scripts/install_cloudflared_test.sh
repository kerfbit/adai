#!/bin/bash
#
# Tests for scripts/cloudflared/install_cloudflared.sh (TD-043).
#
# See install_server_bundle_test.sh's header comment for why running the
# real script as a non-root user is safe here.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/cloudflared/install_cloudflared.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag (required + optional)"
run bash "$TARGET" --help
assert_exit 0
for flag in --tunnel-name --config-src --credentials-src --install-path --user \
            --group --cloudflared-bin --yes --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
t "missing --tunnel-name -> exit 1, before any root check"
run bash "$TARGET" --config-src /tmp/x --credentials-src /tmp/y --yes
assert_exit 1
assert_contains "--tunnel-name is required"

t "--tunnel-name rejects invalid characters"
run bash "$TARGET" --tunnel-name 'bad;value' --yes
assert_exit 1
assert_contains "contains invalid characters"

for flag in --user --group; do
    t "$flag rejects invalid characters"
    run bash "$TARGET" --tunnel-name valid-tunnel "$flag" 'bad;value' --yes
    assert_exit 1
    assert_contains "contains invalid characters"
done

# ---------------------------------------------------------------------------
t "--install-path rejects a relative path"
run bash "$TARGET" --tunnel-name valid-tunnel --install-path "relative/path" --yes
assert_exit 1
assert_contains "must be an absolute path"

t "--install-path regression: an ordinary absolute path must be accepted (TD-043 \$'\\0' bug)"
run bash "$TARGET" --tunnel-name valid-tunnel --install-path "/tmp/adai-td043-regress" \
    --config-src /tmp/x --credentials-src /tmp/y --yes
assert_not_contains "illegal characters"

t "--install-path rejects an embedded newline"
run bash "$TARGET" --tunnel-name valid-tunnel --install-path "$(printf '/tmp/foo\nbar')" --yes
assert_exit 1
assert_contains "illegal characters"

# ---------------------------------------------------------------------------
t "all required flags present, valid, no root -> refuses cleanly"
run bash "$TARGET" --tunnel-name adai-storage-tunnel --config-src /tmp/does-not-exist.yml \
    --credentials-src /tmp/does-not-exist.json --yes
assert_exit 1
assert_contains "requires root privileges"

harness_report
exit $?
