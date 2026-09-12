#!/bin/bash
#
# Tests for scripts/install_chatbot_API.sh (TD-043, TD-157).
#
# See install_server_bundle_test.sh's header comment for why running the
# real script as a non-root user is safe here.
#
# TD-157: --install-path/--user/--group/--port used to be assigned with zero
# validation, unlike every sibling install script (install_metrics_service.sh,
# install_mns_server.sh, install_server_bundle.sh, install_incremental_trainer.sh,
# cloudflared/install_cloudflared.sh) — the same unsanitized-identifier-into-a-
# privileged-command class of bug TD-149 already fixed for install_server_bundle.sh's
# --pg-db-name/--pg-db-user. Fixed by adding validate_identifier/validate_abs_path/
# validate_port calls (copied from install_metrics_service.sh's own already-fixed
# versions); the tests below mirror install_metrics_service_test.sh's own coverage
# of those same validators.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/install_chatbot_API.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --install-path --build-dir --user --group --port --log-level --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
t "--build-dir rejects '..' traversal"
run bash "$TARGET" --build-dir "../../etc"
assert_exit 1
assert_contains "must be a relative path with no '..'"

t "--build-dir rejects an absolute path"
run bash "$TARGET" --build-dir "/etc/passwd"
assert_exit 1
assert_contains "must be a relative path with no '..'"

t "--build-dir accepts an ordinary relative path, proceeds to the EUID guard"
run bash "$TARGET" --build-dir "build/portable"
assert_not_contains "must be a relative path"
assert_contains "must be run as root"

# ---------------------------------------------------------------------------
# TD-157: --install-path/--user/--group/--port validation, added to match every
# sibling install script — see install_metrics_service_test.sh for the same
# coverage of these shared validators.

t "--install-path rejects a relative path"
run bash "$TARGET" --install-path "relative/path"
assert_exit 1
assert_contains "must be an absolute path"

t "--install-path accepts an ordinary absolute path, proceeds to the EUID guard"
run bash "$TARGET" --install-path "/tmp/adai-td157-regress"
assert_not_contains "illegal characters"
assert_contains "must be run as root"

t "--install-path rejects an embedded newline"
run bash "$TARGET" --install-path "$(printf '/tmp/foo\nbar')"
assert_exit 1
assert_contains "illegal characters"

for flag in --user --group; do
    t "$flag rejects invalid characters"
    run bash "$TARGET" "$flag" 'bad;value'
    assert_exit 1
    assert_contains "contains invalid characters"

    t "$flag accepts an ordinary identifier, proceeds to the EUID guard"
    run bash "$TARGET" "$flag" "chatbotsvc"
    assert_not_contains "contains invalid characters"
    assert_contains "must be run as root"
done

t "--port rejects an out-of-range value"
run bash "$TARGET" --port 99999
assert_exit 1
assert_contains "--port: '99999' is not a valid port number"

t "--port rejects a non-numeric value"
run bash "$TARGET" --port abc
assert_exit 1
assert_contains "is not a valid port number"

# ---------------------------------------------------------------------------
t "valid arguments, no root -> refuses cleanly"
run bash "$TARGET" --install-path /tmp/adai-test --user chatbotsvc --group chatbotsvc --port 8080
assert_exit 1
assert_contains "must be run as root"

harness_report
exit $?
