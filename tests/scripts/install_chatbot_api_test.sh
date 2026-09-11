#!/bin/bash
#
# Tests for scripts/install_chatbot_API.sh (TD-043).
#
# See install_server_bundle_test.sh's header comment for why running the
# real script as a non-root user is safe here.
#
# NOTE: unlike every sibling install script, this one currently validates
# only --build-dir — --install-path/--user/--group/--port are assigned with
# zero validation. That gap is real (flagged separately as its own follow-up
# fix, not folded into this TD-043 pass, since it's new validation logic to
# add rather than a broken check to fix) — this test file covers the
# script's actual current behavior, including that gap, rather than
# asserting behavior the script doesn't have yet.

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
t "valid arguments, no root -> refuses cleanly"
run bash "$TARGET" --install-path /tmp/adai-test --port 8080
assert_exit 1
assert_contains "must be run as root"

harness_report
exit $?
