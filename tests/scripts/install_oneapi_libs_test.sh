#!/bin/bash
#
# Tests for scripts/install_oneapi_libs.sh (TD-043).
#
# Unlike the other install_*.sh scripts, this one supports a real --dry-run
# that also skips the EUID check entirely (`[[ $EUID -ne 0 ]] && [[
# "$DRY_RUN" == false ]]`), so its full happy path — auto-detect, library
# counting, the dry-run summary — can be exercised for real as a non-root
# user, not just argument-parsing/error-path coverage.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/install_oneapi_libs.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --lib-dir --install-path --dry-run --uninstall --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
# validate_install_path: rejects every FHS top-level directory (depth < 2),
# accepts a realistically deep path. --dry-run so the EUID guard never
# blocks reaching this check.
for shallow in /home /etc /usr /var /root /tmp /opt /; do
    t "--install-path rejects shallow path '$shallow' (refuses to risk rm -rf on it)"
    run bash "$TARGET" --dry-run --install-path "$shallow" --lib-dir /tmp
    assert_exit 1
    assert_contains "too shallow"
done

t "--install-path accepts a realistically deep path"
run bash "$TARGET" --dry-run --install-path "/opt/adai/lib" --lib-dir /tmp
assert_not_contains "too shallow"

t "--install-path rejects a relative path"
run bash "$TARGET" --dry-run --install-path "relative/lib" --lib-dir /tmp
assert_exit 1
assert_contains "must be an absolute path"

t "--install-path rejects empty value"
run bash "$TARGET" --dry-run --install-path "" --lib-dir /tmp
assert_exit 1
assert_contains "value must not be empty"

# ---------------------------------------------------------------------------
t "--lib-dir pointing to a nonexistent directory -> clean error, not a raw 'find' failure"
run bash "$TARGET" --dry-run --lib-dir "/tmp/does-not-exist-$$"
assert_exit 1
assert_contains "does not exist or is not a directory"
assert_not_contains "No such file or directory" "must be the script's own friendly message, not a raw find(1) error leaking through"

t "--lib-dir pointing to an empty directory -> 'no shared libraries found'"
d="$(mktemp_dir)"
run bash "$TARGET" --dry-run --lib-dir "$d"
assert_exit 1
assert_contains "No shared libraries found"

# ---------------------------------------------------------------------------
t "full dry-run happy path: counts libraries, makes no filesystem changes"
d="$(mktemp_dir)"
touch "$d/libfoo.so" "$d/libbar.so.1" "$d/not_a_lib.txt"
install_dir="$d/fake_install"
run bash "$TARGET" --dry-run --lib-dir "$d" --install-path "$install_dir/adai/lib"
assert_exit 0
assert_contains "Source:      $d (2 libraries)"
assert_contains "--- DRY RUN ---"
assert_contains "Would copy 2 libraries"
assert_file_not_exists "$install_dir"

t "--uninstall --dry-run makes no filesystem changes even if paths exist"
d="$(mktemp_dir)"
fake_install="$d/adai/lib"
mkdir -p "$fake_install"
touch "$fake_install/libfoo.so"
run bash "$TARGET" --uninstall --dry-run --install-path "$fake_install"
assert_exit 0
assert_contains "DRY RUN"
assert_file_exists "$fake_install/libfoo.so"

# ---------------------------------------------------------------------------
t "auto-detect finds lib/ one level above the script's own directory"
d="$(mktemp_dir)"
mkdir -p "$d/scripts" "$d/lib"
touch "$d/lib/libauto.so"
cp "$TARGET" "$d/scripts/install_oneapi_libs.sh"
run bash "$d/scripts/install_oneapi_libs.sh" --dry-run --install-path "$d/opt/adai/lib"
assert_exit 0
assert_contains "Source:      $d/lib (1 libraries)"

# ---------------------------------------------------------------------------
t "without --dry-run and not root -> refuses cleanly, no changes attempted"
d="$(mktemp_dir)"
touch "$d/libfoo.so"
install_dir="$d/would_install/adai/lib"
run bash "$TARGET" --lib-dir "$d" --install-path "$install_dir"
assert_exit 1
assert_contains "must be run as root"
assert_file_not_exists "$install_dir"

harness_report
exit $?
