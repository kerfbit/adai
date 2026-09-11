#!/bin/bash
#
# Tests for scripts/check_tech_debt.sh (TD-043).
#
# The script resolves SRC_DIR="src", TESTS_DIR="tests", and
# DEBT_FILE="docs/development/guides/TECHNICAL_DEBT.md" relative to its own
# CWD (not its own file location) — confirmed by reading the script: none of
# the three is anchored via $(dirname "${BASH_SOURCE[0]}"). That's what makes
# it testable at all without touching the real repo: each test below `cd`s
# into a throwaway fixture directory with its own src/, tests/, and
# docs/development/guides/TECHNICAL_DEBT.md before invoking the script by
# absolute path.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/check_tech_debt.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "no markers anywhere, debt file present -> exit 0, 'No untracked' message"
d="$(mktemp_dir)"
mkdir -p "$d/src" "$d/tests" "$d/docs/development/guides"
echo 'int main() { return 0; }' > "$d/src/main.cpp"
echo '# Technical Debt' > "$d/docs/development/guides/TECHNICAL_DEBT.md"
run_in "$d" bash "$TARGET"
assert_exit 0
assert_contains "No untracked technical debt markers found!"
assert_contains "TODO markers:"

# ---------------------------------------------------------------------------
t "one untracked TODO in src/ -> exit 1, reported in summary and untracked warning"
d="$(mktemp_dir)"
mkdir -p "$d/src" "$d/tests" "$d/docs/development/guides"
printf '// TODO: fix this later\nint main() { return 0; }\n' > "$d/src/main.cpp"
echo '# Technical Debt' > "$d/docs/development/guides/TECHNICAL_DEBT.md"
run_in "$d" bash "$TARGET"
assert_exit 1
assert_contains "Untracked technical debt found!"
assert_contains "main.cpp"

# ---------------------------------------------------------------------------
t "TODO in tests/ is also scanned, not just src/"
d="$(mktemp_dir)"
mkdir -p "$d/src" "$d/tests" "$d/docs/development/guides"
echo 'int main() { return 0; }' > "$d/src/main.cpp"
printf '// TODO: add more coverage\nvoid f() {}\n' > "$d/tests/foo_test.cpp"
echo '# Technical Debt' > "$d/docs/development/guides/TECHNICAL_DEBT.md"
run_in "$d" bash "$TARGET"
assert_exit 1
assert_contains "foo_test.cpp"

# ---------------------------------------------------------------------------
t "FIXME/HACK/XXX each counted independently of TODO"
d="$(mktemp_dir)"
mkdir -p "$d/src" "$d/tests" "$d/docs/development/guides"
cat > "$d/src/main.cpp" <<'EOF'
// FIXME: broken
// HACK: workaround
// XXX: dangerous
int main() { return 0; }
EOF
echo '# Technical Debt' > "$d/docs/development/guides/TECHNICAL_DEBT.md"
run_in "$d" bash "$TARGET"
assert_exit 1
assert_contains "TODO markers:  0"
assert_contains "FIXME markers: 1"
assert_contains "HACK markers:  1"
assert_contains "XXX markers:   1"
assert_contains "Total:         3"

# ---------------------------------------------------------------------------
t "missing TECHNICAL_DEBT.md -> exit 1, explicit warning naming the file"
d="$(mktemp_dir)"
mkdir -p "$d/src" "$d/tests"
echo 'int main() { return 0; }' > "$d/src/main.cpp"
run_in "$d" bash "$TARGET"
assert_exit 1
assert_contains "docs/development/guides/TECHNICAL_DEBT.md not found"

# ---------------------------------------------------------------------------
t "only .cpp/.hpp are scanned, not other extensions"
d="$(mktemp_dir)"
mkdir -p "$d/src" "$d/tests" "$d/docs/development/guides"
printf '# TODO: not a cpp/hpp file, must not be counted\n' > "$d/src/notes.md"
echo '# Technical Debt' > "$d/docs/development/guides/TECHNICAL_DEBT.md"
run_in "$d" bash "$TARGET"
assert_exit 0
assert_contains "No untracked technical debt markers found!"

harness_report
exit $?
