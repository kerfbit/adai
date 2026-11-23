#!/bin/bash
#
# Tests for scripts/package-sycl.sh (TD-043).
#
# (TD-575 added --system-oneapi and --build-dir; the note below predates them.) This script took no CLI arguments at all (confirmed by reading it — no
# `case "$1"`/getopts anywhere) and hardcodes BUILD_DIR="$PROJECT_ROOT/build/
# sycl" with no override. What's safely testable without an actual SYCL
# build is exactly its preflight guard: it refuses cleanly, before doing
# anything else (rm -rf/mkdir/tar), when build/sycl/bin/{incremental_trainer,
# chatbot_api_server} aren't both present — true in any checkout that hasn't
# built the sycl preset, verified below before relying on it.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/package-sycl.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

# ---------------------------------------------------------------------------
t "preflight refuses cleanly when the sycl build doesn't exist, makes no changes"
if [[ -x "${REPO_ROOT}/build/sycl/bin/incremental_trainer" && \
      -x "${REPO_ROOT}/build/sycl/bin/chatbot_api_server" ]]; then
    t "(skipped: build/sycl is actually built in this checkout — preflight would pass)"
else
    run bash "$TARGET"
    assert_exit 1
    assert_contains "Build the sycl preset first"
    assert_contains "cmake --preset sycl"
    # Confirms it exits at the very first preflight check, before ever
    # reaching "rm -rf \"$STAGE_DIR\"" further down.
    assert_not_contains "=== Packaging"
fi

# ---------------------------------------------------------------------------
# TD-575: --system-oneapi against a stand-in SYCL build (no compiler needed): no lib/, the extra
# binaries, deploy.sh, the units, and configs with scripts/ai_machine/*.overrides.conf applied.
t "--system-oneapi packs no libraries, the extra binaries, deploy.sh, the units and the overridden configs"
fake="$(mktemp_dir)"
mkdir -p "$fake/bin"
for b in incremental_trainer chatbot_api_server trainer_service vocab_builder dataset_manager; do
    printf '#!/bin/sh\n' >"$fake/bin/$b"
    chmod +x "$fake/bin/$b"
done
run env ONEAPI_ROOT=/nonexistent bash "$TARGET" --system-oneapi --build-dir "$fake"
assert_exit 0
assert_contains "No oneAPI libraries bundled (--system-oneapi)"
tarball=$(ls "$fake"/adai-sycl-ai-machine-*.tar.gz 2>/dev/null | head -1)
[[ -f "$tarball" ]] && assert_true true || assert_true false "no adai-sycl-ai-machine-*.tar.gz in $fake"
run tar -tzf "$tarball"
assert_contains "/bin/trainer_service"
assert_contains "/deploy.sh"
assert_contains "/units/adai-trainer@.service"
assert_contains "/etc/trainer-lejepa.env"
assert_not_contains "/lib/"
run bash -c 'tar -xzOf "$1" --wildcards "*/etc/config.trainer.conf"' _ "$tarball"
assert_contains "MODEL_NAME=bodacious-badger"
assert_contains "VOCAB_PATH=/opt/adai/vocab/bodacious-badger.vocab"
assert_not_contains "MODEL_NAME=chatbot-v1"
run bash -c 'tar -xzOf "$1" --wildcards "*/etc/config.trainer.conf" | grep -c "^MODEL_NAME="' _ "$tarball"
assert_contains "1"
run bash -c 'tar -xzOf "$1" --wildcards "*/etc/config.chatbot.conf"' _ "$tarball"
assert_contains "HIPPOCAMPAL_STATE_PATH=/opt/adai/models/bodacious-badger.hippocampal"
run bash -c 'tar -xzOf "$1" --wildcards "*/etc/config.*.conf" | grep -v "^#" | grep -c /home/' _ "$tarball"
assert_contains "0"
run bash -c 'tar -xzOf "$1" --wildcards "*/MANIFEST.txt"' _ "$tarball"
assert_contains "oneapi libraries: the host's"

t "refuses a binary older than the newest source, naming the build that brings it up to date"
touch -d '2000-01-01' "$fake/bin/chatbot_api_server"
rm -f "$fake"/adai-sycl-ai-machine-*.tar.gz
run env ONEAPI_ROOT=/nonexistent bash "$TARGET" --system-oneapi --build-dir "$fake"
assert_exit 1
assert_contains "ERROR: older than"
assert_contains ": chatbot_api_server"
assert_contains "--target incremental_trainer chatbot_api_server trainer_service vocab_builder dataset_manager"
assert_not_contains "=== Packaging"
touch "$fake/bin/chatbot_api_server"

t "--system-oneapi refuses a build missing trainer_service"
rm "$fake/bin/trainer_service"
run env ONEAPI_ROOT=/nonexistent bash "$TARGET" --system-oneapi --build-dir "$fake"
assert_exit 1
assert_contains "--system-oneapi packs trainer_service too"

harness_report
exit $?
