#!/bin/bash
#
# Tests for scripts/package_server_bundle.sh (TD-043).
#
# Unlike the install_*.sh scripts, this one needs no root at all, so its
# real happy path is exercised for real here (against the repo's own
# build/debug — the only one of the three auto-detect candidates that
# doesn't apply, since it's not build/portable, build, or build-clang-
# release, so --build-dir is passed explicitly), always with --output-dir
# pointed at a scratch directory so no tarball is ever written into the
# real repo tree.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/package_server_bundle.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help exits 0 and documents every flag"
run bash "$TARGET" --help
assert_exit 0
for flag in --build-dir --output-dir --version --no-strip --include-trainer \
            --include-chatbot-api --help; do
    assert_contains "$flag"
done

t "unknown flag -> exit 1"
run bash "$TARGET" --nonexistent-flag
assert_exit 1
assert_contains "Unknown option: --nonexistent-flag"

# ---------------------------------------------------------------------------
t "--build-dir pointing to a directory with no bin/ -> clean error"
d="$(mktemp_dir)"
run bash "$TARGET" --build-dir "$d" --output-dir "$d"
assert_exit 1
assert_contains "Binary directory not found"

t "--build-dir with bin/ but missing required binaries -> lists what's missing"
d="$(mktemp_dir)"
mkdir -p "$d/bin"
touch "$d/bin/metrics_api_server"
run bash "$TARGET" --build-dir "$d" --output-dir "$d"
assert_exit 1
assert_contains "Missing binaries"
assert_contains "registry_server"
assert_contains "mns_server"
assert_contains "mns_cli"

# ---------------------------------------------------------------------------
t "real build/debug is only found via explicit --build-dir, not auto-detected"
# build/debug is not one of the three auto-detect candidates (build/portable,
# build, build-clang-release) — confirms auto-detect fails cleanly rather
# than silently finding the wrong thing when none of its candidates exist.
if [[ -d "${REPO_ROOT}/build/portable" || -d "${REPO_ROOT}/build-clang-release" ]]; then
    t "(skipped: build/portable or build-clang-release exists in this checkout)"
else
    out_dir="$(mktemp_dir)"
    run_in "$REPO_ROOT" bash "$TARGET" --output-dir "$out_dir"
    assert_exit 1
    assert_contains "Could not auto-detect build directory"
fi

# ---------------------------------------------------------------------------
t "full real run against build/debug produces a real, well-formed tarball"
if [[ ! -x "${REPO_ROOT}/build/debug/bin/metrics_api_server" ]]; then
    t "(skipped: build/debug not built in this checkout)"
else
    out_dir="$(mktemp_dir)"
    run bash "$TARGET" --build-dir "${REPO_ROOT}/build/debug" --output-dir "$out_dir"
    assert_exit 0
    assert_contains "All required files present"

    tarballs=("$out_dir"/adai-server-bundle-*.tar.gz)
    if [[ -f "${tarballs[0]}" ]]; then
        pass
        t "tarball contains the expected top-level layout"
        run tar tzf "${tarballs[0]}"
        assert_contains "/bin/metrics_api_server"
        assert_contains "/bin/registry_server"
        assert_contains "/bin/mns_server"
        assert_contains "/bin/mns_cli"
        assert_contains "/config.conf"
        assert_contains "/scripts/install_server_bundle.sh"
        assert_contains "/scripts/setup_postgres.sql"
        # --include-trainer / --include-chatbot-api were NOT passed —
        # confirms those binaries are correctly left out by default.
        assert_not_contains "/bin/incremental_trainer"
        assert_not_contains "/bin/chatbot_api_server"
    else
        t "tarball produced by full run"
        fail "no adai-server-bundle-*.tar.gz found in $out_dir"
    fi
fi

# ---------------------------------------------------------------------------
t "--include-trainer requires incremental_trainer/dataset_manager to also be present"
if [[ -x "${REPO_ROOT}/build/debug/bin/metrics_api_server" ]]; then
    out_dir="$(mktemp_dir)"
    run bash "$TARGET" --build-dir "${REPO_ROOT}/build/debug" --output-dir "$out_dir" \
        --include-trainer
    if [[ -x "${REPO_ROOT}/build/debug/bin/incremental_trainer" ]]; then
        assert_exit 0
    else
        assert_exit 1
        assert_contains "incremental_trainer"
    fi
else
    t "(skipped: build/debug not built in this checkout)"
fi

harness_report
exit $?
