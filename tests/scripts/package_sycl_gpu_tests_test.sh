#!/bin/bash
#
# Tests for scripts/package_sycl_gpu_tests.sh and the scripts it packages
# (scripts/sycl_gpu_tests/). What's testable without a SYCL build: syntax,
# --help, argument errors, and the refusals that come before any build or
# file change.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PACKAGER="${REPO_ROOT}/scripts/package_sycl_gpu_tests.sh"
RUNNER="${REPO_ROOT}/scripts/sycl_gpu_tests/run_gpu_tests.sh"
TRAIN_PASS="${REPO_ROOT}/scripts/sycl_gpu_tests/train_pass.sh"
source "${SCRIPT_DIR}/harness.sh"

for script in "$PACKAGER" "$RUNNER" "$TRAIN_PASS"; do
    t "syntax check: $(basename "$script")"
    run bash -n "$script"
    assert_exit 0
done

t "packager --help prints its usage"
run bash "$PACKAGER" --help
assert_exit 0
assert_contains "--build-dir"

t "packager rejects an unknown argument"
run bash "$PACKAGER" --bogus
assert_exit 2
assert_contains "unknown argument"

t "packager --build-dir needs a value"
run bash "$PACKAGER" --build-dir
assert_exit 2

t "packager --no-build refuses a directory with no build, before building or staging"
EMPTY="$(mktemp -d)"
run bash "$PACKAGER" --no-build --build-dir "$EMPTY/none" --out "$EMPTY/out"
assert_exit 1
assert_contains "no build in"
assert_file_not_exists "$EMPTY/out"
rm -rf -- "$EMPTY"

t "runner --help prints its usage without running anything"
run bash "$RUNNER" --help
assert_exit 0
assert_contains "--allow-cpu-device"
assert_file_not_exists "${REPO_ROOT}/scripts/sycl_gpu_tests/results"

t "runner rejects an unknown argument before running anything"
run bash "$RUNNER" --bogus
assert_exit 2
assert_contains "Unknown argument"

t "the packaged pairs file is JSON Lines with input and response"
run python3 -c "import json,sys; [json.loads(l)['response'] for l in open(sys.argv[1])]" \
    "${REPO_ROOT}/scripts/sycl_gpu_tests/pairs.jsonl"
assert_exit 0

harness_report
exit $?
