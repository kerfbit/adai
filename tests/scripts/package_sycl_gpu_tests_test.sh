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
LEJEPA_PASS="${REPO_ROOT}/scripts/sycl_gpu_tests/lejepa_pass.sh"
PREPARE="${REPO_ROOT}/scripts/sycl_gpu_tests/prepare_host.sh"
RESTORE="${REPO_ROOT}/scripts/sycl_gpu_tests/restore_host.sh"
LIB="${REPO_ROOT}/scripts/sycl_gpu_tests/gpu_host_lib.sh"
GUC="${REPO_ROOT}/scripts/sycl_gpu_tests/guc_firmware.sh"
source "${SCRIPT_DIR}/harness.sh"

for script in "$PACKAGER" "$RUNNER" "$TRAIN_PASS" "$LEJEPA_PASS" "$PREPARE" "$RESTORE" "$LIB" "$GUC"; do
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

t "runner --soak-passes needs a whole number"
run bash "$RUNNER" --soak-passes abc
assert_exit 2
assert_contains "whole number"

t "runner --lejepa-soak-passes needs a whole number"
run bash "$RUNNER" --lejepa-soak-passes abc
assert_exit 2
assert_contains "whole number"

t "runner --help lists the LeJEPA options"
run bash "$RUNNER" --help
assert_exit 0
assert_contains "--lejepa-only"

t "lejepa_pass.sh without arguments prints its usage"
run bash "$LEJEPA_PASS"
assert_exit 2
assert_contains "lejepa_pass.sh setup"

t "lejepa_pass.sh rejects an unknown command"
run bash "$LEJEPA_PASS" bogus /nonexistent
assert_exit 2
assert_contains "Unknown command"

t "prepare --help prints its usage without sudo or changes"
run bash "$PREPARE" --help
assert_exit 0
assert_contains "--skip-upgrade"
assert_file_not_exists "${REPO_ROOT}/scripts/sycl_gpu_tests/host_state"

t "prepare rejects an unknown argument before using sudo"
run bash "$PREPARE" --bogus
assert_exit 2
assert_contains "Unknown argument"
assert_file_not_exists "${REPO_ROOT}/scripts/sycl_gpu_tests/host_state"

t "restore with nothing stopped says so, without sudo"
run bash "$RESTORE"
assert_exit 0
assert_contains "Nothing to restore"

t "cgroup paths map to the service, user service or container that owns a process"
run bash -c 'source "$1"
    unit_from_cgroup "0::/system.slice/adai.service"
    unit_from_cgroup "0::/user.slice/user-1001.slice/user@1001.service/app.slice/ollama.service"
    unit_from_cgroup "0::/system.slice/docker-0123456789abcdef0123.scope"
    unit_from_cgroup "0::/user.slice/user-1000.slice/session-2.scope"' _ "$LIB"
assert_exit 0
assert_contains "system adai.service"
assert_contains "user 1001 ollama.service"
assert_contains "docker 0123456789ab"
assert_contains "none"

t "a runtime at or above the build's oneAPI release passes, an older one doesn't"
run bash -c 'source "$1"
    oneapi_version_at_least 2026.2.0-863 2026.2.0 && echo newer-ok
    oneapi_version_at_least 2026.1.1-325 2026.2.0 || echo older-refused' _ "$LIB"
assert_exit 0
assert_contains "newer-ok"
assert_contains "older-refused"

t "dpkg's refusals between oneAPI runtime packages are recognised; any other refusal isn't"
LOGS="$(mktemp -d)"
printf '%s\n' "dpkg: error processing archive /tmp/x/02-intel-oneapi-runtime-openmp_2026.2.0-863_amd64.deb (--unpack):" \
    " trying to overwrite '/opt/intel/oneapi/redist/lib/libur_loader.so', which is also in package intel-oneapi-runtime-dpcpp-sycl-core 2026.1.1-325" >"$LOGS/runtime.log"
printf '%s\n' " trying to overwrite '/usr/lib/x86_64-linux-gnu/libze_loader.so.1', which is also in package libze1 1.20" >"$LOGS/driver.log"
: >"$LOGS/none.log"
run bash -c 'source "$1"; cd "$2"
    for f in runtime driver none; do
        overwrite_conflicts $f.log >$f.c
        if only_runtime_conflicts $f.c; then echo "$f: finish"; else echo "$f: refuse"; fi
    done
    cat runtime.c' _ "$LIB" "$LOGS"
assert_exit 0
assert_contains "runtime: finish"
assert_contains "driver: refuse"
assert_contains "none: refuse"
assert_contains "/opt/intel/oneapi/redist/lib/libur_loader.so intel-oneapi-runtime-dpcpp-sycl-core"
rm -rf -- "$LOGS"

t "guc_firmware.sh without a command prints its usage and changes nothing"
run bash "$GUC"
assert_exit 2
assert_contains "status | install | uninstall"

t "guc_firmware.sh reads the GuC version from a firmware header"
FW="$(mktemp)"
python3 -c "import struct,sys; b=bytearray(128); struct.pack_into('<I', b, 64, (70<<16)|(72<<8)|1); open(sys.argv[1],'wb').write(b)" "$FW"
run bash -c 'source <(sed -n "/^guc_version()/,/^}/p" "$1"); guc_version "$2"' _ "$GUC" "$FW"
assert_exit 0
assert_contains "70.72.1"
rm -f -- "$FW"

t "the packaged pairs file is JSON Lines with input and response"
run python3 -c "import json,sys; [json.loads(l)['response'] for l in open(sys.argv[1])]" \
    "${REPO_ROOT}/scripts/sycl_gpu_tests/pairs.jsonl"
assert_exit 0

harness_report
exit $?
