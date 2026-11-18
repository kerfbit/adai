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
COREDUMPS="${REPO_ROOT}/scripts/sycl_gpu_tests/coredumps.sh"
UPDATE="${REPO_ROOT}/scripts/sycl_gpu_tests/update_gpu_stack.sh"
XE_DEBUG="${REPO_ROOT}/scripts/sycl_gpu_tests/xe_debug.sh"
MAINLINE="${REPO_ROOT}/scripts/sycl_gpu_tests/mainline_kernel.sh"
source "${SCRIPT_DIR}/harness.sh"

for script in "$PACKAGER" "$RUNNER" "$TRAIN_PASS" "$LEJEPA_PASS" "$PREPARE" "$RESTORE" "$LIB" "$GUC" "$COREDUMPS" "$UPDATE" "$XE_DEBUG" "$MAINLINE"; do
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

t "runner --bisect-passes needs a whole number"
run bash "$RUNNER" --bisect-passes abc
assert_exit 2
assert_contains "whole number"

t "runner --help lists the engine-reset bisection options (TD-574)"
run bash "$RUNNER" --help
assert_exit 0
assert_contains "--bisect-only"
assert_contains "--skip-bisect"
assert_contains "--engine-tuning"

KERNEL_EVENTS_PATTERN=$(sed -n "s/^KERNEL_EVENTS='\(.*\)'$/\1/p" "$RUNNER")
t "the kernel-event pattern counts a new crash dump, not its deletion (TD-574)"
run bash -c 'printf "%s\n" \
    "xe 0000:2d:00.0: [drm] GT0: Engine reset: engine_class=ccs, logical_mask: 0x1, guc_id=38" \
    "xe 0000:2d:00.0: [drm] Xe device coredump has been created" \
    "xe 0000:2d:00.0: [drm] Xe device coredump has been deleted." | grep -c -E "$1"' _ "$KERNEL_EVENTS_PATTERN"
assert_exit 0
assert_contains "2"

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
assert_contains "status | latest | install | install-latest | uninstall"

t "guc_firmware.sh reads the GuC version from a firmware header"
FW="$(mktemp)"
python3 -c "import struct,sys; b=bytearray(128); struct.pack_into('<I', b, 64, (70<<16)|(72<<8)|1); open(sys.argv[1],'wb').write(b)" "$FW"
run bash -c 'source <(sed -n "/^guc_version()/,/^}/p" "$1"); guc_version "$2"' _ "$GUC" "$FW"
assert_exit 0
assert_contains "70.72.1"
rm -f -- "$FW"

t "coredumps.sh without a command prints its usage and changes nothing"
run bash "$COREDUMPS"
assert_exit 2
assert_contains "status | install | uninstall"

t "coredumps.sh rejects a bad --days"
run bash "$COREDUMPS" install --days 0
assert_exit 2
assert_contains "--days needs"

t "coredumps.sh install keeps both kinds of dump and enables core dumps for adai services"
CD_ROOT="$(mktemp -d)"
run env COREDUMPS_ROOT="$CD_ROOT" COREDUMPS_TEST_UNITS="adai-trainer.service adai.service" \
    bash "$COREDUMPS" install --days 45 --user "$(id -un)"
assert_exit 0
assert_contains "Core dumps enabled for adai-trainer.service"
run cat "$CD_ROOT/etc/tmpfiles.d/00-adai-coredumps.conf" "$CD_ROOT/etc/systemd/coredump.conf.d/50-adai.conf" \
    "$CD_ROOT/etc/udev/rules.d/90-adai-gpu-coredump.rules" "$CD_ROOT/etc/systemd/system/adai.service.d/50-adai-coredump.conf" \
    "$CD_ROOT/etc/security/limits.d/90-adai-core.conf"
assert_exit 0
assert_contains "d /var/lib/systemd/coredump 0755 root root 45d"
assert_contains "d /var/lib/adai/gpu-coredumps 0750 root $(id -gn) 45d"
assert_contains "Storage=external"
assert_contains 'SUBSYSTEM=="devcoredump"'
assert_contains "adai-gpu-coredump@%k.service"
assert_contains "LimitCORE=infinity"
assert_contains "$(id -un) soft core unlimited"

t "the installed saver is valid sh and frees the dump after saving it"
run sh -n "$CD_ROOT/usr/local/sbin/adai-save-gpu-coredump"
assert_exit 0
run grep -c 'echo 1 >"$src"' "$CD_ROOT/usr/local/sbin/adai-save-gpu-coredump"
assert_contains "1"

t "coredumps.sh uninstall removes what install added"
run env COREDUMPS_ROOT="$CD_ROOT" COREDUMPS_TEST_UNITS="adai-trainer.service adai.service" bash "$COREDUMPS" uninstall
assert_exit 0
run find "$CD_ROOT" -type f
assert_exit 0
assert_not_contains ".conf"
assert_not_contains "adai-save-gpu-coredump"
rm -rf -- "$CD_ROOT"

t "the bisection runs queue bounds against an unbounded control (TD-574)"
run bash -c '
    variant_env() {
        local variant=$1 envs=()
        eval "$(sed -n "/^            case \$variant in\$/,/^            esac\$/p" "$2" | head -4)"
        echo "$variant: ${envs[*]:-none}"
    }
    for v in control default depth_8 depth_16 encode_soak encode_soak_depth_8 sync_kernel; do variant_env $v "$1"; done
    grep -A1 "VARIANTS=(\${BISECT_VARIANTS" "$1"' _ "$RUNNER"
assert_exit 0
assert_contains "control: GPU_QUEUE_DEPTH=0"
assert_contains "default: GPU_QUEUE_DEPTH=0"
assert_contains "depth_8: GPU_QUEUE_DEPTH=8"
assert_contains "depth_16: GPU_QUEUE_DEPTH=16"
assert_contains "encode_soak: GPU_QUEUE_DEPTH=0"
assert_contains "encode_soak_depth_8: GPU_QUEUE_DEPTH=8"
assert_contains "sync_kernel: GPU_QUEUE_DEPTH=0"
assert_contains "control depth_8 encode_soak repro_mkl repro_sycl"

t "the sixth round's variants set the debug log, the v1 teardown and the engine settings (TD-574)"
run bash -c '
    variant_env() {
        local variant=$1 envs=() tuned="" dir=/w
        eval "$(sed -n "/^            case \$variant in\$/,/^            esac\$/p" "$2")"
        echo "$variant: ${envs[*]:-none} tuned=${tuned:-none}"
    }
    for v in ur_debug v1_teardown long_timeslice no_preempt repro_mkl_wait_8; do variant_env $v "$1"; done' _ "$RUNNER"
assert_exit 0
assert_contains "ur_debug: GPU_QUEUE_DEPTH=0 UR_LOG_LEVEL_ZERO=level:debug;flush:debug;output:file,/w/ur_level_zero.log"
assert_contains "v1_teardown: GPU_QUEUE_DEPTH=0 SYCL_UR_USE_LEVEL_ZERO_V2=0 ADAI_GPU_KEEP_TEARDOWN=1"
assert_contains "long_timeslice: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=background tuned=timeslice_duration_us:100000"
assert_contains "no_preempt: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=background tuned=preempt_timeout_us:0"

t "the priority variants train at GPU_STRATEGY=normal and full, unbounded, and only when named (TD-574)"
run bash -c '
    variant_env() {
        local variant=$1 envs=() tuned="" dir=/w
        eval "$(sed -n "/^            case \$variant in\$/,/^            esac\$/p" "$2")"
        echo "$variant: ${envs[*]:-none}"
    }
    for v in normal_priority full_priority encode_soak_normal_priority encode_soak_full_priority repro_sycl_normal; do
        variant_env $v "$1"
    done' _ "$RUNNER"
assert_exit 0
assert_contains "normal_priority: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=normal"
assert_contains "full_priority: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=full"
assert_contains "encode_soak_normal_priority: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=normal"
assert_contains "encode_soak_full_priority: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=full"
assert_contains "repro_sycl_normal: GPU_QUEUE_DEPTH=0"
assert_not_contains "repro_sycl_normal: GPU_QUEUE_DEPTH=0 GPU_STRATEGY"

t "every other variant trains at GPU_STRATEGY=background, the earlier rounds' queue, not the new default (TD-574)"
run bash -c '
    variant_env() {
        local variant=$1 envs=() tuned="" dir=/w
        eval "$(sed -n "/^            case \$variant in\$/,/^            esac\$/p" "$2")"
        echo "$variant: ${envs[*]:-none}"
    }
    for v in control depth_8 encode_soak v1_teardown repro_sycl repro_mkl_wait_8; do variant_env $v "$1"; done' _ "$RUNNER"
assert_exit 0
assert_contains "control: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=background"
assert_contains "depth_8: GPU_QUEUE_DEPTH=8 GPU_STRATEGY=background"
assert_contains "encode_soak: GPU_QUEUE_DEPTH=0 GPU_STRATEGY=background"
assert_contains "v1_teardown: GPU_QUEUE_DEPTH=0 SYCL_UR_USE_LEVEL_ZERO_V2=0 ADAI_GPU_KEEP_TEARDOWN=1 GPU_STRATEGY=background"
assert_not_contains "repro_sycl: GPU_QUEUE_DEPTH=0 GPU_STRATEGY"
assert_not_contains "repro_mkl_wait_8: GPU_QUEUE_DEPTH=0 GPU_STRATEGY"
run bash -c 'grep -A2 "VARIANTS=(\${BISECT_VARIANTS" "$1" | grep -c "_priority"' _ "$RUNNER"
assert_contains "0"

t "the eighth round runs the blocking and traced reproducers instead of ur_debug (TD-574)"
run bash -c 'grep -A1 "VARIANTS=(\${BISECT_VARIANTS" "$1"' _ "$RUNNER"
assert_exit 0
assert_contains "repro_sycl_blocking repro_sycl_trace v1_teardown"
assert_not_contains "ur_debug"

t "the eighth round also runs both reproducers on a normal-priority queue (TD-574)"
run bash -c 'grep -A2 "VARIANTS=(\${BISECT_VARIANTS" "$1"' _ "$RUNNER"
assert_exit 0
assert_contains "repro_sycl_normal repro_mkl_normal"

t "runner --help lists --kernel-debug, which adds the drm.debug variant only with sudo (TD-574)"
run bash "$RUNNER" --help
assert_exit 0
assert_contains "--kernel-debug"
run bash -c 'grep -c -F -e "VARIANTS+=(repro_sycl_drm_debug)" -e "drm_debug_set \"\$DRM_DEBUG_MASK\"" -e "== *_drm_debug ]] && drm_debug_restore" "$1"' _ "$RUNNER"
assert_contains "3"

t "the runner reads drm.debug through sudo, since the file is root's alone (TD-574)"
run bash -c 'grep -n -E "(-r \"\\\$DRM_DEBUG_FILE\"|[^O] cat \"\\\$DRM_DEBUG_FILE\"|\\\$\\(cat /sys/module/drm)" "$1"' _ "$RUNNER"
assert_exit 1
run bash -c 'grep -c -F -e "DRM_DEBUG_ORIG=\$(\$SUDO cat \"\$DRM_DEBUG_FILE\"" -e "drm.debug: \$(\$SUDO cat /sys/module/drm/parameters/debug" "$1"' _ "$RUNNER"
assert_contains "2"

t "the reproducer variants get their arguments, and the traced one keeps only the trace's tail (TD-574)"
run bash -c '
    step() { shift 2; echo "step: $*"; }
    BIN=/b
    for variant in repro_mkl repro_sycl_drm_debug repro_sycl_blocking repro_mkl_wait_8 repro_sycl_trace repro_sycl_normal repro_mkl_normal; do
        name=$variant envs=(E=1)
        eval "if false; then :; $(sed -n "/^            elif \[\[ \$variant == repro_\* \]\]; then\$/,/^            elif \[\[ \$variant == v1_teardown \]\]; then\$/p" "$1" | sed "\$d") fi"
    done' _ "$RUNNER"
assert_exit 0
assert_contains "step: env E=1 /b/sycl_engine_reset_repro --seconds 15 --mkl"
assert_contains "step: env E=1 /b/sycl_engine_reset_repro --seconds 15 --blocking-wait"
assert_contains "step: env E=1 /b/sycl_engine_reset_repro --seconds 15 --mkl --wait-every 8"
assert_contains "step: env E=1 SYCL_UR_TRACE=2 bash -c"
assert_contains "tail -c"
assert_contains "_ /b/sycl_engine_reset_repro --seconds 15"
assert_contains "step: env E=1 /b/sycl_engine_reset_repro --seconds 15 --normal-priority"
assert_contains "step: env E=1 /b/sycl_engine_reset_repro --seconds 15 --mkl --normal-priority"
assert_contains "step: env E=1 /b/sycl_engine_reset_repro --seconds 15
"

t "stacks are capped per variant, so later variants still get theirs (TD-574)"
STACK_TMP=$(mktemp -d)
mkdir -p "$STACK_TMP/fakebin"
printf '#!/bin/sh\necho "fake gdb $*"\n' >"$STACK_TMP/fakebin/gdb"
chmod +x "$STACK_TMP/fakebin/gdb"
run env PATH="$STACK_TMP/fakebin:$PATH" bash -c '
    RESULTS=$2/results SUDO=""
    log() { echo "$*"; }
    eval "$(sed -n "/^STACK_AFTER_RESET=/,/^NO_GDB_NOTED=0/p; /^descendants() {/,/^}/p; /^take_stacks() {/,/^}/p" "$1")"
    timeout 5 bash -c "sleep 4; :" &
    p=$!
    for n in bisect_repro_sycl_01 bisect_repro_sycl_02 bisect_repro_sycl_03 bisect_repro_sycl_blocking_01; do
        take_stacks $n $p waiting
    done
    wait
    ls "$RESULTS/stacks"' _ "$RUNNER" "$STACK_TMP"
assert_exit 0
assert_contains "bisect_repro_sycl_01.waiting.txt"
assert_contains "bisect_repro_sycl_02.waiting.txt"
assert_contains "bisect_repro_sycl_blocking_01.waiting.txt"
assert_not_contains "bisect_repro_sycl_03"
rm -rf -- "$STACK_TMP"

t "the package carries the standalone reproducer and its source (TD-574)"
run grep -c -E "sycl_engine_reset_repro|SyclEngineResetRepro.cpp" "$PACKAGER"
assert_contains "2"

t "a step whose wait gave up gets its stack taken, once, without signalling anything (TD-574)"
STACK_TMP=$(mktemp -d)
mkdir -p "$STACK_TMP/fakebin" "$STACK_TMP/results/work/pass1"
printf '#!/bin/sh\necho "fake gdb $*"\n' >"$STACK_TMP/fakebin/gdb"
chmod +x "$STACK_TMP/fakebin/gdb"
run env PATH="$STACK_TMP/fakebin:$PATH" bash -c '
    RESULTS=$2/results SUDO="" GPU_DUMP_DIR=$2/none
    log() { echo "$*"; }
    eval "$(sed -n "/^STACK_AFTER_RESET=/,/^NO_GDB_NOTED=0/p; /^gpu_dump_since() {/,/^}/p; /^descendants() {/,/^}/p;
        /^take_stacks() {/,/^}/p; /^watch_step() {/,/^}/p" "$1")"
    STACK_AFTER_GIVE_UP=1
    touch "$RESULTS/work/pass1/.step_start"
    (timeout 4 bash -c "echo \"wait gave up (GPU_WAIT_TIMEOUT_SECONDS)\"; exec sleep 30") >"$RESULTS/pass1.log" 2>&1 &
    watch_step pass1 $! "$RESULTS/work/pass1/.step_start"
    wait
    cat "$RESULTS/stacks/pass1.exit.txt"
    ls "$RESULTS/stacks"' _ "$RUNNER" "$STACK_TMP"
assert_exit 0
assert_contains "stack of pass1 (exit): stacks/pass1.exit.txt"
assert_contains "(sleep), exit"
assert_contains "fake gdb -p"
assert_contains "thread apply all bt"
assert_not_contains "pass1.waiting.txt"
rm -rf -- "$STACK_TMP"

t "a reproducer whose log stalls with no reset gets a stalled stack; other steps and reported losses don't (TD-574)"
STACK_TMP=$(mktemp -d)
mkdir -p "$STACK_TMP/fakebin" "$STACK_TMP/results/work" "$STACK_TMP/dumps"
printf '#!/bin/sh\necho "fake gdb $*"\n' >"$STACK_TMP/fakebin/gdb"
chmod +x "$STACK_TMP/fakebin/gdb"
run env PATH="$STACK_TMP/fakebin:$PATH" bash -c '
    RESULTS=$2/results SUDO="" GPU_DUMP_DIR=$2/none
    log() { echo "$*"; }
    eval "$(sed -n "/^STACK_AFTER_RESET=/,/^NO_GDB_NOTED=0/p; /^gpu_dump_since() {/,/^}/p; /^descendants() {/,/^}/p;
        /^take_stacks() {/,/^}/p; /^watch_step() {/,/^}/p" "$1")"
    STACK_AFTER_STALL=1 STACK_AFTER_RESET=1
    watch() {
        mkdir -p "$RESULTS/work/$1"
        touch "$RESULTS/work/$1/.step_start"
        (timeout 3 bash -c "echo \"$2\"; sleep 0.1; ${3:-:}; exec sleep 30") >"$RESULTS/$1.log" 2>&1 &
        watch_step "$1" $! "$RESULTS/work/$1/.step_start"
        wait
    }
    watch bisect_repro_sycl_blocking_01 "[22:41:32.801] progress: 6910 steps"
    watch bisect_control_01 "quiet"
    GPU_DUMP_DIR=$2/dumps
    watch bisect_repro_mkl_01 "[22:32:58.307] FAILED: level_zero backend failed" "touch $GPU_DUMP_DIR/dump.gz"
    ls "$RESULTS/stacks"' _ "$RUNNER" "$STACK_TMP"
assert_exit 0
assert_contains "stack of bisect_repro_sycl_blocking_01 (stalled): stacks/bisect_repro_sycl_blocking_01.stalled.txt"
assert_not_contains "bisect_control_01"
assert_not_contains "bisect_repro_mkl_01"
rm -rf -- "$STACK_TMP"

t "update_gpu_stack.sh --help prints its usage, and rejects an unknown argument"
run bash "$UPDATE" --help
assert_exit 0
assert_contains "rollback"
assert_contains "ppa:kobuk-team/intel-graphics"
run bash "$UPDATE" --bogus
assert_exit 2
assert_contains "Unknown argument"

t "update_gpu_stack.sh refuses an apt plan that touches anything outside the compute runtime"
run bash -c '
    eval "$(sed -n "/^RUNTIME_PACKAGES=/p; /^COMPANION_PACKAGES=/p; /^DBGSYM_PACKAGES=/p; /^outside_runtime() {/,/^}/p" "$1")"
    echo "-- runtime only:"
    printf "Inst libze-intel-gpu1 [26.31] (26.40 PPA)\nInst libigc2:amd64 [2.40] (2.42)\nInst libze1 (1.24)\nInst libigc2-dbgsym (2.42)\nInst libze-dev [1.32] (1.34)\nConf libze1\n" | outside_runtime
    echo "-- with others:"
    printf "Inst libze-intel-gpu1 (26.40)\nInst linux-firmware [1] (2)\nInst libigc-dev (2.42)\nRemv intel-level-zero-gpu [1.3]\n" | outside_runtime' _ "$UPDATE"
assert_exit 0
assert_contains "-- runtime only:
-- with others:
linux-firmware
libigc-dev
intel-level-zero-gpu (removed)"

t "update_gpu_stack.sh's debug archive copies the PPA's entry and its key, with main/debug"
UPD_TMP=$(mktemp -d)
printf "Types: deb\nURIs: https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/\nSuites: noble\nComponents: main\nSigned-By:\n -----BEGIN PGP PUBLIC KEY BLOCK-----\n .\n KEYDATA\n" >"$UPD_TMP/ppa.sources"
printf "# comment\ndeb [signed-by=/k.gpg] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main\n" >"$UPD_TMP/ppa.list"
run bash -c '
    eval "$(sed -n "/^PPA_LIST_PATTERN=/p; /^debug_entry() {/,/^}/p" "$1")"
    debug_entry "$2/ppa.sources"; debug_entry "$2/ppa.list"' _ "$UPDATE" "$UPD_TMP"
assert_exit 0
assert_contains "Components: main/debug"
assert_contains " KEYDATA"
assert_contains "deb [signed-by=/k.gpg] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main/debug"
assert_not_contains "Components: main
"
rm -rf -- "$UPD_TMP"

t "guc_firmware.sh finds the newest GuC on the linux-firmware log page, whatever the row markup"
GUC_TMP=$(mktemp -d)
mkdir -p "$GUC_TMP/bin"
cat >"$GUC_TMP/log.html" <<'HTML'
<tr><td><span class='age-weeks' title='2026-10-30 10:00:00 +0000'>2 weeks</span></td><td><a href='/pub/scm/linux/kernel/git/firmware/linux-firmware.git/commit/xe/bmg_guc_70.bin?h=main&amp;id=f51d6d83dbadf2890b7639be6bd6e35803fd02f5'>xe: Update GUC to v70.83.0 for DG2, MTL, LNL, BMG, PTL, NVL-S/U/H</a></td></tr>
<tr><td><span title='2026-08-05 18:43:33 +0000'>2026-08-05</span></td><td><a href='/pub/scm/linux/kernel/git/firmware/linux-firmware.git/commit/xe/bmg_guc_70.bin?h=main&amp;id=4291fa65d3055297136cad6370e54113e27662ff'>xe: Update GUC to v70.72.1 for BMG, LNL, PTL, NVL-S</a></td></tr>
HTML
printf '#!/bin/sh\ncat "%s/log.html"\n' "$GUC_TMP" >"$GUC_TMP/bin/curl"
chmod +x "$GUC_TMP/bin/curl"
run env PATH="$GUC_TMP/bin:$PATH" bash -c '
    eval "$(sed -n "/^LOG_URL=/p; /^latest_upstream() {/,/^}/p; /^version_newer() {/,/^}/p" "$1")"
    latest_upstream
    version_newer 70.83.0 70.72.1 && echo "83 newer than 72"
    version_newer 70.72.1 70.83.0 || echo "72 not newer than 83"
    version_newer 70.72.1 70.72.1 || echo "same is not newer"' _ "$GUC"
assert_exit 0
assert_contains "f51d6d83dbadf2890b7639be6bd6e35803fd02f5 70.83.0"
assert_contains "83 newer than 72"
assert_contains "72 not newer than 83"
assert_contains "same is not newer"
rm -rf -- "$GUC_TMP"

t "the package carries update_gpu_stack.sh"
run grep -c "update_gpu_stack.sh" "$PACKAGER"
assert_contains "1"

t "prepare_host.sh offers gdb for the stacks"
run grep -c "apt-get install -y gdb" "$PREPARE"
assert_contains "1"

t "xe_debug.sh --help prints its usage; it rejects a bad level and an unknown argument, changing nothing"
run bash "$XE_DEBUG" --help
assert_exit 0
assert_contains "guc_log_level"
run bash "$XE_DEBUG" enable --guc-log-level 9
assert_exit 2
assert_contains "0 to 5"
run bash "$XE_DEBUG" --bogus
assert_exit 2
assert_contains "Unknown argument"
assert_file_not_exists "${REPO_ROOT}/scripts/sycl_gpu_tests/host_state"

t "xe_debug.sh writes the xe module option for the GuC log level"
run bash -c 'eval "$(sed -n "/^conf_line() {/,/^}/p" "$1")"; conf_line 3' _ "$XE_DEBUG"
assert_exit 0
assert_contains "options xe guc_log_level=3"

t "mainline_kernel.sh --help prints its usage; it rejects a bad version and an unknown argument"
run bash "$MAINLINE" --help
assert_exit 0
assert_contains "boot-once"
run bash "$MAINLINE" install 6.x
assert_exit 2
assert_contains "Not a mainline release"
run bash "$MAINLINE" --bogus
assert_exit 2
assert_file_not_exists "${REPO_ROOT}/scripts/sycl_gpu_tests/host_state"

t "mainline_kernel.sh picks the newest release that isn't a release candidate, and its generic packages"
run bash -c '
    eval "$(sed -n "/^latest_release() {/,/^}/p; /^package_files() {/,/^}/p; /^kernel_release() {/,/^}/p" "$1")"
    printf "%s\n" "<a href=\"v6.9.12/\">" "<a href=\"v6.18.4/\">" "<a href=\"v6.19-rc3/\">" "<a href=\"v6.18/\">" | latest_release
    printf "%s\n" "<a href=\"linux-image-unsigned-6.18.4-061804-generic_6.18.4-061804.202611101200_amd64.deb\">" \
        "<a href=\"linux-image-unsigned-6.18.4-061804-lowlatency_6.18.4-061804.202611101200_amd64.deb\">" \
        "<a href=\"linux-modules-6.18.4-061804-generic_6.18.4-061804.202611101200_amd64.deb\">" \
        "<a href=\"linux-headers-6.18.4-061804-generic_6.18.4-061804.202611101200_amd64.deb\">" | package_files >"$2"
    cat "$2"; echo "count $(grep -c . "$2")"
    kernel_release "$(head -1 "$2")"' _ "$MAINLINE" "$(mktemp)"
assert_exit 0
assert_contains "6.18.4"
assert_not_contains "rc3"
assert_contains "count 2"
assert_not_contains "lowlatency"
assert_contains "6.18.4-061804-generic"

t "mainline_kernel.sh finds a kernel's GRUB entry inside the submenu, not its recovery entry"
run bash -c '
    eval "$(sed -n "/^grub_entry_in() {/,/^}/p" "$1")"
    printf "%s\n" "menuentry '"'"'Ubuntu'"'"' --class ubuntu {" "}" \
        "submenu '"'"'Advanced options for Ubuntu'"'"' \$menuentry_id_option '"'"'gnulinux-advanced-x'"'"' {" \
        "	menuentry '"'"'Ubuntu, with Linux 6.18.4-061804-generic (recovery mode)'"'"' --class ubuntu {" \
        "	menuentry '"'"'Ubuntu, with Linux 6.18.4-061804-generic'"'"' --class ubuntu {" "}" | grub_entry_in 6.18.4-061804-generic' _ "$MAINLINE"
assert_exit 0
assert_contains "Advanced options for Ubuntu>Ubuntu, with Linux 6.18.4-061804-generic"
assert_not_contains "recovery"

t "mainline_kernel.sh refuses an apt plan that touches anything but the kernel's two packages"
run bash -c '
    eval "$(sed -n "/^outside_kernel() {/,/^}/p" "$1")"
    printf "%s\n" "Inst linux-image-unsigned-6.18.4-061804-generic (6.18.4 x)" "Inst wireless-regdb (2024 x)" \
        "Remv linux-image-6.17.0-1025-oem" | outside_kernel linux-image-unsigned-6.18.4-061804-generic linux-modules-6.18.4-061804-generic' _ "$MAINLINE"
assert_exit 0
assert_contains "wireless-regdb"
assert_contains "linux-image-6.17.0-1025-oem (removed)"
assert_not_contains "unsigned"

t "mainline_kernel.sh's signing key is for code signing, not DKMS's module-only usage that shim refuses for a kernel"
KEYDIR="$(mktemp -d)"
run bash -c 'sudo() { "$@"; }; eval "$(sed -n "/^new_key() {/,/^}/p" "$1")"; new_key "$2" &&
    openssl x509 -in "$2/MOK.pem" -noout -ext extendedKeyUsage,basicConstraints &&
    openssl x509 -inform DER -in "$2/MOK.der" -noout -subject' _ "$MAINLINE" "$KEYDIR"
assert_exit 0
assert_contains "Code Signing"
assert_contains "CA:FALSE"
assert_not_contains "1.3.6.1.4.1.2312.16.1.2"
assert_contains "adai mainline kernel test key"

t "mainline_kernel.sh reads the key's state from mokutil: enrolled, queued or absent"
run bash -c '
    eval "$(sed -n "/^fingerprint_hex() {/,/^}/p; /^mok_state_of() {/,/^}/p" "$1")"
    fp=$(echo "sha1 Fingerprint=43:95:36:BB:05:E7:14:C4:C5:AD:8E:FD:F8:55:75:E6:A7:59:B5:4D" | fingerprint_hex)
    echo "fp $fp"
    echo "a $(mok_state_of "MOK.der is already enrolled" "" "$fp")"
    echo "b $(mok_state_of "MOK.der is not enrolled" "$(printf "[key 1]\nSHA1 Fingerprint: 43:95:36:bb:05:e7:14:c4:c5:ad:8e:fd:f8:55:75:e6:a7:59:b5:4d\n")" "$fp")"
    echo "c $(mok_state_of "MOK.der is not enrolled" "MokNew is empty" "$fp")"
    echo "d $(mok_state_of "MOK.der is not enrolled" "SHA1 Fingerprint: 00:11" "")"' _ "$MAINLINE"
assert_exit 0
assert_contains "fp 439536bb05e714c4c5ad8efdf85575e6a759b54d"
assert_contains "a enrolled"
assert_contains "b pending"
assert_contains "c absent"
assert_contains "d absent"

# A real sign-and-verify of a PE image, when this machine has sbsigntool and an EFI binary to copy.
EFI_SAMPLE=$(ls /usr/lib/shim/fbx64.efi /usr/lib/grub/x86_64-efi-signed/grubx64.efi.signed 2>/dev/null | head -1)
if command -v sbsign >/dev/null && command -v sbverify >/dev/null && [[ -r "$EFI_SAMPLE" ]]; then
    t "mainline_kernel.sh signs a kernel image with its key, keeps the image's mode, and leaves an already-signed one alone"
    cp "$EFI_SAMPLE" "$KEYDIR/vmlinuz-test"
    chmod 0600 "$KEYDIR/vmlinuz-test"
    run bash -c '
        sudo() { case "$1" in mokutil) [[ $2 == --test-key ]] && echo "MOK.der is already enrolled" ;; *) "$@" ;; esac; }
        mokutil() { echo "MokNew is empty"; }
        eval "$(sed -n "/^fingerprint_hex() {/,/^}/p; /^mok_state_of() {/,/^}/p; /^mok_state() {/,/^}/p; /^sign_kernel() {/,/^}/p; /^mok_reminder() {/,/^}/p" "$1")"
        BOOT_DIR=$2; MOK_DIR=$2; MOK_KEY=$2/MOK.priv; MOK_PEM=$2/MOK.pem; MOK_DER=$2/MOK.der
        sign_kernel test && sbverify --cert "$MOK_PEM" "$2/vmlinuz-test" && stat -c "mode %a" "$2/vmlinuz-test" && sign_kernel test' _ "$MAINLINE" "$KEYDIR"
    assert_exit 0
    assert_contains "Signed"
    assert_contains "Signature verification OK"
    assert_contains "mode 600"
    assert_contains "already signed with the key"
    assert_contains "already trusts the key"
fi
rm -rf -- "$KEYDIR"

t "mainline_kernel.sh rebuilds a package whose scripts pass run-parts /usr/share/kernel/X.d for a one-directory run-parts, contents unchanged"
FITDIR="$(mktemp -d)"
mkdir -p "$FITDIR/src/DEBIAN" "$FITDIR/src/boot" "$FITDIR/plain/DEBIAN" "$FITDIR/plain/boot"
printf "Package: linux-image-unsigned-9.9.9-090909-generic\nVersion: 9.9.9\nArchitecture: amd64\nMaintainer: test <t@example.com>\nDescription: test\n" >"$FITDIR/src/DEBIAN/control"
sed 's/image-unsigned/modules/' "$FITDIR/src/DEBIAN/control" >"$FITDIR/plain/DEBIAN/control"
printf '#!/bin/sh -e\nversion=9.9.9\nimage_path=/boot/vmlinuz-$version\nif [ -d /etc/kernel/preinst.d ] || [ -d /usr/share/kernel/preinst.d ]; then\n    DEB_MAINT_PARAMS="$*" run-parts --report --exit-on-error --arg=$version \\\n        --arg=$image_path /etc/kernel/preinst.d /usr/share/kernel/preinst.d\nfi\n' >"$FITDIR/src/DEBIAN/preinst"
sed 's/preinst/postrm/g' "$FITDIR/src/DEBIAN/preinst" >"$FITDIR/src/DEBIAN/postrm"
printf '#!/bin/sh -e\nrun-parts --report /etc/kernel/postinst.d\n' >"$FITDIR/plain/DEBIAN/postinst"
chmod 0755 "$FITDIR"/src/DEBIAN/preinst "$FITDIR"/src/DEBIAN/postrm "$FITDIR"/plain/DEBIAN/postinst
head -c 4096 /dev/urandom >"$FITDIR/src/boot/vmlinuz-9.9.9"
cp "$FITDIR/src/boot/vmlinuz-9.9.9" "$FITDIR/plain/boot/System.map-9.9.9"
dpkg-deb --root-owner-group -b "$FITDIR/src" "$FITDIR/image.deb" >/dev/null
dpkg-deb --root-owner-group -b "$FITDIR/plain" "$FITDIR/modules.deb" >/dev/null
run bash -c '
    eval "$(sed -n "/^one_dir_run_parts() {/,/^}/p; /^drop_share_hook_dirs() {/,/^}/p; /^names_share_hook_dirs() {/,/^}/p; /^fit_debs() {/,/^}/p" "$1")"
    if one_dir_run_parts; then
        echo "one directory"
        out=$(fit_debs "$2" image.deb modules.deb) || exit 1
        echo "$out"
        fitted=$(head -1 <<<"$out")
        dpkg-deb -I "$fitted" preinst; dpkg-deb -I "$fitted" postrm
        cmp <(dpkg-deb --fsys-tarfile "$2/image.deb") <(dpkg-deb --fsys-tarfile "$fitted") && echo "same contents"
        dpkg-deb -c "$fitted" | grep -c "root/root" && dpkg-deb -f "$fitted" Package
        sh -n <(dpkg-deb -I "$fitted" preinst) && echo "preinst parses"
        again=$(fit_debs "$2" image.deb modules.deb 2>&1) && echo "rebuilt again: $(grep -c Rebuilding <<<"$again")"
    else
        echo "multi directory: only checking that nothing is rebuilt"
        fit_debs "$2" image.deb modules.deb
    fi' _ "$MAINLINE" "$FITDIR"
assert_exit 0
if [[ "$RUN_OUT" == *"one directory"* ]]; then
    assert_contains "Rebuilding image.deb"
    assert_not_contains "Rebuilding modules.deb"
    assert_contains "$FITDIR/fitted/image.deb"
    assert_contains "$FITDIR/modules.deb"
    assert_contains "--arg=\$image_path /etc/kernel/preinst.d
fi"
    assert_contains "--arg=\$image_path /etc/kernel/postrm.d
fi"
    assert_not_contains "/etc/kernel/preinst.d /usr/share"
    assert_contains "same contents"
    assert_contains "linux-image-unsigned-9.9.9-090909-generic"
    assert_contains "preinst parses"
    assert_contains "rebuilt again: 0"
fi
rm -rf -- "$FITDIR"

t "mainline_kernel.sh purges only what a failed attempt left unfinished, after fixing its installed scripts' run-parts calls"
run bash -c '
    eval "$(sed -n "/^repair_unfinished() {/,/^}/p" "$1")"
    YES=1
    ask() { echo "$1 [y/N] y (--yes)"; }
    dpkg-query() { case "${@: -1}" in img) printf "iHR" ;; mods) printf "iU " ;; ok) printf "ii " ;; rc) printf "rc " ;; *) return 1 ;; esac; }
    INFO=$2
    sudo() {
        case "$1" in
            test) shift; test "$1" "${2/\/var\/lib\/dpkg\/info/$INFO}" ;;
            sed) shift; local a=("$@"); a[-1]=${a[-1]/\/var\/lib\/dpkg\/info/$INFO}; sed "${a[@]}" ;;
            dpkg) echo "dpkg ${*:2}" ;;
        esac
    }
    printf "    --arg=\$image_path /etc/kernel/postrm.d /usr/share/kernel/postrm.d\n" >"$INFO/img.postrm"
    repair_unfinished img mods ok rc missing && cat "$INFO/img.postrm"
    echo "---"; repair_unfinished ok rc missing && echo "nothing to do"' _ "$MAINLINE" "$(mktemp -d)"
assert_exit 0
assert_contains "Left unfinished by an earlier attempt: img mods"
assert_contains "dpkg --purge --force-remove-reinstreq img mods"
assert_contains "--arg=\$image_path /etc/kernel/postrm.d
---"
assert_contains "nothing to do"

t "mainline_kernel.sh reads the queued keys as root (MokNew is root-only), so a queued key isn't reported absent"
run bash -c '
    sudo() { case "$1" in mokutil) [[ $2 == --test-key ]] && echo "MOK.der is not enrolled"; [[ $2 == --list-new ]] && echo "SHA1 Fingerprint: 43:95"; true ;; test) true ;; openssl) echo "sha1 Fingerprint=43:95" ;; esac; }
    mokutil() { echo "MokNew is empty"; }
    eval "$(sed -n "/^fingerprint_hex() {/,/^}/p; /^mok_state_of() {/,/^}/p; /^mok_state() {/,/^}/p" "$1")"
    MOK_DER=/x; mok_state' _ "$MAINLINE"
assert_exit 0
assert_contains "pending"

t "mainline_kernel.sh lists only the installed kernels newer than the running one that Canonical signed"
run bash -c '
    eval "$(sed -n "/^canonical_kernels_newer() {/,/^}/p" "$1")"
    BOOT_DIR=$2
    touch "$BOOT_DIR/vmlinuz-6.8.0-149-generic" "$BOOT_DIR/vmlinuz-6.17.0-1025-oem" "$BOOT_DIR/vmlinuz-7.0.0-43-generic" \
        "$BOOT_DIR/vmlinuz-7.3.5-070305-generic"
    sudo() {
        [[ $1 == sbverify ]] || return 1
        case "${@: -1}" in
            *7.3.5*) echo "subject: /CN=adai mainline kernel test key" ;;
            *) echo "subject: /C=GB/ST=Isle of Man/O=Canonical Ltd./CN=Canonical Ltd. Secure Boot Signing (2022 v1)" ;;
        esac
    }
    echo "[$(canonical_kernels_newer 6.17.0-1025-oem | tr "\n" " ")]"' _ "$MAINLINE" "$(mktemp -d)"
assert_exit 0
assert_contains "[7.0.0-43-generic ]"

t "mainline_kernel.sh asks whether someone is at the console before queuing a key or arming a kernel that needs one"
run bash -c '
    eval "$(sed -n "/^console_check() {/,/^}/p" "$1")"
    canonical_kernels_newer() { echo 7.0.0-43-generic; }
    grub_entry() { echo "Advanced options for Ubuntu>Ubuntu, with Linux $1"; }
    uname() { echo 6.17.0-1025-oem; }
    ask() { echo "$1"; return 1; }
    console_check pending; echo "rc $?"
    ask() { echo "$1"; return 0; }
    console_check absent; echo "rc $?"' _ "$MAINLINE"
assert_exit 0
assert_contains "nothing over SSH can answer it"
assert_contains "sudo grub-reboot \"Advanced options for Ubuntu>Ubuntu, with Linux 7.0.0-43-generic\""
assert_contains "Is someone at the console for the next boot?"
assert_contains "Not armed: the next boot is 6.17.0-1025-oem as before."
assert_contains "sudo mokutil --revoke-import drops it.
rc 1"
assert_contains "rc 0"
run bash -c 'sed -n "/^sign_kernel() {/,/^}/p" "$1" | grep -c "\[\[ \"\$state\" == enrolled \]\] || console_check \"\$state\" || return 1"' _ "$MAINLINE"
assert_contains "1"

t "mainline_kernel.sh runs every mokutil that asks for a password on the terminal, not through its log pipe"
run bash -c 'grep -n -E "sudo mokutil --(import|delete|revoke-import)" "$1" | grep -v -E "^[^:]+: *(#|echo)" | grep -v -F "</dev/tty >/dev/tty 2>&1"' _ "$MAINLINE"
assert_exit 1
run bash -c 'grep -c -E "sudo mokutil --(import|delete|revoke-import) .*</dev/tty >/dev/tty 2>&1" "$1"' _ "$MAINLINE"
assert_contains "3"

t "xe_debug.sh and mainline_kernel.sh never pipe into grep -q, which fails a pipefail check on a match (SIGPIPE)"
run bash -c 'grep -n -E "\| *grep -q" "$1" "$2" | grep -v -E "^[^:]+:[0-9]+: *#"' _ "$XE_DEBUG" "$MAINLINE"
assert_exit 1

t "the package carries xe_debug.sh and mainline_kernel.sh"
run grep -c "update_gpu_stack.sh xe_debug.sh mainline_kernel.sh" "$PACKAGER"
assert_contains "1"

t "the packaged pairs file is JSON Lines with input and response"
run python3 -c "import json,sys; [json.loads(l)['response'] for l in open(sys.argv[1])]" \
    "${REPO_ROOT}/scripts/sycl_gpu_tests/pairs.jsonl"
assert_exit 0

harness_report
exit $?
