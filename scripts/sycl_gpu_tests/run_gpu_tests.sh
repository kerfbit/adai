#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.5.1
# @adai-reviewed: 2026-11-13

# Runs the packaged SYCL GPU tests on the GPU host (ai-machine) and packs the results into one
# archive to copy back. Built into a package by scripts/package_sycl_gpu_tests.sh; see README.md
# beside this script. Run ./prepare_host.sh first. This script changes nothing outside this
# directory: it stops no services, and every trainer run uses a private config with no MNS,
# registry or metrics. It uses sudo only to read the kernel log, the GPU firmware version and the
# driver's crash dumps (freeing each one it copies, so the next engine reset can write a new one)
# and to see other users' GPU processes. With --engine-tuning it also changes the compute engine's
# preempt timeout and time slice in sysfs for some bisection passes, restoring them after each pass
# and when the script exits.
#
#   ./run_gpu_tests.sh [--quick] [--skip-tests] [--skip-benchmark] [--skip-training] [--skip-soak]
#                      [--soak-passes N] [--soak-epochs N] [--lejepa-only] [--skip-lejepa]
#                      [--lejepa-soak-passes N] [--lejepa-soak-epochs N] [--bisect-passes N]
#                      [--skip-bisect] [--bisect-only] [--engine-tuning] [--runtime-dir DIR]
#                      [--allow-shared-gpu] [--allow-runtime-mismatch] [--no-sudo]
#                      [--allow-cpu-device]
#
# --quick             only the GPU-specific test binaries, a 3-repeat benchmark, a 1-repeat decode
#                     benchmark, one seed pair and a 3-pass soak
# --soak-passes N     training passes per Level Zero variant in the soak (default 20; 0 = no soak)
# --soak-epochs N     epochs per soak pass (default 10)
# --lejepa-only       only the device check and the LeJEPA world-model passes (TD-572)
# --skip-lejepa       no LeJEPA world-model passes
# --lejepa-soak-passes N
#                     LeJEPA soak passes (default 20, --quick 3; 0 = lifecycle checks only)
# --lejepa-soak-epochs N
#                     epochs per LeJEPA soak pass (default 5)
# --bisect-passes N   passes per variant in the engine-reset bisection (TD-574; default 20,
#                     --quick 3; 0 = none)
# --skip-bisect       no engine-reset bisection
# --bisect-only       only the device check and the engine-reset bisection
# --engine-tuning     add two bisection variants that change a compute engine setting in sysfs
#                     for their passes only (needs sudo): long_preempt (preempt timeout 5 s) and
#                     long_timeslice (time slice 100 ms); the original values are restored after
#                     every pass and when the script exits
# --runtime-dir DIR   the oneAPI runtime every binary must load (default /opt/intel/oneapi/redist/lib,
#                     where Intel's intel-oneapi-runtime-* packages install it)
# --allow-shared-gpu  run even while other processes have the GPU open
# --allow-runtime-mismatch
#                     run even when the runtime is older than the build or libraries load from
#                     several directories
# --no-sudo           don't use sudo (no kernel log, no firmware version, own processes only)
# --allow-cpu-device  also accept oneAPI's CPU device (ADAI_SYCL_ALLOW_CPU_DEVICE=1); for trying
#                     the package on a machine without a GPU, never on ai-machine

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/bin"
# shellcheck source=gpu_host_lib.sh
source "$HERE/gpu_host_lib.sh"
QUICK=0
RUN_TESTS=1
RUN_BENCHMARK=1
RUN_TRAINING=1
SOAK_PASSES=20
SOAK_EPOCHS=10
SOAK_PASSES_SET=0
RUN_LEJEPA=1
LEJEPA_SOAK_PASSES=20
LEJEPA_SOAK_EPOCHS=5
LEJEPA_SOAK_SET=0
BISECT_PASSES=20
BISECT_SET=0
ENGINE_TUNING=0
RUNTIME_DIR=/opt/intel/oneapi/redist/lib
ALLOW_SHARED=0
ALLOW_MISMATCH=0
USE_SUDO=1
ALLOW_CPU=0

usage() {
    sed -n '/^#   \.\/run_gpu_tests/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    sed -n '/^# --quick/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}
need_number() {
    [[ "${2:-}" =~ ^[0-9]+$ ]] || {
        echo "$1 needs a whole number" >&2
        exit 2
    }
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1 ;;
        --skip-tests) RUN_TESTS=0 ;;
        --skip-benchmark) RUN_BENCHMARK=0 ;;
        --skip-training) RUN_TRAINING=0 ;;
        --skip-soak)
            SOAK_PASSES=0
            SOAK_PASSES_SET=1
            ;;
        --soak-passes)
            need_number "$1" "${2:-}"
            SOAK_PASSES="$2"
            SOAK_PASSES_SET=1
            shift
            ;;
        --soak-epochs)
            need_number "$1" "${2:-}"
            SOAK_EPOCHS="$2"
            shift
            ;;
        --runtime-dir)
            [[ $# -ge 2 ]] || {
                echo "--runtime-dir needs a value" >&2
                exit 2
            }
            RUNTIME_DIR="$2"
            shift
            ;;
        --lejepa-only)
            RUN_TESTS=0
            RUN_BENCHMARK=0
            RUN_TRAINING=0
            BISECT_PASSES=0
            BISECT_SET=1
            ;;
        --skip-lejepa) RUN_LEJEPA=0 ;;
        --bisect-passes)
            need_number "$1" "${2:-}"
            BISECT_PASSES="$2"
            BISECT_SET=1
            shift
            ;;
        --skip-bisect)
            BISECT_PASSES=0
            BISECT_SET=1
            ;;
        --engine-tuning) ENGINE_TUNING=1 ;;
        --bisect-only)
            RUN_LEJEPA=0
            RUN_TESTS=0
            RUN_BENCHMARK=0
            RUN_TRAINING=0
            ;;
        --lejepa-soak-passes)
            need_number "$1" "${2:-}"
            LEJEPA_SOAK_PASSES="$2"
            LEJEPA_SOAK_SET=1
            shift
            ;;
        --lejepa-soak-epochs)
            need_number "$1" "${2:-}"
            LEJEPA_SOAK_EPOCHS="$2"
            shift
            ;;
        --allow-shared-gpu) ALLOW_SHARED=1 ;;
        --allow-runtime-mismatch) ALLOW_MISMATCH=1 ;;
        --no-sudo) USE_SUDO=0 ;;
        --allow-cpu-device) ALLOW_CPU=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done
[[ $QUICK == 1 && $SOAK_PASSES_SET == 0 ]] && SOAK_PASSES=3
[[ $QUICK == 1 && $LEJEPA_SOAK_SET == 0 ]] && LEJEPA_SOAK_PASSES=3
[[ $QUICK == 1 && $BISECT_SET == 0 ]] && BISECT_PASSES=3

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_START=$(date +%s)
HOST="$(hostname -s 2>/dev/null || hostname)"
RESULTS_NAME="adai-gpu-results-${HOST}-${STAMP}"
RESULTS="$HERE/results/$RESULTS_NAME"
mkdir -p "$RESULTS/xml" "$RESULTS/work"
SUMMARY="$RESULTS/summary.txt"

log() {
    echo "$*" | tee -a "$SUMMARY"
}

# Only this script's own settings reach the binaries: drop any ADAI config keys and oneAPI or
# Level Zero overrides inherited from the shell (a deployed host may export MNS, registry or GPU
# settings). The originals are recorded in environment.txt.
INHERITED=$(env | grep -E '^(ONEAPI_DEVICE_SELECTOR|SYCL_|UR_|ZE_|LD_LIBRARY_PATH)=' || true)
while IFS= read -r name; do
    unset "$name"
done < <(env | cut -d= -f1 | grep -E '^(NAME_SERVICE|REGISTRY|METRICS|MODEL_|SEED$|GPU_|DATASET|RUN_|SESSION_DIR|VOCAB_PATH|LOG_|TRAINER_|WORLD_MODEL|HIPPOCAMPAL|NUM_|D_MODEL|D_FF|MAX_SEQ|BATCH|LEARNING|DROPOUT|EARLY_STOPPING|AUTO_SAVE|FTP|ADAI_|ONEAPI_DEVICE_SELECTOR$|SYCL_|UR_|ZE_)' || true)
if [[ "$ALLOW_CPU" == 1 ]]; then
    export ADAI_SYCL_ALLOW_CPU_DEVICE=1
fi
export GTEST_COLOR=no

SUDO=""
if [[ $USE_SUDO == 1 ]]; then
    echo "sudo is used to read the kernel log and GPU firmware version, to see other GPU users and (with --engine-tuning) to change the compute engine settings."
    if sudo -v; then
        SUDO="sudo -n"
        # Keep the sudo timestamp fresh for the length of the run.
        (while kill -0 "$$" 2>/dev/null; do
            sudo -n -v 2>/dev/null
            sleep 50
        done) &
        KEEPALIVE=$!
        trap 'kill "$KEEPALIVE" 2>/dev/null' EXIT
    else
        echo "WARNING: sudo failed; continuing without the kernel log or firmware version."
    fi
fi

# kernel_log SINCE [UNTIL]: kernel messages between two epoch times.
kernel_log() {
    if [[ -n "$SUDO" ]]; then
        $SUDO journalctl -k --no-pager -o short-iso-precise --since "@$1" ${2:+--until "@$2"} 2>/dev/null
    else
        journalctl -k --no-pager -o short-iso-precise --since "@$1" ${2:+--until "@$2"} 2>/dev/null ||
            dmesg 2>/dev/null
    fi
}
# What the xe driver logs around a GPU hang or device loss.
# Not "coredump has been deleted": capture_coredumps() causes that line itself when it frees a dump,
# so it lands in the next pass's window (TD-574).
KERNEL_EVENTS='Engine reset|Timedout job|wedged|coredump has been created|Fault response|CAT error|GT[0-9]+: .*(reset|hang)'

BUILD_ONEAPI=$(sed -n 's/^oneapi: //p' "$HERE/MANIFEST.txt" 2>/dev/null)
if [[ -z "$BUILD_ONEAPI" ]]; then
    BUILD_ONEAPI=$(grep -o '([0-9]\{4\}\.[0-9]*\.[0-9]*' "$HERE/MANIFEST.txt" 2>/dev/null | head -1 | tr -d '(')
fi

# Every binary loads the oneAPI runtime from RUNTIME_DIR: ahead of the linker cache, which on
# ai-machine also lists /opt/adai/lib (the deployed service's older bundled runtime).
if [[ -d "$RUNTIME_DIR" ]]; then
    export LD_LIBRARY_PATH="$RUNTIME_DIR"
fi
RUNTIME_VERSION=$(dpkg-query -W -f='${Version}' intel-oneapi-runtime-dpcpp-cpp 2>/dev/null)

# ── 1. Environment ────────────────────────────────────────────────────────────

{
    echo "== package"
    cat "$HERE/MANIFEST.txt" 2>/dev/null
    echo
    echo "== host"
    date -u
    uname -a
    echo "kernel command line: $(cat /proc/cmdline 2>/dev/null)"
    grep -m1 'model name' /proc/cpuinfo
    echo "avx2/fma: $(grep -m1 -o -w -e avx2 /proc/cpuinfo) $(grep -m1 -o -w -e fma /proc/cpuinfo)"
    echo "inherited oneAPI/Level Zero settings (cleared for this run):"
    echo "${INHERITED:-  none}" | sed 's/^/  /'
    echo "ADAI_SYCL_ALLOW_CPU_DEVICE=${ADAI_SYCL_ALLOW_CPU_DEVICE:-<unset>}"
    echo
    echo "== oneAPI runtime"
    echo "built with: ${BUILD_ONEAPI:-unknown}"
    echo "runtime dir: $RUNTIME_DIR"
    echo "intel-oneapi-runtime-dpcpp-cpp: ${RUNTIME_VERSION:-not installed}"
    echo
    echo "== GPU"
    lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' || echo "(lspci unavailable)"
    cat /sys/module/xe/version 2>/dev/null || true
    ls -l /dev/dri 2>/dev/null || true
    id
    echo "xe parameters:"
    for p in /sys/module/xe/parameters/*; do
        [[ -r "$p" ]] && echo "  $(basename "$p")=$(cat "$p" 2>/dev/null)"
    done
    echo "compute engine job timeouts (ms):"
    for f in /sys/class/drm/card*/device/tile*/gt*/engines/ccs/job_timeout_ms; do
        [[ -r "$f" ]] && echo "  $f: $(cat "$f")"
    done
    echo
    echo "== GPU firmware (GuC/HuC)"
    if [[ -n "$SUDO" ]]; then
        $SUDO sh -c 'for f in /sys/kernel/debug/dri/*/gt*/uc/guc_info /sys/kernel/debug/dri/*/gt*/uc/huc_info; do
            [ -r "$f" ] && { echo "-- $f"; head -12 "$f"; }; done' 2>/dev/null
        $SUDO journalctl -k -b --no-pager 2>/dev/null | grep -i -E '(guc|huc|dmc).*(firmware|version)|firmware.*(guc|huc)' | head -10
    else
        echo "(needs sudo)"
    fi
    echo
    echo "== sycl-ls"
    command -v sycl-ls >/dev/null && timeout 60 sycl-ls 2>&1 || echo "(sycl-ls not on PATH)"
    echo
    echo "== oneAPI and GPU packages"
    dpkg -l 2>/dev/null | grep -iE 'intel-oneapi-runtime|level-zero|intel-opencl|libigc|intel-igc|libigdgmm|intel-gpu|xe-' || true
    echo
    echo "== other GPU users"
    gpu_report
    echo
    echo "== host preparation (prepare_host.sh)"
    latest_prepare=$(ls -t "$HERE"/host_state/prepare_*.txt 2>/dev/null | head -1)
    if [[ -n "$latest_prepare" ]]; then
        echo "$latest_prepare"
        cat "$latest_prepare"
    else
        echo "not run"
    fi
} >"$RESULTS/environment.txt" 2>&1

log "ADAI SYCL GPU tests on $HOST, $STAMP"
log "Package: $(head -1 "$HERE/MANIFEST.txt" 2>/dev/null)"

if ! grep -q -w avx2 /proc/cpuinfo || ! grep -q -w fma /proc/cpuinfo; then
    log "FATAL: this CPU has no AVX2/FMA; the binaries are built for x86-64-v3."
    exit 1
fi

missing_libs() {
    ldd "$BIN/incremental_trainer" 2>/dev/null | grep 'not found'
}
if [[ -n "$(missing_libs)" && ! -d "$RUNTIME_DIR" && -f /opt/intel/oneapi/setvars.sh ]]; then
    # setvars.sh reads unset variables, which `set -u` would turn into an exit.
    set +u
    # shellcheck source=/dev/null
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi
if [[ -n "$(missing_libs)" ]]; then
    log "FATAL: oneAPI runtime libraries not found:"
    missing_libs | tee -a "$SUMMARY"
    log "Run ./prepare_host.sh to install the oneAPI runtime the build needs (MANIFEST.txt)."
    exit 1
fi
ldd "$BIN/incremental_trainer" >"$RESULTS/ldd.txt" 2>&1

# The runtime must be one consistent oneAPI release, at least the build's.
RUNTIME_PROBLEMS=()
LIB_DIRS=$(grep -E "$ONEAPI_LIB_PATTERN" "$RESULTS/ldd.txt" | awk '$3 ~ /^\// {print $3}' |
    xargs -r -n1 dirname | xargs -r -n1 realpath -m | sort -u)
if [[ "$(grep -c . <<<"$LIB_DIRS")" -gt 1 ]]; then
    RUNTIME_PROBLEMS+=("oneAPI libraries load from several directories: $(tr '\n' ' ' <<<"$LIB_DIRS")")
fi
if [[ -n "$LIB_DIRS" && "$LIB_DIRS" != "$(realpath -m "$RUNTIME_DIR")" ]]; then
    RUNTIME_PROBLEMS+=("oneAPI libraries don't load from $RUNTIME_DIR")
fi
if [[ -z "$RUNTIME_VERSION" ]]; then
    RUNTIME_PROBLEMS+=("intel-oneapi-runtime-dpcpp-cpp isn't installed, so the runtime's version is unknown")
elif [[ -n "$BUILD_ONEAPI" ]] && ! oneapi_version_at_least "$RUNTIME_VERSION" "$BUILD_ONEAPI"; then
    RUNTIME_PROBLEMS+=("the runtime ($RUNTIME_VERSION) is older than the build ($BUILD_ONEAPI)")
fi
log "oneAPI runtime: ${RUNTIME_VERSION:-unknown} from $(tr '\n' ' ' <<<"${LIB_DIRS:-?}")(built with ${BUILD_ONEAPI:-unknown})"
if [[ ${#RUNTIME_PROBLEMS[@]} -gt 0 ]]; then
    for problem in "${RUNTIME_PROBLEMS[@]}"; do
        log "  $problem"
    done
    if [[ $ALLOW_MISMATCH == 0 ]]; then
        log "FATAL: run ./prepare_host.sh, or pass --allow-runtime-mismatch to test anyway."
        exit 1
    fi
    log "  continuing (--allow-runtime-mismatch)"
fi

# Nothing else may use the GPU: other work shares its memory and engines, skews the benchmark,
# and was running during the November 10 device loss.
OTHERS=$(gpu_work_pids | tr '\n' ' ')
if [[ -n "${OTHERS// /}" ]]; then
    log "Other processes have the GPU open:"
    for pid in $OTHERS; do
        log "  $(describe_pid "$pid")"
    done
    if [[ $ALLOW_SHARED == 0 ]]; then
        log "FATAL: run ./prepare_host.sh to stop them, or pass --allow-shared-gpu to share the GPU."
        exit 1
    fi
    log "  continuing with a shared GPU (--allow-shared-gpu)"
elif [[ -z "$SUDO" ]]; then
    log "No other GPU users visible (without sudo only this user's processes are)."
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# capture_coredumps NAME: copies each xe crash dump the driver wrote (one per engine reset, held
# until read and freed, or for 5 minutes) to coredumps/NAME_<card>.txt, then frees it so the next
# reset can write one. TD-574: the dump records the engine and context state of the hung job.
COREDUMP_MAX_BYTES=$((64 * 1024 * 1024))
capture_coredumps() {
    [[ -n "$SUDO" ]] || return 0
    local f card out
    for f in /sys/class/drm/card*/device/devcoredump/data; do
        [[ -e "$f" ]] || continue
        card=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
        mkdir -p "$RESULTS/coredumps"
        out="$RESULTS/coredumps/${1}_${card}.txt"
        $SUDO head -c "$COREDUMP_MAX_BYTES" "$f" >"$out" 2>/dev/null
        $SUDO sh -c "echo 1 > '$f'" 2>/dev/null
        echo "$1 ${card}: $(wc -c <"$out") bytes" >>"$RESULTS/coredumps/index.txt"
    done
}

FAILURES=0
declare -a ROWS

# step NAME TIMEOUT_SECONDS COMMAND...: runs COMMAND in work/NAME, output in NAME.log.
step() {
    step_expect "$1" "$2" 0 "${@:3}"
}

# step_expect NAME TIMEOUT_SECONDS EXPECTED_RC COMMAND...: as step, but EXPECTED_RC is the pass
# result (a LeJEPA resume with nothing pending exits 3).
step_expect() {
    local name="$1" limit="$2" expect="$3"
    shift 3
    local dir="$RESULTS/work/$name"
    mkdir -p "$dir"
    local start end rc
    start=$(date +%s)
    (cd "$dir" && timeout -k 30 "$limit" "$@") >"$RESULTS/$name.log" 2>&1
    rc=$?
    end=$(date +%s)
    capture_coredumps "$name"
    local note=""
    [[ $rc -eq 124 || $rc -eq 137 ]] && note=" (timed out after ${limit}s)"
    local skipped
    skipped=$(grep -c '^\[  SKIPPED \]' "$RESULTS/$name.log" 2>/dev/null)
    [[ "${skipped:-0}" -gt 0 ]] && note+=" (${skipped} SKIPPED lines)"
    [[ $expect -ne 0 ]] && note+=" (expected rc=$expect)"
    ROWS+=("$(printf '%-34s rc=%-4s %6ss%s' "$name" "$rc" "$((end - start))" "$note")")
    if [[ $rc -ne $expect ]]; then
        FAILURES=$((FAILURES + 1))
    fi
    log "  ${ROWS[-1]}"
    return $rc
}

# probe_device NAME: whether the GPU still runs work (TD-566's device check, as a test).
probe_device() {
    step "$1" 120 "$BIN/gpuutilsTests" \
        --gtest_filter='GPUManagerTest.InitializeMatchesProbe:GPUManagerTest.DeviceUsableOnlyOnAnInitializedWorkingDevice' \
        --gtest_output=xml:"$RESULTS/xml/$1.xml"
}

# ── 2. Device check ───────────────────────────────────────────────────────────

log ""
log "Device check"
step device_info 120 "$BIN/gpuutilsTests" --gtest_filter='GPUManagerTest.ProbeDiagnosticIsNonEmpty:GPUManagerTest.InitializeMatchesProbe' \
    --gtest_output=xml:"$RESULTS/xml/device_info.xml"
if grep -q 'No GPU device present' "$RESULTS/device_info.log"; then
    log "FATAL: no SYCL GPU device found (see device_info.log and environment.txt). Nothing else would test the GPU."
    FATAL_NO_DEVICE=1
fi

# ── 3. GPU test binaries ──────────────────────────────────────────────────────

if [[ $RUN_TESTS == 1 && -z "${FATAL_NO_DEVICE:-}" ]]; then
    log ""
    log "Test binaries"
    if [[ $QUICK == 1 ]]; then
        TESTS=(gpuutilsTests matrixgpuTd003Tests gpuOptimizerTests deviceSelectionTests gpuDebugSyncTests)
    else
        TESTS=(deviceSelectionTests gpuutilsTests matrixgpuTd003Tests gpuOptimizerTests matrixTests
            feedforwardTests multiheadattentionTests crossattentionTests encoderdecoderTests initRngTests
            generationQualityAsyncTests chatbottrainerTests incrementaltrainerTests chatbotapiTests
            gpuDebugSyncTests)
    fi
    for test in "${TESTS[@]}"; do
        [[ -x "$BIN/$test" ]] || {
            log "  $test: missing from the package"
            FAILURES=$((FAILURES + 1))
            continue
        }
        step "$test" 3600 "$BIN/$test" --gtest_output=xml:"$RESULTS/xml/$test.xml"
    done
fi

# ── 4. TD-171 and TD-050 benchmarks ───────────────────────────────────────────

if [[ $RUN_BENCHMARK == 1 && -z "${FATAL_NO_DEVICE:-}" ]]; then
    log ""
    log "TD-171 stacked multiply benchmark (results in stacked_gemm_benchmark.log)"
    repeats=15
    [[ $QUICK == 1 ]] && repeats=3
    step stacked_gemm_benchmark 5400 "$BIN/stacked_gemm_benchmark" 768 3072 12 "$repeats"

    # TD-050/TD-033: the GPU KV-cached decode, checked step by step against full recompute and
    # timed against it and the CPU, at a small shape and at config.trainer.conf's architecture.
    log ""
    log "TD-050 GPU decode benchmark (results in decode_benchmark.txt)"
    decode_repeats=3
    [[ $QUICK == 1 ]] && decode_repeats=1
    decode_shapes=(
        "small --d-model 256 --heads 8 --d-ff 1024 --encoder-layers 4 --decoder-layers 4 --max-seq 256 --vocab 8000"
        "configured --d-model 768 --heads 12 --d-ff 3072 --encoder-layers 8 --decoder-layers 24 --max-seq 512 --vocab 16000")
    for shape in "${decode_shapes[@]}"; do
        read -r -a args <<<"$shape"
        name="decode_${args[0]}"
        # A lost device (rc 2, TD-574's engine resets) is retried once in a fresh process, as
        # <name>_retry.
        for run in "$name" "${name}_retry"; do
            step "$run" 3600 "$BIN/gpu_decode_bench" --lengths 50,100 --repeats "$decode_repeats" \
                "${args[@]:1}"
            [[ $? -eq 2 && $run == "$name" ]] || break
            log "  $name lost the device; trying once more"
            FAILURES=$((FAILURES - 1))
        done
    done
    {
        echo "TD-050/TD-033 GPU decode benchmark: greedy decoding without stop tokens, a random model."
        echo "gpu_cached = GPU KV cache (gpu_decode_step), gpu_full = GPU full recompute every step"
        echo "(before TD-050), cpu_cached = CPU KV cache, cpu_dispatch = CPU KV cache with each large"
        echo "multiply sent to the GPU on its own (chatbot_api_server before TD-033)."
        for shape in "${decode_shapes[@]}"; do
            for run in "decode_${shape%% *}" "decode_${shape%% *}_retry"; do
                [[ -f "$RESULTS/$run.log" ]] || continue
                echo
                echo "== $run"
                grep -h -E '^(model:|VALIDATE|RESULT|MISMATCH|FAILED|Last GPU work|the GPU)' \
                    "$RESULTS/$run.log" || echo "  (no output)"
            done
        done
    } >"$RESULTS/decode_benchmark.txt"
fi

# ── 5. TD-368 seed replay and TD-391 backend line ─────────────────────────────

backend_of() {
    grep -h -m1 'Backend:' "$RESULTS/$1.log" "$RESULTS/work/$1/trainer.log"* 2>/dev/null | head -1 |
        sed 's/.*Backend: //'
}

if [[ $RUN_TRAINING == 1 && -z "${FATAL_NO_DEVICE:-}" ]]; then
    log ""
    log "Training passes on the GPU (TD-368 seed replay, TD-391 backend)"
    weights_hash() {
        (cd "$RESULTS/work/$1/sessions" 2>/dev/null &&
            find . -type f -name '*.bin' -not -path '*best_model*' | sort | xargs -r cat | sha256sum | cut -c1-16)
    }
    epoch_losses() {
        grep -ho 'Epoch [0-9]* complete - Loss: [0-9.na]*.*GradNorm: [0-9.na]*' "$RESULTS/$1.log" 2>/dev/null
    }
    passes=(seed7_a seed7_b seed8)
    [[ $QUICK == 1 ]] && passes=(seed7_a seed7_b)
    for pass in "${passes[@]}"; do
        seed=7
        [[ $pass == seed8 ]] && seed=8
        step "$pass" 1800 "$HERE/train_pass.sh" "$RESULTS/work/$pass" "$seed"
    done
    {
        echo "Backend line (TD-391):"
        grep -h -m2 -A1 'Device 0:' "$RESULTS/seed7_a.log" "$RESULTS/work/seed7_a/trainer.log"* 2>/dev/null | head -4
        echo
        echo "Weight hashes (TD-368; seed7_a and seed7_b should match, seed8 should differ):"
        for pass in "${passes[@]}"; do
            echo "  $pass $(weights_hash "$pass")"
        done
        if [[ -n "$(weights_hash seed7_a)" && "$(weights_hash seed7_a)" == "$(weights_hash seed7_b)" ]]; then
            echo "  => SEED=7 replayed byte-for-byte"
        else
            echo "  => SEED=7 did not replay byte-for-byte; compare the losses:"
        fi
        for pass in "${passes[@]}"; do
            echo "  $pass:"
            epoch_losses "$pass" | sed 's/^/    /'
        done
    } >"$RESULTS/seed_replay.txt" 2>&1
    sed 's/^/  /' "$RESULTS/seed_replay.txt" | tee -a "$SUMMARY" >/dev/null
    cat "$RESULTS/seed_replay.txt"
fi

# ── 6. LeJEPA world-model pass on the GPU (TD-572) ────────────────────────────

# The LeJEPA pass (src/LeJEPATrainingPass.cpp) has run only on the CPU and with an injected device
# check since it moved out of IncrementalTrainingTool.cpp. Here it runs on the GPU through
# incremental_trainer --objective=lejepa: train, resume with nothing pending, re-queue and resume
# from the checkpoint, a SIGTERM drain (checkpoint, files released) and a resume after it. Then a
# soak of fresh passes on the default Level Zero v2 adapter, where a real engine reset should reach
# TD-566's check: the pass stops at once, saves nothing and gives its files back.

# The world_model pool's pending entries are unclaimed (every run_id empty).
lejepa_released() {
    local pending="$1/sessions/world_model/pending_files.txt"
    [[ -f "$pending" ]] && grep -q '"path"' "$pending" && ! grep -q '"run_id":"[^"]' "$pending"
}
lejepa_trained() {
    grep -q 'Trained  : 1 file' <("$HERE/lejepa_pass.sh" status "$1" 2>/dev/null)
}
# A soak pass's output and its own trainer log (the lifecycle's passes share one, so its checks
# read each step's output only).
lejepa_log() {
    cat "$RESULTS/$1.log" "$RESULTS/work/$1/trainer.log"* 2>/dev/null
}
# epoch seconds of the first log line matching PATTERN in the given text
log_time() {
    local stamp
    stamp=$(grep -m1 -E "$2" <<<"$1" | grep -o -m1 '^\[[0-9-]* [0-9:.]*\]' | tr -d '[]')
    [[ -n "$stamp" ]] && date -d "$stamp" +%s
}

if [[ $RUN_LEJEPA == 1 && -z "${FATAL_NO_DEVICE:-}" ]]; then
    log ""
    log "LeJEPA world-model pass on the GPU (TD-572; lejepa_report.txt)"
    LJ=lejepa_lifecycle
    LJDIR="$RESULTS/work/$LJ"
    declare -a LJ_CHECKS
    check() {
        local verdict=FAIL
        if eval "$2"; then verdict=ok; else FAILURES=$((FAILURES + 1)); fi
        LJ_CHECKS+=("$(printf '  %-4s %s' "$verdict" "$1")")
    }
    step lejepa_setup 300 "$HERE/lejepa_pass.sh" setup "$LJDIR" 7 3
    step lejepa_train 1800 "$HERE/lejepa_pass.sh" train "$LJDIR" 3
    check "train: a checkpoint is written and the texts are recorded as trained" \
        '[[ -f "$LJDIR/sessions/world_model_checkpoint/config.bin" ]] && lejepa_trained "$LJDIR"'
    check "train: the pass ran on the GPU" \
        'grep -q -E "Device 0:|Backend:" "$RESULTS/lejepa_train.log"'
    step_expect lejepa_resume_nothing_pending 600 3 "$HERE/lejepa_pass.sh" resume "$LJDIR"
    step lejepa_requeue 120 "$HERE/lejepa_pass.sh" requeue "$LJDIR"
    step lejepa_resume_trains 1800 "$HERE/lejepa_pass.sh" resume "$LJDIR"
    check "resume: trains the re-queued texts from the checkpoint" \
        'grep -q "Resumed world-model weights" "$RESULTS/lejepa_resume_trains.log" && lejepa_trained "$LJDIR"'

    # SIGTERM drain: the helper execs the trainer, so $! is the trainer's own PID; only it is
    # signalled.
    step lejepa_requeue_again 120 "$HERE/lejepa_pass.sh" requeue "$LJDIR"
    t0=$(date +%s)
    (cd "$LJDIR" && exec "$HERE/lejepa_pass.sh" resume "$LJDIR") >"$RESULTS/lejepa_sigterm.log" 2>&1 &
    TRAINER_PID=$!
    for ((s = 0; s < 1200; s++)); do
        grep -q '\[lejepa\] Epoch 1/' "$RESULTS/lejepa_sigterm.log" && break
        kill -0 "$TRAINER_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$TRAINER_PID" 2>/dev/null; then
        kill -TERM "$TRAINER_PID"
        for ((s = 0; s < 120; s++)); do
            kill -0 "$TRAINER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$TRAINER_PID" 2>/dev/null; then
            echo "run_gpu_tests.sh: no exit 120 s after SIGTERM; SIGKILL" >>"$RESULTS/lejepa_sigterm.log"
            kill -KILL "$TRAINER_PID"
        fi
    fi
    wait "$TRAINER_PID"
    rc=$?
    ROWS+=("$(printf '%-34s rc=%-4s %6ss (expected rc=1)' lejepa_sigterm "$rc" "$(($(date +%s) - t0))")")
    log "  ${ROWS[-1]}"
    [[ $rc -eq 1 ]] || FAILURES=$((FAILURES + 1))
    check "SIGTERM: the pass checkpoints and gives its texts back" \
        'grep -q "Checkpoint saved (paused)" "$RESULTS/lejepa_sigterm.log" && lejepa_released "$LJDIR"'
    step lejepa_resume_after_sigterm 1800 "$HERE/lejepa_pass.sh" resume "$LJDIR"
    check "resume after SIGTERM: trains from the paused checkpoint" \
        'grep -q "Resumed world-model weights" "$RESULTS/lejepa_resume_after_sigterm.log" && lejepa_trained "$LJDIR"'

    LJ_TSV="$RESULTS/lejepa_soak.tsv"
    printf 'pass\tbackend\trc\tseconds\toutcome\tkernel_events\tskipped_batches\ttd566\n' >"$LJ_TSV"
    for ((i = 1; i <= LEJEPA_SOAK_PASSES; i++)); do
        name=$(printf 'lejepa_soak_%02d' "$i")
        dir="$RESULTS/work/$name"
        "$HERE/lejepa_pass.sh" setup "$dir" "$((200 + i))" "$LEJEPA_SOAK_EPOCHS" >"$RESULTS/${name}_setup.log" 2>&1 || {
            log "  $name: setup failed (${name}_setup.log)"
            FAILURES=$((FAILURES + 1))
            continue
        }
        t0=$(date +%s)
        step "$name" 900 "$HERE/lejepa_pass.sh" train "$dir" "$LEJEPA_SOAK_EPOCHS"
        rc=$?
        t1=$(date +%s)
        events=$(kernel_log "$t0" "$((t1 + 2))" | grep -E "$KERNEL_EVENTS")
        [[ -n "$events" ]] && { echo "== $name"; echo "$events"; } >>"$RESULTS/lejepa_soak_kernel_events.txt"
        text=$(lejepa_log "$name")
        skipped=$(grep -c '\[lejepa\] Skipping batch' <<<"$text")
        outcome=ok
        td566=-
        if grep -q '\[lejepa\] The GPU stopped running work' <<<"$text"; then
            outcome=device_lost
            td566=ok
            grep -q 'Not saving the world model: the GPU was lost' <<<"$text" || td566="no-skip-save"
            [[ -f "$dir/sessions/world_model_checkpoint/config.bin" ]] && td566+=",checkpoint-written"
            lejepa_released "$dir" || td566+=",not-released"
            [[ $rc -eq 1 ]] || td566+=",rc=$rc"
            lost_at=$(log_time "$text" 'The GPU stopped running work')
            [[ -n "$lost_at" ]] && ((t1 - lost_at > 15)) && td566+=",slow-stop"
        elif grep -q -E 'DEVICE_LOST|device lost' <<<"$text"; then
            # A device loss the pass's check didn't see: outside a batch (GPU start-up, a save)
            # or a batch skipped on a device that still answered the probe.
            outcome=device_lost_elsewhere
            td566=missed
        elif [[ $rc -eq 124 || $rc -eq 137 ]]; then
            outcome=hung
        elif [[ $rc -ne 0 ]]; then
            outcome=failed
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$(backend_of "$name")" "$rc" "$((t1 - t0))" \
            "$outcome" "$(grep -c . <<<"$events")" "$skipped" "$td566" >>"$LJ_TSV"
        if [[ $outcome == device_lost* ]] && ! probe_device "probe_after_$name"; then
            log "  the GPU no longer runs work after $name: LeJEPA soak stopped. Reboot before using the GPU again."
            DEVICE_GONE=1
            break
        fi
    done

    {
        echo "LeJEPA world-model pass on the GPU (TD-572)"
        echo
        echo "Lifecycle (one session directory, work/$LJ):"
        printf '%s\n' "${LJ_CHECKS[@]}"
        echo
        echo "Soak: $LEJEPA_SOAK_PASSES fresh passes of $LEJEPA_SOAK_EPOCHS epochs, Level Zero v2 (the default)"
        echo
        column -t -s $'\t' "$LJ_TSV" 2>/dev/null || cat "$LJ_TSV"
        echo
        awk -F'\t' 'NR > 1 {n++; o += ($5 == "ok"); d += ($5 == "device_lost"); e += ($5 == "device_lost_elsewhere");
                    h += ($5 == "hung"); f += ($5 == "failed"); k += ($6 > 0); s += ($7 > 0); t += ($8 == "ok")}
                    END {printf "%d passes: %d ok, %d device lost (%d handled as TD-566 expects), %d device lost outside the check, %d hung, %d other failures; %d with kernel GPU events, %d with skipped batches\n",
                         n, o, d, t, e, h, f, k, s}' "$LJ_TSV"
        echo "td566: ok = the pass stopped within 15 s of the loss, saved nothing, released its texts and exited 1."
        echo "TD-572 is closed by a run where the lifecycle passes and, ideally, at least one device loss is handled."
        [[ -n "${DEVICE_GONE:-}" ]] && echo "The soak stopped early: the GPU no longer ran work after a device loss."
    } >"$RESULTS/lejepa_report.txt"
    sed -n '/^Lifecycle/,$p' "$RESULTS/lejepa_report.txt" | grep -v -E '^pass |^lejepa_soak_' | sed 's/^/  /' | tee -a "$SUMMARY"
fi

# ── 7. Level Zero soak: v2 (default) against v1 ───────────────────────────────

# Back-to-back training passes, alternating the default Level Zero v2 adapter with
# SYCL_UR_USE_LEVEL_ZERO_V2=0, which one Arc report found stopped the engine resets. Each pass
# records its kernel messages; after a device loss the device is probed, the soak stops if the GPU
# no longer runs work, and TD-566's handling is checked: the pass stops at once without copying
# weights back and no epoch is reported as loss 0.
if [[ $RUN_TRAINING == 1 && $SOAK_PASSES -gt 0 && -z "${FATAL_NO_DEVICE:-}" && -z "${DEVICE_GONE:-}" ]]; then
    log ""
    log "Level Zero soak: $SOAK_PASSES passes of $SOAK_EPOCHS epochs per variant, alternating (soak_report.txt)"
    SOAK_TSV="$RESULTS/soak.tsv"
    printf 'pass\tvariant\tbackend\trc\tseconds\toutcome\tkernel_events\ttd566\n' >"$SOAK_TSV"
    for ((i = 1; i <= SOAK_PASSES; i++)); do
        if ((i % 2)); then order=(l0v2 l0v1); else order=(l0v1 l0v2); fi
        for variant in "${order[@]}"; do
            name=$(printf 'soak_%s_%02d' "$variant" "$i")
            envs=()
            [[ $variant == l0v1 ]] && envs=(SYCL_UR_USE_LEVEL_ZERO_V2=0)
            t0=$(date +%s)
            step "$name" 900 env "${envs[@]}" "$HERE/train_pass.sh" "$RESULTS/work/$name" "$((100 + i))" "$SOAK_EPOCHS"
            rc=$?
            t1=$(date +%s)
            events=$(kernel_log "$t0" "$((t1 + 2))" | grep -E "$KERNEL_EVENTS")
            [[ -n "$events" ]] && { echo "== $name"; echo "$events"; } >>"$RESULTS/soak_kernel_events.txt"
            outcome=ok
            td566=-
            if grep -q -E 'DEVICE_LOST|stopped running work' "$RESULTS/$name.log" "$RESULTS/work/$name/trainer.log"* 2>/dev/null; then
                outcome=device_lost
                td566=ok
                grep -q 'Not copying the device weights back' "$RESULTS/$name.log" "$RESULTS/work/$name/trainer.log"* 2>/dev/null ||
                    td566="no-skip-copy"
                grep -q -E 'complete - Loss: 0(\.0+)?([^0-9.]|$)' "$RESULTS/$name.log" 2>/dev/null && td566="loss0-epoch"
                ((t1 - t0 > 15)) && td566+=",slow-stop"
            elif [[ $rc -ne 0 ]]; then
                outcome=failed
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$variant" "$(backend_of "$name")" "$rc" "$((t1 - t0))" \
                "$outcome" "$(grep -c . <<<"$events")" "$td566" >>"$SOAK_TSV"
            if [[ $outcome == device_lost ]] && ! probe_device "probe_after_$name"; then
                log "  the GPU no longer runs work after $name: soak stopped. Reboot before using the GPU again."
                DEVICE_GONE=1
                break 2
            fi
        done
    done
    {
        echo "Level Zero soak ($SOAK_PASSES passes of $SOAK_EPOCHS epochs per variant, alternating)"
        echo
        column -t -s $'\t' "$SOAK_TSV" 2>/dev/null || cat "$SOAK_TSV"
        echo
        echo "Per variant (passes, ok, device lost, other failures, passes with kernel GPU events):"
        awk -F'\t' 'NR > 1 {n[$2]++; o[$2] += ($6 == "ok"); d[$2] += ($6 == "device_lost");
                    f[$2] += ($6 == "failed"); k[$2] += ($7 > 0)}
                    END {for (v in n) printf "  %s: %d passes, %d ok, %d device lost, %d other failures, %d with kernel events\n",
                         v, n[v], o[v], d[v], f[v], k[v]}' "$SOAK_TSV"
        echo "Backends seen: l0v1 should not say Level-Zero V2, l0v2 should."
        awk -F'\t' 'NR > 1 {print "  " $2 ": " $3}' "$SOAK_TSV" | sort -u
        echo "td566: ok = the pass stopped at once without copying weights back and no loss-0 epoch."
        [[ -n "${DEVICE_GONE:-}" ]] && echo "The soak stopped early: the GPU no longer ran work after a device loss."
    } >"$RESULTS/soak_report.txt"
    sed -n '/^Per variant/,$p' "$RESULTS/soak_report.txt" | sed 's/^/  /' | tee -a "$SUMMARY"
fi

# ── 8. Engine-reset bisection (TD-574) ────────────────────────────────────────

# Training passes like the soak's, on the default Level Zero v2 adapter, in variants that each
# change one thing, run round-robin (the starting variant rotates each round) so a burst of resets
# lands on every variant alike:
#   control         as the soak (twice per round)
#   sync_step       ADAI_GPU_SYNC=step: wait for the GPU at each training phase
#   sync_kernel     ADAI_GPU_SYNC=kernel: wait before every kernel; a failure names its kernel
#                   (twice per round: November 13 it was the only variant with no resets)
#   sync_every_N    ADAI_GPU_SYNC=kernel:N for N = 8, 64, 512: at most N operations queued
#   full_priority   GPU_STRATEGY=full: the normal-priority queue instead of the low-priority one
#   no_dropout      DROPOUT=0: no dropout kernel
#   host_embedding  the embedding gather and scatter-add (float atomics) on the host
#   host_optimizer  the optimizer update and gradient norms (float atomics) on the host
#   host_all        embedding, optimizer and dropout on the host
#   encode_soak     forward only: LeJEPAEncoder::gpu_encode() for 10 s, no backward or optimizer
#   long_preempt    --engine-tuning: the compute engine's preempt timeout raised to 5 s
#   long_timeslice  --engine-tuning: the compute engine's time slice raised to 100 ms
# Every pass keeps its last GPU work in gpu_trace.txt (ADAI_GPU_TRACE_FILE), so even a hung pass
# that is killed leaves what it was running.

# The compute engine settings --engine-tuning changes. The xe driver copies them into each exec
# queue when the queue is made, so a value set before a pass starts applies to that whole pass.
declare -A ENGINE_ORIG=()
engine_files() {
    ls /sys/class/drm/card*/device/tile*/gt*/engines/ccs/"$1" 2>/dev/null
}
# engine_set FILE VALUE: writes VALUE (capped at the driver's <setting>_max) to every compute
# engine's FILE; false if a value didn't read back.
engine_set() {
    local f max v ok=0
    for f in $(engine_files "$1"); do
        max=$(cat "${f%_us}_max" 2>/dev/null)
        v=$2
        [[ -n "$max" && "$max" =~ ^[0-9]+$ && $v -gt $max ]] && v=$max
        $SUDO sh -c "echo $v > '$f'" 2>/dev/null
        [[ "$(cat "$f" 2>/dev/null)" == "$v" ]] || ok=1
    done
    return $ok
}
engine_restore() {
    local f
    for f in "${!ENGINE_ORIG[@]}"; do
        $SUDO sh -c "echo ${ENGINE_ORIG[$f]} > '$f'" 2>/dev/null
    done
}

if [[ $BISECT_PASSES -gt 0 && -z "${FATAL_NO_DEVICE:-}" && -z "${DEVICE_GONE:-}" ]]; then
    VARIANTS=(control sync_kernel sync_step sync_every_8 sync_every_64 sync_every_512 full_priority
        control sync_kernel no_dropout host_embedding host_optimizer host_all encode_soak)
    if [[ $ENGINE_TUNING == 1 ]]; then
        if [[ -z "$SUDO" ]]; then
            log "--engine-tuning needs sudo: long_preempt and long_timeslice left out."
        elif [[ -z "$(engine_files preempt_timeout_us)" || -z "$(engine_files timeslice_duration_us)" ]]; then
            log "--engine-tuning: no compute engine preempt_timeout_us/timeslice_duration_us in sysfs: left out."
        else
            for f in $(engine_files preempt_timeout_us) $(engine_files timeslice_duration_us); do
                ENGINE_ORIG[$f]=$(cat "$f")
            done
            trap 'engine_restore; kill "${KEEPALIVE:-}" 2>/dev/null' EXIT
            {
                echo "Compute engine settings before the bisection (restored after each tuned pass):"
                for f in "${!ENGINE_ORIG[@]}"; do echo "  $f: ${ENGINE_ORIG[$f]}"; done
                for f in $(engine_files preempt_timeout_max) $(engine_files timeslice_duration_max); do
                    echo "  $f: $(cat "$f")"
                done
            } >"$RESULTS/engine_tuning.txt"
            VARIANTS+=(long_preempt long_timeslice)
        fi
    fi
    log ""
    log "Engine-reset bisection: $BISECT_PASSES rounds of ${#VARIANTS[@]} passes ($SOAK_EPOCHS epochs each), round-robin (bisect_report.txt)"
    BISECT_TSV="$RESULTS/bisect.tsv"
    printf 'pass\tvariant\trc\tseconds\toutcome\tkernel_events\tcoredump\tlast_gpu_work\n' >"$BISECT_TSV"
    declare -A VARIANT_RUNS=()
    for ((i = 1; i <= BISECT_PASSES; i++)); do
        for ((k = 0; k < ${#VARIANTS[@]}; k++)); do
            variant=${VARIANTS[$(((k + i - 1) % ${#VARIANTS[@]}))]}
            VARIANT_RUNS[$variant]=$((${VARIANT_RUNS[$variant]:-0} + 1))
            run=${VARIANT_RUNS[$variant]}
            name=$(printf 'bisect_%s_%02d' "$variant" "$run")
            dir="$RESULTS/work/$name"
            envs=(ADAI_GPU_TRACE_FILE="$dir/gpu_trace.txt")
            tuned=""
            case $variant in
                sync_step) envs+=(ADAI_GPU_SYNC=step) ;;
                sync_kernel) envs+=(ADAI_GPU_SYNC=kernel) ;;
                sync_every_*) envs+=(ADAI_GPU_SYNC="kernel:${variant#sync_every_}") ;;
                full_priority) envs+=(GPU_STRATEGY=full) ;;
                no_dropout) envs+=(DROPOUT=0) ;;
                host_embedding) envs+=(ADAI_GPU_HOST_KERNELS=embedding) ;;
                host_optimizer) envs+=(ADAI_GPU_HOST_KERNELS=optimizer) ;;
                host_all) envs+=(ADAI_GPU_HOST_KERNELS=embedding,optimizer,dropout) ;;
                long_preempt) tuned=preempt_timeout_us:5000000 ;;
                long_timeslice) tuned=timeslice_duration_us:100000 ;;
            esac
            if [[ -n "$tuned" ]] && ! engine_set "${tuned%%:*}" "${tuned#*:}"; then
                log "  $name: ${tuned%%:*} could not be set to ${tuned#*:}; pass skipped"
                engine_restore
                continue
            fi
            t0=$(date +%s)
            if [[ $variant == encode_soak ]]; then
                step "$name" 600 env "${envs[@]}" "$BIN/gpu_encode_soak" "$HERE/data/pairs.jsonl" 10
            else
                step "$name" 600 env "${envs[@]}" "$HERE/train_pass.sh" "$dir" "$((300 + run))" "$SOAK_EPOCHS"
            fi
            rc=$?
            t1=$(date +%s)
            [[ -n "$tuned" ]] && engine_restore
            events=$(kernel_log "$t0" "$((t1 + 2))" | grep -E "$KERNEL_EVENTS")
            [[ -n "$events" ]] && { echo "== $name"; echo "$events"; } >>"$RESULTS/bisect_kernel_events.txt"
            if [[ $rc -eq 0 ]]; then
                outcome=ok
            elif [[ $rc -eq 124 || $rc -eq 137 ]]; then
                outcome=hung
            elif grep -q -E 'DEVICE_LOST|stopped running work' "$RESULTS/$name.log" "$dir/trainer.log"* 2>/dev/null; then
                outcome=device_lost
            else
                outcome=failed
            fi
            last=$(grep -h -m1 -o 'Last GPU work: kernel [^;]*' "$RESULTS/$name.log" "$dir/trainer.log"* 2>/dev/null |
                head -1 | sed 's/Last GPU work: //')
            if [[ -z "$last" && $outcome != ok && -s "$dir/gpu_trace.txt" ]]; then
                last="trace: $(head -1 "$dir/gpu_trace.txt" | sed 's/ *$//')"
            fi
            dump=no
            grep -q "^$name " "$RESULTS/coredumps/index.txt" 2>/dev/null && dump=yes
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$variant" "$rc" "$((t1 - t0))" "$outcome" \
                "$(grep -c . <<<"$events")" "$dump" "${last:--}" >>"$BISECT_TSV"
            if [[ $outcome == device_lost || $outcome == hung ]] && ! probe_device "probe_after_$name"; then
                log "  the GPU no longer runs work after $name: bisection stopped. Reboot before using the GPU again."
                DEVICE_GONE=1
                break 2
            fi
        done
    done
    [[ ${#ENGINE_ORIG[@]} -gt 0 ]] && engine_restore
    {
        echo "Engine-reset bisection ($BISECT_PASSES rounds, round-robin, Level Zero v2)"
        echo
        column -t -s $'\t' "$BISECT_TSV" 2>/dev/null || cat "$BISECT_TSV"
        echo
        echo "Per variant (passes, ok, device lost, hung, other failures, passes with kernel GPU events,"
        echo "seconds on the GPU, device losses and hangs per minute of it):"
        for v in $(printf '%s\n' "${VARIANTS[@]}" | awk '!seen[$0]++'); do
            awk -F'\t' -v v="$v" 'NR > 1 && $2 == v {n++; o += ($5 == "ok"); d += ($5 == "device_lost");
                h += ($5 == "hung"); f += ($5 == "failed"); k += ($6 > 0); t += $4}
                END {printf "  %-15s %3d passes, %3d ok, %3d device lost, %3d hung, %3d other failures, %3d with kernel events, %5ds, %.2f/min\n",
                     v, n, o, d, h, f, k, t, (t > 0 ? 60 * (d + h) / t : 0)}' "$BISECT_TSV"
        done
        echo
        echo "Last GPU work of each failed pass (exact with sync_kernel, within the last N with sync_every_N):"
        awk -F'\t' 'NR > 1 && $5 != "ok" {print "  " $1 ": " $8}' "$BISECT_TSV"
        echo
        echo "Crash dumps: coredumps/ (index.txt lists which step each came after)."
        [[ -s "$RESULTS/engine_tuning.txt" ]] && echo "Engine settings used by --engine-tuning: engine_tuning.txt."
        [[ -n "${DEVICE_GONE:-}" ]] && echo "The bisection stopped early: the GPU no longer ran work."
    } >"$RESULTS/bisect_report.txt"
    sed -n '/^Per variant/,$p' "$RESULTS/bisect_report.txt" | sed 's/^/  /' | tee -a "$SUMMARY"
fi

# ── 9. Pack the results ───────────────────────────────────────────────────────

log ""
log "Steps that failed: $FAILURES"
kernel_log "$RUN_START" >"$RESULTS/kernel_log.txt"
capture_coredumps end_of_run
[[ -s "$RESULTS/coredumps/index.txt" ]] && log "Driver crash dumps: $(grep -c . "$RESULTS/coredumps/index.txt") (coredumps/)"
EVENTS=$(grep -c -E "$KERNEL_EVENTS" "$RESULTS/kernel_log.txt")
log "Kernel GPU events during the run (engine resets, job timeouts, faults): ${EVENTS:-0} (kernel_log.txt)"
[[ -s "$RESULTS/kernel_log.txt" ]] || log "  (kernel log empty: needs sudo or the systemd-journal group)"
# The trainer's checkpoints are large and not needed: keep logs and configs only.
find "$RESULTS/work" -path '*/sessions/*' -type f \( -name '*.bin*' -o -name '*.optimizer' -o -name '*.lm_head' \) -delete 2>/dev/null
ARCHIVE="$HERE/results/$RESULTS_NAME.tar.gz"
tar -czf "$ARCHIVE" -C "$HERE/results" "$RESULTS_NAME"
log "Results: $ARCHIVE"
echo
echo "Copy $ARCHIVE back to the dev machine."
[[ -s "$HERE/host_state/stopped.txt" ]] && echo "Then ./restore_host.sh starts the services prepare_host.sh stopped."
[[ $FAILURES -eq 0 && -z "${FATAL_NO_DEVICE:-}" ]]
