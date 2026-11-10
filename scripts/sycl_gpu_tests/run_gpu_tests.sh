#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-09

# Runs the packaged SYCL GPU tests on the GPU host (ai-machine) and packs the results into one
# archive to copy back. Built into a package by scripts/package_sycl_gpu_tests.sh; see README.md
# beside this script. Changes nothing outside this directory: no services are stopped, no system
# settings are touched, and every trainer run uses a private config with no MNS, registry or metrics.
#
#   ./run_gpu_tests.sh [--quick] [--skip-tests] [--skip-benchmark] [--skip-training]
#                      [--allow-cpu-device]
#
# --quick             only the GPU-specific test binaries, a 3-repeat benchmark, one seed pair
# --allow-cpu-device  also accept oneAPI's CPU device (ADAI_SYCL_ALLOW_CPU_DEVICE=1); for trying
#                     the package on a machine without a GPU, never on ai-machine

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/bin"
QUICK=0
RUN_TESTS=1
RUN_BENCHMARK=1
RUN_TRAINING=1
ALLOW_CPU=0

usage() {
    sed -n '/^#   \.\/run_gpu_tests/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --skip-tests) RUN_TESTS=0 ;;
        --skip-benchmark) RUN_BENCHMARK=0 ;;
        --skip-training) RUN_TRAINING=0 ;;
        --allow-cpu-device) ALLOW_CPU=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s 2>/dev/null || hostname)"
RESULTS_NAME="adai-gpu-results-${HOST}-${STAMP}"
RESULTS="$HERE/results/$RESULTS_NAME"
mkdir -p "$RESULTS/xml" "$RESULTS/work"
SUMMARY="$RESULTS/summary.txt"

log() {
    echo "$*" | tee -a "$SUMMARY"
}

# Only this script's own settings reach the binaries: drop any ADAI config keys inherited from
# the shell (a deployed host may export MNS, registry or GPU settings).
while IFS= read -r name; do
    unset "$name"
done < <(env | cut -d= -f1 | grep -E '^(NAME_SERVICE|REGISTRY|METRICS|MODEL_|SEED$|GPU_|DATASET|RUN_|SESSION_DIR|VOCAB_PATH|LOG_|TRAINER_|WORLD_MODEL|HIPPOCAMPAL|NUM_|D_MODEL|D_FF|MAX_SEQ|BATCH|LEARNING|DROPOUT|EARLY_STOPPING|AUTO_SAVE|FTP|ADAI_)' || true)
if [[ "$ALLOW_CPU" == 1 ]]; then
    export ADAI_SYCL_ALLOW_CPU_DEVICE=1
fi
export GTEST_COLOR=no

# ── 1. Environment ────────────────────────────────────────────────────────────

{
    echo "== package"
    cat "$HERE/MANIFEST.txt" 2>/dev/null
    echo
    echo "== host"
    date -u
    uname -a
    grep -m1 'model name' /proc/cpuinfo
    echo "avx2/fma: $(grep -m1 -o -w -e avx2 /proc/cpuinfo) $(grep -m1 -o -w -e fma /proc/cpuinfo)"
    echo "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-<unset>}"
    echo "ADAI_SYCL_ALLOW_CPU_DEVICE=${ADAI_SYCL_ALLOW_CPU_DEVICE:-<unset>}"
    echo
    echo "== GPU"
    lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' || echo "(lspci unavailable)"
    cat /sys/module/xe/version 2>/dev/null || true
    ls -l /dev/dri 2>/dev/null || true
    id
    echo
    echo "== sycl-ls"
    command -v sycl-ls >/dev/null && timeout 60 sycl-ls 2>&1 || echo "(sycl-ls not on PATH)"
    echo
    echo "== oneAPI and GPU packages"
    dpkg -l 2>/dev/null | grep -iE 'intel-oneapi-runtime|level-zero|intel-opencl|libigc|intel-gpu|xe-' || true
    echo
    echo "== other ADAI processes (they share the GPU and can skew timings)"
    pgrep -a -x incremental_tra 2>/dev/null
    pgrep -a -x trainer_service 2>/dev/null
    pgrep -a -x chatbot_api_ser 2>/dev/null
    true
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
if [[ -n "$(missing_libs)" && -f /opt/intel/oneapi/setvars.sh ]]; then
    # setvars.sh reads unset variables, which `set -u` would turn into an exit.
    set +u
    # shellcheck source=/dev/null
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi
if [[ -n "$(missing_libs)" ]]; then
    log "FATAL: oneAPI runtime libraries not found:"
    missing_libs | tee -a "$SUMMARY"
    log "Install or update the intel-oneapi-runtime-libs packages to at least the build's oneAPI version (MANIFEST.txt)."
    exit 1
fi
ldd "$BIN/incremental_trainer" >"$RESULTS/ldd.txt" 2>&1

if [[ -n "$(pgrep -x trainer_service; pgrep -x incremental_tra)" ]]; then
    log "WARNING: a trainer is running on this host. Its GPU use skews the benchmark and the tests share GPU memory with it."
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

FAILURES=0
declare -a ROWS

# step NAME TIMEOUT_SECONDS COMMAND...: runs COMMAND in work/NAME, output in NAME.log.
step() {
    local name="$1" limit="$2"
    shift 2
    local dir="$RESULTS/work/$name"
    mkdir -p "$dir"
    local start end rc
    start=$(date +%s)
    (cd "$dir" && timeout -k 30 "$limit" "$@") >"$RESULTS/$name.log" 2>&1
    rc=$?
    end=$(date +%s)
    local note=""
    [[ $rc -eq 124 || $rc -eq 137 ]] && note=" (timed out after ${limit}s)"
    local skipped
    skipped=$(grep -c '^\[  SKIPPED \]' "$RESULTS/$name.log" 2>/dev/null)
    [[ "${skipped:-0}" -gt 0 ]] && note+=" (${skipped} SKIPPED lines)"
    ROWS+=("$(printf '%-34s rc=%-4s %6ss%s' "$name" "$rc" "$((end - start))" "$note")")
    if [[ $rc -ne 0 ]]; then
        FAILURES=$((FAILURES + 1))
    fi
    log "  ${ROWS[-1]}"
    return $rc
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
        TESTS=(gpuutilsTests matrixgpuTd003Tests gpuOptimizerTests deviceSelectionTests)
    else
        TESTS=(deviceSelectionTests gpuutilsTests matrixgpuTd003Tests gpuOptimizerTests matrixTests
            feedforwardTests multiheadattentionTests encoderdecoderTests initRngTests
            generationQualityAsyncTests chatbottrainerTests incrementaltrainerTests chatbotapiTests)
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

# ── 4. TD-171 benchmark ───────────────────────────────────────────────────────

if [[ $RUN_BENCHMARK == 1 && -z "${FATAL_NO_DEVICE:-}" ]]; then
    log ""
    log "TD-171 stacked multiply benchmark (results in stacked_gemm_benchmark.log)"
    repeats=15
    [[ $QUICK == 1 ]] && repeats=3
    step stacked_gemm_benchmark 5400 "$BIN/stacked_gemm_benchmark" 768 3072 12 "$repeats"
fi

# ── 5. TD-368 seed replay and TD-391 backend line ─────────────────────────────

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

# ── 6. Pack the results ───────────────────────────────────────────────────────

log ""
log "Steps that failed: $FAILURES"
dmesg 2>/dev/null | tail -200 >"$RESULTS/dmesg_tail.txt" || true
# The trainer's checkpoints are large and not needed: keep logs and configs only.
find "$RESULTS/work" -path '*/sessions/*' -type f \( -name '*.bin*' -o -name '*.optimizer' -o -name '*.lm_head' \) -delete 2>/dev/null
ARCHIVE="$HERE/results/$RESULTS_NAME.tar.gz"
tar -czf "$ARCHIVE" -C "$HERE/results" "$RESULTS_NAME"
log "Results: $ARCHIVE"
echo
echo "Copy $ARCHIVE back to the dev machine."
[[ $FAILURES -eq 0 && -z "${FATAL_NO_DEVICE:-}" ]]
