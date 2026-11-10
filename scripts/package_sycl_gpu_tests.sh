#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-09

# Builds the SYCL GPU test binaries, the TD-171 benchmark and the trainer, and packs them with
# scripts/sycl_gpu_tests/ into one tarball to run on the GPU host (ai-machine): see that
# directory's README.md. Uses the host's oneAPI runtime rather than bundling one.
#
#   scripts/package_sycl_gpu_tests.sh [--build-dir DIR] [--out DIR] [--no-build]
#
# --build-dir  a SYCL build with tests (default build/sycl-cpudev; created when missing:
#              ENABLE_SYCL=ON, BUILD_TESTING=ON, Release)
# --out        where the tarball goes (default build/)
# --no-build   package what is already built

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/sycl-cpudev"
OUT_DIR="$PROJECT_ROOT/build"
DO_BUILD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --build-dir needs a value" >&2; exit 2; }
            BUILD_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || { echo "ERROR: --out needs a value" >&2; exit 2; }
            OUT_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        --no-build)
            DO_BUILD=0
            shift
            ;;
        -h | --help)
            sed -n '/^#   scripts\/package_sycl_gpu_tests/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

TEST_TARGETS=(deviceSelectionTests gpuutilsTests matrixgpuTd003Tests gpuOptimizerTests matrixTests
    feedforwardTests multiheadattentionTests encoderdecoderTests initRngTests
    generationQualityAsyncTests chatbottrainerTests incrementaltrainerTests chatbotapiTests)
TOOL_TARGETS=(incremental_trainer dataset_manager stacked_gemm_benchmark)

# ── Preflight ──────────────────────────────────────────────────────────────────

if [[ $DO_BUILD == 0 && ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "ERROR: no build in $BUILD_DIR (drop --no-build to build one)"
    exit 1
fi
if [[ -z "${ONEAPI_ROOT:-}" ]]; then
    if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
        set +u
        # shellcheck source=/dev/null
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
        set -u
    else
        echo "ERROR: oneAPI not found. Install it, or run: source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
fi

# ── Build ──────────────────────────────────────────────────────────────────────

if [[ $DO_BUILD == 1 ]]; then
    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DENABLE_SYCL=ON -DBUILD_TESTING=ON \
            -DCMAKE_BUILD_TYPE=Release
    fi
    if ! grep -q '^ENABLE_SYCL:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" ||
        ! grep -q '^BUILD_TESTING:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"; then
        echo "ERROR: $BUILD_DIR isn't a SYCL build with tests (needs ENABLE_SYCL=ON and BUILD_TESTING=ON)"
        exit 1
    fi
    cmake --build "$BUILD_DIR" -j"$(nproc)" --target "${TEST_TARGETS[@]}" "${TOOL_TARGETS[@]}"
fi

find_binary() {
    local found
    found=$(find "$BUILD_DIR/bin" "$BUILD_DIR/src" "$BUILD_DIR/tests" -maxdepth 1 -type f -name "$1" \
        -perm -u+x 2>/dev/null | head -1)
    [[ -n "$found" ]] || { echo "ERROR: $1 not built in $BUILD_DIR" >&2; return 1; }
    echo "$found"
}

# ── Stage ──────────────────────────────────────────────────────────────────────

COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)
DIRTY=""
git -C "$PROJECT_ROOT" diff --quiet HEAD -- src tests benchmarks || DIRTY="-dirty"
PACKAGE_NAME="adai-sycl-gpu-tests-${COMMIT}${DIRTY}-$(date +%Y%m%d)"
STAGE_PARENT=$(mktemp -d)
trap 'rm -rf -- "$STAGE_PARENT"' EXIT
STAGE="$STAGE_PARENT/$PACKAGE_NAME"
mkdir -p "$STAGE/bin" "$STAGE/data"

for target in "${TEST_TARGETS[@]}" "${TOOL_TARGETS[@]}"; do
    cp "$(find_binary "$target")" "$STAGE/bin/"
done
cp "$SCRIPT_DIR/sycl_gpu_tests/run_gpu_tests.sh" "$SCRIPT_DIR/sycl_gpu_tests/train_pass.sh" \
    "$SCRIPT_DIR/sycl_gpu_tests/README.md" "$STAGE/"
cp "$SCRIPT_DIR/sycl_gpu_tests/pairs.jsonl" "$STAGE/data/"
chmod +x "$STAGE/run_gpu_tests.sh" "$STAGE/train_pass.sh"

{
    echo "$PACKAGE_NAME"
    echo "commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)${DIRTY}"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname -s 2>/dev/null || hostname)"
    echo "compiler: $(icpx --version 2>/dev/null | head -1)"
    echo "build dir: $BUILD_DIR ($(grep '^CMAKE_BUILD_TYPE' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2))"
    echo "oneAPI libraries the trainer links:"
    ldd "$STAGE/bin/incremental_trainer" | grep -E 'sycl|mkl|iomp|ur_|level_zero|OpenCL' | awk '{print "  " $1}'
    echo "files:"
    (cd "$STAGE" && find . -type f | sort | xargs sha256sum | sed 's/^/  /')
} >"$STAGE/MANIFEST.txt"

mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/$PACKAGE_NAME.tar.gz"
tar -czf "$TARBALL" -C "$STAGE_PARENT" "$PACKAGE_NAME"
(cd "$OUT_DIR" && sha256sum "$PACKAGE_NAME.tar.gz" >"$PACKAGE_NAME.tar.gz.sha256")

echo "Package: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo "SHA-256: $(cut -d' ' -f1 "$TARBALL.sha256")"
[[ -n "$DIRTY" ]] && echo "WARNING: built from uncommitted changes to src/, tests/ or benchmarks/"
exit 0
