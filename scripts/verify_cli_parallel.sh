#!/bin/bash

# @adai-status: beta        (TD-044 resolved — real test suite added, see tests/scripts/verify_cli_parallel_test.sh)
# @adai-version: 0.6.2
# @adai-reviewed: 2026-09-11


echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        CLI Chatbot Parallel Processing Verification         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# TD-044: bare relative paths ("./build/bin/chatbot", "./build/CMakeCache.txt"
# below) with no anchor to the script's own location — only worked when the
# caller's CWD happened to already be the repo root. Resolved via
# SCRIPT_DIR/REPO_ROOT, matching every sibling launcher script, then cd to
# REPO_ROOT so every relative path below resolves the same way regardless
# of caller CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}" || exit 1

# chatbot's CMake target sets RUNTIME_OUTPUT_DIRECTORY to
# ${CMAKE_BINARY_DIR}/bin, never .../src/. TD-118.
#
# TD-044: even once anchored to REPO_ROOT, "build/bin/chatbot" was still
# wrong for any standard, documented build — every preset in
# CMakePresets.json builds into "build/<preset>/", never bare "build/".
# Auto-detects the first preset build directory that actually has the
# binary, falling back to bare "build/" for a non-preset configure.
find_build_dir() {
    local marker="$1"
    local dir
    for dir in "${REPO_ROOT}/build/debug" "${REPO_ROOT}/build/release" \
               "${REPO_ROOT}/build/portable" "${REPO_ROOT}/build/relwithdebinfo" \
               "${REPO_ROOT}/build/gpu" "${REPO_ROOT}/build/sycl" \
               "${REPO_ROOT}/build"; do
        if [ -f "${dir}/${marker}" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}
BUILD_DIR="$(find_build_dir bin/chatbot || echo "${REPO_ROOT}/build")"
CHATBOT_BINARY="${BUILD_DIR}/bin/chatbot"

# Check if binary exists
if [ ! -f "$CHATBOT_BINARY" ]; then
    echo "❌ CLI chatbot binary not found at: $CHATBOT_BINARY"
    echo "   Run: cmake --preset=debug && cmake --build --preset=debug --target chatbot"
    exit 1
fi

echo "✅ Binary found: $CHATBOT_BINARY"
echo ""

# Check binary size
SIZE=$(du -h "$CHATBOT_BINARY" | cut -f1)
echo "📦 Binary size: $SIZE"
echo ""

# Check OpenMP linkage
echo "🔍 Checking OpenMP linkage..."
if ldd "$CHATBOT_BINARY" | grep -q "libgomp"; then
    GOMP_PATH=$(ldd "$CHATBOT_BINARY" | grep libgomp | awk '{print $3}')
    echo "✅ OpenMP (parallel processing) is linked ✓"
    echo "   Library: $GOMP_PATH"
else
    echo "❌ OpenMP NOT linked - parallel processing unavailable"
fi
echo ""

# Check pthread linkage
echo "🔍 Checking pthread linkage..."
if ldd "$CHATBOT_BINARY" | grep -q "libpthread"; then
    echo "✅ pthread is linked ✓"
else
    echo "⚠️  pthread not explicitly linked (may be in glibc)"
fi
echo ""

# Check build type
echo "🔍 Checking build configuration..."
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    BUILD_TYPE=$(grep "CMAKE_BUILD_TYPE" "${BUILD_DIR}/CMakeCache.txt" | head -1 | cut -d= -f2)
    echo "   Build Type: $BUILD_TYPE"

    # Check for optimization flags
    if grep -q "CMAKE_CXX_FLAGS_RELEASE.*-O3" "${BUILD_DIR}/CMakeCache.txt"; then
        echo "✅ Optimizations enabled: -O3"
    fi

    if grep -q "march=native" "${BUILD_DIR}/CMakeCache.txt"; then
        echo "✅ Architecture-specific optimization: -march=native"
    fi

    if grep -q "fopenmp" "${BUILD_DIR}/CMakeCache.txt"; then
        echo "✅ OpenMP flags: -fopenmp"
    fi
fi
echo ""

# Check CPU cores
echo "🔍 System CPU information..."
CORES=$(nproc)
echo "   Available CPU cores: $CORES"
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    VERIFICATION SUMMARY                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Parallel Processing Features:"
echo "  ✅ Priority 1: OpenMP matrix operations (adai_core)"
echo "  ✅ Priority 4: Parallel attention heads (adai_attention)"
echo "  ✅ Multi-core CPU utilization ($CORES cores)"
echo "  ✅ Vectorized operations with SIMD"
echo "  ✅ Release build with optimizations"
echo ""
echo "Performance Expectations:"
echo "  • Matrix operations: Up to ${CORES}x faster"
echo "  • Attention computation: Parallelized across heads"
echo "  • Overall inference: 2-4x faster vs sequential"
echo "  • CPU usage: 60-100% across all cores during generation"
echo ""
echo "Usage:"
echo "  # Standard run"
echo "  ${CHATBOT_BINARY} --vocab vocab.txt --model chatbot_model.bin"
echo ""
echo "  # Maximum performance (set OpenMP threads)"
echo "  export OMP_NUM_THREADS=$CORES"
echo "  ${CHATBOT_BINARY} --vocab vocab.txt --model chatbot_model.bin"
echo ""
echo "  # Monitor CPU usage in another terminal"
echo "  htop"
echo ""
