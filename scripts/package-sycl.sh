#!/usr/bin/env bash

# @adai-status: beta
# @adai-version: 0.7.0
# @adai-reviewed: 2026-09-07

set -euo pipefail

# Package incremental_trainer and chatbot_api_server (SYCL build) into a
# self-contained tarball with all Intel oneAPI runtime libraries, so the
# target machine only needs the Intel GPU compute runtime installed.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/sycl"
BIN_DIR="$BUILD_DIR/bin"

VERSION=$(grep 'project(adai VERSION' "$PROJECT_ROOT/CMakeLists.txt" \
    | sed 's/.*VERSION \([0-9.]*\).*/\1/')
TIMESTAMP=$(date +%Y%m%d)
PACKAGE_NAME="adai-sycl-${VERSION}-${TIMESTAMP}"
STAGE_DIR="$BUILD_DIR/$PACKAGE_NAME"

# ── Preflight checks ────────────────────────────────────────────────────

if [[ ! -x "$BIN_DIR/incremental_trainer" ]] || [[ ! -x "$BIN_DIR/chatbot_api_server" ]]; then
    echo "ERROR: Build the sycl preset first:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    echo "  cmake --preset sycl"
    echo "  cmake --build build/sycl -j\$(nproc) --target incremental_trainer chatbot_api_server"
    exit 1
fi

if [[ -z "${ONEAPI_ROOT:-}" ]]; then
    if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
        # shellcheck source=/dev/null
        source /opt/intel/oneapi/setvars.sh --force 2>/dev/null
    else
        echo "ERROR: oneAPI environment not set. Run: source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
fi

echo "=== Packaging $PACKAGE_NAME ==="

# ── Stage directory layout ───────────────────────────────────────────────
#
#   adai-sycl-X.Y.Z-YYYYMMDD/
#   ├── bin/                    executables
#   ├── lib/                    Intel oneAPI runtime .so files
#   ├── scripts/                install scripts and systemd unit template
#   ├── config.conf             default configuration file
#   ├── vocab.txt               BPE vocabulary
#   └── run.sh                  wrapper that sets LD_LIBRARY_PATH

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"/{bin,lib,scripts}

# ── Binaries ─────────────────────────────────────────────────────────────

cp "$BIN_DIR/incremental_trainer" "$STAGE_DIR/bin/"
cp "$BIN_DIR/chatbot_api_server"  "$STAGE_DIR/bin/"

# ── Install scripts and support files ────────────────────────────────────

cp "$SCRIPT_DIR/install_chatbot_API.sh"         "$STAGE_DIR/scripts/"
cp "$SCRIPT_DIR/install_incremental_trainer.sh"  "$STAGE_DIR/scripts/"
cp "$SCRIPT_DIR/install_oneapi_libs.sh"          "$STAGE_DIR/scripts/"
cp "$SCRIPT_DIR/adai.service"                    "$STAGE_DIR/scripts/"
chmod +x "$STAGE_DIR/scripts/install_chatbot_API.sh"
chmod +x "$STAGE_DIR/scripts/install_incremental_trainer.sh"
chmod +x "$STAGE_DIR/scripts/install_oneapi_libs.sh"

# install_chatbot_API.sh expects ${REPO_ROOT}/build/chatbot_api_server;
# install_incremental_trainer.sh defaults to ${REPO_ROOT}/build-gpu-clang/bin/.
# Create compatibility symlinks so both scripts work from the tarball layout.
mkdir -p "$STAGE_DIR/build"
ln -sf ../bin/chatbot_api_server "$STAGE_DIR/build/chatbot_api_server"
mkdir -p "$STAGE_DIR/build-gpu-clang"
ln -sf ../bin "$STAGE_DIR/build-gpu-clang/bin"

# ── Config and vocabulary ────────────────────────────────────────────────

if [[ -f "$PROJECT_ROOT/config.conf" ]]; then
    cp "$PROJECT_ROOT/config.conf" "$STAGE_DIR/"
else
    echo "WARNING: config.conf not found at $PROJECT_ROOT/config.conf — skipped"
fi

if [[ -f "$PROJECT_ROOT/vocab.txt" ]]; then
    cp "$PROJECT_ROOT/vocab.txt" "$STAGE_DIR/"
else
    echo "WARNING: vocab.txt not found at $PROJECT_ROOT/vocab.txt — skipped"
fi

# ── Intel oneAPI shared libraries ────────────────────────────────────────
# Collect every .so from /opt/intel that the binaries actually need, plus
# their transitive dependencies (some MKL libs pull in others).

collect_intel_libs() {
    local binary="$1"
    ldd "$binary" 2>/dev/null \
        | grep '/opt/intel/' \
        | awk '{print $3}' \
        | sort -u
}

declare -A SEEN
for binary in "$STAGE_DIR/bin/incremental_trainer" "$STAGE_DIR/bin/chatbot_api_server"; do
    while IFS= read -r lib; do
        [[ -z "$lib" ]] && continue
        real=$(readlink -f "$lib")
        base=$(basename "$real")
        if [[ -z "${SEEN[$base]:-}" ]]; then
            SEEN[$base]=1
            cp "$real" "$STAGE_DIR/lib/$base"

            # Preserve the soname symlink (e.g. libmkl_core.so.3 -> libmkl_core.so.3.x.y)
            soname=$(basename "$lib")
            if [[ "$soname" != "$base" ]] && [[ ! -e "$STAGE_DIR/lib/$soname" ]]; then
                ln -sf "$base" "$STAGE_DIR/lib/$soname"
            fi
        fi
    done < <(collect_intel_libs "$binary")
done

# Also resolve transitive Intel deps of the libs we just copied
resolve_transitive() {
    local changed=1
    while [[ $changed -eq 1 ]]; do
        changed=0
        for staged_lib in "$STAGE_DIR"/lib/*.so*; do
            [[ -L "$staged_lib" ]] && continue
            while IFS= read -r dep; do
                [[ -z "$dep" ]] && continue
                real=$(readlink -f "$dep")
                base=$(basename "$real")
                if [[ -z "${SEEN[$base]:-}" ]]; then
                    SEEN[$base]=1
                    cp "$real" "$STAGE_DIR/lib/$base"
                    soname=$(basename "$dep")
                    if [[ "$soname" != "$base" ]] && [[ ! -e "$STAGE_DIR/lib/$soname" ]]; then
                        ln -sf "$base" "$STAGE_DIR/lib/$soname"
                    fi
                    changed=1
                fi
            done < <(ldd "$staged_lib" 2>/dev/null | grep '/opt/intel/' | awk '{print $3}' | sort -u)
        done
    done
}
resolve_transitive

# ── SYCL backend plugins (dlopen'd at runtime, invisible to ldd) ────────
# The Unified Runtime Level Zero adapter is required for Intel GPU access.
# Also bundle the OpenCL adapter as a fallback.
COMPILER_LIB="${ONEAPI_ROOT}/compiler/latest/lib"
UMF_LIB="${ONEAPI_ROOT}/umf/latest/lib"
for pattern in \
    "libur_adapter_level_zero.so*" \
    "libur_adapter_level_zero_v2.so*" \
    "libur_adapter_opencl.so*" \
    "libze_loader.so*"; do
    for lib in "$COMPILER_LIB"/$pattern; do
        [[ -e "$lib" ]] || continue
        real=$(readlink -f "$lib")
        base=$(basename "$real")
        name=$(basename "$lib")
        # Copy the real file once
        if [[ -z "${SEEN[$base]:-}" ]]; then
            SEEN[$base]=1
            cp "$real" "$STAGE_DIR/lib/$base"
        fi
        # Recreate every symlink (e.g. .so -> .so.0 -> .so.0.12.0)
        if [[ "$name" != "$base" ]] && [[ ! -e "$STAGE_DIR/lib/$name" ]]; then
            ln -sf "$base" "$STAGE_DIR/lib/$name"
        fi
    done
done

# Runtime deps of the Level Zero adapter that live outside compiler/lib/:
#   libumf.so  — Unified Memory Framework (oneAPI umf component)
#   libhwloc.so — hardware locality (oneAPI tcm component)
for extra_dir in "$UMF_LIB" "${ONEAPI_ROOT}/tcm/latest/lib"; do
    [[ -d "$extra_dir" ]] || continue
    for lib in "$extra_dir"/lib*.so*; do
        [[ -e "$lib" ]] || continue
        real=$(readlink -f "$lib")
        base=$(basename "$real")
        name=$(basename "$lib")
        if [[ -z "${SEEN[$base]:-}" ]]; then
            SEEN[$base]=1
            cp "$real" "$STAGE_DIR/lib/$base"
        fi
        if [[ "$name" != "$base" ]] && [[ ! -e "$STAGE_DIR/lib/$name" ]]; then
            ln -sf "$base" "$STAGE_DIR/lib/$name"
        fi
    done
done

echo "  Bundled $(find "$STAGE_DIR/lib" -maxdepth 1 -type f | wc -l) Intel runtime libraries"

# ── Launcher script ──────────────────────────────────────────────────────

cat > "$STAGE_DIR/run.sh" << 'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UR_ADAPTERS_SEARCH_PATH="$SCRIPT_DIR/lib${UR_ADAPTERS_SEARCH_PATH:+:$UR_ADAPTERS_SEARCH_PATH}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  trainer    Run incremental_trainer"
    echo "  api        Run chatbot_api_server"
    echo ""
    echo "Examples:"
    echo "  $0 trainer init vocab.txt"
    echo "  $0 trainer train 10"
    echo "  $0 api --vocab vocab.txt --port 8080"
    exit 1
fi

CMD="$1"; shift
case "$CMD" in
    trainer|train)
        exec "$SCRIPT_DIR/bin/incremental_trainer" "$@"
        ;;
    api|server)
        exec "$SCRIPT_DIR/bin/chatbot_api_server" "$@"
        ;;
    *)
        echo "Unknown command: $CMD (use 'trainer' or 'api')"
        exit 1
        ;;
esac
LAUNCHER
chmod +x "$STAGE_DIR/run.sh"

# ── Create tarball ───────────────────────────────────────────────────────

TARBALL="$BUILD_DIR/$PACKAGE_NAME.tar.gz"
tar -czf "$TARBALL" -C "$BUILD_DIR" "$PACKAGE_NAME"

# ── Summary ──────────────────────────────────────────────────────────────

TARBALL_SIZE=$(du -sh "$TARBALL" | cut -f1)
echo ""
echo "=== Package created ==="
echo "  $TARBALL ($TARBALL_SIZE)"
echo ""
echo "To deploy on a machine with an Intel ARC GPU:"
echo "  1. Install Intel compute runtime: sudo apt install intel-opencl-icd level-zero-gpu"
echo "  2. Extract:  tar xzf $PACKAGE_NAME.tar.gz"
echo "  3. Install oneAPI runtime libraries system-wide:"
echo "       sudo ./$PACKAGE_NAME/scripts/install_oneapi_libs.sh"
echo "  4. Quick run (uses LD_LIBRARY_PATH, no system install needed):"
echo "       ./$PACKAGE_NAME/run.sh trainer --help"
echo "       ./$PACKAGE_NAME/run.sh api --vocab vocab.txt"
echo "  5. Install as systemd services:"
echo "       sudo ./$PACKAGE_NAME/scripts/install_chatbot_API.sh"
echo "       sudo ./$PACKAGE_NAME/scripts/install_incremental_trainer.sh"
