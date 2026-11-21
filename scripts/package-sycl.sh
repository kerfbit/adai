#!/usr/bin/env bash

# @adai-status: beta        (tests: tests/scripts/package_sycl_test.sh)
# @adai-version: 0.8.1
# @adai-reviewed: 2026-11-21

set -euo pipefail

# Package incremental_trainer and chatbot_api_server (SYCL build) into a
# self-contained tarball with all Intel oneAPI runtime libraries, so the
# target machine only needs the Intel GPU compute runtime installed.
#
#   scripts/package-sycl.sh [--system-oneapi] [--build-dir DIR]
#
# --system-oneapi  (TD-575, ai-machine) bundle no oneAPI libraries: the host runs the release the
#                  build used from Intel's apt packages (MANIFEST.txt's "oneapi:" line). Adds
#                  trainer_service, vocab_builder and dataset_manager, scripts/ai_machine/'s deploy.sh,
#                  units and env files, and etc/config.trainer.conf and etc/config.chatbot.conf: the
#                  repo's configs with scripts/ai_machine/*.overrides.conf applied.
# --build-dir      the SYCL build (default build/sycl)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/sycl"
SYSTEM_ONEAPI=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system-oneapi)
            SYSTEM_ONEAPI=1
            shift
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --build-dir needs a value" >&2; exit 2; }
            BUILD_DIR="$(realpath -m "$2")"
            shift 2
            ;;
        -h | --help)
            sed -n '/^#   scripts\/package-sycl/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
BIN_DIR="$BUILD_DIR/bin"

VERSION=$(grep 'project(adai VERSION' "$PROJECT_ROOT/CMakeLists.txt" \
    | sed 's/.*VERSION \([0-9.]*\).*/\1/')
TIMESTAMP=$(date +%Y%m%d)
PACKAGE_NAME="adai-sycl-${VERSION}-${TIMESTAMP}"
[[ $SYSTEM_ONEAPI == 1 ]] && PACKAGE_NAME="adai-sycl-ai-machine-${VERSION}-$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)-${TIMESTAMP}"
STAGE_DIR="$BUILD_DIR/$PACKAGE_NAME"

# ── Preflight checks ────────────────────────────────────────────────────

if [[ ! -x "$BIN_DIR/incremental_trainer" ]] || [[ ! -x "$BIN_DIR/chatbot_api_server" ]]; then
    echo "ERROR: Build the sycl preset first:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    echo "  cmake --preset sycl"
    echo "  cmake --build build/sycl -j\$(nproc) --target incremental_trainer chatbot_api_server"
    exit 1
fi
EXTRA_BINARIES=(trainer_service vocab_builder dataset_manager)
if [[ $SYSTEM_ONEAPI == 1 ]]; then
    for b in "${EXTRA_BINARIES[@]}"; do
        [[ -x "$BIN_DIR/$b" ]] || {
            echo "ERROR: --system-oneapi packs $b too: cmake --build build/sycl --target ${EXTRA_BINARIES[*]}"
            exit 1
        }
    done
fi

if [[ -z "${ONEAPI_ROOT:-}" ]]; then
    if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
        # setvars.sh doesn't run under set -euo pipefail (it reads unset variables and returns
        # non-zero), which ended this script silently; relax the options while it runs.
        set +euo pipefail
        # shellcheck source=/dev/null
        source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
        set -euo pipefail
        [[ -n "${ONEAPI_ROOT:-}" ]] || {
            echo "ERROR: /opt/intel/oneapi/setvars.sh didn't set up oneAPI (ONEAPI_ROOT is unset)"
            exit 1
        }
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
if [[ $SYSTEM_ONEAPI == 1 ]]; then
    for b in "${EXTRA_BINARIES[@]}"; do
        cp "$BIN_DIR/$b" "$STAGE_DIR/bin/"
    done
fi

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

if [[ $SYSTEM_ONEAPI == 0 ]]; then
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
else
    rmdir "$STAGE_DIR/lib"
    # apply_overrides TEMPLATE OVERRIDES OUT: TEMPLATE with each KEY=VALUE of OVERRIDES replacing that
    # key's line (the first one), or appended when the template has none.
    apply_overrides() {
        awk 'NR == FNR {
                 if ($0 ~ /^[A-Z_]+=/) { k = substr($0, 1, index($0, "=") - 1); v[k] = $0; order[++n] = k }
                 next
             }
             {
                 if (match($0, /^[A-Z_]+=/)) {
                     k = substr($0, 1, RLENGTH - 1)
                     if (k in v && !(k in done)) { print v[k]; done[k] = 1; next }
                 }
                 print
             }
             END {
                 for (i = 1; i <= n; i++) if (!(order[i] in done)) {
                     if (!header++) print "\n# Added by package-sycl.sh --system-oneapi (scripts/ai_machine/*.overrides.conf)"
                     print v[order[i]]
                 }
             }' "$2" "$1" >"$3"
    }
    mkdir -p "$STAGE_DIR/etc" "$STAGE_DIR/units"
    apply_overrides "$PROJECT_ROOT/config.trainer.conf" "$SCRIPT_DIR/ai_machine/trainer.overrides.conf" \
        "$STAGE_DIR/etc/config.trainer.conf"
    apply_overrides "$PROJECT_ROOT/config.chatbot.conf" "$SCRIPT_DIR/ai_machine/chatbot.overrides.conf" \
        "$STAGE_DIR/etc/config.chatbot.conf"
    cp "$SCRIPT_DIR/ai_machine/trainer-lejepa.env" "$SCRIPT_DIR/ai_machine/trainer-chatbot.env" "$STAGE_DIR/etc/"
    cp "$SCRIPT_DIR/ai_machine/adai-trainer@.service" "$SCRIPT_DIR/ai_machine/adai.service" "$STAGE_DIR/units/"
    cp "$SCRIPT_DIR/ai_machine/deploy.sh" "$STAGE_DIR/"
    chmod +x "$STAGE_DIR/deploy.sh"
    echo "  No oneAPI libraries bundled (--system-oneapi); configs, units and deploy.sh added"
fi

# ── Launcher script ──────────────────────────────────────────────────────

cat > "$STAGE_DIR/run.sh" << 'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# The bundled libraries, or (a --system-oneapi package) Intel's apt runtime.
LIB_DIR="$SCRIPT_DIR/lib"
[[ -d "$LIB_DIR" ]] || LIB_DIR=/opt/intel/oneapi/redist/lib
export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UR_ADAPTERS_SEARCH_PATH="$LIB_DIR${UR_ADAPTERS_SEARCH_PATH:+:$UR_ADAPTERS_SEARCH_PATH}"

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

# ── Manifest ─────────────────────────────────────────────────────────────

{
    echo "$PACKAGE_NAME"
    echo "commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)$(git -C "$PROJECT_ROOT" diff --quiet HEAD -- src || echo -dirty)"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname -s 2>/dev/null || hostname)"
    echo "compiler: $(icpx --version 2>/dev/null | head -1)"
    # deploy.sh compares this with the host's apt runtime (--system-oneapi).
    echo "oneapi: $(icpx --version 2>/dev/null | head -1 | grep -o '([0-9]\{4\}\.[0-9]*\.[0-9]*' | tr -d '(')"
    echo "oneapi libraries: $([[ $SYSTEM_ONEAPI == 1 ]] && echo "the host's (Intel apt packages)" || echo bundled in lib/)"
    echo "files:"
    (cd "$STAGE_DIR" && find . -type f | sort | xargs sha256sum | sed 's/^/  /')
} >"$STAGE_DIR/MANIFEST.txt"

# ── Create tarball ───────────────────────────────────────────────────────

TARBALL="$BUILD_DIR/$PACKAGE_NAME.tar.gz"
tar -czf "$TARBALL" -C "$BUILD_DIR" "$PACKAGE_NAME"

# ── Summary ──────────────────────────────────────────────────────────────

TARBALL_SIZE=$(du -sh "$TARBALL" | cut -f1)
echo ""
echo "=== Package created ==="
echo "  $TARBALL ($TARBALL_SIZE)"
echo ""
if [[ $SYSTEM_ONEAPI == 1 ]]; then
    echo "SHA-256: $(sha256sum "$TARBALL" | cut -d' ' -f1)"
    echo "On ai-machine: tar xzf $PACKAGE_NAME.tar.gz && cd $PACKAGE_NAME && ./deploy.sh (check), then ./deploy.sh install"
    exit 0
fi
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
