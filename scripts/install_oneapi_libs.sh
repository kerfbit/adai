#!/usr/bin/env bash

# @adai-status: beta        (capped by TD-043 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.0
# @adai-reviewed: 2026-09-07

set -euo pipefail

# Install bundled Intel oneAPI shared libraries into the system linker
# cache so SYCL-built binaries resolve them without LD_LIBRARY_PATH.
#
# Expects to run from an extracted package created by package-sycl.sh,
# which bundles Intel runtime .so files into a lib/ directory.
#
# Usage:
#   sudo ./install_oneapi_libs.sh [OPTIONS]
#
# Options:
#   --lib-dir PATH       Path to bundled lib/ directory (default: auto-detected)
#   --install-path PATH  System destination for libraries (default: /opt/adai/lib)
#   --dry-run            Show what would be done without making changes
#   --uninstall          Remove installed libraries and ldconfig config
#   --help               Show this help message

# ============================================================================
# Configuration
# ============================================================================

LIB_DIR=""
INSTALL_PATH="/opt/adai/lib"
DRY_RUN=false
UNINSTALL=false
LDCONFIG_CONF="/etc/ld.so.conf.d/adai-oneapi.conf"

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lib-dir)
            LIB_DIR="$2"
            shift 2
            ;;
        --install-path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --help)
            sed -n '2,/^$/s/^# \?//p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ============================================================================
# Preflight
# ============================================================================

if [[ $EUID -ne 0 ]] && [[ "$DRY_RUN" == false ]]; then
    echo "ERROR: This script must be run as root (use sudo)."
    exit 1
fi

# ============================================================================
# Uninstall
# ============================================================================

if [[ "$UNINSTALL" == true ]]; then
    echo "Removing installed oneAPI libraries..."
    if [[ -f "$LDCONFIG_CONF" ]]; then
        echo "  Removing $LDCONFIG_CONF"
        [[ "$DRY_RUN" == false ]] && rm "$LDCONFIG_CONF"
    fi
    if [[ -d "$INSTALL_PATH" ]]; then
        echo "  Removing $INSTALL_PATH"
        [[ "$DRY_RUN" == false ]] && rm -rf "$INSTALL_PATH"
    fi
    if [[ "$DRY_RUN" == false ]]; then
        ldconfig
        echo "Done."
    else
        echo "--- DRY RUN — no changes made ---"
    fi
    exit 0
fi

# ============================================================================
# Locate bundled lib/ directory
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$LIB_DIR" ]]; then
    # Walk up from scripts/ to the package root, then into lib/
    PACKAGE_ROOT="$(dirname "$SCRIPT_DIR")"
    if [[ -d "$PACKAGE_ROOT/lib" ]]; then
        LIB_DIR="$PACKAGE_ROOT/lib"
    else
        echo "ERROR: Could not find bundled lib/ directory."
        echo "Expected at $PACKAGE_ROOT/lib"
        echo "Specify it with --lib-dir."
        exit 1
    fi
fi

LIB_COUNT=$(find "$LIB_DIR" -maxdepth 1 \( -name "*.so" -o -name "*.so.*" \) -not -type d | wc -l)
if [[ "$LIB_COUNT" -eq 0 ]]; then
    echo "ERROR: No shared libraries found in $LIB_DIR"
    exit 1
fi

echo "Source:      $LIB_DIR ($LIB_COUNT libraries)"
echo "Destination: $INSTALL_PATH"
echo ""

# ============================================================================
# Install libraries
# ============================================================================

if [[ "$DRY_RUN" == true ]]; then
    echo "--- DRY RUN ---"
    echo "Would copy $LIB_COUNT libraries to $INSTALL_PATH"
    echo "Would write $LDCONFIG_CONF pointing to $INSTALL_PATH"
    echo "Would run ldconfig"
    exit 0
fi

mkdir -p "$INSTALL_PATH"
cp -a "$LIB_DIR"/*.so* "$INSTALL_PATH/"

echo "Copied $LIB_COUNT libraries to $INSTALL_PATH"

# ============================================================================
# Register with ldconfig
# ============================================================================

cat > "$LDCONFIG_CONF" << EOF
# Intel oneAPI runtime libraries for ADAI
# Installed by install_oneapi_libs.sh on $(date -Iseconds)
$INSTALL_PATH
EOF

echo "Wrote $LDCONFIG_CONF"

ldconfig

# ============================================================================
# Verify
# ============================================================================

echo ""
echo "Verifying key libraries..."

FAILED=0
for lib in libsycl.so libmkl_core.so libtbb.so; do
    if ldconfig -p | grep -q "$lib"; then
        echo "  OK: $lib"
    else
        echo "  MISSING: $lib"
        FAILED=1
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "Done. oneAPI runtime libraries installed to $INSTALL_PATH."
else
    echo "Warning: Some expected libraries were not found in the cache."
    echo "This may be normal if the bundle was built with a subset of oneAPI."
fi
