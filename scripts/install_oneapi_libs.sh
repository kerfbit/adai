#!/usr/bin/env bash

# @adai-status: beta        (capped by TD-043 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.2
# @adai-reviewed: 2026-09-10

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
            # TD-120 (same class as TD-107): the previous `sed -n '2,/^$/s/^#
            # \?//p'` intended "print the header comment block up to its first
            # blank line," but the @adai-status/@adai-version/@adai-reviewed
            # tag block right after the shebang has its OWN blank line before
            # the real usage doc even starts — so the range ended there,
            # printing only the three tag lines instead of the actual usage
            # text. Skip the shebang, tag lines, and any non-comment line
            # (blanks, `set -euo pipefail`) by pattern instead of by a
            # blank-line boundary that doesn't reliably mark the real doc's
            # start.
            awk '
                BEGIN { printing = 0 }
                !printing {
                    if ($0 ~ /^#!/ || $0 ~ /^# @adai-/) next
                    if ($0 !~ /^#/) next
                    printing = 1
                }
                printing {
                    if ($0 !~ /^#/) exit
                    sub(/^# ?/, ""); print
                }
            ' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# TD-154: --install-path had no validation at all, unlike every other
# install script's --install-path/--*-dir flags (all use validate_abs_path
# or stricter). The --uninstall branch below does `rm -rf "$INSTALL_PATH"`
# unconditionally (once --dry-run is off) — every other rm -rf-adjacent
# cleanup in this project's scripts moves data aside to a timestamped
# backup instead of deleting it outright specifically to avoid this class
# of mistake (see wipe_old_data() in install_server_bundle.sh and its
# siblings), but this uninstall path never got that treatment. An operator
# typo like `--install-path /home` (instead of the intended
# `/opt/adai/lib`) would recursively delete the given directory in full.
# validate_abs_path alone doesn't catch this — "/home" is a perfectly
# valid absolute path — so this checks path *depth* instead: the default
# and every realistic install path is at least two segments deep
# (/opt/adai/lib); every FHS top-level directory (/home, /etc, /usr, /var,
# /root, /tmp, /opt itself, and / itself) is one segment or fewer.
validate_install_path() {
    local flag="$1" val="$2"
    if [[ -z "${val}" ]]; then
        echo "ERROR: ${flag}: value must not be empty" >&2
        exit 1
    fi
    if [[ "${val}" != /* ]]; then
        echo "ERROR: ${flag}: '${val}' must be an absolute path (starting with /)" >&2
        exit 1
    fi
    local trimmed="${val%/}"
    local depth
    depth=$(tr -s '/' '\n' <<< "${trimmed}" | grep -c .)
    if (( depth < 2 )); then
        echo "ERROR: ${flag}: '${val}' is too shallow — refusing to recursively" >&2
        echo "  remove a top-level system directory. Use a path at least two" >&2
        echo "  segments deep, e.g. the default /opt/adai/lib." >&2
        exit 1
    fi
}
validate_install_path "--install-path" "$INSTALL_PATH"

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

# The auto-detect branch above already confirms $PACKAGE_ROOT/lib exists
# before assigning it, but a user-supplied --lib-dir skips that check
# entirely. Under `set -euo pipefail`, find failing on a nonexistent
# directory makes the `| wc -l` pipeline exit nonzero too (pipefail reports
# the rightmost failing command), which killed the script right here with a
# raw `find: '...': No such file or directory` instead of ever reaching the
# friendly "No shared libraries found" message below. Reproduced with
# `--lib-dir /nonexistent/path`.
if [[ ! -d "$LIB_DIR" ]]; then
    echo "ERROR: --lib-dir '$LIB_DIR' does not exist or is not a directory."
    exit 1
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
