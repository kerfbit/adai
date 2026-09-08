#!/bin/bash

# @adai-status: beta
# @adai-version: 0.8.0
# @adai-reviewed: 2026-09-07

# ADAI Server Bundle — Packaging Script
#
# Builds a self-contained tarball containing everything needed to deploy the
# ADAI server bundle on a fresh machine:
#
#   binaries, install scripts, SQL schema, config templates, dashboard
#
# The resulting tarball can be copied to a target host and extracted:
#
#   scp adai-server-bundle-*.tar.gz target:/tmp/
#   ssh target 'cd /tmp && tar xzf adai-server-bundle-*.tar.gz'
#   ssh target 'cd /tmp/adai-server-bundle-* && sudo scripts/install_server_bundle.sh --build-dir . --yes'
#
# Usage:
#   ./scripts/package_server_bundle.sh [OPTIONS]
#
# See --help for the full option list.

set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================

BUILD_DIR=""
OUTPUT_DIR="."
VERSION=""
STRIP_BINARIES=true
INCLUDE_TRAINER=false
INCLUDE_CHATBOT_API=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat <<'EOF'
ADAI Server Bundle — Packaging Script

Usage: ./scripts/package_server_bundle.sh [OPTIONS]

Options:
  --build-dir DIR       CMake build directory containing bin/ (auto-detected if omitted)
  --output-dir DIR      Directory to write the tarball to (default: .)
  --version TAG         Version string for the tarball name (default: git describe or date)
  --no-strip            Do not strip debug symbols from binaries
  --include-trainer     Also include incremental_trainer and dataset_manager
  --include-chatbot-api Also include chatbot_api_server
  --help                Show this help message

Auto-detection:
  If --build-dir is not specified, the script looks (in order) for:
    1. build/portable/bin/metrics_api_server
    2. build/bin/metrics_api_server
    3. build-clang-release/bin/metrics_api_server

Tarball contents:
  adai-server-bundle-<version>/
    bin/                            Server binaries
    scripts/
      install_server_bundle.sh      Bundle installer (systemd + config)
      install_incremental_trainer.sh  Trainer installer (optional)
      install_metrics_service.sh    Standalone metrics installer
      install_mns_server.sh         Standalone MNS installer
      setup_postgres.sql            PostgreSQL schema for metrics + MNS
    config.conf                     Default configuration template
    config-remote.conf              Remote/distributed configuration template
    vocab.txt                       Default vocabulary file
    dashboard.html                  Training metrics dashboard
    README.txt                      Quick-start instructions

Installation on target host:
  tar xzf adai-server-bundle-<version>.tar.gz
  cd adai-server-bundle-<version>
  sudo scripts/install_server_bundle.sh --build-dir . --yes

  With PostgreSQL:
  sudo scripts/install_server_bundle.sh --build-dir . --setup-postgres --yes

EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-dir)    BUILD_DIR="$2";      shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";     shift 2 ;;
        --version)      VERSION="$2";        shift 2 ;;
        --no-strip)     STRIP_BINARIES=false; shift ;;
        --include-trainer) INCLUDE_TRAINER=true; shift ;;
        --include-chatbot-api) INCLUDE_CHATBOT_API=true; shift ;;
        --help)         show_help; exit 0 ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Auto-detect build directory
# ============================================================================

if [[ -z "${BUILD_DIR}" ]]; then
    for candidate in \
        "${REPO_ROOT}/build/portable" \
        "${REPO_ROOT}/build" \
        "${REPO_ROOT}/build-clang-release"; do
        if [[ -f "${candidate}/bin/metrics_api_server" ]]; then
            BUILD_DIR="${candidate}"
            break
        fi
    done

    if [[ -z "${BUILD_DIR}" ]]; then
        error "Could not auto-detect build directory"
        error "  Build first:  cmake --preset portable && cmake --build --preset portable"
        error "  Or specify:   --build-dir <path>"
        exit 1
    fi
fi

BUILD_BIN="${BUILD_DIR}/bin"

if [[ ! -d "${BUILD_BIN}" ]]; then
    error "Binary directory not found: ${BUILD_BIN}"
    exit 1
fi

# ============================================================================
# Derive version string
# ============================================================================

if [[ -z "${VERSION}" ]]; then
    if git -C "${REPO_ROOT}" describe --tags --always &>/dev/null; then
        VERSION="$(git -C "${REPO_ROOT}" describe --tags --always --dirty 2>/dev/null)"
    else
        VERSION="$(date +%Y%m%d)"
    fi
fi

BUNDLE_NAME="adai-server-bundle-${VERSION}"

info "Packaging ${BUNDLE_NAME}"
info "  Build directory: ${BUILD_DIR}"
info "  Output:          ${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"

# ============================================================================
# Preflight — verify required files exist
# ============================================================================

REQUIRED_BINS=(metrics_api_server registry_server mns_server mns_cli)
if [[ "${INCLUDE_TRAINER}" == true ]]; then
    REQUIRED_BINS+=(incremental_trainer dataset_manager)
fi
if [[ "${INCLUDE_CHATBOT_API}" == true ]]; then
    REQUIRED_BINS+=(chatbot_api_server)
fi

missing=()
for bin in "${REQUIRED_BINS[@]}"; do
    if [[ ! -f "${BUILD_BIN}/${bin}" ]]; then
        missing+=("${bin}")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    error "Missing binaries in ${BUILD_BIN}:"
    for m in "${missing[@]}"; do
        error "  - ${m}"
    done
    error ""
    error "Build with:  cmake --preset portable && cmake --build --preset portable"
    exit 1
fi

for f in config.conf config-remote.conf vocab.txt dashboard.html; do
    if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
        error "Missing file: ${REPO_ROOT}/${f}"
        exit 1
    fi
done

for s in install_server_bundle.sh setup_postgres.sql; do
    if [[ ! -f "${SCRIPT_DIR}/${s}" ]]; then
        error "Missing script: ${SCRIPT_DIR}/${s}"
        exit 1
    fi
done

success "All required files present"

# ============================================================================
# Stage
# ============================================================================

STAGING="$(mktemp -d)"
STAGE="${STAGING}/${BUNDLE_NAME}"
trap 'rm -rf "${STAGING}"' EXIT

mkdir -p "${STAGE}/bin"
mkdir -p "${STAGE}/scripts"

# --- Binaries ---
info "Copying binaries..."
for bin in "${REQUIRED_BINS[@]}"; do
    cp "${BUILD_BIN}/${bin}" "${STAGE}/bin/${bin}"
    if [[ "${STRIP_BINARIES}" == true ]] && command -v strip &>/dev/null; then
        strip "${STAGE}/bin/${bin}" 2>/dev/null || true
    fi
    success "  bin/${bin}"
done

# Optional: include mns_manager_gui if it was built
if [[ -f "${BUILD_BIN}/mns_manager_gui" ]]; then
    cp "${BUILD_BIN}/mns_manager_gui" "${STAGE}/bin/mns_manager_gui"
    if [[ "${STRIP_BINARIES}" == true ]] && command -v strip &>/dev/null; then
        strip "${STAGE}/bin/mns_manager_gui" 2>/dev/null || true
    fi
    success "  bin/mns_manager_gui (bonus)"
fi

# Optional: include vocab_builder if it was built
if [[ -f "${BUILD_BIN}/vocab_builder" ]]; then
    cp "${BUILD_BIN}/vocab_builder" "${STAGE}/bin/vocab_builder"
    if [[ "${STRIP_BINARIES}" == true ]] && command -v strip &>/dev/null; then
        strip "${STAGE}/bin/vocab_builder" 2>/dev/null || true
    fi
    success "  bin/vocab_builder (bonus)"
fi

# --- Scripts ---
info "Copying scripts..."
for s in install_server_bundle.sh install_incremental_trainer.sh \
         install_metrics_service.sh install_mns_server.sh \
         install_chatbot_API.sh setup_postgres.sql; do
    if [[ -f "${SCRIPT_DIR}/${s}" ]]; then
        cp "${SCRIPT_DIR}/${s}" "${STAGE}/scripts/${s}"
        success "  scripts/${s}"
    fi
done
chmod 755 "${STAGE}"/scripts/*.sh 2>/dev/null || true

# --- Config / data files ---
info "Copying configuration and data files..."
cp "${REPO_ROOT}/config.conf"        "${STAGE}/config.conf"
cp "${REPO_ROOT}/config-remote.conf" "${STAGE}/config-remote.conf"
cp "${REPO_ROOT}/vocab.txt"          "${STAGE}/vocab.txt"
cp "${REPO_ROOT}/dashboard.html"     "${STAGE}/dashboard.html"
success "  config.conf, config-remote.conf, vocab.txt, dashboard.html"

# --- README ---
cat > "${STAGE}/README.txt" <<'READMEEOF'
ADAI Server Bundle
==================

Quick start (SQLite — default):

  sudo scripts/install_server_bundle.sh --build-dir . --yes

Quick start (PostgreSQL):

  sudo scripts/install_server_bundle.sh --build-dir . --setup-postgres --yes

What gets installed:

  Service           Port   Description
  ─────────────────────────────────────────────────────────
  adai-mns          8083   Model Name Service
  adai-registry     8082   Dataset Registry Server
  adai-metrics      8081   Training Metrics REST API

  Default install path: /opt/adai
  Systemd units:        /etc/systemd/system/adai-{mns,registry,metrics}.service

Options:

  Run scripts/install_server_bundle.sh --help for all available options.

  Key flags:
    --install-path /custom/path   Change install location
    --storage-backend postgres    Use PostgreSQL instead of SQLite
    --setup-postgres              Install PostgreSQL packages and bootstrap schema
    --include-trainer             Include incremental_trainer (if packaged)

After installation:

  systemctl status adai-mns adai-registry adai-metrics
  curl http://localhost:8081/health
  curl http://localhost:8081/api/sessions

Training metrics dashboard:

  The dashboard.html file is installed to the install path.  Serve it with
  any static HTTP server pointed at the metrics API:

    python3 -m http.server 9090 --directory /opt/adai

READMEEOF
success "  README.txt"

# ============================================================================
# Pack
# ============================================================================

mkdir -p "${OUTPUT_DIR}"

info "Creating tarball..."
tar czf "${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz" \
    -C "${STAGING}" \
    "${BUNDLE_NAME}"

TARBALL="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
TAR_SIZE="$(du -h "${TARBALL}" | cut -f1)"
FILE_COUNT="$(tar tzf "${TARBALL}" | wc -l)"

echo ""
echo "========================================================================"
success "Package created: ${TARBALL}"
echo "========================================================================"
echo ""
echo "  Size:    ${TAR_SIZE}"
echo "  Files:   ${FILE_COUNT}"
echo ""
echo "Contents:"
tar tzf "${TARBALL}" | sed 's/^/  /'
echo ""
echo "Deploy to a target host:"
echo "  scp ${TARBALL} user@host:/tmp/"
echo "  ssh user@host 'cd /tmp && tar xzf ${BUNDLE_NAME}.tar.gz'"
echo "  ssh user@host 'cd /tmp/${BUNDLE_NAME} && sudo scripts/install_server_bundle.sh --build-dir . --yes'"
echo ""
echo "========================================================================"
