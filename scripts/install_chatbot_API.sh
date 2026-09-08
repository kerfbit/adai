#!/bin/bash

# @adai-status: beta
# @adai-version: 0.8.0
# @adai-reviewed: 2026-09-07

# ADAI Chatbot API - systemd Service Installation Script
#
# This script installs and configures the ADAI chatbot API server as a
# systemd service.  It creates necessary directories, users, and
# configuration files.
#
# Usage:
#   sudo ./install_chatbot_API.sh [OPTIONS]
#
# Options:
#   --install-path PATH    Installation directory (default: /opt/adai)
#   --build-dir PATH       Build directory relative to repo root (default: build/portable)
#   --user USER           Service user (default: adai)
#   --group GROUP         Service group (default: adai)
#   --port PORT           Server port (default: 8080)
#   --help                Show this help message
#
# Examples:
#   sudo ./install_chatbot_API.sh
#   sudo ./install_chatbot_API.sh --install-path /usr/local/adai --port 9000
#   sudo ./install_chatbot_API.sh --build-dir build/sycl   # GPU-accelerated build

set -euo pipefail

# ============================================================================
# Configuration Defaults
# ============================================================================

INSTALL_PATH="/opt/adai"
SERVICE_USER="adai"
SERVICE_GROUP="adai"
SERVER_PORT=8080
LOG_LEVEL="INFO"
BUILD_DIR="build/portable"

# Derived paths
BIN_DIR="${INSTALL_PATH}/bin"
VOCAB_DIR="${INSTALL_PATH}/vocab"
MODELS_DIR="${INSTALL_PATH}/models"
LOG_DIR="/var/log/adai"
CONFIG_FILE="/etc/adai/config.conf"
SERVICE_FILE="/etc/systemd/system/adai.service"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# validate_build_dir: relative path with no '..' traversal (allows slashes for preset sub-dirs)
validate_build_dir() {
    local flag="$1" val="$2"
    if [[ -z "${val}" ]]; then
        echo "ERROR: ${flag}: value must not be empty" >&2
        exit 1
    fi
    if [[ ! "${val}" =~ ^[a-zA-Z0-9._/-]+$ ]] || [[ "${val}" =~ \.\. ]]; then
        echo "ERROR: ${flag}: '${val}' must be a relative path with no '..' (allowed: a-z A-Z 0-9 . _ - /)" >&2
        exit 1
    fi
}

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# ============================================================================
# Argument Parsing
# ============================================================================

show_help() {
    cat <<EOF
ADAI Chatbot API - systemd Service Installation Script

Usage: sudo $0 [OPTIONS]

Options:
  --install-path PATH    Installation directory (default: /opt/adai)
  --build-dir PATH       Build directory relative to repo root, e.g. build/portable,
                         build/sycl, build/gpu (default: build/portable)
  --user USER           Service user (default: adai)
  --group GROUP         Service group (default: adai)
  --port PORT           Server port (default: 8080)
  --log-level LEVEL     Log level: DEBUG, INFO, WARN, ERROR (default: INFO)
  --help                Show this help message

Examples:
  # Default installation (portable, CPU-only build)
  sudo $0

  # Custom installation path
  sudo $0 --install-path /usr/local/adai

  # Custom user and port
  sudo $0 --user mychatbot --port 9000

  # Install a GPU-accelerated (SYCL) build — GPU_ENABLED and device group
  # access are configured automatically when a GPU build is detected
  sudo $0 --build-dir build/sycl

Description:
  This script automates the installation of the ADAI chatbot as a systemd
  service. It performs the following actions:

  1. Creates a system user and group for the service
  2. Creates installation directories
  3. Copies the chatbot executable and vocabulary files
  4. Creates a configuration file
  5. Installs and enables the systemd service file
  6. Starts the service

Requirements:
  - Root privileges (run with sudo)
  - Built chatbot_api_server executable at <build-dir>/bin/chatbot_api_server
    (build with: cmake --preset=<preset> && cmake --build --preset=<preset> --target chatbot_api_server)
  - Vocabulary file (vocab.txt)

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-path)
            INSTALL_PATH="$2"
            shift 2
            ;;
        --build-dir)
            validate_build_dir "--build-dir" "$2"
            BUILD_DIR="$2"
            shift 2
            ;;
        --user)
            SERVICE_USER="$2"
            shift 2
            ;;
        --group)
            SERVICE_GROUP="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Update derived paths
BIN_DIR="${INSTALL_PATH}/bin"
VOCAB_DIR="${INSTALL_PATH}/vocab"
MODELS_DIR="${INSTALL_PATH}/models"
BUILD_BIN_DIR="${REPO_ROOT}/${BUILD_DIR}/bin"

# ============================================================================
# Preflight Checks
# ============================================================================

info "Starting ADAI chatbot systemd service installation..."
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    error "systemd is not available on this system"
    exit 1
fi

# Check if chatbot_api_server executable exists
if [[ ! -f "${BUILD_BIN_DIR}/chatbot_api_server" ]]; then
    error "chatbot_api_server executable not found at ${BUILD_BIN_DIR}/"
    error "Build it first, e.g.: cmake --preset=portable && cmake --build --preset=portable --target chatbot_api_server"
    error "(use --build-dir to point at a different build, e.g. build/sycl for a GPU build)"
    exit 1
fi

# Check if vocab.txt exists
if [[ ! -f "${REPO_ROOT}/vocab.txt" ]]; then
    error "vocab.txt not found in ${REPO_ROOT}/"
    exit 1
fi

# Detect whether this binary was built with GPU (SYCL/CUDA) support: the
# "[GPU] GPU ready." log format string is only compiled in under
# #ifdef ADAI_ENABLE_GPU (see ChatbotAPIServer.cpp), so its presence in the
# binary reliably indicates a GPU-enabled build regardless of runtime config.
IS_GPU_BUILD=false
if strings "${BUILD_BIN_DIR}/chatbot_api_server" 2>/dev/null | grep -q "GPU ready\."; then
    IS_GPU_BUILD=true
fi

success "Preflight checks passed"
if [[ "${IS_GPU_BUILD}" == true ]]; then
    info "Detected GPU-enabled binary — will configure GPU_ENABLED, device group access, and relax PrivateDevices"
else
    info "Detected CPU-only binary — GPU device access will remain locked down"
fi
echo ""

# ============================================================================
# Installation Summary
# ============================================================================

info "Installation Configuration:"
echo "  Installation Path: ${INSTALL_PATH}"
echo "  Build Directory:   ${BUILD_BIN_DIR}"
echo "  GPU Build:         ${IS_GPU_BUILD}"
echo "  Binary Directory:  ${BIN_DIR}"
echo "  Vocabulary Path:   ${VOCAB_DIR}"
echo "  Models Directory:  ${MODELS_DIR}"
echo "  Log Directory:     ${LOG_DIR}"
echo "  Service User:      ${SERVICE_USER}"
echo "  Service Group:     ${SERVICE_GROUP}"
echo "  Server Port:       ${SERVER_PORT}"
echo "  Config File:       ${CONFIG_FILE}"
echo "  Service File:      ${SERVICE_FILE}"
echo ""

read -p "Continue with installation? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    warn "Installation cancelled by user"
    exit 0
fi

# ============================================================================
# Step 1: Create Service User and Group
# ============================================================================

info "[1/8] Creating service user and group..."

if id "${SERVICE_USER}" &>/dev/null; then
    warn "User '${SERVICE_USER}' already exists, skipping creation"
else
    useradd -r -s /bin/false -d "${INSTALL_PATH}" -c "ADAI Chatbot Service" "${SERVICE_USER}"
    success "Created user '${SERVICE_USER}'"
fi

# GPU builds need the service account in the group that owns the GPU device
# node (commonly "render" for /dev/dri/renderD128 on Intel Arc) — without
# this, the server silently falls back to CPU even with PrivateDevices
# disabled and GPU_ENABLED=true.
if [[ "${IS_GPU_BUILD}" == true ]]; then
    if getent group render &>/dev/null; then
        usermod -aG render "${SERVICE_USER}"
        success "Added '${SERVICE_USER}' to the 'render' group for GPU device access"
    else
        warn "GPU build detected but no 'render' group exists on this system"
        warn "GPU device access will likely fail — verify GPU drivers are installed"
    fi
fi

# ============================================================================
# Step 2: Create Directory Structure
# ============================================================================

info "[2/8] Creating directory structure..."

mkdir -p "${BIN_DIR}"
mkdir -p "${VOCAB_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${CONFIG_FILE}")"

success "Created directories"

# ============================================================================
# Step 3: Copy Executable and Resources
# ============================================================================

info "[3/8] Copying executable and resources..."

# Copy executable
cp "${BUILD_BIN_DIR}/chatbot_api_server" "${BIN_DIR}/"
chmod 755 "${BIN_DIR}/chatbot_api_server"

# Copy vocabulary
cp "${REPO_ROOT}/vocab.txt" "${VOCAB_DIR}/"

# Copy model files if they exist
if [[ -d "${REPO_ROOT}/models" ]]; then
    cp -r "${REPO_ROOT}/models/"* "${MODELS_DIR}/" 2>/dev/null || true
fi

success "Copied executable and resources"

# ============================================================================
# Step 4: Set Ownership and Permissions
# ============================================================================

info "[4/8] Setting ownership and permissions..."

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"

# Make models directory writable for model saving
chmod 755 "${MODELS_DIR}"

# Ensure vocabulary is readable
chmod 644 "${VOCAB_DIR}/vocab.txt"

success "Set ownership and permissions"

# ============================================================================
# Step 5: Create Configuration File
# ============================================================================

info "[5/8] Creating configuration file..."

GPU_CONFIG_BLOCK=""
if [[ "${IS_GPU_BUILD}" == true ]]; then
    GPU_CONFIG_BLOCK="
# ============================================================================
# GPU Configuration (auto-detected GPU build)
# ============================================================================

GPU_ENABLED=true
GPU_DEVICE_ID=0
GPU_MEMORY_FRACTION=0.9
GPU_STRATEGY=full
"
fi

cat > "${CONFIG_FILE}" <<EOF
# ADAI Chatbot Service Configuration
# Generated by install_chatbot_API.sh on $(date)

# ============================================================================
# Server Configuration
# ============================================================================

VOCAB_PATH=${VOCAB_DIR}/vocab.txt
PORT=${SERVER_PORT}
SESSION_TIMEOUT=30
LOG_LEVEL=${LOG_LEVEL}

# Optional: Path to pretrained model
# MODEL_PATH=${MODELS_DIR}/model.bin
${GPU_CONFIG_BLOCK}
# ============================================================================
# Model Architecture Parameters
# ============================================================================

D_MODEL=512
NUM_HEADS=8
D_FF=2048
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
MAX_SEQ_LENGTH=1024

# ============================================================================
# Text Generation Parameters
# ============================================================================

MAX_LENGTH=100
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=50
BEAM_WIDTH=4
STRATEGY=nucleus
EOF

chmod 644 "${CONFIG_FILE}"
success "Created configuration file at ${CONFIG_FILE}"

# ============================================================================
# Step 6: Install systemd Service File
# ============================================================================

info "[6/8] Installing systemd service file..."

# Update service file with installation paths
# GPU builds need the device group and a private-/dev exemption; CPU-only
# builds keep the hardened defaults (no supplementary groups, private /dev).
SUPPLEMENTARY_GROUPS=""
PRIVATE_DEVICES="yes"
if [[ "${IS_GPU_BUILD}" == true ]]; then
    SUPPLEMENTARY_GROUPS="render"
    PRIVATE_DEVICES="no"
fi

sed -e "s|WorkingDirectory=.*|WorkingDirectory=${INSTALL_PATH}|" \
    -e "s|ExecStart=.*|ExecStart=${BIN_DIR}/chatbot_api_server|" \
    -e "s|User=.*|User=${SERVICE_USER}|" \
    -e "s|Group=.*|Group=${SERVICE_GROUP}|" \
    -e "s|^SupplementaryGroups=.*|SupplementaryGroups=${SUPPLEMENTARY_GROUPS}|" \
    -e "s|Environment=\"VOCAB_PATH=.*\"|Environment=\"VOCAB_PATH=${VOCAB_DIR}/vocab.txt\"|" \
    -e "s|Environment=\"PORT=.*\"|Environment=\"PORT=${SERVER_PORT}\"|" \
    -e "s|Environment=\"LOG_LEVEL=.*\"|Environment=\"LOG_LEVEL=${LOG_LEVEL}\"|" \
    -e "s|Environment=\"CONFIG_FILE=.*\"|Environment=\"CONFIG_FILE=${CONFIG_FILE}\"|" \
    -e "s|ReadWritePaths=.*|ReadWritePaths=${LOG_DIR} ${MODELS_DIR}|" \
    -e "s|^PrivateDevices=.*|PrivateDevices=${PRIVATE_DEVICES}|" \
    "${SCRIPT_DIR}/adai.service" > "${SERVICE_FILE}"

success "Installed systemd service file"
if [[ "${IS_GPU_BUILD}" == true ]]; then
    info "  GPU build: SupplementaryGroups=render, PrivateDevices=no"
fi

# ============================================================================
# Step 7: Reload systemd and Enable Service
# ============================================================================

info "[7/8] Reloading systemd and enabling service..."

systemctl daemon-reload
systemctl enable adai.service

success "Service enabled (will start on boot)"

# ============================================================================
# Step 8: Start Service
# ============================================================================

info "[8/8] Starting service..."

if systemctl start adai.service; then
    success "Service started successfully"
else
    error "Failed to start service"
    warn "Check logs with: sudo journalctl -u adai -n 50"
    exit 1
fi

# ============================================================================
# Verify Installation
# ============================================================================

echo ""
info "Verifying installation..."
sleep 2

# Check service status
if systemctl is-active --quiet adai.service; then
    success "Service is running"
else
    warn "Service is not running"
    systemctl status adai.service --no-pager || true
fi

# Check if port is listening
if command -v ss &> /dev/null; then
    if ss -tlnp | grep -q ":${SERVER_PORT}"; then
        success "Server is listening on port ${SERVER_PORT}"
    else
        warn "Server does not appear to be listening on port ${SERVER_PORT}"
    fi
fi

# ============================================================================
# Installation Complete
# ============================================================================

echo ""
echo "========================================================================"
success "ADAI Chatbot systemd service installation complete!"
echo "========================================================================"
echo ""
echo "Service Information:"
echo "  Status:  systemctl status adai"
echo "  Logs:    journalctl -u adai -f"
echo "  Stop:    systemctl stop adai"
echo "  Start:   systemctl start adai"
echo "  Restart: systemctl restart adai"
echo "  Disable: systemctl disable adai"
echo ""
echo "Configuration:"
echo "  File:    ${CONFIG_FILE}"
echo "  Restart service after editing: sudo systemctl restart adai"
echo ""
echo "Testing:"
echo "  Health:  curl http://localhost:${SERVER_PORT}/health"
echo "  Chat:    curl -X POST http://localhost:${SERVER_PORT}/chat \\"
echo "             -H 'Content-Type: application/json' \\"
echo "             -d '{\"message\": \"Hello\"}'"
echo ""
echo "Logs:"
echo "  View:    journalctl -u adai -n 50"
echo "  Follow:  journalctl -u adai -f"
echo "  Errors:  journalctl -u adai -p err"
echo ""
echo "========================================================================"
echo ""
