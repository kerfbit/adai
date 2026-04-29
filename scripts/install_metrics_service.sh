#!/bin/bash
# ADAI Metrics API Server - systemd Service Installation Script
#
# This script installs and configures the ADAI metrics API server as a
# systemd service. It creates necessary directories, sets permissions,
# and enables/starts the service.
#
# Usage:
#   sudo ./install_metrics_service.sh [OPTIONS]
#
# Options:
#   --install-path PATH    Installation directory (default: /opt/adai)
#   --user USER           Service user (default: adai)
#   --group GROUP         Service group (default: adai)
#   --port PORT           Metrics server port (default: 8081)
#   --metrics-dir DIR     Training sessions/metrics directory
#                           (default: /opt/adai/training_sessions)
#   --help                Show this help message
#
# Examples:
#   sudo ./install_metrics_service.sh
#   sudo ./install_metrics_service.sh --install-path /usr/local/adai --port 9081

set -euo pipefail

# ============================================================================
# Configuration Defaults
# ============================================================================

INSTALL_PATH="/opt/adai"
SERVICE_USER="adai"
SERVICE_GROUP="adai"
METRICS_PORT=8081

# Derived paths (may be overridden by arguments)
METRICS_DIR=""

# Fixed paths
SERVICE_NAME="adai-metrics"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

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
ADAI Metrics API Server - systemd Service Installation Script

Usage: sudo $0 [OPTIONS]

Options:
  --install-path PATH    Installation directory (default: /opt/adai)
  --user USER           Service user (default: adai)
  --group GROUP         Service group (default: adai)
  --port PORT           Metrics server port (default: 8081)
  --metrics-dir DIR     Training sessions/metrics directory
                          (default: <install-path>/training_sessions)
  --help                Show this help message

Examples:
  # Default installation
  sudo $0

  # Custom port
  sudo $0 --port 9081

  # Custom install path with existing metrics data
  sudo $0 --install-path /usr/local/adai --metrics-dir /data/adai/sessions

Description:
  This script automates the installation of the ADAI metrics API server as a
  systemd service. It performs the following actions:

  1. Creates a system user and group for the service (if needed)
  2. Creates installation and metrics directories
  3. Copies the metrics_api_server executable
  4. Sets ownership and permissions
  5. Installs and enables the systemd service file
  6. Starts the service

Requirements:
  - Root privileges (run with sudo)
  - Built metrics_api_server executable (build/src/metrics_api_server)

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-path)
            INSTALL_PATH="$2"
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
            METRICS_PORT="$2"
            shift 2
            ;;
        --metrics-dir)
            METRICS_DIR="$2"
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

# Apply defaults that depend on INSTALL_PATH
BIN_DIR="${INSTALL_PATH}/bin"
if [[ -z "${METRICS_DIR}" ]]; then
    METRICS_DIR="${INSTALL_PATH}/training_sessions"
fi

# ============================================================================
# Preflight Checks
# ============================================================================

info "Starting ADAI metrics API server systemd service installation..."
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

# Check if metrics_api_server executable exists
METRICS_BIN="${REPO_ROOT}/build/src/metrics_api_server"
if [[ ! -f "${METRICS_BIN}" ]]; then
    error "metrics_api_server executable not found at ${METRICS_BIN}"
    error "Please build the project first:"
    error "  cd ${REPO_ROOT} && cmake -B build && cmake --build build --target metrics_api_server"
    exit 1
fi

success "Preflight checks passed"
echo ""

# ============================================================================
# Installation Summary
# ============================================================================

info "Installation Configuration:"
echo "  Installation Path:  ${INSTALL_PATH}"
echo "  Binary Directory:   ${BIN_DIR}"
echo "  Metrics Directory:  ${METRICS_DIR}"
echo "  Service User:       ${SERVICE_USER}"
echo "  Service Group:      ${SERVICE_GROUP}"
echo "  Metrics Port:       ${METRICS_PORT}"
echo "  Service File:       ${SERVICE_FILE}"
echo ""

read -r -p "Continue with installation? (y/N) " -n 1
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    warn "Installation cancelled by user"
    exit 0
fi

# ============================================================================
# Step 1: Create Service User and Group
# ============================================================================

info "[1/6] Creating service user and group..."

if id "${SERVICE_USER}" &>/dev/null; then
    warn "User '${SERVICE_USER}' already exists, skipping creation"
else
    useradd -r -s /bin/false -d "${INSTALL_PATH}" -c "ADAI Service" "${SERVICE_USER}"
    success "Created user '${SERVICE_USER}'"
fi

# ============================================================================
# Step 2: Create Directory Structure
# ============================================================================

info "[2/6] Creating directory structure..."

mkdir -p "${BIN_DIR}"
mkdir -p "${METRICS_DIR}"

success "Created directories"

# ============================================================================
# Step 3: Copy Executable
# ============================================================================

info "[3/6] Copying metrics_api_server executable..."

cp "${METRICS_BIN}" "${BIN_DIR}/metrics_api_server"
chmod 755 "${BIN_DIR}/metrics_api_server"

success "Copied executable to ${BIN_DIR}/metrics_api_server"

# ============================================================================
# Step 4: Set Ownership and Permissions
# ============================================================================

info "[4/6] Setting ownership and permissions..."

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"

# Metrics dir may be outside INSTALL_PATH if --metrics-dir was specified
if [[ "${METRICS_DIR}" != "${INSTALL_PATH}"* ]]; then
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${METRICS_DIR}"
fi

chmod 755 "${METRICS_DIR}"

success "Set ownership and permissions"

# ============================================================================
# Step 5: Install systemd Service File
# ============================================================================

info "[5/6] Installing systemd service file..."

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=ADAI Training Metrics API Server
Documentation=https://github.com/adai/docs/TRAINING_METRICS_API.md
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_PATH}
ExecStart=${BIN_DIR}/metrics_api_server
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

# Environment variables
Environment="METRICS_PORT=${METRICS_PORT}"
Environment="METRICS_DIR=${METRICS_DIR}"

# Security hardening
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${METRICS_DIR}

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "${SERVICE_FILE}"
success "Installed systemd service file at ${SERVICE_FILE}"

# ============================================================================
# Step 6: Reload systemd, Enable, and Start Service
# ============================================================================

info "[6/6] Reloading systemd and enabling service..."

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"

success "Service enabled (will start on boot)"

info "Starting service..."

if systemctl start "${SERVICE_NAME}.service"; then
    success "Service started successfully"
else
    error "Failed to start service"
    warn "Check logs with: sudo journalctl -u ${SERVICE_NAME} -n 50"
    exit 1
fi

# ============================================================================
# Verify Installation
# ============================================================================

echo ""
info "Verifying installation..."
sleep 2

if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
    success "Service is running"
else
    warn "Service is not running"
    systemctl status "${SERVICE_NAME}.service" --no-pager || true
fi

if command -v ss &> /dev/null; then
    if ss -tlnp | grep -q ":${METRICS_PORT}"; then
        success "Server is listening on port ${METRICS_PORT}"
    else
        warn "Server does not appear to be listening on port ${METRICS_PORT} yet"
    fi
fi

# ============================================================================
# Installation Complete
# ============================================================================

echo ""
echo "========================================================================"
success "ADAI metrics API server systemd service installation complete!"
echo "========================================================================"
echo ""
echo "Service Management:"
echo "  Status:  systemctl status ${SERVICE_NAME}"
echo "  Logs:    journalctl -u ${SERVICE_NAME} -f"
echo "  Stop:    systemctl stop ${SERVICE_NAME}"
echo "  Start:   systemctl start ${SERVICE_NAME}"
echo "  Restart: systemctl restart ${SERVICE_NAME}"
echo "  Disable: systemctl disable ${SERVICE_NAME}"
echo ""
echo "Configuration:"
echo "  Metrics Port: ${METRICS_PORT}"
echo "  Metrics Dir:  ${METRICS_DIR}"
echo ""
echo "Testing:"
echo "  Health:   curl http://localhost:${METRICS_PORT}/health"
echo "  Sessions: curl http://localhost:${METRICS_PORT}/api/sessions"
echo "  Metrics:  curl http://localhost:${METRICS_PORT}/api/metrics/latest"
echo ""
echo "Logs:"
echo "  View:    journalctl -u ${SERVICE_NAME} -n 50"
echo "  Follow:  journalctl -u ${SERVICE_NAME} -f"
echo "  Errors:  journalctl -u ${SERVICE_NAME} -p err"
echo ""
echo "========================================================================"
echo ""
