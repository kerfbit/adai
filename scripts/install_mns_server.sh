#!/bin/bash
# ADAI Model Name Service - Installation Script
#
# Installs mns_server as a systemd service.
#
# Usage:
#   sudo ./install_mns_server.sh [OPTIONS]
#
# See --help for the full option list.

set -euo pipefail

# ============================================================================
# Configuration Defaults
# ============================================================================

INSTALL_PATH="/opt/adai"
SERVICE_USER="adai"
SERVICE_GROUP="adai"
BUILD_DIR="build/portable"
MNS_PORT=8083
YES=false
WIPE_DATA=false

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
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat <<EOF
ADAI Model Name Service - Installation Script

Usage: sudo $0 [OPTIONS]

Options:
  --install-path PATH   Installation root directory (default: /opt/adai)
  --user USER           Service user to own installed files (default: adai)
  --group GROUP         Service group (default: adai)
  --build-dir DIR       CMake build directory containing bin/ (default: build/portable)
  --port PORT           Listening port for mns_server (default: 8083)
  --wipe-data           Move any existing data directory aside to a timestamped
                        backup (<dir>.bak-<timestamp>) before reinstalling, so
                        the daemon starts fresh. Never permanently deletes.
  --yes                 Skip confirmation prompts (for non-interactive use)
  --help                Show this help message

Examples:
  # Default local installation
  sudo $0

  # Custom install path and port
  sudo $0 --install-path /usr/local/adai --port 9083

  # Custom build directory
  sudo $0 --build-dir build/release

  # Reinstall with a clean data directory (old data moved to a backup)
  sudo $0 --wipe-data

Description:
  Installs mns_server (ADAI Model Name Service daemon) to a local host.
  Creates the required directory layout, copies the binary, sets permissions
  and ownership, writes and enables a systemd service unit.

  The MNS daemon provides a stable model identity registry across all ADAI
  processes. It uses a SQLite backend stored in the data directory.

  Binary source: <build-dir>/bin/mns_server
  Install target: <install-path>/bin/mns_server
  Data directory: <install-path>/name_service/
  Service name:   adai-mns
  Default port:   8083

  Build first:
    cmake --preset portable && cmake --build --preset portable --target mns_server

EOF
}

# confirm PROMPT — skipped when --yes is set or stdin is not a terminal
confirm() {
    if [[ "${YES}" == true ]]; then
        info "$1 — skipped (--yes)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        error "stdin is not a terminal; use --yes to confirm non-interactively"
        exit 1
    fi
    read -r -p "$1 (y/N) " -n 1
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Installation cancelled by user"
        exit 0
    fi
}

# wipe_old_data SERVICE_NAME DIR [DIR2 ...]
# No-op if none of the dirs exist / are non-empty. Otherwise: warns, lists the
# exact directories, confirms (respects --yes like every other confirm() call),
# stops SERVICE_NAME first if it's currently active (files may be open/locked),
# then renames each existing non-empty dir to "<dir>.bak-<timestamp>" — never
# rm -rf. The subsequent mkdir -p in the normal install flow recreates it empty.
wipe_old_data() {
    local service_name="$1"; shift
    local dirs=("$@")
    local any=false
    for d in "${dirs[@]}"; do
        [[ -d "$d" && -n "$(ls -A "$d" 2>/dev/null)" ]] && any=true
    done
    if [[ "${any}" != true ]]; then
        info "No existing data under: ${dirs[*]} — nothing to wipe"
        return 0
    fi

    warn "This will move the following EXISTING data director$([ ${#dirs[@]} -gt 1 ] && echo ies || echo y) aside:"
    for d in "${dirs[@]}"; do [[ -d "$d" ]] && echo "    $d"; done
    warn "Preserved as <dir>.bak-<timestamp> — not permanently deleted."
    confirm "Wipe old ${service_name} data?"

    if systemctl is-active --quiet "${service_name}.service" 2>/dev/null; then
        info "Stopping ${service_name} before wiping its data..."
        systemctl stop "${service_name}.service"
    fi
    local ts; ts="$(date +%Y%m%d-%H%M%S)"
    for d in "${dirs[@]}"; do
        if [[ -d "$d" && -n "$(ls -A "$d" 2>/dev/null)" ]]; then
            mv "$d" "${d}.bak-${ts}"
            success "Moved ${d} -> ${d}.bak-${ts}"
        fi
    done
}

# ============================================================================
# Argument Parsing
# ============================================================================

validate_identifier() {
    local flag="$1" val="$2"
    if [[ -z "${val}" ]]; then
        error "${flag}: value must not be empty"
        exit 1
    fi
    if [[ ! "${val}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
        error "${flag}: '${val}' contains invalid characters (allowed: a-z A-Z 0-9 . _ -)"
        exit 1
    fi
}

validate_build_dir() {
    local flag="$1" val="$2"
    if [[ -z "${val}" ]]; then
        error "${flag}: value must not be empty"
        exit 1
    fi
    if [[ ! "${val}" =~ ^[a-zA-Z0-9._/-]+$ ]] || [[ "${val}" =~ \.\. ]]; then
        error "${flag}: '${val}' must be a relative path with no '..' (allowed: a-z A-Z 0-9 . _ - /)"
        exit 1
    fi
}

validate_abs_path() {
    local flag="$1" val="$2"
    if [[ -z "${val}" ]]; then
        error "${flag}: value must not be empty"
        exit 1
    fi
    if [[ "${val}" != /* ]]; then
        error "${flag}: '${val}' must be an absolute path (starting with /)"
        exit 1
    fi
    if [[ "${val}" =~ $'\n' || "${val}" =~ $'\0' ]]; then
        error "${flag}: path contains illegal characters"
        exit 1
    fi
}

validate_port() {
    local val="$1"
    if [[ ! "${val}" =~ ^[0-9]+$ ]] || (( val < 1 || val > 65535 )); then
        error "--port: '${val}' is not a valid port number (1-65535)"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-path)
            validate_abs_path "--install-path" "$2"
            INSTALL_PATH="$2"; shift 2 ;;
        --user)
            validate_identifier "--user" "$2"
            SERVICE_USER="$2"; shift 2 ;;
        --group)
            validate_identifier "--group" "$2"
            SERVICE_GROUP="$2"; shift 2 ;;
        --build-dir)
            validate_build_dir "--build-dir" "$2"
            BUILD_DIR="$2"; shift 2 ;;
        --port)
            validate_port "$2"
            MNS_PORT="$2"; shift 2 ;;
        --wipe-data) WIPE_DATA=true; shift ;;
        --yes)  YES=true; shift ;;
        --help) show_help; exit 0 ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Derived Paths
# ============================================================================

BIN_DIR="${INSTALL_PATH}/bin"
DATA_DIR="${INSTALL_PATH}/name_service"
LOG_DIR="/var/log/adai"

BUILD_BIN_DIR="${REPO_ROOT}/${BUILD_DIR}/bin"

SERVICE_NAME="adai-mns"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# ============================================================================
# Preflight Checks
# ============================================================================

preflight_checks() {
    info "Running preflight checks..."

    if [[ $EUID -ne 0 ]]; then
        error "Installation requires root privileges (use sudo)"
        exit 1
    fi

    if ! command -v systemctl &>/dev/null; then
        error "systemd is not available on this system (required for service installation)"
        exit 1
    fi

    if [[ ! -f "${BUILD_BIN_DIR}/mns_server" ]]; then
        error "mns_server not found at ${BUILD_BIN_DIR}/mns_server"
        error "Build with:"
        error "  cmake --preset portable && cmake --build --preset portable --target mns_server"
        exit 1
    fi

    success "Preflight checks passed"
    echo ""
}

# ============================================================================
# Install
# ============================================================================

install_mns_server() {
    local step_total=6

    info "Installation Configuration:"
    echo "  Install Path:   ${INSTALL_PATH}"
    echo "  Binary:         ${BIN_DIR}/mns_server"
    echo "  Data Directory: ${DATA_DIR}/"
    echo "  Log Directory:  ${LOG_DIR}/"
    echo "  Service User:   ${SERVICE_USER}:${SERVICE_GROUP}"
    echo "  Service File:   ${SERVICE_FILE}"
    echo "  Listen Port:    ${MNS_PORT}"
    echo "  Build Source:   ${BUILD_BIN_DIR}/mns_server"
    echo "  Wipe Old Data:  ${WIPE_DATA}"
    echo ""

    confirm "Continue with installation?"

    if [[ "${WIPE_DATA}" == true ]]; then
        wipe_old_data "${SERVICE_NAME}" "${DATA_DIR}"
    fi

    # Step 1: Create system user and group
    info "[1/${step_total}] Creating service user and group..."
    if getent group "${SERVICE_GROUP}" &>/dev/null; then
        warn "Group '${SERVICE_GROUP}' already exists, skipping creation"
    else
        groupadd -r "${SERVICE_GROUP}"
        success "Created system group '${SERVICE_GROUP}'"
    fi
    if id "${SERVICE_USER}" &>/dev/null; then
        warn "User '${SERVICE_USER}' already exists, skipping creation"
    else
        useradd -r -s /usr/sbin/nologin -g "${SERVICE_GROUP}" -d "${INSTALL_PATH}" \
            -c "ADAI Service" "${SERVICE_USER}"
        success "Created system user '${SERVICE_USER}'"
    fi

    # Step 2: Create directory structure
    info "[2/${step_total}] Creating directory structure..."
    mkdir -p "${BIN_DIR}"
    mkdir -p "${DATA_DIR}"
    mkdir -p "${LOG_DIR}"
    success "Directory structure created"

    # Step 3: Copy binary
    info "[3/${step_total}] Installing mns_server binary..."
    cp "${BUILD_BIN_DIR}/mns_server" "${BIN_DIR}/mns_server"
    chmod 755 "${BIN_DIR}/mns_server"
    success "Installed ${BIN_DIR}/mns_server"

    # Step 4: Set ownership and permissions
    info "[4/${step_total}] Setting ownership and permissions..."
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
    success "Ownership and permissions set"

    # Step 5: Write systemd unit file
    info "[5/${step_total}] Writing systemd service unit..."
    cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=ADAI Model Name Service
Documentation=https://github.com/adai/docs
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_PATH}
ExecStart=${BIN_DIR}/mns_server --port ${MNS_PORT} --data-dir ${DATA_DIR}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

# Environment (override at runtime via /etc/systemd/system/${SERVICE_NAME}.service.d/override.conf)
Environment="NAME_SERVICE_PORT=${MNS_PORT}"
Environment="NAME_SERVICE_DIR=${DATA_DIR}"

# Security hardening
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=${DATA_DIR} ${LOG_DIR}

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 "${SERVICE_FILE}"
    success "Wrote ${SERVICE_FILE}"

    # Step 6: Enable and start service
    info "[6/${step_total}] Enabling and starting ${SERVICE_NAME}..."
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service"
    success "Service '${SERVICE_NAME}' enabled (will start on boot)"

    info "Starting mns_server..."
    if systemctl start "${SERVICE_NAME}.service"; then
        success "mns_server started"
    else
        error "Failed to start mns_server"
        warn "Check logs: sudo journalctl -u ${SERVICE_NAME} -n 50"
        exit 1
    fi

    print_summary
}

print_summary() {
    echo ""
    echo "========================================================================"
    success "ADAI Model Name Service installed!"
    echo "========================================================================"
    echo ""
    echo "Installed:"
    echo "  Binary:   ${BIN_DIR}/mns_server"
    echo "  Data:     ${DATA_DIR}/"
    echo "  Service:  ${SERVICE_FILE}"
    echo ""
    echo "Service management:"
    echo "  Status:  systemctl status ${SERVICE_NAME}"
    echo "  Logs:    journalctl -u ${SERVICE_NAME} -f"
    echo "  Stop:    systemctl stop ${SERVICE_NAME}"
    echo "  Restart: systemctl restart ${SERVICE_NAME}"
    echo ""
    echo "API endpoints (port ${MNS_PORT}):"
    echo "  GET  http://localhost:${MNS_PORT}/health"
    echo "  GET  http://localhost:${MNS_PORT}/models"
    echo "  GET  http://localhost:${MNS_PORT}/roles"
    echo ""
    echo "Configure clients with:"
    echo "  MNS_SERVER_URL=http://$(hostname -f 2>/dev/null || hostname):${MNS_PORT}"
    echo ""
    echo "Override environment variables:"
    echo "  sudo systemctl edit ${SERVICE_NAME}"
    echo "  (set NAME_SERVICE_PORT, NAME_SERVICE_DIR, MODEL_NAME, MODEL_ROLE)"
    echo ""
    echo "========================================================================"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

echo ""
info "ADAI Model Name Service — Installation Script"
echo ""

preflight_checks
install_mns_server
