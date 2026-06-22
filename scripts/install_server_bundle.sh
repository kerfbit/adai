#!/bin/bash
# ADAI Server Bundle - Installation Script
#
# Installs metrics_api_server, registry_server, and mns_server as a co-located
# set of systemd services on a single machine.  All three services communicate
# via localhost.
#
# Usage:
#   sudo ./install_server_bundle.sh [OPTIONS]
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
REGISTRY_PORT=8082
METRICS_PORT=8081
METRICS_DIR=""
REGISTRY_DATA_DIR=""
MNS_DATA_DIR=""
YES=false

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
ADAI Server Bundle - Installation Script

Usage: sudo $0 [OPTIONS]

Options:
  --install-path PATH       Installation root directory (default: /opt/adai)
  --user USER               Service user to own installed files (default: adai)
  --group GROUP             Service group (default: adai)
  --build-dir DIR           CMake build directory containing bin/ (default: build/portable)
  --mns-port PORT           Model Name Service port (default: 8083)
  --registry-port PORT      Registry Server port (default: 8082)
  --metrics-port PORT       Metrics API Server port (default: 8081)
  --metrics-dir DIR         Metrics/sessions directory (default: <install-path>/training_sessions)
  --registry-data-dir DIR   Registry data directory (default: <install-path>/registry_sessions)
  --mns-data-dir DIR        MNS data directory (default: <install-path>/name_service)
  --yes                     Skip confirmation prompts (for non-interactive use)
  --help                    Show this help message

Description:
  Installs the following services as a co-located bundle:

    1. mns_server        — Model Name Service (identity registry)
    2. registry_server   — Dataset queue coordination
    3. metrics_api_server — Training metrics REST API

  All three services run on localhost and are configured to discover each other
  locally.  The metrics_api_server queries the MNS on localhost:${MNS_PORT} for
  registered model names exposed at GET /api/models.

  Service names:
    adai-mns          (port ${MNS_PORT})
    adai-registry     (port ${REGISTRY_PORT})
    adai-metrics      (port ${METRICS_PORT})

  Build first:
    cmake --preset portable && cmake --build --preset portable

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
        --mns-port)
            validate_port "$2"
            MNS_PORT="$2"; shift 2 ;;
        --registry-port)
            validate_port "$2"
            REGISTRY_PORT="$2"; shift 2 ;;
        --metrics-port)
            validate_port "$2"
            METRICS_PORT="$2"; shift 2 ;;
        --metrics-dir)
            validate_abs_path "--metrics-dir" "$2"
            METRICS_DIR="$2"; shift 2 ;;
        --registry-data-dir)
            validate_abs_path "--registry-data-dir" "$2"
            REGISTRY_DATA_DIR="$2"; shift 2 ;;
        --mns-data-dir)
            validate_abs_path "--mns-data-dir" "$2"
            MNS_DATA_DIR="$2"; shift 2 ;;
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
LOG_DIR="/var/log/adai"
CONF_DIR="${INSTALL_PATH}/etc"

[[ -z "${METRICS_DIR}" ]]       && METRICS_DIR="${INSTALL_PATH}/training_sessions"
[[ -z "${REGISTRY_DATA_DIR}" ]] && REGISTRY_DATA_DIR="${INSTALL_PATH}/registry_sessions"
[[ -z "${MNS_DATA_DIR}" ]]      && MNS_DATA_DIR="${INSTALL_PATH}/name_service"

BUILD_BIN_DIR="${REPO_ROOT}/${BUILD_DIR}/bin"

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

    local missing=()
    for bin in mns_server registry_server metrics_api_server; do
        if [[ ! -f "${BUILD_BIN_DIR}/${bin}" ]]; then
            missing+=("${bin}")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing binaries in ${BUILD_BIN_DIR}:"
        for m in "${missing[@]}"; do
            error "  - ${m}"
        done
        error ""
        error "Build with:"
        error "  cmake --preset portable && cmake --build --preset portable"
        exit 1
    fi

    success "Preflight checks passed"
    echo ""
}

# ============================================================================
# Install
# ============================================================================

install_bundle() {
    local step_total=8

    info "Installation Configuration:"
    echo "  Install Path:         ${INSTALL_PATH}"
    echo "  Binary Directory:     ${BIN_DIR}"
    echo "  Config Directory:     ${CONF_DIR}"
    echo "  Log Directory:        ${LOG_DIR}"
    echo "  Service User:         ${SERVICE_USER}:${SERVICE_GROUP}"
    echo ""
    echo "  MNS Server:"
    echo "    Port:               ${MNS_PORT}"
    echo "    Data Directory:     ${MNS_DATA_DIR}"
    echo ""
    echo "  Registry Server:"
    echo "    Port:               ${REGISTRY_PORT}"
    echo "    Data Directory:     ${REGISTRY_DATA_DIR}"
    echo ""
    echo "  Metrics API Server:"
    echo "    Port:               ${METRICS_PORT}"
    echo "    Metrics Directory:  ${METRICS_DIR}"
    echo "    Name Service URL:   http://localhost:${MNS_PORT}"
    echo ""
    echo "  Build Source:         ${BUILD_BIN_DIR}/"
    echo ""

    confirm "Continue with installation?"

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
    mkdir -p "${CONF_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${MNS_DATA_DIR}"
    mkdir -p "${REGISTRY_DATA_DIR}"
    mkdir -p "${METRICS_DIR}"
    success "Directory structure created"

    # Step 3: Copy binaries
    info "[3/${step_total}] Installing binaries..."
    for bin in mns_server registry_server metrics_api_server dataset_manager mns_cli; do
        if [[ -f "${BUILD_BIN_DIR}/${bin}" ]]; then
            cp "${BUILD_BIN_DIR}/${bin}" "${BIN_DIR}/${bin}"
            chmod 755 "${BIN_DIR}/${bin}"
            success "  Installed ${bin}"
        fi
    done

    # Step 4: Write config file
    info "[4/${step_total}] Writing configuration..."
    cat > "${CONF_DIR}/config.conf" <<EOF
# ADAI Server Bundle Configuration
# Generated by install_server_bundle.sh

# Model Name Service
NAME_SERVICE_URL=http://localhost:${MNS_PORT}
NAME_SERVICE_PORT=${MNS_PORT}
NAME_SERVICE_DIR=${MNS_DATA_DIR}

# Registry Server
REGISTRY_SERVER_URL=http://localhost:${REGISTRY_PORT}

# Metrics API Server
METRICS_SERVER_URL=http://localhost:${METRICS_PORT}
METRICS_DIR=${METRICS_DIR}

# Session directory
SESSION_DIR=${METRICS_DIR}
EOF
    chmod 644 "${CONF_DIR}/config.conf"
    success "Wrote ${CONF_DIR}/config.conf"

    # Step 5: Set ownership and permissions
    info "[5/${step_total}] Setting ownership and permissions..."
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
    success "Ownership and permissions set"

    # Step 6: Write systemd unit files
    info "[6/${step_total}] Writing systemd service units..."

    # --- adai-mns ---
    cat > "/etc/systemd/system/adai-mns.service" <<EOF
[Unit]
Description=ADAI Model Name Service
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_PATH}
ExecStart=${BIN_DIR}/mns_server --port ${MNS_PORT} --data-dir ${MNS_DATA_DIR}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

Environment="NAME_SERVICE_PORT=${MNS_PORT}"
Environment="NAME_SERVICE_DIR=${MNS_DATA_DIR}"

PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=${MNS_DATA_DIR} ${LOG_DIR}

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 /etc/systemd/system/adai-mns.service
    success "  Wrote adai-mns.service"

    # --- adai-registry ---
    cat > "/etc/systemd/system/adai-registry.service" <<EOF
[Unit]
Description=ADAI Dataset Registry Server
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_PATH}
ExecStart=${BIN_DIR}/registry_server --port ${REGISTRY_PORT} --data-dir ${REGISTRY_DATA_DIR}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

Environment="REGISTRY_PORT=${REGISTRY_PORT}"

PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=${REGISTRY_DATA_DIR} ${LOG_DIR}

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 /etc/systemd/system/adai-registry.service
    success "  Wrote adai-registry.service"

    # --- adai-metrics ---
    cat > "/etc/systemd/system/adai-metrics.service" <<EOF
[Unit]
Description=ADAI Training Metrics API Server
After=network.target adai-mns.service
Wants=adai-mns.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_PATH}
ExecStart=${BIN_DIR}/metrics_api_server --port ${METRICS_PORT} --name-service-url http://localhost:${MNS_PORT}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

Environment="METRICS_PORT=${METRICS_PORT}"
Environment="METRICS_DIR=${METRICS_DIR}"
Environment="NAME_SERVICE_URL=http://localhost:${MNS_PORT}"

PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=${METRICS_DIR} ${LOG_DIR}

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 /etc/systemd/system/adai-metrics.service
    success "  Wrote adai-metrics.service"

    # Step 7: Enable and start services
    info "[7/${step_total}] Enabling and starting services..."
    systemctl daemon-reload

    for svc in adai-mns adai-registry adai-metrics; do
        systemctl enable "${svc}.service"
        success "  Enabled ${svc}"
    done

    info "Starting services (mns first, then registry and metrics)..."
    if systemctl start adai-mns.service; then
        success "  adai-mns started"
    else
        error "  Failed to start adai-mns"
        warn "  Check logs: sudo journalctl -u adai-mns -n 50"
    fi

    if systemctl start adai-registry.service; then
        success "  adai-registry started"
    else
        error "  Failed to start adai-registry"
        warn "  Check logs: sudo journalctl -u adai-registry -n 50"
    fi

    if systemctl start adai-metrics.service; then
        success "  adai-metrics started"
    else
        error "  Failed to start adai-metrics"
        warn "  Check logs: sudo journalctl -u adai-metrics -n 50"
    fi

    # Step 8: Verify
    info "[8/${step_total}] Verifying installation..."
    sleep 2

    local all_ok=true
    for svc in adai-mns adai-registry adai-metrics; do
        if systemctl is-active --quiet "${svc}.service"; then
            success "  ${svc} is running"
        else
            warn "  ${svc} is not running"
            all_ok=false
        fi
    done

    if command -v ss &>/dev/null; then
        for p in "${MNS_PORT}" "${REGISTRY_PORT}" "${METRICS_PORT}"; do
            if ss -tlnp | grep -q ":${p}"; then
                success "  Port ${p} is listening"
            else
                warn "  Port ${p} not yet listening"
            fi
        done
    fi

    print_summary
}

print_summary() {
    echo ""
    echo "========================================================================"
    success "ADAI Server Bundle installed!"
    echo "========================================================================"
    echo ""
    echo "Services:"
    echo "  adai-mns       — Model Name Service       (port ${MNS_PORT})"
    echo "  adai-registry  — Dataset Registry Server   (port ${REGISTRY_PORT})"
    echo "  adai-metrics   — Training Metrics API      (port ${METRICS_PORT})"
    echo ""
    echo "Binaries:  ${BIN_DIR}/"
    echo "Config:    ${CONF_DIR}/config.conf"
    echo "Logs:      journalctl -u adai-mns / adai-registry / adai-metrics"
    echo ""
    echo "Service management:"
    echo "  systemctl status  adai-mns adai-registry adai-metrics"
    echo "  systemctl restart adai-mns adai-registry adai-metrics"
    echo "  systemctl stop    adai-mns adai-registry adai-metrics"
    echo ""
    echo "Health checks:"
    echo "  curl http://localhost:${MNS_PORT}/health"
    echo "  curl http://localhost:${REGISTRY_PORT}/health"
    echo "  curl http://localhost:${METRICS_PORT}/health"
    echo ""
    echo "Name service integration:"
    echo "  curl http://localhost:${METRICS_PORT}/api/models"
    echo ""
    echo "Configure trainers with:"
    echo "  NAME_SERVICE_URL=http://$(hostname -f 2>/dev/null || hostname):${MNS_PORT}"
    echo "  REGISTRY_SERVER_URL=http://$(hostname -f 2>/dev/null || hostname):${REGISTRY_PORT}"
    echo "  METRICS_SERVER_URL=http://$(hostname -f 2>/dev/null || hostname):${METRICS_PORT}"
    echo ""
    echo "========================================================================"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

echo ""
info "ADAI Server Bundle — Installation Script"
echo ""

preflight_checks
install_bundle
