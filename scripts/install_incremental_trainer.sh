#!/bin/bash
# ADAI Incremental Trainer Sub-System - Installation Script
#
# Installs incremental_trainer, dataset_manager, and optionally registry_server
# to a target host. Supports local, remote (SSH+rsync), and coordinator-only modes.
#
# Usage:
#   sudo ./install_incremental_trainer.sh [OPTIONS]
#
# See --help for the full option list.

set -euo pipefail

# ============================================================================
# Configuration Defaults
# ============================================================================

INSTALL_PATH="/opt/adai"
SERVICE_USER="adai"
SERVICE_GROUP="adai"
BUILD_DIR="build-gpu-clang"
CONFIG_SRC=""       # auto-resolved to <repo-root>/config.conf when empty
VOCAB_SRC=""        # auto-resolved to <repo-root>/vocab.txt when empty
WITH_REGISTRY_SERVER=false
COORDINATOR=false
REMOTE_HOST=""
SYNC_SESSIONS=false
SSH_KEY=""

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
ADAI Incremental Trainer Sub-System - Installation Script

Usage: sudo $0 [OPTIONS]

Options:
  --install-path PATH       Installation root directory (default: /opt/adai)
  --user USER               Service user to own installed files (default: adai)
  --group GROUP             Service group (default: adai)
  --build-dir DIR           CMake build directory containing bin/ (default: build-gpu-clang)
  --config-src PATH         Source config.conf (default: <repo-root>/config.conf)
  --vocab-src PATH          Source vocab.txt (default: <repo-root>/vocab.txt)
  --with-registry-server    Also install the registry_server binary
  --coordinator             Install only registry_server as a coordinator node
                            (implies --with-registry-server; no trainer binary required)
  --remote HOST             Install to a remote host via SSH + rsync
  --sync-sessions           (with --remote) Also rsync training_sessions/ to the remote host
  --ssh-key PATH            SSH identity file forwarded to all ssh/rsync calls
  --help                    Show this help message

Examples:
  # Default local installation
  sudo $0

  # Local install with distributed registry server
  sudo $0 --with-registry-server

  # Custom build directory and install path
  sudo $0 --build-dir build-release --install-path /usr/local/adai

  # Coordinator-only node (runs registry_server + systemd unit, no trainer)
  sudo $0 --coordinator --install-path /opt/adai-coord

  # Remote install to another host
  sudo $0 --remote user@192.168.1.7

  # Remote install with custom SSH key and session checkpoint sync
  sudo $0 --remote user@192.168.1.7 --ssh-key ~/.ssh/id_adai --sync-sessions

Description:
  Deploys the incremental_trainer sub-system to a local or remote host.
  Creates the required directory layout, copies binaries and support files,
  sets permissions and ownership, and appends distributed-registry config stubs.

  Binaries:
    incremental_trainer   training loop and session manager
    dataset_manager       stateless data-prep CLI (Gutenberg / HuggingFace)
    registry_server       optional HTTP daemon (port 8082) for distributed pools

  Build first:
    cmake -B <build-dir> [-DBUILD_METRICS_API_SERVER=ON] && cmake --build <build-dir>

EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-path)       INSTALL_PATH="$2";  shift 2 ;;
        --user)               SERVICE_USER="$2";   shift 2 ;;
        --group)              SERVICE_GROUP="$2";  shift 2 ;;
        --build-dir)          BUILD_DIR="$2";      shift 2 ;;
        --config-src)         CONFIG_SRC="$2";     shift 2 ;;
        --vocab-src)          VOCAB_SRC="$2";      shift 2 ;;
        --with-registry-server) WITH_REGISTRY_SERVER=true; shift ;;
        --coordinator)        COORDINATOR=true; WITH_REGISTRY_SERVER=true; shift ;;
        --remote)             REMOTE_HOST="$2";    shift 2 ;;
        --sync-sessions)      SYNC_SESSIONS=true;  shift ;;
        --ssh-key)            SSH_KEY="$2";        shift 2 ;;
        --help)               show_help; exit 0 ;;
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
CONFIG_DIR="${INSTALL_PATH}/config"
SESSIONS_DIR="${INSTALL_PATH}/training_sessions"
DATA_DIR="${INSTALL_PATH}/training_data"
GUTENBERG_DIR="${DATA_DIR}/gutenberg_data"
HUGGINGFACE_DIR="${DATA_DIR}/huggingface_data"
LOG_DIR="/var/log/adai"

BUILD_BIN_DIR="${REPO_ROOT}/${BUILD_DIR}/bin"

[[ -z "${CONFIG_SRC}" ]] && CONFIG_SRC="${REPO_ROOT}/config.conf"
[[ -z "${VOCAB_SRC}" ]]  && VOCAB_SRC="${REPO_ROOT}/vocab.txt"

# Build SSH option arrays for ssh/rsync calls
SSH_ARGS=()
RSYNC_SSH_CMD="ssh"
if [[ -n "${SSH_KEY}" ]]; then
    SSH_ARGS=("-i" "${SSH_KEY}")
    RSYNC_SSH_CMD="ssh -i ${SSH_KEY}"
fi

# ============================================================================
# Preflight Checks
# ============================================================================

preflight_checks() {
    info "Running preflight checks..."

    if [[ -z "${REMOTE_HOST}" ]] && [[ $EUID -ne 0 ]]; then
        error "Local installation requires root privileges (use sudo)"
        exit 1
    fi

    if [[ "${COORDINATOR}" == true ]]; then
        if [[ ! -f "${BUILD_BIN_DIR}/registry_server" ]]; then
            error "registry_server not found at ${BUILD_BIN_DIR}/registry_server"
            error "Build with: cmake -B ${BUILD_DIR} -DBUILD_METRICS_API_SERVER=ON && cmake --build ${BUILD_DIR} --target registry_server"
            exit 1
        fi
    else
        if [[ ! -f "${BUILD_BIN_DIR}/incremental_trainer" ]]; then
            error "incremental_trainer not found at ${BUILD_BIN_DIR}/incremental_trainer"
            error "Build with: cmake -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} --target incremental_trainer"
            exit 1
        fi
        if [[ ! -f "${BUILD_BIN_DIR}/dataset_manager" ]]; then
            error "dataset_manager not found at ${BUILD_BIN_DIR}/dataset_manager"
            error "Build with: cmake -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} --target dataset_manager"
            exit 1
        fi
        if [[ "${WITH_REGISTRY_SERVER}" == true ]] && [[ ! -f "${BUILD_BIN_DIR}/registry_server" ]]; then
            error "registry_server not found at ${BUILD_BIN_DIR}/registry_server"
            error "Build with: cmake -B ${BUILD_DIR} -DBUILD_METRICS_API_SERVER=ON && cmake --build ${BUILD_DIR} --target registry_server"
            exit 1
        fi
    fi

    if [[ ! -f "${VOCAB_SRC}" ]]; then
        error "vocab.txt not found at ${VOCAB_SRC}"
        error "Specify an alternate path with --vocab-src"
        exit 1
    fi

    if [[ ! -f "${CONFIG_SRC}" ]]; then
        error "config.conf not found at ${CONFIG_SRC}"
        error "Specify an alternate path with --config-src"
        exit 1
    fi

    success "Preflight checks passed"
    echo ""
}

# ============================================================================
# Append Distributed-Registry Config Stubs (idempotent)
# ============================================================================

append_registry_stubs() {
    local config_path="$1"
    if grep -q "REGISTRY_SERVER_URL" "${config_path}" 2>/dev/null; then
        warn "Distributed-registry stubs already present in ${config_path}, skipping"
        return
    fi
    cat >> "${config_path}" <<'STUBS'

# ============================================================================
# Distributed Registry (TD-028 Phase 9)
# Set REGISTRY_SERVER_URL to enable multi-node distributed training pools.
# Leave all keys commented out for standalone single-node operation.
# ============================================================================

# URL of the registry_server HTTP daemon (port 8082)
# REGISTRY_SERVER_URL=http://<coordinator>:8082

# Training run group name (partitions work across nodes in the same pool)
# RUN_GROUP=my-training-group

# Per-node run ID (empty = auto-derived from hostname+PID at startup)
# RUN_ID=

# HTTP request timeout for registry operations (milliseconds, default: 5000)
# REGISTRY_TIMEOUT_MS=5000
STUBS
    success "Appended distributed-registry config stubs to ${config_path}"
}

# ============================================================================
# LOCAL INSTALL
# ============================================================================

local_install() {
    local step_total=7

    info "Installation Configuration:"
    echo "  Install Path:      ${INSTALL_PATH}"
    echo "  Binaries:          ${BIN_DIR}/"
    echo "  Config:            ${CONFIG_DIR}/config.conf"
    echo "  Sessions:          ${SESSIONS_DIR}/"
    echo "  Training Data:     ${DATA_DIR}/"
    echo "  Log Directory:     ${LOG_DIR}/"
    echo "  Service User:      ${SERVICE_USER}:${SERVICE_GROUP}"
    echo "  Build Directory:   ${BUILD_BIN_DIR}/"
    [[ "${WITH_REGISTRY_SERVER}" == true ]] && echo "  Registry Server:   yes (port 8082)"
    echo ""

    read -r -p "Continue with installation? (y/N) " -n 1
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Installation cancelled by user"
        exit 0
    fi

    # Step 1: Create system user
    info "[1/${step_total}] Creating service user and group..."
    if id "${SERVICE_USER}" &>/dev/null; then
        warn "User '${SERVICE_USER}' already exists, skipping creation"
    else
        useradd -r -s /usr/sbin/nologin -d "${INSTALL_PATH}" -c "ADAI Service" "${SERVICE_USER}"
        success "Created system user '${SERVICE_USER}'"
    fi

    # Step 2: Create directory structure
    info "[2/${step_total}] Creating directory structure..."
    mkdir -p "${BIN_DIR}"
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${SESSIONS_DIR}"
    mkdir -p "${GUTENBERG_DIR}"
    mkdir -p "${HUGGINGFACE_DIR}"
    mkdir -p "${LOG_DIR}"
    success "Directory structure created"

    # Step 3: Copy binaries
    info "[3/${step_total}] Copying binaries..."
    cp "${BUILD_BIN_DIR}/incremental_trainer" "${BIN_DIR}/incremental_trainer"
    chmod 755 "${BIN_DIR}/incremental_trainer"
    success "Installed incremental_trainer"

    cp "${BUILD_BIN_DIR}/dataset_manager" "${BIN_DIR}/dataset_manager"
    chmod 755 "${BIN_DIR}/dataset_manager"
    success "Installed dataset_manager"

    if [[ "${WITH_REGISTRY_SERVER}" == true ]]; then
        cp "${BUILD_BIN_DIR}/registry_server" "${BIN_DIR}/registry_server"
        chmod 755 "${BIN_DIR}/registry_server"
        success "Installed registry_server"
    fi

    # Step 4: Copy config and vocab
    info "[4/${step_total}] Copying config and vocab..."
    cp "${VOCAB_SRC}" "${CONFIG_DIR}/vocab.txt"
    chmod 644 "${CONFIG_DIR}/vocab.txt"
    success "Installed vocab.txt"

    cp "${CONFIG_SRC}" "${CONFIG_DIR}/config.conf"
    chmod 640 "${CONFIG_DIR}/config.conf"
    success "Installed config.conf"

    # Step 5: Append distributed-registry config stubs
    info "[5/${step_total}] Adding distributed-registry config stubs..."
    append_registry_stubs "${CONFIG_DIR}/config.conf"

    # Step 6: Set ownership
    info "[6/${step_total}] Setting ownership and permissions..."
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
    chmod 640 "${CONFIG_DIR}/config.conf"
    success "Ownership and permissions set"

    # Step 7: Post-install verification
    info "[7/${step_total}] Verifying installation..."
    local ok=true

    if [[ -x "${BIN_DIR}/incremental_trainer" ]]; then
        # 'status' may exit non-zero when no session is running; that's acceptable
        "${BIN_DIR}/incremental_trainer" status &>/dev/null || true
        success "incremental_trainer is installed and executable"
    else
        warn "incremental_trainer is not executable at ${BIN_DIR}/incremental_trainer"
        ok=false
    fi

    if "${BIN_DIR}/dataset_manager" --help &>/dev/null; then
        success "dataset_manager --help returned exit code 0"
    else
        warn "dataset_manager --help returned a non-zero exit code"
        ok=false
    fi

    if [[ "${ok}" == false ]]; then
        warn "Some post-install checks did not pass — review the output above"
    fi

    print_local_summary
}

print_local_summary() {
    echo ""
    echo "========================================================================"
    success "ADAI incremental_trainer sub-system installed!"
    echo "========================================================================"
    echo ""
    echo "Installed paths:"
    echo "  Binaries:     ${BIN_DIR}/"
    echo "  Config:       ${CONFIG_DIR}/config.conf"
    echo "  Vocab:        ${CONFIG_DIR}/vocab.txt"
    echo "  Sessions:     ${SESSIONS_DIR}/"
    echo "  Training data:${DATA_DIR}/"
    echo "  Logs:         ${LOG_DIR}/"
    echo ""
    echo "Start training:"
    echo "  sudo -u ${SERVICE_USER} ${BIN_DIR}/incremental_trainer \\"
    echo "    --config ${CONFIG_DIR}/config.conf"
    echo ""
    echo "Manage datasets:"
    echo "  sudo -u ${SERVICE_USER} ${BIN_DIR}/dataset_manager --help"
    echo ""
    if [[ "${WITH_REGISTRY_SERVER}" == true ]]; then
        echo "Start registry server:"
        echo "  sudo -u ${SERVICE_USER} ${BIN_DIR}/registry_server"
        echo "  (listens on port 8082; configure REGISTRY_SERVER_URL on worker nodes)"
        echo ""
    fi
    echo "Edit config:"
    echo "  sudo \${EDITOR:-nano} ${CONFIG_DIR}/config.conf"
    echo ""
    echo "========================================================================"
    echo ""
}

# ============================================================================
# COORDINATOR INSTALL
# ============================================================================

coordinator_install() {
    local step_total=6
    local service_name="adai-registry"
    local service_file="/etc/systemd/system/${service_name}.service"

    if ! command -v systemctl &>/dev/null; then
        error "systemd is not available on this system (required for --coordinator)"
        exit 1
    fi

    info "Coordinator Installation Configuration:"
    echo "  Install Path:   ${INSTALL_PATH}"
    echo "  Binary:         ${BIN_DIR}/registry_server"
    echo "  Config:         ${CONFIG_DIR}/config.conf"
    echo "  Log Directory:  ${LOG_DIR}/"
    echo "  Service User:   ${SERVICE_USER}:${SERVICE_GROUP}"
    echo "  Service File:   ${service_file}"
    echo "  Listen Port:    8082"
    echo ""

    read -r -p "Continue with coordinator installation? (y/N) " -n 1
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Installation cancelled by user"
        exit 0
    fi

    # Step 1: Create system user
    info "[1/${step_total}] Creating service user and group..."
    if id "${SERVICE_USER}" &>/dev/null; then
        warn "User '${SERVICE_USER}' already exists, skipping creation"
    else
        useradd -r -s /usr/sbin/nologin -d "${INSTALL_PATH}" -c "ADAI Registry Service" "${SERVICE_USER}"
        success "Created system user '${SERVICE_USER}'"
    fi

    # Step 2: Create directories
    info "[2/${step_total}] Creating directory structure..."
    mkdir -p "${BIN_DIR}"
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${LOG_DIR}"
    success "Directories created"

    # Step 3: Copy registry_server binary
    info "[3/${step_total}] Copying registry_server binary..."
    cp "${BUILD_BIN_DIR}/registry_server" "${BIN_DIR}/registry_server"
    chmod 755 "${BIN_DIR}/registry_server"
    success "Installed registry_server"

    # Step 4: Write coordinator config stub
    info "[4/${step_total}] Writing coordinator config..."
    local coordinator_hostname
    coordinator_hostname="$(hostname -f 2>/dev/null || hostname)"
    cat > "${CONFIG_DIR}/config.conf" <<EOF
# ADAI Registry Server — Coordinator Node Configuration
# Generated by install_incremental_trainer.sh --coordinator on $(date)
#
# This host runs registry_server only (no trainer binary required).
# Worker nodes must add to their config.conf:
#   REGISTRY_SERVER_URL=http://${coordinator_hostname}:8082

# ============================================================================
# Distributed Registry (TD-028 Phase 9)
# ============================================================================

# URL workers use to reach this coordinator (update after finalising networking)
# REGISTRY_SERVER_URL=http://${coordinator_hostname}:8082

# Training run group name shared across all workers in the pool
# RUN_GROUP=my-training-group

# Per-node run ID (empty = auto-derived from hostname+PID at startup)
# RUN_ID=

# HTTP request timeout for registry operations (milliseconds, default: 5000)
# REGISTRY_TIMEOUT_MS=5000
EOF
    chmod 640 "${CONFIG_DIR}/config.conf"
    success "Wrote coordinator config.conf"

    # Step 5: Set ownership
    info "[5/${step_total}] Setting ownership and permissions..."
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
    chmod 640 "${CONFIG_DIR}/config.conf"
    success "Ownership and permissions set"

    # Step 6: Install, enable, and start systemd service
    info "[6/${step_total}] Installing systemd service unit..."
    cat > "${service_file}" <<EOF
[Unit]
Description=ADAI Dataset Registry Server
Documentation=https://github.com/adai/docs
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_PATH}
ExecStart=${BIN_DIR}/registry_server
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

# Security hardening
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${LOG_DIR}

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 "${service_file}"
    success "Installed ${service_file}"

    systemctl daemon-reload
    systemctl enable "${service_name}.service"
    success "Service '${service_name}' enabled (will start on boot)"

    info "Starting registry_server..."
    if systemctl start "${service_name}.service"; then
        success "Registry server started"
    else
        error "Failed to start registry server"
        warn "Check logs: sudo journalctl -u ${service_name} -n 50"
        exit 1
    fi

    echo ""
    echo "========================================================================"
    success "ADAI registry_server coordinator installed!"
    echo "========================================================================"
    echo ""
    echo "Installed:"
    echo "  Binary:   ${BIN_DIR}/registry_server"
    echo "  Config:   ${CONFIG_DIR}/config.conf"
    echo "  Service:  ${service_file}"
    echo ""
    echo "Service management:"
    echo "  Status:  systemctl status ${service_name}"
    echo "  Logs:    journalctl -u ${service_name} -f"
    echo "  Stop:    systemctl stop ${service_name}"
    echo "  Restart: systemctl restart ${service_name}"
    echo ""
    echo "On each worker node add to config.conf:"
    echo "  REGISTRY_SERVER_URL=http://${coordinator_hostname}:8082"
    echo "  RUN_GROUP=my-training-group"
    echo ""
    echo "========================================================================"
    echo ""
}

# ============================================================================
# REMOTE INSTALL
# ============================================================================

remote_install() {
    local remote_bin="${INSTALL_PATH}/bin"
    local remote_config="${INSTALL_PATH}/config"
    local remote_sessions="${INSTALL_PATH}/training_sessions"
    local remote_data="${INSTALL_PATH}/training_data"

    local step_total=5
    [[ "${SYNC_SESSIONS}" == true ]] && step_total=6

    info "Remote Installation Configuration:"
    echo "  Remote Host:    ${REMOTE_HOST}"
    echo "  Install Path:   ${INSTALL_PATH}"
    echo "  Build Dir:      ${BUILD_BIN_DIR}/"
    [[ -n "${SSH_KEY}" ]]              && echo "  SSH Key:        ${SSH_KEY}"
    [[ "${WITH_REGISTRY_SERVER}" == true ]] && echo "  Registry Server: yes"
    [[ "${SYNC_SESSIONS}" == true ]]   && echo "  Sync Sessions:  yes"
    echo ""

    read -r -p "Continue with remote installation to ${REMOTE_HOST}? (y/N) " -n 1
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        warn "Installation cancelled by user"
        exit 0
    fi

    # Step 1: Create remote directory structure via SSH
    info "[1/${step_total}] Creating remote directory structure on ${REMOTE_HOST}..."
    ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" bash -s <<REMOTE_MKDIR
set -euo pipefail
mkdir -p "${remote_bin}"
mkdir -p "${remote_config}"
mkdir -p "${remote_sessions}"
mkdir -p "${remote_data}/gutenberg_data"
mkdir -p "${remote_data}/huggingface_data"
mkdir -p "/var/log/adai"
REMOTE_MKDIR
    success "Remote directory structure created"

    # Step 2: Rsync binaries
    info "[2/${step_total}] Transferring binaries to ${REMOTE_HOST}..."
    rsync -az --progress \
        -e "${RSYNC_SSH_CMD}" \
        "${BUILD_BIN_DIR}/incremental_trainer" \
        "${BUILD_BIN_DIR}/dataset_manager" \
        "${REMOTE_HOST}:${remote_bin}/"
    if [[ "${WITH_REGISTRY_SERVER}" == true ]]; then
        rsync -az --progress \
            -e "${RSYNC_SSH_CMD}" \
            "${BUILD_BIN_DIR}/registry_server" \
            "${REMOTE_HOST}:${remote_bin}/"
    fi
    success "Binaries transferred"

    # Step 3: Set remote binary permissions
    info "[3/${step_total}] Setting remote binary permissions..."
    local chmod_cmd="chmod 755 '${remote_bin}/incremental_trainer' '${remote_bin}/dataset_manager'"
    [[ "${WITH_REGISTRY_SERVER}" == true ]] && chmod_cmd+=" '${remote_bin}/registry_server'"
    ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" bash -c "${chmod_cmd}"
    success "Remote permissions set"

    # Step 4: Rsync config and vocab
    info "[4/${step_total}] Transferring config and vocab to ${REMOTE_HOST}..."
    rsync -az --progress \
        -e "${RSYNC_SSH_CMD}" \
        "${CONFIG_SRC}" \
        "${REMOTE_HOST}:${remote_config}/config.conf"
    rsync -az --progress \
        -e "${RSYNC_SSH_CMD}" \
        "${VOCAB_SRC}" \
        "${REMOTE_HOST}:${remote_config}/vocab.txt"
    success "Config and vocab transferred"

    # Step 5: Append distributed-registry stubs on remote
    info "[5/${step_total}] Adding distributed-registry config stubs on ${REMOTE_HOST}..."
    ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" bash -s -- "${remote_config}/config.conf" <<'REMOTE_STUBS'
config_path="$1"
if grep -q "REGISTRY_SERVER_URL" "${config_path}" 2>/dev/null; then
    echo "Distributed-registry stubs already present, skipping"
else
    cat >> "${config_path}" <<'STUBS'

# ============================================================================
# Distributed Registry (TD-028 Phase 9)
# Set REGISTRY_SERVER_URL to enable multi-node distributed training pools.
# ============================================================================

# REGISTRY_SERVER_URL=http://<coordinator>:8082
# RUN_GROUP=my-training-group
# RUN_ID=
# REGISTRY_TIMEOUT_MS=5000
STUBS
    echo "Distributed-registry stubs appended"
fi
REMOTE_STUBS
    success "Registry config stubs added"

    # Step 6 (optional): Sync training_sessions/
    if [[ "${SYNC_SESSIONS}" == true ]]; then
        info "[6/${step_total}] Syncing training_sessions/ to ${REMOTE_HOST}..."
        local local_sessions="${REPO_ROOT}/training_sessions"
        if [[ -d "${local_sessions}" ]]; then
            rsync -az --progress \
                -e "${RSYNC_SSH_CMD}" \
                "${local_sessions}/" \
                "${REMOTE_HOST}:${remote_sessions}/"
            success "training_sessions/ synced"
        else
            warn "Local training_sessions/ not found at ${local_sessions}, skipping"
        fi
    fi

    # Post-install verification over SSH
    info "Verifying remote installation..."
    local remote_ok=true

    if ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
        "${remote_bin}/incremental_trainer status" &>/dev/null || \
       ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
        "test -x '${remote_bin}/incremental_trainer'"; then
        success "incremental_trainer is present and executable on ${REMOTE_HOST}"
    else
        warn "Could not verify incremental_trainer on ${REMOTE_HOST}"
        remote_ok=false
    fi

    if ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
        "'${remote_bin}/dataset_manager' --help" &>/dev/null; then
        success "dataset_manager --help succeeded on ${REMOTE_HOST}"
    else
        warn "dataset_manager --help did not return 0 on ${REMOTE_HOST}"
        remote_ok=false
    fi

    [[ "${remote_ok}" == false ]] && warn "Some remote verification checks did not pass"

    local ssh_cmd="ssh ${REMOTE_HOST}"
    [[ -n "${SSH_KEY}" ]] && ssh_cmd="ssh -i ${SSH_KEY} ${REMOTE_HOST}"

    echo ""
    echo "========================================================================"
    success "ADAI incremental_trainer installed on ${REMOTE_HOST}!"
    echo "========================================================================"
    echo ""
    echo "Installed on ${REMOTE_HOST}:"
    echo "  Binaries:      ${remote_bin}/"
    echo "  Config:        ${remote_config}/config.conf"
    echo "  Vocab:         ${remote_config}/vocab.txt"
    echo "  Sessions:      ${remote_sessions}/"
    echo "  Training data: ${remote_data}/"
    echo ""
    echo "Start training on ${REMOTE_HOST}:"
    echo "  ${ssh_cmd}"
    echo "  ${remote_bin}/incremental_trainer --config ${remote_config}/config.conf"
    echo ""
    echo "Manage datasets:"
    echo "  ${ssh_cmd} '${remote_bin}/dataset_manager --help'"
    echo ""
    echo "========================================================================"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

echo ""
info "ADAI Incremental Trainer Sub-System — Installation Script"
echo ""

preflight_checks

if [[ -n "${REMOTE_HOST}" ]]; then
    remote_install
elif [[ "${COORDINATOR}" == true ]]; then
    if [[ $EUID -ne 0 ]]; then
        error "Coordinator installation requires root privileges (use sudo)"
        exit 1
    fi
    coordinator_install
else
    local_install
fi
