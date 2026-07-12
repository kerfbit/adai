#!/bin/bash
# Cloudflare Tunnel connector - Installation Script
#
# Installs a filled-in cloudflared config + credentials as a systemd service,
# fronting local ADAI services under kerfbit.dev subdomains. Run once per
# tunnel (once on the storage host for adai-storage-tunnel, once on the
# ai-machine for adai-chat-tunnel).
#
# This script does NOT install the cloudflared binary itself, and does NOT
# create the tunnel or its DNS records — see
# docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md for those steps.
#
# Usage:
#   sudo ./install_cloudflared.sh --tunnel-name NAME --config-src PATH \
#       --credentials-src PATH [OPTIONS]
#
# See --help for the full option list.

set -euo pipefail

# ============================================================================
# Configuration Defaults
# ============================================================================

INSTALL_PATH="/etc/cloudflared"
SERVICE_USER="cloudflared"
SERVICE_GROUP="cloudflared"
TUNNEL_NAME=""
CONFIG_SRC=""
CREDENTIALS_SRC=""
CLOUDFLARED_BIN=""
YES=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
Cloudflare Tunnel Connector - Installation Script

Usage: sudo $0 --tunnel-name NAME --config-src PATH --credentials-src PATH [OPTIONS]

Required:
  --tunnel-name NAME     Name given to \`cloudflared tunnel create\` (e.g. adai-storage-tunnel)
  --config-src PATH      Filled-in ingress config (see scripts/cloudflared/*.yml.template)
  --credentials-src PATH Tunnel credentials JSON written by \`cloudflared tunnel create\`
                         (default location: ~/.cloudflared/<TUNNEL_UUID>.json)

Options:
  --install-path PATH   Directory to hold config + credentials (default: /etc/cloudflared)
  --user USER            Service user (default: cloudflared)
  --group GROUP          Service group (default: cloudflared)
  --cloudflared-bin PATH Path to the cloudflared binary (default: auto-detect via PATH)
  --yes                  Skip confirmation prompts (for non-interactive use)
  --help                 Show this help message

Examples:
  sudo $0 --tunnel-name adai-storage-tunnel \\
      --config-src ./config-storage.yml \\
      --credentials-src ~/.cloudflared/1234abcd-....json --yes

Description:
  Installs a cloudflared tunnel connector as a systemd service, running as a
  dedicated "cloudflared" system user (not "adai" — cloudflared is a separate
  trust boundary with no legitimate reason to touch /opt/adai). Copies the
  ingress config and tunnel credentials into place with restrictive
  permissions, writes a hardened systemd unit, and starts the service.

  This script assumes the cloudflared binary is already installed and that
  the tunnel + its DNS records already exist (see
  docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md).

  Service name: cloudflared-<tunnel-name>

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

while [[ $# -gt 0 ]]; do
    case $1 in
        --tunnel-name)
            validate_identifier "--tunnel-name" "$2"
            TUNNEL_NAME="$2"; shift 2 ;;
        --config-src)
            CONFIG_SRC="$2"; shift 2 ;;
        --credentials-src)
            CREDENTIALS_SRC="$2"; shift 2 ;;
        --install-path)
            validate_abs_path "--install-path" "$2"
            INSTALL_PATH="$2"; shift 2 ;;
        --user)
            validate_identifier "--user" "$2"
            SERVICE_USER="$2"; shift 2 ;;
        --group)
            validate_identifier "--group" "$2"
            SERVICE_GROUP="$2"; shift 2 ;;
        --cloudflared-bin)
            CLOUDFLARED_BIN="$2"; shift 2 ;;
        --yes)  YES=true; shift ;;
        --help) show_help; exit 0 ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "${TUNNEL_NAME}" ]]; then
    error "--tunnel-name is required"
    exit 1
fi
if [[ -z "${CONFIG_SRC}" ]]; then
    error "--config-src is required"
    exit 1
fi
if [[ -z "${CREDENTIALS_SRC}" ]]; then
    error "--credentials-src is required (default location: ~/.cloudflared/<TUNNEL_UUID>.json)"
    exit 1
fi

# ============================================================================
# Derived Paths
# ============================================================================

SERVICE_NAME="cloudflared-${TUNNEL_NAME}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CONFIG_DEST="${INSTALL_PATH}/config.yml"

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

    if [[ -z "${CLOUDFLARED_BIN}" ]]; then
        CLOUDFLARED_BIN="$(command -v cloudflared || true)"
    fi
    if [[ -z "${CLOUDFLARED_BIN}" || ! -x "${CLOUDFLARED_BIN}" ]]; then
        error "cloudflared binary not found. Install it first — see"
        error "docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md step 1,"
        error "or pass --cloudflared-bin PATH."
        exit 1
    fi

    if [[ ! -f "${CONFIG_SRC}" ]]; then
        error "--config-src not found: ${CONFIG_SRC}"
        exit 1
    fi
    if ! grep -q '^tunnel:' "${CONFIG_SRC}" || ! grep -q '^credentials-file:' "${CONFIG_SRC}"; then
        error "${CONFIG_SRC} does not look like a filled-in cloudflared ingress config"
        error "(missing 'tunnel:' or 'credentials-file:' key). See scripts/cloudflared/*.yml.template."
        exit 1
    fi
    if grep -q '<TUNNEL_UUID>' "${CONFIG_SRC}"; then
        error "${CONFIG_SRC} still contains the <TUNNEL_UUID> placeholder — fill it in"
        error "with the UUID from 'cloudflared tunnel create ${TUNNEL_NAME}' first."
        exit 1
    fi

    if [[ ! -f "${CREDENTIALS_SRC}" ]]; then
        error "--credentials-src not found: ${CREDENTIALS_SRC}"
        exit 1
    fi

    success "Preflight checks passed"
    echo ""
}

# ============================================================================
# Install
# ============================================================================

install_cloudflared_service() {
    local step_total=6
    local credentials_dest="${INSTALL_PATH}/$(basename "${CREDENTIALS_SRC}")"

    info "Installation Configuration:"
    echo "  Tunnel Name:    ${TUNNEL_NAME}"
    echo "  Install Path:   ${INSTALL_PATH}"
    echo "  Config:         ${CONFIG_DEST}"
    echo "  Credentials:    ${credentials_dest}"
    echo "  Service User:   ${SERVICE_USER}:${SERVICE_GROUP}"
    echo "  Service File:   ${SERVICE_FILE}"
    echo "  cloudflared:    ${CLOUDFLARED_BIN}"
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
            -c "Cloudflare Tunnel connector" "${SERVICE_USER}"
        success "Created system user '${SERVICE_USER}'"
    fi

    # Step 2: Create directory structure
    info "[2/${step_total}] Creating directory structure..."
    mkdir -p "${INSTALL_PATH}"
    chmod 750 "${INSTALL_PATH}"
    success "Directory structure created"

    # Step 3: Copy config + credentials
    info "[3/${step_total}] Installing config and credentials..."
    cp "${CONFIG_SRC}" "${CONFIG_DEST}"
    chmod 640 "${CONFIG_DEST}"
    cp "${CREDENTIALS_SRC}" "${credentials_dest}"
    chmod 600 "${credentials_dest}"
    success "Installed ${CONFIG_DEST} and ${credentials_dest}"

    # Step 4: Set ownership
    info "[4/${step_total}] Setting ownership and permissions..."
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
    success "Ownership and permissions set"

    # Step 5: Render and write systemd unit file
    info "[5/${step_total}] Writing systemd service unit..."
    sed \
        -e "s|@TUNNEL_NAME@|${TUNNEL_NAME}|g" \
        -e "s|@SERVICE_USER@|${SERVICE_USER}|g" \
        -e "s|@SERVICE_GROUP@|${SERVICE_GROUP}|g" \
        -e "s|@CLOUDFLARED_BIN@|${CLOUDFLARED_BIN}|g" \
        -e "s|@CONFIG_PATH@|${CONFIG_DEST}|g" \
        -e "s|@CONFIG_DIR@|${INSTALL_PATH}|g" \
        "${SCRIPT_DIR}/cloudflared.service.template" > "${SERVICE_FILE}"
    chmod 644 "${SERVICE_FILE}"
    success "Wrote ${SERVICE_FILE}"

    # Step 6: Enable and start service
    info "[6/${step_total}] Enabling and starting ${SERVICE_NAME}..."
    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service"
    success "Service '${SERVICE_NAME}' enabled (will start on boot)"

    info "Starting ${SERVICE_NAME}..."
    if systemctl start "${SERVICE_NAME}.service"; then
        success "${SERVICE_NAME} started"
    else
        error "Failed to start ${SERVICE_NAME}"
        warn "Check logs: sudo journalctl -u ${SERVICE_NAME} -n 50"
        exit 1
    fi

    print_summary
}

print_summary() {
    echo ""
    echo "========================================================================"
    success "cloudflared tunnel connector installed!"
    echo "========================================================================"
    echo ""
    echo "Installed:"
    echo "  Config:      ${CONFIG_DEST}"
    echo "  Service:     ${SERVICE_FILE}"
    echo ""
    echo "Service management:"
    echo "  Status:  systemctl status ${SERVICE_NAME}"
    echo "  Logs:    journalctl -u ${SERVICE_NAME} -f"
    echo "  Stop:    systemctl stop ${SERVICE_NAME}"
    echo "  Restart: systemctl restart ${SERVICE_NAME}"
    echo ""
    echo "Next: configure Cloudflare Access on the public hostname(s) in"
    echo "${CONFIG_DEST} — see docs/operations/deployment/CLOUDFLARE_TUNNEL_RELAY.md"
    echo ""
    echo "========================================================================"
    echo ""
}

# ============================================================================
# Main
# ============================================================================

echo ""
info "Cloudflare Tunnel Connector — Installation Script"
echo ""

preflight_checks
install_cloudflared_service
