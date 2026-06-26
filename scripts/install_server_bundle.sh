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
STORAGE_BACKEND="sqlite+file"
DB_PATH=""
DB_URL=""
DB_POOL_SIZE=4
SETUP_POSTGRES=false
PG_DB_NAME="adai"
PG_DB_USER=""
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
  --storage-backend BACKEND Metrics storage backend (default: sqlite+file)
                            Options: file, sqlite, postgres, sqlite+file, postgres+file
  --db-path PATH            SQLite database file path (default: <metrics-dir>/metrics.db)
  --db-url URL              PostgreSQL connection URL (required when backend includes postgres)
  --db-pool-size N          PostgreSQL connection pool size (default: 4)
  --setup-postgres          Install PostgreSQL, create database and role, run schema setup
  --pg-db-name NAME         PostgreSQL database name (default: adai)
  --pg-db-user USER         PostgreSQL role for ADAI (default: same as --user)
  --yes                     Skip confirmation prompts (for non-interactive use)
  --help                    Show this help message

PostgreSQL quick-start:
  sudo $0 --setup-postgres --yes

  Equivalent to:
    1. apt install postgresql           (or dnf on RHEL/Fedora)
    2. CREATE ROLE adai LOGIN
    3. CREATE DATABASE adai OWNER adai
    4. psql -d adai -f scripts/setup_postgres.sql
    5. --storage-backend postgres+file --db-url postgresql://adai@localhost/adai

  If --storage-backend and --db-url are not explicitly set, --setup-postgres
  sets them automatically.

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
        --storage-backend)
            STORAGE_BACKEND="$2"; shift 2 ;;
        --db-path)
            DB_PATH="$2"; shift 2 ;;
        --db-url)
            DB_URL="$2"; shift 2 ;;
        --db-pool-size)
            DB_POOL_SIZE="$2"; shift 2 ;;
        --setup-postgres) SETUP_POSTGRES=true; shift ;;
        --pg-db-name)
            PG_DB_NAME="$2"; shift 2 ;;
        --pg-db-user)
            PG_DB_USER="$2"; shift 2 ;;
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
[[ -z "${DB_PATH}" ]]           && DB_PATH="${METRICS_DIR}/metrics.db"
[[ -z "${PG_DB_USER}" ]]        && PG_DB_USER="${SERVICE_USER}"

# --setup-postgres implies postgres+file backend and auto-derives --db-url
# unless the caller explicitly set them.
if [[ "${SETUP_POSTGRES}" == true ]]; then
    if [[ "${STORAGE_BACKEND}" == "sqlite+file" ]]; then
        STORAGE_BACKEND="postgres+file"
    fi
    if [[ -z "${DB_URL}" ]]; then
        DB_URL="postgresql://${PG_DB_USER}@localhost/${PG_DB_NAME}"
    fi
fi

BUILD_BIN_DIR="${REPO_ROOT}/${BUILD_DIR}/bin"
SETUP_SQL="${SCRIPT_DIR}/setup_postgres.sql"

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

    # Validate storage backend
    case "${STORAGE_BACKEND}" in
        file|sqlite|postgres|sqlite+file|postgres+file) ;;
        *)
            error "Invalid --storage-backend '${STORAGE_BACKEND}'"
            error "  Valid options: file, sqlite, postgres, sqlite+file, postgres+file"
            exit 1
            ;;
    esac

    if [[ "${STORAGE_BACKEND}" == *postgres* && -z "${DB_URL}" && "${SETUP_POSTGRES}" != true ]]; then
        error "--db-url is required when storage backend includes postgres"
        error "  Hint: use --setup-postgres to install PostgreSQL and auto-configure"
        exit 1
    fi

    if [[ "${SETUP_POSTGRES}" == true && ! -f "${SETUP_SQL}" ]]; then
        error "Schema file not found: ${SETUP_SQL}"
        error "  Expected at: scripts/setup_postgres.sql (relative to repo root)"
        exit 1
    fi

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
# PostgreSQL Setup (--setup-postgres)
# ============================================================================

detect_pkg_manager() {
    if command -v apt-get &>/dev/null; then
        echo "apt"
    elif command -v dnf &>/dev/null; then
        echo "dnf"
    elif command -v yum &>/dev/null; then
        echo "yum"
    else
        echo "unknown"
    fi
}

install_postgres_packages() {
    local pkg_mgr
    pkg_mgr="$(detect_pkg_manager)"

    info "Detected package manager: ${pkg_mgr}"

    case "${pkg_mgr}" in
        apt)
            info "Installing PostgreSQL via apt..."
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
                postgresql postgresql-client libpq-dev >/dev/null
            ;;
        dnf)
            info "Installing PostgreSQL via dnf..."
            dnf install -y -q postgresql-server postgresql postgresql-devel
            # Initialize the data directory on first install
            if [[ ! -d /var/lib/pgsql/data/base ]]; then
                postgresql-setup --initdb 2>/dev/null || true
            fi
            ;;
        yum)
            info "Installing PostgreSQL via yum..."
            yum install -y -q postgresql-server postgresql postgresql-devel
            if [[ ! -d /var/lib/pgsql/data/base ]]; then
                postgresql-setup initdb 2>/dev/null || true
            fi
            ;;
        *)
            error "Unsupported package manager — install PostgreSQL manually"
            error "  Debian/Ubuntu:  sudo apt install postgresql postgresql-client libpq-dev"
            error "  RHEL/Fedora:    sudo dnf install postgresql-server postgresql postgresql-devel"
            exit 1
            ;;
    esac

    # Ensure the service is running
    systemctl enable postgresql
    systemctl start postgresql
    success "PostgreSQL server is running"
}

pg_role_exists() {
    sudo -u postgres psql -tAc \
        "SELECT 1 FROM pg_roles WHERE rolname = '${1}'" 2>/dev/null | grep -q 1
}

pg_db_exists() {
    sudo -u postgres psql -tAc \
        "SELECT 1 FROM pg_database WHERE datname = '${1}'" 2>/dev/null | grep -q 1
}

setup_postgres() {
    info "Setting up PostgreSQL for ADAI..."

    # Step 1: Install packages if psql is not available
    if ! command -v psql &>/dev/null; then
        confirm "PostgreSQL is not installed. Install it now?"
        install_postgres_packages
    else
        info "PostgreSQL client already installed ($(psql --version | head -1))"
        # Make sure the server is running
        if ! systemctl is-active --quiet postgresql; then
            warn "PostgreSQL service is not running — starting it"
            systemctl enable postgresql
            systemctl start postgresql
        fi
    fi

    # Step 2: Create the database role
    if pg_role_exists "${PG_DB_USER}"; then
        info "PostgreSQL role '${PG_DB_USER}' already exists"
    else
        info "Creating PostgreSQL role '${PG_DB_USER}'..."
        sudo -u postgres psql -c "CREATE ROLE ${PG_DB_USER} LOGIN;" 2>&1 | \
            grep -v "^$" || true
        success "Created role '${PG_DB_USER}'"
    fi

    # Step 3: Create the database
    if pg_db_exists "${PG_DB_NAME}"; then
        info "PostgreSQL database '${PG_DB_NAME}' already exists"
    else
        info "Creating PostgreSQL database '${PG_DB_NAME}'..."
        sudo -u postgres createdb -O "${PG_DB_USER}" "${PG_DB_NAME}"
        success "Created database '${PG_DB_NAME}' (owner: ${PG_DB_USER})"
    fi

    # Step 4: Ensure peer auth works for the service user.
    # The adai system user needs to connect via Unix socket.  Most default
    # pg_hba.conf files allow "local all all peer", which maps the OS user
    # to the same-named PostgreSQL role — exactly what we need.  If the role
    # was just created above, peer auth already covers it.

    # Step 5: Run the schema setup SQL
    if [[ ! -f "${SETUP_SQL}" ]]; then
        error "Schema file not found: ${SETUP_SQL}"
        error "Expected at: scripts/setup_postgres.sql (relative to repo root)"
        exit 1
    fi

    info "Applying schema from ${SETUP_SQL}..."
    sudo -u postgres psql -d "${PG_DB_NAME}" -f "${SETUP_SQL}" 2>&1 | \
        while IFS= read -r line; do
            case "${line}" in
                BEGIN|COMMIT|"") ;;
                CREATE*|INSERT*) success "  ${line}" ;;
                NOTICE*)         info   "  ${line}" ;;
                ERROR*)          error  "  ${line}" ;;
                *)               echo   "  ${line}" ;;
            esac
        done
    success "PostgreSQL schema applied"

    # Step 6: Quick connectivity check as the service user
    info "Verifying connectivity as '${PG_DB_USER}'..."
    if sudo -u "${PG_DB_USER}" psql -d "${PG_DB_NAME}" -c "SELECT version();" &>/dev/null; then
        success "Role '${PG_DB_USER}' can connect to '${PG_DB_NAME}' via peer auth"
    else
        warn "Peer auth test failed — the service user may need pg_hba.conf adjustment"
        warn "  The metrics_api_server will retry with the connection URL: ${DB_URL}"
        warn "  Verify manually: sudo -u ${PG_DB_USER} psql -d ${PG_DB_NAME} -c 'SELECT 1'"
    fi

    echo ""
}

# ============================================================================
# Install
# ============================================================================

install_bundle() {
    local step_total=8
    if [[ "${SETUP_POSTGRES}" == true ]]; then
        step_total=9
    fi

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
    echo "    Storage Backend:    ${STORAGE_BACKEND}"
    if [[ "${STORAGE_BACKEND}" == *sqlite* ]]; then
        echo "    SQLite DB Path:     ${DB_PATH}"
    fi
    if [[ "${STORAGE_BACKEND}" == *postgres* ]]; then
        echo "    PostgreSQL URL:     ${DB_URL}"
        echo "    Connection Pool:    ${DB_POOL_SIZE}"
    fi
    echo "    Name Service URL:   http://localhost:${MNS_PORT}"
    echo ""
    if [[ "${SETUP_POSTGRES}" == true ]]; then
        echo "  PostgreSQL Setup:"
        echo "    Database:           ${PG_DB_NAME}"
        echo "    Role:               ${PG_DB_USER}"
        echo "    Schema SQL:         ${SETUP_SQL}"
        echo ""
    fi
    echo "  Build Source:         ${BUILD_BIN_DIR}/"
    echo ""

    confirm "Continue with installation?"

    local step=0

    # Step: Create system user and group
    step=$((step + 1))
    info "[${step}/${step_total}] Creating service user and group..."
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

    # Step: Create directory structure
    step=$((step + 1))
    info "[${step}/${step_total}] Creating directory structure..."
    mkdir -p "${BIN_DIR}"
    mkdir -p "${CONF_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${MNS_DATA_DIR}"
    mkdir -p "${REGISTRY_DATA_DIR}"
    mkdir -p "${METRICS_DIR}"
    success "Directory structure created"

    # Step: Copy binaries
    step=$((step + 1))
    info "[${step}/${step_total}] Installing binaries..."
    for bin in mns_server registry_server metrics_api_server dataset_manager mns_cli; do
        if [[ -f "${BUILD_BIN_DIR}/${bin}" ]]; then
            cp "${BUILD_BIN_DIR}/${bin}" "${BIN_DIR}/${bin}"
            chmod 755 "${BIN_DIR}/${bin}"
            success "  Installed ${bin}"
        fi
    done

    # Step: Write config file
    step=$((step + 1))
    info "[${step}/${step_total}] Writing configuration..."
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

# Metrics Database Persistence (TD-020)
METRICS_STORAGE_BACKEND=${STORAGE_BACKEND}
METRICS_DB_PATH=${DB_PATH}
EOF

    if [[ "${STORAGE_BACKEND}" == *postgres* ]]; then
        cat >> "${CONF_DIR}/config.conf" <<EOF
METRICS_DB_URL=${DB_URL}
METRICS_DB_POOL_SIZE=${DB_POOL_SIZE}
EOF
    fi
    chmod 644 "${CONF_DIR}/config.conf"
    success "Wrote ${CONF_DIR}/config.conf"

    # Step: PostgreSQL setup (conditional)
    if [[ "${SETUP_POSTGRES}" == true ]]; then
        step=$((step + 1))
        info "[${step}/${step_total}] Installing and configuring PostgreSQL..."
        setup_postgres
    fi

    # Step: Set ownership and permissions
    step=$((step + 1))
    info "[${step}/${step_total}] Setting ownership and permissions..."
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_PATH}"
    chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
    success "Ownership and permissions set"

    # Step: Write systemd unit files
    step=$((step + 1))
    info "[${step}/${step_total}] Writing systemd service units..."

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
    local metrics_exec="${BIN_DIR}/metrics_api_server --port ${METRICS_PORT} --name-service-url http://localhost:${MNS_PORT}"
    metrics_exec+=" --storage-backend ${STORAGE_BACKEND}"
    if [[ "${STORAGE_BACKEND}" == *sqlite* ]]; then
        metrics_exec+=" --db-path ${DB_PATH}"
    fi
    if [[ "${STORAGE_BACKEND}" == *postgres* ]]; then
        metrics_exec+=" --db-url ${DB_URL} --db-pool-size ${DB_POOL_SIZE}"
    fi

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
ExecStart=${metrics_exec}
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

    # Step: Enable and start services
    step=$((step + 1))
    info "[${step}/${step_total}] Enabling and starting services..."
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

    # Step: Verify
    step=$((step + 1))
    info "[${step}/${step_total}] Verifying installation..."
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
    echo "Storage:   ${STORAGE_BACKEND}"
    if [[ "${STORAGE_BACKEND}" == *sqlite* ]]; then
        echo "           SQLite DB: ${DB_PATH}"
    fi
    if [[ "${STORAGE_BACKEND}" == *postgres* ]]; then
        echo "           PostgreSQL: ${DB_URL} (pool=${DB_POOL_SIZE})"
        echo "           Database:   ${PG_DB_NAME}"
        echo ""
        echo "PostgreSQL administration:"
        echo "  sudo -u postgres psql -d ${PG_DB_NAME}        # superuser shell"
        echo "  sudo -u ${PG_DB_USER} psql -d ${PG_DB_NAME}   # service user shell"
        echo "  sudo systemctl status postgresql               # server status"
    fi
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
if [[ "${SETUP_POSTGRES}" == true ]]; then
    info "PostgreSQL setup enabled"
fi
echo ""

preflight_checks
install_bundle
