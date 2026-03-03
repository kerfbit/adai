#!/usr/bin/env bash
# =============================================================================
# ADAI Model Service Manager
#
# Loads and manages the ADAI chatbot_api_server as a foreground or background
# (daemon) service. Handles build, start, stop, restart, status, and health
# checks without requiring systemd or root privileges.
#
# Usage:
#   ./scripts/model_service.sh <command> [OPTIONS]
#
# Commands:
#   start       Build (if needed) then start the model service
#   stop        Stop a running background service
#   restart     Stop then start the service
#   status      Show whether the service is running and its PID
#   health      Hit the /health REST endpoint and print the response
#   logs        Tail the service log file (Ctrl+C to exit)
#   build       (Re)build the chatbot_api_server binary only
#   help        Show this help message
#
# Options:
#   --config <path>     Config file  (default: ./config.conf)
#   --vocab  <path>     Vocab file   (overrides config)
#   --model  <path>     Model weights file (overrides config; optional)
#   --port   <number>   Listen port  (default from config, or 8080)
#   --log-level <lvl>   DEBUG|INFO|WARN|ERROR (default: INFO)
#   --build-type <type> debug|release (default: release)
#   --foreground        Run in the foreground instead of daemonising
#   --jobs   <n>        Parallel make jobs for build step (default: nproc)
#   --pidfile <path>    PID file path (default: /tmp/adai_model_service.pid)
#   --logfile <path>    Log file path (default: /tmp/adai_model_service.log)
#
# Examples:
#   ./scripts/model_service.sh start
#   ./scripts/model_service.sh start --model models/model.bin --port 9000
#   ./scripts/model_service.sh start --foreground
#   ./scripts/model_service.sh stop
#   ./scripts/model_service.sh restart --log-level DEBUG
#   ./scripts/model_service.sh health
#   ./scripts/model_service.sh logs
#   ./scripts/model_service.sh build --build-type debug
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_FILE="${REPO_ROOT}/config.conf"
VOCAB_PATH=""
MODEL_PATH=""
PORT=""
LOG_LEVEL="INFO"
BUILD_TYPE="release"
FOREGROUND=false
JOBS="$(nproc 2>/dev/null || echo 4)"
PID_FILE="/tmp/adai_model_service.pid"
LOG_FILE="/tmp/adai_model_service.log"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[OK]${NC}      $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC}   $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}==> $*${NC}"; }

die() { error "$*"; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
COMMAND="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG_FILE="$2";  shift 2 ;;
        --vocab)       VOCAB_PATH="$2";   shift 2 ;;
        --model)       MODEL_PATH="$2";   shift 2 ;;
        --port)        PORT="$2";         shift 2 ;;
        --log-level)   LOG_LEVEL="$2";    shift 2 ;;
        --build-type)  BUILD_TYPE="$2";   shift 2 ;;
        --foreground)  FOREGROUND=true;   shift   ;;
        --jobs)        JOBS="$2";         shift 2 ;;
        --pidfile)     PID_FILE="$2";     shift 2 ;;
        --logfile)     LOG_FILE="$2";     shift 2 ;;
        -h|--help)     COMMAND="help";    shift   ;;
        *) die "Unknown option: $1  (run with 'help' for usage)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate the binary for the requested build type
# ---------------------------------------------------------------------------
get_binary() {
    case "${BUILD_TYPE}" in
        release) echo "${REPO_ROOT}/build/release/src/chatbot_api_server" ;;
        debug)   echo "${REPO_ROOT}/build/src/chatbot_api_server" ;;
        *)       die "Unknown build type '${BUILD_TYPE}'. Use 'debug' or 'release'." ;;
    esac
}

get_build_dir() {
    case "${BUILD_TYPE}" in
        release) echo "${REPO_ROOT}/build/release" ;;
        debug)   echo "${REPO_ROOT}/build" ;;
    esac
}

# ---------------------------------------------------------------------------
# Derive effective port (flag > config file > hardcoded default)
# ---------------------------------------------------------------------------
get_effective_port() {
    if [[ -n "${PORT}" ]]; then
        echo "${PORT}"
    elif [[ -f "${CONFIG_FILE}" ]]; then
        grep -E '^PORT=' "${CONFIG_FILE}" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '[:space:]' || echo "8080"
    else
        echo "8080"
    fi
}

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
cmd_build() {
    header "Build: chatbot_api_server [${BUILD_TYPE}]"

    local build_dir
    build_dir="$(get_build_dir)"

    if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
        die "CMake build directory '${build_dir}' is not configured.\n       Run cmake --preset ... or cmake -B '${build_dir}' first."
    fi

    info "Build directory : ${build_dir}"
    info "Parallel jobs   : ${JOBS}"

    cmake --build "${build_dir}" \
          --target chatbot_api_server \
          --parallel "${JOBS}"

    local binary
    binary="$(get_binary)"

    if [[ -x "${binary}" ]]; then
        success "Binary ready: ${binary}"
    else
        die "Build completed but binary not found at: ${binary}"
    fi
}

# ---------------------------------------------------------------------------
# Assemble server arguments from flags / config
# ---------------------------------------------------------------------------
build_server_args() {
    local args=()

    args+=("--config" "${CONFIG_FILE}")
    args+=("--log-level" "${LOG_LEVEL}")

    [[ -n "${VOCAB_PATH}" ]] && args+=("--vocab"  "${VOCAB_PATH}")
    [[ -n "${MODEL_PATH}" ]] && args+=("--model"  "${MODEL_PATH}")
    [[ -n "${PORT}"       ]] && args+=("--port"   "${PORT}")

    echo "${args[@]}"
}

# ---------------------------------------------------------------------------
# Start command
# ---------------------------------------------------------------------------
cmd_start() {
    header "Start: ADAI model service"

    # ---- Build if binary missing ----------------------------------------
    local binary
    binary="$(get_binary)"

    if [[ ! -x "${binary}" ]]; then
        warn "Binary not found at ${binary} — building now..."
        cmd_build
    else
        info "Binary        : ${binary}"
    fi

    # ---- Abort if already running ----------------------------------------
    if [[ -f "${PID_FILE}" ]]; then
        local old_pid
        old_pid="$(cat "${PID_FILE}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            warn "Service is already running (PID ${old_pid})."
            warn "Use 'restart' to restart it."
            exit 0
        else
            info "Stale PID file found — removing."
            rm -f "${PID_FILE}"
        fi
    fi

    # ---- Validate config / vocab ----------------------------------------
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        warn "Config file not found: ${CONFIG_FILE}"
    fi

    # Derive vocab from config if not overridden
    local effective_vocab="${VOCAB_PATH}"
    if [[ -z "${effective_vocab}" && -f "${CONFIG_FILE}" ]]; then
        effective_vocab="$(grep -E '^VOCAB_PATH=' "${CONFIG_FILE}" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '[:space:]' || true)"
    fi

    if [[ -z "${effective_vocab}" || ! -f "${effective_vocab}" ]]; then
        die "Vocabulary file not found: '${effective_vocab:-<unset>}'\n       Set VOCAB_PATH in ${CONFIG_FILE} or pass --vocab <path>."
    fi

    # Warn (not fatal) if model weights are absent
    local effective_model="${MODEL_PATH}"
    if [[ -z "${effective_model}" && -f "${CONFIG_FILE}" ]]; then
        effective_model="$(grep -E '^MODEL_PATH=' "${CONFIG_FILE}" 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '[:space:]' || true)"
    fi
    if [[ -z "${effective_model}" ]]; then
        warn "No MODEL_PATH configured — server will use random initialisation."
    elif [[ ! -f "${effective_model}" ]]; then
        warn "Model weights file not found: ${effective_model}"
        warn "Server will use random initialisation (run training first)."
    else
        info "Model weights : ${effective_model}"
    fi

    # ---- Build argument list ---------------------------------------------
    local server_args=()
    read -ra server_args <<< "$(build_server_args)"

    local effective_port
    effective_port="$(get_effective_port)"

    info "Config file   : ${CONFIG_FILE}"
    info "Vocab file    : ${effective_vocab}"
    info "Listen port   : ${effective_port}"
    info "Log level     : ${LOG_LEVEL}"

    # ---- Launch ----------------------------------------------------------
    if "${FOREGROUND}"; then
        info "Running in foreground (Ctrl+C to stop)..."
        echo ""
        exec "${binary}" "${server_args[@]}"
    else
        info "Log file      : ${LOG_FILE}"
        info "PID file      : ${PID_FILE}"
        info "Launching in background..."

        # Rotate existing log
        if [[ -f "${LOG_FILE}" ]]; then
            mv "${LOG_FILE}" "${LOG_FILE}.$(date +%Y%m%d_%H%M%S).bak"
        fi

        # Start process detached from terminal
        nohup "${binary}" "${server_args[@]}" \
            >> "${LOG_FILE}" 2>&1 &
        local service_pid=$!
        echo "${service_pid}" > "${PID_FILE}"

        # Give it a moment then verify it is still alive
        sleep 2
        if kill -0 "${service_pid}" 2>/dev/null; then
            success "Service started  (PID ${service_pid})"
            info "  Logs : tail -f ${LOG_FILE}"
            info "  Stop : $0 stop"

            # Poll for the HTTP server to become ready (up to 30 s)
            local attempts=0
            local max_attempts=30
            info "Waiting for HTTP server on port ${effective_port}..."
            while (( attempts < max_attempts )); do
                if curl -sf "http://localhost:${effective_port}/health" >/dev/null 2>&1; then
                    success "Server is ready at http://localhost:${effective_port}"
                    break
                fi
                (( attempts++ ))
                sleep 1
            done
            if (( attempts >= max_attempts )); then
                warn "Server did not respond on port ${effective_port} within ${max_attempts}s."
                warn "Check logs: ${LOG_FILE}"
            fi
        else
            rm -f "${PID_FILE}"
            die "Service exited immediately. Check ${LOG_FILE} for details."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Stop command
# ---------------------------------------------------------------------------
cmd_stop() {
    header "Stop: ADAI model service"

    if [[ ! -f "${PID_FILE}" ]]; then
        warn "PID file not found (${PID_FILE}). Service may not be running."
        return 0
    fi

    local pid
    pid="$(cat "${PID_FILE}")"

    if ! kill -0 "${pid}" 2>/dev/null; then
        warn "Process ${pid} is not running. Removing stale PID file."
        rm -f "${PID_FILE}"
        return 0
    fi

    info "Sending SIGTERM to PID ${pid}..."
    kill -TERM "${pid}"

    # Wait for graceful shutdown
    local waited=0
    while kill -0 "${pid}" 2>/dev/null && (( waited < 30 )); do
        sleep 1
        (( waited++ ))
    done

    if kill -0 "${pid}" 2>/dev/null; then
        warn "Process ${pid} did not exit in 30 s — sending SIGKILL..."
        kill -KILL "${pid}" 2>/dev/null || true
        sleep 1
    fi

    rm -f "${PID_FILE}"
    success "Service stopped (was PID ${pid})"
}

# ---------------------------------------------------------------------------
# Restart command
# ---------------------------------------------------------------------------
cmd_restart() {
    cmd_stop  || true
    sleep 1
    cmd_start
}

# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------
cmd_status() {
    header "Status: ADAI model service"

    if [[ ! -f "${PID_FILE}" ]]; then
        echo -e "  Status  : ${RED}STOPPED${NC} (no PID file)"
        return 1
    fi

    local pid
    pid="$(cat "${PID_FILE}")"

    if kill -0 "${pid}" 2>/dev/null; then
        local effective_port
        effective_port="$(get_effective_port)"

        echo -e "  Status  : ${GREEN}RUNNING${NC}"
        echo    "  PID     : ${pid}"
        echo    "  Port    : ${effective_port}"
        echo    "  PID file: ${PID_FILE}"
        echo    "  Log file: ${LOG_FILE}"

        # Process details from /proc
        if [[ -f "/proc/${pid}/status" ]]; then
            local vmrss
            vmrss="$(grep VmRSS /proc/${pid}/status 2>/dev/null | awk '{print $2, $3}' || true)"
            [[ -n "${vmrss}" ]] && echo "  Memory  : ${vmrss}"
        fi

        # Quick health check
        if curl -sf "http://localhost:${effective_port}/health" >/dev/null 2>&1; then
            echo -e "  Health  : ${GREEN}OK${NC} (HTTP /health responded)"
        else
            echo -e "  Health  : ${YELLOW}UNKNOWN${NC} (no response on port ${effective_port})"
        fi
    else
        echo -e "  Status  : ${RED}DEAD${NC} (stale PID ${pid})"
        rm -f "${PID_FILE}"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Health command
# ---------------------------------------------------------------------------
cmd_health() {
    header "Health check"

    local effective_port
    effective_port="$(get_effective_port)"
    local url="http://localhost:${effective_port}/health"

    info "GET ${url}"
    if ! curl -sf --connect-timeout 5 "${url}"; then
        echo ""
        error "Health endpoint did not respond at ${url}"
        exit 1
    fi
    echo ""
    success "Health check passed"
}

# ---------------------------------------------------------------------------
# Logs command
# ---------------------------------------------------------------------------
cmd_logs() {
    if [[ ! -f "${LOG_FILE}" ]]; then
        warn "Log file not found: ${LOG_FILE}"
        exit 1
    fi
    info "Tailing ${LOG_FILE}  (Ctrl+C to exit)"
    tail -n 50 -f "${LOG_FILE}"
}

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
cmd_help() {
    # Print lines from the top-of-file comment block (lines starting with #)
    # Stop at the first non-comment, non-blank line (i.e. the script body)
    awk '
        NR < 3                  { next }
        /^[^#]/ && NR > 3      { exit }
        { sub(/^# ?/, ""); print }
    ' "${BASH_SOURCE[0]}"
}

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
case "${COMMAND}" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_restart ;;
    status)  cmd_status  ;;
    health)  cmd_health  ;;
    logs)    cmd_logs    ;;
    build)   cmd_build   ;;
    help|-h|--help) cmd_help ;;
    *) error "Unknown command: '${COMMAND}'"; cmd_help; exit 1 ;;
esac
