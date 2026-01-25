#!/bin/bash
# Docker deployment script for ADAI Chatbot API Server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONTAINER_NAME="adai-chatbot-api"
IMAGE_NAME="adai-chatbot"
IMAGE_TAG="latest"
PORT=8080
VOCAB_FILE="${PROJECT_ROOT}/vocab.txt"
MODEL_DIR="${PROJECT_ROOT}/models"
LOG_DIR="${PROJECT_ROOT}/logs"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $(basename "$0") COMMAND [OPTIONS]

Deploy and manage ADAI Chatbot API Server containers

Commands:
    start               Start the chatbot API server
    stop                Stop the chatbot API server
    restart             Restart the chatbot API server
    logs                View container logs
    status              Check container status
    shell               Open shell in running container
    clean               Remove stopped containers and images

Options:
    -p, --port PORT             Host port to bind (default: 8080)
    -v, --vocab FILE            Path to vocabulary file
    -m, --model-dir DIR         Path to models directory
    -l, --log-dir DIR           Path to logs directory
    -n, --name NAME             Container name (default: adai-chatbot-api)
    -t, --tag TAG               Image tag (default: latest)
    -d, --detach                Run in detached mode (default)
    -h, --help                  Display this help message

Examples:
    $(basename "$0") start                      # Start with defaults
    $(basename "$0") start -p 9090              # Start on port 9090
    $(basename "$0") stop                       # Stop container
    $(basename "$0") logs                       # View logs
    $(basename "$0") status                     # Check status

EOF
    exit 0
}

# Parse command
if [ $# -eq 0 ]; then
    print_error "No command specified"
    usage
fi

COMMAND=$1
shift

# Parse options
DETACH=true
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -v|--vocab)
            VOCAB_FILE="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        --no-detach)
            DETACH=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Function to start container
start_container() {
    print_info "Starting ADAI Chatbot API Server..."
    
    # Create necessary directories
    mkdir -p "$MODEL_DIR" "$LOG_DIR"
    
    # Check if vocabulary file exists
    if [ ! -f "$VOCAB_FILE" ]; then
        print_warning "Vocabulary file not found: $VOCAB_FILE"
        print_warning "Container will start but may not function correctly"
    fi
    
    # Check if container already exists
    if container_exists; then
        if container_running; then
            print_warning "Container is already running"
            return 0
        else
            print_info "Starting existing container..."
            docker start "$CONTAINER_NAME"
            print_success "Container started: $CONTAINER_NAME"
            return 0
        fi
    fi
    
    # Build run command
    RUN_CMD="docker run"
    
    if [ "$DETACH" = true ]; then
        RUN_CMD="$RUN_CMD -d"
    fi
    
    RUN_CMD="$RUN_CMD --name $CONTAINER_NAME"
    RUN_CMD="$RUN_CMD -p ${PORT}:8080"
    
    # Mount vocabulary file if it exists
    if [ -f "$VOCAB_FILE" ]; then
        RUN_CMD="$RUN_CMD -v ${VOCAB_FILE}:/app/vocab/vocab.txt:ro"
    fi
    
    # Mount directories
    RUN_CMD="$RUN_CMD -v ${MODEL_DIR}:/app/models:ro"
    RUN_CMD="$RUN_CMD -v ${LOG_DIR}:/app/logs:rw"
    
    RUN_CMD="$RUN_CMD ${IMAGE_NAME}:${IMAGE_TAG}"
    
    print_info "Running: $RUN_CMD"
    eval $RUN_CMD
    
    if [ $? -eq 0 ]; then
        print_success "Container started successfully: $CONTAINER_NAME"
        print_info "API available at: http://localhost:${PORT}"
        print_info "Health check: http://localhost:${PORT}/health"
        
        if [ "$DETACH" = true ]; then
            print_info "View logs with: $0 logs"
        fi
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to stop container
stop_container() {
    print_info "Stopping ADAI Chatbot API Server..."
    
    if ! container_exists; then
        print_warning "Container does not exist: $CONTAINER_NAME"
        return 0
    fi
    
    if ! container_running; then
        print_warning "Container is not running: $CONTAINER_NAME"
        return 0
    fi
    
    docker stop "$CONTAINER_NAME"
    print_success "Container stopped: $CONTAINER_NAME"
}

# Function to restart container
restart_container() {
    print_info "Restarting ADAI Chatbot API Server..."
    stop_container
    sleep 2
    start_container
}

# Function to view logs
view_logs() {
    if ! container_exists; then
        print_error "Container does not exist: $CONTAINER_NAME"
        exit 1
    fi
    
    print_info "Viewing logs for: $CONTAINER_NAME"
    print_info "Press Ctrl+C to exit"
    docker logs -f "$CONTAINER_NAME"
}

# Function to check status
check_status() {
    if ! container_exists; then
        print_info "Container does not exist: $CONTAINER_NAME"
        return 0
    fi
    
    if container_running; then
        print_success "Container is running: $CONTAINER_NAME"
        docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
        
        # Check health
        print_info "Checking health endpoint..."
        sleep 2
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            print_success "Health check passed"
            curl -s "http://localhost:${PORT}/health" | python3 -m json.tool 2>/dev/null || cat
        else
            print_warning "Health check failed or endpoint not ready"
        fi
    else
        print_warning "Container exists but is not running: $CONTAINER_NAME"
        docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}"
    fi
}

# Function to open shell
open_shell() {
    if ! container_running; then
        print_error "Container is not running: $CONTAINER_NAME"
        exit 1
    fi
    
    print_info "Opening shell in container: $CONTAINER_NAME"
    docker exec -it "$CONTAINER_NAME" /bin/bash
}

# Function to clean up
cleanup() {
    print_info "Cleaning up Docker resources..."
    
    # Stop and remove container
    if container_exists; then
        print_info "Removing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Ask before removing images
    read -p "Remove image ${IMAGE_NAME}:${IMAGE_TAG}? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null || true
        print_success "Image removed"
    fi
    
    print_success "Cleanup complete"
}

# Execute command
case $COMMAND in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    logs)
        view_logs
        ;;
    status)
        check_status
        ;;
    shell)
        open_shell
        ;;
    clean)
        cleanup
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        usage
        ;;
esac
