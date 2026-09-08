#!/bin/bash

# @adai-status: beta        (capped by TD-043 — see TECHNICAL_DEBT.md)
# @adai-version: 0.8.0
# @adai-reviewed: 2026-09-07

# Docker build script for ADAI Chatbot API Server

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
IMAGE_NAME="adai-chatbot"
IMAGE_TAG="latest"
BUILD_ARGS=""
NO_CACHE=false
PLATFORM=""

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
Usage: $(basename "$0") [OPTIONS]

Build Docker image for ADAI Chatbot API Server

Options:
    -t, --tag TAG           Docker image tag (default: latest)
    -n, --name NAME         Docker image name (default: adai-chatbot)
    --no-cache              Build without using cache
    --platform PLATFORM     Target platform (e.g., linux/amd64, linux/arm64)
    -h, --help              Display this help message

Examples:
    $(basename "$0")                          # Build with defaults
    $(basename "$0") -t v1.0.0                # Build with specific tag
    $(basename "$0") --no-cache               # Build without cache
    $(basename "$0") --platform linux/amd64   # Build for specific platform

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
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

# Build the image
print_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
print_info "Build context: ${PROJECT_ROOT}"

# Construct build command
BUILD_CMD="docker build"

if [ "$NO_CACHE" = true ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

if [ -n "$PLATFORM" ]; then
    BUILD_CMD="$BUILD_CMD --platform $PLATFORM"
fi

BUILD_CMD="$BUILD_CMD -t ${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_CMD="$BUILD_CMD -f ${PROJECT_ROOT}/Dockerfile"
BUILD_CMD="$BUILD_CMD ${PROJECT_ROOT}"

print_info "Running: $BUILD_CMD"
eval $BUILD_CMD

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Display image information
    print_info "Image details:"
    docker images ${IMAGE_NAME}:${IMAGE_TAG}
    
    print_info ""
    print_info "Next steps:"
    print_info "  1. Run container: docker run -p 8080:8080 ${IMAGE_NAME}:${IMAGE_TAG}"
    print_info "  2. Use docker-compose: docker-compose up"
    print_info "  3. Push to registry: docker push ${IMAGE_NAME}:${IMAGE_TAG}"
else
    print_error "Docker build failed"
    exit 1
fi
