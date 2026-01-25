#!/bin/bash
# Script to download and install cpp-httplib header-only library

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXTERNAL_DIR="$SCRIPT_DIR/../external"
HTTPLIB_DIR="$EXTERNAL_DIR/cpp-httplib"

echo "=================================================="
echo "  Installing cpp-httplib (header-only library)"
echo "=================================================="

# Create external directory if it doesn't exist
mkdir -p "$EXTERNAL_DIR"

# Check if already installed
if [ -f "$HTTPLIB_DIR/httplib.h" ]; then
    echo "cpp-httplib already installed at $HTTPLIB_DIR"
    exit 0
fi

# Create directory for cpp-httplib
mkdir -p "$HTTPLIB_DIR"

echo "Downloading cpp-httplib..."
cd "$HTTPLIB_DIR"

# Download the latest release (header-only)
HTTPLIB_VERSION="v0.15.3"
DOWNLOAD_URL="https://raw.githubusercontent.com/yhirose/cpp-httplib/$HTTPLIB_VERSION/httplib.h"

if command -v wget &> /dev/null; then
    wget -O httplib.h "$DOWNLOAD_URL"
elif command -v curl &> /dev/null; then
    curl -L -o httplib.h "$DOWNLOAD_URL"
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

if [ -f httplib.h ]; then
    echo "✓ cpp-httplib installed successfully at $HTTPLIB_DIR/httplib.h"
    echo ""
    echo "You can now build the API server with:"
    echo "  cd build"
    echo "  cmake .. -DBUILD_API_SERVER=ON"
    echo "  make chatbot_api_server"
else
    echo "✗ Failed to download cpp-httplib"
    exit 1
fi
