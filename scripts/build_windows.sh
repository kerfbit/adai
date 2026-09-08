#!/bin/bash

# @adai-status: beta        (capped by TD-043 — see TECHNICAL_DEBT.md)
# @adai-version: 0.8.0
# @adai-reviewed: 2026-09-07

# Cross-compile ADAI chatbot for Windows from Linux
# Requires: mingw-w64 toolchain
# Install on Ubuntu/Debian: sudo apt-get install mingw-w64 g++-mingw-w64

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build-windows"
TOOLCHAIN_FILE="${PROJECT_DIR}/cmake/toolchains/mingw-w64.cmake"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ADAI Windows Cross-Compilation${NC}"
echo -e "${BLUE}========================================${NC}"

# Check for MinGW-w64
echo -e "\n${YELLOW}Checking for MinGW-w64 toolchain...${NC}"
if ! command -v x86_64-w64-mingw32-g++ &> /dev/null; then
    echo -e "${RED}ERROR: MinGW-w64 toolchain not found!${NC}"
    echo -e "${YELLOW}Install it with:${NC}"
    echo -e "  Ubuntu/Debian: ${GREEN}sudo apt-get install mingw-w64 g++-mingw-w64${NC}"
    echo -e "  Fedora/RHEL:   ${GREEN}sudo dnf install mingw64-gcc-c++${NC}"
    echo -e "  Arch Linux:    ${GREEN}sudo pacman -S mingw-w64-gcc${NC}"
    exit 1
fi

MINGW_VERSION=$(x86_64-w64-mingw32-g++ --version | head -n1)
echo -e "${GREEN}✓ Found: ${MINGW_VERSION}${NC}"

# Check for CMake
echo -e "\n${YELLOW}Checking for CMake...${NC}"
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}ERROR: CMake not found!${NC}"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1)
echo -e "${GREEN}✓ Found: ${CMAKE_VERSION}${NC}"

# Clean previous build if requested
if [ "$1" == "clean" ]; then
    echo -e "\n${YELLOW}Cleaning previous build...${NC}"
    rm -rf "${BUILD_DIR}"
    echo -e "${GREEN}✓ Build directory cleaned${NC}"
    shift
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
mkdir -p "${BUILD_DIR}"
echo -e "${GREEN}✓ Build directory: ${BUILD_DIR}${NC}"

# Configure CMake
echo -e "\n${YELLOW}Configuring CMake for Windows...${NC}"
cd "${BUILD_DIR}"

cmake \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_API_SERVER=OFF \
    -DENABLE_GPU=OFF \
    "${PROJECT_DIR}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ CMake configuration successful${NC}"
else
    echo -e "${RED}✗ CMake configuration failed${NC}"
    exit 1
fi

# Build
echo -e "\n${YELLOW}Building Windows executables...${NC}"
echo -e "${BLUE}Using ${JOBS} parallel jobs${NC}"

cmake --build . --config "${BUILD_TYPE}" -j "${JOBS}"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Build failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

# List built executables
echo -e "\n${YELLOW}Built Windows executables:${NC}"
find "${BUILD_DIR}" -name "*.exe" -type f | while read exe; do
    SIZE=$(du -h "$exe" | cut -f1)
    BASENAME=$(basename "$exe")
    echo -e "  ${GREEN}✓${NC} ${BASENAME} (${SIZE})"
done

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Build Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Build directory: ${BUILD_DIR}"
echo -e "  Build type: ${BUILD_TYPE}"
echo -e "  Toolchain: MinGW-w64"
echo -e "  Target: Windows x86_64"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Test executables: ${GREEN}./scripts/package_windows.sh${NC}"
echo -e "  2. Copy to Windows and run"
echo -e "  3. See: ${GREEN}docs/guides/windows-cross-compilation.md${NC}"
echo -e "${BLUE}========================================${NC}\n"
