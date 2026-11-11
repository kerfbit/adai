#!/bin/bash

# @adai-status: beta        (tests: tests/scripts/verify_gui_parallel_test.sh)
# @adai-version: 0.7.0
# @adai-reviewed: 2026-11-10

# Chatbot GUI build verification script.
# TD-260: chatbot_gui is a client of chatbot_api_server, so it no longer links the model libraries or
# OpenMP; inference (and its parallel processing) runs in the server. The name is kept for its callers.

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Chatbot GUI Build Verification                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# TD-044: bare relative paths ("./build/src/...", "./build/CMakeCache.txt"
# below) with no anchor to the script's own location — only worked when the
# caller's CWD happened to already be the repo root. Resolved via
# SCRIPT_DIR/REPO_ROOT, matching every sibling launcher script, then cd to
# REPO_ROOT so every relative path below resolves the same way regardless
# of caller CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}" || exit 1

# TD-044: even once anchored to REPO_ROOT, "build/src/..." was still wrong
# for any standard, documented build — every preset in CMakePresets.json
# builds into "build/<preset>/", never bare "build/". Auto-detects the
# first preset build directory that actually has the GUI binary, falling
# back to bare "build/" for a non-preset configure.
find_build_dir() {
    local marker="$1"
    local dir
    for dir in "${REPO_ROOT}/build/debug" "${REPO_ROOT}/build/release" \
               "${REPO_ROOT}/build/portable" "${REPO_ROOT}/build/relwithdebinfo" \
               "${REPO_ROOT}/build/gpu" "${REPO_ROOT}/build/sycl" \
               "${REPO_ROOT}/build"; do
        if [ -f "${dir}/${marker}" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}
BUILD_DIR="$(find_build_dir src/chatbot_gui_binary || echo "${REPO_ROOT}/build")"
GUI_BINARY="${BUILD_DIR}/src/chatbot_gui_binary"
LAUNCHER="${BUILD_DIR}/src/chatbot_gui"

echo -e "${BLUE}1. Checking Binary Location...${NC}"
if [ -f "$GUI_BINARY" ]; then
    echo -e "${GREEN}   ✓ chatbot_gui_binary found${NC}"
    ls -lh "$GUI_BINARY"
else
    echo "   ✗ chatbot_gui_binary not found"
    exit 1
fi
echo ""

echo -e "${BLUE}2. Checking Wrapper...${NC}"
if [ -f "$LAUNCHER" ]; then
    echo -e "${GREEN}   ✓ chatbot_gui wrapper found${NC}"
    ls -lh "$LAUNCHER"
else
    echo "   ✗ chatbot_gui wrapper not found"
fi
echo ""

echo -e "${BLUE}3. Verifying Qt Linkage...${NC}"
if ldd "$GUI_BINARY" | grep -qE "Qt[56]"; then
    echo -e "${GREEN}   ✓ Qt libraries are linked${NC}"
    ldd "$GUI_BINARY" | grep -E "Qt[56]" | head -3
else
    echo "   ✗ Qt not linked"
fi
echo ""

echo -e "${BLUE}4. Checking It Is a Server Client...${NC}"
# TD-260: the model runs in chatbot_api_server; the GUI only sends it messages over HTTP.
if ldd "$GUI_BINARY" | grep -q libgomp; then
    echo -e "${YELLOW}   ⚠ OpenMP is linked: this looks like a build from before the GUI became a client${NC}"
else
    echo -e "${GREEN}   ✓ No inference libraries: replies come from chatbot_api_server${NC}"
fi
echo ""

echo -e "${BLUE}5. Testing GUI Executable...${NC}"
if "$LAUNCHER" --help 2>&1 | grep -q "Usage:"; then
    echo -e "${GREEN}   ✓ GUI executable runs and shows help${NC}"
else
    echo -e "${YELLOW}   ⚠ GUI help didn't run${NC}"
fi
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Verification Complete                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✨ Chatbot GUI is built.${NC}"
echo ""
echo "To run (start chatbot_api_server first; its URL defaults to http://localhost:8080):"
echo "  ${LAUNCHER} http://localhost:8080"
echo ""
echo "Or use the convenience script:"
echo "  ./scripts/run_chatbot_gui.sh [server_url]"
