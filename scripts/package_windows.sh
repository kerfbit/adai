#!/bin/bash

# @adai-status: beta        (capped by TD-043 — see TECHNICAL_DEBT.md)
# @adai-version: 0.8.0
# @adai-reviewed: 2026-09-07

# Package Windows executables with all dependencies for distribution
# Creates a portable Windows package that can be copied and run on any Windows system

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
PACKAGE_DIR="${PROJECT_DIR}/dist-windows"
PACKAGE_NAME="adai-chatbot-windows-x64"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ADAI Windows Package Creator${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if build exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${RED}ERROR: Build directory not found!${NC}"
    echo -e "${YELLOW}Run ./scripts/build_windows.sh first${NC}"
    exit 1
fi

# Check if executables exist
if [ ! -f "${BUILD_DIR}/src/chatbot.exe" ]; then
    echo -e "${RED}ERROR: chatbot.exe not found!${NC}"
    echo -e "${YELLOW}Run ./scripts/build_windows.sh first${NC}"
    exit 1
fi

# Clean previous package
echo -e "\n${YELLOW}Cleaning previous package...${NC}"
rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/${PACKAGE_NAME}"
echo -e "${GREEN}✓ Package directory created${NC}"

# Copy executables
echo -e "\n${YELLOW}Copying Windows executables...${NC}"
cp "${BUILD_DIR}/src/chatbot.exe" "${PACKAGE_DIR}/${PACKAGE_NAME}/" 2>/dev/null || true
cp "${BUILD_DIR}/src/chatbot_trainer.exe" "${PACKAGE_DIR}/${PACKAGE_NAME}/" 2>/dev/null || true
cp "${BUILD_DIR}/src/chatbot_api_server.exe" "${PACKAGE_DIR}/${PACKAGE_NAME}/" 2>/dev/null || true

# List copied executables
find "${PACKAGE_DIR}/${PACKAGE_NAME}" -name "*.exe" -type f | while read exe; do
    SIZE=$(du -h "$exe" | cut -f1)
    BASENAME=$(basename "$exe")
    echo -e "  ${GREEN}✓${NC} ${BASENAME} (${SIZE})"
done

# Copy required DLLs (if any dynamic libraries are needed)
echo -e "\n${YELLOW}Checking for required DLLs...${NC}"
MINGW_PREFIX="/usr/x86_64-w64-mingw32"
if [ -d "${MINGW_PREFIX}/bin" ]; then
    # Note: With static linking, these shouldn't be needed, but we check anyway
    for dll in libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll; do
        if [ -f "${MINGW_PREFIX}/bin/${dll}" ]; then
            echo -e "  ${YELLOW}Found ${dll} (not needed with static linking)${NC}"
        fi
    done
    echo -e "${GREEN}✓ Using static linking - no DLLs needed${NC}"
else
    echo -e "${YELLOW}MinGW runtime directory not found (DLLs not copied)${NC}"
fi

# Copy sample files
echo -e "\n${YELLOW}Copying sample files and documentation...${NC}"
if [ -f "${PROJECT_DIR}/vocab.txt" ]; then
    cp "${PROJECT_DIR}/vocab.txt" "${PACKAGE_DIR}/${PACKAGE_NAME}/"
    echo -e "  ${GREEN}✓${NC} vocab.txt"
fi

if [ -f "${PROJECT_DIR}/sample_training_data.txt" ]; then
    cp "${PROJECT_DIR}/sample_training_data.txt" "${PACKAGE_DIR}/${PACKAGE_NAME}/"
    echo -e "  ${GREEN}✓${NC} sample_training_data.txt"
fi

# Copy any model files that exist
for model_file in chatbot_model.bin chatbot_model.bin.config chatbot_model.bin.decoder chatbot_model.bin.metadata chatbot_model.bin.vocab; do
    if [ -f "${PROJECT_DIR}/${model_file}" ]; then
        cp "${PROJECT_DIR}/${model_file}" "${PACKAGE_DIR}/${PACKAGE_NAME}/"
        echo -e "  ${GREEN}✓${NC} ${model_file}"
    fi
done

# Create README for Windows users
echo -e "\n${YELLOW}Creating Windows README...${NC}"
cat > "${PACKAGE_DIR}/${PACKAGE_NAME}/README.txt" << 'EOF'
ADAI Chatbot - Windows Edition
================================

This package contains Windows native executables for the ADAI transformer-based chatbot.

Contents:
---------
  chatbot.exe          - Interactive chatbot CLI
  chatbot_trainer.exe  - Model training tool
  vocab.txt            - Vocabulary file
  sample_training_data.txt - Sample training data
  chatbot_model.bin*   - Pre-trained model files (if included)

Quick Start:
------------
1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run the chatbot:

   chatbot.exe

   Or with custom files:
   
   chatbot.exe vocab.txt chatbot_model.bin

Training a Model:
-----------------
To train a new model:

   chatbot_trainer.exe --data sample_training_data.txt --vocab vocab.txt --output my_model.bin

For more options:

   chatbot_trainer.exe --help

Interactive Commands:
---------------------
Once the chatbot is running, you can use these commands:

  /help         - Show available commands
  /save         - Save conversation
  /load         - Load conversation
  /stats        - Show conversation statistics
  /settings     - Show current settings
  /set <param>  - Change generation parameter
  /exit         - Exit the chatbot

Generation Strategies:
----------------------
  greedy   - Always pick most likely token
  beam     - Beam search (multiple hypotheses)
  sampling - Random sampling
  top-k    - Sample from top K tokens
  nucleus  - Sample from top P probability mass

Examples:
---------
Change strategy:
  /set strategy greedy

Adjust temperature:
  /set temperature 0.7

Set max response length:
  /set length 150

Requirements:
-------------
- Windows 7 or later (64-bit)
- No additional dependencies required (statically linked)

Troubleshooting:
----------------
If the executable won't run:
1. Make sure you're using 64-bit Windows
2. Try running from Command Prompt to see error messages
3. Check that all .bin files are in the same directory

For more information, visit:
  https://github.com/rjv717/adai

Built with MinGW-w64 cross-compiler
EOF

echo -e "${GREEN}✓ README.txt created${NC}"

# Create a batch file launcher for convenience
echo -e "\n${YELLOW}Creating launcher script...${NC}"
cat > "${PACKAGE_DIR}/${PACKAGE_NAME}/run_chatbot.bat" << 'EOF'
@echo off
echo ========================================
echo ADAI Chatbot Launcher
echo ========================================
echo.

REM Check if chatbot.exe exists
if not exist chatbot.exe (
    echo ERROR: chatbot.exe not found!
    echo Please make sure you're running this from the correct directory.
    pause
    exit /b 1
)

REM Check if vocab file exists
if not exist vocab.txt (
    echo WARNING: vocab.txt not found!
    echo The chatbot may not work without a vocabulary file.
    echo.
)

REM Check if model exists
if not exist chatbot_model.bin (
    echo WARNING: chatbot_model.bin not found!
    echo You may need to train a model first using chatbot_trainer.exe
    echo.
)

echo Starting chatbot...
echo.
chatbot.exe

pause
EOF

echo -e "${GREEN}✓ run_chatbot.bat created${NC}"

# Create archive
echo -e "\n${YELLOW}Creating ZIP archive...${NC}"
cd "${PACKAGE_DIR}"

if command -v zip &> /dev/null; then
    zip -r "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
    ARCHIVE_SIZE=$(du -h "${PACKAGE_NAME}.zip" | cut -f1)
    echo -e "${GREEN}✓ Created ${PACKAGE_NAME}.zip (${ARCHIVE_SIZE})${NC}"
else
    echo -e "${YELLOW}ZIP not found, skipping archive creation${NC}"
    echo -e "${YELLOW}Install with: sudo apt-get install zip${NC}"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Package Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Package directory: ${PACKAGE_DIR}/${PACKAGE_NAME}"
echo -e "  Package contents:"

du -sh "${PACKAGE_DIR}/${PACKAGE_NAME}"/* | while read size file; do
    BASENAME=$(basename "$file")
    echo -e "    ${size}\t${BASENAME}"
done

echo -e "\n${YELLOW}Ready for Windows deployment!${NC}"
echo -e "\n${GREEN}To distribute:${NC}"
echo -e "  1. Copy ${PACKAGE_NAME}.zip to a Windows machine"
echo -e "  2. Extract the archive"
echo -e "  3. Double-click run_chatbot.bat or run chatbot.exe from command line"
echo -e "${BLUE}========================================${NC}\n"
