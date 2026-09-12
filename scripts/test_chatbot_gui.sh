#!/bin/bash

# @adai-status: beta        (TD-044 resolved — real test suite added, see tests/scripts/test_chatbot_gui_test.sh)
# @adai-version: 0.7.2
# @adai-reviewed: 2026-09-11


# Test script for chatbot_gui
# This script verifies the GUI executable is properly built and linked

echo "========================================"
echo "Chatbot GUI Build Verification"
echo "========================================"
echo ""

# TD-044: bare relative paths ("build/src/...", "vocab.txt" further down)
# with no anchor to the script's own location — only worked when the
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
# first preset build directory that actually has the binary, falling back
# to bare "build/" for a non-preset configure.
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

BUILD_DIR="$(find_build_dir src/chatbot_gui)"
if [ -z "$BUILD_DIR" ]; then
    echo "❌ FAIL: chatbot_gui executable not found in any build/<preset>/src/ directory"
    exit 1
fi
LAUNCHER="${BUILD_DIR}/src/chatbot_gui"

# chatbot_gui is a thin exec() launcher (ChatbotGUI_wrapper.cpp) that execs
# chatbot_gui_binary — the actual Qt GUI logic (ChatbotGUI, BPETokenizer,
# EncoderDecoderModel, ConversationContext) is compiled into
# chatbot_gui_binary, not the wrapper. Every check below that inspects size,
# Qt5 linkage, or symbol content was previously run against the wrapper,
# which contains none of those things by design — confirmed empirically
# (0 matching symbols in chatbot_gui vs. 743 in chatbot_gui_binary; wrapper
# has zero Qt5 linkage and is well under the 1MB size threshold, so the size
# check alone always failed with `exit 1` before this script could report
# anything else). TD-124.
GUI_BINARY="${BUILD_DIR}/src/chatbot_gui_binary"

# Check if launcher exists
if [ ! -f "$LAUNCHER" ]; then
    echo "❌ FAIL: chatbot_gui executable not found"
    exit 1
fi
echo "✅ Executable exists: $LAUNCHER"

# Check if launcher is actually executable
if [ ! -x "$LAUNCHER" ]; then
    echo "❌ FAIL: chatbot_gui is not executable"
    exit 1
fi
echo "✅ Executable has correct permissions"

# Check the real GUI binary exists
if [ ! -f "$GUI_BINARY" ]; then
    echo "❌ FAIL: chatbot_gui_binary not found"
    exit 1
fi
echo "✅ Executable exists: $GUI_BINARY"

# Check file size (should be > 1MB)
SIZE=$(stat -c%s "$GUI_BINARY")
if [ $SIZE -lt 1000000 ]; then
    echo "❌ FAIL: Executable too small ($SIZE bytes)"
    exit 1
fi
echo "✅ Executable size: $(numfmt --to=iec-i --suffix=B $SIZE)"

# Check if it's a proper ELF binary
if ! file "$GUI_BINARY" | grep -q "ELF.*executable"; then
    echo "❌ FAIL: Not a valid ELF executable"
    exit 1
fi
echo "✅ Valid ELF executable"

# Check Qt5 dependencies
echo ""
echo "Qt5 Dependencies:"
ldd "$GUI_BINARY" | grep -i qt5 | head -5

# Check for required symbols
#
# Unlike the existence/permissions/size/ELF checks above (which exit 1
# immediately on failure), these four never fed into the final verdict —
# printing "❌ ... not found" here previously had no effect on anything;
# the script went on to unconditionally print "Build Verification: SUCCESS"
# and exit 0 regardless of how many of these actually failed. Track them in
# SYMBOL_CHECK_FAILED so a genuinely broken/stripped binary is reported as
# a failure instead of a false "SUCCESS".
echo ""
echo "Checking for required components..."
SYMBOL_CHECK_FAILED=false
if nm "$GUI_BINARY" | grep -q "ChatbotGUI"; then
    echo "✅ ChatbotGUI class found"
else
    echo "❌ ChatbotGUI class not found"
    SYMBOL_CHECK_FAILED=true
fi

if nm "$GUI_BINARY" | grep -q "BPETokenizer"; then
    echo "✅ BPETokenizer integration found"
else
    echo "❌ BPETokenizer integration not found"
    SYMBOL_CHECK_FAILED=true
fi

if nm "$GUI_BINARY" | grep -q "EncoderDecoderModel"; then
    echo "✅ EncoderDecoderModel integration found"
else
    echo "❌ EncoderDecoderModel integration not found"
    SYMBOL_CHECK_FAILED=true
fi

if nm "$GUI_BINARY" | grep -q "ConversationContext"; then
    echo "✅ ConversationContext integration found"
else
    echo "❌ ConversationContext integration not found"
    SYMBOL_CHECK_FAILED=true
fi

# Check required files
echo ""
echo "Checking required files:"
if [ -f "vocab.txt" ]; then
    VOCAB_SIZE=$(wc -l < vocab.txt)
    echo "✅ vocab.txt found ($VOCAB_SIZE entries)"
else
    echo "⚠️  vocab.txt not found (required for running)"
fi

if [ -f "chatbot_model.bin.config" ]; then
    echo "✅ Model configuration found"
else
    echo "⚠️  chatbot_model.bin.config not found (will use random init)"
fi

echo ""
echo "========================================"
if [ "$SYMBOL_CHECK_FAILED" = true ]; then
    echo "Build Verification: FAILED ❌"
    echo "========================================"
    echo ""
    echo "One or more required components were not found in $GUI_BINARY."
    echo "See the ❌ lines above."
    echo ""
    exit 1
fi
echo "Build Verification: SUCCESS ✅"
echo "========================================"
echo ""
echo "The chatbot_gui has been successfully built!"
echo ""
echo "Note: To run the GUI, you need:"
echo "  1. An X11 display (graphical environment)"
echo "  2. vocab.txt file"
echo "  3. Optionally: trained model files"
echo ""
echo "Usage:"
echo "  ${LAUNCHER}"
echo "  ${LAUNCHER} vocab.txt"
echo "  ${LAUNCHER} vocab.txt model_prefix"
echo ""
echo "For headless testing, the GUI cannot be fully tested"
echo "but the build is verified to be correct."
echo ""
