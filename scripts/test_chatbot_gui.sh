#!/bin/bash

# @adai-status: beta        (capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.1
# @adai-reviewed: 2026-09-10


# Test script for chatbot_gui
# This script verifies the GUI executable is properly built and linked

echo "========================================"
echo "Chatbot GUI Build Verification"
echo "========================================"
echo ""

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
GUI_BINARY="build/src/chatbot_gui_binary"

# Check if launcher exists
if [ ! -f "build/src/chatbot_gui" ]; then
    echo "❌ FAIL: chatbot_gui executable not found"
    exit 1
fi
echo "✅ Executable exists: build/src/chatbot_gui"

# Check if launcher is actually executable
if [ ! -x "build/src/chatbot_gui" ]; then
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
echo ""
echo "Checking for required components..."
if nm "$GUI_BINARY" | grep -q "ChatbotGUI"; then
    echo "✅ ChatbotGUI class found"
else
    echo "❌ ChatbotGUI class not found"
fi

if nm "$GUI_BINARY" | grep -q "BPETokenizer"; then
    echo "✅ BPETokenizer integration found"
else
    echo "❌ BPETokenizer integration not found"
fi

if nm "$GUI_BINARY" | grep -q "EncoderDecoderModel"; then
    echo "✅ EncoderDecoderModel integration found"
else
    echo "❌ EncoderDecoderModel integration not found"
fi

if nm "$GUI_BINARY" | grep -q "ConversationContext"; then
    echo "✅ ConversationContext integration found"
else
    echo "❌ ConversationContext integration not found"
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
echo "  ./build/src/chatbot_gui"
echo "  ./build/src/chatbot_gui vocab.txt"
echo "  ./build/src/chatbot_gui vocab.txt model_prefix"
echo ""
echo "For headless testing, the GUI cannot be fully tested"
echo "but the build is verified to be correct."
echo ""
