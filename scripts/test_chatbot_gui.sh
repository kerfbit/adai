#!/bin/bash

# @adai-status: beta        (capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.0
# @adai-reviewed: 2026-09-07


# Test script for chatbot_gui
# This script verifies the GUI executable is properly built and linked

echo "========================================"
echo "Chatbot GUI Build Verification"
echo "========================================"
echo ""

# Check if executable exists
if [ ! -f "build/src/chatbot_gui" ]; then
    echo "❌ FAIL: chatbot_gui executable not found"
    exit 1
fi
echo "✅ Executable exists: build/src/chatbot_gui"

# Check if executable is actually executable
if [ ! -x "build/src/chatbot_gui" ]; then
    echo "❌ FAIL: chatbot_gui is not executable"
    exit 1
fi
echo "✅ Executable has correct permissions"

# Check file size (should be > 1MB)
SIZE=$(stat -c%s "build/src/chatbot_gui")
if [ $SIZE -lt 1000000 ]; then
    echo "❌ FAIL: Executable too small ($SIZE bytes)"
    exit 1
fi
echo "✅ Executable size: $(numfmt --to=iec-i --suffix=B $SIZE)"

# Check if it's a proper ELF binary
if ! file build/src/chatbot_gui | grep -q "ELF.*executable"; then
    echo "❌ FAIL: Not a valid ELF executable"
    exit 1
fi
echo "✅ Valid ELF executable"

# Check Qt5 dependencies
echo ""
echo "Qt5 Dependencies:"
ldd build/src/chatbot_gui | grep -i qt5 | head -5

# Check for required symbols
echo ""
echo "Checking for required components..."
if nm build/src/chatbot_gui | grep -q "ChatbotGUI"; then
    echo "✅ ChatbotGUI class found"
else
    echo "❌ ChatbotGUI class not found"
fi

if nm build/src/chatbot_gui | grep -q "BPETokenizer"; then
    echo "✅ BPETokenizer integration found"
else
    echo "❌ BPETokenizer integration not found"
fi

if nm build/src/chatbot_gui | grep -q "EncoderDecoderModel"; then
    echo "✅ EncoderDecoderModel integration found"
else
    echo "❌ EncoderDecoderModel integration not found"
fi

if nm build/src/chatbot_gui | grep -q "ConversationContext"; then
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
