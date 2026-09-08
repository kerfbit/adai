#!/bin/bash

# @adai-status: beta        (capped by TD-044 — see TECHNICAL_DEBT.md)
# @adai-version: 0.7.0
# @adai-reviewed: 2026-09-07


# Comprehensive test suite for chatbot_gui
# Tests build, dependencies, integration, and code quality

echo "╔═══════════════════════════════════════════════════════╗"
echo "║      Chatbot GUI - Comprehensive Test Suite          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Helper functions
pass() {
    echo "✅ PASS: $1"
    ((PASS_COUNT++))
}

fail() {
    echo "❌ FAIL: $1"
    ((FAIL_COUNT++))
}

warn() {
    echo "⚠️  WARN: $1"
    ((WARN_COUNT++))
}

info() {
    echo "ℹ️  INFO: $1"
}

section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Test 1: Source Files
section "Test 1: Source Files"

if [ -f "src/ChatbotGUI.hpp" ]; then
    pass "ChatbotGUI.hpp exists"
else
    fail "ChatbotGUI.hpp not found"
fi

if [ -f "src/ChatbotGUI.cpp" ]; then
    pass "ChatbotGUI.cpp exists"
else
    fail "ChatbotGUI.cpp not found"
fi

if [ -f "src/ChatbotGUI_main.cpp" ]; then
    pass "ChatbotGUI_main.cpp exists"
else
    fail "ChatbotGUI_main.cpp not found"
fi

# Test 2: Build System
section "Test 2: Build System"

if grep -q "BUILD_GUI" src/CMakeLists.txt; then
    pass "BUILD_GUI option in CMakeLists.txt"
else
    fail "BUILD_GUI option not found in CMakeLists.txt"
fi

if grep -q "chatbot_gui" src/CMakeLists.txt; then
    pass "chatbot_gui target in CMakeLists.txt"
else
    fail "chatbot_gui target not found in CMakeLists.txt"
fi

if grep -q "Qt5\\|Qt6" src/CMakeLists.txt; then
    pass "Qt detection in CMakeLists.txt"
else
    fail "Qt detection not found in CMakeLists.txt"
fi

# Test 3: Executable
section "Test 3: Executable"

if [ -f "build/src/chatbot_gui" ]; then
    pass "Executable built successfully"
    
    SIZE=$(stat -c%s "build/src/chatbot_gui")
    if [ $SIZE -gt 1000000 ]; then
        pass "Executable size reasonable ($(numfmt --to=iec-i --suffix=B $SIZE))"
    else
        fail "Executable too small ($SIZE bytes)"
    fi
    
    if [ -x "build/src/chatbot_gui" ]; then
        pass "Executable has correct permissions"
    else
        fail "Executable not executable"
    fi
else
    fail "Executable not found"
fi

# Test 4: Dependencies
section "Test 4: Qt Dependencies"

if command -v qmake >/dev/null 2>&1; then
    QT_VERSION=$(qmake -query QT_VERSION)
    pass "Qt installed: version $QT_VERSION"
else
    warn "qmake not found (Qt may still be available)"
fi

if ldd build/src/chatbot_gui 2>/dev/null | grep -q libQt5; then
    pass "Linked against Qt5"
    QT_WIDGETS=$(ldd build/src/chatbot_gui | grep -c Qt5Widgets)
    QT_GUI=$(ldd build/src/chatbot_gui | grep -c Qt5Gui)
    QT_CORE=$(ldd build/src/chatbot_gui | grep -c Qt5Core)
    
    if [ $QT_WIDGETS -gt 0 ]; then
        pass "Qt5Widgets linked"
    else
        fail "Qt5Widgets not linked"
    fi
    
    if [ $QT_GUI -gt 0 ]; then
        pass "Qt5Gui linked"
    else
        fail "Qt5Gui not linked"
    fi
    
    if [ $QT_CORE -gt 0 ]; then
        pass "Qt5Core linked"
    else
        fail "Qt5Core not linked"
    fi
elif ldd build/src/chatbot_gui 2>/dev/null | grep -q libQt6; then
    pass "Linked against Qt6"
else
    fail "No Qt libraries linked"
fi

# Test 5: Code Integration
section "Test 5: Chatbot Component Integration"

if nm build/src/chatbot_gui 2>/dev/null | grep -q "ChatbotGUI"; then
    pass "ChatbotGUI class integrated"
else
    fail "ChatbotGUI class not found in binary"
fi

if nm build/src/chatbot_gui 2>/dev/null | grep -q "BPETokenizer"; then
    pass "BPETokenizer integrated"
else
    fail "BPETokenizer not integrated"
fi

if nm build/src/chatbot_gui 2>/dev/null | grep -q "EncoderDecoderModel"; then
    pass "EncoderDecoderModel integrated"
else
    fail "EncoderDecoderModel not integrated"
fi

if nm build/src/chatbot_gui 2>/dev/null | grep -q "ConversationContext"; then
    pass "ConversationContext integrated"
else
    fail "ConversationContext not integrated"
fi

# Test 6: Qt Signals/Slots
section "Test 6: Qt Meta-Object System"

if nm build/src/chatbot_gui 2>/dev/null | grep -q "onSendMessage"; then
    pass "onSendMessage slot found"
else
    fail "onSendMessage slot not found"
fi

if nm build/src/chatbot_gui 2>/dev/null | grep -q "onClearConversation"; then
    pass "onClearConversation slot found"
else
    fail "onClearConversation slot not found"
fi

if nm build/src/chatbot_gui 2>/dev/null | grep -q "onStrategyChanged"; then
    pass "onStrategyChanged slot found"
else
    fail "onStrategyChanged slot not found"
fi

# Test 7: Required Files
section "Test 7: Runtime Requirements"

if [ -f "vocab.txt" ]; then
    VOCAB_SIZE=$(wc -l < vocab.txt)
    pass "vocab.txt exists ($VOCAB_SIZE entries)"
else
    warn "vocab.txt not found (required for running)"
fi

if [ -f "chatbot_model.bin.config" ]; then
    pass "Model configuration exists"
    
    # Check for associated model files
    MODEL_PREFIX="chatbot_model.bin"
    if ls ${MODEL_PREFIX}*.decoder >/dev/null 2>&1; then
        pass "Model decoder files found"
    else
        warn "Model decoder files not found"
    fi
else
    warn "Model configuration not found (will use random init)"
fi

# Test 8: Documentation
section "Test 8: Documentation"

if [ -f "docs/guides/chatbot-gui-guide.md" ]; then
    pass "GUI guide documentation exists"
else
    fail "GUI guide documentation not found"
fi

if [ -f "CHATBOT_GUI_README.md" ]; then
    pass "Quick reference README exists"
else
    fail "Quick reference README not found"
fi

# Test 9: Code Quality
section "Test 9: Code Quality Checks"

# Check for common issues in source files
if grep -q "Q_OBJECT" src/ChatbotGUI.hpp; then
    pass "Q_OBJECT macro present (required for MOC)"
else
    fail "Q_OBJECT macro missing in ChatbotGUI.hpp"
fi

if grep -q "slots:" src/ChatbotGUI.hpp; then
    pass "Slot declarations found"
else
    fail "No slot declarations in header"
fi

if grep -q "QMainWindow" src/ChatbotGUI.hpp; then
    pass "Inherits from QMainWindow"
else
    fail "Does not inherit from QMainWindow"
fi

# Check for proper includes
if grep -q "#include.*QApplication" src/ChatbotGUI_main.cpp; then
    pass "QApplication included in main"
else
    fail "QApplication not included in main"
fi

# Test 10: Build Artifacts
section "Test 10: Build Artifacts"

if [ -d "build/src/chatbot_gui_autogen" ]; then
    pass "Qt MOC autogen directory exists"
    
    if ls build/src/chatbot_gui_autogen/mocs_*.cpp >/dev/null 2>&1; then
        pass "MOC files generated"
    else
        warn "MOC files not found"
    fi
else
    warn "Qt autogen directory not found"
fi

# Summary
section "Test Summary"

TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))

echo ""
echo "Results:"
echo "  ✅ Passed:  $PASS_COUNT"
echo "  ❌ Failed:  $FAIL_COUNT"
echo "  ⚠️  Warnings: $WARN_COUNT"
echo "  ━━━━━━━━━━━━━━━━━"
echo "  Total:     $TOTAL"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║              ✅ ALL TESTS PASSED ✅                   ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "The chatbot_gui is ready to use!"
    echo ""
    echo "To run (requires graphical environment):"
    echo "  cd /home/rodney/Repos/adai"
    echo "  ./build/src/chatbot_gui"
    echo ""
    exit 0
else
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║           ❌ SOME TESTS FAILED ❌                     ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""
    echo "Please review the failures above."
    echo ""
    exit 1
fi
