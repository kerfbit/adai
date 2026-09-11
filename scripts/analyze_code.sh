#!/bin/bash

# @adai-status: stable
# @adai-version: 1.0.0
# @adai-reviewed: 2026-09-10

#
# Run static analysis on C++ source files using clang-tidy
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              ADAI Static Analysis Tool                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if clang-tidy is installed
if ! command -v clang-tidy &> /dev/null; then
    echo "❌ Error: clang-tidy not found"
    echo "   Install with: sudo apt-get install clang-tidy"
    exit 1
fi

echo "📋 Using clang-tidy: $(clang-tidy --version | head -1)"
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check if compile_commands.json exists
if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
    echo "⚠️  Warning: compile_commands.json not found"
    echo "   Generating it now..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    echo ""
fi

# Parse arguments
FILES_TO_CHECK=""
if [ $# -gt 0 ]; then
    FILES_TO_CHECK="$@"
    echo "🔍 Analyzing specified files..."
else
    echo "🔍 Finding C++ files in src/..."
    FILES_TO_CHECK=$(find "$PROJECT_ROOT/src" \
        -type f \( -name "*.cpp" -o -name "*.hpp" \) \
        ! -path "*/build/*" \
        ! -path "*/legacy/*" \
        ! -name "*Example.cpp" \
        2>/dev/null)
fi

# wc -w (not -l): when files come from "$@" (explicit args), FILES_TO_CHECK
# is one space-separated line, so -l always reported 1 file regardless of how
# many were actually passed. -w counts correctly for both that case and the
# find-populated (one path per line) case below.
FILE_COUNT=$(echo "$FILES_TO_CHECK" | wc -w)
echo "   Found $FILE_COUNT files to analyze"
echo ""

# Run clang-tidy
echo "🔬 Running static analysis..."
echo ""

ISSUES_FOUND=0
for file in $FILES_TO_CHECK; do
    RELATIVE_PATH=$(realpath --relative-to="$PROJECT_ROOT" "$file" 2>/dev/null || echo "$file")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📄 Analyzing: $RELATIVE_PATH"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if clang-tidy -p "$BUILD_DIR" "$file" 2>&1 | grep -q "warning:"; then
        clang-tidy -p "$BUILD_DIR" "$file" 2>&1
        # `set -e` is active in this script: `((ISSUES_FOUND++))` is a
        # post-increment, so its exit status is the PRE-increment value —
        # 0 the very first time a file with issues is found — and `set -e`
        # treats that as a failure, killing the whole script right there,
        # before any later file is ever analyzed or the summary is printed.
        # Plain arithmetic assignment has no such exit-status trap.
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    else
        echo "✅ No issues found"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Analysis Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Files analyzed: $FILE_COUNT"
echo "   Files with issues: $ISSUES_FOUND"
echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✅ No issues found!"
else
    echo "⚠️  Found issues in $ISSUES_FOUND file(s)"
    echo "💡 Review the warnings above and fix as needed"
fi
