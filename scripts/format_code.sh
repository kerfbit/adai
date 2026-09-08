#!/bin/bash

# @adai-status: stable
# @adai-version: 1.0.0
# @adai-reviewed: 2026-09-07

#
# Format all C++ source files in the project using clang-format
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              ADAI Code Formatting Tool                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if clang-format is installed (prefer clang-format-18 for version consistency with CI)
CLANG_FORMAT=""
if command -v clang-format-18 &> /dev/null; then
    CLANG_FORMAT="clang-format-18"
elif command -v clang-format &> /dev/null; then
    CLANG_FORMAT="clang-format"
else
    echo "❌ Error: clang-format not found"
    echo "   Install with: sudo apt-get install clang-format-18"
    exit 1
fi

echo "📋 Using clang-format: $($CLANG_FORMAT --version | head -1)"
echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Find all C++ source files
echo "🔍 Finding C++ files..."
mapfile -t CPP_FILES < <(find "$PROJECT_ROOT/src" "$PROJECT_ROOT/tests" \
    -type f \( -name "*.cpp" -o -name "*.hpp" \) \
    ! -path "*/build/*" \
    ! -path "*/legacy/*" \
    ! -path "*/.vscode/*" \
    2>/dev/null)

FILE_COUNT=${#CPP_FILES[@]}
echo "   Found $FILE_COUNT files to format"
echo ""

# Format files
echo "✨ Formatting files..."
FORMATTED=0
ERRORS=0
for file in "${CPP_FILES[@]}"; do
    RELATIVE_PATH=$(realpath --relative-to="$PROJECT_ROOT" "$file")
    echo "   Formatting: $RELATIVE_PATH"
    if $CLANG_FORMAT -i "$file" 2>&1; then
        ((FORMATTED++))
    else
        echo "   ⚠️  Error formatting: $RELATIVE_PATH"
        ((ERRORS++))
    fi
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✅ Successfully formatted $FORMATTED files"
else
    echo "⚠️  Formatted $FORMATTED files with $ERRORS errors"
fi
echo ""
echo "💡 Tip: Run 'git diff' to see the formatting changes"
echo "💡 Tip: Add pre-commit hook to format automatically"
