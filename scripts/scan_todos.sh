#!/bin/bash

# Script to scan for TODO comments in the codebase
# This helps ensure all TODOs are tracked in TECHNICAL_DEBT.md

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Scanning for TODO comments in source files...${NC}"
echo ""

# Directories to scan
SCAN_DIRS="src/ tests/ include/"
FILE_PATTERNS="*.cpp *.hpp"

# Find all TODO comments
TODO_RESULTS=$(mktemp)
for pattern in $FILE_PATTERNS; do
    grep -rn "TODO:" --include="$pattern" $SCAN_DIRS >> "$TODO_RESULTS" 2>/dev/null || true
done

# Count TODOs
TODO_COUNT=$(wc -l < "$TODO_RESULTS")

if [ $TODO_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ No TODO comments found${NC}"
    rm "$TODO_RESULTS"
    exit 0
fi

echo -e "${YELLOW}Found $TODO_COUNT TODO comment(s):${NC}"
echo ""
echo "==================== TODO List ===================="
cat "$TODO_RESULTS"
echo "===================================================="
echo ""

# Check if TODOs are tracked in TECHNICAL_DEBT.md
TECH_DEBT_FILE="docs/development/guides/TECHNICAL_DEBT.md"

if [ -f "$TECH_DEBT_FILE" ]; then
    echo -e "${BLUE}📋 Checking if TODOs are tracked in $TECH_DEBT_FILE...${NC}"
    echo ""
    
    NOT_TRACKED=0
    while IFS= read -r line; do
        # Extract file path and line number
        TODO_FILE=$(echo "$line" | cut -d':' -f1)
        TODO_LINE=$(echo "$line" | cut -d':' -f2)
        TODO_TEXT=$(echo "$line" | cut -d':' -f3-)
        
        # Check if this file:line is mentioned in TECHNICAL_DEBT.md
        if grep -q "$TODO_FILE:$TODO_LINE" "$TECH_DEBT_FILE" 2>/dev/null || \
           grep -q "$TODO_FILE.*line $TODO_LINE" "$TECH_DEBT_FILE" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $TODO_FILE:$TODO_LINE is tracked"
        else
            echo -e "${RED}✗${NC} $TODO_FILE:$TODO_LINE is NOT tracked"
            NOT_TRACKED=$((NOT_TRACKED + 1))
        fi
    done < "$TODO_RESULTS"
    
    echo ""
    
    if [ $NOT_TRACKED -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: $NOT_TRACKED TODO(s) are not tracked in $TECH_DEBT_FILE${NC}"
        echo ""
        echo "Please either:"
        echo "  1. Add an entry to $TECH_DEBT_FILE"
        echo "  2. Create a GitHub issue and reference it in the TODO"
        echo "  3. Remove the TODO if it's no longer needed"
        echo ""
    else
        echo -e "${GREEN}✅ All TODOs are properly tracked!${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  $TECH_DEBT_FILE not found - cannot verify tracking${NC}"
fi

# Save report
REPORT_FILE="todo-report-$(date +%Y%m%d-%H%M%S).txt"
{
    echo "TODO Report - Generated on $(date)"
    echo "=================================="
    echo ""
    cat "$TODO_RESULTS"
} > "$REPORT_FILE"

echo ""
echo -e "${BLUE}📄 Report saved to: $REPORT_FILE${NC}"

rm "$TODO_RESULTS"

exit 0
