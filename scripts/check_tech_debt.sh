#!/bin/bash
# Script to scan codebase for technical debt markers and verify tracking

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Technical Debt Scanner ===${NC}\n"

# Configuration
SRC_DIR="src"
TESTS_DIR="tests"
DEBT_FILE="TECHNICAL_DEBT.md"

# Initialize counters
TODO_COUNT=0
FIXME_COUNT=0
HACK_COUNT=0
XXX_COUNT=0

echo -e "${YELLOW}Scanning for technical debt markers...${NC}\n"

# Function to scan for a pattern and display results
scan_pattern() {
    local pattern=$1
    local name=$2
    local color=$3
    
    echo -e "${color}${name} markers:${NC}"
    
    # Search in src/
    if [ -d "$SRC_DIR" ]; then
        results=$(grep -rn "$pattern" "$SRC_DIR" --include="*.cpp" --include="*.hpp" 2>/dev/null || true)
        if [ -n "$results" ]; then
            echo "$results" | while IFS= read -r line; do
                echo "  $line"
            done
            count=$(echo "$results" | wc -l)
        else
            count=0
        fi
    else
        count=0
    fi
    
    # Search in tests/
    if [ -d "$TESTS_DIR" ]; then
        test_results=$(grep -rn "$pattern" "$TESTS_DIR" --include="*.cpp" --include="*.hpp" 2>/dev/null || true)
        if [ -n "$test_results" ]; then
            echo "$test_results" | while IFS= read -r line; do
                echo "  $line"
            done
            test_count=$(echo "$test_results" | wc -l)
            count=$((count + test_count))
        fi
    fi
    
    if [ $count -eq 0 ]; then
        echo -e "  ${GREEN}None found ✓${NC}"
    fi
    
    echo ""
    return $count
}

# Scan for different markers
scan_pattern "TODO" "TODO" "$YELLOW"
TODO_COUNT=$?

scan_pattern "FIXME" "FIXME" "$RED"
FIXME_COUNT=$?

scan_pattern "HACK" "HACK" "$RED"
HACK_COUNT=$?

scan_pattern "XXX" "XXX" "$RED"
XXX_COUNT=$?

# Calculate total
TOTAL=$((TODO_COUNT + FIXME_COUNT + HACK_COUNT + XXX_COUNT))

echo -e "${BLUE}=== Summary ===${NC}"
echo -e "TODO markers:  ${YELLOW}$TODO_COUNT${NC}"
echo -e "FIXME markers: ${RED}$FIXME_COUNT${NC}"
echo -e "HACK markers:  ${RED}$HACK_COUNT${NC}"
echo -e "XXX markers:   ${RED}$XXX_COUNT${NC}"
echo -e "Total:         ${YELLOW}$TOTAL${NC}"
echo ""

# Check if technical debt file exists
if [ ! -f "$DEBT_FILE" ]; then
    echo -e "${RED}WARNING: $DEBT_FILE not found!${NC}"
    echo "Please create this file to track technical debt."
    exit 1
fi

# Verify all markers are tracked
if [ $TOTAL -gt 0 ]; then
    echo -e "${YELLOW}⚠ Untracked technical debt found!${NC}"
    echo ""
    echo "Please ensure all markers are tracked in $DEBT_FILE with:"
    echo "  1. A unique TD-XXX identifier"
    echo "  2. Priority and effort estimate"
    echo "  3. Tasks and affected files"
    echo "  4. Reference in code: // See TD-XXX in TECHNICAL_DEBT.md"
    echo ""
    echo "To create a GitHub issue, use the template:"
    echo "  .github/ISSUE_TEMPLATE/technical-debt.md"
    exit 1
else
    echo -e "${GREEN}✓ No untracked technical debt markers found!${NC}"
    echo ""
    echo "All technical debt should be tracked in $DEBT_FILE"
fi

# Display tracked items summary
echo -e "${BLUE}=== Tracked Technical Debt ===${NC}"
if [ -f "$DEBT_FILE" ]; then
    # Count active items (lines starting with "### " in Active section)
    active_count=$(awk '/^## Active Technical Debt/,/^## Resolved Items/' "$DEBT_FILE" | grep -c "^### " || true)
    echo "Active items in $DEBT_FILE: $active_count"
    
    # Show high priority items
    echo ""
    echo "High priority items:"
    awk '/^## Active Technical Debt/,/^## Resolved Items/' "$DEBT_FILE" | \
        grep -A 2 "Priority.*High" | grep "^### " || echo "  None"
fi

echo ""
echo -e "${GREEN}Scan complete!${NC}"
