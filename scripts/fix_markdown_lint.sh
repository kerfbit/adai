#!/bin/bash

# Script to find and fix markdown lint issues in all docs/ markdown files
# 
# This script will:
# 1. Find all .md files in the docs/ directory
# 2. Run fix_all_markdown.py on all files (fixes MD022, MD031, MD032, MD036, MD040, MD060, MD009, MD029)
# 3. Optionally try markdownlint --fix if available for additional fixes
# 4. Report summary of fixes
#
# Common issues fixed automatically:
# - MD022: Blank lines around headings
# - MD031: Blank lines around code blocks
# - MD032: Blank lines around lists
# - MD036: Emphasis used as heading (converts to plain text)
# - MD040: Code block language specification
# - MD060: Table formatting (compact style)
# - MD009: Trailing whitespace
# - MD029: Ordered list numbering

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
PYTHON_FIXER="$SCRIPT_DIR/fix_all_markdown.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Markdown Lint Fixer ===${NC}"
echo ""

# Check if docs directory exists
if [ ! -d "$DOCS_DIR" ]; then
    echo -e "${RED}Error: docs/ directory not found at $DOCS_DIR${NC}"
    exit 1
fi

# Find all markdown files
echo -e "${BLUE}Finding markdown files in docs/...${NC}"
MD_FILES=$(find "$DOCS_DIR" -type f -name "*.md" | sort)
FILE_COUNT=$(echo "$MD_FILES" | wc -l)

if [ -z "$MD_FILES" ]; then
    echo -e "${YELLOW}No markdown files found in docs/${NC}"
    exit 0
fi

echo -e "${GREEN}Found $FILE_COUNT markdown files${NC}"
echo ""

# Method 1: Use our Python fixer
if [ -f "$PYTHON_FIXER" ]; then
    echo -e "${BLUE}Method 1: Using Python fixer (fix_all_markdown.py)...${NC}"
    echo ""
    
    # Convert newline-separated list to space-separated for python script
    MD_FILES_ARRAY=($MD_FILES)
    
    python3 "$PYTHON_FIXER" "${MD_FILES_ARRAY[@]}"
    
    echo ""
    echo -e "${GREEN}✓ Python fixer complete!${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ Python fixer not found at $PYTHON_FIXER${NC}"
fi

# Method 2: Try markdownlint if available
if command -v markdownlint &> /dev/null; then
    echo -e "${BLUE}Method 2: Running markdownlint --fix for additional fixes...${NC}"
    echo ""
    
    FIXED_COUNT=0
    ERROR_COUNT=0
    
    while IFS= read -r file; do
        # Get relative path for display
        REL_PATH="${file#$PROJECT_ROOT/}"
        
        # Run markdownlint with fix
        if markdownlint --fix "$file" 2>&1 >/dev/null; then
            echo -e "  ${GREEN}✓${NC} $REL_PATH"
            ((FIXED_COUNT++))
        else
            echo -e "  ${YELLOW}⚠${NC} $REL_PATH - has remaining issues"
            ((ERROR_COUNT++))
        fi
    done <<< "$MD_FILES"
    
    echo ""
    echo -e "${BLUE}=== Markdownlint Summary ===${NC}"
    echo -e "${GREEN}Files processed: $FILE_COUNT${NC}"
    
    if [ $ERROR_COUNT -gt 0 ]; then
        echo -e "${YELLOW}Files with remaining issues: $ERROR_COUNT${NC}"
        echo ""
        echo -e "${YELLOW}To see details, run: markdownlint docs/**/*.md${NC}"
    else
        echo -e "${GREEN}All files passed markdownlint!${NC}"
    fi
    
    echo ""
else
    echo -e "${YELLOW}Method 2: markdownlint-cli not available (optional)${NC}"
    echo ""
    echo "To install markdownlint-cli for additional checks:"
    echo -e "  ${BLUE}npm install -g markdownlint-cli${NC}"
    echo ""
fi

echo -e "${GREEN}✓ Markdown lint fixing complete!${NC}"
echo ""
echo "Files processed: $FILE_COUNT"
echo ""
echo "To verify fixes, you can run:"
echo -e "  ${BLUE}find docs -name '*.md' -exec markdownlint {} +${NC}"
