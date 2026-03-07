# Markdown Lint Fixing Tools

This directory contains scripts to automatically fix markdown lint issues in documentation files.

## Quick Start

To fix all markdown files in the `docs/` directory:

```bash
./scripts/fix_markdown_lint.sh
```

This will:

1. Find all `.md` files in `docs/`
1. Run the Python fixer (`fix_all_markdown.py`) to apply common fixes
1. Optionally run `markdownlint --fix` if available for additional fixes
1. Report a summary of changes

## Available Scripts

### fix_markdown_lint.sh (Recommended)

Bash wrapper script that runs both fixers

```bash
./scripts/fix_markdown_lint.sh
```

What it fixes:

- MD022: Blank lines around headings
- MD031: Blank lines around fenced code blocks
- MD032: Blank lines around lists
- MD040: Language specification for code blocks
- MD060: Table formatting
- MD009: Trailing whitespace
- And more via markdownlint if available

Requirements:

- Python 3 (required)
- `markdownlint-cli` (optional, for additional fixes)

### fix_all_markdown.py

Python script for programmatic fixes

```bash
# Fix specific files
python3 scripts/fix_all_markdown.py docs/README.md docs/development/STEP1_COMPLETE.md

# Fix all files in docs/
python3 scripts/fix_all_markdown.py docs/**/*.md
```

What it fixes:

- MD022: Headings surrounded by blank lines
- MD031: Fenced code blocks surrounded by blank lines
- MD032: Lists surrounded by blank lines
- MD040: Code blocks have language specification
- MD060: Table column formatting
- MD009: Trailing whitespace
- MD029: Ordered list numbering

Features:

- No external dependencies (pure Python 3)
- In-place file modification
- Handles nested lists and code blocks
- Preserves code block content

## Installing markdownlint-cli (Optional)

For the most comprehensive fixes, install `markdownlint-cli`:

### Global Installation

```bash
npm install -g markdownlint-cli
```

### Project-Local Installation

```bash
npm install --save-dev markdownlint-cli
```

## Manual Usage

### Check for Issues

```bash
# Check all files
markdownlint docs/**/*.md

# Check specific file
markdownlint docs/README.md
```

### Fix Issues

```bash
# Fix all files
markdownlint --fix docs/**/*.md

# Fix specific file
markdownlint --fix docs/README.md
```

### Using Python Fixer Only

```bash
# Single file
python3 scripts/fix_all_markdown.py docs/README.md

# Multiple files
find docs -name "*.md" -print0 | xargs -0 python3 scripts/fix_all_markdown.py
```

## Common Lint Rules Fixed

|Rule|Description|Fix|
|---|---|---|
|MD022|Headings should be surrounded by blank lines|Adds blank lines before/after headings|
|MD031|Fenced code blocks should be surrounded by blank lines|Adds blank lines before/after code blocks|
|MD032|Lists should be surrounded by blank lines|Adds blank lines before/after lists|
|MD036|Emphasis used instead of heading|Converts `**text**` to `## text`|
|MD040|Fenced code blocks should have a language|Adds `text` or inferred language|
|MD060|Table column style|Formats table columns consistently|
|MD009|Trailing spaces|Removes trailing whitespace|
|MD029|Ordered list item prefix|Uses consistent numbering (1/1/1)|

## Workflow Integration

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Fix markdown lint issues before commit
./scripts/fix_markdown_lint.sh
git add docs/**/*.md
```

### CI/CD Check

```yaml
# .github/workflows/lint.yml
- name: Check Markdown
  run: |
    npm install -g markdownlint-cli
    markdownlint docs/**/*.md
```

## Troubleshooting

### Python script not fixing all issues

Some issues require manual intervention:

- Duplicate headings (MD024)
- Heading level increments (MD001)
- Link reference definitions

Use `markdownlint` to identify these.

### markdownlint not installed

The bash script will work without it, using only the Python fixer. Install it for comprehensive fixes.

### Files still have issues after running

Check the output for specific error messages. Some rules may require manual fixes or configuration changes.

## Configuration

Markdown lint rules can be configured via `.markdownlintrc`:

```json
{
  "default": true,
  "MD013": false,
  "MD033": false
}
```

Place this in the project root to customize which rules are enforced.

## Examples

### Fix all docs before committing

```bash
./scripts/fix_markdown_lint.sh
git add docs/
git commit -m "Fix markdown lint issues"
```

### Check a specific directory

```bash
python3 scripts/fix_all_markdown.py docs/development/*.md
```

### Verify fixes

```bash
# Before
markdownlint docs/README.md

# Fix
./scripts/fix_markdown_lint.sh

# After
markdownlint docs/README.md  # Should report no issues
```

## See Also

- [markdownlint rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
- [Markdown guide](https://www.markdownguide.org/)
- Project documentation: `docs/`
