# Development Scripts

This directory contains utility scripts for ADAI development.

## Available Scripts

### 📊 check_tech_debt.sh
Scans codebase for technical debt markers and verifies tracking.

**Usage:**
```bash
./scripts/check_tech_debt.sh
```

**What it does:**
- Scans for TODO, FIXME, HACK, and XXX markers in code
- Verifies all markers are tracked in TECHNICAL_DEBT.md
- Reports summary of tracked technical debt items
- Highlights high-priority items

**Output:**
- Lists any untracked technical debt markers
- Shows count of active tracked items
- Displays high-priority debt items
- Exit code 0 if all debt is tracked, 1 if untracked markers found

**Best Practices:**
- Run before committing code
- All new debt markers must be tracked in TECHNICAL_DEBT.md
- Reference debt items in code: `// See TD-XXX in TECHNICAL_DEBT.md`
- Create GitHub issues for new debt using `.github/ISSUE_TEMPLATE/technical-debt.md`

---

### 🎨 format_code.sh
Formats all C++ source files using clang-format.

**Usage:**
```bash
./scripts/format_code.sh
```

**Requirements:** clang-format
```bash
sudo apt-get install clang-format
```

---

### 🔬 analyze_code.sh
Runs static analysis on C++ source files using clang-tidy.

**Usage:**
```bash
# Analyze all source files
./scripts/analyze_code.sh

# Analyze specific files
./scripts/analyze_code.sh src/Matrix.cpp src/Optimizer.cpp
```

**Requirements:** clang-tidy
```bash
sudo apt-get install clang-tidy
```

---

### 🧪 run_tests.sh
Runs the test suite with optional sanitizers and coverage.

**Usage:**
```bash
# Run tests normally
./scripts/run_tests.sh

# Run with AddressSanitizer
./scripts/run_tests.sh --asan

# Run with UndefinedBehaviorSanitizer
./scripts/run_tests.sh --ubsan

# Run with ThreadSanitizer
./scripts/run_tests.sh --tsan

# Run with coverage analysis
./scripts/run_tests.sh --coverage

# Verbose output
./scripts/run_tests.sh --verbose

# Combine options
./scripts/run_tests.sh --asan --verbose
```

**Requirements:** 
- AddressSanitizer/UndefinedBehaviorSanitizer: GCC/Clang with sanitizer support
- Coverage: lcov
```bash
sudo apt-get install lcov
```

---

## Quick Start

```bash
# Format code before committing
./scripts/format_code.sh

# Check for code quality issues
./scripts/analyze_code.sh

# Run all tests
./scripts/run_tests.sh

# Run tests with memory leak detection
./scripts/run_tests.sh --asan
```

---

## Pre-commit Hook (Optional)

To automatically format code before commits:

```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
./scripts/format_code.sh
git add -u
EOF

chmod +x .git/hooks/pre-commit
```
