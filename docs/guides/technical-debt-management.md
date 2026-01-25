# Technical Debt Management Guide

This guide explains how to identify, track, and resolve technical debt in the ADAI project.

## What is Technical Debt?

Technical debt represents code that works but could be improved. It includes:

- **Design Shortcuts**: Quick solutions that need refactoring
- **Incomplete Features**: Partially implemented functionality
- **Code Smells**: Patterns that indicate deeper problems
- **Missing Tests**: Untested or under-tested code
- **Documentation Gaps**: Undocumented or poorly documented code
- **Performance Issues**: Code that works but is inefficient
- **Deprecated APIs**: Use of outdated or deprecated interfaces

## When to Track Technical Debt

Track debt when:
- You implement a temporary workaround
- You skip a feature to meet a deadline
- You discover code that needs refactoring
- You find missing tests or documentation
- You identify performance bottlenecks
- You see duplicated or overly complex code

Don't track:
- Simple typos or formatting issues (just fix them)
- Planned features (use feature requests instead)
- Actual bugs (use bug reports instead)

## How to Track Technical Debt

### Step 1: Document in TECHNICAL_DEBT.md

Add a new entry in the "Active Technical Debt" section:

```markdown
### 3. Your Debt Item Title

**ID:** TD-003  
**Priority:** Medium  
**Component:** Component Name  
**Effort:** 2-4 hours  
**Status:** Open  

**Description:**  
Clear description of the debt item and why it exists.

**Impact:**
- What problems does this cause?
- Why should we fix it?

**Location:**
- `src/File.cpp:123-145` - Description of problem location

**Tasks:**
- [ ] Task 1
- [ ] Task 2
- [ ] Add tests
- [ ] Update documentation

**Files Affected:**
- `src/File.hpp`
- `src/File.cpp`

**Related Issues:** [Create issue in GitHub]

**Notes:**
Any additional context or constraints.
```

### Step 2: Create GitHub Issue (Optional)

For significant debt items, create a GitHub issue:

1. Use template: `.github/ISSUE_TEMPLATE/technical-debt.md`
2. Add label: `technical-debt`
3. Add priority label: `priority-high`, `priority-medium`, or `priority-low`
4. Link to TECHNICAL_DEBT.md
5. Update TECHNICAL_DEBT.md with issue number

### Step 3: Update Code Comments

Replace generic TODOs with tracked references:

```cpp
// ❌ BAD - Untracked TODO
// TODO: Optimize this algorithm

// ✅ GOOD - Tracked reference
// See TD-003 in TECHNICAL_DEBT.md - Algorithm optimization needed
```

### Step 4: Verify Tracking

Run the technical debt scanner:

```bash
./scripts/check_tech_debt.sh
```

This will find any untracked TODOs, FIXMEs, HACKs, or XXX markers.

## Priority Guidelines

### High Priority

Characteristics:
- Blocks new features
- Causes bugs or incorrect behavior
- Security or stability concerns
- Affects multiple components
- Significant technical risk

Examples:
- Memory leaks
- Race conditions
- Broken APIs
- Security vulnerabilities

**Action:** Address in current sprint or next sprint

### Medium Priority

Characteristics:
- Improves maintainability significantly
- Reduces technical complexity
- Enables future features
- Clear refactoring path
- Localized impact

Examples:
- Code duplication
- Missing abstractions
- Incomplete feature implementation
- Suboptimal algorithms

**Action:** Address within 1-2 months

### Low Priority

Characteristics:
- Nice-to-have improvements
- Cosmetic cleanups
- Non-critical optimizations
- Developer convenience

Examples:
- Better variable names
- More comments
- Stylistic improvements
- Minor refactoring

**Action:** Address opportunistically or when convenient

## Resolving Technical Debt

### Before You Start

1. Review the debt item in TECHNICAL_DEBT.md
2. Check for related GitHub issues
3. Understand the scope and impact
4. Plan your approach
5. Consider backward compatibility

### Implementation Process

1. **Create a branch**
   ```bash
   git checkout -b fix/td-003-algorithm-optimization
   ```

2. **Make changes incrementally**
   - Fix one thing at a time
   - Keep commits focused
   - Run tests frequently

3. **Add tests**
   - Ensure behavior is preserved
   - Add tests for edge cases
   - Verify no regressions

4. **Update documentation**
   - Update code comments
   - Update API docs if needed
   - Update user guides if behavior changes

5. **Run quality checks**
   ```bash
   ./scripts/format_code.sh
   ./scripts/analyze_code.sh
   ./scripts/run_tests.sh
   ./scripts/check_tech_debt.sh
   ```

### Completing the Resolution

1. **Update TECHNICAL_DEBT.md**
   - Move item to "Resolved Items" section
   - Add resolution date
   - Add brief summary of changes

   ```markdown
   ## Resolved Items
   
   ### TD-003: Algorithm Optimization (Resolved: 2026-01-24)
   **Resolution:** Replaced O(n²) algorithm with O(n log n) sort-based approach.
   Achieved 10x performance improvement on large datasets.
   **PR:** #123
   ```

2. **Close GitHub issue** (if exists)
   - Reference PR in closing comment
   - Add metrics if applicable

3. **Update statistics** in TECHNICAL_DEBT.md
   - Decrement active count
   - Update priority distribution

4. **Remove code comments**
   - Delete `// See TD-XXX` references
   - Add normal documentation as needed

## Common Patterns

### Incomplete Feature Implementation

```markdown
**ID:** TD-XXX  
**Priority:** Medium  
**Component:** Feature Name  

**Description:**  
Feature X was partially implemented. Currently supports Y but needs Z.

**Tasks:**
- [ ] Implement missing functionality Z
- [ ] Add tests for Z
- [ ] Update documentation
```

### Code Duplication

```markdown
**ID:** TD-XXX  
**Priority:** Low  
**Component:** Core/Utils  

**Description:**  
Similar logic duplicated in multiple files. Should extract to common utility.

**Tasks:**
- [ ] Create shared utility function
- [ ] Replace duplicated code with utility calls
- [ ] Add tests for utility
```

### Performance Issue

```markdown
**ID:** TD-XXX  
**Priority:** Medium  
**Component:** Matrix Operations  

**Description:**  
Matrix multiplication uses naive O(n³) algorithm. Should optimize for large matrices.

**Tasks:**
- [ ] Implement Strassen or BLAS-based multiplication
- [ ] Add performance benchmarks
- [ ] Verify numerical accuracy
```

## Anti-Patterns to Avoid

### ❌ Vague Descriptions

```markdown
### Fix the code
**Description:** Code is bad and needs fixing.
```

**Why bad:** Doesn't explain what's wrong or how to fix it.

### ❌ Too Many Tasks

```markdown
**Tasks:**
- [ ] Refactor entire codebase
- [ ] Rewrite all tests
- [ ] Add documentation
- [ ] Improve performance
```

**Why bad:** Too broad. Break into multiple debt items.

### ❌ Missing Context

```markdown
### Optimize function
**Location:** src/File.cpp
```

**Why bad:** No line numbers, no explanation of what needs optimization.

### ❌ Untracked TODOs

```cpp
// TODO: Fix this later
```

**Why bad:** Will be forgotten. Always track debt items.

## Monthly Review Process

1. **Review TECHNICAL_DEBT.md**
   - Are priorities still accurate?
   - Have circumstances changed?
   - Can any items be closed?

2. **Run scanner**
   ```bash
   ./scripts/check_tech_debt.sh
   ```

3. **Update statistics**
   - Total items
   - Priority distribution
   - Component distribution

4. **Plan next month**
   - Which items to tackle?
   - Allocate time for debt reduction
   - Balance new features vs. debt

## Integration with Development Workflow

### During Code Review

Reviewers should check:
- [ ] New TODOs are tracked in TECHNICAL_DEBT.md
- [ ] Existing debt references are accurate
- [ ] No untracked technical debt introduced

### Before Release

1. Review high-priority debt items
2. Decide if any must be addressed before release
3. Document known limitations in release notes

### Sprint Planning

1. Reserve 10-20% of sprint capacity for debt
2. Pick debt items that:
   - Align with current work
   - Unblock upcoming features
   - Reduce risk

## Tools and Automation

### Scripts

- `./scripts/check_tech_debt.sh` - Scan for untracked debt
- `./scripts/analyze_code.sh` - Static analysis
- `./scripts/format_code.sh` - Code formatting

### IDE Integration

Consider configuring your IDE to:
- Highlight TODO/FIXME comments
- Link to TECHNICAL_DEBT.md
- Show debt items in task list

### Git Hooks

Add pre-commit hook to check for untracked debt:

```bash
#!/bin/bash
# .git/hooks/pre-commit
./scripts/check_tech_debt.sh
```

## Metrics and Reporting

### Track Over Time

Monitor these metrics:
- Total debt items (should trend down)
- Average age of debt items
- High-priority debt count
- Debt resolution rate

### Reporting

Monthly debt report template:

```markdown
# Technical Debt Report - January 2026

**Summary:**
- Total items: 5 (↓ 2 from last month)
- High priority: 0 (↓ 1)
- Medium priority: 3 (↔ 0)
- Low priority: 2 (↓ 1)

**Resolved This Month:**
- TD-001: Optimizer parameter exposure
- TD-007: Error handling in tokenizer

**New This Month:**
- TD-008: Memory pool for matrix operations

**Next Month Focus:**
- TD-003: Algorithm optimization
- TD-005: Increase test coverage
```

## Questions?

- Check [TECHNICAL_DEBT.md](../TECHNICAL_DEBT.md) for active items
- See [Contributing Guide](../docs/guides/contributing.md) for code standards
- Review [Process Improvement Plan](../PROCESS_IMPROVEMENT_PLAN.md) for guidelines

---

**Remember:** Technical debt isn't bad - it's a tool for managing trade-offs. The key is tracking it consciously and paying it down systematically.
