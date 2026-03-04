# Git Workflow

This document outlines the Git workflow, branching strategy, and commit conventions for the ADAI project.

## Branch Strategy

We follow a **Git Flow** inspired branching model:

### Main Branches

- **`main`** - Production-ready code
  - Always stable and deployable
  - Protected branch - requires pull request reviews
  - Tagged for releases

- **`develop`** - Integration branch for ongoing development
  - Latest development changes
  - Source for feature branches
  - Target for feature merges

### Supporting Branches

- **`feature/*`** - New features or enhancements
  - Branch from: `develop`
  - Merge back to: `develop`
  - Naming: `feature/short-description`
  - Examples: `feature/beam-search`, `feature/add-gelu-activation`

- **`bugfix/*`** - Bug fixes for development
  - Branch from: `develop`
  - Merge back to: `develop`
  - Naming: `bugfix/issue-description`
  - Examples: `bugfix/matrix-transpose-bounds`, `bugfix/tokenizer-encoding`

- **`hotfix/*`** - Critical fixes for production
  - Branch from: `main`
  - Merge back to: `main` AND `develop`
  - Naming: `hotfix/critical-issue`
  - Examples: `hotfix/memory-leak`, `hotfix/gradient-nan`

- **`release/*`** - Preparation for production release
  - Branch from: `develop`
  - Merge back to: `main` AND `develop`
  - Naming: `release/vX.Y.Z`
  - Examples: `release/v1.0.0`, `release/v1.1.0`

## Commit Message Convention

We use **Conventional Commits** for clear and consistent commit history.

### Format

```text
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

- **`feat`** - New feature
- **`fix`** - Bug fix
- **`docs`** - Documentation changes
- **`style`** - Code style changes (formatting, missing semi-colons, etc.)
- **`refactor`** - Code restructuring without changing functionality
- **`perf`** - Performance improvements
- **`test`** - Adding or updating tests
- **`build`** - Build system or dependency changes
- **`ci`** - CI/CD configuration changes
- **`chore`** - Other changes (maintenance, tooling)
- **`revert`** - Reverting a previous commit

### Scope (Optional but Recommended)

The scope specifies the affected component:

- `matrix` - Matrix operations
- `optimizer` - Optimizer implementations
- `encoder` - Encoder components
- `decoder` - Decoder components
- `attention` - Attention mechanisms
- `tokenizer` - BPE tokenizer
- `chatbot` - Chatbot applications
- `tests` - Test infrastructure
- `docs` - Documentation
- `build` - Build system

### Subject

- Use imperative mood ("add" not "added" or "adds")
- Don't capitalize first letter
- No period at the end
- Maximum 50 characters

### Examples

```bash
# Good commit messages
feat(decoder): add beam search generation
fix(matrix): correct transpose operation bounds check
docs(api): update MultiHeadAttention documentation
test(encoder): add gradient flow tests
refactor(optimizer): extract momentum calculation to helper
perf(attention): optimize softmax computation
style(core): format code with clang-format
build(cmake): add sanitizer build options
```

```bash
# Bad commit messages (avoid these)
Update code                    # Too vague
Fixed bug                      # Not descriptive
Added new feature!!!           # Don't use punctuation/emotion
WIP - still working on this    # Don't commit WIP to shared branches
```

## Workflow Examples

### Creating a Feature Branch

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/add-gelu-activation

# Make changes and commit
git add src/Activation.cpp src/Activation.hpp
git commit -m "feat(activation): add GELU activation function"

# Push to remote
git push -u origin feature/add-gelu-activation
```

### Creating a Pull Request

1. **Ensure your branch is up to date**

   ```bash
   git checkout develop
   git pull origin develop
   git checkout feature/your-feature
   git rebase develop
   ```

2. **Run tests locally**

   ```bash
   cd build
   cmake ..
   make
   ctest --output-on-failure
   ```

3. **Push to remote**

   ```bash
   git push origin feature/your-feature
   ```

4. **Create PR on GitHub**
   - Go to repository on GitHub
   - Click "New Pull Request"
   - Base: `develop`, Compare: `feature/your-feature`
   - Fill in PR template (title, description, linked issues)

### Pull Request Guidelines

**PR Title Format:**
Same as commit message format: `<type>(<scope>): <description>`

**PR Description Should Include:**

- What changes were made
- Why the changes were necessary
- How to test the changes
- Related issue numbers (if applicable)

**Before Submitting:**

- [ ] All tests pass locally
- [ ] Code follows project style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] Branch is up to date with target branch

### Merging Strategy

- **Feature/Bugfix to Develop**: Squash and merge (clean history)
- **Develop to Main**: Merge commit (preserve history)
- **Hotfix to Main**: Merge commit
- **Release to Main**: Merge commit with version tag

## Commit Best Practices

### Do

- ✅ Commit early and often
- ✅ Make atomic commits (one logical change per commit)
- ✅ Write descriptive commit messages
- ✅ Test before committing
- ✅ Keep commits focused and small

### Don't

- ❌ Commit directly to `main` or `develop`
- ❌ Commit generated files (build artifacts, binaries)
- ❌ Commit IDE-specific files
- ❌ Mix unrelated changes in one commit
- ❌ Commit commented-out code
- ❌ Use `git commit -m "WIP"` on shared branches

## Handling Merge Conflicts

```bash
# Update your branch with latest develop
git checkout feature/your-feature
git fetch origin
git rebase origin/develop

# If conflicts occur
# 1. Open conflicting files and resolve markers
# 2. Test the changes
# 3. Stage resolved files
git add <resolved-files>

# 4. Continue rebase
git rebase --continue

# 5. Force push (only on your feature branch!)
git push --force-with-lease origin feature/your-feature
```

## Release Process

### Creating a Release

1. **Create release branch from develop**

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.0.0
   ```

2. **Update version numbers**
   - Update `CMakeLists.txt` version
   - Update `README.md` if needed
   - Update `CHANGELOG.md`

3. **Commit version bump**

   ```bash
   git commit -am "chore(release): bump version to 1.0.0"
   ```

4. **Merge to main and tag**

   ```bash
   git checkout main
   git pull origin main
   git merge --no-ff release/v1.0.0
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin main --tags
   ```

5. **Merge back to develop**

   ```bash
   git checkout develop
   git pull origin develop
   git merge --no-ff release/v1.0.0
   git push origin develop
   ```

6. **Delete release branch**

   ```bash
   git branch -d release/v1.0.0
   git push origin --delete release/v1.0.0
   ```

## Git Hooks (Optional but Recommended)

### Pre-commit Hook

Automatically format code before commit:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Format staged C++ files
for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp |hpp)$'); do
    if command -v clang-format &> /dev/null; then
        clang-format -i "$file"
        git add "$file"
    fi
done
```

### Commit-msg Hook

Validate commit message format:

```bash
#!/bin/bash
# .git/hooks/commit-msg

commit_msg=$(cat "$1")
pattern='^(feat| fix | docs | style | refactor | perf | test | build | ci | chore |revert)(\([a-z]+\))?: .{1,50}'

if ! echo "$commit_msg" | grep -qE "$pattern"; then
    echo "Error: Commit message does not follow conventional commits format"
    echo "Format: <type>(<scope>): <subject>"
    echo "Example: feat(decoder): add beam search"
    exit 1
fi
```

## Useful Git Commands

```bash
# View commit history in one line
git log --oneline --graph --decorate

# See what changed in last commit
git show HEAD

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes) - BE CAREFUL!
git reset --hard HEAD~1

# Stash changes temporarily
git stash
git stash pop

# Cherry-pick a commit from another branch
git cherry-pick <commit-hash>

# Interactive rebase to clean up commits
git rebase -i HEAD~5

# See which branch contains a commit
git branch --contains <commit-hash>

# Clean untracked files (dry run first)
git clean -n
git clean -fd
```

## Branch Protection Rules

For `main` and `develop` branches:

- ✅ Require pull request reviews (minimum 1 reviewer)
- ✅ Require status checks to pass (CI tests)
- ✅ Require branches to be up to date before merging
- ✅ Require linear history (no merge commits from feature branches)
- ✅ Include administrators in restrictions
- ❌ Allow force pushes
- ❌ Allow deletions

## Additional Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow Workflow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

---

**Questions or Issues?** Contact the maintainers or open an issue on GitHub.
