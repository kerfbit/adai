# CI/CD Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) setup for the ADAI project.

## Overview

The project uses GitHub Actions for automated testing, quality checks, and releases. All workflows are defined in `.github/workflows/`.

## Workflows

### 1. CI Workflow (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Jobs:**

#### Build and Test
- **Matrix Strategy:** Tests across multiple configurations
  - OS: Ubuntu 22.04, Ubuntu 20.04
  - Compilers: GCC, Clang
  - Build Types: Debug, Release
- **Steps:**
  1. Checkout code with submodules
  2. Install dependencies (CMake, build tools, clang-tidy, clang-format)
  3. Configure CMake with specified compiler and build type
  4. Build all targets
  5. Run all tests with CTest
  6. Upload test results on failure

#### Code Quality
- **Checks:**
  - Code formatting with clang-format (dry-run)
  - Static analysis with clang-tidy
- **Failure Criteria:**
  - Unformatted code (run `./scripts/format_code.sh` to fix)
  - Static analysis warnings (informational only for now)

#### Sanitizer Tests
- **Sanitizers:**
  - AddressSanitizer (ASAN) - Memory error detection
  - UndefinedBehaviorSanitizer (UBSAN) - Undefined behavior detection
- **Purpose:** Catch memory leaks, buffer overflows, and undefined behavior

**Status Badge:**
```markdown
[![CI](https://github.com/rjv717/adai/actions/workflows/ci.yml/badge.svg)](https://github.com/rjv717/adai/actions/workflows/ci.yml)
```

---

### 2. Code Coverage Workflow (`coverage.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` branch

**Process:**
1. Build with coverage instrumentation (`--coverage` flag)
2. Run all tests
3. Generate coverage report with lcov
4. Filter out external dependencies and test code
5. Upload to Codecov (requires `CODECOV_TOKEN` secret)
6. Generate HTML coverage report
7. Display coverage summary in GitHub Action summary

**Outputs:**
- Coverage report uploaded to Codecov
- HTML coverage report as artifact (30-day retention)
- Coverage summary in action logs

**Status Badge:**
```markdown
[![codecov](https://codecov.io/gh/rjv717/adai/branch/main/graph/badge.svg)](https://codecov.io/gh/rjv717/adai)
```

---

### 3. Release Workflow (`release.yml`)

**Triggers:**
- Push of version tags (e.g., `v1.0.0`, `v1.2.3`)
- Manual workflow dispatch

**Process:**
1. Build Release configuration on multiple platforms
2. Run full test suite
3. Package binaries with documentation
4. Create release artifacts (`.tar.gz` archives)
5. Create GitHub Release with artifacts

**Platforms:**
- Ubuntu 22.04
- Ubuntu 20.04
- macOS (latest)

**Artifacts Include:**
- Compiled executables (`chatbot`, `chatbot_trainer`)
- Documentation
- README

---

## Setup Requirements

### Repository Secrets

Add these secrets in GitHub repository settings:

1. **CODECOV_TOKEN** (optional)
   - Sign up at [codecov.io](https://codecov.io)
   - Add your repository
   - Copy the upload token
   - Add as repository secret

### Branch Protection Rules

Recommended settings for `main` branch:

1. **Require pull request reviews before merging**
   - Require 1 approval
   - Dismiss stale reviews when new commits are pushed

2. **Require status checks to pass before merging**
   - Require branches to be up to date
   - Required status checks:
     - `Build and Test (ubuntu-22.04, gcc, Release)`
     - `Build and Test (ubuntu-22.04, clang, Release)`
     - `Code Quality Checks`

3. **Require conversation resolution before merging**

4. **Do not allow bypassing the above settings**

### Setting Up Branch Protection

1. Go to repository Settings → Branches
2. Click "Add rule" under "Branch protection rules"
3. Branch name pattern: `main`
4. Enable the settings above
5. Click "Create" or "Save changes"

---

## Local Testing

### Before Pushing

Run these commands locally to catch issues before CI:

```bash
# Format code
./scripts/format_code.sh

# Run static analysis
./scripts/analyze_code.sh

# Build and test
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure

# Test with sanitizers
cd ..
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
make -j$(nproc)
ctest --output-on-failure
```

### Coverage Report Locally

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="--coverage"
make -j$(nproc)
ctest

# Generate report
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/gtest/*' '*/tests/*' \
  --output-file coverage.info
genhtml coverage.info --output-directory coverage_html

# View report
xdg-open coverage_html/index.html  # Linux
open coverage_html/index.html      # macOS
```

---

## Troubleshooting

### CI Build Fails but Local Succeeds

**Common causes:**
1. **Missing dependencies** - Check workflow installs all required packages
2. **Environment differences** - CI uses clean Ubuntu/macOS environment
3. **Cached build artifacts** - CI always builds from scratch

**Solutions:**
- Test in clean environment: `rm -rf build && mkdir build && cd build && cmake .. && make`
- Check CI logs for specific error messages
- Ensure all dependencies are in `ci.yml` install steps

### Code Formatting Check Fails

**Error message:** "Code formatting issues found"

**Solution:**
```bash
./scripts/format_code.sh
git add .
git commit --amend --no-edit
git push --force-with-lease
```

### Sanitizer Failures

**Error:** AddressSanitizer detects memory leak/error

**Steps:**
1. Run locally with sanitizer enabled (see above)
2. Review sanitizer output for specific issue
3. Fix the memory issue
4. Re-run tests locally to verify fix
5. Push changes

### Coverage Upload Fails

**Error:** Codecov upload failed

**Causes:**
- Missing or invalid `CODECOV_TOKEN` secret
- Codecov service outage

**Note:** Coverage upload failure does not fail the workflow (`fail_ci_if_error: false`)

---

## Workflow Optimization

### Reducing CI Time

Current strategies:
- **Matrix exclusions:** Skip some OS/compiler/build type combinations
- **Parallel jobs:** Multiple configurations run in parallel
- **Dependency caching:** Could be added for faster builds

### Future Improvements

1. **Cache CMake builds** - Use `actions/cache` for faster rebuilds
2. **Docker containers** - Pre-built containers with all dependencies
3. **Selective testing** - Run only affected tests for small changes
4. **Nightly builds** - Extended test suites on schedule

---

## Monitoring

### Status Checks

View workflow runs:
- Repository → Actions tab
- See all runs, logs, and artifacts
- Download test results and coverage reports

### Badges

Add to README.md:
```markdown
[![CI](https://github.com/rjv717/adai/actions/workflows/ci.yml/badge.svg)](https://github.com/rjv717/adai/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/rjv717/adai/branch/main/graph/badge.svg)](https://codecov.io/gh/rjv717/adai)
```

### Notifications

Configure notifications:
- GitHub Settings → Notifications
- Email on workflow failures
- Slack/Discord integration available

---

## Release Process

### Creating a Release

1. **Update version** in relevant files
2. **Commit and push** changes
3. **Create and push tag:**
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
4. **Release workflow** automatically creates GitHub release
5. **Download artifacts** from release page

### Manual Release

Trigger manually:
1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Select branch
4. Click "Run workflow"

---

## Best Practices

1. **Always run tests locally** before pushing
2. **Format code** before committing
3. **Review CI logs** if workflow fails
4. **Keep workflows updated** with dependency versions
5. **Monitor coverage trends** - aim for >80% coverage
6. **Fix sanitizer issues immediately** - they indicate real bugs

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CMake Testing Documentation](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [Clang-Tidy Checks](https://clang.llvm.org/extra/clang-tidy/checks/list.html)
- [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [Codecov Documentation](https://docs.codecov.com/)

---

**Last Updated:** January 24, 2026  
**Maintained By:** ADAI Development Team
