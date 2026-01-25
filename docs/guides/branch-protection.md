# Branch Protection Setup Guide

This guide walks through setting up branch protection rules for the ADAI repository to ensure code quality and prevent accidental changes to critical branches.

## Why Branch Protection?

Branch protection rules:
- **Prevent force pushes** to important branches
- **Require code reviews** before merging
- **Ensure CI passes** before allowing merges
- **Maintain code quality** through automated checks
- **Prevent accidental deletions** of branches
- **Enforce development workflow** standards

## Recommended Rules for `main` Branch

### 1. Require Pull Request Reviews

**Settings:**
- ✅ Require a pull request before merging
- ✅ Require approvals: **1**
- ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require review from Code Owners (if CODEOWNERS file exists)

**Why:** Ensures all code is reviewed by at least one other developer before merging.

### 2. Require Status Checks

**Settings:**
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging

**Required Status Checks:**
- `Build and Test (ubuntu-22.04, gcc, Release)`
- `Build and Test (ubuntu-22.04, clang, Release)`
- `Build and Test (ubuntu-20.04, gcc, Release)`
- `Code Quality Checks`
- `Sanitizer Tests / asan`
- `Sanitizer Tests / ubsan`

**Why:** Ensures code passes all tests and quality checks before being merged.

### 3. Additional Protections

**Settings:**
- ✅ Require conversation resolution before merging
- ✅ Require signed commits (optional but recommended)
- ✅ Require linear history (prevents merge commits, enforces squash or rebase)
- ✅ Include administrators (applies rules to repository admins too)
- ❌ Allow force pushes (keep disabled)
- ❌ Allow deletions (keep disabled)

## Setup Instructions

### Method 1: GitHub Web Interface

1. **Navigate to Settings**
   - Go to your repository on GitHub
   - Click **Settings** tab
   - Click **Branches** in the left sidebar

2. **Add Branch Protection Rule**
   - Click **Add rule** button
   - Branch name pattern: `main`

3. **Configure Protection Settings**
   
   **Protect matching branches:**
   - ✅ Require a pull request before merging
     - Required approvals: `1`
     - ✅ Dismiss stale pull request approvals when new commits are pushed
     - ✅ Require review from Code Owners
   
   - ✅ Require status checks to pass before merging
     - ✅ Require branches to be up to date before merging
     - Search and add each required status check:
       - `Build and Test (ubuntu-22.04, gcc, Release)`
       - `Build and Test (ubuntu-22.04, clang, Release)`
       - `Code Quality Checks`
       - And others as they appear in your Actions
   
   - ✅ Require conversation resolution before merging
   - ✅ Require signed commits (optional)
   - ✅ Require linear history
   - ✅ Include administrators
   
   - ❌ Allow force pushes: **Nobody**
   - ❌ Allow deletions: Disabled

4. **Create/Save Rule**
   - Click **Create** or **Save changes**

### Method 2: GitHub CLI

If you have `gh` CLI installed:

```bash
# Install gh if needed
# Ubuntu: sudo apt install gh
# macOS: brew install gh

# Authenticate
gh auth login

# Create branch protection rule
gh api repos/rjv717/adai/branches/main/protection \
  -X PUT \
  --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Build and Test (ubuntu-22.04, gcc, Release)",
      "Build and Test (ubuntu-22.04, clang, Release)",
      "Code Quality Checks"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismissal_restrictions": {},
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
EOF
```

## Optional: Protect `develop` Branch

For repositories using a `develop` integration branch:

**Settings (slightly relaxed from `main`):**
- ✅ Require pull request reviews: **1 approval**
- ✅ Require status checks to pass
- ✅ Require conversation resolution
- ✅ Include administrators
- ❌ Allow force pushes: **Nobody**
- ❌ Allow deletions: Disabled

**Difference from `main`:**
- May not require linear history (allows merge commits)
- May not require signed commits
- Same CI checks required

## Bypass Options (Not Recommended)

For emergency situations, repository admins can:
1. Temporarily disable branch protection
2. Make emergency changes
3. Re-enable protection immediately

**Better approach:** Use hotfix branches even for emergencies, maintaining the protection rules.

## Code Owners (Optional Enhancement)

Create `.github/CODEOWNERS` file to automatically assign reviewers:

```
# Default owners for everything
* @rjv717

# Core components
/src/core/* @rjv717
/src/attention/* @rjv717

# Documentation
/docs/* @rjv717
*.md @rjv717

# CI/CD
/.github/workflows/* @rjv717
```

**Benefits:**
- Automatic reviewer assignment
- Can require review from code owners
- Clear ownership of code areas

## Verifying Protection Rules

### Test the Protection

1. **Try to push directly to main:**
   ```bash
   git checkout main
   echo "test" >> test.txt
   git add test.txt
   git commit -m "test: Direct push"
   git push origin main
   ```
   
   **Expected result:** Push rejected with message about branch protection

2. **Try force push:**
   ```bash
   git push --force origin main
   ```
   
   **Expected result:** Force push rejected

3. **Create PR without CI passing:**
   - Create branch and PR
   - Don't wait for CI
   - Try to merge
   
   **Expected result:** Merge button disabled until CI passes

### View Protection Status

```bash
# Using GitHub CLI
gh api repos/rjv717/adai/branches/main/protection | jq

# Check status checks
gh api repos/rjv717/adai/branches/main/protection/required_status_checks | jq
```

## Troubleshooting

### Status Checks Not Appearing

**Problem:** Required status checks don't appear in the list

**Solution:**
1. Run workflows at least once on the branch
2. Wait for workflows to complete
3. Refresh branch protection settings page
4. Status checks should now be selectable

### Can't Merge Even Though CI Passed

**Problem:** Merge button disabled despite passing checks

**Causes:**
- Branch not up to date with base branch
- Conversations not resolved
- Required reviewers haven't approved
- Administrator enforcement enabled but admin hasn't approved

**Solutions:**
1. Update branch: `git pull origin main && git push`
2. Resolve all conversations
3. Request/wait for review approval
4. Check all protection requirements are met

### Emergency Bypass Needed

**Problem:** Need to push critical fix but can't bypass protection

**Solution:**
1. Create hotfix branch from main
2. Make fix
3. Create PR
4. Get fast-track review
5. Merge through normal process

**Never disable protection rules permanently.**

## Best Practices

1. **Set up protection early** - Before project gains contributors
2. **Apply to all important branches** - main, develop, release/*
3. **Include administrators** - Rules apply to everyone
4. **Require up-to-date branches** - Prevents integration issues
5. **Use squash merging** - Keeps history clean
6. **Document exceptions** - If you must bypass, document why
7. **Review rules quarterly** - Adjust as project evolves

## Monitoring

### Review Protection Effectiveness

Monthly review:
- Number of direct push attempts (should be 0)
- PRs merged without review (should be 0)
- CI failures caught before merge
- Time from PR creation to merge

### Adjust Rules As Needed

As project matures:
- Add more required status checks
- Increase required reviewers (2+)
- Add required code owners
- Tighten restrictions

## Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [GitHub Status Checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/about-status-checks)
- [CODEOWNERS Syntax](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)

---

**Last Updated:** January 24, 2026  
**Status:** Recommended for implementation
