#!/usr/bin/env python3
"""Validate the per-file @adai-status / @adai-version / @adai-reviewed tag.

See docs/development/guides/file-status-standard.md for the full standard.

Usage:
    ./scripts/check_file_status.py                    # whole repo, warn-only
    ./scripts/check_file_status.py --strict            # whole repo, fail on problems
    ./scripts/check_file_status.py --changed           # files changed vs origin/main
    ./scripts/check_file_status.py --changed --strict  # PR-gating mode
    ./scripts/check_file_status.py --changed HEAD~5     # custom base ref
"""
# @adai-status: beta        (validated via manual test cases this session; no formal test suite; capped by TD-043 — see TECHNICAL_DEBT.md)
# @adai-version: 0.9.0
# @adai-reviewed: 2026-09-07
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories/patterns whose contents are never in scope, even if they match an
# in-scope extension below (vendored code, build output, generated artifacts).
EXCLUDE_DIR_PARTS = {
    "build", "build-clang14", "build-clang-release", "build-gpu", "build-ubsan",
    "build-windows", "dist-windows", "external", "legacy", "tests", "gtest",
    "Testing", "node_modules", ".git", "test", "androidTest",
}

# (glob, ) pairs defining what's in scope. Extend deliberately, not opportunistically.
IN_SCOPE_GLOBS = [
    "src/**/*.cpp",
    "src/**/*.hpp",
    "src/**/*.h",
    "src/**/*.cu",
    "android/**/src/**/*.kt",
    "android/**/src/**/*.java",
    "tizen-metrics-app/js/*.js",
    "scripts/*.sh",
    "scripts/*.py",
]

TAG_RE = re.compile(
    r"@adai-status:\s*(?P<status>\S+).*?\n"
    r".*?@adai-version:\s*(?P<version>\S+).*?\n"
    r".*?@adai-reviewed:\s*(?P<reviewed>\S+)",
    re.MULTILINE,
)
VALID_STATUSES = {"experimental", "beta", "stable", "deprecated", "legacy"}
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
MAX_HEADER_LINES = 40


def in_scope_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in IN_SCOPE_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and not any(part in EXCLUDE_DIR_PARTS for part in path.parts):
                files.add(path)
    return sorted(files)


def changed_files(base_ref: str) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACM", f"{base_ref}...HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"error: git diff against {base_ref!r} failed: {e.stderr.strip()}", file=sys.stderr)
        sys.exit(2)
    scoped = set(in_scope_files())
    result = []
    for line in out.splitlines():
        p = (REPO_ROOT / line).resolve()
        if p in scoped:
            result.append(p)
    return sorted(result)


def check_file(path: Path) -> list[str]:
    """Return a list of problems (empty if the file is fine)."""
    try:
        header = "\n".join(path.read_text(errors="replace").splitlines()[:MAX_HEADER_LINES])
    except OSError as e:
        return [f"could not read file: {e}"]

    m = TAG_RE.search(header)
    if not m:
        return ["missing @adai-status/@adai-version/@adai-reviewed tag in first "
                f"{MAX_HEADER_LINES} lines"]

    problems = []
    status = m.group("status")
    version = m.group("version")
    reviewed = m.group("reviewed")

    if status not in VALID_STATUSES:
        problems.append(f"invalid @adai-status {status!r} (expected one of {sorted(VALID_STATUSES)})")

    semver = SEMVER_RE.match(version)
    if not semver:
        problems.append(f"invalid @adai-version {version!r} (expected MAJOR.MINOR.PATCH)")
    elif status == "stable" and int(semver.group(1)) < 1:
        problems.append(f"@adai-status is 'stable' but @adai-version {version!r} has MAJOR < 1 "
                         "(stable requires MAJOR >= 1)")

    try:
        reviewed_date = datetime.date.fromisoformat(reviewed)
        if reviewed_date > datetime.date.today():
            problems.append(f"@adai-reviewed {reviewed!r} is in the future")
    except ValueError:
        problems.append(f"invalid @adai-reviewed {reviewed!r} (expected YYYY-MM-DD)")

    if status == "deprecated":
        line = next((l for l in header.splitlines() if "@adai-status" in l), "")
        if "->" not in line and "replaced by" not in line.lower() and "see " not in line.lower():
            problems.append("@adai-status is 'deprecated' but the line names no replacement "
                             "(add e.g. '(replaced by Foo.hpp)')")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--changed", nargs="?", const="origin/main", metavar="BASE_REF",
                         help="only check files changed vs BASE_REF (default: origin/main)")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any problems found")
    args = parser.parse_args()

    files = changed_files(args.changed) if args.changed is not None else in_scope_files()

    if not files:
        print("No in-scope files to check.")
        return 0

    total_problems = 0
    missing = 0
    for path in files:
        problems = check_file(path)
        if problems:
            total_problems += len(problems)
            if "missing @adai-status" in problems[0]:
                missing += 1
            rel = path.relative_to(REPO_ROOT)
            for p in problems:
                print(f"{rel}: {p}")

    print(f"\nChecked {len(files)} file(s); {missing} missing the tag entirely; "
          f"{total_problems} problem(s) total.")

    if args.strict and total_problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
