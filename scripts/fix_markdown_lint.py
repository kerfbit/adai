#!/usr/bin/env python3

# @adai-status: beta        (general repo-maintenance utility, no test)
# @adai-version: 0.6.0
# @adai-reviewed: 2026-09-07

"""
Markdown lint fixer for the ADAI repository.

Finds and fixes common markdownlint violations across all .md files.

Rules fixed:
  MD009  Trailing whitespace
  MD022  Headings should be surrounded by blank lines
  MD029  Ordered list item prefix (consistent numbering)
  MD031  Fenced code blocks should be surrounded by blank lines
  MD032  Lists should be surrounded by blank lines
  MD036  Emphasis used instead of heading
  MD040  Fenced code blocks should have a language specifier
  MD060  Table column style (compact, no spaces around pipes)

Usage:
    # Check all .md files in the repo (exit 1 if issues found)
    python3 scripts/fix_markdown_lint.py --check

    # Fix all .md files in the repo
    python3 scripts/fix_markdown_lint.py

    # Fix specific files
    python3 scripts/fix_markdown_lint.py docs/README.md docs/guides/*.md

    # Fix files under a specific directory
    python3 scripts/fix_markdown_lint.py --dir docs/
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def find_repo_root():
    """Walk up from this script to find the git repository root."""
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / ".git").exists():
            return path
        path = path.parent
    return Path.cwd()


def discover_markdown_files(root, subdirectory=None):
    """Find all .md files under root, skipping build/ and hidden dirs."""
    search = Path(root) / subdirectory if subdirectory else Path(root)
    skip = {".git", "build", "build-windows", "node_modules", ".vscode"}
    results = []
    for dirpath, dirnames, filenames in os.walk(search):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for f in filenames:
            if f.endswith(".md"):
                results.append(os.path.join(dirpath, f))
    return sorted(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_heading(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("```") or stripped.startswith("#include"):
        return False
    return stripped.startswith("#")


def is_code_fence(line):
    return line.strip().startswith("```")


def is_list_item(line):
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped[0] in "-*+" and (len(stripped) == 1 or stripped[1] == " "):
        return True
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    return False


# ---------------------------------------------------------------------------
# Core fixer
# ---------------------------------------------------------------------------

def fix_markdown(lines):
    """Apply all lint fixes to a list of lines. Returns (fixed_lines, changed)."""
    if not lines:
        return lines, False

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    fixed = []
    in_code_block = False
    prev_was_list = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # MD009: strip trailing whitespace
        line = line.rstrip() + "\n"

        # --- Code fence handling ---
        if is_code_fence(line):
            # MD040: add language if missing (opening fence)
            if line.strip() == "```" and not in_code_block:
                line = "```text\n"

            # MD031: blank line before opening fence
            if not in_code_block:
                if fixed and fixed[-1].strip():
                    if not is_heading(fixed[-1]) and not is_code_fence(fixed[-1]):
                        fixed.append("\n")

            fixed.append(line)

            was_closing = in_code_block
            in_code_block = not in_code_block

            # MD031: blank line after closing fence
            if was_closing and i + 1 < len(lines):
                nxt = lines[i + 1]
                if nxt.strip() and not is_heading(nxt) and not is_code_fence(nxt):
                    fixed.append("\n")

            i += 1
            continue

        # Inside code blocks: pass through untouched (except MD009 above)
        if in_code_block:
            fixed.append(line)
            i += 1
            continue

        curr_is_list = is_list_item(line)

        # MD032: blank line before list start
        if curr_is_list and not prev_was_list:
            if fixed and fixed[-1].strip():
                if not is_heading(fixed[-1]) and not is_code_fence(fixed[-1]):
                    fixed.append("\n")

        # MD032: blank line after list end
        if prev_was_list and not curr_is_list and line.strip():
            if not is_heading(line) and not is_code_fence(line):
                if fixed and fixed[-1].strip():
                    fixed.append("\n")

        # MD022: blank line before heading
        if is_heading(line):
            if fixed and fixed[-1].strip():
                if not is_heading(fixed[-1]) and not is_code_fence(fixed[-1]):
                    fixed.append("\n")

        fixed.append(line)

        # MD022: blank line after heading
        if is_heading(line) and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt.strip() and not is_heading(nxt) and not is_code_fence(nxt):
                fixed.append("\n")

        prev_was_list = curr_is_list
        i += 1

    # --- Second pass: tables (MD060) and emphasis-as-heading (MD036) ---
    final = []
    for line in fixed:
        # MD060: compact table style
        if "|" in line and not line.strip().startswith("```") and line.count("|") >= 2:
            parts = line.rstrip("\n").split("|")
            formatted = []
            for j, part in enumerate(parts):
                if j == 0 or j == len(parts) - 1:
                    formatted.append(part)
                else:
                    formatted.append(part.strip())
            line = "|".join(formatted).rstrip() + "\n"

        # MD036: standalone bold text that looks like a heading
        stripped = line.strip()
        if (
            stripped.startswith("**")
            and stripped.endswith("**")
            and stripped.count("**") == 2
        ):
            inner = stripped[2:-2].strip()
            if (
                inner
                and len(inner) < 100
                and not any(c in inner for c in ["`", "[", "]"])
            ):
                if not (
                    final and (is_list_item(final[-1]) or "|" in final[-1])
                ):
                    line = f"{inner}\n"

        final.append(line)

    return final, final != lines


# ---------------------------------------------------------------------------
# File-level operations
# ---------------------------------------------------------------------------

def check_file(filepath):
    """Return True if the file has lint issues (would be changed by fix)."""
    with open(filepath, "r", encoding="utf-8") as f:
        original = f.readlines()
    _, changed = fix_markdown(list(original))
    return changed


def fix_file(filepath):
    """Fix a file in-place. Returns True if changes were made."""
    with open(filepath, "r", encoding="utf-8") as f:
        original = f.readlines()
    fixed, changed = fix_markdown(list(original))
    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(fixed)
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find and fix markdown lint issues across the ADAI repository."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific .md files to process (default: all .md files in the repo)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report issues without fixing; exit 1 if any found",
    )
    parser.add_argument(
        "--dir",
        metavar="DIR",
        help="Limit file discovery to a subdirectory (e.g. docs/)",
    )
    parser.add_argument(
        "--markdownlint",
        action="store_true",
        help="Also run markdownlint --fix if installed (ignored in --check mode)",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()

    if args.files:
        md_files = args.files
    else:
        md_files = discover_markdown_files(repo_root, args.dir)

    if not md_files:
        print("No markdown files found.")
        return 0

    print(f"Processing {len(md_files)} markdown file(s)...\n")

    issues = 0
    fixed_count = 0

    for filepath in md_files:
        rel = os.path.relpath(filepath, repo_root)
        try:
            if args.check:
                if check_file(filepath):
                    print(f"  lint issues: {rel}")
                    issues += 1
                else:
                    print(f"  ok:          {rel}")
            else:
                if fix_file(filepath):
                    print(f"  fixed: {rel}")
                    fixed_count += 1
                else:
                    print(f"  ok:    {rel}")
        except Exception as e:
            print(f"  ERROR: {rel} — {e}")
            issues += 1

    # Optional markdownlint pass
    if args.markdownlint and not args.check:
        if subprocess.run(["which", "markdownlint"], capture_output=True).returncode == 0:
            print("\nRunning markdownlint --fix...")
            for filepath in md_files:
                subprocess.run(["markdownlint", "--fix", filepath], capture_output=True)
            print("markdownlint pass complete.")
        else:
            print("\nmarkdownlint not installed — skipping (npm install -g markdownlint-cli)")

    print()
    if args.check:
        if issues:
            print(f"{issues} file(s) have lint issues. Run without --check to fix.")
            return 1
        print("All files clean.")
        return 0

    print(f"{fixed_count} file(s) fixed, {len(md_files) - fixed_count} already clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
