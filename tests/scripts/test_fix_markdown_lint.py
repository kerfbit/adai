#!/usr/bin/env python3
"""Tests for scripts/fix_markdown_lint.py (TD-045).

Runnable standalone: `python3 tests/scripts/test_fix_markdown_lint.py`.
Standard library only (see test_check_file_status.py's module docstring for
why unittest rather than pytest).

fix_markdown(lines) is a pure function (list of lines in, (fixed_lines,
changed) out) — most coverage here exercises it directly, one rule at a
time, plus a regression test for TD-119's specific fenced-code-block
corruption bug. CLI-level tests are careful to NEVER invoke the script in
its default (no positional files, no --dir) fixing mode from inside this
repo: find_repo_root() walks up from the SCRIPT's own file location (not
CWD) to find .git, so an unscoped, non---check invocation would rewrite
every .md file in the REAL repo in place. Every CLI test here either uses
--check (read-only) or passes explicit scratch file paths.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "fix_markdown_lint.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import fix_markdown_lint as fml  # noqa: E402


def lines_of(text: str) -> list[str]:
    """Split text into lines the way readlines() would (newline-terminated)."""
    return text.splitlines(keepends=True)


class Md009TrailingWhitespaceTests(unittest.TestCase):
    def test_trailing_spaces_stripped(self):
        fixed, changed = fml.fix_markdown(lines_of("Some text.   \n"))
        self.assertTrue(changed)
        self.assertEqual(fixed, ["Some text.\n"])

    def test_clean_line_unchanged(self):
        fixed, changed = fml.fix_markdown(lines_of("Some text.\n"))
        self.assertFalse(changed)


class Md022HeadingBlankLinesTests(unittest.TestCase):
    def test_blank_line_inserted_before_and_after_heading(self):
        fixed, changed = fml.fix_markdown(lines_of("Para one.\n# Heading\nPara two.\n"))
        self.assertTrue(changed)
        self.assertEqual(fixed, ["Para one.\n", "\n", "# Heading\n", "\n", "Para two.\n"])

    def test_already_surrounded_heading_unchanged(self):
        text = "Para one.\n\n# Heading\n\nPara two.\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertFalse(changed)


class Md031CodeFenceBlankLinesTests(unittest.TestCase):
    def test_blank_line_before_and_after_fence(self):
        text = "Some text.\n```python\ncode\n```\nMore text.\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertTrue(changed)
        self.assertEqual(
            fixed,
            ["Some text.\n", "\n", "```python\n", "code\n", "```\n", "\n", "More text.\n"],
        )


class Md032ListBlankLinesTests(unittest.TestCase):
    def test_blank_line_before_and_after_list(self):
        text = "Para.\n- item one\n- item two\nAfter.\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertTrue(changed)
        self.assertEqual(
            fixed,
            ["Para.\n", "\n", "- item one\n", "- item two\n", "\n", "After.\n"],
        )

    def test_numbered_list_recognized(self):
        self.assertTrue(fml.is_list_item("1. first"))
        self.assertTrue(fml.is_list_item("2) second"))
        self.assertFalse(fml.is_list_item("Not a list."))


class Md040FenceLanguageTests(unittest.TestCase):
    def test_bare_fence_gets_text_language(self):
        fixed, changed = fml.fix_markdown(lines_of("```\ncode\n```\n"))
        self.assertTrue(changed)
        self.assertEqual(fixed[0], "```text\n")

    def test_fence_with_language_unchanged(self):
        fixed, changed = fml.fix_markdown(lines_of("```python\ncode\n```\n"))
        self.assertEqual(fixed[0], "```python\n")


class Md060TableCompactionTests(unittest.TestCase):
    def test_spaces_around_pipes_removed_from_inner_cells(self):
        # split("|") on a line that both starts and ends with "|" produces
        # an EMPTY string for parts[0] and parts[-1] (the boundary before
        # the first pipe and after the last) — those two are left alone by
        # this rule (already empty, nothing to strip); every part *between*
        # them is an inner cell and gets its surrounding whitespace
        # stripped. So "| a | b | c |" -> parts ["", " a ", " b ", " c ",
        # ""] -> ["", "a", "b", "c", ""] -> "|a|b|c|", not "| a|b|c |" as a
        # naive "leave the outermost cell text alone" reading might suggest.
        fixed, changed = fml.fix_markdown(lines_of("| a | b | c |\n"))
        self.assertTrue(changed)
        self.assertEqual(fixed, ["|a|b|c|\n"])

    def test_already_compact_table_unchanged_by_md060(self):
        # The genuinely idempotent, fully-compact form per the rule above:
        # no space anywhere around any pipe, including immediately after
        # the opening "|" and before the closing "|".
        text = "|a|b|c|\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertFalse(changed)


class Md036EmphasisAsHeadingTests(unittest.TestCase):
    def test_standalone_bold_line_converted(self):
        fixed, changed = fml.fix_markdown(lines_of("**Bold Heading**\n"))
        self.assertTrue(changed)
        self.assertEqual(fixed, ["Bold Heading\n"])

    def test_bold_with_backtick_left_alone(self):
        # Documented exclusion: inline code inside the bold text disqualifies
        # it from being treated as a heading-like line.
        text = "**Use `foo()` here**\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertFalse(changed)

    def test_bold_line_in_table_row_left_alone(self):
        text = "| **Bold** | value |\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        # MD060 still compacts the table row itself, but the MD036 rule must
        # not also fire on it (the line was never a standalone bold line).
        self.assertNotEqual(fixed, ["Bold\n"])


class Md029NotImplementedTests(unittest.TestCase):
    def test_inconsistent_ordered_list_numbering_is_not_touched(self):
        # Documented, deliberate non-feature (TD-119): this tool does NOT
        # renumber ordered lists. A file whose only issue is numbering must
        # come back unchanged, not silently "fixed" in a way the docstring
        # doesn't promise.
        text = "1. first\n3. second\n5. third\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertFalse(changed)
        self.assertEqual(fixed, lines_of(text))


class Td119CodeFenceProtectionRegressionTests(unittest.TestCase):
    """TD-119: the second pass (tables/MD036) used to have no code-fence
    tracking of its own and reached into fenced code blocks, corrupting any
    code example containing 2+ literal "|" characters."""

    def test_bash_pipeline_inside_fence_not_corrupted(self):
        text = "```bash\ncat foo.txt | grep bar | wc -l\n```\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertIn("cat foo.txt | grep bar | wc -l\n", fixed)

    def test_cpp_logical_or_inside_fence_not_corrupted(self):
        text = "```cpp\nif (a || b) { return c || d; }\n```\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertIn("if (a || b) { return c || d; }\n", fixed)

    def test_bold_line_inside_fence_not_converted_to_heading_text(self):
        text = "```text\n**not a heading, just text in a code block**\n```\n"
        fixed, changed = fml.fix_markdown(lines_of(text))
        self.assertIn("**not a heading, just text in a code block**\n", fixed)


class DiscoverAndFileOpsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_discover_finds_md_files_recursively_and_skips_build_dirs(self):
        (self.tmp_path / "docs").mkdir()
        (self.tmp_path / "docs" / "a.md").write_text("# A\n")
        (self.tmp_path / "build").mkdir()
        (self.tmp_path / "build" / "generated.md").write_text("# skip me\n")
        found = fml.discover_markdown_files(self.tmp_path)
        rels = [str(Path(f).relative_to(self.tmp_path)) for f in found]
        self.assertIn(str(Path("docs") / "a.md"), rels)
        self.assertNotIn(str(Path("build") / "generated.md"), rels)

    def test_check_file_does_not_modify_the_file(self):
        p = self.tmp_path / "dirty.md"
        p.write_text("Trailing space   \n")
        original = p.read_text()
        has_issues = fml.check_file(p)
        self.assertTrue(has_issues)
        self.assertEqual(p.read_text(), original, "check_file must be read-only")

    def test_fix_file_modifies_in_place_and_reports_true(self):
        p = self.tmp_path / "dirty.md"
        p.write_text("Trailing space   \n")
        changed = fml.fix_file(p)
        self.assertTrue(changed)
        self.assertEqual(p.read_text(), "Trailing space\n")

    def test_fix_file_on_already_clean_file_reports_false_and_does_not_rewrite(self):
        p = self.tmp_path / "clean.md"
        p.write_text("Clean line.\n")
        mtime_before = p.stat().st_mtime_ns
        changed = fml.fix_file(p)
        self.assertFalse(changed)
        self.assertEqual(p.stat().st_mtime_ns, mtime_before, "must not rewrite an already-clean file")


class CliTests(unittest.TestCase):
    """Subprocess-level smoke tests. NEVER invokes the script's default
    (unscoped, non---check) fixing mode — that would rewrite every .md file
    in the real repo, since find_repo_root() resolves via the script's own
    file location, not CWD."""

    def run_cli(self, *args, cwd=None):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            cwd=cwd or REPO_ROOT, capture_output=True, text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_help_exits_0(self):
        code, out, _ = self.run_cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("--check", out)
        self.assertIn("--dir", out)

    def test_check_mode_default_scope_is_read_only_and_exits_0_or_1(self):
        # Safe specifically because --check never writes anything, even
        # though it still scans the whole real repo by default.
        code, out, _ = self.run_cli("--check")
        self.assertIn(code, (0, 1))
        self.assertIn("file(s)", out)

    def test_explicit_scratch_files_are_fixed_in_place_without_touching_the_repo(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scratch.md"
            p.write_text("Trailing space   \n")
            code, out, _ = self.run_cli(str(p))
            self.assertEqual(code, 0)
            self.assertIn("fixed:", out)
            self.assertEqual(p.read_text(), "Trailing space\n")

    def test_nonexistent_explicit_file_reports_error_but_does_not_crash(self):
        # Fix mode (no --check) returns 0 unconditionally at the end of
        # main() regardless of per-file errors along the way — only
        # --check mode's own `if issues: return 1` branch considers the
        # issues counter. Confirmed this is main()'s actual, deliberate
        # control flow (not asserting a bug): the important behavior this
        # test pins down is that a missing file is caught and reported,
        # not an unhandled traceback.
        code, out, err = self.run_cli("/tmp/definitely-does-not-exist-td045.md")
        self.assertEqual(code, 0)
        self.assertIn("ERROR", out)
        self.assertEqual(err, "", "must not crash with a traceback")

    def test_nonexistent_explicit_file_with_check_mode_exits_1(self):
        code, out, _ = self.run_cli("--check", "/tmp/definitely-does-not-exist-td045.md")
        self.assertEqual(code, 1)
        self.assertIn("ERROR", out)


if __name__ == "__main__":
    unittest.main()
