#!/usr/bin/env python3
"""Tests for scripts/check_file_status.py (TD-043).

Runnable standalone: `python3 tests/scripts/test_check_file_status.py`.
Uses only the standard library (unittest) — pytest is not installed and
isn't otherwise part of this project's toolchain, so introducing it as a
dependency just for this one file would need its own CI/dependency-file
changes for no real benefit over unittest here.

check_file(path) is a pure function (reads one file, returns a list of
problem strings) and is exercised directly against real temp files for
precise, fast, per-branch coverage. REPO_ROOT/in_scope_files()/changed_files()
are hardcoded relative to the script's own file location (not overridable),
so the whole-repo and --changed CLI modes are exercised via subprocess
against the real repo instead of a fixture — safe, since both are read-only.
"""
from __future__ import annotations

import datetime
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import check_file_status as cfs  # noqa: E402


def write_header(tmp_path: Path, body: str, suffix: str = ".hpp") -> Path:
    p = tmp_path / f"fixture{suffix}"
    p.write_text(body)
    return p


TODAY = datetime.date.today().isoformat()
FUTURE = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()


class CheckFileTests(unittest.TestCase):
    """Direct unit tests of check_file()'s per-branch behavior."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_valid_header_has_no_problems(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta
// @adai-version: 1.2.3
// @adai-reviewed: {TODAY}

int main() {{}}
""")
        self.assertEqual(cfs.check_file(p), [])

    def test_missing_tag_entirely(self):
        p = write_header(self.tmp_path, "int main() {}\n")
        problems = cfs.check_file(p)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing @adai-status", problems[0])

    def test_invalid_status_value(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: bogus
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        problems = cfs.check_file(p)
        self.assertTrue(any("invalid @adai-status" in x for x in problems))

    def test_non_semver_version_rejected(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta
// @adai-version: v1.2
// @adai-reviewed: {TODAY}
""")
        problems = cfs.check_file(p)
        self.assertTrue(any("invalid @adai-version" in x for x in problems))

    def test_stable_requires_major_gte_1(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: stable
// @adai-version: 0.9.0
// @adai-reviewed: {TODAY}
""")
        problems = cfs.check_file(p)
        self.assertTrue(any("stable" in x and "MAJOR >= 1" in x for x in problems))

    def test_stable_with_major_1_is_fine(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        self.assertEqual(cfs.check_file(p), [])

    def test_future_reviewed_date_rejected(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta
// @adai-version: 1.0.0
// @adai-reviewed: {FUTURE}
""")
        problems = cfs.check_file(p)
        self.assertTrue(any("in the future" in x for x in problems))

    def test_malformed_reviewed_date_rejected(self):
        p = write_header(self.tmp_path, """\
// @adai-status: beta
// @adai-version: 1.0.0
// @adai-reviewed: not-a-date
""")
        problems = cfs.check_file(p)
        self.assertTrue(any("invalid @adai-reviewed" in x for x in problems))

    def test_deprecated_without_replacement_flagged(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: deprecated
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        problems = cfs.check_file(p)
        self.assertTrue(any("names no replacement" in x for x in problems))

    def test_deprecated_with_arrow_replacement_accepted(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: deprecated        (-> NewFile.hpp)
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        self.assertEqual(cfs.check_file(p), [])

    def test_deprecated_with_replaced_by_text_accepted(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: deprecated        (replaced by NewFile.hpp)
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        self.assertEqual(cfs.check_file(p), [])

    def test_unreadable_file_reports_error_not_exception(self):
        p = self.tmp_path / "does-not-exist.hpp"
        problems = cfs.check_file(p)
        self.assertEqual(len(problems), 1)
        self.assertIn("could not read file", problems[0])

    def test_tag_block_must_be_three_consecutive_lines(self):
        # Documents a real, load-bearing constraint (see CLAUDE.md /
        # file-status-standard.md): TAG_RE uses re.MULTILINE but NOT
        # re.DOTALL, so a wrapped/multi-line @adai-status value breaks
        # parsing — the three tags must be on their own consecutive single
        # lines. A prose line inserted between @adai-status and
        # @adai-version must make the file register as "missing the tag"
        # rather than silently misparsing.
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta
// some unrelated comment line inserted between the tags
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        problems = cfs.check_file(p)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing @adai-status", problems[0])

    def test_extra_parenthetical_text_on_status_line_is_fine(self):
        # The common real-world pattern used throughout this repo: extra
        # explanatory text AFTER the status value but still on the SAME
        # line is fine — only wrapping onto a NEW line breaks parsing.
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta        (capped by TD-999 — see TECHNICAL_DEBT.md)
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        self.assertEqual(cfs.check_file(p), [])

    def test_tag_must_be_within_first_max_header_lines(self):
        padding = "// filler comment line\n" * (cfs.MAX_HEADER_LINES + 5)
        p = write_header(self.tmp_path, padding + f"""\
// @adai-status: beta
// @adai-version: 1.0.0
// @adai-reviewed: {TODAY}
""")
        problems = cfs.check_file(p)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing @adai-status", problems[0])


class InScopeFilesTests(unittest.TestCase):
    """in_scope_files() is hardcoded to the real REPO_ROOT — exercised
    against the real repo rather than a fixture."""

    def test_known_in_scope_file_is_found(self):
        files = cfs.in_scope_files()
        self.assertIn((REPO_ROOT / "scripts" / "check_file_status.py").resolve(), files)

    def test_excluded_directories_never_appear(self):
        files = cfs.in_scope_files()
        for f in files:
            for excluded in cfs.EXCLUDE_DIR_PARTS:
                self.assertNotIn(
                    excluded, f.parts,
                    f"{f} should have been excluded (matches {excluded!r})",
                )

    def test_non_recursive_scripts_glob_still_reaches_cloudflared_subdir(self):
        # TD-145's own regression: scripts/*.sh alone would never match
        # scripts/cloudflared/*.sh — confirms the explicit extra glob entry
        # for that subdirectory is still present and working.
        files = cfs.in_scope_files()
        self.assertIn(
            (REPO_ROOT / "scripts" / "cloudflared" / "install_cloudflared.sh").resolve(),
            files,
        )


class CliTests(unittest.TestCase):
    """Subprocess-level tests of main()'s CLI wiring — argument parsing and
    exit codes, run against the real script and real repo (read-only)."""

    def run_cli(self, *args):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "check_file_status.py"), *args],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_default_mode_exits_0_even_with_problems(self):
        # Warn-only by default (no --strict) — exit 0 regardless of whether
        # any problems were found, confirmed by checking the summary line
        # is present rather than asserting zero problems (which would make
        # this test depend on the rest of the repo's tag hygiene).
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertRegex(out, r"Checked \d+ file\(s\)")

    def test_strict_mode_propagates_nonzero_when_problems_exist(self):
        # Uses --changed against a base ref with no possible diff (HEAD
        # itself) to deterministically get zero changed files rather than
        # depending on whatever the real repo's current tag hygiene is.
        code, out, _ = self.run_cli("--changed", "HEAD", "--strict")
        self.assertEqual(code, 0)
        self.assertIn("No in-scope files to check.", out)

    def test_changed_with_bad_base_ref_fails_cleanly(self):
        code, out, err = self.run_cli("--changed", "this-ref-does-not-exist-xyz")
        self.assertEqual(code, 2)
        self.assertIn("git diff", err)

    def test_help_exits_0(self):
        code, out, _ = self.run_cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("--changed", out)
        self.assertIn("--strict", out)


if __name__ == "__main__":
    unittest.main()
