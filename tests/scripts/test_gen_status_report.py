#!/usr/bin/env python3
"""Tests for scripts/gen_status_report.py (TD-043).

Runnable standalone: `python3 tests/scripts/test_gen_status_report.py`.
Standard library only (see test_check_file_status.py's module docstring for
why unittest rather than pytest).

main() writes directly to the real, hardcoded OUT_PATH (docs/development/
PRODUCTION_READINESS.md) — every test that calls main() monkeypatches
OUT_PATH to a scratch file first and restores it in a finally block, so this
suite never overwrites the real generated report.
"""
from __future__ import annotations

import datetime
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gen_status_report as gsr  # noqa: E402

TODAY = datetime.date.today()


def write_header(tmp_path: Path, body: str, name: str = "fixture.hpp") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


class ComponentOfTests(unittest.TestCase):
    def test_top_level_directory_name(self):
        p = REPO_ROOT / "src" / "Foo.cpp"
        self.assertEqual(gsr.component_of(p), "src")

    def test_nested_path_still_returns_top_level_component(self):
        p = REPO_ROOT / "android" / "app" / "src" / "main" / "Foo.kt"
        self.assertEqual(gsr.component_of(p), "android")

    def test_scripts_subdirectory_still_returns_scripts(self):
        p = REPO_ROOT / "scripts" / "cloudflared" / "install_cloudflared.sh"
        self.assertEqual(gsr.component_of(p), "scripts")


class ReadTagTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_basic_tag_extraction(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta
// @adai-version: 1.2.3
// @adai-reviewed: {TODAY.isoformat()}
""")
        tag = gsr.read_tag(p)
        self.assertEqual(tag["status"], "beta")
        self.assertEqual(tag["version"], "1.2.3")
        self.assertEqual(tag["reviewed"], TODAY.isoformat())
        self.assertIsNone(tag["td_ref"])

    def test_td_ref_extracted_from_status_line(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta        (capped by TD-043 — see TECHNICAL_DEBT.md)
// @adai-version: 1.2.3
// @adai-reviewed: {TODAY.isoformat()}
""")
        tag = gsr.read_tag(p)
        self.assertEqual(tag["td_ref"], "TD-043")

    def test_first_td_ref_wins_when_multiple_are_mentioned(self):
        p = write_header(self.tmp_path, f"""\
// @adai-status: beta        (TD-001 fully resolved; capped by TD-002)
// @adai-version: 1.2.3
// @adai-reviewed: {TODAY.isoformat()}
""")
        tag = gsr.read_tag(p)
        self.assertEqual(tag["td_ref"], "TD-001")

    def test_missing_tag_returns_none(self):
        p = write_header(self.tmp_path, "int main() {}\n")
        self.assertIsNone(gsr.read_tag(p))

    def test_docstring_mentioning_adai_status_before_the_real_tag_is_not_mistaken_for_it(self):
        # The exact scenario this function's own comment describes: a file
        # (like check_file_status.py and gen_status_report.py themselves)
        # whose module docstring mentions "@adai-status" in prose, ahead of
        # the real tag block further down. Must extract the REAL tag's
        # values, not misparse the docstring line.
        p = write_header(self.tmp_path, f"""\
\"\"\"Some tool that validates the @adai-status tag on other files.\"\"\"
# @adai-status: beta
# @adai-version: 2.0.0
# @adai-reviewed: {TODAY.isoformat()}
""", name="fixture.py")
        tag = gsr.read_tag(p)
        self.assertEqual(tag["status"], "beta")
        self.assertEqual(tag["version"], "2.0.0")


class MainReportGenerationTests(unittest.TestCase):
    """main() is exercised for real against the real repo's in_scope_files(),
    with OUT_PATH monkeypatched to a scratch file so the real generated
    report is never touched."""

    def setUp(self):
        # main()'s last line does OUT_PATH.relative_to(REPO_ROOT) — the
        # scratch path has to actually live under REPO_ROOT for that not to
        # raise, so the temp dir is created there rather than under the
        # system default (/tmp), instead of monkeypatching REPO_ROOT itself
        # (which in_scope_files() etc. also depend on more broadly).
        self._tmpdir = tempfile.TemporaryDirectory(dir=str(REPO_ROOT))
        self.scratch_out = Path(self._tmpdir.name) / "SCRATCH_PRODUCTION_READINESS.md"
        self._real_out_path = gsr.OUT_PATH
        gsr.OUT_PATH = self.scratch_out

    def tearDown(self):
        gsr.OUT_PATH = self._real_out_path
        self._tmpdir.cleanup()

    def test_real_out_path_is_genuinely_untouched_by_this_suite(self):
        # Sanity check on the monkeypatch itself, not on main()'s logic —
        # confirms this test file cannot accidentally clobber the real
        # generated report no matter what main() does.
        self.assertNotEqual(gsr.OUT_PATH, self._real_out_path)

    def test_main_returns_0_and_writes_the_scratch_file(self):
        rc = gsr.main()
        self.assertEqual(rc, 0)
        self.assertTrue(self.scratch_out.exists())

    def test_report_has_expected_structure(self):
        gsr.main()
        content = self.scratch_out.read_text()
        self.assertIn("# Production Readiness Report", content)
        self.assertIn("do not hand-edit, re-run the script instead", content)
        self.assertIn("## Summary by component", content)
        self.assertIn("## Untagged files", content)
        self.assertIn("## Experimental / beta files", content)
        self.assertIn(f"## Stable files overdue for re-attestation (> {gsr.STALE_AFTER_DAYS} days)", content)
        self.assertRegex(content, r"Generated: \d{4}-\d{2}-\d{2} · \d+ in-scope file\(s\) scanned\.")

    def test_percentage_line_is_internally_consistent(self):
        gsr.main()
        content = self.scratch_out.read_text()
        m = re.search(r"\*\*(\d+)/(\d+) files \((\d+)%\) are tagged `stable`\.\*\*", content)
        self.assertIsNotNone(m, f"percentage summary line not found in:\n{content[:500]}")
        stable, total, pct = int(m.group(1)), int(m.group(2)), int(m.group(3))
        self.assertLessEqual(stable, total)
        expected_pct = round(100 * stable / total) if total else 0
        # Python's round() and the script's f"{pct:.0f}" both round-half-to-even;
        # allow a difference of at most 1 to absorb that edge case rather than
        # replicating the exact float-formatting rule here.
        self.assertLessEqual(abs(pct - expected_pct), 1)

    def test_summary_table_row_counts_match_column_sum(self):
        gsr.main()
        content = self.scratch_out.read_text()
        table_start = content.index("| Component |")
        table_section = content[table_start:content.index("\n\n", table_start)]
        for line in table_section.splitlines()[2:]:  # skip header + separator
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            counts = [int(c) for c in cells[1:-1]]
            total_cell = int(cells[-1])
            self.assertEqual(
                sum(counts), total_cell,
                f"row {cells!r}: status counts don't sum to the row's own Total column",
            )

    def test_idempotent_when_repo_state_is_unchanged(self):
        gsr.main()
        first = self.scratch_out.read_text()
        gsr.main()
        second = self.scratch_out.read_text()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
