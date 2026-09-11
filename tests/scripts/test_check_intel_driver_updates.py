#!/usr/bin/env python3
"""Tests for scripts/check_intel_driver_updates.py (TD-045).

Runnable standalone: `python3 tests/scripts/test_check_intel_driver_updates.py`.
Standard library only.

The pure functions (diff_snapshot, load_state/save_state, print_report) are
tested directly with no network I/O at all. run()'s real per-source fetch
path is exercised by monkeypatching the module's SOURCES list to point at
127.0.0.1:1 (reserved, always-refused, no real network dependency) rather
than the 5 real Intel/GitHub/Launchpad URLs — deterministic and fast, while
still running the actual, unmodified fetch()/check_github_release()/run()
code. A separate, optional live-network test exercises the real URLs but
skips gracefully (does not fail) if the network is unavailable, since CI
environments and sandboxes vary in outbound access.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_intel_driver_updates.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import check_intel_driver_updates as cidu  # noqa: E402

UNREACHABLE_URL = "http://127.0.0.1:1/nonexistent"


def network_available() -> bool:
    try:
        urllib.request.urlopen("https://api.github.com", timeout=3)
        return True
    except Exception:
        return False


class DiffSnapshotTests(unittest.TestCase):
    def test_first_check_has_no_prior_state(self):
        changes = cidu.diff_snapshot(None, {"tag": "v1.0"})
        self.assertEqual(changes, ["first check (no prior state to compare against)"])

    def test_identical_snapshots_produce_no_changes(self):
        snap = {"tag": "v1.0", "name": "release"}
        self.assertEqual(cidu.diff_snapshot(snap, dict(snap)), [])

    def test_changed_field_is_reported(self):
        old = {"tag": "v1.0"}
        new = {"tag": "v2.0"}
        changes = cidu.diff_snapshot(old, new)
        self.assertEqual(len(changes), 1)
        self.assertIn("tag", changes[0])
        self.assertIn("v1.0", changes[0])
        self.assertIn("v2.0", changes[0])

    def test_new_field_appearing_is_reported(self):
        old = {"tag": "v1.0"}
        new = {"tag": "v1.0", "extra": "value"}
        changes = cidu.diff_snapshot(old, new)
        self.assertEqual(len(changes), 1)
        self.assertIn("extra", changes[0])


class StatePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.state_file = str(Path(self._tmpdir.name) / "state.json")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_nonexistent_state_returns_empty_dict(self):
        self.assertEqual(cidu.load_state(self.state_file), {})

    def test_save_then_load_round_trips(self):
        state = {"source_a": {"snapshot": {"tag": "v1.0"}}}
        cidu.save_state(self.state_file, state)
        self.assertEqual(cidu.load_state(self.state_file), state)

    def test_save_creates_parent_directories(self):
        nested = str(Path(self._tmpdir.name) / "a" / "b" / "state.json")
        cidu.save_state(nested, {"x": 1})
        self.assertTrue(Path(nested).exists())

    def test_corrupted_state_file_is_treated_as_empty_not_a_crash(self):
        Path(self.state_file).write_text("{not valid json")
        self.assertEqual(cidu.load_state(self.state_file), {})

    def test_save_state_is_atomic_via_tmp_plus_rename(self):
        # A save that overwrites existing valid state must never leave the
        # real file half-written if something goes wrong mid-write — the
        # os.replace(tmp, path) pattern is what guarantees that. Confirms
        # the .tmp file doesn't linger after a successful save.
        cidu.save_state(self.state_file, {"a": 1})
        self.assertFalse(Path(self.state_file + ".tmp").exists())


class PrintReportExitSemanticsTests(unittest.TestCase):
    """print_report()'s return value drives main()'s documented exit-code
    contract (0 = no changes, 2 = changes found) — tested directly against
    crafted result lists, no I/O needed."""

    def test_no_changes_returns_falsy(self):
        results = [{"id": "a", "label": "A", "link": "l", "status": "unchanged", "changes": []}]
        self.assertFalse(cidu.print_report(results, verbose=False, as_json=False))

    def test_a_change_returns_truthy(self):
        results = [{"id": "a", "label": "A", "link": "l", "status": "changed",
                    "changes": ["tag: 'v1' -> 'v2'"]}]
        self.assertTrue(cidu.print_report(results, verbose=False, as_json=False))

    def test_errors_alone_do_not_count_as_changes(self):
        results = [{"id": "a", "label": "A", "link": "l", "status": "error", "error": "timeout"}]
        self.assertFalse(cidu.print_report(results, verbose=False, as_json=False))

    def test_json_mode_still_reflects_change_status(self):
        results = [{"id": "a", "label": "A", "link": "l", "status": "changed", "changes": []}]
        self.assertTrue(cidu.print_report(results, verbose=False, as_json=True))


class RunWithUnreachableSourcesTests(unittest.TestCase):
    """Exercises the real run()/fetch()/check_github_release() code paths
    end-to-end against a deterministic, always-unreachable local URL —
    genuine network-layer behavior, zero dependency on real connectivity."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.state_file = str(Path(self._tmpdir.name) / "state.json")
        self._real_sources = cidu.SOURCES
        cidu.SOURCES = [
            {
                "id": "fake_source",
                "label": "Fake unreachable source",
                "kind": "github_release",
                "url": UNREACHABLE_URL,
                "link": UNREACHABLE_URL,
            },
        ]

    def tearDown(self):
        cidu.SOURCES = self._real_sources
        self._tmpdir.cleanup()

    def test_unreachable_source_reported_as_error_not_a_crash(self):
        results = cidu.run(self.state_file, verbose=False)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("id", results[0])

    def test_a_failed_source_does_not_corrupt_saved_state(self):
        # Documented behavior: "A per-source fetch failure ... does not
        # fail the other sources or corrupt their saved state." — for a
        # single always-failing source, that means no snapshot entry is
        # ever written for it at all.
        cidu.run(self.state_file, verbose=False)
        state = cidu.load_state(self.state_file)
        self.assertNotIn("fake_source", state)

    def test_repeated_runs_against_the_same_failure_stay_stable(self):
        first = cidu.run(self.state_file, verbose=False)
        second = cidu.run(self.state_file, verbose=False)
        self.assertEqual(first[0]["status"], second[0]["status"])


class LiveNetworkTests(unittest.TestCase):
    """Exercises the real, unmodified SOURCES list against the real
    internet. Skips (does not fail) if outbound network access isn't
    available in the environment running this suite."""

    @classmethod
    def setUpClass(cls):
        if not network_available():
            raise unittest.SkipTest("no outbound network access in this environment")

    def test_real_run_against_live_sources_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            state_file = str(Path(d) / "state.json")
            results = cidu.run(state_file, verbose=False)
            self.assertEqual(len(results), len(cidu.SOURCES))
            for r in results:
                self.assertIn(r["status"], ("changed", "unchanged", "error"))


class CliTests(unittest.TestCase):
    def run_cli(self, *args, timeout=10):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_help_exits_0(self):
        code, out, _ = self.run_cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("--state-file", out)
        self.assertIn("--json", out)

    def test_full_live_run_exits_a_documented_code_and_never_crashes(self):
        if not network_available():
            self.skipTest("no outbound network access in this environment")
        with tempfile.TemporaryDirectory() as d:
            state_file = str(Path(d) / "state.json")
            code, out, err = self.run_cli("--state-file", state_file, "--json", timeout=120)
            self.assertIn(code, (0, 1, 2), f"undocumented exit code; stderr:\n{err}")
            self.assertEqual(err, "", "must not crash with a traceback")
            parsed = json.loads(out)
            self.assertEqual(len(parsed), len(cidu.SOURCES))


if __name__ == "__main__":
    unittest.main()
