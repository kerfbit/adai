#!/usr/bin/env python3
"""Tests for scripts/monitor_training.py (TD-045).

Runnable standalone: `python3 tests/scripts/test_monitor_training.py`.
Standard library only.

The formatting helpers (format_duration, create_progress_bar,
format_loss_trend, the three display_*_dashboard functions) are pure and
tested directly. monitor_training()'s main loop never exits on its own
(only via KeyboardInterrupt/SIGINT — "Ctrl+C to quit" is the documented
way to stop it) and calls os.system('clear') every iteration, so it's
exercised as a real subprocess with a short --refresh-rate: let it run a
couple of refresh cycles, then send it a real SIGINT and confirm the
documented graceful-shutdown message and a clean exit code — the same
technique already used for model_service.sh (TD-043) and serve_dashboard.py
(TD-045) real-process tests.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "monitor_training.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import monitor_training as mt  # noqa: E402


class FormatDurationTests(unittest.TestCase):
    def test_sub_minute_shows_seconds(self):
        self.assertEqual(mt.format_duration(45.2), "45.2s")

    def test_sub_hour_shows_minutes(self):
        self.assertEqual(mt.format_duration(150), "2.5m")

    def test_hour_and_above_shows_hours(self):
        self.assertEqual(mt.format_duration(7200), "2.00h")


class CreateProgressBarTests(unittest.TestCase):
    def test_zero_total_is_an_empty_bar_not_a_division_error(self):
        bar = mt.create_progress_bar(0, 0)
        self.assertEqual(bar, "[" + " " * 40 + "]")

    def test_half_progress_is_roughly_half_filled(self):
        bar = mt.create_progress_bar(5, 10, width=20)
        self.assertIn("50.0%", bar)

    def test_full_progress_shows_100_percent(self):
        bar = mt.create_progress_bar(10, 10)
        self.assertIn("100.0%", bar)


class FormatLossTrendTests(unittest.TestCase):
    def test_fewer_than_two_points_is_not_available(self):
        self.assertEqual(mt.format_loss_trend([1.0]), "N/A")
        self.assertEqual(mt.format_loss_trend([]), "N/A")

    def test_decreasing_loss_shows_down_arrow(self):
        trend = mt.format_loss_trend([1.0, 0.8, 0.5])
        self.assertTrue(trend.startswith("↓"))

    def test_increasing_loss_shows_up_arrow(self):
        trend = mt.format_loss_trend([0.5, 0.8, 1.0])
        self.assertTrue(trend.startswith("↑"))

    def test_flat_loss_shows_flat_arrow(self):
        trend = mt.format_loss_trend([0.5, 0.5, 0.5])
        self.assertTrue(trend.startswith("→"))


class DisplayDashboardTests(unittest.TestCase):
    """These print directly to stdout — captured via capsys-equivalent
    (redirect_stdout) since unittest has no capsys. Verifies each renders
    without raising on a realistic data dict, and on a deliberately sparse
    one (missing optional keys) — every field access in the source uses
    .get() with a default, so a partial metrics file must not crash it."""

    def _capture(self, fn, data):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(data)
        return buf.getvalue()

    FULL_DATA = {
        "session_id": "run-01", "is_training": True, "timestamp": "2026-09-11T00:00:00Z",
        "current_epoch": 3, "total_epochs": 10, "current_sample": 500, "total_samples": 1000,
        "current_loss": 0.4321, "running_loss": 0.45, "current_validation_loss": 0.5,
        "current_learning_rate": 0.001, "current_gradient_norm": 1.2, "current_perplexity": 2.1,
        "best_validation_loss": 0.48, "best_epoch": 2,
        "total_samples_trained": 5000, "total_training_time_seconds": 3661,
        "samples_per_second": 12.5, "estimated_time_remaining_seconds": 1800,
        "epoch_losses": [0.9, 0.7, 0.5, 0.43],
        "epoch_validation_losses": [0.95, 0.75, 0.55],
        "epoch_learning_rates": [0.01, 0.005, 0.001],
        "epoch_durations": [120, 118, 121],
    }

    def test_full_dashboard_renders_realistic_data(self):
        out = self._capture(mt.display_full_dashboard, self.FULL_DATA)
        self.assertIn("run-01", out)
        self.assertIn("TRAINING", out)
        self.assertIn("EPOCH HISTORY", out)

    def test_full_dashboard_handles_sparse_data_without_crashing(self):
        out = self._capture(mt.display_full_dashboard, {})
        self.assertIn("N/A", out)
        self.assertIn("IDLE", out)

    def test_compact_dashboard_renders(self):
        out = self._capture(mt.display_compact_dashboard, self.FULL_DATA)
        self.assertIn("run-01", out)

    def test_compact_dashboard_handles_sparse_data(self):
        self._capture(mt.display_compact_dashboard, {})  # must not raise

    def test_minimal_dashboard_renders_one_line(self):
        out = self._capture(mt.display_minimal_dashboard, self.FULL_DATA)
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertIn("run-01", out)

    def test_minimal_dashboard_handles_sparse_data(self):
        self._capture(mt.display_minimal_dashboard, {})  # must not raise

    def test_more_than_ten_epochs_truncates_with_a_count_message(self):
        data = dict(self.FULL_DATA)
        data["epoch_losses"] = [1.0 - i * 0.05 for i in range(15)]
        out = self._capture(mt.display_full_dashboard, data)
        self.assertIn("and 5 more epochs", out)


class MainProcessSmokeTests(unittest.TestCase):
    """Real subprocess tests. --refresh-rate is set very low so a couple of
    iterations happen quickly; every invocation is torn down with a real
    SIGINT to exercise the documented graceful-shutdown path rather than
    just killing the process.

    The SIGINT is sent once the output line the test is about appears, not
    after a fixed sleep: under a parallel ctest run, interpreter startup alone
    can take longer than the old 0.5 s/2.8 s waits, which sent the signal
    before the script was in the state the test meant to exercise.
    """

    # Generous: only the wait for a marker that never appears runs this long.
    MARKER_TIMEOUT_S = 60
    EXIT_TIMEOUT_S = 15

    def _run_and_interrupt(self, extra_args, marker, delay_after_marker_s=0.0, env=None):
        proc = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT_PATH), "--refresh-rate", "0.2", *extra_args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        out_lines, err_lines = [], []
        seen = threading.Event()

        def read_stdout():
            for line in proc.stdout:
                out_lines.append(line)
                if marker in line:
                    seen.set()

        def read_stderr():
            for line in proc.stderr:
                err_lines.append(line)

        readers = [threading.Thread(target=read_stdout, daemon=True),
                   threading.Thread(target=read_stderr, daemon=True)]
        for t in readers:
            t.start()
        try:
            if not seen.wait(self.MARKER_TIMEOUT_S):
                self.fail(f"{marker!r} never appeared; stdout:\n{''.join(out_lines)}"
                          f"\nstderr:\n{''.join(err_lines)}")
            if delay_after_marker_s:
                time.sleep(delay_after_marker_s)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=self.EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.fail(f"process did not exit after SIGINT; stdout:\n{''.join(out_lines)}"
                          f"\nstderr:\n{''.join(err_lines)}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for t in readers:
                t.join(timeout=5)
        return proc.returncode, "".join(out_lines), "".join(err_lines)

    def assert_no_traceback(self, err):
        # clear_screen()'s os.system('clear') legitimately writes "TERM
        # environment variable not set." to stderr once per refresh
        # iteration when run headless (no TTY/$TERM, as in any subprocess
        # test harness) — harmless, expected noise, not a script defect.
        # What must never appear is an actual unhandled Python exception.
        self.assertNotIn("Traceback (most recent call last)", err,
                          f"process crashed with a traceback; stderr was:\n{err}")

    def test_sigint_during_the_initial_startup_delay_exits_cleanly(self):
        # TD-045 regression: main()'s fixed 2-second startup delay (before
        # the monitoring loop even begins) used to sit outside any
        # KeyboardInterrupt handling, so a SIGINT landing in that window —
        # a perfectly plausible time to press Ctrl+C — crashed with a raw
        # traceback and exit code 130 instead of the same friendly
        # "Monitoring stopped." + exit 0 that Ctrl+C anywhere else in the
        # tool's lifetime already produced. The banner's last line is printed
        # just before that delay, so signalling on it lands inside the window.
        code, out, err = self._run_and_interrupt(
            ["--summary-file", "/tmp/td045-regress-unused.json"], marker="Refresh rate:",
        )
        self.assertNotIn("Waiting for metrics file", out)  # still in the startup delay
        self.assertEqual(code, 0)
        self.assertIn("Monitoring stopped", out)
        self.assert_no_traceback(err)

    def test_help_exits_0_immediately(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--summary-file", proc.stdout)
        self.assertIn("--refresh-rate", proc.stdout)

    def test_rejects_an_invalid_format_choice(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--format", "bogus"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", proc.stderr)

    def test_missing_summary_file_shows_waiting_message_and_exits_cleanly_on_sigint(self):
        with tempfile.TemporaryDirectory() as d:
            missing = str(Path(d) / "does-not-exist.json")
            code, out, err = self._run_and_interrupt(
                ["--summary-file", missing], marker="Waiting for metrics file")
            self.assertEqual(code, 0)
            self.assertIn("Waiting for metrics file", out)
            self.assertIn("Monitoring stopped", out)
            self.assert_no_traceback(err)

    def test_invalid_json_summary_file_reports_cleanly_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text("{not valid json")
            code, out, err = self._run_and_interrupt(
                ["--summary-file", str(bad)], marker="Invalid JSON")
            self.assertEqual(code, 0)
            self.assertIn("Invalid JSON", out)
            self.assert_no_traceback(err)

    def test_valid_summary_file_renders_and_shuts_down_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            summary = Path(d) / "metrics_summary.json"
            summary.write_text(json.dumps({
                "session_id": "smoke-test-run",
                "is_training": True,
                "current_epoch": 1,
                "total_epochs": 5,
                "current_loss": 0.6,
            }))
            code, out, err = self._run_and_interrupt(
                ["--summary-file", str(summary), "--format", "minimal"],
                marker="smoke-test-run",
            )
            self.assertEqual(code, 0)
            self.assertIn("smoke-test-run", out)
            self.assert_no_traceback(err)

    def test_sigint_while_clearing_the_screen_still_stops_it(self):
        # The refresh loop used to clear the screen with os.system('clear'), and
        # system() ignores SIGINT in the parent while its child runs, so a SIGINT
        # landing during a clear was discarded and the monitor kept running (the
        # parallel-ctest flake). A `clear` that takes 2 s makes that window
        # certain: the old script ignored this SIGINT and never exited.
        with tempfile.TemporaryDirectory() as d:
            slow_clear = Path(d) / "clear"
            slow_clear.write_text("#!/bin/sh\nsleep 2\n")
            slow_clear.chmod(0o755)
            env = dict(os.environ, PATH=f"{d}{os.pathsep}{os.environ.get('PATH', '')}")
            code, out, err = self._run_and_interrupt(
                ["--summary-file", str(Path(d) / "missing.json")],
                marker="Waiting for metrics file", delay_after_marker_s=0.5, env=env,
            )
            self.assertEqual(code, 0)
            self.assertIn("Monitoring stopped", out)
            self.assert_no_traceback(err)


if __name__ == "__main__":
    unittest.main()
