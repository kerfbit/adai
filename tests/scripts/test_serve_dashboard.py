#!/usr/bin/env python3
"""Tests for scripts/serve_dashboard.py (TD-045).

Runnable standalone: `python3 tests/scripts/test_serve_dashboard.py`.
Standard library only.

DashboardOnlyHandler is exercised against a real socketserver.TCPServer
bound to 127.0.0.1 on port 0 (OS-assigned free port) in a background
thread — genuine HTTP request/response behavior, not a mock. DASHBOARD_PATH
is monkeypatched on the imported module to a scratch file rather than
writing into the real repo root (the module computes it from its own file
location, so there's no CLI/env override to use instead) — a handler
method referencing the global by name at call time picks up the patched
value with no need to touch the real dashboard.html (which doesn't even
exist in this checkout — deliberately untracked as a generated artifact).
"""
from __future__ import annotations

import http.client
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "serve_dashboard.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import serve_dashboard as sd  # noqa: E402
import socketserver  # noqa: E402


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


class DashboardHandlerTests(unittest.TestCase):
    """Real HTTP server, bound to an OS-assigned free port (never the
    script's own hardcoded 8082, so this never collides with anything real
    that might be using it)."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._real_dashboard_path = sd.DASHBOARD_PATH
        self.server = socketserver.TCPServer(("127.0.0.1", 0), sd.DashboardOnlyHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        sd.DASHBOARD_PATH = self._real_dashboard_path
        self._tmpdir.cleanup()

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp, body

    def test_root_serves_dashboard_content(self):
        dashboard = Path(self._tmpdir.name) / "dashboard.html"
        dashboard.write_text("<html>TD-045 fixture dashboard</html>")
        sd.DASHBOARD_PATH = str(dashboard)

        resp, body = self.get("/")
        self.assertEqual(resp.status, 200)
        self.assertEqual(body, b"<html>TD-045 fixture dashboard</html>")
        self.assertEqual(resp.getheader("Content-Type"), "text/html; charset=utf-8")
        self.assertEqual(resp.getheader("Content-Length"), str(len(body)))

    def test_explicit_dashboard_html_path_serves_same_content(self):
        dashboard = Path(self._tmpdir.name) / "dashboard.html"
        dashboard.write_text("<html>fixture</html>")
        sd.DASHBOARD_PATH = str(dashboard)

        resp, body = self.get("/dashboard.html")
        self.assertEqual(resp.status, 200)
        self.assertEqual(body, b"<html>fixture</html>")

    def test_cors_headers_present_on_a_successful_response(self):
        dashboard = Path(self._tmpdir.name) / "dashboard.html"
        dashboard.write_text("x")
        sd.DASHBOARD_PATH = str(dashboard)

        resp, _ = self.get("/")
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", resp.getheader("Access-Control-Allow-Methods"))

    def test_missing_dashboard_file_returns_404_not_a_crash(self):
        sd.DASHBOARD_PATH = str(Path(self._tmpdir.name) / "does-not-exist.html")
        resp, _ = self.get("/")
        self.assertEqual(resp.status, 404)

    def test_any_other_path_returns_404(self):
        dashboard = Path(self._tmpdir.name) / "dashboard.html"
        dashboard.write_text("x")
        sd.DASHBOARD_PATH = str(dashboard)

        for path in ("/etc/passwd", "/.git/config", "/config.conf", "/some/random/path"):
            with self.subTest(path=path):
                resp, _ = self.get(path)
                self.assertEqual(resp.status, 404)

    def test_cors_headers_present_even_on_a_404(self):
        resp, _ = self.get("/nonexistent")
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "*")


class MainSmokeTest(unittest.TestCase):
    """Subprocess-level smoke test of the real __main__ block, including
    its hardcoded PORT=8082 bind — skipped if that port is already in use
    (this test can't safely force a different port; the script has no
    override), and torn down with SIGINT to also exercise the documented
    graceful-shutdown path."""

    def test_real_process_binds_serves_and_shuts_down_cleanly(self):
        if port_in_use(8082):
            self.skipTest("port 8082 is already in use in this environment")

        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_PATH)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.time() + 5
            started = False
            while time.time() < deadline:
                if port_in_use(8082):
                    started = True
                    break
                time.sleep(0.1)
            self.assertTrue(started, "server never started listening on 8082")

            conn = http.client.HTTPConnection("127.0.0.1", 8082, timeout=5)
            conn.request("GET", "/")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            # dashboard.html doesn't exist in this checkout, so 404 is the
            # correct, expected real-world response here — the point is
            # that the real process answered at all, cleanly.
            self.assertEqual(resp.status, 404)
        finally:
            import signal
            proc.send_signal(signal.SIGINT)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Shutting down", out)

    def test_quick_restart_after_serving_a_request_does_not_crash(self):
        # Regression test: plain socketserver.TCPServer defaults
        # allow_reuse_address to False (unlike http.server.HTTPServer,
        # which this file doesn't use). A client connection followed by a
        # restart within the OS's TIME_WAIT window (typically ~60s) used to
        # crash with "OSError: [Errno 98] Address already in use" before
        # ever printing a line — hit on every systemd auto-restart after a
        # crash, or a developer's Ctrl+C-then-immediately-rerun. Exercises
        # exactly that sequence for real: serve one request, SIGINT, then
        # immediately start a second instance and confirm IT also binds
        # and serves successfully, with no restart delay at all.
        if port_in_use(8082):
            self.skipTest("port 8082 is already in use in this environment")

        import signal

        def start_and_stop_after_one_request():
            proc = subprocess.Popen(
                [sys.executable, str(SCRIPT_PATH)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            deadline = time.time() + 5
            while time.time() < deadline:
                if port_in_use(8082):
                    break
                time.sleep(0.1)
            else:
                proc.kill()
                proc.communicate()
                self.fail("server never started listening on 8082")

            conn = http.client.HTTPConnection("127.0.0.1", 8082, timeout=5)
            conn.request("GET", "/")
            conn.getresponse().read()
            conn.close()

            proc.send_signal(signal.SIGINT)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            return proc.returncode, out, err

        code1, out1, err1 = start_and_stop_after_one_request()
        self.assertEqual(code1, 0, f"first instance failed: {err1}")

        # No delay here at all — this is the exact window the bug hit.
        code2, out2, err2 = start_and_stop_after_one_request()
        self.assertEqual(
            code2, 0,
            f"second instance failed to restart immediately after the first "
            f"served a request (TIME_WAIT regression); stderr:\n{err2}",
        )
        self.assertNotIn("Address already in use", err2)


if __name__ == "__main__":
    unittest.main()
