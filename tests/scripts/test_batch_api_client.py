#!/usr/bin/env python3
"""Tests for scripts/batch_api_client.py (TD-045).

Runnable standalone: `python3 tests/scripts/test_batch_api_client.py`.
Standard library + requests only (requests is already a real dependency of
this script, confirmed installed).

BatchChatbotClient's methods are exercised directly against a definitely-
unreachable host (127.0.0.1 on a reserved, essentially-never-listened-on
port) so this suite has no dependency on whether a real chatbot_api_server
happens to be running, or on whatever else might happen to be bound to the
script's own hardcoded default port 8080 in a given environment. The one
subprocess-level smoke test of the whole script (main()) is tolerant of
either outcome for the same reason.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "batch_api_client.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import batch_api_client as bac  # noqa: E402

# Port 1 is reserved (requires root to bind on Linux) and nothing this
# suite runs would ever legitimately listen there — connections to it fail
# fast and deterministically with ConnectionRefusedError/ConnectionError,
# with zero dependency on real network access or the ambient port state of
# whatever machine runs this test.
UNREACHABLE_BASE_URL = "http://127.0.0.1:1"


class BatchChatbotClientTests(unittest.TestCase):
    def setUp(self):
        self.client = bac.BatchChatbotClient(base_url=UNREACHABLE_BASE_URL)

    def test_base_url_is_configurable(self):
        self.assertEqual(self.client.base_url, UNREACHABLE_BASE_URL)

    def test_health_against_unreachable_host_raises_connection_error(self):
        with self.assertRaises(Exception) as ctx:
            self.client.health()
        # requests.exceptions.ConnectionError specifically, not some other
        # unrelated exception class — confirms this is genuinely a network-
        # level failure, not e.g. an AttributeError from a typo.
        self.assertIsInstance(ctx.exception, Exception)
        import requests
        self.assertIsInstance(ctx.exception, requests.exceptions.ConnectionError)

    def test_chat_against_unreachable_host_raises(self):
        with self.assertRaises(Exception):
            self.client.chat("hello")

    def test_batch_chat_against_unreachable_host_raises(self):
        with self.assertRaises(Exception):
            self.client.batch_chat(["hello", "world"])

    def test_batch_chat_session_without_session_ids_omits_the_key(self):
        # Pure request-construction behavior, verified without a network
        # call by inspecting what requests.Session.post would be given —
        # patched here rather than exercised over a real socket.
        captured = {}

        class FakeResponse:
            def json(self):
                return {"success": True}

        def fake_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

        self.client.session.post = fake_post
        self.client.batch_chat_session(["hi"])
        self.assertNotIn("session_ids", captured["json"])

    def test_batch_chat_session_with_session_ids_includes_the_key(self):
        captured = {}

        class FakeResponse:
            def json(self):
                return {"success": True}

        def fake_post(url, json=None, **kwargs):
            captured["json"] = json
            return FakeResponse()

        self.client.session.post = fake_post
        self.client.batch_chat_session(["hi"], session_ids=["abc"])
        self.assertEqual(captured["json"]["session_ids"], ["abc"])


class MainSmokeTests(unittest.TestCase):
    """Subprocess-level smoke test of the whole script, per TD-045's own
    bar ("invoke it, assert it doesn't crash"). main() hardcodes base_url
    to localhost:8080 with no CLI override, so this can't force a
    guaranteed-unreachable target — it instead accepts either of the two
    outcomes the (TD-045-fixed) exception handling now covers: a clean
    "cannot connect" message, or a clean "not valid JSON" message if
    something non-JSON happens to be bound to 8080 in the test
    environment. Either way, main() must exit 0 with no traceback."""

    def test_main_does_not_crash_regardless_of_what_if_anything_is_on_8080(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(proc.returncode, 0, f"stderr was:\n{proc.stderr}")
        self.assertEqual(proc.stderr, "", "must not crash with a traceback")
        self.assertIn("Batch Processing API Client Examples", proc.stdout)
        self.assertTrue(
            "Cannot connect to server" in proc.stdout
            or "not with valid JSON" in proc.stdout
            or "✓ Server Status" in proc.stdout,
            f"expected one of the three documented health-check outcomes; got:\n{proc.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
