#!/usr/bin/env python3

# @adai-status: beta        (TD-045 resolved — real test suite added, see tests/scripts/test_serve_dashboard.py)
# @adai-version: 0.6.3
# @adai-reviewed: 2026-09-11

"""
Simple HTTP server to serve the dashboard
Usage: python3 serve_dashboard.py
"""
import http.server
import socketserver
import os

PORT = 8082
# dashboard.html lives at the repo root (see docs/operations/OPERATIONS_MANUAL.md's
# component table and package_server_bundle.sh, which both place it there), not
# alongside this script in scripts/ — serving from this script's own directory
# 404ed on the documented path. TD-110.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_PATH = os.path.join(REPO_ROOT, "dashboard.html")


class DashboardOnlyHandler(http.server.BaseHTTPRequestHandler):
    # TD-144: TD-110 pointed this server at the repo root so it could actually
    # find dashboard.html there, but the handler at the time (SimpleHTTPRequestHandler
    # with directory=REPO_ROOT) serves EVERY file under that directory, not just
    # dashboard.html — including .git/ (the full commit history, the same tree
    # that once carried the now-rotated GitHub PAT), config.conf, vocab.txt, and
    # training_sessions/ checkpoints, plus a browsable directory listing of all of
    # it. Reproduced directly: a plain `python3 -m http.server` bound at a repo
    # root served `GET /.git/config` and `GET /config.conf` with 200 OK and their
    # real contents, and listed both in the directory index — and this server
    # binds to all interfaces ("", PORT), not just localhost, so the exposure
    # isn't even confined to the local machine. dashboard.html is a single
    # self-contained file (confirmed against the last tracked revision before it
    # was untracked as a generated artifact: its only external references are a
    # CDN <script> and two /favicon requests browsers already tolerate 404ing) —
    # it has no sibling assets that need serving. So instead of trying to
    # allowlist individual sensitive paths within the repo tree, stop serving a
    # directory at all: only ever answer GET /dashboard.html or / with that one
    # file's contents, and 404 everything else.
    def do_GET(self):
        if self.path in ("/", "/dashboard.html"):
            self._serve_dashboard()
        else:
            self.send_error(404, "Not Found")

    def _serve_dashboard(self):
        try:
            with open(DASHBOARD_PATH, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404, "dashboard.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, fmt, *args):
        # Match SimpleHTTPRequestHandler's default behavior (log to stderr).
        http.server.BaseHTTPRequestHandler.log_message(self, fmt, *args)


if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), DashboardOnlyHandler) as httpd:
        print(f"Dashboard server running at http://localhost:{PORT}/dashboard.html")
        print(f"Make sure metrics-api-server is running on port 8081")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
