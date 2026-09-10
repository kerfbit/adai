#!/usr/bin/env python3

# @adai-status: beta        (capped by TD-045 — see TECHNICAL_DEBT.md)
# @adai-version: 0.6.1
# @adai-reviewed: 2026-09-10

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
DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"Dashboard server running at http://localhost:{PORT}/dashboard.html")
        print(f"Make sure metrics-api-server is running on port 8081")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
