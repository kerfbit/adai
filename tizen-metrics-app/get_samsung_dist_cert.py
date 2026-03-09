#!/usr/bin/env python3
"""
Samsung TV Tizen Distributor Certificate Generator + App Installer

This script:
1. Opens Samsung account OAuth in your browser
2. Captures the access token via local callback server
3. Generates RSA-2048 key + CSR with device DUID as SAN
4. Calls Samsung CA API (svdca.samsungqbe.com) to get signed cert
5. Creates .p12 keystore
6. Registers Tizen security profile with Samsung-issued certs
7. Packages the app WGT and installs to the TV

Usage: python3 get_samsung_dist_cert.py
"""

import http.server
import threading
import urllib.parse
import json
import ssl
import urllib.request
import subprocess
import os
import sys
import time
import webbrowser
import socket
import shutil

from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DUID = "2DCIFS3MMTRFM"
TV_HOST = "10.0.0.10"
TV_PORT = "26101"
TV_DEVICE = f"{TV_HOST}:{TV_PORT}"

AUTHOR_P12 = "/home/rodney/.tizen-extension-platform/server/sdktools/sdk-data/keystore/author/ADAI Dashboard_auth.p12"
AUTHOR_PASSWORD = "Tizen123"

SAMSUNG_CA_DIR = "/home/rodney/.tizen-extension-platform/server/dist/assets/certificate-manager/samsung-tv-ca"
CA_PUBLIC_CERT = f"{SAMSUNG_CA_DIR}/vd_tizen_dev_public2.crt"

OUTPUT_DIR = Path("/home/rodney/Repos/adai/tizen-metrics-app/certs/samsung_dist")
DIST_P12 = str(OUTPUT_DIR / "samsung-distributor.p12")
DIST_PASSWORD = "Tizen123"

STAGE_DIR = Path("/tmp/adai-stage")
WGT_OUT_DIR = Path("/tmp/adai-wgt")
APP_SRC = Path("/home/rodney/Repos/adai/tizen-metrics-app")

TIZEN_CLI = "/home/rodney/tizen-studio/tools/ide/bin/tizen"
SDB = "/home/rodney/.tizen-extension-platform/server/sdktools/data/tools/sdb"
PROFILE_NAME = "AdaiSamsung"

# Samsung OAuth
SERVICE_ID = "v285zxnl3h"
SAMSUNG_DIST_URL_V1 = "https://svdca.samsungqbe.com/apis/v1/distributors"
SAMSUNG_DIST_URL_V3 = "https://svdca.samsungqbe.com/apis/v3/distributors"

# ──────────────────────────────────────────────
# Step 1: Samsung OAuth
# ──────────────────────────────────────────────
class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    code = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/signin/callback":
            params = urllib.parse.parse_qs(parsed.query)
            code_val = params.get("code", [""])[0]
            if code_val:
                OAuthCallbackHandler.code = code_val
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                <html><body style="font-family:Arial;text-align:center;margin-top:80px">
                <h2 style="color:green">&#10003; Samsung Login Successful</h2>
                <p>You can close this window and return to the terminal.</p>
                </body></html>""")
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Samsung Authentication Service</h2>")

    def log_message(self, fmt, *args):
        pass  # suppress access log


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def samsung_oauth_login():
    """Open Samsung account in browser, wait for OAuth callback, return auth dict."""
    port = find_free_port()
    server = http.server.HTTPServer(('localhost', port), OAuthCallbackHandler)

    auth_url = (
        f"https://account.samsung.com/mobile/account/check.do"
        f"?serviceID={SERVICE_ID}"
        f"&actionID=StartOAuth2"
        f"&accessToken=Y"
        f"&redirect_uri=http://localhost:{port}/signin/callback"
    )

    print(f"\n{'='*60}")
    print("SAMSUNG ACCOUNT LOGIN REQUIRED")
    print('='*60)
    print(f"\nOpening Samsung login in your browser...")
    print(f"If the browser doesn't open, visit:\n  {auth_url}\n")
    print("Waiting for login (up to 3 minutes)...\n")

    webbrowser.open(auth_url)

    # Run server with 3-minute timeout
    server.timeout = 180
    deadline = time.time() + 180
    while not OAuthCallbackHandler.code and time.time() < deadline:
        server.handle_request()

    server.server_close()

    if not OAuthCallbackHandler.code:
        raise RuntimeError("OAuth timeout — no callback received in 3 minutes")

    # The 'code' parameter is URL-encoded JSON
    raw = urllib.parse.unquote(OAuthCallbackHandler.code)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Unexpected callback format: {raw[:200]}")

    access_token = data.get("access_token") or data.get("accessToken")
    user_id = data.get("userId") or data.get("user_id") or ""
    email = data.get("inputEmailID") or data.get("email") or ""

    if not access_token:
        raise RuntimeError(f"No access_token in response: {data}")

    print(f"✓ Logged in as: {email or user_id}")
    return {"accessToken": access_token, "userId": user_id, "email": email}


# ──────────────────────────────────────────────
# Step 2: Generate CSR with DUID SAN (via openssl)
# ──────────────────────────────────────────────
def generate_key_and_csr(output_dir: Path, duid: str, password: str):
    """Generate RSA-2048 private key and CSR with DUID as SubjectAltName using OpenSSL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_key_path = output_dir / "distributor.raw.key"
    pri_path = output_dir / "distributor.pri"
    csr_path = output_dir / "distributor.csr"
    san_conf_path = output_dir / "san.cnf"

    # Write OpenSSL config with SAN extension
    san_conf = f"""[req]
distinguished_name = req_dn
req_extensions = v3_req
prompt = no

[req_dn]
CN = TizenSDK

[v3_req]
subjectAltName = @alt_names

[alt_names]
URI.1 = URN:tizen:packageid=
URI.2 = URN:tizen:deviceid={duid}
"""
    san_conf_path.write_text(san_conf)

    print("  Generating RSA-2048 key pair...")
    run(["openssl", "genrsa", "-out", str(raw_key_path), "2048"])

    print("  Encrypting private key...")
    run(["openssl", "rsa", "-in", str(raw_key_path), "-out", str(pri_path),
         "-aes256", "-passout", f"pass:{password}"])
    raw_key_path.unlink(missing_ok=True)
    print(f"  Saved private key: {pri_path}")

    print("  Generating CSR with DUID SAN...")
    run(["openssl", "req", "-new",
         "-key", str(pri_path),
         "-passin", f"pass:{password}",
         "-out", str(csr_path),
         "-config", str(san_conf_path),
         "-sha512"])
    print(f"  Saved CSR: {csr_path}")

    # Verify SAN is in CSR
    result = run(["openssl", "req", "-in", str(csr_path), "-text", "-noout"], capture=True)
    if duid in (result.stdout or ""):
        print(f"  ✓ DUID {duid} confirmed in CSR SAN")
    else:
        print(f"  ⚠ Could not confirm DUID in CSR — check {csr_path}")

    return pri_path, csr_path


# ──────────────────────────────────────────────
# Step 3: Call Samsung CA API
# ──────────────────────────────────────────────
def multipart_form(fields: dict, files: dict, boundary: str) -> bytes:
    """Create multipart/form-data body."""
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Disposition: form-data; name={name}\r\n\r\n"
            f"{value}\r\n"
        )
    for name, (filename, content) in files.items():
        header = (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name={name}; filename={filename}\r\n\r\n"
        )
        parts.append(header + content + "\r\n")
    body = "".join(parts) + f"--{boundary}--\r\n"
    return body.encode("utf-8")


def call_samsung_ca(url: str, access_token: str, user_id: str, csr_path: Path, output_path: Path) -> bool:
    """POST CSR to Samsung CA, save response cert."""
    boundary = f"----{int(time.time() * 1000)}"
    csr_content = csr_path.read_text()

    fields = {
        "access_token": access_token,
        "user_id": user_id,
        "privilege_level": "Public",
        "developer_type": "Individual",
        "platform": "VD",
    }
    files = {
        "csr": ("distributor.csr", csr_content),
    }

    body = multipart_form(fields, files, boundary)
    content_type = f"multipart/form-data; boundary={boundary}"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
        method="POST"
    )

    print(f"  POST {url}")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            status = resp.getcode()
            if status == 200:
                cert_data = resp.read().decode("utf-8")
                output_path.write_text(cert_data)
                print(f"  ✓ Saved response → {output_path}")
                return True
            else:
                print(f"  ✗ HTTP {status}")
                return False
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Samsung CA API error {e.code}: {body_err[:500]}")


# ──────────────────────────────────────────────
# Step 4: Build P12 from cert + key + CA
# ──────────────────────────────────────────────
def build_p12(crt_path: Path, pri_path: Path, ca_cert_path: str, p12_path: str, password: str):
    """Combine signed cert + private key + CA cert into a PKCS#12 using openssl."""
    import tempfile

    # Decrypt private key to temp file for openssl pkcs12
    tmp_key = tempfile.NamedTemporaryFile(suffix=".key", delete=False)
    tmp_key.close()
    try:
        run(["openssl", "rsa",
             "-in", str(pri_path), "-passin", f"pass:{password}",
             "-out", tmp_key.name])

        # Build p12: key + signed cert + CA cert, using legacy PBE for Tizen CLI compat
        run(["openssl", "pkcs12", "-export",
             "-inkey", tmp_key.name,
             "-in", str(crt_path),
             "-certfile", ca_cert_path,
             "-out", p12_path,
             "-passout", f"pass:{password}",
             "-name", "samsung-distributor",
             "-legacy",          # SHA1+3DES — required for Java/Tizen CLI
             ])
        print(f"  ✓ PKCS#12 saved: {p12_path}")
        # Verify it's readable
        run(["openssl", "pkcs12", "-in", p12_path,
             "-passin", f"pass:{password}", "-noout", "-info"], capture=True)
        print("  ✓ P12 verified readable")
    finally:
        os.unlink(tmp_key.name)



# ──────────────────────────────────────────────
# Step 5: Stage app, sign, install
# ──────────────────────────────────────────────
def run(cmd, check=True, capture=False):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=capture, text=True)
    if check and result.returncode != 0:
        out = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {out[-500:]}")
    return result


def stage_app():
    print("\n[5a] Staging clean app...")
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)
    for item in ["config.xml", "index.html", "icon.png"]:
        shutil.copy(APP_SRC / item, STAGE_DIR / item)
    for folder in ["css", "js"]:
        shutil.copytree(APP_SRC / folder, STAGE_DIR / folder)
    print(f"  Staged to {STAGE_DIR}")


def register_profile():
    print(f"\n[5b] Registering security profile '{PROFILE_NAME}'...")
    # Remove if exists
    run([TIZEN_CLI, "security-profiles", "remove", "-n", PROFILE_NAME], check=False)
    run([
        TIZEN_CLI, "security-profiles", "add",
        "-n", PROFILE_NAME,
        "-a", AUTHOR_P12, "-p", AUTHOR_PASSWORD,
        "-d", DIST_P12, "-p", DIST_PASSWORD,
        "--distributor", "1",
    ])
    print(f"  ✓ Profile '{PROFILE_NAME}' registered")


def package_app():
    print("\n[5c] Packaging WGT...")
    if WGT_OUT_DIR.exists():
        shutil.rmtree(WGT_OUT_DIR)
    WGT_OUT_DIR.mkdir(parents=True)
    run([TIZEN_CLI, "package", "-t", "wgt", "-s", PROFILE_NAME, "-o", str(WGT_OUT_DIR), "--", str(STAGE_DIR)])
    wgts = list(WGT_OUT_DIR.glob("*.wgt"))
    if not wgts:
        raise RuntimeError("No WGT produced")
    wgt = wgts[0]
    print(f"  ✓ Package: {wgt} ({wgt.stat().st_size // 1024}KB)")
    return wgt


def install_app(wgt: Path):
    print("\n[5d] Connecting to TV and installing...")
    run([SDB, "connect", TV_HOST])
    time.sleep(2)
    result = run([TIZEN_CLI, "install", "-s", TV_DEVICE, "-n", str(wgt)], check=False, capture=True)
    output = result.stdout + result.stderr
    print(output)
    if result.returncode == 0 and "successfully" in output.lower():
        print("\n✅ APP INSTALLED SUCCESSFULLY!")
        return True
    else:
        print(f"\n✗ Install failed (rc={result.returncode})")
        return False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Samsung TV Tizen App Installer")
    print(f"  DUID:  {DUID}")
    print(f"  TV:    {TV_DEVICE}")
    print("=" * 60)

    # Step 1: OAuth
    print("\n[1] Samsung OAuth login...")
    auth = samsung_oauth_login()

    # Step 2: Generate CSR
    print(f"\n[2] Generating RSA-2048 key + CSR for DUID {DUID}...")
    pri_path, csr_path = generate_key_and_csr(OUTPUT_DIR, DUID, DIST_PASSWORD)

    # Step 3: Get Samsung cert (v1 then v3)
    print("\n[3] Requesting Samsung distributor certificate...")
    dev_profile_path = OUTPUT_DIR / "device-profile.xml"
    dist_crt_path = OUTPUT_DIR / "distributor.crt"

    print("  [v1] Getting device profile...")
    call_samsung_ca(SAMSUNG_DIST_URL_V1, auth["accessToken"], auth["userId"],
                    csr_path, dev_profile_path)

    print("  [v3] Getting distributor cert...")
    call_samsung_ca(SAMSUNG_DIST_URL_V3, auth["accessToken"], auth["userId"],
                    csr_path, dist_crt_path)

    # Step 4: Build P12
    print("\n[4] Building PKCS#12 keystore...")
    build_p12(dist_crt_path, pri_path, CA_PUBLIC_CERT, DIST_P12, DIST_PASSWORD)

    # Step 5: Stage, sign, install
    stage_app()
    register_profile()
    wgt = package_app()
    success = install_app(wgt)

    if success:
        print(f"\nDistributor cert saved to: {DIST_P12}")
        print(f"App installed on TV {TV_DEVICE}")
    else:
        print(f"\nCert generated at: {DIST_P12}")
        print("Installation failed — check TV platform log")
        sys.exit(1)


if __name__ == "__main__":
    main()
