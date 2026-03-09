# Samsung TV Tizen App Installation Guide

**Target Device:** Samsung UN55DU7200DXZA  
**Tizen Version:** 8.0  
**Platform:** Samsung Smart TV (VD)  
**Date:** March 8, 2026  

---

## Prerequisites

| Tool | Path |
|------|------|
| Tizen Studio CLI | `~/tizen-studio/tools/ide/bin/tizen` |
| Samsung sdb | `~/.tizen-extension-platform/server/sdktools/data/tools/sdb` |
| Tizen Extension Server | `http://localhost:45653` (VS Code Samsung Tizen Extension) |
| Samsung cert dir | `~/SamsungCertificate/` |
| Extension profiles.xml | `~/.tizen-extension-platform/server/sdktools/sdk-data/profile/profiles.xml` |

---

## Step 1 — Enable Developer Mode on the TV

1. On the TV remote: **Settings → General → System Manager → Developer Mode**
2. Toggle on, enter the host machine IP (e.g., `10.0.0.12`)
3. Reboot the TV when prompted

---

## Step 2 — Connect via sdb

```bash
SDB="$HOME/.tizen-extension-platform/server/sdktools/data/tools/sdb"
$SDB connect 10.0.0.10
$SDB devices
# Expected: 10.0.0.10:26101  device  UN55DU7200DXZA
```

**Note:** The default sdb port on Samsung TVs is `26101` (not `26101`—confirm with `$SDB devices`).  
The TV uses `secure_protocol:enabled`, meaning sdb communicates over a secured channel.

---

## Step 3 — Get the TV DUID

The DUID (Device Unique ID) is required to bind the Samsung distributor certificate to your TV.

```bash
# Via extension API (TV must be connected via sdb first):
curl -s http://localhost:45653/api/v1/getDuidList
# Returns: {"status":"success","duidList":["2DCIFS3MMTRFM"]}
```

**TV DUID:** `2DCIFS3MMTRFM`

The DUID format `2DCIFS3MMTRFM` (no `#` prefix, no `1.0#` prefix) maps to **version V2** in Samsung's cert versioning scheme.

---

## Step 4 — Certificate Setup

### 4a. Author Certificate

The author cert proves who published the app. Use the Samsung Tizen Extension to create it — it provisions through the Samsung CA (`svdca.samsungqbe.com/apis/v3/authors`).

**Working author cert:**  
- Path: `~/.tizen-extension-platform/server/sdktools/sdk-data/keystore/author/ADAI Dashboard_auth.p12`  
- Password: `Tizen123`  
- Subject: `CN=Rodney Vanmarter`  
- Issuer: `O=Tizen Association, OU=Tizen Association, CN=Tizen Developers CA`  

> **Note:** This author cert is from the Tizen Association CA (not Samsung's VD Author CA). Despite that, installation succeeded. Samsung's distributor cert (Step 4b) is the critical one for TV acceptance.

### 4b. Samsung Distributor Certificate (CRITICAL)

Generic Tizen Foundation/Association distributor certs are **rejected** by Samsung TVs. You must obtain a Samsung CA-issued distributor cert bound to your TV's DUID.

**Generate via Samsung Tizen Extension API:**

```bash
curl -s -X POST http://localhost:45653/api/v1/generateSamsungCert \
  -H "Content-Type: application/json" \
  -d '{
    "profileName": "AdaiDashboard",
    "password": "Tizen123",
    "certificateType": "distributor",
    "duidList": ["2DCIFS3MMTRFM"],
    "isTVSelected": true,
    "privilege": "Public"
  }'
```

**What happens internally:**
1. Extension opens Samsung OAuth login (`https://account.samsung.com/`) in a browser
2. After you log in, the extension exchanges the OAuth code for an access token
3. It submits a CSR to `https://svdca.samsungqbe.com/apis/v1/distributors` with DUID and privilege
4. Samsung CA returns a DUID-bound certificate
5. Extension packages into `distributor.p12` and saves to `~/SamsungCertificate/{profileName}/`

**Resulting files:**
```
~/SamsungCertificate/AdaiDashboard/
├── device-profile.xml    # Contains DUID binding and full cert chain (XML)
├── distributor.crt       # The DER certificate
├── distributor.csr       # Certificate signing request
├── distributor.p12       # PKCS#12 bundle (what Tizen CLI signs with)
└── distributor.pri       # Private key
```

**Verify the cert has your DUID:**
```bash
openssl pkcs12 -in ~/SamsungCertificate/AdaiDashboard/distributor.p12 \
  -passin pass:Tizen123 -nokeys -clcerts 2>/dev/null \
  | openssl x509 -noout -text \
  | grep -A2 "Subject Alternative"
# Expected: URI:URN:tizen:packageid=, URI:URN:tizen:deviceid=2DCIFS3MMTRFM
```

**Samsung CA chain:**
- Leaf: `CN=TizenSDK`, issued by `VD DEVELOPER Public CA Class`
- Intermediate: `VD DEVELOPER Public CA Class`, issued by `VD DEVELOPER Public Root Class`

### 4c. Samsung TV CA Files (bundled with extension)

The extension bundles Samsung's CA roots at:
```
~/.tizen-extension-platform/server/dist/assets/certificate-manager/samsung-tv-ca/
├── vd_tizen_dev_author_ca.cer    # Samsung VD Author CA
├── vd_tizen_dev_public2.crt      # Samsung VD Distributor Public CA
└── vd_tizen_dev_partner2.crt     # Samsung VD Distributor Partner CA
```

---

## Step 5 — Create Security Profile

Add a named profile in the Tizen CLI that pairs your author and distributor certs:

```bash
TIZEN="$HOME/tizen-studio/tools/ide/bin/tizen"
AUTHOR_P12="$HOME/.tizen-extension-platform/server/sdktools/sdk-data/keystore/author/ADAI Dashboard_auth.p12"
DIST_P12="$HOME/SamsungCertificate/AdaiDashboard/distributor.p12"

# Point CLI at extension's profiles.xml
$TIZEN cli-config "profiles.path=$HOME/.tizen-extension-platform/server/sdktools/sdk-data/profile/profiles.xml"

# Create profile
$TIZEN security-profiles add \
  -n "ADAI Samsung" \
  -A \
  -a "$AUTHOR_P12" \
  -p "Tizen123" \
  -d "$DIST_P12" \
  -dp "Tizen123"
```

Verify:
```bash
$TIZEN security-profiles list
```

---

## Step 6 — config.xml Requirements

### Working config.xml (critical fields):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<widget xmlns="http://www.w3.org/ns/widgets"
        xmlns:tizen="http://tizen.org/ns/widgets"
        id="http://rodney.adai/training-metrics"
        version="1.0.0">

    <tizen:application id="AdaiMtrcs1.trainingmetrics"
                       package="AdaiMtrcs1"
                       required_version="3.0"/>
    ...
    <tizen:profile name="tv"/>
    <access origin="*" subdomains="true"/>
    <tizen:privilege name="http://tizen.org/privilege/internet"/>
    ...
</widget>
```

### Rules that were discovered by trial and error:

| Field | Correct Value | Common Mistakes |
|-------|--------------|-----------------|
| `package` | Alphanumeric, exactly 10 chars (`AdaiMtrcs1`) | Too short/long, or hyphens |
| `tizen:application id` | `{package}.{appname}` | Must match package prefix |
| `required_version` | `"3.0"` | `"2.4"` caused silent failures on Tizen 8.0 |
| `tizen:profile name` | `"tv"` | `"tv-samsung"` caused install failure |
| Privileges | Only `http://tizen.org/privilege/internet` | `http://tizen.org/privilege/tv.display` requires partner certs |

---

## Step 7 — Package the WGT

```bash
STAGE_DIR="/tmp/adai-stage"
WGT_DIR="/tmp/adai-wgt"

# Clean old signatures (REQUIRED before re-signing)
rm -f "$STAGE_DIR/author-signature.xml" "$STAGE_DIR/signature1.xml"

# Package
$TIZEN package -t wgt -s "ADAI Samsung" -o "$WGT_DIR" -- "$STAGE_DIR"
```

> **CRITICAL:** The output filename will be based on the `<name>` tag in config.xml (e.g., `ADAI Training Metrics.wgt`). **Rename it to remove spaces before installing.**

---

## Step 8 — Permit-to-Install

This step pushes your author certificate to the TV's trusted cert store, allowing it to accept apps signed by that cert.

```bash
curl -s -X POST "http://localhost:45653/api/v1/devices/10.0.0.10:26101/permit-to-install" \
  -H "Content-Type: application/json" \
  -d '{"certificate_path":"/home/rodney/.tizen-extension-platform/server/sdktools/sdk-data/keystore/author/ADAI Dashboard_auth.p12"}'
# Expected: {"status":"success","message":"Permit to install succeeded"}
```

**What this does internally:** Extracts the p12 filename and pushes it to `/home/owner/share/tmp/sdk_tools/{filename}` on the TV via sdb push.

> Run this step before each install if you've rebooted the TV.

---

## Step 9 — Install the WGT

```bash
# RENAME to remove spaces - MANDATORY
mv "$WGT_DIR/ADAI Training Metrics.wgt" "$WGT_DIR/adai.wgt"

# Install
$TIZEN install -s "10.0.0.10:26101" -n "$WGT_DIR/adai.wgt"
```

**Expected output:**
```
Transferring the package...
Installing the package...
install AdaiMtrcs1.trainingmetrics
app_id[AdaiMtrcs1.trainingmetrics] install start
app_id[AdaiMtrcs1.trainingmetrics] installing[9] ... [100]
app_id[AdaiMtrcs1.trainingmetrics] install completed
Installed the package: Id(AdaiMtrcs1.trainingmetrics)
Tizen application is successfully installed.
```

---

## Step 10 — Launch the App

```bash
$TIZEN run -s "10.0.0.10:26101" -p "AdaiMtrcs1.trainingmetrics"
# Expected: ... successfully launched pid = NNN with debug 0
```

---

## Troubleshooting

### "Failed to install Tizen application" (instant, ~1 second)

Work through these causes in order:

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| **Spaces in WGT filename** | Filename contains spaces | Rename to `adai.wgt` before installing |
| **Generic distributor cert** | Distributor cert issuer is Tizen Foundation/Association | Generate Samsung CA cert via `generateSamsungCert` API |
| **Wrong profile name in config.xml** | `tizen:profile name="tv-samsung"` | Change to `name="tv"` |
| **Wrong required_version** | `required_version="2.4"` | Change to `"3.0"` |
| **tv.display privilege** | App uses `http://tizen.org/privilege/tv.display` | Remove it (needs partner cert) |
| **Package ID format** | Package != 10 alphanumeric chars | Fix to exactly 10 chars, alphanumeric only |
| **Author cert not trusted** | permit-to-install not run / TV rebooted | Re-run permit-to-install step |
| **Cert/key in WGT** | Private key or cert files packaged inside WGT | Use a clean staging dir with only app files |

### "Failed to install" with no platform log output

The TV's dlogutil logging is **disabled by default** (`log_enable:disabled` in sdb capability). You cannot read TV logs without enabling developer logging first. Instead, diagnose by methodically eliminating causes from the table above.

### Extension API unreachable

```bash
curl -s http://localhost:45653/api/v1/health
```

If not running, open VS Code and ensure the Samsung Tizen Extension is active (it starts the local API server).

### sdb "closed" immediately / shell returns nothing

The TV shell (`sdb shell`) sessions are highly restricted on Samsung consumer TVs. Many commands (pkgcmd, dlogutil with filters) exit immediately. Use `sdb install` or `tizen install` instead of trying to run commands directly.

---

## Extension API Reference

The Samsung Tizen VS Code extension runs an HTTP server at `http://localhost:45653`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Server health check |
| `/api/v1/getDuidList` | GET | Get DUIDs of connected TV devices |
| `/api/v1/getProfiles` | GET | List security profiles |
| `/api/v1/generateSamsungCert` | POST | Generate Samsung CA-issued cert (opens browser for OAuth) |
| `/api/v1/devices/:serial/permit-to-install` | POST | Push author cert to TV trust store |
| `/api/v1/devices` | GET | List connected devices with capabilities |

---

## Key Technical Findings

1. **Spaces in WGT filename crash pkgcmd** — The `tizen install` command transfers the WGT and triggers `pkgcmd` on the TV. If the filename contains spaces, `pkgcmd` fails immediately with no useful error message. Always rename the WGT to a no-space filename before installing.

2. **Samsung TVs reject non-Samsung distributor certs** — The TV's pkgmgr validates the distributor signature against Samsung's CA root (`VD DEVELOPER Public Root Class`). Tizen Foundation/Association certs are rejected silently (~1 second failure).

3. **`permit-to-install` only pushes the p12 file** — It does not register the cert in a TV trust database. It simply copies the p12 to `/home/owner/share/tmp/sdk_tools/` so the TV's pkgmgr can verify the author signature during install.

4. **The Samsung cert API requires a browser OAuth login** — `generateSamsungCert` opens `https://account.samsung.com/` for authentication. After login, the extension handles token exchange and Samsung CA API calls automatically. Auth tokens are cached at `~/SamsungCertificate/{profileName}/samsung-auth-data.json`.

5. **`tizen:profile name="tv-samsung"` causes install failure** — Use `"tv"` for all Samsung TV apps. The `tv-samsung` value is a recognized profile selector but causes pkgmgr to apply stricter validation.

6. **`required_version="2.4"` is rejected by Tizen 8.0 TVs** — Use `"3.0"` or higher.

7. **TV logs (dlogutil) are disabled** — Samsung consumer TVs ship with logging disabled. You cannot capture install error logs without root access. Diagnose install failures by elimination.

8. **The extension port is `45653`** — Not configurable; hardcoded in the Samsung Tizen VS Code Extension (v10.3.0+).

---

## Quick Reference: Full Deploy Sequence

```bash
TIZEN="$HOME/tizen-studio/tools/ide/bin/tizen"
SDB="$HOME/.tizen-extension-platform/server/sdktools/data/tools/sdb"
TV="10.0.0.10:26101"
AUTHOR_P12="$HOME/.tizen-extension-platform/server/sdktools/sdk-data/keystore/author/ADAI Dashboard_auth.p12"
STAGE="/tmp/adai-stage"
WGT_DIR="/tmp/adai-wgt"

# 1. Configure CLI
$TIZEN cli-config "profiles.path=$HOME/.tizen-extension-platform/server/sdktools/sdk-data/profile/profiles.xml"

# 2. Clean & package
rm -f "$STAGE/author-signature.xml" "$STAGE/signature1.xml"
$TIZEN package -t wgt -s "ADAI Samsung" -o "$WGT_DIR" -- "$STAGE"

# 3. RENAME (remove spaces)
mv "$WGT_DIR/ADAI Training Metrics.wgt" "$WGT_DIR/adai.wgt"

# 4. Connect
$SDB connect 10.0.0.10

# 5. Permit-to-install
curl -s -X POST "http://localhost:45653/api/v1/devices/$TV/permit-to-install" \
  -H "Content-Type: application/json" \
  -d "{\"certificate_path\":\"$AUTHOR_P12\"}"

# 6. Install
$TIZEN install -s "$TV" -n "$WGT_DIR/adai.wgt"

# 7. Launch
$TIZEN run -s "$TV" -p "AdaiMtrcs1.trainingmetrics"
```

Or use the provided [deploy.sh](deploy.sh) script which automates all of the above.
