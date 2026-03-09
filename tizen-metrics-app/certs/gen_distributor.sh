#!/usr/bin/env bash
# =============================================================
# gen_distributor.sh — Generate Samsung TV Distributor certificate
#
# This script creates a self-signed Distributor certificate for
# development sideloading and (optionally) wraps the DUID into
# the certificate CN so Samsung's signing workflow can identify
# the target device.
#
# For production distribution, the distributor cert MUST come
# from Samsung's Developer Portal. This script creates a cert
# suitable for development/sideloading.
#
# Usage:
#   ./gen_distributor.sh [DUID]
#
#   If DUID is provided: cert CN includes DUID and is targeted
#   If DUID is omitted:  uses the generic Tizen dev distributor
# =============================================================

set -euo pipefail

DUID="${1:-}"
CERTS_DIR="$(cd "$(dirname "$0")" && pwd)"
TIZEN_DIST_CA=/home/rodney/tizen-studio/tools/certificate-generator/certificates/distributor

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; BLU='\033[0;34m'; NC='\033[0m'
bold() { echo -e "${BLU}$*${NC}"; }
ok()   { echo -e "${GRN}[OK]${NC}  $*"; }
warn() { echo -e "${YEL}[WARN]${NC} $*"; }

bold "================================================"
bold " ADAI Tizen TV — Distributor Certificate Generator"
bold "================================================"
echo ""

if [[ -n "$DUID" ]]; then
    ok "Targeting DUID: ${DUID}"
    CERT_CN="ADAI Distributor (${DUID})"
    OUT_BASE="${CERTS_DIR}/adai-distributor-${DUID:0:8}"
else
    warn "No DUID provided. Generating generic development distributor cert."
    warn "This cert lets you sideload on ANY device in developer mode."
    CERT_CN="ADAI Distributor (dev)"
    OUT_BASE="${CERTS_DIR}/adai-distributor-dev"
fi

KEY_FILE="${OUT_BASE}.key"
CSR_FILE="${OUT_BASE}.csr"
CRT_FILE="${OUT_BASE}.crt"
P12_FILE="${OUT_BASE}.p12"
PWD_FILE="${OUT_BASE}.pwd"

# ── Step 1: Generate distributor private key ───────────────────
bold "Generating RSA-2048 distributor key..."
openssl genrsa -out "$KEY_FILE" 2048
ok "Key: ${KEY_FILE}"

# ── Step 2: Create CSR config ─────────────────────────────────
CNF_FILE=$(mktemp /tmp/adai_dist_XXXXXX.cnf)
cat > "$CNF_FILE" << CONF
[req]
default_bits       = 2048
default_md         = sha256
distinguished_name = dn
x509_extensions    = v3_dist
prompt             = no

[dn]
C  = US
ST = California
L  = San Francisco
O  = ADAI Project
OU = TV Distribution
CN = ${CERT_CN}

[v3_dist]
subjectKeyIdentifier   = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints       = critical,CA:false
keyUsage               = critical,digitalSignature
extendedKeyUsage       = emailProtection
CONF

# ── Step 3: Self-signed distributor certificate ────────────────
bold "Generating distributor certificate (10 years)..."
openssl req -new -x509 \
  -key    "$KEY_FILE" \
  -out    "$CRT_FILE" \
  -days   3650 \
  -config "$CNF_FILE"
ok "Certificate: ${CRT_FILE}"
rm -f "$CNF_FILE"

# ── Step 4: Export to PKCS#12 ─────────────────────────────────
bold "Bundling into PKCS#12..."

# Check if we can chain against Tizen CA (for proper trust chain)
if [[ -f "${TIZEN_DIST_CA}/tizen-distributor-ca.cer" ]]; then
    openssl pkcs12 -export \
        -inkey  "$KEY_FILE" \
        -in     "$CRT_FILE" \
        -certfile "${TIZEN_DIST_CA}/tizen-distributor-ca.cer" \
        -name   "ADAI Distributor" \
        -out    "$P12_FILE" \
        -passout pass:
    ok "Bundled with Tizen CA chain"
else
    openssl pkcs12 -export \
        -inkey  "$KEY_FILE" \
        -in     "$CRT_FILE" \
        -name   "ADAI Distributor" \
        -out    "$P12_FILE" \
        -passout pass:
    warn "Tizen CA not found; bundled without CA chain."
fi
ok "PKCS#12: ${P12_FILE}"

# Empty password file (Tizen Studio format)
echo -n "" > "$PWD_FILE"

# ── Step 5: Show cert info ────────────────────────────────────
bold "Certificate details:"
openssl x509 -in "$CRT_FILE" -noout -subject -issuer -dates 2>/dev/null || true

# ── Step 6: Update / create profiles.xml ─────────────────────
PROFILES_XML="${CERTS_DIR}/profiles.xml"
AUTHOR_P12="${CERTS_DIR}/adai-author.p12"
AUTHOR_PWD="${CERTS_DIR}/adai-author.pwd"
TIZEN_CA_CER="${TIZEN_DIST_CA}/tizen-distributor-ca.cer"

bold "Writing profiles.xml..."
cat > "$PROFILES_XML" << PROFILES
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!--
    ADAI TV Metrics App — Tizen Signing Profile
    Generated: $(date '+%Y-%m-%d %H:%M:%S')
    TV Target DUID: ${DUID:-<not set — run get_tv_duid.sh>}

    To use in Tizen Studio:
      Tools → Certificate Manager → Import → select this profiles.xml
    Or set TIZEN_SDK_DATA_PATH to point here and restart Tizen Studio.
-->
<profiles active="AdaiTV" version="3.1">
    <profile name="AdaiTV">
        <!-- Author certificate (signs app identity) -->
        <profileitem
            distributor="0"
            key="${AUTHOR_P12}"
            password="${AUTHOR_PWD}"
            ca=""
            rootca=""/>

        <!-- Distributor certificate (device trust) -->
        <profileitem
            distributor="1"
            key="${P12_FILE}"
            password="${PWD_FILE}"
            ca="${TIZEN_CA_CER}"
            rootca=""/>

        <!-- Distributor 2 (reserved) -->
        <profileitem distributor="2" key="" password="" ca="" rootca=""/>
    </profile>
</profiles>
PROFILES

ok "Profiles: ${PROFILES_XML}"
echo ""
bold "================================================"
ok "Distributor certificate generated successfully!"
echo ""
echo "Files created:"
echo "  ${KEY_FILE}"
echo "  ${CRT_FILE}"
echo "  ${P12_FILE}"
echo "  ${PWD_FILE}"
echo "  ${PROFILES_XML}"
echo ""

if [[ -n "$DUID" ]]; then
    bold "Next: For production Samsung signing, upload to Samsung Developer Portal:"
    echo "  https://developer.samsung.com/smarttv/develop/getting-started/"
    echo "  setting-up-sdk/creating-certificates.html"
    echo ""
    bold "For development sideloading (TV in Developer Mode), this cert is ready."
    echo "  Run:  tizen package -t wgt -s AdaiTV -- /path/to/tizen-metrics-app"
    echo "  Then: sdb install adai.trainingmetrics.wgt"
else
    bold "To generate a DUID-specific certificate:"
    echo "  1. Run: ./get_tv_duid.sh 10.0.0.10"
    echo "  2. Run: ./gen_distributor.sh <DUID>"
fi
bold "================================================"
