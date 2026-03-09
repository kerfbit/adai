#!/usr/bin/env bash
set -e
TIZEN=/home/rodney/tizen-studio/tools/ide/bin/tizen
CERTS=/home/rodney/Repos/adai/tizen-metrics-app/certs
CERT_BASE=/home/rodney/tizen-studio/tools/certificate-generator/certificates

AUTHOR_P12="${CERTS}/adai-author.p12"
AUTHOR_PASS=tizen123

DIST_P12="${CERT_BASE}/distributor/sdk-public/tizen-distributor-signer.p12"
DIST_CA="${CERT_BASE}/distributor/sdk-public/tizen-distributor-ca.cer"
DIST_PASS=tizenpkcs12passfordsigner

echo "==> Removing old AdaiTV profile..."
$TIZEN security-profiles remove -n AdaiTV 2>/dev/null && echo "  removed" || echo "  (not found)"

echo "==> Adding AdaiTV profile with official Tizen distributor cert..."
$TIZEN security-profiles add -n AdaiTV \
  -a "${AUTHOR_P12}" -p "${AUTHOR_PASS}" \
  -d "${DIST_P12}" -dc "${DIST_CA}" -dp "${DIST_PASS}"
echo "profile_status=$?"

echo "==> Listing profiles..."
$TIZEN security-profiles list

echo "==> Done"
