#!/usr/bin/env bash
set -e
TIZEN=/home/rodney/tizen-studio/tools/ide/bin/tizen
CERTS=/home/rodney/Repos/adai/tizen-metrics-app/certs
DIST_CA=/home/rodney/tizen-studio/tools/certificate-generator/certificates/distributor/tizen-distributor-ca.cer
PASS=tizen123

echo "==> Removing old AdaiTV profile (if any)..."
$TIZEN security-profiles remove -n AdaiTV 2>/dev/null && echo "  removed" || echo "  (not found, ok)"

echo "==> Adding AdaiTV profile..."
$TIZEN security-profiles add -n AdaiTV \
  -a "${CERTS}/adai-author.p12" -p "${PASS}" \
  -d "${CERTS}/adai-distributor-90629FEC.p12" -dc "${DIST_CA}" -dp "${PASS}"
echo "profile_add_status=$?"

echo "==> Listing profiles..."
$TIZEN security-profiles list
