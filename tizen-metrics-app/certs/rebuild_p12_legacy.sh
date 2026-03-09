#!/usr/bin/env bash
# Rebuild p12 files using legacy PBE algorithms compatible with Tizen CLI 2.x (Java)
set -e
CERTS=/home/rodney/Repos/adai/tizen-metrics-app/certs
PASS=tizen123
DIST_CA=/home/rodney/tizen-studio/tools/certificate-generator/certificates/distributor/tizen-distributor-ca.cer

echo "==> Rebuilding adai-author.p12 (legacy algorithms)..."
openssl pkcs12 -export \
  -inkey "${CERTS}/adai-author.key" \
  -in    "${CERTS}/adai-author.crt" \
  -name  "ADAI TV Author" \
  -out   "${CERTS}/adai-author.p12" \
  -passout "pass:${PASS}" \
  -keypbe PBE-SHA1-3DES \
  -certpbe PBE-SHA1-3DES \
  -macalg SHA1
echo "${PASS}" > "${CERTS}/adai-author.pwd"
echo "  author_ok"

echo "==> Rebuilding adai-distributor-90629FEC.p12 (legacy algorithms)..."
openssl pkcs12 -export \
  -inkey "${CERTS}/adai-distributor-90629FEC.key" \
  -in    "${CERTS}/adai-distributor-90629FEC.crt" \
  -certfile "${DIST_CA}" \
  -name  "ADAI Distributor" \
  -out   "${CERTS}/adai-distributor-90629FEC.p12" \
  -passout "pass:${PASS}" \
  -keypbe PBE-SHA1-3DES \
  -certpbe PBE-SHA1-3DES \
  -macalg SHA1
echo "${PASS}" > "${CERTS}/adai-distributor-90629FEC.pwd"
echo "  dist_ok"

echo "==> Verifying author..."
openssl pkcs12 -in "${CERTS}/adai-author.p12" -nokeys -passin "pass:${PASS}" -legacy 2>/dev/null | openssl x509 -noout -subject
echo "==> Verifying distributor..."
openssl pkcs12 -in "${CERTS}/adai-distributor-90629FEC.p12" -nokeys -passin "pass:${PASS}" -legacy 2>/dev/null | openssl x509 -noout -subject
echo "==> ALL DONE"
