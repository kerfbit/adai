#!/usr/bin/env bash
set -e
CERTS=/home/rodney/Repos/adai/tizen-metrics-app/certs
PASS=tizen123
DIST_CA=/home/rodney/tizen-studio/tools/certificate-generator/certificates/distributor/tizen-distributor-ca.cer

echo "==> Rebuilding adai-author.p12 ..."
openssl pkcs12 -export \
  -inkey "${CERTS}/adai-author.key" \
  -in    "${CERTS}/adai-author.crt" \
  -name  "ADAI TV Author" \
  -out   "${CERTS}/adai-author.p12" \
  -passout "pass:${PASS}"
echo "${PASS}" > "${CERTS}/adai-author.pwd"
echo "  author_ok"

echo "==> Rebuilding adai-distributor-90629FEC.p12 ..."
openssl pkcs12 -export \
  -inkey "${CERTS}/adai-distributor-90629FEC.key" \
  -in    "${CERTS}/adai-distributor-90629FEC.crt" \
  -certfile "${DIST_CA}" \
  -name  "ADAI Distributor" \
  -out   "${CERTS}/adai-distributor-90629FEC.p12" \
  -passout "pass:${PASS}"
echo "${PASS}" > "${CERTS}/adai-distributor-90629FEC.pwd"
echo "  dist_ok"

echo "==> Verifying author..."
openssl pkcs12 -in "${CERTS}/adai-author.p12" -nokeys -passin "pass:${PASS}" 2>&1 | openssl x509 -noout -subject
echo "==> Verifying distributor..."
openssl pkcs12 -in "${CERTS}/adai-distributor-90629FEC.p12" -nokeys -passin "pass:${PASS}" 2>&1 | openssl x509 -noout -subject
echo "==> ALL DONE"
