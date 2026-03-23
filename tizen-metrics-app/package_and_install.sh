#!/usr/bin/env bash
set -e
TIZEN=/home/rodney/tizen-studio/tools/ide/bin/tizen
APP_DIR=/home/rodney/Repos/adai/tizen-metrics-app
OUT_DIR=/tmp/adai-wgt
SDB=/home/rodney/.tizen-extension-platform/server/sdktools/data/tools/sdb

echo "==> Setting CLI profiles path..."
$TIZEN cli-config "profiles.path=/home/rodney/tizen-studio-data/profile/profiles.xml"

echo "==> Cleaning output dir..."
rm -rf "${OUT_DIR}" && mkdir -p "${OUT_DIR}"

echo "==> Packaging app as .wgt with AdaiTV profile..."
$TIZEN package -t wgt -s AdaiTV -o "${OUT_DIR}" -- "${APP_DIR}"
echo "package_status=$?"
mv "${OUT_DIR}"/*.wgt "${OUT_DIR}/adai-metrics.wgt"

echo "==> Listing output..."
ls -lh "${OUT_DIR}"

echo "==> Connecting to Samsung TV (10.0.0.10)..."
$SDB connect 10.0.0.10 2>&1 || true
sleep 2
$SDB devices

echo "==> Installing .wgt on TV..."
WGT=$(ls "${OUT_DIR}"/*.wgt 2>/dev/null | head -1)
if [ -z "$WGT" ]; then
  echo "ERROR: No .wgt file found in ${OUT_DIR}"
  ls "${OUT_DIR}"
  exit 1
fi
echo "Installing: $WGT"
$TIZEN install -s 10.0.0.10:26101 -n "$WGT" 2>&1
echo "install_done"
