#!/bin/bash
# ADAI Training Metrics - Samsung TV Deploy Script
# Tested on: Samsung UN55DU7200DXZA, Tizen 8.0, DUID=2DCIFS3MMTRFM
# Requirements:
#   - Tizen Studio at ~/tizen-studio
#   - Samsung Tizen Extension at ~/.tizen-extension-platform
#   - TV connected via sdb (developer mode enabled)

set -e

TV_SERIAL="192.168.1.17:26101"
SDB="$HOME/tizen-studio/tools/sdb"
TIZEN="$HOME/tizen-studio/tools/ide/bin/tizen"
PROFILE="AdaiTV"
STAGE_DIR="/tmp/adai-stage"
WGT_DIR="/tmp/adai-wgt"
# IMPORTANT: no spaces in the WGT filename - spaces cause pkgcmd failure on TV
WGT_FILE="adai.wgt"
APP_ID="AdaiMtrcs1.trainingmetrics"
AUTHOR_P12="$SCRIPT_DIR/certs/adai-author.p12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ADAI TV Deploy ==="

# 1. Point Tizen CLI at extension profiles
$TIZEN cli-config "profiles.path=$HOME/tizen-studio-data/profile/profiles.xml"

# 2. Sync app files to staging dir
mkdir -p "$STAGE_DIR"
cp "$SCRIPT_DIR/config.xml" "$STAGE_DIR/"
cp "$SCRIPT_DIR/index.html" "$STAGE_DIR/"
cp "$SCRIPT_DIR/icon.png" "$STAGE_DIR/"
cp -r "$SCRIPT_DIR/css" "$STAGE_DIR/"
cp -r "$SCRIPT_DIR/js" "$STAGE_DIR/"

# 3. Clean old signatures
rm -f "$STAGE_DIR/author-signature.xml" "$STAGE_DIR/signature1.xml"

# 4. Package with Samsung cert profile
mkdir -p "$WGT_DIR"
echo "Packaging..."
$TIZEN package -t wgt -s "$PROFILE" -o "$WGT_DIR" -- "$STAGE_DIR"

# Rename to remove spaces (REQUIRED - TV pkgcmd fails with spaces in path)
mv "$WGT_DIR/ADAI Training Metrics.wgt" "$WGT_DIR/$WGT_FILE" 2>/dev/null || true

echo "WGT: $WGT_DIR/$WGT_FILE"

# 5. Connect and permit-to-install
echo "Connecting to TV..."
"$SDB" connect 192.168.1.17

echo "Running permit-to-install..."
curl -s -X POST "http://localhost:45653/api/v1/devices/$TV_SERIAL/permit-to-install" \
  -H "Content-Type: application/json" \
  -d "{\"certificate_path\":\"$AUTHOR_P12\"}" | python3 -m json.tool 2>/dev/null || true

# 6. Install
echo "Installing..."
$TIZEN install -s "$TV_SERIAL" -n "$WGT_DIR/$WGT_FILE"

# 7. Launch
if [[ "${1}" != "--no-launch" ]]; then
  echo "Launching..."
  $TIZEN run -s "$TV_SERIAL" -p "$APP_ID"
fi

echo "=== Done ==="
