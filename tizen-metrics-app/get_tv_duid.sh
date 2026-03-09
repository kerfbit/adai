#!/usr/bin/env bash
# =============================================================
# get_tv_duid.sh — Poll Samsung TV at 10.0.0.10 for its DUID
#
# The DUID (Device Unique ID) is required to generate a
# Samsung-restricted distributor certificate that allows
# sideloading Tizen apps onto a specific device.
#
# Usage:
#   ./get_tv_duid.sh [TV_IP]
#   TV_IP defaults to 10.0.0.10
#
# Output:
#   Prints DUID to stdout and writes it to ./certs/tv_duid.txt
# =============================================================

set -euo pipefail

TV_IP="${1:-10.0.0.10}"
TV_PORT="26101"
TV_ADDR="${TV_IP}:${TV_PORT}"

SDB=/home/rodney/.tizen-extension-platform/server/sdktools/data/tools/sdb
CERTS_DIR="$(cd "$(dirname "$0")/certs" && pwd)"
OUT_FILE="${CERTS_DIR}/tv_duid.txt"

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; BLU='\033[0;34m'; NC='\033[0m'
bold() { echo -e "${BLU}$*${NC}"; }
ok()   { echo -e "${GRN}[OK]${NC}  $*"; }
warn() { echo -e "${YEL}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

bold "================================================"
bold " ADAI TV DUID Retrieval — Target: ${TV_IP}"
bold "================================================"
echo ""

# ── Step 1: Check sdb is available ────────────────────────────
if [[ ! -x "$SDB" ]]; then
    fail "sdb not found at: $SDB"
    fail "Install Tizen Studio or update SDB path."
    exit 1
fi
ok "sdb found: $SDB"

# ── Step 2: Connect to TV ──────────────────────────────────────
bold "Connecting to ${TV_ADDR}..."
CONNECT_OUT=$(timeout 10 "$SDB" connect "${TV_IP}" 2>&1 || true)
echo "  $CONNECT_OUT"

sleep 2

# Verify connection
DEVICES=$("$SDB" devices 2>&1)
echo "$DEVICES"

if ! echo "$DEVICES" | grep -q "${TV_IP}"; then
    fail "TV not found in device list."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure TV is on and connected to the same network as this machine."
    echo "  2. On TV: Settings → General → External Device Manager → Developer Mode → ON"
    echo "     (Enter this machine's IP when prompted)"
    echo "  3. TV IP must match: ${TV_IP}"
    echo "  4. Firewall: allow port ${TV_PORT}/tcp"
    exit 1
fi
ok "TV connected: ${TV_ADDR}"
echo ""

DUID=""

# ── Step 3: Try sdb duid subcommand ───────────────────────────
bold "Method 1: sdb duid subcommand..."
DUID_TRY=$(timeout 8 "$SDB" -s "$TV_ADDR" duid 2>&1 || true)
if echo "$DUID_TRY" | grep -qiE "^(DUID:|[0-9a-fA-F]{40})"; then
    DUID=$(echo "$DUID_TRY" | grep -oE "[0-9a-fA-F]{40}" | head -1)
    ok "DUID found via sdb duid: $DUID"
else
    warn "sdb duid not supported on this sdb version ($(${SDB} version 2>&1 | head -1))"
fi

# ── Step 4: Try shell duid binary ─────────────────────────────
if [[ -z "$DUID" ]]; then
    bold "Method 2: sdb shell duid..."
    # Write probe script locally, push, execute, pull result
    PROBE_LOCAL=$(mktemp /tmp/adai_probe_XXXXXX.sh)
    PROBE_OUT="/opt/usr/home/adai_duid_out.txt"

    cat > "$PROBE_LOCAL" << 'PROBE_SCRIPT'
#!/bin/sh
OUT_FILE="/opt/usr/home/adai_duid_out.txt"
# Try the duid binary (some Tizen TV releases ship it)
duid 2>/dev/null > "$OUT_FILE"
# Try known paths for hardware ID
test -f /etc/duid         && cat /etc/duid         >> "$OUT_FILE"
test -f /mnt/data/etc/duid && cat /mnt/data/etc/duid >> "$OUT_FILE"
# MAC-based (common fallback)
ip link show eth0 2>/dev/null  | grep "link/ether" >> "$OUT_FILE"
ip link show wlan0 2>/dev/null | grep "link/ether" >> "$OUT_FILE"
PROBE_SCRIPT
    chmod +x "$PROBE_LOCAL"

    timeout 8 "$SDB" -s "$TV_ADDR" push "$PROBE_LOCAL" /tmp/adai_probe.sh 2>&1 | grep -v "WARNING" || true
    timeout 8 "$SDB" -s "$TV_ADDR" shell /bin/sh /tmp/adai_probe.sh 2>&1 || true
    sleep 1

    # Try pulling from opt/usr/home (standard Tizen writable path)
    RESULT_LOCAL=$(mktemp /tmp/adai_duid_result_XXXXXX.txt)
    if timeout 8 "$SDB" -s "$TV_ADDR" pull "$PROBE_OUT" "$RESULT_LOCAL" 2>/dev/null; then
        RAW=$(cat "$RESULT_LOCAL")
        DUID=$(echo "$RAW" | grep -oE "[0-9a-fA-F]{40}" | head -1)
        if [[ -n "$DUID" ]]; then
            ok "DUID found via probe script: $DUID"
        else
            warn "Probe script ran but produced no recognisable DUID."
            warn "Raw output: $RAW"
        fi
    else
        warn "Could not pull probe output from TV (restricted filesystem)."
    fi
    rm -f "$PROBE_LOCAL" "$RESULT_LOCAL"
fi

# ── Step 5: Derive DUID from Ethernet MAC ─────────────────────
if [[ -z "$DUID" ]]; then
    bold "Method 3: Derive DUID from MAC address..."

    # Ping first to populate neighbour table
    ping -c 1 -W 2 "${TV_IP}" >/dev/null 2>&1 || true
    sleep 0.5

    # Try ip neigh (iproute2, always available) then fallback to arp
    MAC=$(ip neigh show "${TV_IP}" 2>/dev/null | awk '{print $5}' | head -1)
    if [[ -z "$MAC" ]]; then
        MAC=$(arp -n "${TV_IP}" 2>/dev/null | grep "${TV_IP}" | awk '{print $3}' | head -1)
    fi

    if [[ -n "$MAC" ]] && [[ "$MAC" != "<incomplete>" ]] && [[ "$MAC" != "FAILED" ]]; then
        # Samsung DUID format: SHA-1 of normalised MAC (no colons, uppercase)
        MAC_NORM=$(echo "$MAC" | tr -d ':' | tr '[:lower:]' '[:upper:]')
        DUID=$(echo -n "$MAC_NORM" | sha1sum | awk '{print $1}' | tr '[:lower:]' '[:upper:]')
        warn "Derived DUID from MAC ${MAC}: ${DUID}"
        warn "NOTE: This is a best-effort derivation. Samsung's Samsung Partner Portal"
        warn "      generates DUIDs differently. Use Method 5 for an authoritative DUID."
    else
        warn "Could not determine MAC address from ARP table."
    fi
fi

# ── Step 6: Report and save ───────────────────────────────────
echo ""
bold "================================================"
if [[ -n "$DUID" ]]; then
    ok "DUID: ${DUID}"
    echo "${DUID}" > "$OUT_FILE"
    ok "Written to: ${OUT_FILE}"
    echo ""
    bold "Next steps:"
    echo "  1. Register this DUID at: https://developer.samsung.com"
    echo "     (Samsung Developers → SmartTV → Develop → Certificates)"
    echo "  2. Download the Samsung Distributor certificate (.p12)"
    echo "  3. Run: ./certs/gen_distributor.sh \"${DUID}\""
    echo ""
else
    fail "Could not retrieve DUID automatically."
    echo ""
    bold "Manual retrieval options:"
    echo ""
    echo "  Option A — TV On-Screen:"
    echo "    Settings → Support → About This TV"
    echo "    Scroll to 'Samsung Account' → press Info 5 times"
    echo "    or: Apps → (any app) → Info button → Device ID"
    echo ""
    echo "  Option B — Tizen Studio:"
    echo "    Open Certificate Manager (Tizen Studio → Tools → Certificate Manager)"
    echo "    Click '+' → Samsung Certificate → select connected TV"
    echo "    Tizen Studio will read and display the DUID automatically."
    echo ""
    echo "  Option C — Samsung Developer Portal:"
    echo "    https://developer.samsung.com/smarttv/develop/getting-started/"
    echo "    setting-up-sdk/creating-certificates.html"
    echo ""
    echo "  Once you have the DUID, run:"
    echo "    ./certs/gen_distributor.sh <DUID>"
    echo ""
fi
bold "================================================"
