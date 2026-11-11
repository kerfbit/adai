#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-10

# Installs or removes a newer GuC firmware for the Arc B60 (Battlemage) on the GPU host.
#
#   ./guc_firmware.sh status | install | uninstall
#
# ai-machine's xe driver asks for GuC 70.49.4 but loads 70.44.1, the newest Ubuntu 24.04's
# linux-firmware package has. Upstream linux-firmware ships 70.72.1. `install` downloads that file
# from the upstream commit below, checks its SHA-256 and the version in its header, and puts it in
# /lib/firmware/updates/xe/, which the kernel searches before Ubuntu's own firmware and which
# package upgrades leave alone; then it rebuilds the initramfs. Reboot afterwards. `uninstall`
# removes only the file this script installed and rebuilds the initramfs, which restores Ubuntu's
# firmware on the next boot. `status` shows the firmware the driver loaded and what is installed.
# Uses sudo.

set -uo pipefail

COMMIT=4291fa65d3055297136cad6370e54113e27662ff # "xe: Update GUC to v70.72.1 for BMG, LNL, PTL, NVL-S"
URL="https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/xe/bmg_guc_70.bin?id=$COMMIT"
SHA256=de81c75f46a127c33cd59f604d800e9ffc7ed3495967ba0d8767cd6985ab398b
VERSION=70.72.1
TARGET=/lib/firmware/updates/xe/bmg_guc_70.bin
MARKER="$TARGET.adai"

usage() {
    sed -n '/^#   \.\/guc_firmware/,/^# Uses sudo/p' "$0" | sed 's/^# \{0,1\}//'
}

# guc_version FILE: the firmware version in the GuC header (sw_version, byte 64: major.minor.patch).
guc_version() {
    local v
    v=$(od -An -tu4 -j64 -N4 "$1" 2>/dev/null | tr -d ' ') || return 1
    [[ -n "$v" ]] || return 1
    echo "$(((v >> 16) & 255)).$(((v >> 8) & 255)).$((v & 255))"
}

status() {
    echo "Kernel: $(uname -r)"
    echo "GuC the driver loaded (wanted / found):"
    sudo sh -c 'for f in /sys/kernel/debug/dri/*/gt0/uc/guc_info; do [ -r "$f" ] && grep -E "wanted|found release" "$f" && break; done' |
        sed 's/^[[:space:]]*/  /'
    if [[ -f "$TARGET" ]]; then
        echo "Installed override: $TARGET, version $(guc_version "$TARGET")$([[ -f "$MARKER" ]] && echo ", installed by this script")"
    else
        echo "Installed override: none (Ubuntu's /lib/firmware/xe/bmg_guc_70.bin.zst is used)"
    fi
    local initrd="/boot/initrd.img-$(uname -r)"
    if command -v lsinitramfs >/dev/null && sudo test -r "$initrd"; then
        echo "In the running kernel's initramfs:"
        sudo lsinitramfs "$initrd" 2>/dev/null | grep -E 'bmg_guc' | sed 's/^/  /' || echo "  (no B60 GuC file; the driver loads it from the root filesystem)"
    fi
}

case "${1:-}" in
    status)
        sudo -v || exit 1
        status
        ;;
    install)
        sudo -v || exit 1
        TMP=$(mktemp -d)
        trap 'rm -rf -- "$TMP"' EXIT
        echo "Downloading GuC $VERSION from upstream linux-firmware ($COMMIT)..."
        if ! curl -fsSL --max-time 120 -o "$TMP/bmg_guc_70.bin" "$URL"; then
            echo "FATAL: download failed: $URL"
            exit 1
        fi
        got=$(sha256sum "$TMP/bmg_guc_70.bin" | cut -d' ' -f1)
        if [[ "$got" != "$SHA256" ]]; then
            echo "FATAL: SHA-256 mismatch (got $got, want $SHA256); nothing installed."
            exit 1
        fi
        if [[ "$(guc_version "$TMP/bmg_guc_70.bin")" != "$VERSION" ]]; then
            echo "FATAL: the file's header says version $(guc_version "$TMP/bmg_guc_70.bin"), not $VERSION; nothing installed."
            exit 1
        fi
        if [[ -f "$TARGET" && ! -f "$MARKER" ]]; then
            echo "FATAL: $TARGET exists and wasn't installed by this script; left alone."
            exit 1
        fi
        sudo install -D -m 0644 "$TMP/bmg_guc_70.bin" "$TARGET" &&
            printf 'GuC %s for Battlemage from linux-firmware %s\nsha256 %s\ninstalled %s by guc_firmware.sh\n' \
                "$VERSION" "$COMMIT" "$SHA256" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | sudo tee "$MARKER" >/dev/null || {
            echo "FATAL: couldn't write $TARGET"
            exit 1
        }
        echo "Installed $TARGET (GuC $VERSION). Rebuilding the initramfs for every installed kernel..."
        sudo update-initramfs -u -k all || {
            echo "FATAL: update-initramfs failed; the firmware file is installed, run it again by hand."
            exit 1
        }
        status
        echo
        echo "Reboot, then check with: ./guc_firmware.sh status  (found release version $VERSION)"
        ;;
    uninstall)
        sudo -v || exit 1
        if [[ ! -f "$MARKER" ]]; then
            echo "Nothing to remove: no GuC firmware installed by this script."
            exit 0
        fi
        if [[ "$(sha256sum "$TARGET" 2>/dev/null | cut -d' ' -f1)" != "$SHA256" ]]; then
            echo "FATAL: $TARGET was changed since this script installed it; left alone."
            exit 1
        fi
        sudo rm -f -- "$TARGET" "$MARKER"
        sudo rmdir --ignore-fail-on-non-empty /lib/firmware/updates/xe 2>/dev/null
        echo "Removed $TARGET. Rebuilding the initramfs for every installed kernel..."
        sudo update-initramfs -u -k all || exit 1
        echo "Reboot to go back to Ubuntu's GuC firmware."
        ;;
    -h | --help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
