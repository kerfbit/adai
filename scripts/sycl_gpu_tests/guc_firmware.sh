#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.2.0
# @adai-reviewed: 2026-11-15

# Installs or removes a newer GuC firmware for the Arc B60 (Battlemage) on the GPU host.
#
#   ./guc_firmware.sh status | latest | install | install-latest | uninstall
#
# ai-machine's xe driver asks for GuC 70.49.4 but loads 70.44.1, the newest Ubuntu 24.04's
# linux-firmware package has. Upstream linux-firmware ships 70.72.1. `install` downloads that file
# from the upstream commit below, checks its SHA-256 and the version in its header, and puts it in
# /lib/firmware/updates/xe/, which the kernel searches before Ubuntu's own firmware and which
# package upgrades leave alone; then it rebuilds the initramfs. Reboot afterwards. `uninstall`
# removes only the file this script installed and rebuilds the initramfs, which restores Ubuntu's
# firmware on the next boot. `status` shows the firmware the driver loaded and what is installed.
# `latest` looks up the newest Battlemage GuC in upstream linux-firmware (the newest commit that
# changed xe/bmg_guc_70.bin, and the version in its title); `install-latest` installs that one the
# same way, checking the version in the file's header against the commit's title instead of a
# pinned SHA-256, and records the file's SHA-256 so `uninstall` can recognise it. Uses sudo.

set -uo pipefail

COMMIT=4291fa65d3055297136cad6370e54113e27662ff # "xe: Update GUC to v70.72.1 for BMG, LNL, PTL, NVL-S"
URL="https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/xe/bmg_guc_70.bin?id=$COMMIT"
SHA256=de81c75f46a127c33cd59f604d800e9ffc7ed3495967ba0d8767cd6985ab398b
VERSION=70.72.1
TARGET=/lib/firmware/updates/xe/bmg_guc_70.bin
MARKER="$TARGET.adai"

LOG_URL="https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/log/xe/bmg_guc_70.bin"

usage() {
    sed -n '/^#   \.\/guc_firmware/,/Uses sudo\.$/p' "$0" | sed 's/^# \{0,1\}//'
}

# latest_upstream: "<commit> <version>" of the newest upstream commit that changed the B60 GuC,
# from the first row of cgit's log page whose title names a GuC version.
latest_upstream() {
    curl -fsSL --max-time 60 "$LOG_URL" |
        grep -o "id=[0-9a-f]\{40\}'>xe: Update GUC to v[0-9.]*" | head -1 |
        sed -E "s/^id=([0-9a-f]{40})'>xe: Update GUC to v([0-9.]*[0-9]).*/\1 \2/"
}

# version_newer A B: whether version A is newer than B (dotted numbers).
version_newer() {
    [[ "$1" != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]
}

# install_file COMMIT VERSION [SHA256]: downloads xe/bmg_guc_70.bin at COMMIT, checks it (against
# SHA256 if given) and the version in its header, and installs it with a marker.
install_file() {
    local commit="$1" version="$2" want="${3:-}" got
    TMP=$(mktemp -d)
    trap 'rm -rf -- "$TMP"' EXIT
    echo "Downloading GuC $version from upstream linux-firmware ($commit)..."
    if ! curl -fsSL --max-time 120 -o "$TMP/bmg_guc_70.bin" \
        "https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/plain/xe/bmg_guc_70.bin?id=$commit"; then
        echo "FATAL: download failed"
        exit 1
    fi
    got=$(sha256sum "$TMP/bmg_guc_70.bin" | cut -d' ' -f1)
    if [[ -n "$want" && "$got" != "$want" ]]; then
        echo "FATAL: SHA-256 mismatch (got $got, want $want); nothing installed."
        exit 1
    fi
    if [[ "$(guc_version "$TMP/bmg_guc_70.bin")" != "$version" ]]; then
        echo "FATAL: the file's header says version $(guc_version "$TMP/bmg_guc_70.bin"), not $version; nothing installed."
        exit 1
    fi
    if [[ -f "$TARGET" && ! -f "$MARKER" ]]; then
        echo "FATAL: $TARGET exists and wasn't installed by this script; left alone."
        exit 1
    fi
    sudo install -D -m 0644 "$TMP/bmg_guc_70.bin" "$TARGET" &&
        printf 'GuC %s for Battlemage from linux-firmware %s\nsha256 %s\ninstalled %s by guc_firmware.sh\n' \
            "$version" "$commit" "$got" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | sudo tee "$MARKER" >/dev/null || {
        echo "FATAL: couldn't write $TARGET"
        exit 1
    }
    echo "Installed $TARGET (GuC $version). Rebuilding the initramfs for every installed kernel..."
    sudo update-initramfs -u -k all || {
        echo "FATAL: update-initramfs failed; the firmware file is installed, run it again by hand."
        exit 1
    }
    status
    echo
    echo "Reboot, then check with: ./guc_firmware.sh status  (found release version $version)"
}

# loaded_version: the GuC release the driver loaded, from debugfs.
loaded_version() {
    sudo sh -c 'for f in /sys/kernel/debug/dri/*/gt0/uc/guc_info; do [ -r "$f" ] && grep -m1 "found release" "$f" && break; done' 2>/dev/null |
        grep -o '[0-9]*\.[0-9]*\.[0-9]*' | head -1
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
        install_file "$COMMIT" "$VERSION" "$SHA256"
        ;;
    latest)
        read -r commit version < <(latest_upstream)
        [[ -n "${version:-}" ]] || { echo "FATAL: couldn't read the newest GuC from $LOG_URL"; exit 1; }
        sudo -v || exit 1
        loaded=$(loaded_version)
        echo "Newest upstream Battlemage GuC: $version (linux-firmware $commit)"
        echo "Loaded by the driver: ${loaded:-unknown}"
        if [[ -n "$loaded" ]] && version_newer "$version" "$loaded"; then
            echo "A newer GuC is available: ./guc_firmware.sh install-latest"
        else
            echo "Up to date."
        fi
        ;;
    install-latest)
        read -r commit version < <(latest_upstream)
        [[ -n "${version:-}" ]] || { echo "FATAL: couldn't read the newest GuC from $LOG_URL"; exit 1; }
        sudo -v || exit 1
        install_file "$commit" "$version"
        ;;
    uninstall)
        sudo -v || exit 1
        if [[ ! -f "$MARKER" ]]; then
            echo "Nothing to remove: no GuC firmware installed by this script."
            exit 0
        fi
        recorded=$(sed -n 's/^sha256 //p' "$MARKER")
        if [[ "$(sha256sum "$TARGET" 2>/dev/null | cut -d' ' -f1)" != "${recorded:-$SHA256}" ]]; then
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
