#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-15

# Raises the xe driver's GuC log level on the GPU host (ai-machine), so the GuC log in every GPU crash
# dump says more about an engine reset, for TD-574's report to Intel; and puts it back. The level is
# a module option the driver reads when it loads, so each change needs a reboot. Uses sudo; asks
# before each change.
#
#   ./xe_debug.sh [status] | enable [--guc-log-level N] [--yes] | disable [--yes]
#
# status    (the default) changes nothing: the GuC log level the driver loaded with, the level the
#           crash dumps record, and the xe module options set in /etc/modprobe.d.
# enable    writes /etc/modprobe.d/adai-xe-debug.conf with "options xe guc_log_level=N" (default 3;
#           the driver takes 0 = off up to 5 = most verbose, and loads with 1), and rebuilds the
#           running kernel's initramfs if the xe module is in it. Takes effect at the next boot.
#           A more verbose GuC log costs the GuC some time, so rates measured with it are compared
#           only with passes of the same run.
# disable   removes that file (the driver's default level again) and rebuilds the initramfs the
#           same way. Takes effect at the next boot.
#
# The DRM core's debug messages (drm.debug) need no reboot: run_gpu_tests.sh --kernel-debug turns
# them on for its own passes only. A report of each run goes to host_state/.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONF=/etc/modprobe.d/adai-xe-debug.conf
DEFAULT_LEVEL=3

COMMAND=status
YES=0
LEVEL=$DEFAULT_LEVEL
usage() {
    sed -n '/^#   \.\/xe_debug/,/^# The DRM core/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        status | enable | disable) COMMAND=$1 ;;
        --yes) YES=1 ;;
        --guc-log-level)
            [[ "${2:-}" =~ ^[0-5]$ ]] || {
                echo "--guc-log-level needs a level from 0 to 5" >&2
                exit 2
            }
            LEVEL=$2
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

STATE="$HERE/host_state"
mkdir -p "$STATE"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee -a "$STATE/xe_debug_${COMMAND}_$STAMP.txt") 2>&1

ask() {
    if [[ $YES == 1 ]]; then
        echo "$1 [y/N] y (--yes)"
        return 0
    fi
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty || return 1
    [[ "$reply" =~ ^[Yy] ]]
}

# conf_line LEVEL: the modprobe.d line enable writes.
conf_line() {
    echo "options xe guc_log_level=$1"
}

status() {
    echo "Kernel: $(uname -r)"
    echo "GuC log level the driver loaded with: $(sudo cat /sys/module/xe/parameters/guc_log_level 2>/dev/null || echo '? (xe not loaded, or no sudo)')"
    local dump
    dump=$(sudo sh -c 'cat /sys/class/devcoredump/devcd*/data 2>/dev/null' | grep -a -m1 '^Log level:')
    [[ -n "$dump" ]] && echo "GuC log level in the crash dump waiting to be read: ${dump#Log level: }"
    echo "xe module options in /etc/modprobe.d:"
    grep -H -E '^[[:space:]]*options[[:space:]]+xe' /etc/modprobe.d/*.conf 2>/dev/null | sed 's/^/  /' || echo "  none"
    if [[ -f "$CONF" ]]; then
        echo "$CONF: $(cat "$CONF")"
    fi
}

# rebuild_initramfs: the running kernel's initramfs, when the xe module is in it (it then loads
# from there, with the options the initramfs was built with).
rebuild_initramfs() {
    local initrd="/boot/initrd.img-$(uname -r)"
    if command -v lsinitramfs >/dev/null && [[ -r "$initrd" ]] && lsinitramfs "$initrd" 2>/dev/null | grep -c '/xe\.ko' >/dev/null; then
        echo "The xe module is in $initrd: rebuilding it."
        sudo update-initramfs -u -k "$(uname -r)"
    else
        echo "The xe module isn't in the running kernel's initramfs: no rebuild needed."
    fi
}

case "$COMMAND" in
    status)
        status
        ;;
    enable)
        # Not `| grep -q`: with pipefail, grep stopping at the first match makes the pipeline fail.
        params=$(modinfo -p xe 2>&1)
        if ! grep -q '^guc_log_level:' <<<"$params" && [[ ! -e /sys/module/xe/parameters/guc_log_level ]]; then
            echo "FATAL: this kernel's xe module has no guc_log_level option; nothing changed. modinfo -p xe:"
            sed 's/^/  /' <<<"$params"
            exit 1
        fi
        status
        echo
        echo "Will write $CONF: $(conf_line "$LEVEL")"
        ask "Set the GuC log level to $LEVEL from the next boot?" || {
            echo "Nothing changed."
            exit 0
        }
        conf_line "$LEVEL" | sudo tee "$CONF" >/dev/null || {
            echo "FATAL: could not write $CONF."
            exit 1
        }
        rebuild_initramfs
        echo
        echo "Done. Reboot, then check with ./xe_debug.sh status (and ./prepare_host.sh before the run)."
        ;;
    disable)
        if [[ ! -f "$CONF" ]]; then
            echo "$CONF doesn't exist: the driver already loads with its default level."
            status
            exit 0
        fi
        ask "Remove $CONF ($(cat "$CONF")), back to the driver's default from the next boot?" || {
            echo "Nothing changed."
            exit 0
        }
        sudo rm -f -- "$CONF"
        rebuild_initramfs
        echo
        echo "Done. Reboot for the driver's default GuC log level."
        ;;
esac
