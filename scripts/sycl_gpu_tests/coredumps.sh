#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-14

# Makes the GPU host (ai-machine) keep crash dumps for diagnosis (TD-574).
#
#   ./coredumps.sh status | install | uninstall [--days N] [--user NAME]
#
# Two kinds of dump, both lost by default:
# - Process core dumps (a SIGSEGV in libigc, an abort): systemd-coredump stores them, but its own
#   tmpfiles rule deletes them after 3 days, and the adai services run with LimitCORE=0. install
#   keeps them N days (default 30, at most 20 GB and never below 10 GB free) and enables core dumps
#   for every adai* service (from its next restart) and for the user's logins (from the next one).
# - xe GPU crash dumps (one per engine reset: the engine, context and GuC state of the hung job): the
#   driver holds one in /sys/class/devcoredump until it is read and freed, or for 5 minutes, and
#   writes no other while it holds one. install adds a udev rule that saves each dump, gzipped, to
#   /var/lib/adai/gpu-coredumps/ the moment the driver writes it and then frees it, so every reset
#   gets one. Kept N days too, readable by the user's group.
# run_gpu_tests.sh copies the dumps saved during a run into its results.
# uninstall removes what install added and leaves the saved dumps in place. Uses sudo.
#
# COREDUMPS_ROOT=DIR (tests only) writes the files under DIR and runs no system command.

set -uo pipefail

ROOT="${COREDUMPS_ROOT:-}"
DAYS=30
TARGET_USER="${SUDO_USER:-$(id -un)}"
GPU_DUMP_DIR=/var/lib/adai/gpu-coredumps
SAVER=/usr/local/sbin/adai-save-gpu-coredump
UNIT=adai-gpu-coredump@.service
FILES=(
    /etc/systemd/coredump.conf.d/50-adai.conf
    /etc/tmpfiles.d/00-adai-coredumps.conf
    /etc/udev/rules.d/90-adai-gpu-coredump.rules
    /etc/systemd/system/$UNIT
    $SAVER
    /etc/security/limits.d/90-adai-core.conf
)
DROPIN=50-adai-coredump.conf

usage() {
    sed -n '/^#   \.\/coredumps/,/^# COREDUMPS_ROOT/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

# as_root CMD...: runs CMD with sudo, or directly under COREDUMPS_ROOT.
as_root() {
    if [[ -n "$ROOT" ]]; then "$@"; else sudo "$@"; fi
}

# system CMD...: a system command, skipped under COREDUMPS_ROOT.
system() {
    [[ -n "$ROOT" ]] && return 0
    sudo "$@"
}

# put PATH MODE: writes stdin to ROOT/PATH.
put() {
    as_root mkdir -p "$(dirname "$ROOT$1")" &&
        as_root tee "$ROOT$1" >/dev/null &&
        as_root chmod "$2" "$ROOT$1"
}

# adai_units: the installed adai* services (names only).
adai_units() {
    if [[ -n "$ROOT" ]]; then
        printf '%s\n' ${COREDUMPS_TEST_UNITS:-}
        return
    fi
    systemctl list-unit-files --type=service --no-legend 'adai*' 2>/dev/null | awk '{print $1}' | grep -v '@'
}

status() {
    echo "Kernel core pattern: $(cat /proc/sys/kernel/core_pattern 2>/dev/null)"
    if grep -q systemd-coredump /proc/sys/kernel/core_pattern 2>/dev/null; then
        echo "  systemd-coredump receives process core dumps."
    else
        echo "  not systemd-coredump: process core dumps are not kept (install offers to fix this)."
    fi
    echo "This shell's core size limit: soft $(ulimit -S -c), hard $(ulimit -H -c)"
    local f
    for f in "${FILES[@]}"; do
        if [[ -e "$ROOT$f" ]]; then echo "installed: $f"; else echo "missing:   $f"; fi
    done
    local u
    for u in $(adai_units); do
        if [[ -e "$ROOT/etc/systemd/system/$u.d/$DROPIN" ]]; then
            echo "core dumps on: $u (from its next restart)"
        else
            echo "core dumps off: $u"
        fi
    done
    if [[ -d "$ROOT$GPU_DUMP_DIR" ]]; then
        echo "Saved GPU dumps: $(find "$ROOT$GPU_DUMP_DIR" -name '*.gz' 2>/dev/null | wc -l) in $GPU_DUMP_DIR"
    fi
    if [[ -z "$ROOT" ]] && command -v coredumpctl >/dev/null; then
        echo "Process core dumps (last 5):"
        sudo coredumpctl list --no-pager 2>/dev/null | tail -5 | sed 's/^/  /'
    fi
}

write_saver() {
    put "$SAVER" 0755 <<EOF
#!/bin/sh
# Installed by coredumps.sh (adai, TD-574): saves one xe GPU crash dump, then frees it so the driver
# can write the next. Run by $UNIT from a udev rule.
dev="\$1"
src="/sys/class/devcoredump/\$dev/data"
[ -r "\$src" ] || exit 0
pci=\$(basename "\$(readlink -f "/sys/class/devcoredump/\$dev/failing_device" 2>/dev/null)")
out="$GPU_DUMP_DIR/\$(date +%Y%m%dT%H%M%S)_\${pci:-unknown}_\$dev.txt.gz"
mkdir -p "$GPU_DUMP_DIR"
# At most 256 MiB of one dump, and nothing while the directory holds 4 GiB or more.
used=\$(du -sm "$GPU_DUMP_DIR" | cut -f1)
if [ "\${used:-0}" -lt 4096 ]; then
    head -c 268435456 "\$src" | gzip -6 >"\$out.tmp" && mv "\$out.tmp" "\$out" && chmod 0640 "\$out"
    echo "saved \$out (\$(stat -c %s "\$out") bytes)"
else
    echo "$GPU_DUMP_DIR holds \${used} MiB: dump from \$pci not saved"
fi
echo 1 >"\$src"
EOF
}

install_files() {
    local group
    group=$(id -gn "$TARGET_USER" 2>/dev/null) || {
        echo "FATAL: no user $TARGET_USER"
        return 1
    }
    put /etc/systemd/coredump.conf.d/50-adai.conf 0644 <<EOF
# Installed by coredumps.sh (adai, TD-574): keep process core dumps for diagnosis.
[Coredump]
Storage=external
Compress=yes
ProcessSizeMax=32G
ExternalSizeMax=32G
MaxUse=20G
KeepFree=10G
EOF
    # The earliest file name wins for a path several tmpfiles rules name, so this replaces systemd's
    # own 3-day rule for its coredump directory (systemd-tmpfiles logs the other as a duplicate).
    put /etc/tmpfiles.d/00-adai-coredumps.conf 0644 <<EOF
# Installed by coredumps.sh (adai, TD-574): keep crash dumps $DAYS days.
d /var/lib/systemd/coredump 0755 root root ${DAYS}d
d $GPU_DUMP_DIR 0750 root $group ${DAYS}d
EOF
    put /etc/udev/rules.d/90-adai-gpu-coredump.rules 0644 <<EOF
# Installed by coredumps.sh (adai, TD-574): save every GPU crash dump as soon as it is written.
ACTION=="add", SUBSYSTEM=="devcoredump", RUN+="/bin/systemctl --no-block start adai-gpu-coredump@%k.service"
EOF
    put "/etc/systemd/system/$UNIT" 0644 <<EOF
# Installed by coredumps.sh (adai, TD-574).
[Unit]
Description=Save GPU crash dump %i

[Service]
Type=oneshot
ExecStart=$SAVER %i
EOF
    write_saver
    put /etc/security/limits.d/90-adai-core.conf 0644 <<EOF
# Installed by coredumps.sh (adai, TD-574): core dumps for $TARGET_USER's logins.
$TARGET_USER soft core unlimited
$TARGET_USER hard core unlimited
EOF
    local u
    for u in $(adai_units); do
        put "/etc/systemd/system/$u.d/$DROPIN" 0644 <<EOF
# Installed by coredumps.sh (adai, TD-574): overrides the unit's LimitCORE=0.
[Service]
LimitCORE=infinity
EOF
        echo "Core dumps enabled for $u from its next restart."
    done
    as_root mkdir -p "$ROOT$GPU_DUMP_DIR"
    as_root chown "root:$group" "$ROOT$GPU_DUMP_DIR" 2>/dev/null
    as_root chmod 0750 "$ROOT$GPU_DUMP_DIR"
}

ensure_systemd_coredump() {
    [[ -n "$ROOT" ]] && return 0
    grep -q systemd-coredump /proc/sys/kernel/core_pattern 2>/dev/null && return 0
    echo "Process core dumps currently go to: $(cat /proc/sys/kernel/core_pattern)"
    if ! dpkg-query -W -f='${Status}' systemd-coredump 2>/dev/null | grep -q 'install ok installed'; then
        read -r -p "Install systemd-coredump with apt (replaces apport's core handling)? [y/N] " a
        [[ "$a" == [yY]* ]] || {
            echo "Left as it is: process core dumps won't be kept."
            return 0
        }
        sudo apt-get install -y systemd-coredump || return 1
    fi
    if systemctl is-enabled apport.service >/dev/null 2>&1; then
        read -r -p "apport resets the core pattern at boot. Disable apport? [y/N] " a
        [[ "$a" == [yY]* ]] && sudo systemctl disable --now apport.service
    fi
    sudo sysctl -q -p /usr/lib/sysctl.d/50-coredump.conf
}

uninstall_files() {
    local f u
    for f in "${FILES[@]}"; do
        as_root rm -f -- "$ROOT$f"
    done
    for u in $(adai_units); do
        as_root rm -f -- "$ROOT/etc/systemd/system/$u.d/$DROPIN"
        as_root rmdir --ignore-fail-on-non-empty "$ROOT/etc/systemd/system/$u.d" 2>/dev/null
    done
    as_root rmdir --ignore-fail-on-non-empty "$ROOT/etc/systemd/coredump.conf.d" 2>/dev/null
}

reload() {
    system systemctl daemon-reload
    system udevadm control --reload
}

CMD="${1:-}"
[[ $# -gt 0 ]] && shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --days)
            DAYS="${2:-}"
            [[ "$DAYS" =~ ^[1-9][0-9]{0,3}$ ]] || {
                echo "--days needs a whole number from 1 to 9999" >&2
                exit 2
            }
            shift 2
            ;;
        --user)
            TARGET_USER="${2:-}"
            [[ -n "$TARGET_USER" ]] || {
                echo "--user needs a name" >&2
                exit 2
            }
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$CMD" in
    status)
        [[ -n "$ROOT" ]] || sudo -v || exit 1
        status
        ;;
    install)
        [[ -n "$ROOT" ]] || sudo -v || exit 1
        ensure_systemd_coredump || exit 1
        install_files || exit 1
        reload
        system systemd-tmpfiles --create /etc/tmpfiles.d/00-adai-coredumps.conf
        echo
        status
        echo
        echo "Process core dumps: kept $DAYS days (coredumpctl list). GPU dumps: $GPU_DUMP_DIR, $DAYS days."
        echo "Log in again for the core limit in your shell; restart adai services for theirs."
        ;;
    uninstall)
        [[ -n "$ROOT" ]] || sudo -v || exit 1
        uninstall_files
        reload
        echo "Removed. Saved GPU dumps stay in $GPU_DUMP_DIR; process core dumps are deleted on systemd's own 3-day schedule again."
        ;;
    -h | --help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
