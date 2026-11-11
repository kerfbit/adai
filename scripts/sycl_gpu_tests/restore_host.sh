#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.2.0
# @adai-reviewed: 2026-11-10

# Starts again the services and Docker containers prepare_host.sh stopped (host_state/stopped.txt).
# The oneAPI runtime upgrade stays. When prepare_host.sh found that a deployed binary in /opt/adai/bin
# takes some oneAPI libraries from the upgraded system runtime rather than its own bundle, this
# script asks before starting a service that runs one, since it would then mix oneAPI releases.
#
#   ./restore_host.sh

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STOPPED="$HERE/host_state/stopped.txt"

case "${1:-}" in
    "") ;;
    -h | --help)
        sed -n '/^# Starts again/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        exit 2
        ;;
esac

if [[ ! -s "$STOPPED" ]]; then
    echo "Nothing to restore: prepare_host.sh stopped nothing (no $STOPPED)."
    exit 0
fi

sudo -v || {
    echo "FATAL: sudo is needed to start the services again."
    exit 1
}
OUTSIDE="$HERE/host_state/deployed_outside_bundle.txt"
# mixes_runtime UNIT: true when UNIT runs a deployed binary that loads libraries from outside its bundle.
mixes_runtime() {
    [[ -s "$OUTSIDE" ]] || return 1
    local exec bin
    exec=$(systemctl show -p ExecStart --value "$1" 2>/dev/null)
    for bin in $(awk '{print $1}' "$OUTSIDE" | sort -u); do
        [[ "$exec" == *"/opt/adai/bin/$bin"* ]] && return 0
    done
    return 1
}
FAILED=0
while read -r kind what rest; do
    case "$kind" in
        unit)
            if mixes_runtime "$what"; then
                echo "$what runs a deployed binary that loads these oneAPI libraries from the upgraded system"
                echo "runtime rather than its bundle in /opt/adai/lib (binary, library, path):"
                sed 's/^/  /' "$OUTSIDE"
                echo "Started now, it would mix its bundled oneAPI release with the system's newer one."
                read -r -p "Start $what anyway? [y/N] " reply </dev/tty || reply=n
                if [[ ! "$reply" =~ ^[Yy] ]]; then
                    echo "not started: $what (start it with: sudo systemctl start $what)"
                    echo "unit $what" >>"$STOPPED.kept"
                    continue
                fi
            fi
            if sudo systemctl start "$what"; then echo "started $what"; else echo "FAILED to start $what" && FAILED=1; fi
            ;;
        userunit)
            if systemctl --user start "$what"; then echo "started $what"; else echo "FAILED to start $what" && FAILED=1; fi
            ;;
        container)
            docker=(docker)
            docker ps -q >/dev/null 2>&1 || docker=(sudo docker)
            if "${docker[@]}" start "$what" >/dev/null; then echo "started container $what"; else echo "FAILED to start container $what" && FAILED=1; fi
            ;;
        process)
            echo "not restarted: process $what ($rest) was not a service; start it by hand if it's needed"
            ;;
        ldconf)
            echo "left set aside: linker configuration $rest (moved to $what so the deployed binaries keep their libraries)"
            ;;
    esac
done <"$STOPPED"
mv -- "$STOPPED" "$STOPPED.restored-$(date -u +%Y%m%dT%H%M%SZ)"
exit $FAILED
