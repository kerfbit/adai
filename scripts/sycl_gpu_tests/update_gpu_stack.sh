#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-15

# Updates the GPU host's (ai-machine's) Intel compute runtime and GuC firmware to the newest
# available, for TD-574's report to Intel, and can put the runtime back. Uses sudo; asks before
# each change.
#
#   ./update_gpu_stack.sh [check] | install [--yes] [--dbgsym] [--skip-guc] [--allow-shared-gpu]
#                       | rollback [--yes]
#
# check     (the default) changes nothing: shows the installed compute runtime packages, what the
#           Intel graphics PPA (ppa:kobuk-team/intel-graphics, where ai-machine's runtime comes from)
#           offers for them, and the GuC firmware loaded against the newest upstream one.
# install   1. Refuses while another process has the GPU open (stop the services first with
#              ./prepare_host.sh, or pass --allow-shared-gpu).
#           2. Keeps a copy of every runtime package it will upgrade, for rollback: from apt's cache,
#              or rebuilt from the installed files with dpkg-repack (the PPA keeps only its newest
#              versions, so they can't be downloaded again). Refuses if a copy can't be made.
#           3. Upgrades the runtime packages from the PPA: Level Zero (libze-intel-gpu1, libze1),
#              OpenCL (intel-opencl-icd, intel-ocloc), the graphics compiler (libigc2, libigdfcl2) and
#              gmmlib (libigdgmm12), and their development packages if installed (libze-dev and the
#              like must move in step with their libraries). It refuses if apt would install, upgrade
#              or remove anything else.
#              With --dbgsym it also installs their debug symbols from the PPA's debug archive (added
#              to apt as a copy of the PPA's own entry with the component main/debug), so the stacks
#              run_gpu_tests.sh takes name functions inside the runtime.
#           4. Checks which libigc and Level Zero driver now load, and that the GPU still runs work.
#           5. Installs the newest upstream GuC firmware with ./guc_firmware.sh install-latest if the
#              driver loads an older one (--skip-guc leaves it). That needs a reboot.
# rollback  Reinstalls the runtime packages the last install kept (allowing downgrades) and
#           removes the debug symbols it added. The GuC goes back with ./guc_firmware.sh install
#           (70.72.1) or ./guc_firmware.sh uninstall (Ubuntu's own), then a reboot.
#
# A report of each run goes to host_state/. The kernel and the xe driver stay as they are.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=gpu_host_lib.sh
source "$HERE/gpu_host_lib.sh"

PPA_LIST_PATTERN='kobuk-team/intel-graphics'
PPA_DEBUG_SOURCES=/etc/apt/sources.list.d/adai-intel-graphics-debug.sources
PPA_DEBUG_LIST=/etc/apt/sources.list.d/adai-intel-graphics-debug.list
# The compute runtime's packages: what install may upgrade or add, and nothing else.
RUNTIME_PACKAGES=(libze-intel-gpu1 libze1 intel-opencl-icd intel-ocloc libigc2 libigdfcl2 libigdgmm12)
# Development packages built from the same sources, pinned to their library's exact version: upgraded
# (and kept for rollback) along with it when installed, never added.
COMPANION_PACKAGES=(libze-dev libigdgmm-dev libigc-dev libigdfcl-dev intel-ocloc-dev)
DBGSYM_PACKAGES=(libze-intel-gpu1-dbgsym intel-opencl-icd-dbgsym libigc2-dbgsym libigdgmm12-dbgsym)

COMMAND=check
YES=0
DBGSYM=0
SKIP_GUC=0
ALLOW_SHARED=0
usage() {
    sed -n '/^#   \.\/update_gpu_stack/,/^# A report/p' "$0" | sed 's/^# \{0,1\}//'
}
for arg in "$@"; do
    case "$arg" in
        check | install | rollback) COMMAND=$arg ;;
        --yes) YES=1 ;;
        --dbgsym) DBGSYM=1 ;;
        --skip-guc) SKIP_GUC=1 ;;
        --allow-shared-gpu) ALLOW_SHARED=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

STATE="$HERE/host_state"
BACKUP="$STATE/runtime_backup"
mkdir -p "$STATE"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="$STATE/update_gpu_stack_${COMMAND}_$STAMP.txt"
exec > >(tee -a "$REPORT") 2>&1

say() { echo "$*"; }
ask() {
    if [[ $YES == 1 ]]; then
        echo "$1 [y/N] y (--yes)"
        return 0
    fi
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty || return 1
    [[ "$reply" =~ ^[Yy] ]]
}

# installed_runtime: "<package> <version>" for each installed runtime (or companion) package.
installed_runtime() {
    dpkg-query -W -f='${db:Status-Abbrev} ${Package} ${Version}\n' "${RUNTIME_PACKAGES[@]}" "${COMPANION_PACKAGES[@]}" 2>/dev/null |
        awk '$1 == "ii" {print $2, $3}'
}
# candidate PACKAGE: the version apt would install.
candidate() {
    apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/ {print $2; exit}'
}
# Which libigc and Level Zero GPU driver the dynamic linker finds, in the order it searches.
library_resolution() {
    ldconfig -p | grep -E 'libigc\.so\.2|libze_intel_gpu\.so\.1|libze_loader\.so\.1|libigdgmm\.so\.12' | sed 's/^\t*/  /'
}
# outside_runtime: the packages an `apt-get -s` plan on stdin would install, upgrade or remove that
# aren't compute runtime packages (or their debug symbols), a companion package that isn't an upgrade
# of an installed one, and every removal.
outside_runtime() {
    local allowed=" ${RUNTIME_PACKAGES[*]} ${DBGSYM_PACKAGES[*]} " companions=" ${COMPANION_PACKAGES[*]} " verb p rest
    while read -r verb p rest; do
        case "$verb" in
            Remv) echo "$p (removed)" ;;
            Inst)
                [[ "$allowed" == *" ${p%%:*} "* ]] && continue
                [[ "$companions" == *" ${p%%:*} "* && "$rest" == \[* ]] && continue
                echo "$p"
                ;;
        esac
    done
}
# debug_entry FILE: the PPA's apt entry in FILE (deb822 .sources or one-line .list) with the component
# main/debug instead of main, keeping its signing key.
debug_entry() {
    case "$1" in
        *.sources) sed -E 's/^Components:.*/Components: main\/debug/; s/^Types:.*/Types: deb/' "$1" ;;
        *) grep -m1 "^deb .*$PPA_LIST_PATTERN" "$1" | sed -E 's/ main( |$)/ main\/debug\1/' ;;
    esac
}
# gpu_probe: whether the GPU runs work, with the packaged device check.
gpu_probe() {
    [[ -x "$HERE/bin/gpuutilsTests" ]] || { say "  (no bin/gpuutilsTests: device check skipped)"; return 0; }
    LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib timeout 120 "$HERE/bin/gpuutilsTests" \
        --gtest_filter='GPUManagerTest.ProbeDiagnosticIsNonEmpty:GPUManagerTest.InitializeMatchesProbe' >"$STATE/gpu_probe_$STAMP.log" 2>&1
    local rc=$?
    grep -i -E 'device 0|backend|driver|PASSED|FAILED' "$STATE/gpu_probe_$STAMP.log" | head -8 | sed 's/^/  /'
    return $rc
}

say "ADAI GPU stack update ($COMMAND), $STAMP, $(hostname -s 2>/dev/null || hostname)"
say "Report: $REPORT"
say "Kernel: $(uname -r) (left as it is)"
if ! sudo -v; then
    say "FATAL: sudo is needed."
    exit 1
fi
SUDO="sudo -n"

if ! grep -rqs "$PPA_LIST_PATTERN" /etc/apt/sources.list /etc/apt/sources.list.d/; then
    say "FATAL: the Intel graphics PPA (ppa:kobuk-team/intel-graphics) isn't configured; this script only"
    say "       upgrades from it, the source ai-machine's runtime came from."
    exit 1
fi

# ── check ──────────────────────────────────────────────────────────────────────

if [[ $COMMAND == check || $COMMAND == install ]]; then
    say ""
    say "== Compute runtime"
    say "Refreshing the package lists (apt-get update)..."
    sudo apt-get update -qq || say "WARNING: apt-get update reported errors; continuing with the lists it has."
    installed_runtime >"$STATE/runtime_packages_before_$STAMP.txt"
    printf '  %-20s %-34s %s\n' package installed "newest in the PPA"
    for pkg in "${RUNTIME_PACKAGES[@]}"; do
        inst=$(awk -v p="$pkg" '$1 == p {print $2}' "$STATE/runtime_packages_before_$STAMP.txt")
        printf '  %-20s %-34s %s\n' "$pkg" "${inst:-(not installed)}" "$(candidate "$pkg")"
    done
    for pkg in "${COMPANION_PACKAGES[@]}"; do
        inst=$(awk -v p="$pkg" '$1 == p {print $2}' "$STATE/runtime_packages_before_$STAMP.txt")
        [[ -n "$inst" ]] && printf '  %-20s %-34s %s\n' "$pkg" "$inst" "$(candidate "$pkg")"
    done
    say "Libraries the linker finds:"
    library_resolution
    say ""
    say "== GuC firmware"
    "$HERE/guc_firmware.sh" latest | sed 's/^/  /'
fi
if [[ $COMMAND == check ]]; then
    say ""
    say "Nothing changed. To update: ./update_gpu_stack.sh install [--dbgsym]"
    exit 0
fi

# ── install ────────────────────────────────────────────────────────────────────

if [[ $COMMAND == install ]]; then
    say ""
    say "== Other GPU users"
    OTHERS=$(gpu_work_pids | tr '\n' ' ')
    if [[ -n "${OTHERS// /}" ]]; then
        for pid in $OTHERS; do describe_pid "$pid" | sed 's/^/  /'; done
        if [[ $ALLOW_SHARED == 0 ]]; then
            say "FATAL: processes have the GPU open; stop them first (./prepare_host.sh), or pass --allow-shared-gpu."
            exit 1
        fi
        say "  continuing (--allow-shared-gpu); those processes keep the old runtime until restarted"
    else
        say "  none"
    fi

    if [[ $DBGSYM == 1 && ! -f "$PPA_DEBUG_SOURCES" && ! -f "$PPA_DEBUG_LIST" ]]; then
        # The debug archive is the PPA's own entry with the component main/debug, so it keeps the
        # PPA's signing key (often embedded in a .sources file's Signed-By).
        src=$(grep -l -s "$PPA_LIST_PATTERN" /etc/apt/sources.list.d/*.sources | head -1)
        lst=$(grep -l -s "^deb .*$PPA_LIST_PATTERN" /etc/apt/sources.list.d/*.list | head -1)
        if [[ -n "$src" ]]; then
            entry=$(debug_entry "$src")
            target=$PPA_DEBUG_SOURCES
        elif [[ -n "$lst" ]]; then
            entry=$(debug_entry "$lst")
            target=$PPA_DEBUG_LIST
        fi
        if [[ -z "${entry:-}" ]]; then
            say "WARNING: couldn't read the PPA's apt entry; no debug symbols."
            DBGSYM=0
        else
            say "The PPA's debug-symbol archive, to add as $target:"
            sed 's/^/  /' <<<"$entry" | grep -v -E '^  [[:space:]]' | grep -v 'BEGIN PGP\|END PGP'
            if ask "Add it?"; then
                echo "$entry" | sudo tee "$target" >/dev/null
                sudo apt-get update -qq || say "WARNING: apt-get update reported errors."
            else
                DBGSYM=0
            fi
        fi
    fi

    # Upgrade every runtime package that's installed, plus what they now need from the same set.
    mapfile -t PKGS < <(awk '{print $1}' "$STATE/runtime_packages_before_$STAMP.txt")
    [[ ${#PKGS[@]} -gt 0 ]] || { say "FATAL: no compute runtime packages installed."; exit 1; }
    WANT=("${PKGS[@]}")
    [[ $DBGSYM == 1 ]] && WANT+=("${DBGSYM_PACKAGES[@]}")
    PLAN=$(LC_ALL=C sudo apt-get -s install --no-install-recommends "${WANT[@]}" 2>&1)
    if grep -q -E '^E: |unmet dependencies' <<<"$PLAN"; then
        say "FATAL: apt can't plan the upgrade:"
        grep -E '^E: |^ ' <<<"$PLAN" | sed 's/^/  /'
        exit 1
    fi
    INST=$(grep '^Inst ' <<<"$PLAN" || true)
    OUTSIDE=$(outside_runtime <<<"$PLAN")
    if [[ -n "$OUTSIDE" ]]; then
        say "FATAL: the upgrade would also change packages outside the compute runtime; nothing changed:"
        sed 's/^/  /' <<<"$OUTSIDE"
        exit 1
    fi
    if [[ -z "$INST" ]]; then
        say ""
        say "== Compute runtime: already the newest the PPA has"
    else
        say ""
        say "== Compute runtime: apt would install"
        sed 's/^Inst /  /' <<<"$INST"

        # Copies of what will be replaced, for rollback.
        say "Keeping copies of the installed packages for rollback ($BACKUP)..."
        mkdir -p "$BACKUP.new"
        missing=()
        while read -r pkg ver; do
            grep -q "^Inst $pkg " <<<"$INST" || continue
            file="${pkg}_${ver//:/%3a}_amd64.deb"
            if [[ -f "/var/cache/apt/archives/$file" ]]; then
                cp -- "/var/cache/apt/archives/$file" "$BACKUP.new/"
            elif command -v dpkg-repack >/dev/null; then
                (cd "$BACKUP.new" && sudo dpkg-repack "$pkg" >/dev/null 2>&1 && sudo chown "$(id -u):$(id -g)" ./*.deb)
                ls "$BACKUP.new/${pkg}_"*.deb >/dev/null 2>&1 || missing+=("$pkg $ver")
            else
                missing+=("$pkg $ver")
            fi
        done <"$STATE/runtime_packages_before_$STAMP.txt"
        if [[ ${#missing[@]} -gt 0 ]]; then
            say "  no copy of: ${missing[*]}"
            if ! command -v dpkg-repack >/dev/null && ask "Install dpkg-repack (Ubuntu) to rebuild them from the installed files?"; then
                sudo apt-get install -y -qq dpkg-repack
                for m in "${missing[@]}"; do
                    (cd "$BACKUP.new" && sudo dpkg-repack "${m%% *}" >/dev/null 2>&1 && sudo chown "$(id -u):$(id -g)" ./*.deb)
                done
                missing=()
                while read -r pkg ver; do
                    grep -q "^Inst $pkg " <<<"$INST" || continue
                    ls "$BACKUP.new/${pkg}_"*.deb >/dev/null 2>&1 || missing+=("$pkg $ver")
                done <"$STATE/runtime_packages_before_$STAMP.txt"
            fi
        fi
        if [[ ${#missing[@]} -gt 0 ]]; then
            say "FATAL: no rollback copy of ${missing[*]}; nothing changed."
            exit 1
        fi
        # The newest backup replaces the previous one only once it is complete.
        [[ -d "$BACKUP" ]] && mv -- "$BACKUP" "$BACKUP.$STAMP.old"
        mv -- "$BACKUP.new" "$BACKUP"
        cp -- "$STATE/runtime_packages_before_$STAMP.txt" "$BACKUP/versions.txt"
        [[ $DBGSYM == 1 ]] && printf '%s\n' "${DBGSYM_PACKAGES[@]}" >"$BACKUP/dbgsym_added.txt"
        ls "$BACKUP" | sed 's/^/  /'

        if ! ask "Install these compute runtime packages?"; then
            say "Nothing changed."
            exit 0
        fi
        if ! sudo apt-get install -y --no-install-recommends "${WANT[@]}"; then
            say "FATAL: the install failed. ./update_gpu_stack.sh rollback restores the packages kept above."
            exit 1
        fi
    fi
    installed_runtime >"$STATE/runtime_packages_after_$STAMP.txt"
    say ""
    say "== After"
    diff "$STATE/runtime_packages_before_$STAMP.txt" "$STATE/runtime_packages_after_$STAMP.txt" | grep '^[<>]' | sed 's/^/  /'
    say "Libraries the linker finds:"
    library_resolution
    if ls /usr/local/lib/libigc.so.2 >/dev/null 2>&1; then
        say "WARNING: /usr/local/lib/libigc.so.2 exists (a GitHub IGC install) and is found before the PPA's libigc2."
    fi
    say "Device check:"
    if gpu_probe; then
        say "  the GPU runs work"
    else
        say "WARNING: the device check failed ($STATE/gpu_probe_$STAMP.log). ./update_gpu_stack.sh rollback"
        say "         restores the previous runtime."
    fi

    if [[ $SKIP_GUC == 0 ]]; then
        say ""
        say "== GuC firmware"
        read -r commit newest < <(bash -c 'source <(sed -n "/^LOG_URL=/p; /^latest_upstream() {/,/^}/p" "$1"); latest_upstream' _ "$HERE/guc_firmware.sh")
        loaded=$($SUDO sh -c 'for f in /sys/kernel/debug/dri/*/gt0/uc/guc_info; do [ -r "$f" ] && grep -m1 "found release" "$f" && break; done' 2>/dev/null |
            grep -o '[0-9]*\.[0-9]*\.[0-9]*' | head -1)
        say "  loaded: ${loaded:-unknown}; newest upstream: ${newest:-unknown}"
        if [[ -n "${newest:-}" && -n "$loaded" && "$newest" != "$loaded" &&
            "$(printf '%s\n%s\n' "$newest" "$loaded" | sort -V | tail -1)" == "$newest" ]]; then
            if ask "Install GuC $newest (./guc_firmware.sh install-latest; needs a reboot)?"; then
                "$HERE/guc_firmware.sh" install-latest || say "WARNING: the GuC install failed; the runtime update stands."
                REBOOT=1
            fi
        else
            say "  up to date"
        fi
    fi

    say ""
    say "Done. Kept for rollback: $BACKUP"
    if [[ -n "${REBOOT:-}" ]]; then
        say "Reboot for the new GuC firmware, then check: ./guc_firmware.sh status"
    else
        say "Restart anything that uses the GPU (or reboot) so it loads the new runtime."
    fi
    say "Then ./prepare_host.sh and ./run_gpu_tests.sh; environment.txt records the new versions."
    exit 0
fi

# ── rollback ───────────────────────────────────────────────────────────────────

if [[ $COMMAND == rollback ]]; then
    if [[ ! -s "$BACKUP/versions.txt" ]]; then
        say "FATAL: nothing kept to roll back to ($BACKUP/versions.txt missing)."
        exit 1
    fi
    say ""
    say "== Rollback to"
    sed 's/^/  /' "$BACKUP/versions.txt"
    OTHERS=$(gpu_work_pids | tr '\n' ' ')
    if [[ -n "${OTHERS// /}" && $ALLOW_SHARED == 0 ]]; then
        say "FATAL: processes have the GPU open (pids $OTHERS); stop them first, or pass --allow-shared-gpu."
        exit 1
    fi
    if ! ask "Reinstall these packages (downgrading)?"; then
        say "Nothing changed."
        exit 0
    fi
    sudo apt-get install -y --allow-downgrades --no-install-recommends "$BACKUP"/*.deb || {
        say "FATAL: the reinstall failed."
        exit 1
    }
    if [[ -s "$BACKUP/dbgsym_added.txt" ]]; then
        mapfile -t added < <(dpkg-query -W -f='${Package}\n' $(cat "$BACKUP/dbgsym_added.txt") 2>/dev/null)
        [[ ${#added[@]} -gt 0 ]] && sudo apt-get remove -y "${added[@]}"
    fi
    say ""
    say "== After"
    installed_runtime | sed 's/^/  /'
    library_resolution
    gpu_probe && say "  the GPU runs work" || say "WARNING: the device check failed ($STATE/gpu_probe_$STAMP.log)."
    say "Restart anything that uses the GPU (or reboot). The GuC firmware is unchanged: ./guc_firmware.sh install"
    say "(70.72.1) or uninstall (Ubuntu's own), then reboot, to change it."
    exit 0
fi
