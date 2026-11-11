#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.2.3
# @adai-reviewed: 2026-11-10

# Prepares the GPU host (ai-machine) for run_gpu_tests.sh. Uses sudo; asks before each change.
#
#   ./prepare_host.sh [--yes] [--skip-upgrade] [--skip-gpu-check]
#
# 1. Upgrades the oneAPI runtime packages (intel-oneapi-runtime-*) from Intel's apt repository to
#    at least the oneAPI release the package was built with (MANIFEST.txt). Only those packages:
#    it refuses if apt would install, upgrade or remove anything else, so the GPU driver and the
#    kernel stay as they are. Afterwards it checks that the deployed binaries in /opt/adai/bin
#    still load the same libraries as before (the deployed chatbot_api_server uses /opt/adai/lib),
#    undoes any new linker configuration if they don't, and names any library they take from the
#    upgraded system runtime instead of their own bundle. Intel's 2026.2 runtime moved files between
#    its own packages without declaring it, which stops a plain upgrade half done; this script
#    recognises that, explains it, and finishes the upgrade (also when re-run after such a failure).
# 2. Lists every process using the GPU, GPU programs found by name (Ollama, llama.cpp, vLLM, Qwen
#    runs...), matching services and Docker containers, and offers to stop each one. Services
#    and containers it stops are recorded so ./restore_host.sh can start them again.
#
# --yes             answer yes to the upgrade and to stopping services and containers (never to
#                   signalling a process that isn't a service or container)
# --skip-upgrade    leave the oneAPI runtime as it is
# --skip-gpu-check  don't look for other GPU users

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=gpu_host_lib.sh
source "$HERE/gpu_host_lib.sh"
YES=0
DO_UPGRADE=1
DO_GPU_CHECK=1

usage() {
    sed -n '/^#   \.\/prepare_host/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    sed -n '/^# --yes/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        --yes) YES=1 ;;
        --skip-upgrade) DO_UPGRADE=0 ;;
        --skip-gpu-check) DO_GPU_CHECK=0 ;;
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
mkdir -p "$STATE"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="$STATE/prepare_$STAMP.txt"
STOPPED="$STATE/stopped.txt"
exec > >(tee -a "$REPORT") 2>&1

say() { echo "$*"; }
ask() {
    # ask QUESTION [auto]: true for yes; "auto" lets --yes answer.
    if [[ "${2:-}" == auto && $YES == 1 ]]; then
        echo "$1 [y/N] y (--yes)"
        return 0
    fi
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty || return 1
    [[ "$reply" =~ ^[Yy] ]]
}

say "ADAI GPU host preparation, $STAMP, $(hostname -s 2>/dev/null || hostname)"
say "Report: $REPORT"
say ""
if ! sudo -v; then
    say "FATAL: sudo is needed (to upgrade packages, see every process's GPU use and stop services)."
    exit 1
fi
SUDO="sudo -n"

BUILD_ONEAPI=$(sed -n 's/^oneapi: //p' "$HERE/MANIFEST.txt" 2>/dev/null)
if [[ -z "$BUILD_ONEAPI" ]]; then
    BUILD_ONEAPI=$(grep -o '([0-9]\{4\}\.[0-9]*\.[0-9]*' "$HERE/MANIFEST.txt" 2>/dev/null | head -1 | tr -d '(')
fi

# ── 1. oneAPI runtime ─────────────────────────────────────────────────────────

installed_runtime() {
    dpkg-query -W -f='${db:Status-Abbrev} ${Package} ${Version}\n' 'intel-oneapi-runtime-*' 2>/dev/null |
        awk '$1 == "ii" {print $2, $3}'
}
driver_packages() {
    dpkg-query -W -f='${db:Status-Abbrev} ${Package} ${Version}\n' 'intel-opencl-icd' 'intel-ocloc*' 'libze*' \
        'libigc*' 'intel-igc*' 'libigdgmm*' 'intel-level-zero*' 'level-zero*' 'linux-image-*' 2>/dev/null |
        awk '$1 == "ii" {print $2, $3}'
}
# Where each deployed binary's oneAPI libraries resolve, the thing that must not change.
deployed_resolution() {
    local bin
    for bin in /opt/adai/bin/*; do
        [[ -f "$bin" && -x "$bin" ]] || continue
        ldd "$bin" 2>/dev/null | grep -E "$ONEAPI_LIB_PATTERN" | awk -v b="$(basename "$bin")" '{print b, $1, $3}'
    done
}
ldconf_files() {
    find /etc/ld.so.conf.d -maxdepth 1 -type f -name '*.conf' 2>/dev/null | sort | xargs -r sha256sum
}
# Installed packages dpkg left part way through (unpacked or half-configured), one name per line.
unfinished_packages() {
    dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null | awk '$1 ~ /^i/ && $1 != "ii" {print $2}'
}
# blocking_packages: the unfinished packages outside the oneAPI runtime, except those only waiting
# for a trigger to run (status t or W: libc-bin's ldconfig trigger is queued whenever libraries are
# unpacked, and runs when the other packages are configured).
blocking_packages() {
    dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null |
        awk '$1 ~ /^i/ && $1 != "ii" && $2 !~ /^intel-oneapi-/ && substr($1, 2, 1) !~ /[tW]/ {print $2}'
}

# The state before the first upgrade on this host, kept outside the package directory so a re-run
# or a later package compares against it rather than against a half-upgraded state.
BASELINE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/adai-gpu-tests"

# Intel's 2026.2 runtime moved files between its own packages without declaring it: the Unified
# Runtime libraries (libur_loader and the Level Zero adapters) went from
# intel-oneapi-runtime-dpcpp-sycl-core to intel-oneapi-runtime-openmp, whose Replaces field doesn't
# say so. apt unpacks the new openmp package while the old sycl-core package still owns the files,
# dpkg refuses to overwrite them, and the upgrade stops half done.
APT_LOG="$STATE/apt_$STAMP.log"
# apt_logged ARGS...: sudo apt-get -y ARGS, its output shown and kept in APT_LOG and APT_LOG.last.
# NEEDRESTART_MODE=l makes Ubuntu's needrestart only list services using upgraded libraries: left to
# itself it restarted the deployed adai.service in the middle of the upgrade.
apt_logged() {
    sudo LC_ALL=C DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get -y -q "$@" 2>&1 | tee "$APT_LOG.last"
    local rc=${PIPESTATUS[0]}
    cat "$APT_LOG.last" >>"$APT_LOG"
    return "$rc"
}
# upgrade_runtime ARGS...: runs apt-get ARGS. When dpkg refuses only Intel's undeclared file moves,
# it explains, asks, and finishes the job letting the new packages take those files
# (--force-overwrite, for this one apt run): the old packages' new versions no longer contain them.
upgrade_runtime() {
    apt_logged "$@" && return 0
    overwrite_conflicts "$APT_LOG.last" >"$STATE/conflicts.txt"
    if ! only_runtime_conflicts "$STATE/conflicts.txt"; then
        say "  the failure isn't one of Intel's file moves between oneAPI runtime packages; see $APT_LOG"
        return 1
    fi
    say ""
    say "  dpkg refused to overwrite files that Intel's oneAPI runtime moved between two of its own"
    say "  packages without declaring it (a packaging bug in Intel's release). Conflicts (file, old owner):"
    sed 's/^/    /' "$STATE/conflicts.txt"
    say "  The new package takes these files and the old package's new version no longer contains them,"
    say "  so letting dpkg overwrite them finishes the upgrade correctly."
    ask "Finish the upgrade, letting dpkg overwrite these files?" auto || return 1
    apt_logged -o Dpkg::Options::=--force-overwrite install -f || return 1
    apt_logged -o Dpkg::Options::=--force-overwrite "$@" || return 1
    local path owner now
    while read -r path owner; do
        now=$(dpkg -S "$path" 2>/dev/null | cut -d: -f1)
        say "    $path now belongs to ${now:-no package}"
    done <"$STATE/conflicts.txt"
}

# check_deployed: after any upgrade attempt, whether the deployed binaries in /opt/adai/bin still
# load the libraries they loaded before the first upgrade, and which of them come from outside their
# own bundle (/opt/adai/lib), i.e. from the oneAPI runtime this script upgrades.
check_deployed() {
    sudo ldconfig
    deployed_resolution >"$STATE/deployed_libs_after.txt"
    if [[ ! -s "$STATE/deployed_libs_before.txt" && ! -s "$STATE/deployed_libs_after.txt" ]]; then
        say "  no deployed binaries in /opt/adai/bin"
        return 0
    fi
    if ! diff -q "$STATE/deployed_libs_before.txt" "$STATE/deployed_libs_after.txt" >/dev/null; then
        say "WARNING: the deployed binaries in /opt/adai/bin now resolve different oneAPI libraries:"
        diff "$STATE/deployed_libs_before.txt" "$STATE/deployed_libs_after.txt" | sed 's/^/  /'
        ldconf_files >"$STATE/ldconf_after.txt"
        local conf
        mapfile -t NEW_CONF < <(join -v1 -1 2 -2 2 <(sort -k2 "$STATE/ldconf_after.txt") \
            <(sort -k2 "$STATE/ldconf_before.txt") | awk '{print $1}')
        for conf in "${NEW_CONF[@]}"; do
            say "  setting aside the new linker configuration $conf"
            sudo mv -- "$conf" "$conf.disabled-by-adai-$STAMP"
            echo "ldconf $conf.disabled-by-adai-$STAMP $conf" >>"$STOPPED"
        done
        sudo ldconfig
        deployed_resolution >"$STATE/deployed_libs_after.txt"
        if diff -q "$STATE/deployed_libs_before.txt" "$STATE/deployed_libs_after.txt" >/dev/null; then
            say "  the deployed binaries resolve the same libraries as before again"
        else
            say "WARNING: still different. Don't restart the deployed services until this is checked:"
            diff "$STATE/deployed_libs_before.txt" "$STATE/deployed_libs_after.txt" | sed 's/^/  /'
        fi
    else
        say "  the deployed binaries in /opt/adai/bin resolve the same library paths as before"
    fi
    # Same paths can still mean new contents: a library the deployed bundle lacks is taken from the
    # runtime the packages install in /opt/intel/oneapi/redist, which is now a newer release than the
    # bundle. (A full toolkit's versioned directories, such as /opt/intel/oneapi/mkl/2026.1, aren't
    # touched by this upgrade.)
    awk '$3 ~ "^/opt/intel/+oneapi/+redist/"' "$STATE/deployed_libs_after.txt" >"$STATE/deployed_outside_bundle.txt"
    if [[ -s "$STATE/deployed_outside_bundle.txt" ]]; then
        say "WARNING: deployed binaries load these oneAPI libraries from the runtime this script upgrades"
        say "         rather than from their bundle in /opt/adai/lib (binary, library, path):"
        sed 's/^/    /' "$STATE/deployed_outside_bundle.txt"
        say "         A running service keeps the copies it loaded; after a restart it would mix its bundled"
        say "         oneAPI release with the system's. restore_host.sh asks before restarting one."
    fi
}

if [[ $DO_UPGRADE == 1 ]]; then
    say "== oneAPI runtime"
    say "Built with oneAPI: ${BUILD_ONEAPI:-unknown (MANIFEST.txt has no oneapi line)}"
    installed_runtime >"$STATE/runtime_before.txt"
    driver_packages >"$STATE/drivers_before.txt"
    mkdir -p "$BASELINE_DIR"
    if [[ ! -s "$BASELINE_DIR/deployed_libs.txt" ]]; then
        # An earlier package kept its snapshot in its own directory.
        earlier=$(ls -tr "$HERE"/../adai-sycl-gpu-tests-*/host_state/deployed_libs_before.txt 2>/dev/null | head -1)
        if [[ -n "$earlier" ]]; then
            cp -- "$earlier" "$BASELINE_DIR/deployed_libs.txt"
            cp -- "$(dirname "$earlier")/ldconf_before.txt" "$BASELINE_DIR/ldconf.txt" 2>/dev/null
            say "  comparing against the snapshot taken before the first upgrade attempt: $earlier"
        else
            deployed_resolution >"$BASELINE_DIR/deployed_libs.txt"
            ldconf_files >"$BASELINE_DIR/ldconf.txt"
        fi
    fi
    cp -- "$BASELINE_DIR/deployed_libs.txt" "$STATE/deployed_libs_before.txt"
    cp -- "$BASELINE_DIR/ldconf.txt" "$STATE/ldconf_before.txt" 2>/dev/null || ldconf_files >"$STATE/ldconf_before.txt"
    sed 's/^/  installed: /' "$STATE/runtime_before.txt"

    say "Refreshing the package lists (apt-get update)..."
    sudo apt-get update -qq || say "WARNING: apt-get update reported errors; continuing with the lists it has."

    UNFINISHED=$(unfinished_packages)
    if [[ -n "$UNFINISHED" ]]; then
        say "  an earlier upgrade stopped part way; these packages are unfinished:"
        sed 's/^/    /' <<<"$UNFINISHED"
        if [[ -n "$(blocking_packages)" ]]; then
            say "FATAL: packages outside the oneAPI runtime are unfinished too; repair them with apt first:"
            blocking_packages | sed 's/^/  /'
            exit 1
        fi
        if ! ask "Finish that upgrade now?" auto || ! upgrade_runtime install -f; then
            say "FATAL: the unfinished upgrade wasn't completed (log: $APT_LOG)."
            check_deployed
            exit 1
        fi
    fi

    mapfile -t PKGS < <(installed_runtime | awk '{print $1}')
    if [[ ${#PKGS[@]} -eq 0 ]]; then
        PKGS=(intel-oneapi-runtime-libs)
        say "  no oneAPI runtime packages installed; will install intel-oneapi-runtime-libs"
    fi
    PLAN=$(LC_ALL=C sudo apt-get -s install --only-upgrade --no-install-recommends "${PKGS[@]}" 2>&1)
    INST=$(grep '^Inst ' <<<"$PLAN" || true)
    REMV=$(grep '^Remv ' <<<"$PLAN" || true)
    OTHER=$(awk '/^Inst / && $2 !~ /^intel-oneapi-/ {print $2}' <<<"$PLAN")
    if grep -q -E '^E: |unmet dependencies' <<<"$PLAN"; then
        say "FATAL: apt can't plan the upgrade:"
        grep -E '^E: |^ ' <<<"$PLAN" | sed 's/^/  /'
        exit 1
    fi
    if [[ -n "$REMV" || -n "$OTHER" ]]; then
        say "FATAL: the upgrade would also change packages outside the oneAPI runtime; nothing changed:"
        printf '%s\n%s\n' "$REMV" "$OTHER" | grep . | sed 's/^/  /'
        exit 1
    fi
    if [[ -z "$INST" ]]; then
        say "  already up to date"
    else
        say "  apt would upgrade:"
        sed 's/^Inst /    /' <<<"$INST"
        if ask "Upgrade these oneAPI runtime packages?" auto; then
            if ! upgrade_runtime install --only-upgrade --no-install-recommends "${PKGS[@]}"; then
                say "FATAL: the upgrade failed (log: $APT_LOG)."
                check_deployed
                exit 1
            fi
        else
            say "  upgrade skipped"
        fi
    fi
    if [[ -n "$(unfinished_packages)" ]]; then
        say "FATAL: packages are still unfinished after the upgrade:"
        unfinished_packages | sed 's/^/  /'
        check_deployed
        exit 1
    fi
    installed_runtime >"$STATE/runtime_after.txt"
    driver_packages >"$STATE/drivers_after.txt"

    if ! diff -q "$STATE/drivers_before.txt" "$STATE/drivers_after.txt" >/dev/null; then
        say "WARNING: GPU driver or kernel packages changed:"
        diff "$STATE/drivers_before.txt" "$STATE/drivers_after.txt" | sed 's/^/  /'
    fi
    check_deployed

    RT_VERSION=$(awk '$1 == "intel-oneapi-runtime-dpcpp-cpp" {print $2}' "$STATE/runtime_after.txt")
    if [[ -n "$BUILD_ONEAPI" && -n "$RT_VERSION" ]]; then
        if oneapi_version_at_least "$RT_VERSION" "$BUILD_ONEAPI"; then
            say "  runtime $RT_VERSION matches the build ($BUILD_ONEAPI)"
        else
            say "WARNING: the runtime ($RT_VERSION) is still older than the build ($BUILD_ONEAPI)."
            say "         Intel's repository may not have it yet; run_gpu_tests.sh will refuse."
        fi
    fi
    say ""
fi

# ── 2. Other GPU users ────────────────────────────────────────────────────────

# record_stopped LINE: notes something stopped for restore_host.sh, once.
record_stopped() {
    grep -q -x -F -- "$1" "$STOPPED" 2>/dev/null || echo "$1" >>"$STOPPED"
}

# stop_pid PID: stops the service or container owning PID, or after a separate confirmation sends
# the process SIGTERM. Records what it stopped for restore_host.sh.
stop_pid() {
    local pid="$1" line owner kind
    [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || return 0
    line=$(describe_pid "$pid")
    owner=$(cut -f3 <<<"$line")
    read -r kind _ <<<"$owner"
    case "$kind" in
        system)
            local unit="${owner#system }"
            if ask "Stop service $unit (pid $pid: $(cut -f5 <<<"$line" | cut -c1-80))?" auto; then
                sudo systemctl stop "$unit" && record_stopped "unit $unit" && say "  stopped $unit"
            fi
            ;;
        user)
            local uid unit
            read -r _ uid unit <<<"$owner"
            if [[ "$uid" == "$(id -u)" ]]; then
                if ask "Stop your user service $unit (pid $pid)?" auto; then
                    systemctl --user stop "$unit" && record_stopped "userunit $unit" && say "  stopped $unit"
                fi
            else
                say "  pid $pid is user $uid's service $unit; stop it as that user: systemctl --user stop $unit"
            fi
            ;;
        docker)
            local id="${owner#docker }"
            if ask "Stop Docker container $id (pid $pid)?" auto; then
                local docker=(docker)
                docker ps -q >/dev/null 2>&1 || docker=(sudo docker)
                "${docker[@]}" stop "$id" >/dev/null && record_stopped "container $id" && say "  stopped container $id"
            fi
            ;;
        *)
            local start
            start=$(process_start_time "$pid") || return 0
            # Never answered by --yes: restore_host.sh can't restart a plain process.
            if ask "Send SIGTERM to pid $pid ($(cut -f2 <<<"$line"): $(cut -f5 <<<"$line" | cut -c1-80))?"; then
                if [[ "$(process_start_time "$pid")" != "$start" ]]; then
                    say "  pid $pid has exited or been reused; left alone"
                    return 0
                fi
                sudo kill -TERM -- "$pid"
                local i
                for i in $(seq 30); do
                    [[ "$(process_start_time "$pid" 2>/dev/null)" == "$start" ]] || break
                    sleep 1
                done
                if [[ -d "/proc/$pid" && "$(process_start_time "$pid")" == "$start" ]]; then
                    say "  pid $pid is still running after 30 s; stop it by hand (it is not killed)"
                else
                    echo "process $pid $(cut -f5 <<<"$line" | cut -c1-120)" >>"$STOPPED"
                    say "  pid $pid exited"
                fi
            fi
            ;;
    esac
}

if [[ $DO_GPU_CHECK == 1 ]]; then
    say "== Other GPU users"
    gpu_report
    say ""
    mapfile -t CANDIDATES < <({
        gpu_work_pids
        gpu_suspect_pids
    } | sort -un)
    for pid in "${CANDIDATES[@]}"; do
        [[ -d "/proc/$pid" ]] && stop_pid "$pid"
    done
    # Running services found by name that hold no process listed above (an idle model server), and
    # services restarting in a loop ("activating": a failing trainer_service relaunches, and each
    # launch can open the GPU). What is running now decides, not stopped.txt: a service recorded as
    # stopped before a reboot is running again.
    while IFS=$'\t' read -r scope unit active _; do
        [[ "$active" == active || "$active" == activating || "$active" == reloading ]] || continue
        if [[ "$scope" == system ]]; then
            ask "Stop $active service $unit?" auto && sudo systemctl stop "$unit" &&
                record_stopped "unit $unit" && say "  stopped $unit"
        else
            ask "Stop your $active user service $unit?" auto && systemctl --user stop "$unit" &&
                record_stopped "userunit $unit" && say "  stopped $unit"
        fi
    done < <(gpu_suspect_units)
    while read -r id name; do
        [[ -n "$id" ]] || continue
        if ask "Stop Docker container $name ($id), which has the GPU?" auto; then
            docker=(docker)
            docker ps -q >/dev/null 2>&1 || docker=(sudo docker)
            "${docker[@]}" stop "$id" >/dev/null && record_stopped "container $id" && say "  stopped container $name"
        fi
    done < <(gpu_docker_containers)

    say ""
    say "== After"
    REMAINING=$(gpu_work_pids | tr '\n' ' ')
    gpu_report
    say ""
    if [[ -n "${REMAINING// /}" ]]; then
        say "Processes still have the GPU open (pids: $REMAINING). run_gpu_tests.sh refuses to start"
        say "while they do, unless given --allow-shared-gpu."
    else
        say "Nothing else has the GPU open."
    fi
    if [[ -s "$STOPPED" ]]; then
        say "Stopped (./restore_host.sh starts the services and containers again):"
        sed 's/^/  /' "$STOPPED"
        say "Note: a GPU process stopped in the middle of GPU work can make the next one hang early"
        say "(reported on the B60, XeStrata issue #62). If one was busy, a reboot gives the cleanest"
        say "device; services that start at boot then need stopping again (re-run this script)."
    fi
fi

say ""
say "Next: ./run_gpu_tests.sh"
exit 0
