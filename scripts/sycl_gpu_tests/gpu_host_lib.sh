#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.2.0
# @adai-reviewed: 2026-11-10

# Shared helpers for prepare_host.sh, run_gpu_tests.sh and restore_host.sh: finding processes that
# use the GPU, and comparing oneAPI versions. Sourced, not run. Callers set SUDO to "sudo -n" (or
# leave it empty to see only their own processes).

# Library names of the oneAPI runtime the trainer loads; used to check where they resolve.
ONEAPI_LIB_PATTERN='libsycl|libmkl|libiomp|libur_|libimf|libsvml|libintlc|libirng|libumf|libtcm'

# oneapi_version_at_least HAVE WANT: true when Debian version HAVE is at least WANT's release
# (WANT like 2026.2.0; HAVE like 2026.2.0-863).
oneapi_version_at_least() {
    dpkg --compare-versions "$1" ge "$2"
}

# For dpkg's refusals to overwrite a file another package owns (Intel's 2026.2 runtime moved files
# between its own packages without declaring it; see prepare_host.sh).
# overwrite_conflicts LOG: dpkg's refusals in LOG, as "path owning-package" lines.
overwrite_conflicts() {
    grep -o "trying to overwrite '[^']*', which is also in package [^ ]*" "$1" 2>/dev/null |
        sed "s/trying to overwrite '\([^']*\)', which is also in package \([^ ]*\)/\1 \2/" | sort -u
}
# only_runtime_conflicts FILE: true when FILE lists conflicts and every one is between Intel's
# oneAPI runtime packages over a file under /opt/intel/oneapi.
only_runtime_conflicts() {
    [[ -s "$1" ]] && awk '$1 !~ "^/opt/intel/oneapi/" || $2 !~ /^intel-oneapi-runtime-/ {bad = 1} END {exit bad}' "$1"
}
# unit_from_cgroup TEXT: the systemd unit, docker container or user unit a /proc/<pid>/cgroup
# line belongs to, as "system <unit>", "user <uid> <unit>", "docker <id>" or "none".
unit_from_cgroup() {
    local path="${1##*::}"
    path="${path##*:}"
    if [[ "$path" =~ docker-([0-9a-f]{12,64})\.scope ]] || [[ "$path" =~ /docker/([0-9a-f]{12,64}) ]]; then
        echo "docker ${BASH_REMATCH[1]:0:12}"
    elif [[ "$path" =~ ^/user\.slice/user-([0-9]+)\.slice/user@[0-9]+\.service/(.*/)?([^/]+\.service)$ ]]; then
        echo "user ${BASH_REMATCH[1]} ${BASH_REMATCH[3]}"
    elif [[ "$path" =~ ^/system\.slice/(.*/)?([^/]+\.service)$ ]]; then
        echo "system ${BASH_REMATCH[2]}"
    else
        echo "none"
    fi
}

# process_start_time PID: field 22 of /proc/PID/stat (start time in clock ticks), so a PID can be
# checked to still be the same process before it is signalled.
process_start_time() {
    local stat
    stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
    stat="${stat##*) }"
    # shellcheck disable=SC2086
    set -- $stat
    echo "${20}"
}

# gpu_holder_pids: PIDs that have a /dev/dri device open, one per line. Needs root (SUDO) to see
# other users' processes.
gpu_holder_pids() {
    ${SUDO:-} find /proc/[0-9]*/fd -maxdepth 1 -lname '/dev/dri/*' -printf '%h\n' 2>/dev/null |
        cut -d/ -f3 | sort -un
}

# Display servers, desktop shells and browsers keep the GPU open to draw the screen; they are
# reported but never stopped, and don't count as other GPU work. Matched against /proc/<pid>/comm
# (at most 15 characters).
DESKTOP_PATTERN='^(Xorg|Xwayland|gnome-shell|mutter.*|kwin.*|plasmashell|sway|Hyprland|weston|gdm.*|sddm.*|lightdm|xdg-desktop-po.*|gsd-.*|nautilus|firefox|Web Content|RDD Process|chrome|chromium.*|code|.*desktop.*)$'

is_desktop_pid() {
    grep -q -E "$DESKTOP_PATTERN" "/proc/$1/comm" 2>/dev/null
}

# gpu_work_pids: gpu_holder_pids without the desktop.
gpu_work_pids() {
    local pid
    for pid in $(gpu_holder_pids); do
        is_desktop_pid "$pid" || echo "$pid"
    done
}

# gpu_vram_mib PID: device memory the process holds on the GPU, from the xe driver's per-client
# fdinfo (drm-total-vram0, KiB), summed over its distinct DRM clients; empty if unknown.
gpu_vram_mib() {
    ${SUDO:-} sh -c 'for f in /proc/'"$1"'/fdinfo/*; do cat "$f" 2>/dev/null; echo; done' 2>/dev/null |
        awk '/^drm-client-id:/ {id=$2} /^drm-total-vram0:/ && id != "" && !(id in seen) {seen[id]=1; kib+=$2}
             END {if (kib > 0) printf "%.0f", kib / 1024}'
}

# describe_pid PID: one tab-separated line: pid, user, owner (unit_from_cgroup), VRAM MiB, command.
describe_pid() {
    local pid="$1" user owner vram cmd
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    owner=$(unit_from_cgroup "$(head -1 "/proc/$pid/cgroup" 2>/dev/null)")
    vram=$(gpu_vram_mib "$pid")
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)
    printf '%s\t%s\t%s\t%s\t%s\n' "$pid" "${user:-?}" "$owner" "${vram:-?}" "${cmd:-?}"
}

# GPU programs worth looking for even when they hold no GPU device right now (an idle Ollama
# loads a model onto the GPU on the next request).
GPU_SUSPECT_PATTERN='ollama|llama[-_.]?(server|cli|cpp)|vllm|qwen|sglang|ipex|koboldcpp|lmstudio|text-generation|comfyui|stable-diffusion|whisper|ramalama|xpu-smi|intel-gpu-top|incremental_trainer|trainer_service|chatbot_api_server'

# gpu_suspect_pids: running processes whose command line matches GPU_SUSPECT_PATTERN, except this
# shell and its children.
gpu_suspect_pids() {
    local pid
    for pid in $(pgrep -f -i -E "$GPU_SUSPECT_PATTERN" 2>/dev/null); do
        [[ "$pid" == "$$" || "$pid" == "$BASHPID" ]] && continue
        [[ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$$" ]] && continue
        grep -q -E 'gpu_host_lib|prepare_host|restore_host|run_gpu_tests' "/proc/$pid/cmdline" 2>/dev/null && continue
        echo "$pid"
    done
}

# gpu_suspect_units: systemd services (system and this user's) whose names match the pattern, with
# their state and whether they start at boot.
gpu_suspect_units() {
    {
        systemctl list-unit-files --type=service --no-legend --no-pager 2>/dev/null | awk '{print "system", $1, $2}'
        systemctl --user list-unit-files --type=service --no-legend --no-pager 2>/dev/null | awk '{print "user", $1, $2}'
    } | grep -i -E "$GPU_SUSPECT_PATTERN|adai" | while read -r scope unit enabled; do
        local active
        if [[ "$scope" == user ]]; then
            active=$(systemctl --user is-active "$unit" 2>/dev/null)
        else
            active=$(systemctl is-active "$unit" 2>/dev/null)
        fi
        printf '%s\t%s\t%s\t%s\n' "$scope" "$unit" "${active:-unknown}" "$enabled"
    done
}

# gpu_docker_containers: running containers given a /dev/dri device, as "id name".
gpu_docker_containers() {
    command -v docker >/dev/null || return 0
    local docker=(docker)
    docker ps -q >/dev/null 2>&1 || docker=(${SUDO:-} docker)
    "${docker[@]}" ps -q 2>/dev/null | while read -r id; do
        "${docker[@]}" inspect --format '{{.Id}} {{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' \
            "$id" 2>/dev/null | awk '/\/dev\/dri/ {print substr($1, 1, 12), substr($2, 2)}'
    done
}

# gpu_report: a readable report of everything above.
gpu_report() {
    local pid any=0 desktop=()
    echo "Processes with the GPU open (pid, user, owner, VRAM MiB, command):"
    for pid in $(gpu_holder_pids); do
        if is_desktop_pid "$pid"; then
            desktop+=("$pid")
            continue
        fi
        echo "  $(describe_pid "$pid")"
        any=1
    done
    [[ $any == 0 ]] && echo "  none"
    if [[ ${#desktop[@]} -gt 0 ]]; then
        echo "Desktop programs with the GPU open (left alone):"
        for pid in "${desktop[@]}"; do
            echo "  $(describe_pid "$pid" | cut -c1-140)"
        done
    fi
    [[ -z "${SUDO:-}" ]] && echo "  (without sudo only this user's processes are visible)"
    any=0
    echo "Running GPU programs by name (may load the GPU on demand):"
    for pid in $(gpu_suspect_pids); do
        echo "  $(describe_pid "$pid")"
        any=1
    done
    [[ $any == 0 ]] && echo "  none"
    echo "Services by name (scope, unit, state, start at boot):"
    gpu_suspect_units | sed 's/^/  /' | grep . || echo "  none"
    echo "Docker containers given the GPU:"
    gpu_docker_containers | sed 's/^/  /' | grep . || echo "  none"
}
