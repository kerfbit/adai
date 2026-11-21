#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.4.0
# @adai-reviewed: 2026-11-20

# Replaces the binaries an existing server-bundle host runs (the adai-mns, adai-registry and adai-metrics-api
# or adai-metrics units, as install_server_bundle.sh, install_metrics_service.sh or a person installed
# them) with the ones in this package, keeping
# the host's units, configs and data. install_server_bundle.sh is for a fresh host: it rewrites all three
# units and the config. This keeps whatever the host's units run with (ports, FTP, storage backend, data
# directories). Packed by package_server_bundle.sh; runs from the unpacked bundle, using sudo.
#
#   scripts/redeploy_server_bundle.sh [check] | install [--yes] | rollback [STAMP] [--with-data] [--yes]
#
# check     (the default) changes nothing: per service, the units installed for it and which one this
#           script would use (the running one; a host can have both adai-metrics-api and an old, stopped
#           adai-metrics), the binary it runs (and whether it's this package's already), its port, /health
#           and its data; the shared libraries this package's binaries need that the host lacks; the
#           backups so far. The units it doesn't use are left alone.
# install   stops the services (metrics, registry, MNS), backs up into /var/backups/adai-server-bundle/STAMP/
#           each binary they run, mns_cli, dataset_manager and vocab_builder next to it, the unit files, each unit's --config
#           file, /etc/adai and /opt/adai/config, MNS's data directory, the SQLite metrics database (from
#           --db-path, the unit's environment or its --config file) and the registry's queue files
#           (not its datasets), replaces the binaries (same path, owner and group), starts again the units
#           that were running (MNS, registry, metrics) and waits for each /health. Asks first.
# rollback  puts back STAMP's binaries (default the newest backup) and restarts; --with-data also puts
#           back MNS's data directory, the metrics database and the registry's queue files as they were
#           before that install (a newer binary may have migrated them).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_BIN="$(cd "$HERE/.." && pwd)/bin"
BACKUP_ROOT=${ADAI_SERVER_BACKUP_ROOT:-/var/backups/adai-server-bundle}
# The host's config directories, copied into each backup.
CONF_DIRS=${ADAI_SERVER_CONF_DIRS:-/etc/adai /opt/adai/config}

# service, binary, default port, then its unit names in order of preference: started in this order,
# stopped in reverse. install_metrics_service.sh names the metrics unit adai-metrics-api;
# install_server_bundle.sh, adai-metrics.
SERVICES=("mns mns_server 8083 adai-mns" "registry registry_server 8082 adai-registry"
    "metrics metrics_api_server 8081 adai-metrics-api adai-metrics")
CLIS=(mns_cli dataset_manager vocab_builder)

COMMAND=check
YES=0
STAMP_ARG=""
WITH_DATA=0
usage() { sed -n '/^#   scripts\/redeploy_server_bundle/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        check | install | rollback) COMMAND=$1 ;;
        --yes) YES=1 ;;
        --with-data) WITH_DATA=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        [0-9]*T[0-9]*Z)
            [[ $COMMAND == rollback ]] || { echo "ERROR: a backup stamp only goes with rollback" >&2; exit 2; }
            STAMP_ARG=$1
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

ask() {
    if [[ $YES == 1 ]]; then
        echo "$1 [y/N] y (--yes)"
        return 0
    fi
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty || return 1
    [[ "$reply" =~ ^[Yy] ]]
}

unit_installed() { systemctl cat "$1.service" >/dev/null 2>&1; }
unit_active() { [[ "$(systemctl is-active "$1.service" 2>/dev/null)" == active ]]; }
# pick_unit ENTRY: the unit to use for a SERVICES entry: the first of its units that is running, else the
# first installed; empty when none is installed.
pick_unit() {
    local service bin port units u
    read -r service bin port units <<<"$1"
    for u in $units; do
        unit_installed "$u" && unit_active "$u" && { echo "$u"; return; }
    done
    for u in $units; do
        unit_installed "$u" && { echo "$u"; return; }
    done
}
# setting UNIT FLAG KEY DEFAULT: what the unit's server uses for a setting, by the servers' own precedence:
# the command-line FLAG, else KEY in the unit's Environment=, else KEY in its --config file, else DEFAULT.
setting() {
    local line v cfg
    line=$(exec_line "$1")
    v=$(flag_value "$line" "$2" || true)
    [[ -n "$v" ]] && { echo "$v"; return; }
    v=$(systemctl show -p Environment --value "$1.service" 2>/dev/null | tr ' ' '\n' | sed -n "s/^$3=//p" | tail -1)
    [[ -n "$v" ]] && { echo "$v"; return; }
    cfg=$(flag_value "$line" --config || true)
    if [[ -n "$cfg" ]]; then
        v=$(sudo sed -n "s/^[[:space:]]*$3[[:space:]]*=[[:space:]]*//p" "$cfg" 2>/dev/null | tail -1 | sed 's/[[:space:]]*$//')
        [[ -n "$v" ]] && { echo "$v"; return; }
    fi
    echo "$4"
}
# in_workdir UNIT PATH: PATH, made absolute against the unit's WorkingDirectory when relative.
in_workdir() {
    [[ "$2" == /* ]] && { echo "$2"; return; }
    local wd
    wd=$(systemctl show -p WorkingDirectory --value "$1.service" 2>/dev/null)
    echo "${wd:-/}/$2"
}
# metrics_db UNIT / metrics_backend UNIT: the metrics server's SQLite database and storage backend
# (Config.hpp's defaults: training_sessions/metrics.db, sqlite+file).
metrics_db() { in_workdir "$1" "$(setting "$1" --db-path METRICS_DB_PATH training_sessions/metrics.db)"; }
metrics_backend() { setting "$1" --storage-backend METRICS_STORAGE_BACKEND sqlite+file; }
# pkg_has_postgres: whether this package's metrics_api_server was built with PostgreSQL (links libpq).
pkg_has_postgres() { ldd "$PKG_BIN/metrics_api_server" 2>/dev/null | grep -c libpq >/dev/null; }

# unit_port UNIT DEFAULT: its --port, else DEFAULT.
unit_port() {
    local p
    p=$(flag_value "$(exec_line "$1")" --port || true)
    echo "${p:-$2}"
}
# exec_line UNIT: the unit's ExecStart as systemd parsed it ("{ path=... ; argv[]=... ; ... }").
exec_line() { systemctl show -p ExecStart --value "$1.service" 2>/dev/null | head -1; }
# exec_path EXEC_LINE: the program it runs.
exec_path() { sed -n 's/.*path=\([^ ;]*\).*/\1/p' <<<"$1" | head -1; }
# flag_value EXEC_LINE FLAG: the value after FLAG in its argv, if any.
flag_value() {
    sed -n 's/.*argv\[\]=\([^;]*\);.*/\1/p' <<<"$1" | tr ' ' '\n' | awk -v f="$2" 'p {print; exit} $0 == f {p = 1}'
}
health() { curl -fsS -m 3 "http://127.0.0.1:$1/health" 2>/dev/null; }
sha() { sudo sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# missing_libs: the shared libraries this package's binaries need that the host can't find.
missing_libs() { ldd "$PKG_BIN"/* 2>/dev/null | awk '/not found/ {print $1}' | sort -u; }

check() {
    local entry unit bin port line path data db cfg backend
    echo "Package binaries: $PKG_BIN"
    local libs
    libs=$(missing_libs)
    if [[ -n "$libs" ]]; then
        echo "MISSING shared libraries (install them before 'install'):"
        sed 's/^/  /' <<<"$libs"
    else
        echo "Shared libraries: all found"
    fi
    local service units u
    for entry in "${SERVICES[@]}"; do
        read -r service bin port units <<<"$entry"
        echo
        unit=$(pick_unit "$entry")
        if [[ -z "$unit" ]]; then
            echo "$service: no unit installed ($units)"
            continue
        fi
        for u in $units; do
            [[ "$u" == "$unit" ]] && continue
            unit_installed "$u" && echo "$service: $u is installed too ($(systemctl is-active "$u.service" 2>/dev/null)); left alone"
        done
        line=$(exec_line "$unit")
        path=$(exec_path "$line")
        port=$(unit_port "$unit" "$port")
        echo "$service: $unit, $(systemctl is-active "$unit.service" 2>/dev/null), $(systemctl is-enabled "$unit.service" 2>/dev/null)"
        echo "  runs: $path ($(sudo stat -c '%y' "$path" 2>/dev/null | cut -d. -f1))"
        if [[ "$(sha "$path")" == "$(sha "$PKG_BIN/$bin")" ]]; then
            echo "  already this package's $bin"
        fi
        echo "  port $port: /health $(health "$port" >/dev/null && echo ok || echo 'not answering')"
        data=$(flag_value "$line" --data-dir || true)
        cfg=$(flag_value "$line" --config || true)
        [[ -n "$cfg" ]] && echo "  config: $cfg"
        [[ -n "$data" ]] && echo "  data: $data ($(sudo du -sh "$data" 2>/dev/null | cut -f1))"
        if [[ "$service" == metrics ]]; then
            db=$(metrics_db "$unit")
            backend=$(metrics_backend "$unit")
            echo "  storage backend: $backend"
            if [[ "$backend" == *sqlite* ]]; then
                echo "  database: $db ($(sudo du -sh "$db" 2>/dev/null | cut -f1 || true))"
            fi
            if [[ "$backend" == *postgres* ]] && ! pkg_has_postgres; then
                echo "  PROBLEM: this package's metrics_api_server has no PostgreSQL support; install refuses"
            fi
        fi
        echo "  argv: $(sed -n 's/.*argv\[\]=\([^;]*\);.*/\1/p' <<<"$line")"
    done
    echo
    if sudo test -d "$BACKUP_ROOT"; then
        echo "Backups in $BACKUP_ROOT: $(sudo ls "$BACKUP_ROOT" | tr '\n' ' ')"
    else
        echo "Backups: none yet"
    fi
}

# wait_healthy PORT: up to 30 s for /health.
wait_healthy() {
    local i
    for i in $(seq 30); do
        health "$1" >/dev/null && return 0
        sleep 1
    done
    return 1
}

# start_units UNITS_FILE: starts, in SERVICES order, the units UNITS_FILE ("service unit binary port
# active|inactive" lines) says were running, and waits for each one's /health. A unit that wasn't running
# stays stopped.
start_units() {
    local service unit bin port state failed=0
    while read -r service unit bin port state; do
        if [[ "$state" != active ]]; then
            echo "Left stopped: $unit (it wasn't running)"
            continue
        fi
        sudo systemctl start "$unit.service"
        if wait_healthy "$port"; then
            echo "OK: $unit answers on port $port"
        else
            echo "FAILED: $unit doesn't answer /health on port $port; its log:"
            sudo journalctl -u "$unit.service" -n 25 --no-pager
            failed=1
        fi
    done < <(sudo cat "$1")
    return $failed
}

# stop_units UNITS_FILE: stops its units, in reverse order.
stop_units() {
    local service unit rest
    while read -r service unit rest; do
        sudo systemctl stop "$unit.service"
    done < <(sudo tac "$1")
}

# backup_registry_queue DATA_DIR DEST: the registry's queue and registry files (.txt, .json, .db), not
# the datasets it serves.
backup_registry_queue() {
    sudo mkdir -p "$2"
    (cd "$1" && sudo find . -type f \( -name '*.txt' -o -name '*.json' -o -name '*.db' \) -print0) |
        sudo tar -C "$1" --null -T - -cf - | sudo tar -C "$2" -xf -
}

install() {
    local entry service units unit bin port line path data db cfg libs stamp dir clidir chosen=()
    for entry in "${SERVICES[@]}"; do
        unit=$(pick_unit "$entry")
        [[ -n "$unit" ]] && chosen+=("$entry|$unit")
    done
    if [[ ${#chosen[@]} -eq 0 ]]; then
        echo "No adai-mns, adai-registry or adai-metrics(-api) unit on this host: for a fresh install, run"
        echo "  sudo scripts/install_server_bundle.sh --build-dir . --yes"
        exit 1
    fi
    libs=$(missing_libs)
    if [[ -n "$libs" ]]; then
        echo "FATAL: this host lacks shared libraries the new binaries need:"
        sed 's/^/  /' <<<"$libs"
        exit 1
    fi
    for entry in "${chosen[@]}"; do
        unit=${entry##*|}
        if [[ "${entry%% *}" == metrics && "$(metrics_backend "$unit")" == *postgres* ]] && ! pkg_has_postgres; then
            echo "FATAL: $unit uses PostgreSQL ($(metrics_backend "$unit")), and this package's metrics_api_server"
            echo "was built without it. Rebuild with ENABLE_POSTGRES_METRICS=ON."
            exit 1
        fi
    done
    echo "Will replace, keeping each unit, its config and its data:"
    for entry in "${chosen[@]}"; do
        unit=${entry##*|}
        read -r service bin port units <<<"${entry%|*}"
        echo "  $unit ($(systemctl is-active "$unit.service" 2>/dev/null)): $(exec_path "$(exec_line "$unit")") <- $PKG_BIN/$bin"
    done
    ask "Stop the services, back up, replace and restart?" || {
        echo "Nothing changed."
        exit 0
    }

    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    # Never reuse a backup: a second install in the same second would overwrite the first's old binaries.
    while sudo test -e "$BACKUP_ROOT/$stamp"; do
        sleep 1
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
    done
    dir="$BACKUP_ROOT/$stamp"
    sudo mkdir -p "$dir/bin" "$dir/units" "$dir/data" "$dir/configs"
    for entry in "${chosen[@]}"; do
        unit=${entry##*|}
        read -r service bin port units <<<"${entry%|*}"
        echo "$service $unit $bin $(unit_port "$unit" "$port") $(unit_active "$unit" && echo active || echo inactive)"
    done | sudo tee "$dir/units.txt" >/dev/null
    stop_units "$dir/units.txt"
    while read -r service unit bin port _; do
        line=$(exec_line "$unit")
        path=$(exec_path "$line")
        sudo test -f "$path" || { echo "FATAL: $unit runs $path, which isn't there"; start_units "$dir/units.txt"; exit 1; }
        sudo cp -a "$path" "$dir/bin/$bin"
        echo "$bin $path" | sudo tee -a "$dir/binaries.txt" >/dev/null
        systemctl cat "$unit.service" | sudo tee "$dir/units/$unit.service.txt" >/dev/null
        cfg=$(flag_value "$line" --config || true)
        [[ -n "$cfg" ]] && sudo test -f "$cfg" && sudo cp -a "$cfg" "$dir/configs/$unit.conf"
        data=$(flag_value "$line" --data-dir || true)
        db=""
        [[ "$service" == metrics && "$(metrics_backend "$unit")" == *sqlite* ]] && db=$(metrics_db "$unit")
        case "$service" in
            mns) [[ -n "$data" ]] && sudo cp -a "$data" "$dir/data/mns" && echo "mns $data" | sudo tee -a "$dir/data.txt" >/dev/null ;;
            registry) [[ -n "$data" ]] && backup_registry_queue "$data" "$dir/data/registry" && echo "registry $data" | sudo tee -a "$dir/data.txt" >/dev/null ;;
            metrics) [[ -n "$db" ]] && sudo test -f "$db" && sudo cp -a "$db" "$dir/data/metrics.db" && echo "metrics $db" | sudo tee -a "$dir/data.txt" >/dev/null ;;
        esac
    done < <(sudo cat "$dir/units.txt")
    clidir=$(dirname "$(sudo head -1 "$dir/binaries.txt" | cut -d' ' -f2)")
    for bin in "${CLIS[@]}"; do
        [[ -f "$PKG_BIN/$bin" ]] || continue
        if sudo test -f "$clidir/$bin"; then
            sudo cp -a "$clidir/$bin" "$dir/bin/$bin"
        fi
        echo "$bin $clidir/$bin" | sudo tee -a "$dir/binaries.txt" >/dev/null
    done
    for conf in $CONF_DIRS; do
        sudo test -d "$conf" && sudo cp -a "$conf" "$dir/$(basename "$conf")-dir"
    done
    echo "Backed up to $dir"

    local owner
    while read -r bin path; do
        owner=$(sudo stat -c '%U:%G' "$path" 2>/dev/null || sudo stat -c '%U:%G' "$(dirname "$path")")
        sudo install -m 0755 -o "${owner%%:*}" -g "${owner##*:}" "$PKG_BIN/$bin" "$path" &&
            echo "Installed $path" || { echo "FATAL: could not install $path; ./scripts/redeploy_server_bundle.sh rollback $stamp"; exit 1; }
    done < <(sudo cat "$dir/binaries.txt")

    if start_units "$dir/units.txt"; then
        echo
        echo "Done. The previous binaries are in $dir; ./scripts/redeploy_server_bundle.sh rollback $stamp puts them back."
    else
        echo
        echo "A service didn't come up. ./scripts/redeploy_server_bundle.sh rollback $stamp puts the previous binaries back"
        echo "(add --with-data for the data as it was too)."
        exit 1
    fi
}

rollback() {
    local stamp=$STAMP_ARG dir bin path what target
    if [[ -z "$stamp" ]]; then
        stamp=$(sudo ls "$BACKUP_ROOT" 2>/dev/null | sort | tail -1)
    fi
    dir="$BACKUP_ROOT/$stamp"
    [[ -n "$stamp" ]] && sudo test -f "$dir/binaries.txt" && sudo test -f "$dir/units.txt" || {
        echo "No backup ${stamp:+$stamp }in $BACKUP_ROOT."
        exit 1
    }
    echo "Will put back the binaries from $dir$([[ $WITH_DATA == 1 ]] && echo ', and the data as it was')."
    ask "Stop the services and roll back?" || exit 0
    stop_units "$dir/units.txt"
    while read -r bin path; do
        if sudo test -f "$dir/bin/$bin"; then
            sudo cp -a "$dir/bin/$bin" "$path" && echo "Restored $path"
        fi
    done < <(sudo cat "$dir/binaries.txt")
    if [[ $WITH_DATA == 1 ]] && sudo test -f "$dir/data.txt"; then
        local now
        now=$(date -u +%Y%m%dT%H%M%SZ)
        while read -r what target; do
            case "$what" in
                mns) sudo mv -- "$target" "$target.rollback-$now" && sudo cp -a "$dir/data/mns" "$target" &&
                    echo "Restored $target (the newer one is $target.rollback-$now)" ;;
                registry) sudo tar -C "$dir/data/registry" -cf - . | sudo tar -C "$target" -xf - && echo "Restored the queue files in $target" ;;
                metrics) sudo mv -- "$target" "$target.rollback-$now" && sudo cp -a "$dir/data/metrics.db" "$target" &&
                    echo "Restored $target (the newer one is $target.rollback-$now)" ;;
            esac
        done < <(sudo cat "$dir/data.txt")
    fi
    start_units "$dir/units.txt"
}

case "$COMMAND" in
    check) check ;;
    install) install ;;
    rollback) rollback ;;
esac
