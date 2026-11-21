#!/bin/bash
#
# Tests for scripts/redeploy_server_bundle.sh (TD-575): an installed server bundle's binaries replaced in
# place, with a backup and a rollback. Runs against a scratch "host" with stand-in systemctl, sudo,
# curl and journalctl on PATH; nothing touches the real system.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/redeploy_server_bundle.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help prints the usage; an unknown argument is an error"
run bash "$TARGET" --help
assert_exit 0
assert_contains "check] | install [--yes] | rollback [STAMP] [--with-data] [--yes]"
run bash "$TARGET" --bogus
assert_exit 2
assert_contains "unknown argument: --bogus"

t "it reads a unit's program and flags from systemctl's ExecStart"
run bash -c '
    eval "$(sed -n "/^exec_path() {/p; /^flag_value() {/,/^}/p" "$1")"
    line="{ path=/opt/adai/bin/registry_server ; argv[]=/opt/adai/bin/registry_server --port 8082 --data-dir /opt/adai/registry_sessions --ftp-enabled ; ignore_errors=no ; start_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
    echo "path=$(exec_path "$line") port=$(flag_value "$line" --port) data=$(flag_value "$line" --data-dir) db=[$(flag_value "$line" --db-path)]"' _ "$TARGET"
assert_exit 0
assert_contains "path=/opt/adai/bin/registry_server port=8082 data=/opt/adai/registry_sessions db=[]"

# A scratch host: a package (bin/ beside scripts/), installed "old" binaries, data, and stand-ins.
host="$(mktemp_dir)"
pkg="$host/pkg"
mkdir -p "$pkg/bin" "$pkg/scripts" "$host/opt/bin" "$host/mns" "$host/registry/aa01/chatbot" "$host/metrics" "$host/etc" \
    "$host/stub"
cp "$TARGET" "$pkg/scripts/"
for b in mns_server registry_server metrics_api_server mns_cli dataset_manager vocab_builder; do
    printf '#!/bin/sh\necho new %s\n' "$b" >"$pkg/bin/$b"
    chmod +x "$pkg/bin/$b"
done
for b in mns_server registry_server metrics_api_server mns_cli; do
    printf '#!/bin/sh\necho old %s\n' "$b" >"$host/opt/bin/$b"
    chmod +x "$host/opt/bin/$b"
done
# Like 192.168.1.16: the running metrics unit is adai-metrics-api; an old, stopped adai-metrics runs another copy.
mkdir -p "$host/old"
printf '#!/bin/sh\necho stale metrics_api_server\n' >"$host/old/metrics_api_server"
echo "models v1" >"$host/mns/models.db"
echo "pending v1" >"$host/registry/aa01/chatbot/pending_files.txt"
echo '{"big":"dataset"}' >"$host/registry/aa01/chatbot/data.jsonl"
echo "metrics v1" >"$host/metrics/metrics.db"
echo "KEY=1" >"$host/etc/config.conf"
# Like 192.168.1.16's adai-metrics-api: its settings come from --config, the database path relative to
# the unit's WorkingDirectory.
printf 'METRICS_PORT=8081\nMETRICS_STORAGE_BACKEND = sqlite+file\nMETRICS_DB_PATH=metrics/metrics.db\n' >"$host/etc/metrics.conf"
cat >"$host/stub/systemctl" <<EOF
#!/bin/bash
# Stand-in systemctl for three installed units; records start/stop in $host/systemctl.log.
unit=\${@: -1}; unit=\${unit%.service}
case "\$unit" in adai-mns|adai-registry|adai-metrics-api|adai-metrics) ;; *) exit 1 ;; esac
case "\$1" in
    cat) echo "# /etc/systemd/system/\$unit.service"; echo "ExecStart=..." ;;
    show)
        [[ "\$3" == WorkingDirectory ]] && { echo "$host"; exit 0; }
        [[ "\$3" == Environment ]] && exit 0
        case "\$unit" in
            adai-mns) echo "{ path=$host/opt/bin/mns_server ; argv[]=$host/opt/bin/mns_server --port 8083 --data-dir $host/mns ; status=0/0 }" ;;
            adai-registry) echo "{ path=$host/opt/bin/registry_server ; argv[]=$host/opt/bin/registry_server --port 8082 --data-dir $host/registry ; status=0/0 }" ;;
            adai-metrics-api) echo "{ path=$host/opt/bin/metrics_api_server ; argv[]=$host/opt/bin/metrics_api_server --config $host/etc/metrics.conf ; status=0/0 }" ;;
            adai-metrics) echo "{ path=$host/old/metrics_api_server ; argv[]=$host/old/metrics_api_server --port 8081 ; status=0/0 }" ;;
        esac ;;
    start|stop) echo "\$1 \$unit" >>"$host/systemctl.log" ;;
    is-active) [[ "\$unit" == adai-metrics ]] && { echo inactive; exit 3; }; echo active ;;
    is-enabled) echo enabled ;;
esac
EOF
printf '#!/bin/bash\nexec "$@"\n' >"$host/stub/sudo"
printf '#!/bin/bash\n[[ -e %q/unhealthy ]] && exit 7\necho ok\n' "$host" >"$host/stub/curl"
printf '#!/bin/bash\necho "(journal of $2)"\n' >"$host/stub/journalctl"
chmod +x "$host/stub/"*
export PATH="$host/stub:$PATH" ADAI_SERVER_BACKUP_ROOT="$host/backups" ADAI_SERVER_CONF_DIRS="$host/etc"

t "check reports each unit, its binary, port, health and data, and changes nothing"
run bash "$pkg/scripts/redeploy_server_bundle.sh"
assert_exit 0
assert_contains "Shared libraries: all found"
assert_contains "registry: adai-registry, active, enabled"
assert_contains "metrics: adai-metrics is installed too (inactive); left alone"
assert_contains "metrics: adai-metrics-api, active, enabled"
assert_contains "runs: $host/opt/bin/registry_server"
assert_contains "port 8082: /health ok"
assert_contains "data: $host/registry"
assert_contains "config: $host/etc/metrics.conf"
assert_contains "storage backend: sqlite+file"
assert_contains "database: $host/metrics/metrics.db"
assert_contains "Backups: none yet"
run cat "$host/opt/bin/mns_server"
assert_contains "old mns_server"

t "install stops the services, backs up binaries, units, configs and data (not datasets), replaces, restarts in order"
run bash "$pkg/scripts/redeploy_server_bundle.sh" install --yes
assert_exit 0
assert_contains "Done."
for b in mns_server registry_server metrics_api_server mns_cli dataset_manager vocab_builder; do
    run cat "$host/opt/bin/$b"
    assert_contains "new $b"
done
stamp=$(ls "$host/backups")
run cat "$host/backups/$stamp/bin/mns_server"
assert_contains "old mns_server"
assert_file_exists "$host/backups/$stamp/units/adai-registry.service.txt"
assert_file_exists "$host/backups/$stamp/etc-dir/config.conf"
assert_file_exists "$host/backups/$stamp/data/mns/models.db"
assert_file_exists "$host/backups/$stamp/data/metrics.db"
assert_file_exists "$host/backups/$stamp/configs/adai-metrics-api.conf"
assert_file_exists "$host/backups/$stamp/data/registry/aa01/chatbot/pending_files.txt"
assert_file_not_exists "$host/backups/$stamp/data/registry/aa01/chatbot/data.jsonl"
run cat "$host/systemctl.log"
assert_contains "stop adai-metrics-api
stop adai-registry
stop adai-mns
start adai-mns
start adai-registry
start adai-metrics-api"
assert_not_contains "adai-metrics
"
run cat "$host/old/metrics_api_server"
assert_contains "stale metrics_api_server"

t "a service that doesn't come up fails the install and names the rollback"
: >"$host/systemctl.log"
touch "$host/unhealthy"
run bash -c 'sed -i "s/for i in \$(seq 30)/for i in 1/" "$1"; bash "$1" install --yes' _ "$pkg/scripts/redeploy_server_bundle.sh"
assert_exit 1
assert_contains "FAILED: adai-mns doesn't answer /health on port 8083"
assert_contains "(journal of adai-mns.service)"
assert_contains "rollback"
rm -f "$host/unhealthy"

t "each install keeps its own backup, even in the same second"
run bash -c 'ls "$1" | wc -l' _ "$host/backups"
assert_contains "2"
run cat "$host/backups/$stamp/bin/registry_server"
assert_contains "old registry_server"

t "rollback STAMP --with-data puts back that backup's binaries and data, keeping the newer data aside"
echo "models v2 (migrated)" >"$host/mns/models.db"
echo "pending v2" >"$host/registry/aa01/chatbot/pending_files.txt"
run bash "$pkg/scripts/redeploy_server_bundle.sh" rollback "$stamp" --with-data --yes
assert_exit 0
run cat "$host/opt/bin/registry_server"
assert_contains "old registry_server"
run cat "$host/mns/models.db"
assert_contains "models v1"
run cat "$host/registry/aa01/chatbot/pending_files.txt"
assert_contains "pending v1"
run bash -c 'cat "$1"/mns.rollback-*/models.db' _ "$host"
assert_contains "models v2 (migrated)"
assert_file_exists "$host/registry/aa01/chatbot/data.jsonl"

t "a metrics server on PostgreSQL: check says so, and install refuses a package built without it"
cp "$host/etc/metrics.conf" "$host/metrics.conf.keep"
echo "METRICS_STORAGE_BACKEND=postgres+file" >>"$host/etc/metrics.conf"
run bash "$pkg/scripts/redeploy_server_bundle.sh"
assert_contains "storage backend: postgres+file"
assert_contains "PROBLEM: this package's metrics_api_server has no PostgreSQL support"
: >"$host/systemctl.log"
run bash "$pkg/scripts/redeploy_server_bundle.sh" install --yes
assert_exit 1
assert_contains "FATAL: adai-metrics-api uses PostgreSQL (postgres+file)"
run cat "$host/systemctl.log"
assert_not_contains "stop"
cp "$host/metrics.conf.keep" "$host/etc/metrics.conf"

t "install refuses a host with none of the three units"
run bash -c 'PATH="$2:$PATH" bash "$1" install --yes' _ "$pkg/scripts/redeploy_server_bundle.sh" "$(mktemp_dir)"
sed -i 's/adai-mns|adai-registry|adai-metrics-api|adai-metrics) ;;/none) ;;/' "$host/stub/systemctl"
run bash "$pkg/scripts/redeploy_server_bundle.sh" install --yes
assert_exit 1
assert_contains "install_server_bundle.sh --build-dir . --yes"

harness_report
exit $?
