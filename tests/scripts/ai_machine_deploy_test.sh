#!/bin/bash
#
# Tests for scripts/ai_machine/deploy.sh (TD-575): the SYCL trainer and chatbot server installed on
# ai-machine with Intel's apt oneAPI runtime on the library path instead of the bundled /opt/adai/lib, and
# rolled back. Runs in a scratch root (ADAI_DEPLOY_ROOT) with stand-in sudo, systemctl, ldconfig, ldd and
# dpkg-query on PATH; nothing touches the real system.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="${REPO_ROOT}/scripts/ai_machine/deploy.sh"
source "${SCRIPT_DIR}/harness.sh"

t "syntax check"
run bash -n "$TARGET"
assert_exit 0

t "--help prints the usage; an unknown argument is an error"
run bash "$TARGET" --help
assert_exit 0
assert_contains "./deploy.sh [check] | install [--yes] | rollback [STAMP] [--yes]"
run bash "$TARGET" --bogus
assert_exit 2

# A scratch ai-machine as it was on November 18, 2026: the September trainer and July chatbot server,
# /opt/adai/lib on the library path, the apt runtime's own file set aside, the old adai-trainer.service.
host="$(mktemp_dir)"
root="$host/root"
pkg="$host/pkg"
redist="$host/redist"
mkdir -p "$root/opt/adai/bin" "$root/opt/adai/config" "$root/etc/adai" "$root/etc/systemd/system/adai-trainer.service.d" \
    "$root/etc/ld.so.conf.d" "$root/var/log" "$pkg/bin" "$pkg/etc" "$pkg/units" "$redist" "$host/stub" "$host/state"
for b in incremental_trainer chatbot_api_server; do echo "old $b" >"$root/opt/adai/bin/$b"; done
echo "VOCAB_PATH=/home/rodney/Repos/adai/vocab.txt" >"$root/opt/adai/config/config.conf"
echo "KEY=old" >"$root/etc/adai/config.conf"
echo "[Service] old trainer" >"$root/etc/systemd/system/adai-trainer.service"
echo "LimitCORE=infinity" >"$root/etc/systemd/system/adai-trainer.service.d/50-adai-coredump.conf"
echo "[Service] old chatbot" >"$root/etc/systemd/system/adai.service"
echo "/opt/adai/lib" >"$root/etc/ld.so.conf.d/adai-oneapi.conf"
# Intel's apt file writes its directories with doubled slashes (/opt/intel//oneapi/redist//lib).
printf '%s\n%s\n' "${redist%/*}//${redist##*/}" "${redist}32" >"$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf.disabled-20260809T230000Z"
cp "$TARGET" "$pkg/deploy.sh"
for b in incremental_trainer trainer_service chatbot_api_server vocab_builder dataset_manager; do echo "new $b" >"$pkg/bin/$b"; done
printf 'MODEL_NAME=bodacious-badger\nVOCAB_PATH=/opt/adai/vocab/bodacious-badger.vocab\n' >"$pkg/etc/config.trainer.conf"
printf 'MODEL_NAME=bodacious-badger\nVOCAB_PATH=/opt/adai/vocab/bodacious-badger.vocab\n' >"$pkg/etc/config.chatbot.conf"
echo "TRAINER_ADMIN_PORT=8086" >"$pkg/etc/trainer-lejepa.env"
echo "TRAINER_ADMIN_PORT=8084" >"$pkg/etc/trainer-chatbot.env"
echo "[Service] new template" >"$pkg/units/adai-trainer@.service"
echo "[Service] new chatbot" >"$pkg/units/adai.service"
printf 'adai-sycl-ai-machine-test\noneapi: 2026.2.0\n' >"$pkg/MANIFEST.txt"
echo active >"$host/state/adai.active"
echo enabled >"$host/state/adai.enabled"
echo disabled >"$host/state/adai-trainer.enabled"

cat >"$host/stub/systemctl" <<EOF
#!/bin/bash
# Stand-in systemctl: unit states in $host/state/<unit>.{active,enabled}; actions logged.
quiet=0; args=()
for a in "\$@"; do [[ \$a == --quiet ]] && quiet=1 || args+=("\$a"); done
cmd=\${args[0]}; unit=\${args[-1]%.service}
case "\$cmd" in
    is-active) s=\$(cat "$host/state/\$unit.active" 2>/dev/null || echo inactive); ((quiet)) || echo "\$s"; [[ \$s == active ]] ;;
    is-enabled) s=\$(cat "$host/state/\$unit.enabled" 2>/dev/null || echo disabled); ((quiet)) || echo "\$s"; [[ \$s == enabled ]] ;;
    start) echo active >"$host/state/\$unit.active"; echo "start \$unit" >>"$host/systemctl.log" ;;
    stop) echo inactive >"$host/state/\$unit.active"; echo "stop \$unit" >>"$host/systemctl.log" ;;
    enable) echo enabled >"$host/state/\$unit.enabled"; echo "enable \$unit" >>"$host/systemctl.log" ;;
    disable) echo disabled >"$host/state/\$unit.enabled"; echo "disable \$unit" >>"$host/systemctl.log" ;;
    daemon-reload) echo daemon-reload >>"$host/systemctl.log" ;;
esac
EOF
# ldconfig -p: libsycl.so.9 from every directory the scratch ld.so.conf.d names (the first one, as ai-machine's
# lib32 has none), spelled as the file spells it.
cat >"$host/stub/ldconfig" <<EOF
#!/bin/bash
[[ "\${1:-}" == -p ]] || exit 0
for f in "$root/etc/ld.so.conf.d"/*.conf; do
    [[ -e \$f ]] && head -1 "\$f" | while read -r d; do printf '\tlibsycl.so.9 (libc6,x86-64) => %s/libsycl.so.9\n' "\$d"; done
done
EOF
printf '#!/bin/bash\nfor f in "$@"; do echo "$f:"; printf "\\tlibsycl.so.9 => /x/libsycl.so.9 (0x0)\\n"; [[ -n "${MISSING:-}" ]] && printf "\\t%%s => not found\\n" "$MISSING"; done\nexit 0\n' >"$host/stub/ldd"
printf '#!/bin/bash\necho "${RUNTIME_VERSION:-2026.2.0-863}"\n' >"$host/stub/dpkg-query"
printf '#!/bin/bash\nexec "$@"\n' >"$host/stub/sudo"
chmod +x "$host/stub/"*
export PATH="$host/stub:$PATH" ADAI_DEPLOY_ROOT="$root" ADAI_DEPLOY_BACKUP_ROOT="$host/backups" ADAI_ONEAPI_REDIST="$redist"

t "check reports the release, the library path, the binaries and the units, and changes nothing"
run bash "$pkg/deploy.sh"
assert_exit 0
assert_contains "oneAPI runtime: 2026.2.0 from apt, as built"
assert_contains "libsycl.so.9 resolves from: /opt/adai/lib/libsycl.so.9"
assert_contains "trainer_service: none"
assert_contains "adai: active, enabled"
assert_contains "VOCAB_PATH=/home/rodney/Repos/adai/vocab.txt"
assert_contains "Backups: none yet"

t "install refuses an apt runtime at another release, or missing libraries, and changes nothing"
run env RUNTIME_VERSION=2026.1.1-325 bash "$pkg/deploy.sh" install --yes
assert_exit 1
assert_contains "the apt oneAPI runtime (intel-oneapi-runtime-dpcpp-cpp) is 2026.1.1; the package was built with 2026.2.0"
run env MISSING=libmkl_sycl_blas.so.5 bash "$pkg/deploy.sh" install --yes
assert_exit 1
assert_contains "libmkl_sycl_blas.so.5"
run cat "$root/opt/adai/bin/incremental_trainer"
assert_contains "old incremental_trainer"
assert_file_not_exists "$host/systemctl.log"

t "install stops the units, backs up, installs, switches the library path to the apt runtime, starts nothing"
run bash "$pkg/deploy.sh" install --yes
assert_exit 0
assert_contains "libsycl.so.9 resolves from $redist only"
for b in incremental_trainer trainer_service chatbot_api_server vocab_builder dataset_manager; do
    run cat "$root/opt/adai/bin/$b"
    assert_contains "new $b"
done
run cat "$root/etc/adai/config.trainer.conf"
assert_contains "MODEL_NAME=bodacious-badger"
assert_file_exists "$root/etc/adai/trainer-lejepa.env"
assert_file_exists "$root/etc/systemd/system/adai-trainer@.service"
assert_file_not_exists "$root/etc/systemd/system/adai-trainer.service"
run cat "$root/etc/systemd/system/adai.service"
assert_contains "new chatbot"
assert_file_not_exists "$root/etc/ld.so.conf.d/adai-oneapi.conf"
run cat "$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf"
assert_contains "${redist%/*}//${redist##*/}"
assert_file_exists "$root/opt/adai/vocab"
stamp=$(ls "$host/backups")
run cat "$host/backups/$stamp/bin/incremental_trainer"
assert_contains "old incremental_trainer"
assert_file_exists "$host/backups/$stamp/units/adai-trainer.service"
assert_file_exists "$host/backups/$stamp/units/adai-trainer.service.d/50-adai-coredump.conf"
assert_file_exists "$host/backups/$stamp/opt-adai-config/config.conf"
assert_file_exists "$host/backups/$stamp/ldconf/adai-oneapi.conf.removed"
run cat "$host/systemctl.log"
assert_contains "stop adai"
assert_contains "disable adai"
assert_not_contains "start"
run cat "$host/state/adai.active"
assert_contains "inactive"

t "rollback puts the binaries, units and library path back and starts what was running"
: >"$host/systemctl.log"
run bash "$pkg/deploy.sh" rollback "$stamp" --yes
assert_exit 0
assert_contains "libsycl.so.9 resolves from: /opt/adai/lib/libsycl.so.9"
assert_contains "Started adai"
run cat "$root/opt/adai/bin/incremental_trainer"
assert_contains "old incremental_trainer"
run cat "$root/etc/systemd/system/adai.service"
assert_contains "old chatbot"
assert_file_exists "$root/etc/systemd/system/adai-trainer.service"
assert_file_not_exists "$root/etc/systemd/system/adai-trainer@.service"
assert_file_not_exists "$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf"
assert_file_exists "$root/etc/ld.so.conf.d/adai-oneapi.conf"
run cat "$host/state/adai.enabled"
assert_contains "enabled"

t "ai-machine on November 18: both oneAPI files active; install keeps the apt one, rollback puts both back"
cp "$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf.disabled-20260809T230000Z" "$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf"
run bash "$pkg/deploy.sh"
assert_contains "libsycl.so.9 resolves from: /opt/adai/lib/libsycl.so.9 $redist/libsycl.so.9"
run bash "$pkg/deploy.sh" install --yes
assert_exit 0
assert_contains "libsycl.so.9 resolves from $redist only"
assert_file_not_exists "$root/etc/ld.so.conf.d/adai-oneapi.conf"
assert_file_exists "$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf"
stamp2=$(ls "$host/backups" | sort | tail -1)
run bash "$pkg/deploy.sh" rollback "$stamp2" --yes
assert_exit 0
assert_file_exists "$root/etc/ld.so.conf.d/adai-oneapi.conf"
assert_file_exists "$root/etc/ld.so.conf.d/intel-oneapi-runtime.conf"

harness_report
exit $?
