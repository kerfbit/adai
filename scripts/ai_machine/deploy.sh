#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.1.0
# @adai-reviewed: 2026-11-17

# Deploys the SYCL trainer and chatbot server on ai-machine from a package-sycl.sh --system-oneapi
# package (TD-575): incremental_trainer, trainer_service, chatbot_api_server, vocab_builder and
# dataset_manager into /opt/adai/bin; config.trainer.conf, config.chatbot.conf and the two trainer
# instances' env files into /etc/adai; the adai-trainer@.service template and adai.service. The oneAPI
# runtime is Intel's apt package (intel-oneapi-runtime-*, /opt/intel/oneapi/redist/lib) at the release
# the package was built with: the old bundled /opt/adai/lib is taken off the library path (kept on
# disk). Every unit is left stopped and disabled. Runs from the unpacked package, using sudo.
#
#   ./deploy.sh [check] | install [--yes] | rollback [STAMP] [--yes]
#
# check     (the default) changes nothing: the installed binaries, the adai units, which oneAPI the
#           library path resolves (ldconfig), the apt runtime's release against the package's, the
#           libraries the new binaries would miss, the configs and the vocabulary.
# install   backs up /opt/adai/bin, /etc/adai, /opt/adai/config, the adai units and the oneAPI
#           ld.so.conf.d files into /var/backups/adai-ai-machine/STAMP/, stops every adai unit, installs
#           the binaries, configs and units (the old single-instance adai-trainer.service goes to the
#           backup), switches the library path to the apt runtime and checks it. Asks first.
# rollback  puts STAMP's backup back (default the newest): binaries, configs, units, the library path,
#           and starts the units that were running before that install.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG_BIN="$HERE/bin"
BACKUP_ROOT=${ADAI_DEPLOY_BACKUP_ROOT:-/var/backups/adai-ai-machine}
ROOT=${ADAI_DEPLOY_ROOT:-}  # tests only: a scratch prefix for /opt, /etc and /var paths
OPT="$ROOT/opt/adai"
ETC="$ROOT/etc/adai"
UNIT_DIR="$ROOT/etc/systemd/system"
LDCONF_DIR="$ROOT/etc/ld.so.conf.d"
REDIST=${ADAI_ONEAPI_REDIST:-/opt/intel/oneapi/redist/lib}
BINARIES=(incremental_trainer trainer_service chatbot_api_server vocab_builder dataset_manager)
UNITS=(adai adai-trainer adai-trainer@lejepa adai-trainer@chatbot)
RUNTIME_PACKAGE=intel-oneapi-runtime-dpcpp-cpp

COMMAND=check
YES=0
STAMP_ARG=""
usage() { sed -n '/^#   \.\/deploy\.sh/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        check | install | rollback) COMMAND=$1 ;;
        --yes) YES=1 ;;
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

# The oneAPI release the package was built with (MANIFEST.txt's "oneapi:" line).
wanted_oneapi() { sed -n 's/^oneapi: *//p' "$HERE/MANIFEST.txt" 2>/dev/null | head -1; }
# The apt runtime's release, e.g. 2026.2.0 from 2026.2.0-863.
apt_oneapi() { dpkg-query -W -f='${Version}' "$RUNTIME_PACKAGE" 2>/dev/null | sed 's/-.*//'; }
# Libraries the package's binaries can't find with the apt runtime on the path.
missing_libs() { LD_LIBRARY_PATH="$REDIST" ldd "$PKG_BIN"/* 2>/dev/null | awk '/not found/ {print $1}' | sort -u; }
# squash PATH...: each path with repeated slashes collapsed (Intel's apt file names /opt/intel//oneapi/redist//lib).
squash() { sed 's#//*#/#g'; }
# Where the library path resolves libsycl.so.9 (one line per provider, slashes collapsed).
sycl_providers() { ldconfig -p 2>/dev/null | awk '/libsycl\.so\.9 / {print $NF}' | squash | sort -u; }
unit_state() { echo "$(systemctl is-active "$1.service" 2>/dev/null), $(systemctl is-enabled "$1.service" 2>/dev/null)"; }

preflight() {
    local want have libs ok=0
    want=$(wanted_oneapi)
    have=$(apt_oneapi)
    if [[ -z "$want" ]]; then
        echo "PROBLEM: the package's MANIFEST.txt names no oneAPI release (package-sycl.sh --system-oneapi writes it)"
        ok=1
    elif [[ "$have" != "$want" ]]; then
        echo "PROBLEM: the apt oneAPI runtime ($RUNTIME_PACKAGE) is ${have:-not installed}; the package was built with $want."
        echo "         The GPU test package's prepare_host.sh installs the matching runtime from Intel's repository."
        ok=1
    else
        echo "oneAPI runtime: $have from apt, as built"
    fi
    libs=$(missing_libs)
    if [[ -n "$libs" ]]; then
        echo "PROBLEM: with $REDIST on the path, the new binaries still miss:"
        sed 's/^/  /' <<<"$libs"
        ok=1
    else
        echo "Libraries: all found with $REDIST"
    fi
    return $ok
}

check() {
    local b u f
    echo "Package: $(head -1 "$HERE/MANIFEST.txt" 2>/dev/null)"
    preflight
    echo
    echo "Installed in $OPT/bin:"
    for b in "${BINARIES[@]}"; do
        if sudo test -f "$OPT/bin/$b"; then
            echo "  $b ($(sudo stat -c '%y' "$OPT/bin/$b" | cut -d. -f1))$(cmp -s "$OPT/bin/$b" "$PKG_BIN/$b" && echo ', this package')"
        else
            echo "  $b: none"
        fi
    done
    echo
    for u in "${UNITS[@]}"; do
        echo "$u: $(unit_state "$u")"
    done
    echo
    echo "libsycl.so.9 resolves from: $(sycl_providers | tr '\n' ' ')"
    for f in "$LDCONF_DIR"/adai-oneapi.conf "$LDCONF_DIR"/intel-oneapi*; do
        [[ -e "$f" ]] && echo "  $f: $(tr '\n' ' ' <"$f")"
    done
    echo
    # The legacy single config the old units read, and the per-service ones this package installs.
    for f in "$OPT/config/config.conf" "$ETC/config.conf" "$ETC/config.trainer.conf" "$ETC/config.chatbot.conf"; do
        if sudo test -f "$f"; then
            echo "$f: MODEL_NAME=$(sudo sed -n 's/^MODEL_NAME=//p' "$f" | tail -1), VOCAB_PATH=$(sudo sed -n 's/^VOCAB_PATH=//p' "$f" | tail -1)"
        else
            echo "$f: none"
        fi
    done
    f=$(sed -n 's/^VOCAB_PATH=//p' "$HERE/etc/config.trainer.conf" | tail -1)
    echo "Vocabulary $ROOT$f: $(sudo test -s "$ROOT$f" && echo present || echo 'not there yet')"
    echo
    if sudo test -d "$BACKUP_ROOT"; then
        echo "Backups in $BACKUP_ROOT: $(sudo ls "$BACKUP_ROOT" | tr '\n' ' ')"
    else
        echo "Backups: none yet"
    fi
}

# switch_library_path: the apt runtime on the library path, /opt/adai/lib off it.
switch_library_path() {
    local disabled
    if [[ -e "$LDCONF_DIR/adai-oneapi.conf" ]]; then
        sudo mv -- "$LDCONF_DIR/adai-oneapi.conf" "$1/ldconf/adai-oneapi.conf.removed"
        echo "Took /opt/adai/lib off the library path ($LDCONF_DIR/adai-oneapi.conf is in the backup)"
    fi
    if [[ ! -e "$LDCONF_DIR/intel-oneapi-runtime.conf" ]]; then
        # The apt package's own file, set aside in August 2026; else a file naming the runtime directory.
        disabled=$(ls -1 "$LDCONF_DIR"/intel-oneapi-runtime.conf.disabled-* 2>/dev/null | sort | tail -1)
        if [[ -n "$disabled" ]]; then
            sudo cp -a -- "$disabled" "$LDCONF_DIR/intel-oneapi-runtime.conf"
            echo "Put the apt runtime back on the library path (from $disabled)"
        else
            echo "$REDIST" | sudo tee "$LDCONF_DIR/intel-oneapi-runtime.conf" >/dev/null
            echo "Put $REDIST on the library path"
        fi
        echo "intel-oneapi-runtime.conf" | sudo tee -a "$1/ldconf/added.txt" >/dev/null
    fi
    sudo ldconfig
}

install() {
    local stamp dir u b f owner
    if ! preflight; then
        echo "FATAL: fix the above first; nothing changed."
        exit 1
    fi
    echo
    echo "Will stop every adai unit (the running chatbot server too), back up, then install:"
    echo "  $OPT/bin: ${BINARIES[*]}"
    echo "  $ETC: config.trainer.conf, config.chatbot.conf, trainer-lejepa.env, trainer-chatbot.env"
    echo "  $UNIT_DIR: adai-trainer@.service, adai.service (the old adai-trainer.service goes to the backup)"
    echo "  the library path: $REDIST instead of /opt/adai/lib"
    echo "All units are left stopped and disabled."
    ask "Go ahead?" || {
        echo "Nothing changed."
        exit 0
    }

    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    while sudo test -e "$BACKUP_ROOT/$stamp"; do
        sleep 1
        stamp=$(date -u +%Y%m%dT%H%M%SZ)
    done
    dir="$BACKUP_ROOT/$stamp"
    sudo mkdir -p "$dir/units" "$dir/ldconf"
    for u in "${UNITS[@]}"; do
        echo "$u $(systemctl is-active "$u.service" 2>/dev/null || true) $(systemctl is-enabled "$u.service" 2>/dev/null || true)"
    done | sudo tee "$dir/states.txt" >/dev/null
    for u in "${UNITS[@]}"; do
        systemctl is-active --quiet "$u.service" 2>/dev/null && sudo systemctl stop "$u.service"
    done
    sudo test -d "$OPT/bin" && sudo cp -a "$OPT/bin" "$dir/bin"
    sudo test -d "$ETC" && sudo cp -a "$ETC" "$dir/etc-adai"
    sudo test -d "$OPT/config" && sudo cp -a "$OPT/config" "$dir/opt-adai-config"
    for f in "$UNIT_DIR"/adai.service "$UNIT_DIR"/adai.service.d "$UNIT_DIR"/adai-trainer.service \
        "$UNIT_DIR"/adai-trainer.service.d "$UNIT_DIR"/adai-trainer@.service; do
        sudo test -e "$f" && sudo cp -a "$f" "$dir/units/"
    done
    for f in "$LDCONF_DIR"/adai-oneapi.conf "$LDCONF_DIR"/intel-oneapi*; do
        [[ -e "$f" ]] && sudo cp -a "$f" "$dir/ldconf/"
    done
    echo "Backed up to $dir"

    owner=$(sudo stat -c '%U:%G' "$OPT" 2>/dev/null || echo adai:adai)
    sudo mkdir -p "$OPT/bin" "$OPT/vocab" "$OPT/models" "$OPT/training_sessions" "$OPT/trainer_admin/lejepa" \
        "$OPT/trainer_admin/chatbot" "$OPT/chat_sessions" "$OPT/downloads" "$OPT/tokenized_cache" "$ROOT/var/log/adai" "$ETC"
    sudo chown "$owner" "$OPT/vocab" "$OPT/models" "$OPT/training_sessions" "$OPT/trainer_admin" "$OPT/trainer_admin/lejepa" \
        "$OPT/trainer_admin/chatbot" "$OPT/chat_sessions" "$OPT/downloads" "$OPT/tokenized_cache" "$ROOT/var/log/adai"
    for b in "${BINARIES[@]}"; do
        sudo install -m 0755 -o "${owner%%:*}" -g "${owner##*:}" "$PKG_BIN/$b" "$OPT/bin/$b" &&
            echo "Installed $OPT/bin/$b" || { echo "FATAL: could not install $b; ./deploy.sh rollback $stamp"; exit 1; }
    done
    for f in config.trainer.conf config.chatbot.conf trainer-lejepa.env trainer-chatbot.env; do
        sudo install -m 0640 -o root -g "${owner##*:}" "$HERE/etc/$f" "$ETC/$f" && echo "Installed $ETC/$f"
    done

    for u in adai adai-trainer; do
        systemctl is-enabled --quiet "$u.service" 2>/dev/null && sudo systemctl disable "$u.service" 2>/dev/null
    done
    if sudo test -e "$UNIT_DIR/adai-trainer.service"; then
        sudo rm -f -- "$UNIT_DIR/adai-trainer.service"
        echo "Removed the old $UNIT_DIR/adai-trainer.service (in the backup)"
    fi
    sudo install -m 0644 "$HERE/units/adai-trainer@.service" "$UNIT_DIR/adai-trainer@.service"
    sudo install -m 0644 "$HERE/units/adai.service" "$UNIT_DIR/adai.service"
    echo "Installed $UNIT_DIR/adai-trainer@.service and $UNIT_DIR/adai.service"
    sudo systemctl daemon-reload

    switch_library_path "$dir"
    local providers
    providers=$(sycl_providers)
    if [[ "$providers" != "$(squash <<<"$REDIST/libsycl.so.9")" ]]; then
        echo "FATAL: libsycl.so.9 now resolves from: $(tr '\n' ' ' <<<"$providers")(expected $REDIST only)."
        echo "./deploy.sh rollback $stamp puts everything back."
        exit 1
    fi
    if ldd "$OPT/bin/incremental_trainer" "$OPT/bin/chatbot_api_server" | grep -q 'not found'; then
        echo "FATAL: the installed binaries miss libraries:"
        ldd "$OPT/bin/incremental_trainer" "$OPT/bin/chatbot_api_server" | grep 'not found' | sort -u
        echo "./deploy.sh rollback $stamp puts everything back."
        exit 1
    fi
    echo "libsycl.so.9 resolves from $REDIST only; every library found."
    echo
    echo "Done; every adai unit is stopped and disabled. ./deploy.sh rollback $stamp undoes this install."
    echo "Next: put the vocabulary at $(sed -n 's/^VOCAB_PATH=//p' "$HERE/etc/config.trainer.conf" | tail -1),"
    echo "then: sudo systemctl enable --now adai-trainer@lejepa"
}

rollback() {
    local stamp=$STAMP_ARG dir u active enabled f
    [[ -n "$stamp" ]] || stamp=$(sudo ls "$BACKUP_ROOT" 2>/dev/null | sort | tail -1)
    dir="$BACKUP_ROOT/$stamp"
    [[ -n "$stamp" ]] && sudo test -f "$dir/states.txt" || {
        echo "No backup ${stamp:+$stamp }in $BACKUP_ROOT."
        exit 1
    }
    echo "Will put back $dir: binaries, configs, units, the library path; then start what was running."
    ask "Stop the adai units and roll back?" || exit 0
    for u in "${UNITS[@]}"; do
        systemctl is-active --quiet "$u.service" 2>/dev/null && sudo systemctl stop "$u.service"
        systemctl is-enabled --quiet "$u.service" 2>/dev/null && sudo systemctl disable "$u.service" 2>/dev/null
    done
    if sudo test -d "$dir/bin"; then
        sudo cp -a "$dir/bin/." "$OPT/bin/" && echo "Restored $OPT/bin"
    fi
    if sudo test -d "$dir/etc-adai"; then
        sudo cp -a "$dir/etc-adai/." "$ETC/" && echo "Restored $ETC"
    fi
    if sudo test -d "$dir/opt-adai-config"; then
        sudo cp -a "$dir/opt-adai-config/." "$OPT/config/" && echo "Restored $OPT/config"
    fi
    sudo test -e "$dir/units/adai-trainer@.service" || sudo rm -f -- "$UNIT_DIR/adai-trainer@.service"
    for f in adai.service adai.service.d adai-trainer.service adai-trainer.service.d adai-trainer@.service; do
        sudo test -e "$dir/units/$f" && sudo cp -a "$dir/units/$f" "$UNIT_DIR/" && echo "Restored $UNIT_DIR/$f"
    done
    sudo systemctl daemon-reload
    if sudo test -f "$dir/ldconf/added.txt"; then
        while read -r f; do
            sudo rm -f -- "$LDCONF_DIR/$f"
        done < <(sudo cat "$dir/ldconf/added.txt")
    fi
    if sudo test -f "$dir/ldconf/adai-oneapi.conf.removed"; then
        sudo cp -a "$dir/ldconf/adai-oneapi.conf.removed" "$LDCONF_DIR/adai-oneapi.conf"
    fi
    sudo ldconfig
    echo "libsycl.so.9 resolves from: $(sycl_providers | tr '\n' ' ')"
    while read -r u active enabled; do
        if [[ "$enabled" == enabled ]]; then
            sudo systemctl enable "$u.service" 2>/dev/null
        fi
        if [[ "$active" == active ]]; then
            sudo systemctl start "$u.service" && echo "Started $u"
        fi
    done < <(sudo cat "$dir/states.txt")
    return 0
}

case "$COMMAND" in
    check) check ;;
    install) install ;;
    rollback) rollback ;;
esac
