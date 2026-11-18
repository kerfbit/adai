#!/usr/bin/env bash

# @adai-status: experimental
# @adai-version: 0.4.1
# @adai-reviewed: 2026-11-18

# Installs a current upstream kernel on the GPU host (ai-machine) from Ubuntu's mainline kernel
# builds (kernel.ubuntu.com/mainline), boots it once for a test run, and removes it again: the xe
# driver's developers ask for a result on a current kernel (TD-574's report to Intel). The stock
# kernel stays the default throughout, so if the upstream one doesn't come up, a power cycle boots
# the stock one again. With Secure Boot on, it signs the kernel with a key of its own that the
# firmware is asked to trust (a MOK). Uses sudo; asks before each change.
#
#   ./mainline_kernel.sh [check] | install [VERSION] [--yes] | boot-once [VERSION] [--yes]
#                       | remove [--yes]
#
# check      (the default) changes nothing: Secure Boot, the signing key's state, the running
#            kernel, GRUB's default, the newest mainline release, and the mainline kernels this
#            script installed.
# install    VERSION is a mainline release such as 6.18.4 (default: the newest one that isn't a
#            release candidate).
#            1. Downloads its generic linux-image-unsigned and linux-modules packages and checks
#               them against the release's CHECKSUMS file.
#            2. Installs them with apt, refusing if apt would also install, upgrade or remove
#               anything else. Packages built for a newer Ubuntu pass run-parts two hook
#               directories (/etc/kernel/X.d and /usr/share/kernel/X.d), which 24.04's run-parts
#               refuses ("missing operand"); those are rebuilt in host_state/mainline/VERSION/fitted/
#               with only /etc/kernel/X.d, the kernel and modules unchanged. Packages an earlier
#               failed attempt left half-installed are purged first (asks).
#            3. With Secure Boot on (the mainline builds are unsigned): creates a code-signing key
#               in /var/lib/adai-mainline-mok (root only) unless it exists, signs the kernel image
#               with it (sbsign; needs the sbsigntool package), and queues the key for enrollment
#               (mokutil --import, which asks for a one-time password) unless the firmware already
#               trusts it. At the next boot shim's MokManager (blue screen, 10 s to press a key)
#               asks for it: Enroll MOK, Continue, Yes, the password, Reboot. That needs someone at
#               the console, so it asks first, and names any installed Canonical-signed kernel newer
#               than the running one (those boot with no key; a headless host should use one). Missed
#               it? The boot stops at shim; power-cycle, then run boot-once again. The kernel's modules need nothing: the build signs them with a
#               key built into that kernel.
#            4. Makes GRUB boot a saved entry (GRUB_DEFAULT=saved; /etc/default/grub is kept in
#               host_state/ first), sets it to the running kernel, and arms the upstream kernel for
#               the next boot only (grub-reboot).
#            Then: sudo reboot, ./prepare_host.sh, ./run_gpu_tests.sh ..., and a reboot back.
# boot-once  arms the installed upstream kernel for the next boot again (another run); with Secure
#            Boot on, signs it and queues the key's enrollment first if either is missing.
# remove     removes the kernels install added (not while one of them is running) and puts
#            /etc/default/grub back as it was. Then, if the key exists, asks to take it off the
#            firmware's list (mokutil --delete, confirmed in MokManager at the next boot the same
#            way; or --revoke-import if it was never enrolled) and deletes it.
#
# The compute runtime, the GuC firmware and xe_debug.sh's module option stay as they are. A report of
# each run goes to host_state/.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MAINLINE=https://kernel.ubuntu.com/mainline

COMMAND=check
YES=0
VERSION=""
usage() {
    sed -n '/^#   \.\/mainline_kernel/,/^# The compute runtime/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        check | install | boot-once | remove) COMMAND=$1 ;;
        --yes) YES=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        [0-9]*)
            [[ "$1" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || {
                echo "Not a mainline release: $1 (expected e.g. 6.18.4)" >&2
                exit 2
            }
            VERSION=$1
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
DOWNLOADS="$STATE/mainline"
INSTALLED="$STATE/mainline_installed.txt"
GRUB_BACKUP="$STATE/grub.default.before_mainline"
BOOT_DIR=/boot
MOK_DIR=/var/lib/adai-mainline-mok
MOK_KEY="$MOK_DIR/MOK.priv"
MOK_PEM="$MOK_DIR/MOK.pem"
MOK_DER="$MOK_DIR/MOK.der"
mkdir -p "$STATE"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee -a "$STATE/mainline_kernel_${COMMAND}_$STAMP.txt") 2>&1

ask() {
    if [[ $YES == 1 ]]; then
        echo "$1 [y/N] y (--yes)"
        return 0
    fi
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty || return 1
    [[ "$reply" =~ ^[Yy] ]]
}

# latest_release INDEX_HTML: the newest vX.Y[.Z] directory that isn't a release candidate.
latest_release() {
    grep -o -E 'href="v[0-9]+\.[0-9]+(\.[0-9]+)?/"' | sed -E 's/^href="v//; s/\/"$//' | sort -V | tail -1
}
# package_files AMD64_INDEX_HTML: the generic image and modules .debs a release's amd64/ lists.
package_files() {
    grep -o -E 'href="linux-(image-unsigned|modules)-[^"]*-generic_[^"]*_amd64\.deb"' |
        sed -E 's/^href="//; s/"$//' | sort -u
}
# kernel_release DEB: the kernel release a linux-image-unsigned/linux-modules package installs.
kernel_release() {
    basename "$1" | sed -E 's/^linux-(image-unsigned|modules)-//; s/_.*$//'
}
# grub_entry KERNEL_RELEASE: GRUB's "<submenu>><entry>" title for that kernel, if grub.cfg has it.
grub_entry() {
    sudo cat /boot/grub/grub.cfg 2>/dev/null | grub_entry_in "$1"
}
# grub_entry_in KERNEL_RELEASE: the same, from a grub.cfg on stdin.
grub_entry_in() {
    awk -v k="$1" '
        /^submenu / { if (match($0, /\047[^\047]*\047/)) sub_title = substr($0, RSTART + 1, RLENGTH - 2) }
        /^[[:space:]]+menuentry / && index($0, "\047Ubuntu, with Linux " k "\047") {
            print sub_title ">Ubuntu, with Linux " k; exit
        }'
}
secure_boot_on() {
    command -v mokutil >/dev/null && mokutil --sb-state 2>/dev/null | grep -c -i 'SecureBoot enabled' >/dev/null
}
# new_key DIR: a code-signing key and certificate (MOK.priv, MOK.pem, MOK.der) in DIR. Code signing,
# not the module-signing-only usage of DKMS's MOK, which shim refuses for a kernel image.
new_key() {
    sudo openssl req -new -x509 -newkey rsa:2048 -nodes -sha256 -days 365 \
        -subj "/CN=adai mainline kernel test key ($(hostname))/" \
        -addext extendedKeyUsage=codeSigning -addext basicConstraints=critical,CA:FALSE \
        -addext keyUsage=digitalSignature -keyout "$1/MOK.priv" -out "$1/MOK.pem" 2>/dev/null &&
        sudo openssl x509 -in "$1/MOK.pem" -outform DER -out "$1/MOK.der"
}
# fingerprint_hex: the lowercase hex digits of a "SHA1 Fingerprint" line on stdin, separators gone.
fingerprint_hex() {
    sed -n -E 's/^.*[Ff]ingerprint[=:] *//p' | head -1 | tr -d ': ' | tr 'A-F' 'a-f'
}
# mok_state_of TEST_KEY_OUTPUT LIST_NEW_OUTPUT FINGERPRINT: enrolled, pending (queued for MokManager)
# or absent, from mokutil --test-key's and --list-new's output and the key's SHA-1 fingerprint.
mok_state_of() {
    if grep -c -i 'already enrolled' <<<"$1" >/dev/null; then
        echo enrolled
    elif [[ -n "$3" ]] && grep -i 'fingerprint' <<<"$2" | tr -d ': ' | tr 'A-F' 'a-f' | grep -c -F "$3" >/dev/null; then
        echo pending
    else
        echo absent
    fi
}
# mok_state: the state of this script's key, or none when there is no key.
mok_state() {
    sudo test -s "$MOK_DER" || {
        echo none
        return
    }
    local fp
    fp=$(sudo openssl x509 -inform DER -in "$MOK_DER" -noout -fingerprint -sha1 | fingerprint_hex)
    # MokNew is readable by root only: without sudo, --list-new sees an empty request.
    mok_state_of "$(sudo mokutil --test-key "$MOK_DER" 2>&1)" "$(sudo mokutil --list-new 2>&1)" "$fp"
}
# signing_tools: refuses (non-zero, with a message) when a tool the signing needs is missing.
signing_tools() {
    local t missing=()
    for t in mokutil openssl sbsign sbverify; do command -v "$t" >/dev/null || missing+=("$t"); done
    [[ ${#missing[@]} -eq 0 ]] && return 0
    echo "FATAL: Secure Boot is on, and signing the kernel needs ${missing[*]}: sudo apt-get install sbsigntool mokutil openssl"
    return 1
}
# sign_kernel KERNEL_RELEASE: $BOOT_DIR/vmlinuz-KERNEL_RELEASE signed with this script's key (made if
# missing), and the key queued for enrollment unless the firmware trusts it or already has it queued.
sign_kernel() {
    local img="$BOOT_DIR/vmlinuz-$1" state
    if ! sudo test -s "$MOK_KEY"; then
        echo "Creating the signing key in $MOK_DIR."
        sudo install -d -m 0700 "$MOK_DIR" && new_key "$MOK_DIR" || {
            echo "FATAL: could not create the signing key."
            return 1
        }
    fi
    if sudo sbverify --cert "$MOK_PEM" "$img" >/dev/null 2>&1; then
        echo "$img is already signed with the key."
    else
        sudo sbsign --key "$MOK_KEY" --cert "$MOK_PEM" --output "$img.adai-signed" "$img" &&
            sudo chmod --reference="$img" "$img.adai-signed" &&
            sudo sbverify --cert "$MOK_PEM" "$img.adai-signed" >/dev/null &&
            sudo mv -f -- "$img.adai-signed" "$img" || {
            sudo rm -f -- "$img.adai-signed"
            echo "FATAL: could not sign $img."
            return 1
        }
        echo "Signed $img."
    fi
    state=$(mok_state)
    [[ "$state" == enrolled ]] || console_check "$state" || return 1
    case "$state" in
        enrolled) echo "The firmware already trusts the key." ;;
        pending) echo "The key is queued for enrollment: MokManager asks at the next boot." ;;
        *)
            echo "Queueing the key for enrollment. mokutil asks for a one-time password; MokManager asks"
            echo "for it again at the next boot, at the console."
            # On the terminal itself: through this script's log pipe its password prompt never showed.
            sudo mokutil --import "$MOK_DER" </dev/tty >/dev/tty 2>&1 || {
                echo "FATAL: mokutil --import failed (rc $?); the kernel is signed but the firmware won't trust it yet."
                echo "Queue it by hand: sudo mokutil --import $MOK_DER, then ./mainline_kernel.sh boot-once."
                return 1
            }
            ;;
    esac
    [[ "$state" == enrolled ]] || mok_reminder
}
mok_reminder() {
    echo
    echo "At the next boot, at the console: shim's blue MokManager screen waits 10 s for a key press."
    echo "Press one, then Enroll MOK, Continue, Yes, the password, Reboot. If it's missed, shim refuses the"
    echo "upstream kernel and the boot stops there; a power cycle boots $(uname -r) again (the one-time"
    echo "entry is used up). Then run ./mainline_kernel.sh boot-once again."
}
# console_check MOK_STATE: before a key that isn't enrolled is queued or a kernel armed that needs it,
# says that enrolling needs someone at the console (ai-machine has none: November 17, 2026, the boot
# stopped at shim and SSH never came back), names the installed Canonical-signed kernels newer than
# the running one (they boot with Secure Boot and no key), and asks. False if declined.
console_check() {
    local k newer
    echo
    echo "Enrolling the key needs someone at this machine's console at the next boot: MokManager waits"
    echo "for a key press there, and nothing over SSH can answer it. Missed, shim refuses the upstream kernel,"
    echo "the boot stops, and SSH stays down until the machine is power-cycled."
    newer=$(canonical_kernels_newer "$(uname -r)")
    if [[ -n "$newer" ]]; then
        echo "Installed kernels newer than $(uname -r) that Canonical signed (no key needed) - to boot one once:"
        for k in $newer; do
            echo "  sudo grub-reboot \"$(grub_entry "$k")\""
        done
    fi
    ask "Is someone at the console for the next boot?" && return 0
    echo "Not armed: the next boot is $(uname -r) as before."
    if [[ "$1" == pending ]]; then
        echo "The key is still queued; sudo mokutil --revoke-import drops it."
    fi
    return 1
}
# canonical_kernels_newer RELEASE: the kernels in $BOOT_DIR newer than RELEASE (sort -V) whose image
# carries Canonical's signature.
canonical_kernels_newer() {
    local img k
    for img in "$BOOT_DIR"/vmlinuz-*; do
        [[ -e "$img" ]] || continue
        k=${img#"$BOOT_DIR"/vmlinuz-}
        [[ "$k" != "$1" && "$(printf '%s\n%s\n' "$1" "$k" | sort -V | tail -1)" == "$k" ]] || continue
        sudo sbverify --list "$img" 2>/dev/null | grep -i 'Canonical' >/dev/null && echo "$k"
    done
}
# outside_kernel: the packages an `apt-get -s` plan on stdin would touch that aren't the ones given.
outside_kernel() {
    local allowed=" $* " verb p rest
    while read -r verb p rest; do
        case "$verb" in
            Remv) echo "$p (removed)" ;;
            Inst) [[ "$allowed" == *" ${p%%:*} "* ]] || echo "$p" ;;
        esac
    done
}
# one_dir_run_parts: true when this host's run-parts takes only one directory (debianutils 5.17,
# Ubuntu 24.04). Newer mainline packages' maintainer scripts pass it two, /etc/kernel/X.d and
# /usr/share/kernel/X.d, and fail with "run-parts: missing operand".
one_dir_run_parts() {
    local d rc
    d=$(mktemp -d) || return 1
    run-parts --list "$d" "$d" >/dev/null 2>&1
    rc=$?
    rmdir "$d"
    [[ $rc -ne 0 ]]
}
# drop_share_hook_dirs FILE: removes /usr/share/kernel/X.d where a maintainer script passes it to
# run-parts after /etc/kernel/X.d, leaving the hooks this host has (it has no /usr/share/kernel).
drop_share_hook_dirs() {
    sed -i -E 's#(/etc/kernel/[a-z]+\.d)[[:space:]]+/usr/share/kernel/[a-z]+\.d#\1#g' "$1"
}
# names_share_hook_dirs FILE...: true when a maintainer script still names /usr/share/kernel/X.d
# in a run-parts call's directories.
names_share_hook_dirs() {
    grep -E '/etc/kernel/[a-z]+\.d[[:space:]]+/usr/share/kernel/[a-z]+\.d' "$@" >/dev/null 2>&1
}
# fit_debs DIR FILE...: prints the package files to install. A package whose maintainer scripts pass
# run-parts two directories, on a host whose run-parts takes one, is rebuilt into DIR/fitted/ with
# only /etc/kernel/X.d in those calls; the kernel and modules themselves are unchanged.
fit_debs() {
    local dir=$1 f ctl work out s stamp
    shift
    for f in "$@"; do
        ctl=$(mktemp -d) || return 1
        dpkg-deb -e "$dir/$f" "$ctl" || return 1
        if ! one_dir_run_parts || ! names_share_hook_dirs "$ctl"/*; then
            rm -rf -- "$ctl"
            echo "$dir/$f"
            continue
        fi
        rm -rf -- "$ctl"
        out="$dir/fitted/$f"
        if [[ ! -s "$out" ]]; then
            echo "Rebuilding $f: its scripts pass run-parts /usr/share/kernel/*.d too, which this run-parts can't take." >&2
            if compgen -G "/usr/share/kernel/*.d" >/dev/null; then
                echo "FATAL: this host has /usr/share/kernel/*.d hooks the rebuilt package would skip." >&2
                return 1
            fi
            mkdir -p "$dir/fitted" && work=$(mktemp -d -p "$dir/fitted") || return 1
            # TD-578: not dpkg-deb -R, which unpacks DEBIAN/ into the extracted root and so stamps it
            # with the current time; the root keeps the original's mtime, so only the scripts change.
            dpkg-deb -x "$dir/$f" "$work/pkg" && stamp=$(stat -c %y "$work/pkg") &&
                dpkg-deb -e "$dir/$f" "$work/pkg/DEBIAN" || return 1
            for s in preinst postinst prerm postrm; do
                [[ -f "$work/pkg/DEBIAN/$s" ]] && drop_share_hook_dirs "$work/pkg/DEBIAN/$s"
            done
            if names_share_hook_dirs "$work"/pkg/DEBIAN/*; then
                echo "FATAL: $f still names /usr/share/kernel hook directories after the rewrite." >&2
                return 1
            fi
            touch -d "$stamp" "$work/pkg" || return 1
            dpkg-deb --root-owner-group -b "$work/pkg" "$out.part" >/dev/null && mv -- "$out.part" "$out" || return 1
            rm -rf -- "$work"
        fi
        echo "$out"
    done
}
# repair_unfinished PACKAGE...: purges any of them left half-installed by an earlier attempt that
# failed in their own scripts, after removing /usr/share/kernel/X.d from their installed scripts so
# the purge can run. Packages that are installed, or absent, are left alone.
repair_unfinished() {
    local p st s broken=()
    for p in "$@"; do
        st=$(dpkg-query -W -f='${db:Status-Abbrev}' "$p" 2>/dev/null) || continue
        # Status (2nd letter) unpacked, half-installed, half-configured or awaiting triggers, or
        # an error (3rd letter) such as reinstall-required: what a failed maintainer script leaves.
        if [[ "${st:1:1}" == [UHFWt] || "${st:2:1}" == R ]]; then broken+=("$p"); fi
    done
    [[ ${#broken[@]} -eq 0 ]] && return 0
    echo "Left unfinished by an earlier attempt: ${broken[*]}"
    ask "Purge them first (their scripts' /usr/share/kernel hook directories removed so the purge can run)?" || return 1
    for p in "${broken[@]}"; do
        for s in preinst postinst prerm postrm; do
            sudo test -f "/var/lib/dpkg/info/$p.$s" &&
                sudo sed -i -E 's#(/etc/kernel/[a-z]+\.d)[[:space:]]+/usr/share/kernel/[a-z]+\.d#\1#g' "/var/lib/dpkg/info/$p.$s"
        done
    done
    sudo dpkg --purge --force-remove-reinstreq "${broken[@]}"
}
# arm KERNEL_RELEASE: GRUB boots the running kernel by default and KERNEL_RELEASE next time only.
arm() {
    local stock target
    if [[ ! -f "$GRUB_BACKUP" ]]; then
        cp /etc/default/grub "$GRUB_BACKUP" || return 1
    fi
    if ! grep -q -E '^GRUB_DEFAULT=saved$' /etc/default/grub; then
        sudo sed -i -E 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT=saved/' /etc/default/grub
        grep -q -E '^GRUB_DEFAULT=' /etc/default/grub || echo 'GRUB_DEFAULT=saved' | sudo tee -a /etc/default/grub >/dev/null
        sudo update-grub || return 1
    fi
    stock=$(grub_entry "$(uname -r)")
    target=$(grub_entry "$1")
    if [[ -z "$stock" ]]; then
        echo "FATAL: GRUB has no entry for the running kernel $(uname -r) (grub.cfg)."
        return 1
    fi
    if [[ -z "$target" ]]; then
        echo "FATAL: GRUB has no entry for $1 (grub.cfg)."
        return 1
    fi
    sudo grub-set-default "$stock" && sudo grub-reboot "$target" || return 1
    echo "GRUB: default \"$stock\", next boot only \"$target\"."
}

check() {
    echo "Running kernel: $(uname -r)"
    if secure_boot_on; then
        echo "Secure Boot: on (install signs the kernel with this script's key)"
    else
        echo "Secure Boot: off (or mokutil missing)"
    fi
    echo "Signing key ($MOK_DIR): $(mok_state)"
    echo "GRUB_DEFAULT: $(grep -E '^GRUB_DEFAULT=' /etc/default/grub 2>/dev/null | cut -d= -f2-)"
    sudo grub-editenv list 2>/dev/null | sed 's/^/  grubenv: /'
    echo "Newest mainline release: $(curl -fsSL "$MAINLINE/" 2>/dev/null | latest_release || echo '? (no network)')"
    if [[ -s "$INSTALLED" ]]; then
        echo "Installed by this script:"
        sed 's/^/  /' "$INSTALLED"
    else
        echo "Installed by this script: none"
    fi
}

case "$COMMAND" in
    check)
        check
        ;;
    install)
        SIGN=0
        if secure_boot_on; then
            signing_tools || {
                echo "Nothing changed."
                exit 1
            }
            SIGN=1
        fi
        if [[ -z "$VERSION" ]]; then
            VERSION=$(curl -fsSL "$MAINLINE/" | latest_release)
            [[ -n "$VERSION" ]] || {
                echo "FATAL: could not read the mainline index ($MAINLINE/); nothing changed."
                exit 1
            }
        fi
        base="$MAINLINE/v$VERSION/amd64"
        mapfile -t files < <(curl -fsSL "$base/" | package_files)
        if [[ ${#files[@]} -ne 2 ]]; then
            echo "FATAL: $base/ doesn't list one generic image and one modules package (found ${#files[@]});"
            echo "the amd64 build of v$VERSION may have failed. Pick another release. Nothing changed."
            exit 1
        fi
        krel=$(kernel_release "${files[0]}")
        echo "Mainline v$VERSION: kernel $krel"
        dir="$DOWNLOADS/v$VERSION"
        mkdir -p "$dir"
        for f in "${files[@]}" CHECKSUMS; do
            [[ -s "$dir/$f" ]] || curl -fsSL -o "$dir/$f" "$base/$f" || {
                echo "FATAL: download of $base/$f failed; nothing installed."
                exit 1
            }
        done
        # CHECKSUMS lists SHA-1 and SHA-256 sums; check the SHA-256 ones of the two packages.
        if ! (cd "$dir" && grep -E '^[0-9a-f]{64}  ' CHECKSUMS | grep -F -e "${files[0]}" -e "${files[1]}" >sha256.txt &&
            [[ $(grep -c . sha256.txt) -eq 2 ]] && sha256sum -c sha256.txt); then
            echo "FATAL: the packages don't match the release's CHECKSUMS; nothing installed."
            exit 1
        fi
        pkgs=()
        for f in "${files[@]}"; do pkgs+=("$(dpkg-deb -f "$dir/$f" Package)"); done
        repair_unfinished "${pkgs[@]}" || {
            echo "FATAL: ${pkgs[*]} are still unfinished; nothing installed."
            exit 1
        }
        debs_txt=$(fit_debs "$dir" "${files[@]}") || {
            echo "FATAL: could not prepare the packages for this host's run-parts; nothing installed."
            exit 1
        }
        mapfile -t debs <<<"$debs_txt"
        plan=$(sudo apt-get install -s "${debs[@]}" 2>&1)
        outside=$(outside_kernel "${pkgs[@]}" <<<"$plan")
        if [[ -n "$outside" ]]; then
            echo "FATAL: apt would also change packages outside the kernel; nothing changed:"
            echo "$outside" | sed 's/^/  /'
            exit 1
        fi
        echo "Will install ${pkgs[*]}, then make GRUB boot $(uname -r) by default and $krel once."
        [[ $SIGN == 1 ]] && echo "Secure Boot is on: will sign $krel with the key in $MOK_DIR (key now: $(mok_state)) and have the firmware trust it."
        ask "Install kernel $krel?" || {
            echo "Nothing changed."
            exit 0
        }
        sudo apt-get install -y "${debs[@]}" || {
            echo "FATAL: apt failed; see above."
            exit 1
        }
        grep -q -x -F "$krel ${pkgs[*]}" "$INSTALLED" 2>/dev/null || echo "$krel ${pkgs[*]}" >>"$INSTALLED"
        if [[ $SIGN == 1 ]]; then
            sign_kernel "$krel" || {
                echo "GRUB not armed: the stock kernel boots next. Fix the above, then ./mainline_kernel.sh boot-once."
                exit 1
            }
        fi
        arm "$krel" || {
            echo "FATAL: GRUB not set up; the new kernel may now be GRUB's default. Check before rebooting."
            exit 1
        }
        echo
        echo "Done. sudo reboot boots $krel once; then ./prepare_host.sh and the run. The boot after that is $(uname -r) again."
        ;;
    boot-once)
        [[ -s "$INSTALLED" ]] || {
            echo "No upstream kernel installed by this script."
            exit 1
        }
        if [[ -n "$VERSION" ]]; then
            krel=$(awk -v v="$VERSION" 'index($1, v "-") == 1 {print $1}' "$INSTALLED" | tail -1)
        else
            krel=$(awk '{print $1}' "$INSTALLED" | tail -1)
        fi
        [[ -n "$krel" ]] || {
            echo "v$VERSION isn't one this script installed."
            exit 1
        }
        ask "Boot $krel once, at the next reboot?" || exit 0
        if secure_boot_on; then
            signing_tools && sign_kernel "$krel" || exit 1
        fi
        arm "$krel" || exit 1
        ;;
    remove)
        [[ -s "$INSTALLED" ]] || {
            echo "No upstream kernel installed by this script."
            exit 0
        }
        if grep -q -E "^$(uname -r) " "$INSTALLED"; then
            echo "FATAL: $(uname -r) is running; reboot into the stock kernel first."
            exit 1
        fi
        mapfile -t pkgs < <(cut -d' ' -f2- "$INSTALLED" | tr ' ' '\n' | grep .)
        echo "Will remove ${pkgs[*]} and put /etc/default/grub back."
        ask "Remove them?" || exit 0
        sudo apt-get remove -y "${pkgs[@]}" || exit 1
        if [[ -f "$GRUB_BACKUP" ]]; then
            sudo cp "$GRUB_BACKUP" /etc/default/grub && rm -f -- "$GRUB_BACKUP"
        fi
        sudo update-grub
        rm -f -- "$INSTALLED"
        state=$(mok_state)
        if [[ "$state" != none ]]; then
            echo "The signing key in $MOK_DIR is $state."
            if ask "Take it off the firmware's list and delete it?"; then
                case "$state" in
                    enrolled)
                        echo "mokutil asks for a one-time password; at the next boot, at the console, MokManager"
                        echo "asks to Delete MOK (Continue, Yes, the password, Reboot)."
                        sudo mokutil --delete "$MOK_DER" </dev/tty >/dev/tty 2>&1 || exit 1
                        ;;
                    pending) sudo mokutil --revoke-import </dev/tty >/dev/tty 2>&1 || exit 1 ;;
                esac
                sudo rm -f -- "$MOK_KEY" "$MOK_PEM" "$MOK_DER" && sudo rmdir -- "$MOK_DIR"
            fi
        fi
        echo "Done."
        ;;
esac
