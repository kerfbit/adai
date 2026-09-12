#!/bin/bash
#
# Smoke test for src/MnsManagerGUI_main.cpp (TD-036).
#
# Like ChatbotGUI_main.cpp, this file constructs a real QApplication before
# parsing argv, so QT_QPA_PLATFORM=offscreen is set explicitly rather than
# relying on ambient environment. Also covers its --url/--config flags'
# basic parsing (both take a following value; --help wins even after them).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${SCRIPT_DIR}/../scripts/harness.sh"

find_build_dir() {
    local marker="$1"
    local dir
    for dir in "${REPO_ROOT}/build/debug" "${REPO_ROOT}/build/release" \
               "${REPO_ROOT}/build/portable" "${REPO_ROOT}/build/relwithdebinfo" \
               "${REPO_ROOT}/build/gpu" "${REPO_ROOT}/build/sycl" \
               "${REPO_ROOT}/build"; do
        if [[ -x "${dir}/${marker}" ]]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

BUILD_DIR="$(find_build_dir "bin/mns_manager_gui")"
if [[ -z "$BUILD_DIR" ]]; then
    t "(skipped: mns_manager_gui not built in this checkout — BUILD_GUI off, or no Qt/httplib)"
    harness_report
    exit $?
fi
BIN="${BUILD_DIR}/bin/mns_manager_gui"
export QT_QPA_PLATFORM=offscreen

# 10s timeout: a broken --help path here would fall through to
# window.show(); app.exec() and hang in the event loop rather than exit
# non-zero (confirmed against the sibling ChatbotGUI_main.cpp binary).
t "mns_manager_gui --help exits 0"
run timeout 10 "$BIN" --help
assert_exit 0

t "mns_manager_gui --help prints usage without opening a real window"
assert_contains "Usage:"
assert_contains "--url"
assert_contains "--config"

t "mns_manager_gui -h (short flag) exits 0 with the same usage text"
run timeout 10 "$BIN" -h
assert_exit 0
assert_contains "Usage:"

t "--help after --url/--config still wins and exits 0 (argv scanned left to right)"
run timeout 10 "$BIN" --url http://localhost:9999 --help
assert_exit 0
assert_contains "Usage:"

harness_report
exit $?
