#!/bin/bash
#
# Smoke test for src/ChatbotGUI_main.cpp (TD-036).
#
# Targets chatbot_gui_binary directly (the executable ChatbotGUI_main.cpp
# actually compiles into — see src/CMakeLists.txt's chatbot_gui_binary
# target), not the chatbot_gui launcher wrapper (ChatbotGUI_wrapper.cpp,
# covered separately by chatbot_gui_wrapper_test.sh). QT_QPA_PLATFORM=
# offscreen is set explicitly rather than relying on ambient environment.
# Since TD-266, main() handles --help before constructing QApplication, which
# the last case checks with no platform plugin and no display at all.

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

BUILD_DIR="$(find_build_dir "src/chatbot_gui_binary")"
if [[ -z "$BUILD_DIR" ]]; then
    t "(skipped: chatbot_gui_binary not built in this checkout — BUILD_GUI off, or no Qt)"
    harness_report
    exit $?
fi
BIN="${BUILD_DIR}/src/chatbot_gui_binary"
export QT_QPA_PLATFORM=offscreen

# 10s timeout: if --help ever stopped short-circuiting before window
# construction, this would otherwise block forever in app.exec()'s event
# loop instead of failing fast (confirmed directly: reproduces as a hang,
# not a clean non-zero exit, when the --help check is disabled).
t "chatbot_gui_binary --help exits 0"
run timeout 10 "$BIN" --help
assert_exit 0

t "chatbot_gui_binary --help prints usage without opening a real window"
assert_contains "Usage:"
assert_contains "[server_url]"  # TD-260: a client of chatbot_api_server
assert_not_contains "vocab_file"

t "chatbot_gui_binary -h (short flag) exits 0 with the same usage text"
run timeout 10 "$BIN" -h
assert_exit 0
assert_contains "Usage:"

t "TD-266: --help needs no display or Qt platform plugin"
run env -u DISPLAY -u WAYLAND_DISPLAY -u QT_QPA_PLATFORM timeout 10 "$BIN" --help
assert_exit 0
assert_contains "Usage:"

t "TD-260: the old [vocab_file] [model_file] form is refused with an explanation"
run timeout 10 "$BIN" vocab.txt chatbot_model.bin
assert_exit 1
assert_contains "no longer loads a vocabulary"

harness_report
exit $?
