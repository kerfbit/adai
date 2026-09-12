#!/bin/bash
#
# Smoke test for src/ChatbotGUI_main.cpp (TD-036).
#
# Targets chatbot_gui_binary directly (the executable ChatbotGUI_main.cpp
# actually compiles into — see src/CMakeLists.txt's chatbot_gui_binary
# target), not the chatbot_gui launcher wrapper (ChatbotGUI_wrapper.cpp,
# covered separately by chatbot_gui_wrapper_test.sh). QT_QPA_PLATFORM=
# offscreen is set explicitly rather than relying on ambient environment —
# this file's main() constructs a real QApplication before it ever checks
# argv for --help, so it needs a usable Qt platform plugin even just to
# print usage and exit; "offscreen" is the standard headless one and is
# confirmed installed in this environment.

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
assert_contains "vocab_file"
assert_contains "model_file"

t "chatbot_gui_binary -h (short flag) exits 0 with the same usage text"
run timeout 10 "$BIN" -h
assert_exit 0
assert_contains "Usage:"

harness_report
exit $?
