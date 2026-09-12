#!/bin/bash
#
# Smoke test for src/ChatbotGUI_wrapper.cpp (TD-036).
#
# This binary has no argv handling of its own — it unsets a few snap-related
# env vars, fixes up LD_LIBRARY_PATH/QT_QPA_PLATFORM_PLUGIN_PATH, then
# execvp()s chatbot_gui_binary (found next to itself via /proc/self/exe) with
# argv passed straight through. So a --help smoke test exercises its real
# job end to end: resolve the sibling binary correctly and hand off to it
# cleanly, rather than just re-testing chatbot_gui_binary's own argument
# parsing (covered separately by chatbot_gui_binary_main_test.sh).
#
# Also regression-tests the specific thing this wrapper's own resolve_exe_dir()
# comment calls out (TD-103): it must locate chatbot_gui_binary via
# /proc/self/exe, not argv[0]/CWD, so it has to keep working when invoked by
# absolute path from a CWD that isn't the binary's own directory.

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

BUILD_DIR="$(find_build_dir "src/chatbot_gui")"
if [[ -z "$BUILD_DIR" || ! -x "${BUILD_DIR}/src/chatbot_gui_binary" ]]; then
    t "(skipped: chatbot_gui wrapper and/or chatbot_gui_binary not built in this checkout)"
    harness_report
    exit $?
fi
BIN="${BUILD_DIR}/src/chatbot_gui"
export QT_QPA_PLATFORM=offscreen

# 10s timeout: --help is handled entirely inside chatbot_gui_binary (this
# wrapper has no argv logic of its own), so a broken hand-off here would
# either exit non-zero/print "Failed to execute" (a bad exec) or, if it did
# exec successfully into a chatbot_gui_binary whose own --help broke, hang
# in that binary's event loop rather than failing fast — same failure mode
# covered directly in chatbot_gui_binary_main_test.sh.
t "chatbot_gui (wrapper) --help exits 0"
run timeout 10 "$BIN" --help
assert_exit 0

t "chatbot_gui (wrapper) passes --help through to chatbot_gui_binary's real usage text"
assert_contains "Usage:"
assert_contains "vocab_file"

t "TD-103 regression: wrapper resolves its sibling binary via /proc/self/exe, not CWD, when invoked by absolute path from elsewhere"
SCRATCH="$(mktemp_dir)"
run_in "$SCRATCH" timeout 10 "$BIN" --help
assert_exit 0
assert_contains "Usage:"
assert_not_contains "Failed to execute"

harness_report
exit $?
