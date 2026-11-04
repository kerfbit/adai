#!/bin/bash
#
# Smoke test for src/ChatbotGUI_wrapper.cpp (TD-036).
#
# This binary has no argv handling of its own — it unsets a few snap-related
# env vars, removes snap entries from LD_LIBRARY_PATH/QT_PLUGIN_PATH (TD-264), then
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
assert_contains "[server_url]"  # TD-260

t "TD-103 regression: wrapper resolves its sibling binary via /proc/self/exe, not CWD, when invoked by absolute path from elsewhere"
SCRATCH="$(mktemp_dir)"
run_in "$SCRATCH" timeout 10 "$BIN" --help
assert_exit 0
assert_contains "Usage:"
assert_not_contains "Failed to execute"

# TD-264: the environment the wrapper hands chatbot_gui_binary, read from /proc once it has exec'd.
# Started with a snap-like environment; the GUI runs offscreen against an unreachable server and is
# stopped by its own PID (only this test's child is signalled).
gui_environment() {
    env -i HOME="$HOME" PATH="$PATH" QT_QPA_PLATFORM=offscreen "$@" "$BIN" http://127.0.0.1:1 >/dev/null 2>&1 &
    local pid=$! i
    for i in $(seq 1 100); do
        [[ "$(readlink "/proc/${pid}/exe" 2>/dev/null)" == *chatbot_gui_binary* ]] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
}

t "TD-264: under snap, only snap entries leave LD_LIBRARY_PATH/QT_PLUGIN_PATH, in order, with no system paths added"
RUN_OUT="$(gui_environment SNAP=/snap/code/180 GTK_PATH=/snap/code/180/usr/lib/gtk-3.0 \
    LD_LIBRARY_PATH=/opt/td264-b:/snap/core20/current/lib:/opt/td264-a \
    QT_PLUGIN_PATH=/snap/kde/1/plugins:/opt/td264-plugins)"
assert_contains "LD_LIBRARY_PATH=/opt/td264-b:/opt/td264-a"
assert_contains "QT_PLUGIN_PATH=/opt/td264-plugins"
assert_not_contains "x86_64-linux-gnu:/lib/x86_64-linux-gnu"
assert_not_contains "SNAP="
assert_not_contains "GTK_PATH="

t "TD-264: under snap, QT_QPA_PLATFORM_PLUGIN_PATH is the build's own platform plugin directory"
assert_contains "QT_QPA_PLATFORM_PLUGIN_PATH="
assert_contains "/platforms"

t "TD-264: outside snap the user's LD_LIBRARY_PATH is untouched and Qt's plugin path isn't set"
RUN_OUT="$(gui_environment LD_LIBRARY_PATH=/opt/td264-b:/opt/td264-a)"
assert_contains "LD_LIBRARY_PATH=/opt/td264-b:/opt/td264-a"
assert_not_contains "QT_QPA_PLATFORM_PLUGIN_PATH="

harness_report
exit $?
