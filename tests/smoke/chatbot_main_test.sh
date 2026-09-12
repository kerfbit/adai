#!/bin/bash
#
# Smoke test for src/ChatbotCLI_main.cpp (TD-036).
#
# A GTest unit test isn't a good fit for a main() — this invokes the real
# built `chatbot` binary and checks its argv-parsing shim end to end: exit
# code and non-empty --help/-h output. The interactive run() path (talking
# to a real chatbot_api_server) is intentionally out of scope here — that's
# ChatbotCLI's own behavior, not this thin wrapper's.

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

BUILD_DIR="$(find_build_dir "bin/chatbot")"
if [[ -z "$BUILD_DIR" ]]; then
    t "(skipped: chatbot binary not built in this checkout)"
    harness_report
    exit $?
fi
BIN="${BUILD_DIR}/bin/chatbot"

t "chatbot --help exits 0"
run "$BIN" --help
assert_exit 0

t "chatbot --help prints usage"
assert_contains "Usage:"
assert_contains "server_url"
assert_contains "conversation_save_file"

t "chatbot -h (short flag) exits 0 with the same usage text"
run "$BIN" -h
assert_exit 0
assert_contains "Usage:"

harness_report
exit $?
