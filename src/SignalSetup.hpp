#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-28

/**
 * SignalSetup — SIGHUP handling for binaries that don't reload their configuration (TD-283).
 *
 * Only chatbot_api_server reloads on SIGHUP. Every other long-running binary used to register only
 * SIGTERM/SIGINT, so SIGHUP took its default action and terminated the process: `kill -HUP` on a
 * trainer killed a pass mid-epoch, and on trainer_service it killed the supervisor and its admin
 * API. ignore_sighup() sets SIGHUP to SIG_IGN (no handler, so nothing async-signal-unsafe runs,
 * TD-249) and logs that a restart is needed to apply changes. An ignored disposition survives
 * exec(), so trainer_service's child passes ignore it too.
 */

#include <csignal>
#include <string>

#include "Logger.hpp"

namespace adai {

inline void ignore_sighup(const std::string& binary) {
#ifdef SIGHUP
    std::signal(SIGHUP, SIG_IGN);
    Logger::info("{}: SIGHUP is ignored; this binary doesn't reload its configuration (restart it to "
                 "apply changes)",
                 binary);
#else
    (void)binary;  // no SIGHUP on Windows
#endif
}

}  // namespace adai
