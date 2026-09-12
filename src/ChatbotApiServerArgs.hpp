#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

// TD-035: chatbot_api_server's argv parsing and required-config validation, pulled out of
// ChatbotAPIServer.cpp so it's testable without loading a real tokenizer/model or starting a
// real HTTP server — see ChatbotApiServerArgs_test.cpp. The actual server lifecycle (ChatbotAPI
// class, request handlers) is already covered by chatbotapiTests.

#include <optional>
#include <string>
#include "Config.hpp"

namespace adai {

// First pass: scans argv for just --config, before any file config can be loaded (mirrors
// ChatbotAPIServer.cpp's own two-pass design — --config must be known before
// ConfigLoader::discover_config_path()/load() run, but every other flag overrides the loaded
// file, so it can't be applied until after).
std::optional<std::string> extract_config_path_arg(int argc, char* argv[]);

struct ChatbotApiServerArgsResult {
    bool help = false;
    bool error = false;
    std::string error_message;
};

// Second pass: applies every flag except --config (already consumed) onto an already-loaded
// ServiceConfig, in place — CLI flags win over whatever the file/env already set. Returns
// .help=true on --help/-h (config is left unmodified in that case) or .error=true with a message
// on an unrecognized argument.
ChatbotApiServerArgsResult apply_chatbot_api_server_args(int argc, char* argv[],
                                                         ServiceConfig& config);

// Mirrors the "Validate required configuration" check in main(): returns an error message if
// `config` isn't startable, or std::nullopt if it is.
std::optional<std::string> validate_chatbot_api_server_config(const ServiceConfig& config);

}  // namespace adai
