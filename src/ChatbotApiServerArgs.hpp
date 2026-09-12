#pragma once

// @adai-status: experimental
// @adai-version: 0.4.0
// @adai-reviewed: 2026-09-12

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
    // TD-038: enables the GET /admin/profile endpoint (ChatbotAPI::enable_profiling()) — a
    // runtime behavior toggle, not part of ServiceConfig, so it lives on the result rather than
    // being applied to `config` like every other flag here.
    bool profile = false;
    // TD-038: enables ChatbotAPI::enable_batched_inference() — same reasoning as `profile` above:
    // a runtime mode toggle for this one process, not a persisted/hot-reloadable setting, so it
    // lives here rather than on ServiceConfig. batch_timeout_ms only takes effect when
    // batched_inference is true; it mirrors BatchedInferenceConfig::timeout_ms's own default (see
    // BatchedInferenceEngine.hpp) so an unset flag reproduces that default exactly.
    bool batched_inference = false;
    int batch_timeout_ms = 50;
    // TD-038: enables ChatbotAPI::enable_pipeline_inference() — same "runtime toggle on the
    // result" reasoning as batched_inference above. Requires vocab_path to be set (validated the
    // same place --vocab itself is), since enable_pipeline_inference() needs an actual file to
    // reload into the model's encoder-internal tokenizer.
    bool pipeline_inference = false;
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
