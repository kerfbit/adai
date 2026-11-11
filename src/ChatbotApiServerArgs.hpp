#pragma once

// @adai-status: beta
// @adai-version: 0.11.0
// @adai-reviewed: 2026-11-10

// TD-035: chatbot_api_server's argv parsing and required-config validation, pulled out of
// ChatbotAPIServer.cpp so it's testable without loading a real tokenizer/model or starting a
// real HTTP server — see tests/chatbot_api_server_args_test.cpp. The actual server lifecycle (ChatbotAPI
// class, request handlers) is already covered by chatbotapiTests.

#include <optional>
#include <set>
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
    // lives here rather than on ServiceConfig. batch_timeout_ms needs batched_inference (passing
    // --batch-timeout-ms alone is an error, TD-243); it mirrors BatchedInferenceConfig::timeout_ms's own default (see
    // BatchedInferenceEngine.hpp) so an unset flag reproduces that default exactly.
    bool batched_inference = false;
    int batch_timeout_ms = 50;
    // TD-242: the config keys the flags set (e.g. "TEMPERATURE" for --temperature), so a SIGHUP
    // reload can re-apply the flags and report file changes those flags hold.
    std::set<std::string> cli_keys;
    // --pipeline-inference and --integrated-inference were removed with their engines (TD-474/TD-475,
    // TD-417–TD-420); --batched-inference is the only engine mode left, so several engine modes at
    // once (TD-243) can't happen.
};

// Second pass: applies every flag except --config (already consumed) onto an already-loaded
// ServiceConfig, in place — CLI flags win over whatever the file/env already set. Returns
// .help=true on --help/-h (config is left unmodified in that case) or .error=true with a message
// on an unrecognized argument, a flag missing its value (TD-245), a malformed number (TD-240) or
// --batch-timeout-ms without --batched-inference (TD-243).
ChatbotApiServerArgsResult apply_chatbot_api_server_args(int argc, char* argv[],
                                                         ServiceConfig& config);

// Mirrors the "Validate required configuration" check in main(): returns an error message if
// `config` isn't startable (no vocab path, or any ConfigLoader::validate() error, all listed; TD-241),
// or std::nullopt if it is.
std::optional<std::string> validate_chatbot_api_server_config(const ServiceConfig& config);

// TD-251: copies the settings that describe the model chatbot_api_server loaded at startup (model
// and vocabulary paths, the architecture, the draft model, the world-model and hippocampal setup)
// from the running config into a reloaded one. Several of them can come from MNS rather than the
// file, and none applies before a restart, so a SIGHUP reload used to replace them with the file's
// values while the old model kept serving. The save interval isn't kept: a reload applies it.
void keep_loaded_model_settings(const ServiceConfig& running, ServiceConfig& reloaded);

}  // namespace adai
