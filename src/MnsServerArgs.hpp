#pragma once

// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-01

// TD-035: argv parsing and config-precedence resolution for mns_server's main(), pulled out of
// ModelNameServiceServer.cpp so it's testable without spinning up a real ModelNameService
// instance — see tests/mns_server_args_test.cpp.

#include <map>
#include <optional>
#include <string>
#include "Config.hpp"

namespace adai {


struct MnsServerArgs {
    std::optional<std::string> config_path;
    std::optional<int> port;
    std::optional<std::string> data_dir;
    std::optional<std::string> registry_url;
    std::optional<std::string> registry_group;
    std::optional<bool> admin_enabled;

    bool help = false;
    // Set on an invalid --port (not a whole number 1-65535) or --admin-enabled (not a boolean)
    // value, a flag missing its value, or an unrecognized argument (TD-456). error_message is
    // human-readable and already ends without a trailing newline.
    bool error = false;
    std::string error_message;
};

// Parses argv[1..] (argv[0], the program name, is not consulted). Never throws — an invalid
// value, a missing value or an unrecognized flag sets .error/.error_message instead.
MnsServerArgs parse_mns_server_args(int argc, char* argv[]);

struct MnsServerEffectiveConfig {
    int port = 0;
    std::string data_dir;
    std::string registry_url;
    std::string registry_group;
    bool admin_enabled = true;  // --admin-enabled, else MNS_ADMIN_ENABLED (TD-456)
};

// Resolves the effective startup config from (lowest to highest precedence): `file_config` <
// `admin_overrides` (persisted via PUT /admin/config, see CLAUDE.md "Daemon admin config API") <
// `args`' explicit CLI flags. port/data_dir are never admin-mutable, so `admin_overrides` is
// only consulted for registry_url/registry_group — matching ModelNameServiceServer.cpp's
// original inline logic exactly, including the case where registry_group is set (e.g. via a
// persisted override) even when registry_url is empty.
MnsServerEffectiveConfig resolve_mns_server_config(
    const MnsServerArgs& args, const ServiceConfig& file_config,
    const std::map<std::string, std::string>& admin_overrides);

}  // namespace adai
