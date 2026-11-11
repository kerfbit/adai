// @adai-status: beta
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-10

#include "MnsServerArgs.hpp"
#include <set>

namespace adai {

MnsServerArgs parse_mns_server_args(int argc, char* argv[]) {
    MnsServerArgs result;
    auto fail = [&](const std::string& message) {
        result.error = true;
        result.error_message = message;
        return result;
    };
    static const std::set<std::string> kValueFlags = {"--config",       "--port",           "-p",
                                                      "--data-dir",     "--registry-url",   "--registry-group",
                                                      "--admin-enabled"};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
        // TD-456: a flag given no value used to be reported as "Unknown argument".
        if (kValueFlags.count(arg) != 0 && (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
            return fail("Missing value for " + arg);
        }
        if (arg == "--config") {
            result.config_path = argv[++i];
        } else if (arg == "--port" || arg == "-p") {
            // TD-456: std::stoi took "8083x" as 8083, and 99999 listened on 34463.
            const std::string value = argv[++i];
            const auto port = parse_strict_int(value);
            if (!port || *port < 1 || *port > 65535) {
                return fail("Invalid value for " + arg + ": '" + value + "' (expected 1-65535)");
            }
            result.port = *port;
        } else if (arg == "--data-dir") {
            result.data_dir = argv[++i];
        } else if (arg == "--registry-url") {
            result.registry_url = argv[++i];
        } else if (arg == "--registry-group") {
            result.registry_group = argv[++i];
        } else if (arg == "--admin-enabled") {
            // TD-456: an unrecognised value fell back to true, so "--admin-enabled flase" left
            // PUT /admin/config open.
            const std::string value = argv[++i];
            const auto enabled = parse_strict_bool(value);
            if (!enabled) {
                return fail("Invalid value for --admin-enabled: '" + value +
                            "' (expected true/false, yes/no, on/off, enabled/disabled or 1/0)");
            }
            result.admin_enabled = *enabled;
        } else {
            return fail("Unknown argument: " + arg);
        }
    }

    return result;
}

MnsServerEffectiveConfig resolve_mns_server_config(
    const MnsServerArgs& args, const ServiceConfig& file_config,
    const std::map<std::string, std::string>& admin_overrides) {
    MnsServerEffectiveConfig cfg;
    cfg.port = args.port.value_or(file_config.name_service_port);
    cfg.data_dir = args.data_dir.value_or(file_config.name_service_dir);

    cfg.registry_url = file_config.registry_server_url;
    cfg.registry_group = file_config.run_group.empty() ? "default" : file_config.run_group;

    if (auto it = admin_overrides.find("registry_url"); it != admin_overrides.end()) {
        cfg.registry_url = it->second;
    }
    if (auto it = admin_overrides.find("registry_group"); it != admin_overrides.end()) {
        cfg.registry_group = it->second;
    }

    // This run's explicit CLI flags win over everything, including persisted admin overrides
    // (they don't overwrite the persisted value, just this run).
    if (args.registry_url)
        cfg.registry_url = *args.registry_url;
    if (args.registry_group)
        cfg.registry_group = *args.registry_group;
    cfg.admin_enabled = args.admin_enabled.value_or(file_config.name_service_admin_enabled);  // TD-456

    return cfg;
}

}  // namespace adai
