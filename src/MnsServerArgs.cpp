// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

#include "MnsServerArgs.hpp"
#include <cctype>

namespace adai {

bool parse_bool_flag(const std::string& value, bool default_value) {
    std::string lower = value;
    for (auto& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "true" || lower == "1" || lower == "yes" || lower == "on")
        return true;
    if (lower == "false" || lower == "0" || lower == "no" || lower == "off")
        return false;
    return default_value;
}

MnsServerArgs parse_mns_server_args(int argc, char* argv[]) {
    MnsServerArgs result;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            result.config_path = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            try {
                result.port = std::stoi(argv[++i]);
            } catch (...) {
                result.error = true;
                result.error_message = "Invalid port: " + std::string(argv[i]);
                return result;
            }
        } else if (arg == "--data-dir" && i + 1 < argc) {
            result.data_dir = argv[++i];
        } else if (arg == "--registry-url" && i + 1 < argc) {
            result.registry_url = argv[++i];
        } else if (arg == "--registry-group" && i + 1 < argc) {
            result.registry_group = argv[++i];
        } else if (arg == "--admin-enabled" && i + 1 < argc) {
            result.admin_enabled = parse_bool_flag(argv[++i], true);
        } else if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        } else {
            result.error = true;
            result.error_message = "Unknown argument: " + arg;
            return result;
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
    cfg.admin_enabled = args.admin_enabled.value_or(true);

    return cfg;
}

}  // namespace adai
