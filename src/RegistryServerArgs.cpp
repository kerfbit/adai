// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

#include "RegistryServerArgs.hpp"
#include <cctype>

namespace adai {

RegistryServerArgs parse_registry_server_args(int argc, char* argv[]) {
    RegistryServerArgs result;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            result.config_path = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            result.port = std::stoi(argv[++i]);
        } else if (arg == "--data-dir" && i + 1 < argc) {
            result.data_dir = argv[++i];
        } else if (arg == "--ftp-enabled") {
            result.ftp_enabled = true;
        } else if (arg == "--ftp-port" && i + 1 < argc) {
            result.ftp_port = std::stoi(argv[++i]);
        } else if (arg == "--ftp-ip" && i + 1 < argc) {
            result.ftp_advertise_ip = argv[++i];
        } else if (arg == "--ftp-pasv-min" && i + 1 < argc) {
            result.ftp_pasv_min = std::stoi(argv[++i]);
        } else if (arg == "--ftp-pasv-max" && i + 1 < argc) {
            result.ftp_pasv_max = std::stoi(argv[++i]);
        } else if (arg == "--ftp-ttl" && i + 1 < argc) {
            result.ftp_ttl_minutes = std::stoi(argv[++i]);
        } else if (arg == "--ftp-secret" && i + 1 < argc) {
            result.ftp_server_secret = argv[++i];
        } else if (arg == "--ftps") {
            result.ftps_enabled = true;
        } else if (arg == "--ftp-cert" && i + 1 < argc) {
            result.ftp_cert_file = argv[++i];
        } else if (arg == "--ftp-key" && i + 1 < argc) {
            result.ftp_key_file = argv[++i];
        } else if (arg == "--ftp-max-sessions" && i + 1 < argc) {
            result.ftp_max_sessions = std::stoi(argv[++i]);
        } else if (arg == "--admin-enabled" && i + 1 < argc) {
            std::string v = argv[++i];
            for (auto& c : v)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            result.admin_enabled = (v == "true" || v == "1" || v == "yes" || v == "on");
        } else if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
        // Note: an unrecognized argument is silently ignored here, matching the original inline
        // loop exactly — see the doc comment on this declaration in RegistryServerArgs.hpp.
    }

    return result;
}

RegistryServerEffectiveConfig resolve_registry_server_config(
    const RegistryServerArgs& args, const ServiceConfig& file_config,
    const std::map<std::string, std::string>& admin_overrides) {
    RegistryServerEffectiveConfig cfg;
    cfg.port = args.port.value_or(file_config.registry_listen_port);
    cfg.data_dir = args.data_dir.value_or(file_config.registry_data_dir);

    cfg.ftp_ttl_minutes = file_config.ftp_token_ttl_minutes;
    cfg.ftp_max_sessions = file_config.ftp_max_sessions_per_run;

    if (auto it = admin_overrides.find("ftp_token_ttl_minutes"); it != admin_overrides.end()) {
        try {
            cfg.ftp_ttl_minutes = std::stoi(it->second);
        } catch (...) {
        }
    }
    if (auto it = admin_overrides.find("ftp_max_sessions_per_run"); it != admin_overrides.end()) {
        try {
            cfg.ftp_max_sessions = std::stoi(it->second);
        } catch (...) {
        }
    }

    // This run's explicit CLI flags win over everything, including persisted admin overrides.
    if (args.ftp_ttl_minutes)
        cfg.ftp_ttl_minutes = *args.ftp_ttl_minutes;
    if (args.ftp_max_sessions)
        cfg.ftp_max_sessions = *args.ftp_max_sessions;
    cfg.admin_enabled = args.admin_enabled.value_or(true);

    return cfg;
}

}  // namespace adai
