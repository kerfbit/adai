// @adai-status: experimental
// @adai-version: 0.3.0
// @adai-reviewed: 2026-10-31

#include "RegistryServerArgs.hpp"
#include <algorithm>
#include <cctype>
#include <type_traits>

namespace adai {

RegistryServerArgs parse_registry_server_args(int argc, char* argv[]) {
    RegistryServerArgs result;

    // TODO: See TD-488 in TECHNICAL_DEBT.md - bare std::stoi ("abc" aborts, "8090x" -> 8090, 99999 binds 34463, no
    // ranges); a value flag with no value is dropped silently (bare --admin-enabled leaves admin open);
    // unknown arguments are ignored.
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
        } else if (arg == "--claim-lease-minutes" && i + 1 < argc) {
            result.claim_lease_minutes = std::stoi(argv[++i]);
        } else if (arg == "--max-upload-mb" && i + 1 < argc) {
            result.max_upload_mb = std::stoi(argv[++i]);
        } else if (arg == "--max-concurrent-fetches" && i + 1 < argc) {
            result.max_concurrent_fetches = std::stoi(argv[++i]);
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
    cfg.claim_lease_minutes =
        std::max(0, args.claim_lease_minutes.value_or(file_config.registry_claim_lease_minutes));
    cfg.max_upload_mb = std::max(1, args.max_upload_mb.value_or(file_config.registry_max_upload_mb));
    cfg.max_concurrent_fetches = std::max(
        1, args.max_concurrent_fetches.value_or(file_config.registry_max_concurrent_fetches));

    // TD-284: FTP listener settings, config file < CLI. They used to come from the CLI only, so the
    // file's FTPS_ENABLED and FTP_DATA_SERVER_SECRET silently did nothing.
    // The CLI value if the flag was passed (warning when it differs from a non-empty file value),
    // else the file's.
    auto pick = [&cfg](const auto& cli, const auto& file, const char* key, const char* flag) {
        using T = std::decay_t<decltype(file)>;
        if (!cli) return T(file);
        if (*cli != file && file != T{}) {
            cfg.warnings.push_back(std::string(flag) + " overrides " + key + " from the config file");
        }
        return T(*cli);
    };
    cfg.ftp_enabled = pick(args.ftp_enabled, file_config.ftp_server_enabled, "FTP_SERVER_ENABLED", "--ftp-enabled");
    cfg.ftp_port = pick(args.ftp_port, file_config.ftp_server_port, "FTP_SERVER_PORT", "--ftp-port");
    cfg.ftp_advertise_ip = pick(args.ftp_advertise_ip, file_config.ftp_advertise_ip, "FTP_ADVERTISE_IP", "--ftp-ip");
    cfg.ftp_pasv_min = pick(args.ftp_pasv_min, file_config.ftp_pasv_port_min, "FTP_PASV_PORT_MIN", "--ftp-pasv-min");
    cfg.ftp_pasv_max = pick(args.ftp_pasv_max, file_config.ftp_pasv_port_max, "FTP_PASV_PORT_MAX", "--ftp-pasv-max");
    cfg.ftps_enabled = pick(args.ftps_enabled, file_config.ftps_enabled, "FTPS_ENABLED", "--ftps");
    cfg.ftp_cert_file = pick(args.ftp_cert_file, file_config.ftp_cert_file, "FTP_CERT_FILE", "--ftp-cert");
    cfg.ftp_key_file = pick(args.ftp_key_file, file_config.ftp_key_file, "FTP_KEY_FILE", "--ftp-key");

    // The shipped config file used to carry FTP_DATA_SERVER_SECRET=change-me-in-production. A
    // publicly known HMAC key would let anyone compute token passwords, so it counts as unset.
    std::string file_secret = file_config.ftp_data_server_secret;
    if (file_secret == "change-me-in-production") {
        cfg.warnings.push_back(
            "FTP_DATA_SERVER_SECRET is the shipped placeholder 'change-me-in-production'; ignoring it "
            "(each FTP token gets a random password). Set a real secret or remove the key.");
        file_secret.clear();
    }
    cfg.ftp_server_secret =
        pick(args.ftp_server_secret, file_secret, "FTP_DATA_SERVER_SECRET", "--ftp-secret");

    if (cfg.ftp_enabled && !cfg.ftps_enabled && !cfg.ftp_advertise_ip.empty() &&
        cfg.ftp_advertise_ip != "127.0.0.1" && cfg.ftp_advertise_ip != "::1" &&
        cfg.ftp_advertise_ip != "localhost") {
        cfg.warnings.push_back("FTP is enabled without FTPS on " + cfg.ftp_advertise_ip +
                               ": token passwords and dataset files cross the network in plaintext "
                               "(set FTPS_ENABLED=true or --ftps)");
    }
    return cfg;
}

}  // namespace adai
