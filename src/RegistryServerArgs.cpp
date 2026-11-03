// @adai-status: experimental
// @adai-version: 0.6.0
// @adai-reviewed: 2026-11-03

#include "RegistryServerArgs.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>
#include "nlohmann/json.hpp"

namespace adai {

namespace {
constexpr int kMaxInt = std::numeric_limits<int>::max();

template <typename Flag>
std::string int_range_text(const Flag& f) {
    if (f.max == kMaxInt) return f.min == 0 ? "a whole number >= 0" : "a whole number >= 1";
    return std::to_string(f.min) + "-" + std::to_string(f.max);
}
}  // namespace

RegistryServerArgs parse_registry_server_args(int argc, char* argv[]) {
    RegistryServerArgs result;
    auto fail = [&](const std::string& message) {
        result.error = true;
        result.error_message = message;
        return result;
    };
    // TD-488: every number is a whole number in range (bare std::stoi aborted on "abc", took "8090x"
    // as 8090 and bound 34463 for 99999); a value flag given no value, and an unknown argument, are
    // errors (a bare --admin-enabled used to leave the admin endpoint open, and --portt 8091 started
    // on 8082).
    struct IntFlag {
        const char* flag;
        std::optional<int> RegistryServerArgs::*field;
        int min;
        int max;
    };
    static const IntFlag kIntFlags[] = {
        {"--port", &RegistryServerArgs::port, 1, 65535},
        {"-p", &RegistryServerArgs::port, 1, 65535},
        {"--ftp-port", &RegistryServerArgs::ftp_port, 1, 65535},
        {"--ftp-pasv-min", &RegistryServerArgs::ftp_pasv_min, 1, 65535},
        {"--ftp-pasv-max", &RegistryServerArgs::ftp_pasv_max, 1, 65535},
        {"--ftp-ttl", &RegistryServerArgs::ftp_ttl_minutes, 1, kMaxInt},
        {"--ftp-max-sessions", &RegistryServerArgs::ftp_max_sessions, 0, kMaxInt},  // 0 = unlimited
        {"--claim-lease-minutes", &RegistryServerArgs::claim_lease_minutes, 0, kMaxInt},  // 0 = never expire
        {"--max-upload-mb", &RegistryServerArgs::max_upload_mb, 1, kMaxInt},
        {"--max-concurrent-fetches", &RegistryServerArgs::max_concurrent_fetches, 1, kMaxInt},
    };
    static const std::set<std::string> kStringValueFlags = {"--config",     "--data-dir", "--ftp-ip",  "--ftp-secret",
                                                            "--ftp-cert",   "--ftp-key",  "--admin-enabled"};

    // TD-489: --help anywhere wins; it used to count only if no bad argument came before it.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const IntFlag* int_flag = nullptr;
        for (const auto& f : kIntFlags) {
            if (arg == f.flag) int_flag = &f;
        }
        if ((int_flag || kStringValueFlags.count(arg) != 0) &&
            (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
            return fail("Missing value for " + arg);
        }
        if (int_flag) {
            const std::string value = argv[++i];
            const auto v = parse_strict_int(value);
            if (!v || *v < int_flag->min || *v > int_flag->max) {
                return fail("Invalid value for " + arg + ": '" + value + "' (expected " + int_range_text(*int_flag) + ")");
            }
            result.*(int_flag->field) = *v;
        } else if (arg == "--config") {
            result.config_path = argv[++i];
        } else if (arg == "--data-dir") {
            result.data_dir = argv[++i];
        } else if (arg == "--ftp-enabled") {
            result.ftp_enabled = true;
        } else if (arg == "--ftp-ip") {
            result.ftp_advertise_ip = argv[++i];
        } else if (arg == "--ftp-secret") {
            result.ftp_server_secret = argv[++i];
        } else if (arg == "--ftps") {
            result.ftps_enabled = true;
        } else if (arg == "--ftp-cert") {
            result.ftp_cert_file = argv[++i];
        } else if (arg == "--ftp-key") {
            result.ftp_key_file = argv[++i];
        } else if (arg == "--admin-enabled") {
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
    if (result.ftp_pasv_min && result.ftp_pasv_max && *result.ftp_pasv_min > *result.ftp_pasv_max) {
        return fail("--ftp-pasv-min (" + std::to_string(*result.ftp_pasv_min) + ") is above --ftp-pasv-max (" +
                    std::to_string(*result.ftp_pasv_max) + ")");
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

    // TD-488: a saved override must be a whole number in range, like the flag; a bad one is skipped
    // with a warning (stoi took "50abc" as 50 and applied a TTL of 0 or below).
    // The ranges are registry_admin_int_settings(), which PUT /admin/config checks too (TD-306).
    auto override_int = [&](const char* key, int& field) {
        auto it = admin_overrides.find(key);
        if (it == admin_overrides.end()) return;
        const auto& s = *find_admin_int_setting(registry_admin_int_settings(), key);
        const auto v = parse_strict_int(it->second);
        if (v && *v >= s.min && *v <= s.max) {
            field = *v;
        } else {
            cfg.override_warnings.push_back(std::string("Ignoring saved admin setting ") + key + "='" + it->second +
                                            "' (expected " + admin_int_range_text(s) + ")");
        }
    };
    override_int("ftp_token_ttl_minutes", cfg.ftp_ttl_minutes);
    override_int("ftp_max_sessions_per_run", cfg.ftp_max_sessions);

    // This run's explicit CLI flags win over everything, including persisted admin overrides.
    if (args.ftp_ttl_minutes)
        cfg.ftp_ttl_minutes = *args.ftp_ttl_minutes;
    if (args.ftp_max_sessions)
        cfg.ftp_max_sessions = *args.ftp_max_sessions;
    cfg.admin_enabled = args.admin_enabled.value_or(file_config.registry_admin_enabled);  // TD-489
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

    // TD-489: with FTP disabled the listener flags do nothing; they used to be dropped silently.
    if (!cfg.ftp_enabled) {
        const std::pair<bool, const char*> listener_flags[] = {
            {args.ftp_port.has_value(), "--ftp-port"},         {args.ftp_advertise_ip.has_value(), "--ftp-ip"},
            {args.ftp_pasv_min.has_value(), "--ftp-pasv-min"}, {args.ftp_pasv_max.has_value(), "--ftp-pasv-max"},
            {args.ftp_server_secret.has_value(), "--ftp-secret"}, {args.ftps_enabled.has_value(), "--ftps"},
            {args.ftp_cert_file.has_value(), "--ftp-cert"},    {args.ftp_key_file.has_value(), "--ftp-key"},
            {args.ftp_ttl_minutes.has_value(), "--ftp-ttl"},   {args.ftp_max_sessions.has_value(), "--ftp-max-sessions"},
        };
        for (const auto& [given, flag] : listener_flags) {
            if (given) {
                cfg.warnings.push_back(std::string(flag) +
                                       " has no effect: FTP is disabled (pass --ftp-enabled or set FTP_SERVER_ENABLED=true)");
            }
        }
    }

    // TD-488: a PASV range emptied by the file and the flags together (each alone is checked by
    // validate() and the parser) left every data connection failing with 425.
    if (cfg.ftp_pasv_min > cfg.ftp_pasv_max) {
        cfg.errors.push_back("the FTP passive port range is empty: minimum " + std::to_string(cfg.ftp_pasv_min) +
                             " is above maximum " + std::to_string(cfg.ftp_pasv_max));
    }

    if (cfg.ftp_enabled && !cfg.ftps_enabled && !cfg.ftp_advertise_ip.empty() &&
        cfg.ftp_advertise_ip != "127.0.0.1" && cfg.ftp_advertise_ip != "::1" &&
        cfg.ftp_advertise_ip != "localhost") {
        cfg.warnings.push_back("FTP is enabled without FTPS on " + cfg.ftp_advertise_ip +
                               ": token passwords and dataset files cross the network in plaintext "
                               "(set FTPS_ENABLED=true or --ftps)");
    }
    return cfg;
}

RegistryAdminConfigUpdate parse_registry_admin_config(const std::string& body) {
    RegistryAdminConfigUpdate u;
    const auto j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        u.error = "body must be a JSON object";
        return u;
    }
    for (const char* key : {"port", "data_dir", "ftp_server_port", "ftp_pasv_port_min", "ftp_pasv_port_max",
                            "ftps_enabled", "ftp_cert_file", "ftp_key_file"}) {
        if (j.contains(key)) {
            u.error = std::string("'") + key +
                      "' is immutable at runtime; set it via config.registry.conf or the matching --flag and restart";
            return u;
        }
    }
    const std::pair<const char*, std::optional<int>*> fields[] = {
        {"ftp_token_ttl_minutes", &u.ftp_token_ttl_minutes},
        {"ftp_max_sessions_per_run", &u.ftp_max_sessions_per_run}};
    for (const auto& [key, out] : fields) {
        const auto it = j.find(key);
        if (it == j.end()) continue;
        const auto& s = *find_admin_int_setting(registry_admin_int_settings(), key);
        if (!it->is_number_integer() || it->get<long long>() < s.min || it->get<long long>() > s.max) {
            u.error = std::string("'") + key + "' must be " + admin_int_range_text(s);
            u.ftp_token_ttl_minutes.reset();
            u.ftp_max_sessions_per_run.reset();
            return u;
        }
        *out = static_cast<int>(it->get<long long>());
    }
    if (!u.ftp_token_ttl_minutes && !u.ftp_max_sessions_per_run) {
        u.error = "no recognized mutable keys in body (ftp_token_ttl_minutes, ftp_max_sessions_per_run)";
    }
    return u;
}

}  // namespace adai
