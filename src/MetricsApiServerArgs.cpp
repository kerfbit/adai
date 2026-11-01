// @adai-status: experimental
// @adai-version: 0.4.0
// @adai-reviewed: 2026-11-01

#include "MetricsApiServerArgs.hpp"
#include <algorithm>
#include <optional>
#include <set>
#include <vector>
#include "Logger.hpp"

namespace adai {

MetricsApiServerConfig seed_metrics_server_config_from_file(const ServiceConfig& file_config) {
    MetricsApiServerConfig config;
    config.port = file_config.metrics_api_port;
    config.enable_persistence = file_config.metrics_enable_persistence;
    config.metrics_file = file_config.metrics_file;
    config.summary_file = file_config.metrics_summary_file;
    config.prometheus_file = file_config.metrics_prometheus_file;
    config.persist_every_samples = file_config.metrics_persist_every_samples;
    config.persist_every_seconds = file_config.metrics_persist_every_seconds;
    config.max_records_in_memory = file_config.metrics_max_records_in_memory;
    config.max_records_on_disk = file_config.metrics_max_records_on_disk;
    config.retention_days = file_config.metrics_retention_days;
    config.metrics_file_max_mb = file_config.metrics_file_max_mb;
    config.metrics_file_backups = file_config.metrics_file_backups;
    config.max_live_sessions = file_config.metrics_max_live_sessions;
    config.completed_ttl_seconds = file_config.metrics_completed_ttl_seconds;
    config.sweep_interval_seconds = file_config.metrics_sweep_interval_seconds;
    config.staleness_threshold_seconds = file_config.metrics_staleness_threshold_seconds;
    config.enable_prometheus = file_config.metrics_enable_prometheus;
    config.allow_control = file_config.metrics_api_allow_control;
    config.cors_allowed_origins = file_config.metrics_api_cors_allowed_origins;
    config.name_service_url = file_config.name_service_url;
    config.storage_backend = file_config.metrics_storage_backend;
    config.db_path = file_config.metrics_db_path;
    config.db_url = file_config.metrics_db_url;
    config.db_pool_size = file_config.metrics_db_pool_size;
    return config;
}

namespace {
// TD-438: every numeric setting, with its admin-store key (empty if not admin-mutable), its flag and
// its allowed range. Flags and saved admin overrides both go through this table; they used
// atoi/strtoul/stoull, so --port abc listened on a random port, --port 99999 on 34463, and
// --max-live-sessions -1 (or a saved "-1") removed the session cap. The ranges match
// ConfigLoader::validate() (TD-288); 0 means "off" or "no limit" where it's allowed.
struct NumericSetting {
    const char* admin_key;
    const char* flag;
    int* int_field;
    size_t* size_field;
    long long min;
    long long max;
};

constexpr long long kIntMax = 2147483647LL;

std::vector<NumericSetting> numeric_settings(MetricsApiServerConfig& c) {
    return {
        {"", "--port", &c.port, nullptr, 1, 65535},
        {"persist_every_samples", "--persist-samples", &c.persist_every_samples, nullptr, 0, kIntMax},
        {"persist_every_seconds", "--persist-seconds", &c.persist_every_seconds, nullptr, 0, kIntMax},
        {"max_records_in_memory", "--max-memory-records", &c.max_records_in_memory, nullptr, 1, kIntMax},
        {"max_records_on_disk", "--max-disk-records", &c.max_records_on_disk, nullptr, 0, kIntMax},
        {"retention_days", "--retention-days", &c.retention_days, nullptr, 0, kIntMax},
        {"metrics_file_max_mb", "--metrics-file-max-mb", &c.metrics_file_max_mb, nullptr, 0, kIntMax},
        {"metrics_file_backups", "--metrics-file-backups", &c.metrics_file_backups, nullptr, 0, kIntMax},
        {"max_live_sessions", "--max-live-sessions", nullptr, &c.max_live_sessions, 1, kIntMax},
        {"completed_ttl_seconds", "--completed-ttl-seconds", &c.completed_ttl_seconds, nullptr, 0, kIntMax},
        // 0 turns the sweep off (TD-446); negative values used to as well.
        {"sweep_interval_seconds", "--sweep-interval-seconds", &c.sweep_interval_seconds, nullptr, 0, kIntMax},
        {"staleness_threshold_seconds", "--staleness-threshold-seconds", &c.staleness_threshold_seconds, nullptr, 1,
         kIntMax},
        {"", "--db-pool-size", &c.db_pool_size, nullptr, 1, kIntMax},
    };
}

std::string range_text(const NumericSetting& s) {
    if (s.max == kIntMax) return s.min == 0 ? "a whole number >= 0" : "a whole number >= 1";
    return std::to_string(s.min) + "-" + std::to_string(s.max);
}

// The value if it's a whole number in range, else nullopt.
std::optional<long long> parse_setting(const NumericSetting& s, const std::string& text) {
    const auto v = parse_strict_int(text);
    if (!v || *v < s.min || *v > s.max) return std::nullopt;
    return *v;
}

void store(const NumericSetting& s, long long v) {
    if (s.int_field) *s.int_field = static_cast<int>(v);
    if (s.size_field) *s.size_field = static_cast<size_t>(v);
}
}  // namespace

void apply_metrics_admin_overrides(MetricsApiServerConfig& config,
                                   const std::map<std::string, std::string>& overrides) {
    for (const auto& s : numeric_settings(config)) {
        if (s.admin_key[0] == '\0') continue;
        auto it = overrides.find(s.admin_key);
        if (it == overrides.end()) continue;
        if (auto v = parse_setting(s, it->second)) {
            store(s, *v);
        } else {
            // TD-438: a bad saved value (TD-306 lets PUT /admin/config store one) used to be applied.
            Logger::warn("Ignoring saved admin setting {}='{}' (expected {})", s.admin_key, it->second,
                         range_text(s));
        }
    }
    if (auto it = overrides.find("enable_prometheus"); it != overrides.end()) {
        config.enable_prometheus = (it->second == "true");
    }
}

MetricsApiServerArgsResult parse_metrics_api_server_args(int argc, char* argv[],
                                                         MetricsApiServerConfig& config) {
    MetricsApiServerArgsResult result;
    const auto settings = numeric_settings(config);
    auto fail = [&](const std::string& message) {
        result.error = true;
        result.error_message = message;
        return result;
    };
    // TD-438: a value-taking flag given no value (last, or followed by another "--" flag) says so;
    // it used to be reported as "Unknown option".
    static const std::set<std::string> kStringValueFlags = {
        "--config",      "--metrics-file", "--summary-file",    "--prometheus-file", "--cors-allowed-origins",
        "--name-service-url", "--storage-backend", "--db-path", "--db-url"};

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
        const auto numeric = std::find_if(settings.begin(), settings.end(),
                                          [&](const NumericSetting& s) { return arg == s.flag; });
        const bool takes_value = numeric != settings.end() || kStringValueFlags.count(arg) != 0;
        if (takes_value && (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
            return fail("Missing value for " + arg);
        }
        if (numeric != settings.end()) {
            const std::string value = argv[++i];
            const auto v = parse_setting(*numeric, value);
            if (!v) return fail("Invalid value for " + arg + ": '" + value + "' (expected " + range_text(*numeric) + ")");
            store(*numeric, *v);
        } else if (arg == "--metrics-file") {
            config.metrics_file = argv[++i];
        } else if (arg == "--summary-file") {
            config.summary_file = argv[++i];
        } else if (arg == "--prometheus-file") {
            config.prometheus_file = argv[++i];
        } else if (arg == "--config") {
            ++i;  // handled in a first pass before this runs — see extract_config_path_arg-style
                 // helpers in ChatbotApiServerArgs.hpp for the same pattern.
        } else if (arg == "--no-persistence") {
            config.enable_persistence = false;
        } else if (arg == "--enable-prometheus") {
            config.enable_prometheus = true;
        } else if (arg == "--no-control") {
            config.allow_control = false;
        } else if (arg == "--cors-allowed-origins") {
            config.cors_allowed_origins = argv[++i];
        } else if (arg == "--name-service-url") {
            config.name_service_url = argv[++i];
        } else if (arg == "--no-name-service") {
            config.name_service_url.clear();
        } else if (arg == "--storage-backend") {
            // TODO: See TD-439 in TECHNICAL_DEBT.md - unknown backend accepted; registry falls back to file-only
            config.storage_backend = argv[++i];
        } else if (arg == "--db-path") {
            config.db_path = argv[++i];
        } else if (arg == "--db-url") {
            config.db_url = argv[++i];
        } else {
            return fail("Unknown option: " + arg);
        }
    }

    return result;
}

}  // namespace adai
