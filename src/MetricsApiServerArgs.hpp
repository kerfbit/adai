#pragma once

// @adai-status: beta
// @adai-version: 0.5.0
// @adai-reviewed: 2026-11-10

// TD-035: metrics_api_server's config-precedence resolution (file < persisted admin overrides <
// CLI flags) and argv parsing, pulled out of TrainingMetricsAPIServer.cpp so it's testable
// without starting a real MetricsSessionRegistry/HTTP server — see metrics_api_server_args_test.cpp.
// TrainingMetricsAPI's own request-handling is already covered by
// TrainingMetricsAPIRoutesTests/TrainingMetricsAPILiveTests.

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include "Config.hpp"

namespace adai {

// TD-439: whether this build has the PostgreSQL metrics backend (ENABLE_POSTGRES_METRICS, which
// defines ADAI_ENABLE_POSTGRES for everything linking adai_core).
#ifdef ADAI_ENABLE_POSTGRES
inline constexpr bool kMetricsPostgresBuilt = true;
#else
inline constexpr bool kMetricsPostgresBuilt = false;
#endif

struct MetricsApiServerConfig {
    int port = 8081;
    bool enable_persistence = true;
    bool enable_prometheus = false;
    bool allow_control = true;
    std::string cors_allowed_origins;  // TD-526: METRICS_API_CORS_ALLOWED_ORIGINS / --cors-allowed-origins
    std::string metrics_file = "training_sessions/metrics.jsonl";
    std::string summary_file = "training_sessions/metrics_summary.json";
    std::string prometheus_file = "training_sessions/metrics.prom";
    int persist_every_samples = 100;
    int persist_every_seconds = 30;
    int max_records_in_memory = 10000;
    int max_records_on_disk = 100000;
    int retention_days = 90;         // TD-440: METRICS_RETENTION_DAYS / --retention-days
    int metrics_file_max_mb = 100;   // TD-440: METRICS_FILE_MAX_MB / --metrics-file-max-mb
    int metrics_file_backups = 3;    // TD-440: METRICS_FILE_BACKUPS / --metrics-file-backups
    size_t max_live_sessions = 16;
    int completed_ttl_seconds = 3600;
    int sweep_interval_seconds = 60;
    int staleness_threshold_seconds = 60;
    std::string name_service_url = "http://localhost:8083";
    std::string storage_backend = "sqlite+file";
    std::string db_path = "training_sessions/metrics.db";
    std::string db_url;
    int db_pool_size = 4;
};

// Seeds a MetricsApiServerConfig from an already-loaded ServiceConfig (config.metrics.conf) —
// the lowest-precedence layer; persisted admin overrides and CLI flags are applied on top by
// the two functions below.
MetricsApiServerConfig seed_metrics_server_config_from_file(const ServiceConfig& file_config);

// Applies persisted admin overrides (see CLAUDE.md "Daemon admin config API") on top of an
// already-seeded config, in place. Only the documented admin-mutable subset is consulted. A numeric
// value that isn't a whole number in the setting's range is skipped with a warning (TD-438),
// leaving that field at whatever it already was.
void apply_metrics_admin_overrides(MetricsApiServerConfig& config,
                                   const std::map<std::string, std::string>& overrides);

struct MetricsApiServerArgsResult {
    bool help = false;
    bool error = false;
    std::string error_message;
    // TD-439: the admin-store keys whose flag this run passed, so main() can warn when one replaces a
    // saved PUT /admin/config value (flags win, so a flag in ExecStart undoes the PUT on every restart).
    std::set<std::string> admin_keys_from_flags;
};

// TD-439: the backends `with_postgres` builds can use, in metrics_storage_backend_names() order.
std::vector<std::string> supported_metrics_storage_backends(bool with_postgres = kMetricsPostgresBuilt);

// TD-439: nullopt when this build can use `backend`, else the startup error. An unknown or
// unsupported name used to run the server file-only while it printed the name as its backend.
std::optional<std::string> check_metrics_storage_backend(const std::string& backend,
                                                         bool with_postgres = kMetricsPostgresBuilt);

// TD-439: where daemon_config.db lives — the directory of the final db_path (after CLI flags), or "."
// when db_path has none. It used to be the file-configured db_path's directory, so --db-path under the
// installer's unit (ProtectSystem=strict, only the metrics dir writable) left it read-only.
std::string metrics_admin_store_dir(const MetricsApiServerConfig& config);

// TD-439: one warning per saved admin setting that a flag in `admin_keys_from_flags` replaced with a
// different value; `config` is the final, flag-applied config.
std::vector<std::string> flag_overridden_admin_settings(const MetricsApiServerConfig& config,
                                                       const std::map<std::string, std::string>& overrides,
                                                       const std::set<std::string>& admin_keys_from_flags);

// Applies CLI flags on top of an already-seeded/overridden config, in place — the
// highest-precedence layer. --config is recognized (and its value skipped) but not applied here;
// its path must already have been extracted and used to load file_config before this runs.
// .error is set for an unknown option, a flag missing its value, a number that isn't a whole
// number in the setting's range (TD-438; e.g. --port must be 1-65535), or a --storage-backend that
// isn't one of metrics_storage_backend_names() (TD-439; matched in any case, stored lower-case).
MetricsApiServerArgsResult parse_metrics_api_server_args(int argc, char* argv[],
                                                         MetricsApiServerConfig& config);

}  // namespace adai
