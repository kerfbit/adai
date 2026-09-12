#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

// TD-035: metrics_api_server's config-precedence resolution (file < persisted admin overrides <
// CLI flags) and argv parsing, pulled out of TrainingMetricsAPIServer.cpp so it's testable
// without starting a real MetricsSessionRegistry/HTTP server — see MetricsApiServerArgs_test.cpp.
// TrainingMetricsAPI's own request-handling is already covered by
// TrainingMetricsAPIRoutesTests/TrainingMetricsAPILiveTests.

#include <map>
#include <string>
#include "Config.hpp"

namespace adai {

struct MetricsApiServerConfig {
    int port = 8081;
    bool enable_persistence = true;
    bool enable_prometheus = false;
    bool allow_control = true;
    std::string metrics_file = "training_sessions/metrics.jsonl";
    std::string summary_file = "training_sessions/metrics_summary.json";
    std::string prometheus_file = "training_sessions/metrics.prom";
    int persist_every_samples = 100;
    int persist_every_seconds = 30;
    int max_records_in_memory = 10000;
    int max_records_on_disk = 100000;
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
// already-seeded config, in place. Only the documented admin-mutable subset is consulted — an
// unparseable override value for a numeric field is silently skipped (same as the original
// inline try/catch), leaving that field at whatever it already was.
void apply_metrics_admin_overrides(MetricsApiServerConfig& config,
                                   const std::map<std::string, std::string>& overrides);

struct MetricsApiServerArgsResult {
    bool help = false;
    bool error = false;
    std::string error_message;
};

// Applies CLI flags on top of an already-seeded/overridden config, in place — the
// highest-precedence layer. --config is recognized (and its value skipped) but not applied here;
// its path must already have been extracted and used to load file_config before this runs.
MetricsApiServerArgsResult parse_metrics_api_server_args(int argc, char* argv[],
                                                         MetricsApiServerConfig& config);

}  // namespace adai
