// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

#include "MetricsApiServerArgs.hpp"
#include <cstdlib>

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
    config.max_live_sessions = file_config.metrics_max_live_sessions;
    config.completed_ttl_seconds = file_config.metrics_completed_ttl_seconds;
    config.sweep_interval_seconds = file_config.metrics_sweep_interval_seconds;
    config.staleness_threshold_seconds = file_config.metrics_staleness_threshold_seconds;
    config.enable_prometheus = file_config.metrics_enable_prometheus;
    config.allow_control = file_config.metrics_api_allow_control;
    config.name_service_url = file_config.name_service_url;
    config.storage_backend = file_config.metrics_storage_backend;
    config.db_path = file_config.metrics_db_path;
    config.db_url = file_config.metrics_db_url;
    config.db_pool_size = file_config.metrics_db_pool_size;
    return config;
}

void apply_metrics_admin_overrides(MetricsApiServerConfig& config,
                                   const std::map<std::string, std::string>& overrides) {
    auto apply_int = [&](const char* key, int& field) {
        if (auto it = overrides.find(key); it != overrides.end()) {
            try {
                field = std::stoi(it->second);
            } catch (...) {
            }
        }
    };
    auto apply_size_t = [&](const char* key, size_t& field) {
        if (auto it = overrides.find(key); it != overrides.end()) {
            try {
                field = static_cast<size_t>(std::stoull(it->second));
            } catch (...) {
            }
        }
    };
    apply_size_t("max_live_sessions", config.max_live_sessions);
    apply_int("completed_ttl_seconds", config.completed_ttl_seconds);
    apply_int("sweep_interval_seconds", config.sweep_interval_seconds);
    apply_int("persist_every_samples", config.persist_every_samples);
    apply_int("persist_every_seconds", config.persist_every_seconds);
    apply_int("max_records_in_memory", config.max_records_in_memory);
    apply_int("max_records_on_disk", config.max_records_on_disk);
    apply_int("staleness_threshold_seconds", config.staleness_threshold_seconds);
    if (auto it = overrides.find("enable_prometheus"); it != overrides.end()) {
        config.enable_prometheus = (it->second == "true");
    }
}

MetricsApiServerArgsResult parse_metrics_api_server_args(int argc, char* argv[],
                                                         MetricsApiServerConfig& config) {
    MetricsApiServerArgsResult result;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
        if (arg == "--port" && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (arg == "--metrics-file" && i + 1 < argc) {
            config.metrics_file = argv[++i];
        } else if (arg == "--summary-file" && i + 1 < argc) {
            config.summary_file = argv[++i];
        } else if (arg == "--prometheus-file" && i + 1 < argc) {
            config.prometheus_file = argv[++i];
        } else if (arg == "--persist-samples" && i + 1 < argc) {
            config.persist_every_samples = std::atoi(argv[++i]);
        } else if (arg == "--persist-seconds" && i + 1 < argc) {
            config.persist_every_seconds = std::atoi(argv[++i]);
        } else if (arg == "--max-memory-records" && i + 1 < argc) {
            config.max_records_in_memory = std::atoi(argv[++i]);
        } else if (arg == "--max-disk-records" && i + 1 < argc) {
            config.max_records_on_disk = std::atoi(argv[++i]);
        } else if (arg == "--max-live-sessions" && i + 1 < argc) {
            config.max_live_sessions = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--completed-ttl-seconds" && i + 1 < argc) {
            config.completed_ttl_seconds = std::atoi(argv[++i]);
        } else if (arg == "--sweep-interval-seconds" && i + 1 < argc) {
            config.sweep_interval_seconds = std::atoi(argv[++i]);
        } else if (arg == "--staleness-threshold-seconds" && i + 1 < argc) {
            config.staleness_threshold_seconds = std::atoi(argv[++i]);
        } else if (arg == "--config" && i + 1 < argc) {
            ++i;  // handled in a first pass before this runs — see extract_config_path_arg-style
                 // helpers in ChatbotApiServerArgs.hpp for the same pattern.
        } else if (arg == "--no-persistence") {
            config.enable_persistence = false;
        } else if (arg == "--enable-prometheus") {
            config.enable_prometheus = true;
        } else if (arg == "--no-control") {
            config.allow_control = false;
        } else if (arg == "--name-service-url" && i + 1 < argc) {
            config.name_service_url = argv[++i];
        } else if (arg == "--no-name-service") {
            config.name_service_url.clear();
        } else if (arg == "--storage-backend" && i + 1 < argc) {
            config.storage_backend = argv[++i];
        } else if (arg == "--db-path" && i + 1 < argc) {
            config.db_path = argv[++i];
        } else if (arg == "--db-url" && i + 1 < argc) {
            config.db_url = argv[++i];
        } else if (arg == "--db-pool-size" && i + 1 < argc) {
            config.db_pool_size = std::atoi(argv[++i]);
        } else {
            result.error = true;
            result.error_message = "Unknown option: " + arg;
            return result;
        }
    }

    return result;
}

}  // namespace adai
