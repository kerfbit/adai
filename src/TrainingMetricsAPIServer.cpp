#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include "TrainingMetricsAPI.hpp"
#include "MetricsSessionRegistry.hpp"
#include "TrainingMetricsService.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// Global atomic flag for shutdown (async-signal-safe)
static std::atomic<bool> shutdown_requested{false};

// Global pointer for signal handling (only used after signal handler sets flag)
static TrainingMetricsAPI* g_api_server = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

/**
 * @brief Signal handler for graceful shutdown
 *
 * This handler is async-signal-safe and only sets atomic flags.
 * The actual cleanup is performed in the main thread.
 */
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        // Set shutdown flag (atomic operation is async-signal-safe)
        shutdown_requested.store(true);

        // Note: Do NOT stop the server here. It's unsafe to call complex functions
        // from a signal handler. The main loop will detect the flag and stop the server.
    }
}

/**
 * @brief Print usage information
 */
void print_usage(const char* program_name) {
    std::cout << "Training Metrics REST API Server\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --port PORT                  Port number (default: 8081)\n";
    std::cout << "  --metrics-file FILE          Metrics JSONL file path (default: "
                 "training_sessions/metrics.jsonl)\n";
    std::cout << "  --summary-file FILE          Summary JSON file path (default: "
                 "training_sessions/metrics_summary.json)\n";
    std::cout << "  --prometheus-file FILE       Prometheus file path (default: "
                 "training_sessions/metrics.prom)\n";
    std::cout << "  --persist-samples N          Persist every N samples (default: 100)\n";
    std::cout << "  --persist-seconds N          Persist every N seconds (default: 30)\n";
    std::cout << "  --max-memory-records N       Max records in memory (default: 10000)\n";
    std::cout << "  --max-disk-records N         Max records on disk (default: 100000)\n";
    std::cout << "  --max-live-sessions N        Max live metrics sessions (default: 16)\n";
    std::cout << "  --completed-ttl-seconds N    Completed session TTL in seconds (default: 3600)\n";
    std::cout << "  --no-persistence             Disable persistence to disk\n";
    std::cout << "  --enable-prometheus          Enable Prometheus format output\n";
    std::cout << "  --no-control                 Disable control endpoints (flush, clear)\n";
    std::cout << "  --help                       Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "\n";
    std::cout << "    Start server on port 8081 with default settings\n\n";
    std::cout << "  " << program_name << " --port 9090 --persist-samples 50\n";
    std::cout << "    Start on port 9090, persist every 50 samples\n\n";
    std::cout << "Endpoints:\n";
    std::cout << "  GET  /api/metrics/current       - Current training snapshot\n";
    std::cout << "  GET  /api/metrics/summary       - Aggregated metrics summary\n";
    std::cout << "  GET  /api/metrics/history       - Historical records (query params: "
                 "max_records, session_id)\n";
    std::cout << "  GET  /api/sessions              - List tracked sessions\n";
    std::cout << "  GET  /api/metrics/aggregate     - Aggregate live session metrics\n";
    std::cout << "  GET  /api/sessions/{key}/...    - Session-scoped endpoints\n";
    std::cout << "  GET  /api/metrics/prometheus    - Prometheus format metrics\n";
    std::cout << "  GET  /api/metrics/csv           - CSV format metrics\n";
    std::cout << "  GET  /api/session/status        - Session status and progress\n";
    std::cout << "  GET  /api/session/epochs        - Per-epoch metrics\n";
    std::cout << "  POST /api/control/flush         - Force flush to disk\n";
    std::cout << "  POST /api/control/clear         - Clear historical metrics\n";
    std::cout << "  GET  /health                    - Health check\n";
}

/**
 * @brief Configuration structure for command-line parsing
 */
struct ServerConfig {
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
    int sweep_interval_seconds = 60;  // TD-021: background eviction sweep interval
};

/**
 * @brief Parse command-line arguments
 */
bool parse_args(int argc, char** argv, ServerConfig& config) {  // NOLINT(modernize-avoid-c-arrays)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
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
        } else if (arg == "--no-persistence") {
            config.enable_persistence = false;
        } else if (arg == "--enable-prometheus") {
            config.enable_prometheus = true;
        } else if (arg == "--no-control") {
            config.allow_control = false;
        } else {
            std::cerr << "Unknown option: " << arg << '\n';
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================\n";
    std::cout << "  Training Metrics REST API Server\n";
    std::cout << "==================================================\n\n";

    // Parse command-line arguments
    ServerConfig server_config;
    if (!parse_args(argc, argv, server_config)) {
        return 0;  // Help was shown or invalid args
    }

    try {
        // Create metrics service configuration
        MetricsServiceConfig metrics_config;
        metrics_config.enable_persistence = server_config.enable_persistence;
        metrics_config.metrics_file = server_config.metrics_file;
        metrics_config.summary_file = server_config.summary_file;
        metrics_config.prometheus_file = server_config.prometheus_file;
        metrics_config.persist_every_samples = server_config.persist_every_samples;
        metrics_config.persist_every_seconds = server_config.persist_every_seconds;
        metrics_config.max_records_in_memory = server_config.max_records_in_memory;
        metrics_config.max_records_on_disk = server_config.max_records_on_disk;
        metrics_config.enable_prometheus_format = server_config.enable_prometheus;

<<<<<<< HEAD
        // Create registry-backed metrics services
        std::cout << "[1/3] Initializing metrics session registry...\n";
        auto session_registry = std::make_shared<MetricsSessionRegistry>(
            metrics_config, server_config.max_live_sessions, server_config.completed_ttl_seconds,
            server_config.sweep_interval_seconds);
        auto metrics_service = session_registry->create_or_get_session("0-default");
        if (!metrics_service) {
            std::cerr << "Error: Unable to initialize default metrics session\n";
            return 1;
        }
        std::cout << "  ✓ Metrics session registry initialized\n";
        std::cout << "    - Max live sessions: " << server_config.max_live_sessions << "\n";
        std::cout << "    - Completed session TTL: " << server_config.completed_ttl_seconds
                  << " seconds\n";
        std::cout << "    - Sweep interval: " << server_config.sweep_interval_seconds
                  << " seconds\n";
=======
        // Create metrics service
        // TODO: See TECHNICAL_DEBT.md TD-018 - Replace single TrainingMetricsService with
        //   MetricsSessionRegistry; seed "0-default" slot from metrics_config for backwards compat;
        //   inject registry into TrainingMetricsAPI instead of a single service pointer.
        std::cout << "[1/3] Initializing metrics service...\n";
        auto metrics_service = std::make_shared<TrainingMetricsService>(metrics_config);
        std::cout << "  ✓ Metrics service initialized\n";
>>>>>>> ed717615298f1636afc2d8ea1e25ef1ea07c8c6e

        if (server_config.enable_persistence) {
            std::cout << "  ✓ Persistence enabled:\n";
            std::cout << "    - Metrics file: " << server_config.metrics_file << "\n";
            std::cout << "    - Summary file: " << server_config.summary_file << "\n";
            std::cout << "    - Persist every " << server_config.persist_every_samples
                      << " samples or " << server_config.persist_every_seconds << " seconds\n";
        } else {
            std::cout << "  ⚠ Persistence disabled\n";
        }

        if (server_config.enable_prometheus) {
            std::cout << "  ✓ Prometheus format enabled: " << server_config.prometheus_file << "\n";
        }

        std::cout << "\n[2/3] Creating REST API...\n";
        auto api = std::make_unique<TrainingMetricsAPI>(session_registry, server_config.port,
                                                        server_config.allow_control);
        std::cout << "  ✓ API initialized\n";

        if (!server_config.allow_control) {
            std::cout << "  ⚠ Control endpoints disabled (flush, clear)\n";
        }

        // Set up signal handlers
        g_api_server = api.get();
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Start server
        std::cout << "\n[3/3] Starting API server...\n";
        std::cout << "==================================================\n";
        std::cout << "Server starting on http://0.0.0.0:" << server_config.port << "\n";
        std::cout << "Available endpoints:\n";
        std::cout << "  GET  /api/metrics/current       - Current training snapshot\n";
        std::cout << "  GET  /api/metrics/summary       - Aggregated metrics summary\n";
        std::cout << "  GET  /api/metrics/history       - Historical records\n";
        std::cout << "  GET  /api/sessions              - List tracked sessions\n";
        std::cout << "  GET  /api/metrics/aggregate     - Aggregate live sessions\n";
        std::cout << "  GET  /api/sessions/{key}/...    - Session-scoped endpoints\n";
        std::cout << "  GET  /api/metrics/prometheus    - Prometheus format\n";
        std::cout << "  GET  /api/metrics/csv           - CSV format\n";
        std::cout << "  GET  /api/session/status        - Session status\n";
        std::cout << "  GET  /api/session/epochs        - Per-epoch metrics\n";

        if (server_config.allow_control) {
            std::cout << "  POST /api/control/flush         - Force flush to disk\n";
            std::cout << "  POST /api/control/clear         - Clear history\n";
        }

        std::cout << "  GET  /health                    - Health check\n";
        std::cout << "==================================================\n";
        std::cout << "Press Ctrl+C to stop the server\n\n";

        // Start server in a background thread to allow main thread to handle signals
        std::atomic<bool> server_error{false};
        std::thread server_thread([&]() {
            if (!api->start()) {
                std::cerr << "\nFailed to start server on port " << server_config.port << '\n';
                std::cerr << "Port may already be in use. Try a different port with --port" << '\n';
                server_error = true;
                shutdown_requested.store(true);
            }
        });

        // Main service loop - check for shutdown requests
        while (!shutdown_requested.load() && !server_error) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Graceful shutdown sequence
        if (shutdown_requested.load()) {
            std::cout << "\n==================================================\n";
            std::cout << "  Initiating Graceful Shutdown\n";
            std::cout << "==================================================\n";

            std::cout << "[1/2] Stopping API server...\n";
            if (g_api_server) {
                g_api_server->stop();
            }
            if (server_thread.joinable()) {
                server_thread.join();
            }
            std::cout << "  ✓ Server stopped\n";

            std::cout << "[2/2] Flushing metrics to disk...\n";
            metrics_service->flush_to_disk();
            const auto sessions = session_registry->list_sessions();
            for (const auto& summary : sessions) {
                if (summary.key == "0-default") {
                    continue;
                }
                auto service = session_registry->get_session(summary.key);
                if (service.has_value()) {
                    service.value()->flush_to_disk();
                }
            }
            std::cout << "  ✓ Metrics persisted\n";

            std::cout << "\nServer stopped successfully\n";
        }

        if (server_error) {
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
