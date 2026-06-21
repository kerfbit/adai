/**
 * mns_server — Model Name Service daemon
 *
 * Authoritative registry for model identity across all ADAI processes.
 * Assigns stable UUIDs, tracks lifecycle state, and resolves roles to
 * production artifact locations.
 *
 * Default port: 8083  (metrics=8081, registry=8082, mns=8083)
 *
 * Endpoints:
 *   POST   /models                       — register a new model
 *   GET    /models                       — list models (?state=&role=&limit=)
 *   GET    /models/{name}                — get full ModelRecord
 *   GET    /models/{name}/resolve        — get artifact location
 *   PUT    /models/{name}/state          — lifecycle state transition
 *   DELETE /models/{name}                — hard-delete (retired/initializing only)
 *   GET    /roles                        — list roles and production models
 *   GET    /roles/{role}/production      — resolve production model for a role
 *   PUT    /roles/{role}/production      — promote a candidate to production
 *   GET    /health                       — liveness check
 */

#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>
#include "Logger.hpp"
#include "ModelNameService.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<bool>        shutdown_requested{false};
static adai::ModelNameService*  g_service = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static void signal_handler(int) {
    shutdown_requested.store(true);
    if (g_service) g_service->stop();
}

static void print_usage(const char* prog) {
    std::cout
        << "ADAI Model Name Service — model identity registry\n"
        << "Usage: " << prog << " [OPTIONS]\n\n"
        << "Options:\n"
        << "  --port PORT           Listening port (default: 8083)\n"
        << "  --data-dir DIR        Storage directory for SQLite DB (default: name_service)\n"
        << "  --registry-url URL    Registry server URL for dataset proxy (e.g. http://localhost:8082)\n"
        << "  --registry-group GRP  Registry group name (default: default)\n"
        << "  --help                Show this message\n\n"
        << "Endpoints:\n"
        << "  POST   /models\n"
        << "  GET    /models[?state=&role=&limit=]\n"
        << "  GET    /models/{name}\n"
        << "  GET    /models/{name}/resolve\n"
        << "  GET    /models/{name}/datasets\n"
        << "  PUT    /models/{name}/state\n"
        << "  DELETE /models/{name}\n"
        << "  GET    /roles\n"
        << "  GET    /roles/{role}/production\n"
        << "  PUT    /roles/{role}/production\n"
        << "  GET    /health\n";
}

int main(int argc, char* argv[]) {
    int         port           = 8083;
    std::string data_dir       = "name_service";
    std::string registry_url;
    std::string registry_group = "default";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            try { port = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "Invalid port: " << argv[i] << '\n'; return 1; }
        } else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--registry-url" && i + 1 < argc) {
            registry_url = argv[++i];
        } else if (arg == "--registry-group" && i + 1 < argc) {
            registry_group = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }

    adai::Logger::init(adai::Logger::Level::INFO, "mns_server");

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        adai::ModelNameService service(data_dir, port);
        if (!registry_url.empty()) {
            service.set_registry(registry_url, registry_group);
            adai::Logger::info("Registry proxy: {} (group={})", registry_url, registry_group);
        }
        g_service = &service;

        std::atomic<bool> server_error{false};
        std::thread server_thread([&] {
            if (!service.start()) {
                server_error.store(true);
                shutdown_requested.store(true);
            }
        });

        while (!shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        service.stop();
        server_thread.join();

        if (server_error.load()) {
            adai::Logger::error("mns_server exited with error");
            return 1;
        }
    } catch (const std::exception& e) {
        adai::Logger::error("mns_server fatal: {}", e.what());
        return 1;
    }

    adai::Logger::info("mns_server stopped");
    return 0;
}
