// @adai-status: beta        (shipped as mns_server; no dedicated test file)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07

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
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#include "Config.hpp"
#include "DaemonConfigStore.hpp"
#include "Logger.hpp"
#include "ModelNameService.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<bool> shutdown_requested{false};
static adai::ModelNameService* g_service = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

static void signal_handler(int) {
    shutdown_requested.store(true);
    if (g_service)
        g_service->stop();
}

static void print_usage(const char* prog) {
    std::cout << "ADAI Model Name Service — model identity registry\n"
              << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --config PATH         Path to config.mns.conf (default: ./config.mns.conf or "
                 "/etc/adai/config.mns.conf)\n"
              << "  --port PORT           Listening port (default: 8083)\n"
              << "  --data-dir DIR        Storage directory for SQLite DB (default: name_service)\n"
              << "  --registry-url URL    Registry server URL for dataset proxy (e.g. "
                 "http://localhost:8082)\n"
              << "  --registry-group GRP  Registry group name (default: default)\n"
              << "  --admin-enabled BOOL  Allow PUT /admin/config to mutate settings (default: "
                 "true)\n"
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
              << "  GET    /admin/config\n"
              << "  PUT    /admin/config\n"
              << "  GET    /health\n";
}

static bool parse_bool_flag(const std::string& value, bool default_value) {
    std::string lower = value;
    for (auto& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "true" || lower == "1" || lower == "yes" || lower == "on")
        return true;
    if (lower == "false" || lower == "0" || lower == "no" || lower == "off")
        return false;
    return default_value;
}

int main(int argc, char* argv[]) {
    // Single pass: collect raw CLI values without applying any precedence yet.
    // Precedence (config.mns.conf < persisted admin overrides < this run's CLI
    // flags) is resolved explicitly below — see CLAUDE.md "Daemon admin config API".
    std::optional<std::string> cli_config_path;
    std::optional<int> cli_port;
    std::optional<std::string> cli_data_dir;
    std::optional<std::string> cli_registry_url;
    std::optional<std::string> cli_registry_group;
    std::optional<bool> cli_admin_enabled;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            cli_config_path = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            try {
                cli_port = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid port: " << argv[i] << '\n';
                return 1;
            }
        } else if (arg == "--data-dir" && i + 1 < argc) {
            cli_data_dir = argv[++i];
        } else if (arg == "--registry-url" && i + 1 < argc) {
            cli_registry_url = argv[++i];
        } else if (arg == "--registry-group" && i + 1 < argc) {
            cli_registry_group = argv[++i];
        } else if (arg == "--admin-enabled" && i + 1 < argc) {
            cli_admin_enabled = parse_bool_flag(argv[++i], true);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            print_usage(argv[0]);
            return 1;
        }
    }

    const std::string config_path =
        adai::ConfigLoader::discover_config_path(cli_config_path.value_or(""), "config.mns.conf");
    adai::ServiceConfig file_config = adai::ConfigLoader::load(config_path);

    const int port = cli_port.value_or(file_config.name_service_port);
    const std::string data_dir = cli_data_dir.value_or(file_config.name_service_dir);

    std::string registry_url = file_config.registry_server_url;
    std::string registry_group = file_config.run_group.empty() ? "default" : file_config.run_group;

    // Overlay persisted admin overrides (see ModelNameService::handle_admin_put_config)
    // on top of the file defaults. port/data_dir are never admin-mutable, so they're
    // resolved above without consulting the store.
    try {
        std::filesystem::create_directories(data_dir);
        adai::DaemonConfigStore config_store(data_dir + "/daemon_config.db");
        const auto overrides = config_store.load_all();
        if (auto it = overrides.find("registry_url"); it != overrides.end()) {
            registry_url = it->second;
        }
        if (auto it = overrides.find("registry_group"); it != overrides.end()) {
            registry_group = it->second;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: daemon_config.db unavailable (" << e.what()
                  << "); using file/CLI registry settings\n";
    }

    // This run's explicit CLI flags win over everything, including persisted
    // admin overrides (they don't overwrite the persisted value, just this run).
    if (cli_registry_url)
        registry_url = *cli_registry_url;
    if (cli_registry_group)
        registry_group = *cli_registry_group;
    const bool admin_enabled = cli_admin_enabled.value_or(true);

    adai::Logger::init(adai::Logger::Level::INFO, "mns_server");

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        adai::ModelNameService service(data_dir, port);
        // Always apply, even with an empty registry_url: registry_group can be a
        // persisted admin override independent of registry_url (e.g. only the
        // group changed via PUT /admin/config, url was never set) and must still
        // reach the service — this mirrors ModelNameService's own defaults so it's
        // a no-op when both are unset.
        service.set_registry(registry_url, registry_group);
        if (!registry_url.empty()) {
            adai::Logger::info("Registry proxy: {} (group={})", registry_url, registry_group);
        }
        service.set_admin_enabled(admin_enabled);
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
