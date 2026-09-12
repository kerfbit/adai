// @adai-status: beta        (capped by TD-035 — shipped as mns_cli, no dedicated test)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-10

/**
 * mns_cli — Command-line interface for the ADAI Model Name Service
 *
 * Manages model names, lifecycle states, and role promotions via the
 * mns_server HTTP API.
 *
 * Usage:  mns_cli [--url URL] <command> [args...]
 *
 * Default server URL: http://localhost:8083
 * Override via --url flag or NAME_SERVICE_URL environment variable / config.conf.
 */

#include <httplib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "Config.hpp"
#include "MnsCliCommands.hpp"
#include "ModelNameClient.hpp"

using adai::HttpMethod;
using adai::MnsCliRequest;
using ParsedUrl = adai::ParsedUrl;
using adai::parse_url;

// ============================================================================
// HTTP helper — issues GET/POST/PUT/DELETE and prints the response
// ============================================================================

static httplib::Client make_http(const ParsedUrl& u) {
    httplib::Client c(u.host, u.port);
    c.set_connection_timeout(5, 0);
    c.set_read_timeout(10, 0);
    return c;
}

static int http_get(const ParsedUrl& u, const std::string& path) {
    auto c = make_http(u);
    auto res = c.Get(path);
    if (!res) {
        std::cerr << "Error: connection to " << u.host << ":" << u.port << " failed\n";
        return 1;
    }
    std::cout << res->body << "\n";
    return (res->status >= 200 && res->status < 300) ? 0 : 1;
}

static int http_post(const ParsedUrl& u, const std::string& path, const std::string& body) {
    auto c = make_http(u);
    auto res = c.Post(path, body, "application/json");
    if (!res) {
        std::cerr << "Error: connection to " << u.host << ":" << u.port << " failed\n";
        return 1;
    }
    std::cout << res->body << "\n";
    return (res->status >= 200 && res->status < 300) ? 0 : 1;
}

static int http_put(const ParsedUrl& u, const std::string& path, const std::string& body) {
    auto c = make_http(u);
    auto res = c.Put(path, body, "application/json");
    if (!res) {
        std::cerr << "Error: connection to " << u.host << ":" << u.port << " failed\n";
        return 1;
    }
    std::cout << res->body << "\n";
    return (res->status >= 200 && res->status < 300) ? 0 : 1;
}

static int http_delete(const ParsedUrl& u, const std::string& path) {
    auto c = make_http(u);
    auto res = c.Delete(path);
    if (!res) {
        std::cerr << "Error: connection to " << u.host << ":" << u.port << " failed\n";
        return 1;
    }
    std::cout << res->body << "\n";
    return (res->status >= 200 && res->status < 300) ? 0 : 1;
}

// ============================================================================
// Dispatch — sends a request already built by one of MnsCliCommands.hpp's build_*_request()
// functions, or prints its usage error and returns 1 without touching the network at all.
// ============================================================================

static int send(const ParsedUrl& u, const MnsCliRequest& req) {
    if (req.error) {
        std::cerr << req.error_message << "\n";
        return 1;
    }
    switch (req.method) {
        case HttpMethod::Get:
            return http_get(u, req.path);
        case HttpMethod::Post:
            return http_post(u, req.path, req.body);
        case HttpMethod::Put:
            return http_put(u, req.path, req.body);
        case HttpMethod::Delete:
            return http_delete(u, req.path);
    }
    return 1;  // unreachable — silences -Wreturn-type for the enum switch above
}

// ============================================================================
// Usage
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "ADAI Model Name Service CLI\n\n"
              << "Usage: " << prog << " [--url URL] [--config PATH] <command> [args...]\n\n"
              << "Global options:\n"
              << "  --url URL              MNS server URL (default: http://localhost:8083)\n"
              << "                         Also settable via NAME_SERVICE_URL env/config key.\n"
              << "  --config PATH          Path to config.mns.conf for URL and arch defaults.\n\n"
              << "Commands:\n"
              << "  list [--state STATE] [--role ROLE] [--limit N]\n"
              << "      List registered models.  Optional filters narrow results.\n\n"
              << "  get <name>\n"
              << "      Show the full record for a model.\n\n"
              << "  register <name> <role> [--d-model N] [--num-heads N] [--d-ff N]\n"
              << "           [--encoder-layers N] [--decoder-layers N] [--max-seq-length N]\n"
              << "           [--run-group GROUP] [--tag key=value ...]\n"
              << "      Register a new model.  Architecture defaults come from config.mns.conf,\n"
              << "      and — unlike --run-group — is immutable after register (changing it would\n"
              << "      break checkpoint compatibility).\n"
              << "      --run-group sets the dataset-registry group this model's trainer should\n"
              << "      use (see DatasetRegistry); omitted/empty means clients fall back to their\n"
              << "      own local RUN_GROUP config / SESSION_DIR-basename derivation. Safe to\n"
              << "      change later with 'update' (see below) — no checkpoint-compatibility\n"
              << "      concern the way architecture has.\n\n"
              << "  update <name> --run-group <value>\n"
              << "      Set/change an already-registered model's run_group. The only per-model\n"
              << "      field with an update-after-register path — everything else (architecture,\n"
              << "      role) is register-time-only.\n\n"
              << "  resolve <name>\n"
              << "      Resolve a model by name (artifact location + state).\n\n"
              << "  set-training <name> [--new-run] [session-key]\n"
              << "      Transition model to \"training\" state. MNS allocates run_id\n"
              << "      (definitive standard); --new-run requests a fresh run (like retrain),\n"
              << "      omitted continues the current run. Allocated run_id is printed in the\n"
              << "      response.\n\n"
              << "  set-candidate <name> <run-id> [--artifact-path PATH] [--artifact-host HOST]\n"
              << "                [--artifact-checksum CHK] [--artifact-format FMT]\n"
              << "                [--summary key=value ...]\n"
              << "      Transition model to \"candidate\" state with artifact.\n\n"
              << "  delete <name>\n"
              << "      Hard-delete a model (only initializing or retired).\n\n"
              << "  roles\n"
              << "      List all roles and their production models.\n\n"
              << "  resolve-role <role>\n"
              << "      Resolve the production model for a role.\n\n"
              << "  promote <role> <model-name>\n"
              << "      Promote a candidate model to production for a role.\n\n"
              << "  health\n"
              << "      Check MNS server health.\n\n"
              << "Examples:\n"
              << "  " << prog << " list\n"
              << "  " << prog << " register my-chatbot-v3 chatbot --d-model 128 --num-heads 4\n"
              << "  " << prog << " set-training my-chatbot-v3\n"
              << "  " << prog
              << " set-candidate my-chatbot-v3 run-42 --artifact-path /opt/adai/models/v3.bin\n"
              << "  " << prog << " promote chatbot my-chatbot-v3\n"
              << "  " << prog << " resolve-role chatbot\n";
}

// ============================================================================
// Command handlers
// ============================================================================

static int cmd_list(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_list_request(args));
}

static int cmd_get(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_get_request(args));
}

static int cmd_register(const ParsedUrl& u, const std::vector<std::string>& args,
                        const adai::ServiceConfig& cfg) {
    return send(u, adai::build_register_request(args, cfg));
}

static int cmd_resolve(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_resolve_request(args));
}

static int cmd_set_training(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_set_training_request(args));
}

static int cmd_set_candidate(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_set_candidate_request(args));
}

static int cmd_delete(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_delete_request(args));
}

static int cmd_roles(const ParsedUrl& u) {
    return send(u, adai::build_roles_request());
}

static int cmd_resolve_role(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_resolve_role_request(args));
}

static int cmd_promote(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_promote_request(args));
}

static int cmd_update_run_group(const ParsedUrl& u, const std::vector<std::string>& args) {
    return send(u, adai::build_update_run_group_request(args));
}

static int cmd_health(const ParsedUrl& u) {
    return send(u, adai::build_health_request());
}


// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string server_url;
    std::string config_path;
    std::vector<std::string> args;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--url" && i + 1 < argc) {
            server_url = argv[++i];
        } else if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            args.push_back(a);
        }
    }

    // Load config for arch defaults and NAME_SERVICE_URL fallback.
    // Discovery: --config > ./config.mns.conf > /etc/adai/config.mns.conf
    // > ./config.conf (legacy) > /etc/adai/config.conf (legacy).
    config_path = adai::ConfigLoader::discover_config_path(config_path, "config.mns.conf");
    adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);

    // URL priority: --url flag > env/config NAME_SERVICE_URL > default
    if (server_url.empty()) {
        server_url = svc_config.name_service_url;
    }
    if (server_url.empty()) {
        server_url = "http://localhost:8083";
    }

    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    ParsedUrl url = parse_url(server_url);
    const std::string command = args[0];
    std::vector<std::string> cmd_args(args.begin() + 1, args.end());

    if (command == "list")
        return cmd_list(url, cmd_args);
    if (command == "get")
        return cmd_get(url, cmd_args);
    if (command == "register")
        return cmd_register(url, cmd_args, svc_config);
    if (command == "update")
        return cmd_update_run_group(url, cmd_args);
    if (command == "resolve")
        return cmd_resolve(url, cmd_args);
    if (command == "set-training")
        return cmd_set_training(url, cmd_args);
    if (command == "set-candidate")
        return cmd_set_candidate(url, cmd_args);
    if (command == "delete")
        return cmd_delete(url, cmd_args);
    if (command == "roles")
        return cmd_roles(url);
    if (command == "resolve-role")
        return cmd_resolve_role(url, cmd_args);
    if (command == "promote")
        return cmd_promote(url, cmd_args);
    if (command == "health")
        return cmd_health(url);

    std::cerr << "Unknown command: " << command << "\n";
    std::cerr << "Run '" << argv[0] << " --help' for usage.\n";
    return 1;
}
