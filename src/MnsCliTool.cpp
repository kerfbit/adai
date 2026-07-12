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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <httplib.h>
#include "Config.hpp"
#include "ModelNameClient.hpp"

// ============================================================================
// URL parsing (lightweight, same approach as ModelNameClient::ParsedUrl)
// ============================================================================

struct ParsedUrl {
    std::string host = "localhost";
    int         port = 8083;
};

static ParsedUrl parse_url(const std::string& raw) {
    ParsedUrl p;
    std::string s = raw;
    if (s.rfind("http://", 0) == 0)  s = s.substr(7);
    if (s.rfind("https://", 0) == 0) s = s.substr(8);
    auto slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    auto colon = s.find(':');
    if (colon != std::string::npos) {
        p.host = s.substr(0, colon);
        try { p.port = std::stoi(s.substr(colon + 1)); } catch (...) {}
    } else {
        p.host = s;
    }
    return p;
}

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
// JSON helpers
// ============================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += static_cast<char>(c); break;
        }
    }
    return out;
}

// ============================================================================
// Usage
// ============================================================================

static void print_usage(const char* prog) {
    std::cout
        << "ADAI Model Name Service CLI\n\n"
        << "Usage: " << prog << " [--url URL] [--config PATH] <command> [args...]\n\n"
        << "Global options:\n"
        << "  --url URL              MNS server URL (default: http://localhost:8083)\n"
        << "                         Also settable via NAME_SERVICE_URL env/config key.\n"
        << "  --config PATH          Path to config.conf for URL and arch defaults.\n\n"
        << "Commands:\n"
        << "  list [--state STATE] [--role ROLE] [--limit N]\n"
        << "      List registered models.  Optional filters narrow results.\n\n"
        << "  get <name>\n"
        << "      Show the full record for a model.\n\n"
        << "  register <name> <role> [--d-model N] [--num-heads N] [--d-ff N]\n"
        << "           [--encoder-layers N] [--decoder-layers N] [--max-seq-length N]\n"
        << "           [--tag key=value ...]\n"
        << "      Register a new model.  Architecture defaults come from config.conf.\n\n"
        << "  resolve <name>\n"
        << "      Resolve a model by name (artifact location + state).\n\n"
        << "  set-training <name> <run-id> [session-key]\n"
        << "      Transition model to \"training\" state.\n\n"
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
        << "  " << prog << " set-training my-chatbot-v3 run-42\n"
        << "  " << prog << " set-candidate my-chatbot-v3 run-42 --artifact-path /opt/adai/models/v3.bin\n"
        << "  " << prog << " promote chatbot my-chatbot-v3\n"
        << "  " << prog << " resolve-role chatbot\n";
}

// ============================================================================
// Command handlers
// ============================================================================

static int cmd_list(const ParsedUrl& u, const std::vector<std::string>& args) {
    std::string state, role;
    int limit = 0;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--state" && i + 1 < args.size()) { state = args[++i]; }
        else if (args[i] == "--role" && i + 1 < args.size()) { role = args[++i]; }
        else if (args[i] == "--limit" && i + 1 < args.size()) { limit = std::stoi(args[++i]); }
    }
    std::string path = "/models";
    std::string sep = "?";
    if (!state.empty()) { path += sep + "state=" + state; sep = "&"; }
    if (!role.empty())  { path += sep + "role=" + role; sep = "&"; }
    if (limit > 0)      { path += sep + "limit=" + std::to_string(limit); }
    return http_get(u, path);
}

static int cmd_get(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.empty()) { std::cerr << "Usage: get <name>\n"; return 1; }
    return http_get(u, "/models/" + args[0]);
}

static int cmd_register(const ParsedUrl& u, const std::vector<std::string>& args,
                         const adai::ServiceConfig& cfg) {
    if (args.size() < 2) {
        std::cerr << "Usage: register <name> <role> [options...]\n";
        return 1;
    }
    const std::string& name = args[0];
    const std::string& role = args[1];

    size_t d_model       = cfg.d_model;
    size_t num_heads     = cfg.num_heads;
    size_t d_ff          = cfg.d_ff;
    size_t enc_layers    = cfg.num_encoder_layers;
    size_t dec_layers    = cfg.num_decoder_layers;
    size_t max_seq       = cfg.max_seq_length;
    std::map<std::string, std::string> tags;

    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--d-model" && i + 1 < args.size())
            d_model = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--num-heads" && i + 1 < args.size())
            num_heads = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--d-ff" && i + 1 < args.size())
            d_ff = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--encoder-layers" && i + 1 < args.size())
            enc_layers = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--decoder-layers" && i + 1 < args.size())
            dec_layers = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--max-seq-length" && i + 1 < args.size())
            max_seq = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--tag" && i + 1 < args.size()) {
            const auto& kv = args[++i];
            auto eq = kv.find('=');
            if (eq != std::string::npos)
                tags[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(name) << "\""
         << ",\"role\":\""       << json_escape(role) << "\""
         << ",\"arch\":{"
            << "\"d_model\":"            << d_model
            << ",\"num_heads\":"         << num_heads
            << ",\"d_ff\":"              << d_ff
            << ",\"num_encoder_layers\":" << enc_layers
            << ",\"num_decoder_layers\":" << dec_layers
            << ",\"max_seq_length\":"    << max_seq
         << "}";
    if (!tags.empty()) {
        body << ",\"tags\":{";
        bool first = true;
        for (const auto& [k, v] : tags) {
            if (!first) body << ',';
            first = false;
            body << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
        }
        body << "}";
    }
    body << "}";

    return http_post(u, "/models", body.str());
}

static int cmd_resolve(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.empty()) { std::cerr << "Usage: resolve <name>\n"; return 1; }
    return http_get(u, "/models/" + args[0] + "/resolve");
}

static int cmd_set_training(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: set-training <name> <run-id> [session-key]\n";
        return 1;
    }
    const std::string& name   = args[0];
    const std::string& run_id = args[1];

    std::ostringstream body;
    body << "{\"state\":\"training\""
         << ",\"run_id\":\"" << json_escape(run_id) << "\"";
    if (args.size() >= 3)
        body << ",\"metrics_session_key\":\"" << json_escape(args[2]) << "\"";
    body << "}";

    return http_put(u, "/models/" + name + "/state", body.str());
}

static int cmd_set_candidate(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: set-candidate <name> <run-id> [options...]\n";
        return 1;
    }
    const std::string& name   = args[0];
    const std::string& run_id = args[1];

    std::string art_path, art_host, art_checksum, art_format = "adai-native";
    std::map<std::string, std::string> summary;

    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--artifact-path" && i + 1 < args.size())
            art_path = args[++i];
        else if (args[i] == "--artifact-host" && i + 1 < args.size())
            art_host = args[++i];
        else if (args[i] == "--artifact-checksum" && i + 1 < args.size())
            art_checksum = args[++i];
        else if (args[i] == "--artifact-format" && i + 1 < args.size())
            art_format = args[++i];
        else if (args[i] == "--summary" && i + 1 < args.size()) {
            const auto& kv = args[++i];
            auto eq = kv.find('=');
            if (eq != std::string::npos)
                summary[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }

    std::ostringstream body;
    body << "{\"state\":\"candidate\""
         << ",\"run_id\":\"" << json_escape(run_id) << "\""
         << ",\"artifact\":{"
            << "\"host\":\""     << json_escape(art_host)     << "\""
            << ",\"path\":\""    << json_escape(art_path)     << "\""
            << ",\"checksum\":\"" << json_escape(art_checksum) << "\""
            << ",\"format\":\""  << json_escape(art_format)   << "\""
         << "}";
    if (!summary.empty()) {
        body << ",\"training_summary\":{";
        bool first = true;
        for (const auto& [k, v] : summary) {
            if (!first) body << ',';
            first = false;
            body << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
        }
        body << "}";
    }
    body << "}";

    return http_put(u, "/models/" + name + "/state", body.str());
}

static int cmd_delete(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.empty()) { std::cerr << "Usage: delete <name>\n"; return 1; }
    return http_delete(u, "/models/" + args[0]);
}

static int cmd_roles(const ParsedUrl& u) {
    return http_get(u, "/roles");
}

static int cmd_resolve_role(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.empty()) { std::cerr << "Usage: resolve-role <role>\n"; return 1; }
    return http_get(u, "/roles/" + args[0] + "/production");
}

static int cmd_promote(const ParsedUrl& u, const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: promote <role> <model-name>\n";
        return 1;
    }
    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(args[1]) << "\"}";
    return http_put(u, "/roles/" + args[0] + "/production", body.str());
}

static int cmd_health(const ParsedUrl& u) {
    return http_get(u, "/health");
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

    // Load config for arch defaults and NAME_SERVICE_URL fallback
    if (config_path.empty()) {
        std::ifstream local_check("config.conf");
        if (local_check.good()) config_path = "config.conf";
    }
    adai::ServiceConfig svc_config =
        config_path.empty() ? adai::ConfigLoader::load()
                            : adai::ConfigLoader::load(config_path);

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

    if (command == "list")           return cmd_list(url, cmd_args);
    if (command == "get")            return cmd_get(url, cmd_args);
    if (command == "register")       return cmd_register(url, cmd_args, svc_config);
    if (command == "resolve")        return cmd_resolve(url, cmd_args);
    if (command == "set-training")   return cmd_set_training(url, cmd_args);
    if (command == "set-candidate")  return cmd_set_candidate(url, cmd_args);
    if (command == "delete")         return cmd_delete(url, cmd_args);
    if (command == "roles")          return cmd_roles(url);
    if (command == "resolve-role")   return cmd_resolve_role(url, cmd_args);
    if (command == "promote")        return cmd_promote(url, cmd_args);
    if (command == "health")         return cmd_health(url);

    std::cerr << "Unknown command: " << command << "\n";
    std::cerr << "Run '" << argv[0] << " --help' for usage.\n";
    return 1;
}
