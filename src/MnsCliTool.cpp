// @adai-status: stable
// @adai-version: 1.3.0
// @adai-reviewed: 2026-11-01

/**
 * mns_cli — Command-line interface for the ADAI Model Name Service
 *
 * Manages model names, lifecycle states, and role promotions via the
 * mns_server HTTP API.
 *
 * Usage:  mns_cli [global options] <command> [args...]
 *
 * Default server URL: http://localhost:8083
 * Override via --url, or NAME_SERVICE_URL in the environment or mns_cli's config file
 * (config.mns.conf by discovery; see main()).
 */

#include <httplib.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "Config.hpp"
#include "MnsCliCommands.hpp"

#ifdef ADAI_METRICS_PUSH_CURL
#include <curl/curl.h>
#endif

using adai::HttpMethod;
using adai::MnsCliRequest;
using ParsedUrl = adai::ParsedUrl;
using adai::parse_url;

namespace {

// How to reach the server: the URL plus TD-451's extra headers and CA bundle.
struct Connection {
    ParsedUrl url;
    std::vector<std::string> headers;  // "Name: value"
    std::string ca_file;               // empty = the system store
};

constexpr long kConnectTimeoutSeconds = 5;
constexpr long kReadTimeoutSeconds = 10;

const char* method_name(HttpMethod m) {
    switch (m) {
        case HttpMethod::Get:
            return "GET";
        case HttpMethod::Post:
            return "POST";
        case HttpMethod::Put:
            return "PUT";
        case HttpMethod::Delete:
            return "DELETE";
    }
    return "GET";
}

// ============================================================================
// HTTP — one request, printing the response body. TD-451: there used to be four copies of this
// (one per method), always plain HTTP, so an https:// URL was sent unencrypted to port 8083. The
// libcurl build (whenever adai_core has libcurl, as for MetricsPushClient, TD-444) speaks https://
// with certificate checks; the httplib fallback refuses https://.
// ============================================================================

#ifdef ADAI_METRICS_PUSH_CURL
size_t collect_body(char* data, size_t size, size_t count, void* out) {
    static_cast<std::string*>(out)->append(data, size * count);
    return size * count;
}

int send_http(const Connection& c, const MnsCliRequest& req) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Error: could not initialise libcurl\n";
        return 1;
    }
    curl_slist* headers = nullptr;
    if (!req.body.empty()) headers = curl_slist_append(headers, "Content-Type: application/json");
    for (const auto& h : c.headers) headers = curl_slist_append(headers, h.c_str());

    const std::string url = c.url.url_for(req.path);
    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method_name(req.method));
    if (req.method == HttpMethod::Post || req.method == HttpMethod::Put) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req.body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(req.body.size()));
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, kConnectTimeoutSeconds + kReadTimeoutSeconds);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, collect_body);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    if (!c.ca_file.empty()) curl_easy_setopt(curl, CURLOPT_CAINFO, c.ca_file.c_str());

    const CURLcode code = curl_easy_perform(curl);
    long status = 0;
    if (code == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (code != CURLE_OK) {
        std::cerr << "Error: request to " << c.url.url_for(req.path) << " failed: " << curl_easy_strerror(code) << "\n";
        return 1;
    }
    std::cout << response << "\n";
    return (status >= 200 && status < 300) ? 0 : 1;
}
#else
int send_http(const Connection& c, const MnsCliRequest& req) {
    if (c.url.https) {
        std::cerr << "Error: this mns_cli was built without libcurl, so it can't use https:// URLs\n";
        return 1;
    }
    std::string host = c.url.host;
    if (host.size() > 2 && host.front() == '[') host = host.substr(1, host.size() - 2);  // [IPv6]
    httplib::Client client(host, c.url.port);
    client.set_connection_timeout(kConnectTimeoutSeconds, 0);
    client.set_read_timeout(kReadTimeoutSeconds, 0);
    httplib::Headers headers;
    for (const auto& h : c.headers) {  // "Name: value", as parse_http_header_list() returns them
        const auto colon = h.find(':');
        const auto value = h.find_first_not_of(" \t", colon + 1);
        headers.emplace(h.substr(0, colon), value == std::string::npos ? "" : h.substr(value));
    }
    const std::string path = c.url.base_path + req.path;
    httplib::Result res;
    switch (req.method) {
        case HttpMethod::Get:
            res = client.Get(path, headers);
            break;
        case HttpMethod::Post:
            res = client.Post(path, headers, req.body, "application/json");
            break;
        case HttpMethod::Put:
            res = client.Put(path, headers, req.body, "application/json");
            break;
        case HttpMethod::Delete:
            res = client.Delete(path, headers);
            break;
    }
    if (!res) {
        std::cerr << "Error: connection to " << c.url.host << ":" << c.url.port << " failed\n";
        return 1;
    }
    std::cout << res->body << "\n";
    return (res->status >= 200 && res->status < 300) ? 0 : 1;
}
#endif

// Sends a request already built by one of MnsCliCommands.hpp's build_*_request() functions, or
// prints its usage error and returns 1 without touching the network at all.
int send(const Connection& c, const MnsCliRequest& req) {
    if (req.error) {
        std::cerr << req.error_message << "\n";
        return 1;
    }
    if (!req.notice.empty()) std::cerr << req.notice << "\n";
    return send_http(c, req);
}

// ============================================================================
// Usage. TD-451: `<command> --help` prints that command's entry (--help used to count only as the
// first word, so `list --help` sent a request).
// ============================================================================

struct CommandHelp {
    const char* name;
    const char* text;
};

const CommandHelp kCommands[] = {
    {"list", R"(  list [--state STATE] [--role ROLE] [--kind KIND] [--limit N]
      List registered models.  Optional filters narrow results.
)"},
    {"get", R"(  get <name>
      Show the full record for a model.
)"},
    {"register", R"(  register <name> <role> [--kind encoder|decoder|world_model|chatbot]
           [--d-model N] [--num-heads N] [--d-ff N] [--num-layers N]
           [--encoder-layers N] [--decoder-layers N] [--max-seq-length N]
           [--encoder NAME] [--decoder NAME]
           [--from-config TRAINER_CONFIG]
           [--sigreg-lambda F] [--sigreg-num-sketches N]
           [--run-group GROUP] [--tag key=value ...]
      Register a new model.  --kind defaults to "chatbot".
      The architecture must be complete: every field as a flag, or the ones not
      given taken from --from-config, the trainer's config file (D_MODEL,
      NUM_HEADS, D_FF, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, MAX_SEQ_LENGTH).
      A missing field is an error; nothing comes from --config. The architecture
      being registered is printed (stderr) before the request is sent.
      For --kind encoder/decoder/world_model: register a standalone entity;
      --num-layers sets that kind's own layer count (num_encoder_layers for
      encoder/world_model, num_decoder_layers for decoder) — the other layer
      count field must stay 0. world_model also accepts --sigreg-lambda/
      --sigreg-num-sketches.
      For --kind chatbot: EITHER legacy inline architecture (--d-model etc, as
      before) OR --encoder NAME --decoder NAME to link to already-registered
      encoder/decoder records (mutually exclusive with the inline architecture
      flags) — the server rejects a dimensionally-incompatible pairing with 409.
      Architecture is immutable after register (changing it would break
      checkpoint compatibility).
      --run-group sets the dataset-registry group this model's trainer should
      use (see DatasetRegistry); omitted/empty means clients fall back to their
      own local RUN_GROUP config / SESSION_DIR-basename derivation. Safe to
      change later with 'update' (see below) — no checkpoint-compatibility
      concern the way architecture has.
)"},
    {"link-world-model", R"(  link-world-model <chatbot-name> --world-model <name>
                   [--inject-every-n-layers N] [--hippocampal-enabled]
                   [--hippocampal-capacity N] [--hippocampal-repetition-alpha F]
                   [--hippocampal-repetition-decay F]
                   [--hippocampal-cross-reference-alpha F]
                   [--hippocampal-association-decay F]
      Attach/replace a chatbot-kind record's world-model connection (pass
      --world-model "" to detach). Only valid on a chatbot-kind record; the
      target must be a registered world_model. Mutable — unlike --encoder/
      --decoder, swapping the world model doesn't break checkpoint compatibility.
      Rejected with 409 if the world model's d_model doesn't match the chatbot's
      own (via its --encoder link, or its inline d_model) whenever
      --inject-every-n-layers > 0.
)"},
    {"update", R"(  update <name> --run-group <value>
      Set/change an already-registered model's run_group. The only per-model
      field with an update-after-register path — everything else (architecture,
      role) is register-time-only.
)"},
    {"resolve", R"(  resolve <name>
      Resolve a model by name (artifact location + state).
)"},
    {"set-training", R"(  set-training <name> [--new-run] [session-key]
      Transition model to "training" state. MNS allocates run_id
      (definitive standard); --new-run requests a fresh run (like retrain),
      omitted continues the current run. Allocated run_id is printed in the
      response.
)"},
    {"set-candidate", R"(  set-candidate <name> <run-id> [--artifact-path PATH] [--artifact-host HOST]
                [--artifact-checksum CHK] [--artifact-format FMT]
                [--summary key=value ...]
      Transition model to "candidate" state with artifact.
)"},
    {"delete", R"(  delete <name>
      Hard-delete a model (only initializing or retired).
)"},
    {"roles", R"(  roles
      List all roles and their production models.
)"},
    {"resolve-role", R"(  resolve-role <role>
      Resolve the production model for a role.
)"},
    {"promote", R"(  promote <role> <model-name>
      Promote a candidate model to production for a role.
)"},
    {"health", R"(  health
      Check MNS server health.
)"},
};

const CommandHelp* find_command(const std::string& name) {
    for (const auto& c : kCommands) {
        if (name == c.name) return &c;
    }
    return nullptr;
}

void print_usage(const char* prog) {
    std::cout << "ADAI Model Name Service CLI\n\n"
              << "Usage: " << prog << " [global options] <command> [args...]\n"
              << "       " << prog << " <command> --help\n\n"
              << "Global options (before the command):\n"
              << "  --url URL              MNS server URL, http:// (default port 8083) or https://\n"
              << "                         (default port 443); a path is kept as a prefix.\n"
              << "                         Default: NAME_SERVICE_URL, else http://localhost:8083.\n"
              << "  --config PATH          mns_cli's own config (default discovery: config.mns.conf)\n"
              << "                         for NAME_SERVICE_URL, NAME_SERVICE_HEADERS,\n"
              << "                         NAME_SERVICE_CA_FILE and register's RUN_GROUP default.\n"
              << "  --header 'Name: value' Extra request header (repeatable), added to\n"
              << "                         NAME_SERVICE_HEADERS (\"Name: value\" pairs separated by\n"
              << "                         ';'), e.g. a Cloudflare Access service token. Prefer the\n"
              << "                         config key for secrets: a flag shows in the process list.\n"
              << "  --ca-file PATH         CA bundle for an https:// server (default:\n"
              << "                         NAME_SERVICE_CA_FILE, else the system store).\n"
              << "  --help, -h             This text.\n\n"
              << "Commands:\n";
    for (const auto& c : kCommands) std::cout << c.text << "\n";
    std::cout << "Examples:\n"
              << "  " << prog << " list\n"
              << "  " << prog << " register my-chatbot-v3 chatbot --from-config /etc/adai/config.trainer.conf\n"
              << "  " << prog << " register my-enc encoder --kind encoder --d-model 256 --num-heads 4 \\\n"
              << "      --d-ff 1024 --num-layers 4 --max-seq-length 512\n"
              << "  " << prog << " set-training my-chatbot-v3\n"
              << "  " << prog << " set-candidate my-chatbot-v3 run-01 --artifact-path /opt/adai/models/v3.bin\n"
              << "  " << prog << " promote chatbot my-chatbot-v3\n"
              << "  " << prog << " --url https://mns.example.com resolve-role chatbot\n";
}

MnsCliRequest build_request(const std::string& command, const std::vector<std::string>& args,
                            const adai::ServiceConfig& cfg) {
    if (command == "list") return adai::build_list_request(args);
    if (command == "get") return adai::build_get_request(args);
    if (command == "register") return adai::build_register_request(args, cfg);
    if (command == "update") return adai::build_update_run_group_request(args);
    if (command == "link-world-model") return adai::build_link_world_model_request(args);
    if (command == "resolve") return adai::build_resolve_request(args);
    if (command == "set-training") return adai::build_set_training_request(args);
    if (command == "set-candidate") return adai::build_set_candidate_request(args);
    if (command == "delete") return adai::build_delete_request(args);
    if (command == "roles") return adai::build_roles_request(args);
    if (command == "resolve-role") return adai::build_resolve_role_request(args);
    if (command == "promote") return adai::build_promote_request(args);
    return adai::build_health_request(args);  // the only command left (main() checked the name)
}

int run(int argc, char* argv[]);

}  // namespace

// ============================================================================
// Main
// ============================================================================

// TD-448: a malformed number (--d-model abc) used to throw out of a builder and abort mns_cli with a
// core dump. The builders now report it as a usage error; anything still thrown exits 2.
int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "mns_cli: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "mns_cli: unexpected error\n";
    }
    return 2;
}

namespace {

int usage_error(const char* prog, const std::string& message) {
    std::cerr << message << "\n";
    std::cerr << "Run '" << prog << " --help' for usage.\n";
    return 1;
}

int run(int argc, char* argv[]) {
    std::string server_url;
    std::string config_path;
    std::vector<std::string> cli_headers;
    std::string ca_file;

    // TD-451: global options end at the command word; everything after it belongs to the command
    // (they used to be taken from anywhere, so no command argument could be "--url").
    int i = 1;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        const bool value_flag = a == "--url" || a == "--config" || a == "--header" || a == "--ca-file";
        if (value_flag && (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
            return usage_error(argv[0], "Missing value for " + a);  // TD-448
        }
        if (a == "--url") {
            server_url = argv[++i];
        } else if (a == "--config") {
            config_path = argv[++i];
        } else if (a == "--header") {
            const auto parsed = adai::parse_http_header_list(argv[++i]);
            if (parsed.size() != 1) {
                return usage_error(argv[0], std::string("Invalid value for --header: '") + argv[i] +
                                                "' (expected one 'Name: value')");
            }
            cli_headers.push_back(parsed.front());
        } else if (a == "--ca-file") {
            ca_file = argv[++i];
        } else if (a.rfind("-", 0) == 0) {
            return usage_error(argv[0], "Unknown global option: " + a);
        } else {
            break;
        }
    }
    if (i >= argc) {
        print_usage(argv[0]);
        return 0;
    }
    const std::string command = argv[i];
    const std::vector<std::string> cmd_args(argv + i + 1, argv + argc);
    const CommandHelp* help = find_command(command);
    if (!help) return usage_error(argv[0], "Unknown command: " + command);
    for (const auto& a : cmd_args) {
        if (a == "--help" || a == "-h") {
            std::cout << "Usage:\n" << help->text;
            return 0;
        }
    }

    // Load config for the NAME_SERVICE_URL fallback, NAME_SERVICE_HEADERS/CA_FILE and register's
    // RUN_GROUP default. Discovery: --config > ./config.mns.conf > /etc/adai/config.mns.conf >
    // ./config.conf (legacy) > /etc/adai/config.conf (legacy). TD-450: register's architecture no
    // longer comes from here (config.mns.conf's 512/8 used to be registered over the trainer's
    // 768/12); see --from-config.
    config_path = adai::ConfigLoader::discover_config_path(config_path, "config.mns.conf");
    adai::ServiceConfig svc_config = adai::ConfigLoader::load(config_path);

    // URL priority: --url flag > env/config NAME_SERVICE_URL > default
    if (server_url.empty()) server_url = svc_config.name_service_url;
    if (server_url.empty()) server_url = "http://localhost:8083";

    Connection conn;
    conn.url = parse_url(server_url);
    if (!conn.url.error.empty()) return usage_error(argv[0], conn.url.error);
    conn.headers = adai::parse_http_header_list(svc_config.name_service_headers);
    conn.headers.insert(conn.headers.end(), cli_headers.begin(), cli_headers.end());
    conn.ca_file = ca_file.empty() ? svc_config.name_service_ca_file : ca_file;

    return send(conn, build_request(command, cmd_args, svc_config));
}

}  // namespace
