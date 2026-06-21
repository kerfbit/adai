/**
 * registry_server — Distributed dataset queue coordination daemon (TD-028 Phase 9)
 *                   + FTP dataset delivery server (Phase 10)
 *
 * Provides a shared pending-file queue and data registry for multi-machine
 * training pools.  Each training run calls acquire (POST /registry/<group>/acquire)
 * to atomically claim a disjoint subset of pending files, trains independently,
 * then calls trained (POST /registry/<group>/trained) to commit results.
 *
 * Phase 10 extension: when --ftp-enabled is passed (or FTP_ENABLED=1 env var),
 * the /acquire response is extended with per-file FTP tokens and an embedded
 * FtpDataServer is started on --ftp-port (default 2121).  Trainers on remote
 * machines use these credentials to download their files before training.
 *
 * State is persisted to flat files in --data-dir/<group>/ using the same format
 * as LocalTransport, so manual inspection and recovery are always possible.
 *
 * Endpoints:
 *   GET  /registry/<group>/queue     — all pending entries with run assignments
 *   POST /registry/<group>/acquire   — atomically claim up to N files for run_id
 *   POST /registry/<group>/release   — return reserved files to the unassigned pool
 *   POST /registry/<group>/trained   — mark files trained and remove from queue
 *   GET  /registry/<group>/registry  — full data registry
 *   GET  /registry/<group>/runs      — files currently assigned per run_id
 *   GET  /health                     — liveness check
 */

#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <httplib.h>
#include "FtpDataServer.hpp"
#include "Logger.hpp"
#include "RegistryTransport.hpp"

namespace fs = std::filesystem;
using adai::Logger;

// ============================================================================
// Globals
// ============================================================================

static std::atomic<bool> shutdown_requested{false};
static httplib::Server*  g_server = nullptr;

static void signal_handler(int) {
    shutdown_requested.store(true);
    if (g_server) g_server->stop();
}

// ============================================================================
// Per-group state
// ============================================================================

struct GroupState {
    std::mutex                   mtx;
    std::unique_ptr<LocalTransport> transport;
};

static std::mutex                              groups_mtx;
static std::map<std::string, GroupState>       groups;
static std::string                             data_dir = "registry_sessions";

static GroupState& get_group(const std::string& group) {
    std::lock_guard<std::mutex> lock(groups_mtx);
    auto& gs = groups[group];
    if (!gs.transport) {
        const std::string dir = data_dir + "/" + group;
        fs::create_directories(dir);
        gs.transport = std::make_unique<LocalTransport>(
            dir + "/data_registry.txt",
            dir + "/pending_files.txt");
    }
    return gs;
}

// ============================================================================
// FTP server (Phase 10) — optional, started when --ftp-enabled is passed
// ============================================================================

static std::unique_ptr<FtpDataServer> g_ftp_server;
static bool        ftp_enabled           = false;
static int         ftp_port              = 2121;
static int         ftp_pasv_min          = 50000;
static int         ftp_pasv_max          = 50099;
static int         ftp_token_ttl_min     = 30;
static std::string ftp_advertise_ip;        // set from --ftp-ip; falls back to empty
// Phase 3: security hardening options
static std::string ftp_server_secret;       // HMAC key (empty = random passwords)
static bool        ftps_enabled           = false;  // FTPS (explicit TLS via AUTH TLS)
static std::string ftp_cert_file;           // PEM cert (empty = self-signed)
static std::string ftp_key_file;            // PEM key  (empty = self-signed)
static int         ftp_max_sessions        = 4;     // max concurrent sessions per run_id

// ============================================================================
// Minimal JSON helpers
// ============================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string json_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return {};
    const auto start = pos + needle.size();
    const auto end   = body.find('"', start);
    if (end == std::string::npos) return {};
    return body.substr(start, end - start);
}

static int json_int(const std::string& body, const std::string& key, int def = 0) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return def;
    try { return std::stoi(body.substr(pos + needle.size())); }
    catch (...) { return def; }
}

static std::vector<std::string> json_string_array(const std::string& body,
                                                   const std::string& key) {
    std::vector<std::string> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        cur = body.find('"', cur);
        if (cur == std::string::npos) break;
        ++cur;
        const auto end = body.find('"', cur);
        if (end == std::string::npos) break;
        result.push_back(body.substr(cur, end - cur));
        cur = end + 1;
        const auto next = body.find_first_of(",]", cur);
        if (next == std::string::npos || body[next] == ']') break;
        cur = next + 1;
    }
    return result;
}

static std::vector<int> json_int_array(const std::string& body, const std::string& key) {
    std::vector<int> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos) return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        const auto end = body.find_first_of(",]", cur);
        if (end == std::string::npos) break;
        try { result.push_back(std::stoi(body.substr(cur, end - cur))); }
        catch (...) { result.push_back(0); }
        if (body[end] == ']') break;
        cur = end + 1;
    }
    return result;
}

// ============================================================================
// Route handlers
// ============================================================================

// GET /registry/<group>/queue
static void handle_queue(const httplib::Request& req, httplib::Response& res,
                          const std::string& group) {
    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    std::ostringstream json;
    json << "{\"entries\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i) json << ',';
        json << "{\"path\":\"" << json_escape(entries[i].path) << "\","
             << "\"run_id\":\"" << json_escape(entries[i].run_id) << "\"}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
}

// POST /registry/<group>/acquire  {"run_id":"...","max_files":N}
//
// Phase 10: when ftp_enabled the response is extended with per-file FTP tokens.
// Legacy format ("acquired":[...]) is used when ftp_enabled is false so that
// old RemoteTransport clients continue to work.
static void handle_acquire(const httplib::Request& req, httplib::Response& res,
                            const std::string& group) {
    const std::string run_id   = json_string(req.body, "run_id");
    const int         max_files = json_int(req.body, "max_files", 0);

    if (run_id.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"run_id required\"}", "application/json");
        return;
    }

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    const int limit = (max_files > 0) ? max_files : static_cast<int>(entries.size());
    std::vector<std::string> acquired;
    for (auto& e : entries) {
        if (e.run_id.empty() && static_cast<int>(acquired.size()) < limit) {
            e.run_id = run_id;
            acquired.push_back(e.path);
        }
    }

    if (!acquired.empty()) {
        gs.transport->save_pending(entries);
    }

    if (!ftp_enabled || !g_ftp_server) {
        // Legacy response format
        std::ostringstream json;
        json << "{\"acquired\":[";
        for (std::size_t i = 0; i < acquired.size(); ++i) {
            if (i) json << ',';
            json << '"' << json_escape(acquired[i]) << '"';
        }
        json << "]}";
        res.set_content(json.str(), "application/json");
        Logger::info("[{}] acquire: run='{}' claimed {} files", group, run_id, acquired.size());
        return;
    }

    // Phase 10: extended response with per-file FTP tokens
    // ftp_path is the registry path made relative to data_dir.
    const fs::path data_root(data_dir);
    std::ostringstream json;
    json << "{\"run_id\":\"" << json_escape(run_id) << "\","
         << "\"ftp_server_host\":\"" << json_escape(ftp_advertise_ip) << "\","
         << "\"ftp_server_port\":" << ftp_port << ","
         << "\"ftps_enabled\":" << (ftps_enabled ? "true" : "false") << ","
         << "\"files\":[";

    for (std::size_t i = 0; i < acquired.size(); ++i) {
        if (i) json << ',';

        const fs::path file_path(acquired[i]);
        std::string ftp_path;
        std::size_t size_bytes = 0;
        std::string checksum;

        // Compute ftp_path relative to data_dir
        try {
            ftp_path = file_path.lexically_relative(data_root).string();
        } catch (...) {
            ftp_path = file_path.filename().string();
        }
        // Normalise path separators to '/'
        std::replace(ftp_path.begin(), ftp_path.end(), '\\', '/');

        // File metadata for the token
        if (fs::exists(file_path) && fs::is_regular_file(file_path)) {
            size_bytes = static_cast<std::size_t>(fs::file_size(file_path));
            const auto ftime = fs::last_write_time(file_path);
            std::ostringstream cs;
            cs << size_bytes << "_" << static_cast<long long>(ftime.time_since_epoch().count());
            checksum = cs.str();
        }

        // Mint per-file FTP token (Phase 3: run_id is first param for audit/rate-limit)
        const auto tok = g_ftp_server->issue_token(run_id, ftp_path, ftp_token_ttl_min);

        json << "{"
             << "\"registry_path\":\"" << json_escape(acquired[i]) << "\","
             << "\"ftp_path\":\""      << json_escape(ftp_path)    << "\","
             << "\"ftp_username\":\""  << json_escape(tok.username) << "\","
             << "\"ftp_password\":\""  << json_escape(tok.password) << "\","
             << "\"checksum\":\""      << json_escape(checksum)     << "\","
             << "\"size_bytes\":"      << size_bytes << ","
             << "\"token_expires_utc\":\"" << json_escape(tok.token_expires_utc) << "\""
             << "}";
    }

    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] acquire: run='{}' claimed {} files (with FTP tokens)",
                 group, run_id, acquired.size());
}

// POST /registry/<group>/release  {"run_id":"...","files":[...]}
static void handle_release(const httplib::Request& req, httplib::Response& res,
                            const std::string& group) {
    const std::string          run_id = json_string(req.body, "run_id");
    const std::vector<std::string> files  = json_string_array(req.body, "files");

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    const std::set<std::string> to_release(files.begin(), files.end());
    int released = 0;
    for (auto& e : entries) {
        if ((run_id.empty() || e.run_id == run_id) && to_release.count(e.path)) {
            e.run_id.clear();
            ++released;
        }
    }

    gs.transport->save_pending(entries);

    std::ostringstream json;
    json << "{\"released\":" << released << "}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] release: run='{}' released {} files", group, run_id, released);
}

// POST /registry/<group>/trained  {"run_id":"...","files":[...],"samples":[...],"model_id":"..."}
static void handle_trained(const httplib::Request& req, httplib::Response& res,
                            const std::string& group) {
    const std::string             run_id   = json_string(req.body, "run_id");
    const std::string             model_id = json_string(req.body, "model_id");
    const std::vector<std::string> files   = json_string_array(req.body, "files");
    const std::vector<int>         samples = json_int_array(req.body, "samples");

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    // Build new DataVersion entries
    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);
    std::set<std::string> existing;
    for (const auto& dv : reg) existing.insert(dv.data_file);

    int trained = 0;
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (!existing.count(files[i])) {
            DataVersion dv;
            dv.data_file   = files[i];
            // Server has no filesystem access to compute a real checksum;
            // use a placeholder so the space-separated flat-file format
            // keeps its column alignment when loaded back by LocalTransport.
            dv.checksum    = "REMOTE";
            dv.num_samples = (i < samples.size()) ? samples[i] : 0;
            dv.trained     = true;
            dv.model_id    = model_id;
            reg.push_back(std::move(dv));
            ++trained;
        }
    }
    gs.transport->save_registry(reg);

    // Remove trained files from pending
    std::vector<PendingEntry> pending;
    gs.transport->load_pending(pending);
    const std::set<std::string> trained_set(files.begin(), files.end());
    pending.erase(std::remove_if(pending.begin(), pending.end(), [&](const PendingEntry& e) {
        return trained_set.count(e.path) &&
               (run_id.empty() || e.run_id == run_id);
    }), pending.end());
    gs.transport->save_pending(pending);

    std::ostringstream json;
    json << "{\"trained\":" << trained << "}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] trained: run='{}' committed {} new files", group, run_id, trained);
}

// GET /registry/<group>/registry
static void handle_registry(const httplib::Request& req, httplib::Response& res,
                             const std::string& group) {
    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);

    std::ostringstream json;
    json << "{\"entries\":[";
    for (std::size_t i = 0; i < reg.size(); ++i) {
        if (i) json << ',';
        json << "{\"data_file\":\"" << json_escape(reg[i].data_file) << "\","
             << "\"checksum\":\""   << json_escape(reg[i].checksum)  << "\","
             << "\"num_samples\":"  << reg[i].num_samples << ","
             << "\"trained\":"      << (reg[i].trained ? "true" : "false") << "}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
}

// GET /registry/<group>/runs
static void handle_runs(const httplib::Request& req, httplib::Response& res,
                         const std::string& group) {
    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    std::map<std::string, std::vector<std::string>> by_run;
    for (const auto& e : entries) {
        if (!e.run_id.empty()) {
            by_run[e.run_id].push_back(e.path);
        }
    }

    std::ostringstream json;
    json << "{\"runs\":{";
    bool first_run = true;
    for (const auto& [run, files] : by_run) {
        if (!first_run) json << ',';
        first_run = false;
        json << '"' << json_escape(run) << "\":[";
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i) json << ',';
            json << '"' << json_escape(files[i]) << '"';
        }
        json << ']';
    }
    json << "}}";
    res.set_content(json.str(), "application/json");
}

// GET /registry/<group>/history?model_id=<uuid>
static void handle_history(const httplib::Request& req, httplib::Response& res,
                            const std::string& group) {
    const std::string filter_id = req.has_param("model_id") ? req.get_param_value("model_id") : "";

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);

    std::ostringstream json;
    json << "{\"entries\":[";
    bool first = true;
    for (const auto& dv : reg) {
        if (!filter_id.empty() && dv.model_id != filter_id) continue;
        if (!first) json << ',';
        first = false;
        json << "{\"data_file\":\""  << json_escape(dv.data_file) << "\","
             << "\"checksum\":\""    << json_escape(dv.checksum)  << "\","
             << "\"num_samples\":"   << dv.num_samples << ","
             << "\"trained\":"       << (dv.trained ? "true" : "false") << ","
             << "\"model_id\":\""    << json_escape(dv.model_id)  << "\"}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
}

// POST /registry/<group>/pending/add  {"path":"..."}
static void handle_pending_add(const httplib::Request& req, httplib::Response& res,
                                const std::string& group) {
    const std::string path = json_string(req.body, "path");
    if (path.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"path required\"}", "application/json");
        return;
    }

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);
    for (const auto& e : entries) {
        if (e.path == path) {
            res.set_content("{\"added\":false,\"reason\":\"already_pending\"}", "application/json");
            Logger::info("[{}] pending/add: '{}' already queued", group, path);
            return;
        }
    }

    entries.push_back({path, {}});
    gs.transport->save_pending(entries);
    res.set_content("{\"added\":true}", "application/json");
    Logger::info("[{}] pending/add: '{}'", group, path);
}

// ============================================================================
// Usage / main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "ADAI Registry Server — distributed dataset queue coordination\n"
              << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --port PORT           Listening port (default: 8082)\n"
              << "  --data-dir DIR        Root directory for group state (default: registry_sessions)\n"
              << "  --ftp-enabled         Enable embedded FTP data server (Phase 10)\n"
              << "  --ftp-port PORT       FTP control port (default: 2121)\n"
              << "  --ftp-ip IP           IP address advertised in PASV responses\n"
              << "  --ftp-pasv-min PORT   Lower bound of PASV data port range (default: 50000)\n"
              << "  --ftp-pasv-max PORT   Upper bound of PASV data port range (default: 50099)\n"
              << "  --ftp-ttl MINUTES     Per-file token lifetime in minutes (default: 30)\n"
              << "  --ftp-secret SECRET   HMAC-SHA256 key for token signing (default: random)\n"
              << "  --ftps                Enable FTPS (explicit TLS via AUTH TLS)\n"
              << "  --ftp-cert FILE       PEM certificate file for FTPS (default: self-signed)\n"
              << "  --ftp-key FILE        PEM private key file for FTPS (default: self-signed)\n"
              << "  --ftp-max-sessions N  Max concurrent FTP sessions per run_id (default: 4)\n"
              << "  --help                Show this message\n\n"
              << "Endpoints per group:\n"
              << "  GET  /registry/<group>/queue\n"
              << "  POST /registry/<group>/acquire  {\"run_id\":\"...\",\"max_files\":N}\n"
              << "  POST /registry/<group>/release  {\"run_id\":\"...\",\"files\":[...]}\n"
              << "  POST /registry/<group>/trained  {\"run_id\":\"...\",\"files\":[...],\"samples\":[...]}\n"
              << "  GET  /registry/<group>/registry\n"
              << "  GET  /registry/<group>/runs\n"
              << "  GET  /registry/<group>/history[?model_id=<uuid>]\n"
              << "  GET  /health\n";
}

int main(int argc, char* argv[]) {
    int port = 8082;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--ftp-enabled") {
            ftp_enabled = true;
        } else if (arg == "--ftp-port" && i + 1 < argc) {
            ftp_port = std::stoi(argv[++i]);
        } else if (arg == "--ftp-ip" && i + 1 < argc) {
            ftp_advertise_ip = argv[++i];
        } else if (arg == "--ftp-pasv-min" && i + 1 < argc) {
            ftp_pasv_min = std::stoi(argv[++i]);
        } else if (arg == "--ftp-pasv-max" && i + 1 < argc) {
            ftp_pasv_max = std::stoi(argv[++i]);
        } else if (arg == "--ftp-ttl" && i + 1 < argc) {
            ftp_token_ttl_min = std::stoi(argv[++i]);
        } else if (arg == "--ftp-secret" && i + 1 < argc) {
            ftp_server_secret = argv[++i];
        } else if (arg == "--ftps") {
            ftps_enabled = true;
        } else if (arg == "--ftp-cert" && i + 1 < argc) {
            ftp_cert_file = argv[++i];
        } else if (arg == "--ftp-key" && i + 1 < argc) {
            ftp_key_file = argv[++i];
        } else if (arg == "--ftp-max-sessions" && i + 1 < argc) {
            ftp_max_sessions = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    Logger::init(Logger::Level::INFO, "registry_server");

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Start FTP server if enabled (Phase 3: pass security params)
    if (ftp_enabled) {
        g_ftp_server = std::make_unique<FtpDataServer>(
            data_dir, ftp_port, ftp_pasv_min, ftp_pasv_max, ftp_advertise_ip,
            ftp_server_secret, ftps_enabled, ftp_cert_file, ftp_key_file, ftp_max_sessions);
        g_ftp_server->start();
    }

    httplib::Server svr;
    g_server = &svr;

    // Route all /registry/<group>/... paths
    svr.Get(R"(/registry/([^/]+)/queue)",    [](const httplib::Request& r, httplib::Response& res) {
        handle_queue(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/acquire)", [](const httplib::Request& r, httplib::Response& res) {
        handle_acquire(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/release)", [](const httplib::Request& r, httplib::Response& res) {
        handle_release(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/trained)", [](const httplib::Request& r, httplib::Response& res) {
        handle_trained(r, res, r.matches[1]);
    });
    svr.Get(R"(/registry/([^/]+)/registry)", [](const httplib::Request& r, httplib::Response& res) {
        handle_registry(r, res, r.matches[1]);
    });
    svr.Get(R"(/registry/([^/]+)/runs)",     [](const httplib::Request& r, httplib::Response& res) {
        handle_runs(r, res, r.matches[1]);
    });
    svr.Get(R"(/registry/([^/]+)/history)", [](const httplib::Request& r, httplib::Response& res) {
        handle_history(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/pending/add)", [](const httplib::Request& r, httplib::Response& res) {
        handle_pending_add(r, res, r.matches[1]);
    });
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    Logger::info("registry_server listening on port {}", port);
    Logger::info("Data directory: {}", data_dir);
    if (ftp_enabled) {
        Logger::info("FTP data server enabled on port {} (PASV {}–{})",
                     ftp_port, ftp_pasv_min, ftp_pasv_max);
    }

    svr.listen("0.0.0.0", port);

    if (g_ftp_server) g_ftp_server->stop();
    Logger::info("registry_server stopped");
    return 0;
}
