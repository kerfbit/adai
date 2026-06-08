/**
 * registry_server — Distributed dataset queue coordination daemon (TD-028 Phase 9)
 *
 * Provides a shared pending-file queue and data registry for multi-machine
 * training pools.  Each training run calls acquire (POST /registry/<group>/acquire)
 * to atomically claim a disjoint subset of pending files, trains independently,
 * then calls trained (POST /registry/<group>/trained) to commit results.
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

    std::ostringstream json;
    json << "{\"acquired\":[";
    for (std::size_t i = 0; i < acquired.size(); ++i) {
        if (i) json << ',';
        json << '"' << json_escape(acquired[i]) << '"';
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] acquire: run='{}' claimed {} files", group, run_id, acquired.size());
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

// POST /registry/<group>/trained  {"run_id":"...","files":[...],"samples":[...]}
static void handle_trained(const httplib::Request& req, httplib::Response& res,
                            const std::string& group) {
    const std::string             run_id  = json_string(req.body, "run_id");
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
            dv.num_samples = (i < samples.size()) ? samples[i] : 0;
            dv.trained     = true;
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

// ============================================================================
// Usage / main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "ADAI Registry Server — distributed dataset queue coordination\n"
              << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --port PORT       Listening port (default: 8082)\n"
              << "  --data-dir DIR    Root directory for group state (default: registry_sessions)\n"
              << "  --help            Show this message\n\n"
              << "Endpoints per group:\n"
              << "  GET  /registry/<group>/queue\n"
              << "  POST /registry/<group>/acquire  {\"run_id\":\"...\",\"max_files\":N}\n"
              << "  POST /registry/<group>/release  {\"run_id\":\"...\",\"files\":[...]}\n"
              << "  POST /registry/<group>/trained  {\"run_id\":\"...\",\"files\":[...],\"samples\":[...]}\n"
              << "  GET  /registry/<group>/registry\n"
              << "  GET  /registry/<group>/runs\n"
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
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    Logger::init(Logger::Level::INFO, "registry_server");

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

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
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    Logger::info("registry_server listening on port {}", port);
    Logger::info("Data directory: {}", data_dir);

    svr.listen("0.0.0.0", port);

    Logger::info("registry_server stopped");
    return 0;
}
