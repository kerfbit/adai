// @adai-status: beta        (capped by TD-035 (no dedicated unit test) and TD-040 (embeds FtpDataServer's unreviewed auth path))
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07

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
 *   POST /registry/<group>/assign    — set model_name on N pending entries, by path,
 *                                       count, or all (Phase 14; count added Phase 16)
 *   POST /registry/<group>/unassign  — clear model_name back to unassigned (Phase 16)
 *   POST /registry/<group>/delete    — purge entries from pending and/or registry by
 *                                       path, optionally unlinking the file (Phase 16)
 *   POST /registry/<group>/trained   — mark files trained and remove from queue
 *   GET  /registry/<group>/registry  — full data registry
 *   GET  /registry/<group>/runs      — files currently assigned per run_id
 *   POST /registry/<group>/fetch/gutenberg   — download a Gutenberg book server-side (Phase 11)
 *   POST /registry/<group>/fetch/huggingface — download a HuggingFace dataset server-side (Phase 11)
 *   POST /registry/<group>/upload?filename=  — upload a local file's bytes server-side (Phase 11)
 *   GET  /health                     — liveness check
 *
 * Phase 11 extension: the three endpoints above make the registry itself
 * responsible for getting dataset bytes onto its own data_dir (instead of
 * expecting the caller to already have them locally), so the existing FTP
 * delivery pipeline always has something real to serve.
 */

#include <httplib.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "Config.hpp"
#include "DaemonConfigStore.hpp"
#include "DataFetcher.hpp"
#include "FtpDataServer.hpp"
#include "Logger.hpp"
#include "RegistryTransport.hpp"

namespace fs = std::filesystem;
using adai::Logger;

// ============================================================================
// Globals
// ============================================================================

static std::atomic<bool> shutdown_requested{false};
static httplib::Server* g_server = nullptr;

static void signal_handler(int) {
    shutdown_requested.store(true);
    if (g_server)
        g_server->stop();
}

// ============================================================================
// Per-group state
// ============================================================================

struct GroupState {
    std::mutex mtx;
    std::unique_ptr<LocalTransport> transport;
};

static std::mutex groups_mtx;
static std::map<std::string, GroupState> groups;
static std::string data_dir = "registry_sessions";

static GroupState& get_group(const std::string& group) {
    std::lock_guard<std::mutex> lock(groups_mtx);
    auto& gs = groups[group];
    if (!gs.transport) {
        const std::string dir = data_dir + "/" + group;
        fs::create_directories(dir);
        gs.transport = std::make_unique<LocalTransport>(dir + "/data_registry.txt",
                                                        dir + "/pending_files.txt");
    }
    return gs;
}

// ============================================================================
// FTP server (Phase 10) — optional, started when --ftp-enabled is passed
// ============================================================================

static std::unique_ptr<FtpDataServer> g_ftp_server;
static bool ftp_enabled = false;
static int ftp_port = 2121;
static int ftp_pasv_min = 50000;
static int ftp_pasv_max = 50099;
// Admin-mutable (PUT /admin/config): read fresh per-acquire by handle_acquire,
// not baked into FtpDataServer at construction, so atomic is enough to make
// live mutation safe — no restart required. Contrast with ftp_max_sessions
// below, which FtpDataServer only reads once, in its constructor.
static std::atomic<int> ftp_token_ttl_min{30};
static std::string ftp_advertise_ip;  // set from --ftp-ip; falls back to empty
// Phase 3: security hardening options
static std::string ftp_server_secret;  // HMAC key (empty = random passwords)
static bool ftps_enabled = false;      // FTPS (explicit TLS via AUTH TLS)
static std::string ftp_cert_file;      // PEM cert (empty = self-signed)
static std::string ftp_key_file;       // PEM key  (empty = self-signed)
// Baked into FtpDataServer at construction (see main()) — PUT /admin/config
// can still record a new value for the *next* restart, but it won't take
// effect on the running FtpDataServer instance.
static int ftp_max_sessions = 4;

static bool admin_enabled = true;
static std::unique_ptr<adai::DaemonConfigStore> g_config_store;

// Session-number counter (Phase 3 run/session numbering): keyed by
// "model_name|run_id" -> current count, persisted separately from
// daemon_config.db since it's business data, not an admin setting.
static std::unique_ptr<adai::DaemonConfigStore> g_session_store;
static std::mutex g_session_store_mtx;

// ============================================================================
// Minimal JSON helpers
// ============================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

static std::string json_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return {};
    const auto start = pos + needle.size();
    const auto end = body.find('"', start);
    if (end == std::string::npos)
        return {};
    return body.substr(start, end - start);
}

static int json_int(const std::string& body, const std::string& key, int def = 0) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    try {
        return std::stoi(body.substr(pos + needle.size()));
    } catch (...) {
        return def;
    }
}

// Bare-literal boolean (true/false, no quotes) — json_string can't parse
// these since it requires a '"' immediately after the colon, and json_int's
// std::stoi would throw on "true"/"false" and silently fall back to @p def
// without ever telling the caller it didn't understand the value.
static bool json_bool(const std::string& body, const std::string& key, bool def = false) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    const auto start = pos + needle.size();
    return body.compare(start, 4, "true") == 0;
}

// Presence check — distinguishes "key not sent" from "sent with its zero/default value".
static bool json_has_key(const std::string& body, const std::string& key) {
    return body.find("\"" + key + "\"") != std::string::npos;
}

static std::vector<std::string> json_string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        // An empty array (or one that has already been fully consumed) closes
        // with ']' before any further '"' — check that first, otherwise an
        // empty array immediately followed by another JSON field (e.g.
        // "paths":[],"count":0) would wrongly treat that field's key as an
        // array element, since the ']' would never be observed.
        const auto quote_pos = body.find('"', cur);
        const auto close_pos = body.find(']', cur);
        if (quote_pos == std::string::npos ||
            (close_pos != std::string::npos && close_pos < quote_pos)) {
            break;
        }
        cur = quote_pos + 1;
        const auto end = body.find('"', cur);
        if (end == std::string::npos)
            break;
        result.push_back(body.substr(cur, end - cur));
        cur = end + 1;
        const auto next = body.find_first_of(",]", cur);
        if (next == std::string::npos || body[next] == ']')
            break;
        cur = next + 1;
    }
    return result;
}

static std::vector<int> json_int_array(const std::string& body, const std::string& key) {
    std::vector<int> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        const auto end = body.find_first_of(",]", cur);
        if (end == std::string::npos)
            break;
        try {
            result.push_back(std::stoi(body.substr(cur, end - cur)));
        } catch (...) {
            result.push_back(0);
        }
        if (body[end] == ']')
            break;
        cur = end + 1;
    }
    return result;
}

// ============================================================================
// Phase 15: dataset metadata helpers (size/checksum/entry count/timestamps)
// ============================================================================

// ISO-8601 UTC timestamp, e.g. "2026-08-02T14:30:00Z".
static std::string utc_now_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

struct LocalFileStat {
    std::size_t size_bytes = 0;
    std::string checksum;
};

// Best-effort size+mtime fingerprint for a registry-local path (same
// convention as FileToken::checksum in handle_acquire, which now delegates
// here too) — not cryptographic, logging/display only. Returns nullopt if
// the path isn't a locally-readable regular file (e.g. a legacy /pending/add
// path that was never actually placed under data_dir).
static std::optional<LocalFileStat> stat_local_file(const std::string& path) {
    const fs::path p(path);
    if (!fs::exists(p) || !fs::is_regular_file(p))
        return std::nullopt;
    LocalFileStat st;
    st.size_bytes = static_cast<std::size_t>(fs::file_size(p));
    const auto ftime = fs::last_write_time(p);
    std::ostringstream cs;
    cs << st.size_bytes << "_" << static_cast<long long>(ftime.time_since_epoch().count());
    st.checksum = cs.str();
    return st;
}

// Best-effort non-empty-line count for a JSONL file; -1 if it can't be opened.
static int count_jsonl_entries(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        return -1;
    int count = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty())
            ++count;
    }
    return count;
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
        if (i)
            json << ',';
        json << "{\"path\":\"" << json_escape(entries[i].path) << "\"," << "\"run_id\":\""
             << json_escape(entries[i].run_id) << "\"," << "\"model_name\":\""
             << json_escape(entries[i].model_name) << "\"," << "\"source\":\""
             << json_escape(entries[i].source) << "\"," << "\"added_utc\":\""
             << json_escape(entries[i].added_utc) << "\"," << "\"size_bytes\":"
             << entries[i].size_bytes << "," << "\"num_entries\":" << entries[i].num_entries
             << "," << "\"checksum\":\"" << json_escape(entries[i].checksum) << "\"}";
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
    const std::string run_id = json_string(req.body, "run_id");
    const std::string model_name = json_string(req.body, "model_name");
    const int max_files = json_int(req.body, "max_files", 0);

    if (run_id.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"run_id required\"}", "application/json");
        return;
    }

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    // Assignment-aware: an entry is claimable iff unassigned or assigned to
    // this exact model_name — never another model's assigned entry. See
    // RegistryTransport::acquire's doc comment (same rule, LocalTransport side).
    const auto eligible = [&model_name](const PendingEntry& e) {
        return e.model_name.empty() || e.model_name == model_name;
    };

    const int limit = (max_files > 0) ? max_files : static_cast<int>(entries.size());
    std::vector<std::string> acquired;
    for (auto& e : entries) {
        // Unclaimed, OR already claimed by this exact run_id — see the
        // matching comment in LocalTransport::acquire() (RegistryTransport.cpp).
        if ((e.run_id.empty() || e.run_id == run_id) && eligible(e) &&
            static_cast<int>(acquired.size()) < limit) {
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
            if (i)
                json << ',';
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
    json << "{\"run_id\":\"" << json_escape(run_id) << "\"," << "\"ftp_server_host\":\""
         << json_escape(ftp_advertise_ip) << "\"," << "\"ftp_server_port\":" << ftp_port << ","
         << "\"ftps_enabled\":" << (ftps_enabled ? "true" : "false") << "," << "\"files\":[";

    for (std::size_t i = 0; i < acquired.size(); ++i) {
        if (i)
            json << ',';

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
        if (const auto stat = stat_local_file(acquired[i])) {
            size_bytes = stat->size_bytes;
            checksum = stat->checksum;
        }

        // Mint per-file FTP token (Phase 3: run_id is first param for audit/rate-limit)
        const auto tok = g_ftp_server->issue_token(run_id, ftp_path, ftp_token_ttl_min);

        json << "{" << "\"registry_path\":\"" << json_escape(acquired[i]) << "\","
             << "\"ftp_path\":\"" << json_escape(ftp_path) << "\"," << "\"ftp_username\":\""
             << json_escape(tok.username) << "\"," << "\"ftp_password\":\""
             << json_escape(tok.password) << "\"," << "\"checksum\":\"" << json_escape(checksum)
             << "\"," << "\"size_bytes\":" << size_bytes << "," << "\"token_expires_utc\":\""
             << json_escape(tok.token_expires_utc) << "\"" << "}";
    }

    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] acquire: run='{}' claimed {} files (with FTP tokens)", group, run_id,
                 acquired.size());
}

// POST /registry/<group>/release  {"run_id":"...","files":[...]}
// Model names flow straight into a cursor-file path (see *_cursor_path below)
// and are also written verbatim into the tab-separated pending-file format
// (LocalTransport), so unlike dataset_id/split this is deliberately stricter —
// no '/' at all, since a model name has no legitimate reason to contain one
// and it doubles as path-traversal protection. Empty is allowed for the fetch
// endpoints (buckets into a shared cursor) but rejected by handle_assign,
// which requires a real, non-empty model name.
static const std::regex kSafeModelName(R"(^[A-Za-z0-9._-]*$)");

static bool is_safe_model_name(const std::string& s) {
    return std::regex_match(s, kSafeModelName);
}

static void handle_release(const httplib::Request& req, httplib::Response& res,
                           const std::string& group) {
    const std::string run_id = json_string(req.body, "run_id");
    const std::vector<std::string> files = json_string_array(req.body, "files");

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

// POST /registry/<group>/assign  {"model_name":"...","paths":[...],"count":N}
//
// Mirrors DatasetRegistry::assign_model()'s semantics (DatasetRegistry.cpp) for
// remote/distributed mode, which previously had no server-side equivalent —
// RemoteTransport::save_pending() is a documented no-op, so dataset_manager's
// `assign` command silently did nothing against a real registry_server before
// this endpoint existed. Three modes, checked in this order:
//   - non-empty paths        — assign exactly those paths (count is ignored)
//   - empty paths, count > 0 — assign the first `count` currently-unassigned
//                              (model_name empty) pending entries, mirroring
//                              handle_acquire's counting-with-limit loop
//   - empty paths, count<=0  — assign every pending entry (original default,
//                              preserved for backward compatibility)
// The response always includes the exact list of paths touched, since the
// count-based mode doesn't otherwise tell the caller which files were picked.
static void handle_assign(const httplib::Request& req, httplib::Response& res,
                          const std::string& group) {
    const std::string model_name = json_string(req.body, "model_name");
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const int count = json_int(req.body, "count", 0);

    if (!is_safe_model_name(model_name) || model_name.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"non-empty model_name matching [A-Za-z0-9._-]+ required\"}",
                        "application/json");
        Logger::warn("[{}] assign: rejected model_name='{}'", group, model_name);
        return;
    }

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    const bool by_paths = !paths.empty();
    const bool by_count = !by_paths && count > 0;
    const bool assign_all = !by_paths && !by_count;
    const std::set<std::string> target_paths(paths.begin(), paths.end());
    std::vector<std::string> assigned_paths;
    for (auto& e : entries) {
        if (by_count && static_cast<int>(assigned_paths.size()) >= count) {
            break;
        }
        const bool matches = by_paths ? target_paths.count(e.path) > 0
                            : by_count ? e.model_name.empty()
                                       : assign_all;
        if (matches) {
            e.model_name = model_name;
            assigned_paths.push_back(e.path);
        }
    }

    if (!assigned_paths.empty()) {
        gs.transport->save_pending(entries);
    }

    std::ostringstream json;
    json << "{\"assigned\":" << assigned_paths.size() << ",\"paths\":[";
    for (std::size_t i = 0; i < assigned_paths.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "\"" << json_escape(assigned_paths[i]) << "\"";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] assign: model='{}' assigned {} file(s) (count={})", group, model_name,
                assigned_paths.size(), count);
}

// POST /registry/<group>/unassign  {"model_name":"...","paths":[...],"force":bool}
//
// Reverses `assign`: clears model_name back to "" (unassigned). Two modes:
//   - paths empty, model_name non-empty — bulk-clear every pending entry
//     currently assigned to that model
//   - paths non-empty — clear only the listed paths; if model_name is also
//     given it acts as an ownership filter (only clears entries currently
//     owned by that model), mirroring handle_release's run_id ownership
//     check; an empty model_name here means "clear regardless of owner"
// Both paths and model_name empty is rejected (400) — there is no implicit
// "unassign everything" the way assign's empty-paths defaults to "assign
// everything", since that would be far too easy to trigger by accident.
// A match whose run_id is non-empty (actively claimed by a training run) is
// skipped unless force:true is passed, to avoid yanking the assignment out
// from under an in-flight run.
static void handle_unassign(const httplib::Request& req, httplib::Response& res,
                            const std::string& group) {
    const std::string model_name = json_string(req.body, "model_name");
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const bool force = json_bool(req.body, "force", false);

    if (paths.empty() && model_name.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"either paths or model_name (or both) required\"}",
                        "application/json");
        Logger::warn("[{}] unassign: rejected — both paths and model_name empty", group);
        return;
    }
    if (!model_name.empty() && !is_safe_model_name(model_name)) {
        res.status = 400;
        res.set_content("{\"error\":\"model_name must match [A-Za-z0-9._-]+\"}",
                        "application/json");
        Logger::warn("[{}] unassign: rejected model_name='{}'", group, model_name);
        return;
    }

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    const bool bulk_by_model = paths.empty();
    const std::set<std::string> target_paths(paths.begin(), paths.end());
    std::vector<std::string> unassigned_paths;
    int skipped = 0;
    for (auto& e : entries) {
        const bool matches = bulk_by_model
                                ? e.model_name == model_name
                                : (target_paths.count(e.path) > 0 &&
                                   (model_name.empty() || e.model_name == model_name));
        if (!matches) {
            continue;
        }
        if (!e.run_id.empty() && !force) {
            ++skipped;
            continue;
        }
        e.model_name.clear();
        unassigned_paths.push_back(e.path);
    }

    if (!unassigned_paths.empty()) {
        gs.transport->save_pending(entries);
    }

    std::ostringstream json;
    json << "{\"unassigned\":" << unassigned_paths.size() << ",\"skipped\":" << skipped
        << ",\"paths\":[";
    for (std::size_t i = 0; i < unassigned_paths.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "\"" << json_escape(unassigned_paths[i]) << "\"";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] unassign: model='{}' unassigned {} file(s), {} skipped (force={})", group,
                model_name, unassigned_paths.size(), skipped, force);
}

// POST /registry/<group>/delete  {"paths":[...],"force":bool,"delete_files":bool}
//
// Permanently purges entries matching `paths` from BOTH the pending queue and
// the trained DataVersion registry (a path may legitimately match either,
// neither, or — not expected in normal operation, since pending->registry is
// a one-way transition via /trained — both). `paths` must be non-empty; there
// is deliberately no "delete everything" bulk mode, since this is destructive
// and spans two stores.
//
// Pending-queue removal respects the same active-run-claim guard as
// /unassign: an entry with a non-empty run_id is left alone (reported
// "skipped_active_run") unless force:true. The registry (trained) half has no
// run_id concept, so it is always purged unconditionally when matched.
//
// delete_files:true additionally unlinks the physical file from disk, but
// ONLY when it resolves to a location inside this group's own data_dir root
// (i.e. files the server itself fetched/uploaded) — arbitrary paths that were
// merely registered via /pending/add (e.g. a trainer-local path) are never
// touched, since the server has no business reaching outside its own managed
// directory tree.
static void handle_delete(const httplib::Request& req, httplib::Response& res,
                          const std::string& group) {
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const bool force = json_bool(req.body, "force", false);
    const bool delete_files = json_bool(req.body, "delete_files", false);

    if (paths.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"paths required\"}", "application/json");
        Logger::warn("[{}] delete: rejected — empty paths", group);
        return;
    }

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> pending;
    gs.transport->load_pending(pending);
    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);

    fs::path group_root;
    bool have_root = false;
    try {
        group_root = fs::weakly_canonical(fs::path(data_dir) / group);
        have_root = true;
    } catch (...) {
        have_root = false;
    }

    bool pending_changed = false;
    bool registry_changed = false;
    int deleted = 0;
    int skipped = 0;
    int not_found = 0;
    std::ostringstream details;
    details << "[";
    bool first_detail = true;

    for (const auto& p : paths) {
        bool pending_blocked = false;
        bool found_anywhere = false;

        auto pit = std::find_if(pending.begin(), pending.end(),
                                [&](const PendingEntry& e) { return e.path == p; });
        if (pit != pending.end()) {
            found_anywhere = true;
            if (!pit->run_id.empty() && !force) {
                pending_blocked = true;
            } else {
                pending.erase(pit);
                pending_changed = true;
            }
        }

        auto rit = std::find_if(reg.begin(), reg.end(),
                                [&](const DataVersion& dv) { return dv.data_file == p; });
        if (rit != reg.end()) {
            found_anywhere = true;
            reg.erase(rit);
            registry_changed = true;
        }

        std::string status;
        bool file_deleted = false;
        if (pending_blocked) {
            status = "skipped_active_run";
            ++skipped;
        } else if (found_anywhere) {
            status = "deleted";
            ++deleted;
            if (delete_files && have_root) {
                try {
                    if (fs::exists(p)) {
                        const auto canon = fs::weakly_canonical(p);
                        const auto rel = canon.lexically_relative(group_root);
                        const bool contained =
                            !rel.empty() && rel.native().compare(0, 2, "..") != 0;
                        if (contained) {
                            std::error_code ec;
                            fs::remove(canon, ec);
                            file_deleted = !ec;
                        } else {
                            Logger::warn(
                                "[{}] delete: refusing to unlink '{}' — outside group data_dir",
                                group, p);
                        }
                    }
                } catch (...) {
                    // Path doesn't resolve (e.g. dangling symlink) — leave file_deleted false.
                }
            }
        } else {
            status = "not_found";
            ++not_found;
        }

        if (!first_detail)
            details << ",";
        first_detail = false;
        details << "{\"path\":\"" << json_escape(p) << "\",\"status\":\"" << status
               << "\",\"file_deleted\":" << (file_deleted ? "true" : "false") << "}";
    }
    details << "]";

    if (pending_changed) {
        gs.transport->save_pending(pending);
    }
    if (registry_changed) {
        gs.transport->save_registry(reg);
    }

    std::ostringstream json;
    json << "{\"deleted\":" << deleted << ",\"skipped\":" << skipped << ",\"not_found\":"
        << not_found << ",\"details\":" << details.str() << "}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] delete: {} deleted, {} skipped, {} not_found (force={}, delete_files={})",
                group, deleted, skipped, not_found, force, delete_files);
}

// POST /registry/<group>/trained  {"run_id":"...","files":[...],"samples":[...],"model_id":"..."}
static void handle_trained(const httplib::Request& req, httplib::Response& res,
                           const std::string& group) {
    const std::string run_id = json_string(req.body, "run_id");
    const std::string model_id = json_string(req.body, "model_id");
    const std::vector<std::string> files = json_string_array(req.body, "files");
    const std::vector<int> samples = json_int_array(req.body, "samples");

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

    // Phase 15: look up each trained path's originating PendingEntry before it's
    // erased below, so source/added_utc survive the pending → trained transition
    // (added_utc then reflects when the file first entered the system, not when
    // training happened to finish).
    std::vector<PendingEntry> pending;
    gs.transport->load_pending(pending);
    std::map<std::string, const PendingEntry*> pending_by_path;
    for (const auto& e : pending)
        pending_by_path[e.path] = &e;

    // Build new DataVersion entries
    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);
    std::set<std::string> existing;
    for (const auto& dv : reg)
        existing.insert(dv.data_file);

    int trained = 0;
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (!existing.count(files[i])) {
            DataVersion dv;
            dv.data_file = files[i];
            dv.num_samples = (i < samples.size()) ? samples[i] : 0;
            dv.trained = true;
            dv.model_id = model_id;

            if (const auto it = pending_by_path.find(files[i]); it != pending_by_path.end()) {
                dv.source = it->second->source;
                dv.added_utc = it->second->added_utc;
            }
            if (dv.added_utc.empty())
                dv.added_utc = utc_now_string();  // no matching pending entry — best available

            // Real checksum when the file is still locally readable (true for
            // anything this registry fetched/uploaded itself); "REMOTE" placeholder
            // otherwise — e.g. a trainer on a different machine reporting a file
            // this registry never staged. Placeholder also keeps the flat-file
            // column non-empty (see LocalTransport::save_registry's "-" convention).
            if (const auto stat = stat_local_file(files[i])) {
                dv.checksum = stat->checksum;
            } else {
                dv.checksum = "REMOTE";
            }

            reg.push_back(std::move(dv));
            ++trained;
        }
    }
    gs.transport->save_registry(reg);

    // Remove trained files from pending
    const std::set<std::string> trained_set(files.begin(), files.end());
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](const PendingEntry& e) {
                                     return trained_set.count(e.path) &&
                                            (run_id.empty() || e.run_id == run_id);
                                 }),
                  pending.end());
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
        if (i)
            json << ',';
        json << "{\"data_file\":\"" << json_escape(reg[i].data_file) << "\"," << "\"checksum\":\""
             << json_escape(reg[i].checksum) << "\"," << "\"num_samples\":" << reg[i].num_samples
             << "," << "\"trained\":" << (reg[i].trained ? "true" : "false") << ","
             << "\"added_utc\":\"" << json_escape(reg[i].added_utc) << "\"," << "\"source\":\""
             << json_escape(reg[i].source) << "\"}";
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
        if (!first_run)
            json << ',';
        first_run = false;
        json << '"' << json_escape(run) << "\":[";
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i)
                json << ',';
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
        if (!filter_id.empty() && dv.model_id != filter_id)
            continue;
        if (!first)
            json << ',';
        first = false;
        json << "{\"data_file\":\"" << json_escape(dv.data_file) << "\"," << "\"checksum\":\""
             << json_escape(dv.checksum) << "\"," << "\"num_samples\":" << dv.num_samples << ","
             << "\"trained\":" << (dv.trained ? "true" : "false") << "," << "\"model_id\":\""
             << json_escape(dv.model_id) << "\"}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
}

// Shared helper: enqueue @p path as a pending entry for @p group unless already
// present. Caller must already hold gs.mtx. Returns true if newly added.
// Phase 15: source identifies how the entry was created ("manual"|"gutenberg"|
// "huggingface"|"upload"); known_num_entries lets a caller that already knows
// the exact count (e.g. a fetch handler's pairs_written) pass it directly
// instead of paying for a redundant re-read. -1 (default) means "count it
// yourself if the file is locally readable, else leave it unknown."
static bool add_pending_path_locked(GroupState& gs, const std::string& path,
                                    const std::string& source, int known_num_entries = -1) {
    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);
    for (const auto& e : entries) {
        if (e.path == path) {
            return false;
        }
    }

    PendingEntry entry;
    entry.path = path;
    entry.source = source;
    entry.added_utc = utc_now_string();
    if (const auto stat = stat_local_file(path)) {
        entry.size_bytes = stat->size_bytes;
        entry.checksum = stat->checksum;
    }
    entry.num_entries = (known_num_entries >= 0) ? known_num_entries : count_jsonl_entries(path);

    entries.push_back(std::move(entry));
    gs.transport->save_pending(entries);
    return true;
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

    if (!add_pending_path_locked(gs, path, "manual")) {
        res.set_content("{\"added\":false,\"reason\":\"already_pending\"}", "application/json");
        Logger::info("[{}] pending/add: '{}' already queued", group, path);
        return;
    }

    res.set_content("{\"added\":true}", "application/json");
    Logger::info("[{}] pending/add: '{}'", group, path);
}

// ============================================================================
// Phase 11: server-side dataset fetch
// ============================================================================

// HuggingFace dataset IDs are "name" or "org/name"; DataFetcher shells out with
// dataset_id interpolated unescaped into curl/python commands, so this endpoint
// is a remote command-injection vector unless the id is strictly allow-listed
// before it ever reaches DataFetcher. Same allow-list applies to `split`.
static const std::regex kSafeHfIdentifier(R"(^[A-Za-z0-9._\-/]+$)");

static bool is_safe_hf_identifier(const std::string& s) {
    return !s.empty() && std::regex_match(s, kSafeHfIdentifier);
}

// Upload filenames must be a bare basename — no directory separators or ".."
// components — so a malicious filename can't write outside data_dir/<group>/uploads/.
static bool is_safe_upload_filename(const std::string& s) {
    if (s.empty() || s == "." || s == "..")
        return false;
    if (s.find('/') != std::string::npos || s.find('\\') != std::string::npos)
        return false;
    return true;
}

// ============================================================================
// Phase 12/13: per-model rotating-slice serving cursors
//
// Tracks, per (group, source, model_name), the next offset to serve from a
// cached full download (HuggingFace dataset or Gutenberg book) — so repeated
// fetch/* calls for the same model advance through the source instead of
// always returning the same slice, while different models each get their own
// independent cursor into the same cached download. Source-specific path
// builders below (hf_cursor_path / gutenberg_cursor_path) share the same
// flat-file storage and read/write helpers.
// ============================================================================

static std::string flatten_for_path(std::string s) {
    std::replace(s.begin(), s.end(), '/', '_');
    return s;
}

static std::string hf_cursor_path(const std::string& group, const std::string& dataset_id,
                                  const std::string& split, const std::string& model_name) {
    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    return data_dir + "/" + group + "/fetch_cursors/hf_" + flatten_for_path(dataset_id) + "__" +
          flatten_for_path(split) + "__" + safe_model + ".txt";
}

static std::string gutenberg_cursor_path(const std::string& group, int book_id,
                                         const std::string& model_name) {
    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    return data_dir + "/" + group + "/fetch_cursors/gutenberg_" + std::to_string(book_id) + "__" +
          safe_model + ".txt";
}

static int read_cursor(const std::string& path) {
    std::ifstream f(path);
    int offset = 0;
    if (f.is_open())
        f >> offset;
    return offset >= 0 ? offset : 0;
}

static void write_cursor(const std::string& path, int offset) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::trunc);
    f << offset;
}

// POST /registry/<group>/fetch/gutenberg  {"book_id":N,"num_pairs":N}
// POST /registry/<group>/fetch/gutenberg  {"book_id":N,"num_pairs":N,"model_name":"..."}
//
// Phase 13: the book is downloaded, cleaned, and split into sentences once
// and cached (DataFetcher::ensure_gutenberg_cached); each call serves the
// next num_pairs QA pairs from a per-(book_id,model_name) cursor rather than
// always the first num_pairs, wrapping around once the book is exhausted —
// same response shape as fetch/huggingface so both sources are served
// identically to the requesting trainer.
static void handle_fetch_gutenberg(const httplib::Request& req, httplib::Response& res,
                                   const std::string& group) {
    const int book_id = json_int(req.body, "book_id", -1);
    const int num_pairs = json_int(req.body, "num_pairs", 500);
    const std::string model_name = json_string(req.body, "model_name");
    if (book_id <= 0) {
        res.status = 400;
        res.set_content("{\"error\":\"positive integer book_id required\"}", "application/json");
        return;
    }
    if (!is_safe_model_name(model_name)) {
        res.status = 400;
        res.set_content("{\"error\":\"model_name must match [A-Za-z0-9._-]*\"}",
                        "application/json");
        Logger::warn("[{}] fetch/gutenberg: rejected unsafe model_name='{}'", group, model_name);
        return;
    }

    const std::string out_dir = data_dir + "/" + group + "/datasets";
    FetcherConfig fcfg;
    fcfg.gutenberg_output_dir = out_dir;
    DataFetcher fetcher(fcfg);

    Logger::info("[{}] fetch/gutenberg: ensuring book #{} is cached...", group, book_id);
    const std::string cached_sentences = fetcher.ensure_gutenberg_cached(book_id);
    if (cached_sentences.empty()) {
        res.status = 502;
        res.set_content("{\"added\":false,\"reason\":\"fetch_failed\"}", "application/json");
        Logger::error("[{}] fetch/gutenberg: failed to download/cache book #{}", group, book_id);
        return;
    }

    const std::string cursor_path = gutenberg_cursor_path(group, book_id, model_name);
    const int offset = read_cursor(cursor_path);

    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    const std::string path = out_dir + "/gutenberg_" + std::to_string(book_id) + "_" + safe_model +
                             "_row" + std::to_string(offset) + "_training.jsonl";

    int next_offset = offset;
    const int pairs_written = DataFetcher::convert_gutenberg_slice(cached_sentences, path, offset,
                                                                    num_pairs, next_offset);
    if (pairs_written <= 0) {
        fs::remove(path);
        res.status = 502;
        res.set_content("{\"added\":false,\"reason\":\"fetch_failed\"}", "application/json");
        Logger::error("[{}] fetch/gutenberg: no pairs extracted from book #{} at sentence {}",
                     group, book_id, offset);
        return;
    }
    write_cursor(cursor_path, next_offset);

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);
    add_pending_path_locked(gs, path, "gutenberg", pairs_written);

    std::ostringstream json;
    json << "{\"added\":true,\"path\":\"" << json_escape(path) << "\",\"served_from_row\":" << offset
         << ",\"next_row\":" << next_offset << ",\"pairs_written\":" << pairs_written << "}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] fetch/gutenberg: book #{} model='{}' sentences[{},{}) → '{}' ({} pairs)",
                group, book_id, model_name.empty() ? "_unassigned" : model_name, offset,
                next_offset, path, pairs_written);
}

// POST /registry/<group>/fetch/huggingface
// {"dataset_id":"...","num_pairs":N,"split":"...","input_field":"...",
//  "output_field":"...","model_name":"..."}
//
// Phase 12: the full dataset is downloaded once and cached (DataFetcher::
// ensure_huggingface_cached); each call serves the next num_pairs rows from a
// per-(dataset_id,split,model_name) cursor rather than always the first
// num_pairs rows, wrapping around to row 0 once the dataset is exhausted.
static void handle_fetch_huggingface(const httplib::Request& req, httplib::Response& res,
                                     const std::string& group) {
    const std::string dataset_id = json_string(req.body, "dataset_id");
    const int num_pairs = json_int(req.body, "num_pairs", 500);
    std::string split = json_string(req.body, "split");
    if (split.empty())
        split = "train";
    const std::string input_field = json_string(req.body, "input_field");
    const std::string output_field = json_string(req.body, "output_field");
    const std::string model_name = json_string(req.body, "model_name");

    if (!is_safe_hf_identifier(dataset_id) || !is_safe_hf_identifier(split)) {
        res.status = 400;
        res.set_content(
            "{\"error\":\"dataset_id/split must match [A-Za-z0-9._-/]+\"}", "application/json");
        Logger::warn("[{}] fetch/huggingface: rejected unsafe dataset_id='{}' split='{}'", group,
                     dataset_id, split);
        return;
    }
    if (!is_safe_model_name(model_name)) {
        res.status = 400;
        res.set_content("{\"error\":\"model_name must match [A-Za-z0-9._-]*\"}",
                        "application/json");
        Logger::warn("[{}] fetch/huggingface: rejected unsafe model_name='{}'", group, model_name);
        return;
    }

    const std::string out_dir = data_dir + "/" + group + "/datasets";
    FetcherConfig fcfg;
    fcfg.huggingface_output_dir = out_dir;
    DataFetcher fetcher(fcfg);

    Logger::info("[{}] fetch/huggingface: ensuring '{}' (split={}) is cached...", group, dataset_id,
                 split);
    const std::string cached_jsonl = fetcher.ensure_huggingface_cached(dataset_id, split);
    if (cached_jsonl.empty()) {
        res.status = 502;
        res.set_content("{\"added\":false,\"reason\":\"fetch_failed\"}", "application/json");
        Logger::error("[{}] fetch/huggingface: failed to download/cache '{}'", group, dataset_id);
        return;
    }

    const std::string cursor_path = hf_cursor_path(group, dataset_id, split, model_name);
    const int offset = read_cursor(cursor_path);

    const std::string safe_id = flatten_for_path(dataset_id);
    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    const std::string path = out_dir + "/" + safe_id + "_" + split + "_" + safe_model + "_row" +
                             std::to_string(offset) + "_training.jsonl";

    int next_offset = offset;
    const int pairs_written = DataFetcher::convert_huggingface_slice(
        cached_jsonl, path, input_field, output_field, offset, num_pairs, next_offset);
    if (pairs_written <= 0) {
        fs::remove(path);
        res.status = 502;
        res.set_content("{\"added\":false,\"reason\":\"fetch_failed\"}", "application/json");
        Logger::error("[{}] fetch/huggingface: no pairs extracted from '{}' at row {}", group,
                     dataset_id, offset);
        return;
    }
    write_cursor(cursor_path, next_offset);

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);
    add_pending_path_locked(gs, path, "huggingface", pairs_written);

    std::ostringstream json;
    json << "{\"added\":true,\"path\":\"" << json_escape(path) << "\",\"served_from_row\":" << offset
         << ",\"next_row\":" << next_offset << ",\"pairs_written\":" << pairs_written << "}";
    res.set_content(json.str(), "application/json");
    Logger::info(
        "[{}] fetch/huggingface: '{}' model='{}' rows[{},{}) → '{}' ({} pairs)", group, dataset_id,
        model_name.empty() ? "_unassigned" : model_name, offset, next_offset, path, pairs_written);
}

// POST /registry/<group>/upload?filename=<name>  (raw body = file bytes)
static void handle_upload(const httplib::Request& req, httplib::Response& res,
                          const std::string& group) {
    const std::string filename =
        req.has_param("filename") ? req.get_param_value("filename") : "";
    if (!is_safe_upload_filename(filename)) {
        res.status = 400;
        res.set_content("{\"error\":\"filename must be a bare basename with no path separators\"}",
                        "application/json");
        Logger::warn("[{}] upload: rejected unsafe filename='{}'", group, filename);
        return;
    }
    if (req.body.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"empty request body\"}", "application/json");
        return;
    }

    const std::string dir = data_dir + "/" + group + "/uploads";
    fs::create_directories(dir);
    const std::string path = dir + "/" + filename;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open() || !(out << req.body) || !out.good()) {
        res.status = 500;
        res.set_content("{\"added\":false,\"reason\":\"write_failed\"}", "application/json");
        Logger::error("[{}] upload: failed to write '{}'", group, path);
        return;
    }
    out.close();

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);
    add_pending_path_locked(gs, path, "upload");

    std::ostringstream json;
    json << "{\"added\":true,\"path\":\"" << json_escape(path) << "\"}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] upload: '{}' ({} bytes)", group, path, req.body.size());
}

// ============================================================================
// Admin config (GET/PUT /admin/config) — see CLAUDE.md "Daemon admin config API"
// ============================================================================

static std::string admin_config_json() {
    std::ostringstream j;
    j << "{\"ftp_token_ttl_minutes\":" << ftp_token_ttl_min.load()
      << ",\"ftp_max_sessions_per_run\":" << ftp_max_sessions
      << ",\"admin_enabled\":" << (admin_enabled ? "true" : "false") << "}";
    return j.str();
}

static void handle_admin_get_config(const httplib::Request&, httplib::Response& res) {
    res.set_content(admin_config_json(), "application/json");
}

static void handle_admin_put_config(const httplib::Request& req, httplib::Response& res) {
    if (!admin_enabled) {
        res.status = 403;
        res.set_content("{\"error\":\"admin config mutation disabled (--admin-enabled=false)\"}",
                        "application/json");
        return;
    }

    static const char* kImmutableKeys[] = {
        "port",         "data_dir",   "ftp_server_port", "ftp_pasv_port_min",
        "ftp_pasv_port_max", "ftps_enabled", "ftp_cert_file",    "ftp_key_file"};
    for (const auto* key : kImmutableKeys) {
        if (json_has_key(req.body, key)) {
            res.status = 400;
            res.set_content(std::string("{\"error\":\"'") + key +
                                "' is immutable at runtime; set it via config.registry.conf or "
                                "the matching --flag and restart\"}",
                            "application/json");
            return;
        }
    }

    bool changed = false;

    if (json_has_key(req.body, "ftp_token_ttl_minutes")) {
        ftp_token_ttl_min.store(json_int(req.body, "ftp_token_ttl_minutes", ftp_token_ttl_min));
        changed = true;
    }
    // ftp_max_sessions_per_run is recorded for next restart but does not affect
    // the already-constructed FtpDataServer instance — see the comment on
    // ftp_max_sessions's declaration.
    bool max_sessions_applied_live = true;
    if (json_has_key(req.body, "ftp_max_sessions_per_run")) {
        ftp_max_sessions = json_int(req.body, "ftp_max_sessions_per_run", ftp_max_sessions);
        max_sessions_applied_live = !ftp_enabled;  // only "live" if there's no running instance
        changed = true;
    }

    if (!changed) {
        res.status = 400;
        res.set_content(
            "{\"error\":\"no recognized mutable keys in body (ftp_token_ttl_minutes, "
            "ftp_max_sessions_per_run)\"}",
            "application/json");
        return;
    }

    if (g_config_store) {
        g_config_store->set("ftp_token_ttl_minutes", std::to_string(ftp_token_ttl_min.load()));
        g_config_store->set("ftp_max_sessions_per_run", std::to_string(ftp_max_sessions));
    }
    Logger::info("registry_server: admin config updated (ftp_token_ttl_minutes={}, "
                "ftp_max_sessions_per_run={})",
                ftp_token_ttl_min.load(), ftp_max_sessions);

    std::ostringstream j;
    j << "{\"ftp_token_ttl_minutes\":" << ftp_token_ttl_min.load()
      << ",\"ftp_max_sessions_per_run\":" << ftp_max_sessions
      << ",\"admin_enabled\":" << (admin_enabled ? "true" : "false")
      << ",\"ftp_max_sessions_per_run_applied\":" << (max_sessions_applied_live ? "true" : "false")
      << "}";
    res.set_content(j.str(), "application/json");
}

// ============================================================================
// Handler: POST /registry/<group>/session/next  {"model_name":..., "run_id":...}
//
// Allocates the next session number for (model_name, run_id) — "session-01"
// the first call for a given run_id, "session-02" the next, and so on. A
// run_id never seen before naturally starts at 1, so a new MNS-allocated run
// (see CLAUDE.md "Configuration") resets the session counter for free, with
// no explicit reset signal needed. Group-agnostic: shared across all groups
// on this daemon since (model_name, run_id) is already globally unique.
// ============================================================================

static void handle_session_next(const httplib::Request& req, httplib::Response& res,
                                const std::string& /*group*/) {
    const std::string model_name = json_string(req.body, "model_name");
    const std::string run_id = json_string(req.body, "run_id");
    if (run_id.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"run_id required\"}", "application/json");
        return;
    }

    std::lock_guard<std::mutex> lock(g_session_store_mtx);
    if (!g_session_store) {
        res.status = 503;
        res.set_content("{\"error\":\"session_counters.db unavailable\"}", "application/json");
        return;
    }

    const std::string key = model_name + "|" + run_id;
    int next = 1;
    if (const auto all = g_session_store->load_all(); all.count(key)) {
        try {
            next = std::stoi(all.at(key)) + 1;
        } catch (...) {
        }
    }
    g_session_store->set(key, std::to_string(next));

    const std::string session_str = std::to_string(next);
    const std::string padded = session_str.size() < 2 ? "0" + session_str : session_str;
    res.set_content("{\"session_id\":\"session-" + padded + "\"}", "application/json");
}

// ============================================================================
// Usage / main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout
        << "ADAI Registry Server — distributed dataset queue coordination\n"
        << "Usage: " << prog << " [OPTIONS]\n\n"
        << "Options:\n"
        << "  --config PATH         Path to config.registry.conf (default: "
           "./config.registry.conf or /etc/adai/config.registry.conf)\n"
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
        << "  --admin-enabled BOOL  Allow PUT /admin/config to mutate settings (default: true)\n"
        << "  --help                Show this message\n\n"
        << "Endpoints per group:\n"
        << "  GET  /registry/<group>/queue\n"
        << "  POST /registry/<group>/acquire  "
           "{\"run_id\":\"...\",\"max_files\":N,\"model_name\":\"...\"}\n"
        << "  POST /registry/<group>/session/next {\"model_name\":\"...\",\"run_id\":\"...\"}\n"
        << "  POST /registry/<group>/release  {\"run_id\":\"...\",\"files\":[...]}\n"
        << "  POST /registry/<group>/assign   "
           "{\"model_name\":\"...\",\"paths\":[...],\"count\":N}\n"
        << "  POST /registry/<group>/unassign "
           "{\"model_name\":\"...\",\"paths\":[...],\"force\":bool}\n"
        << "  POST /registry/<group>/delete   "
           "{\"paths\":[...],\"force\":bool,\"delete_files\":bool}\n"
        << "  POST /registry/<group>/trained  "
           "{\"run_id\":\"...\",\"files\":[...],\"samples\":[...]}\n"
        << "  GET  /registry/<group>/registry\n"
        << "  GET  /registry/<group>/runs\n"
        << "  GET  /registry/<group>/history[?model_id=<uuid>]\n"
        << "  POST /registry/<group>/fetch/gutenberg   "
           "{\"book_id\":N,\"num_pairs\":N}\n"
        << "  POST /registry/<group>/fetch/huggingface "
           "{\"dataset_id\":\"...\",\"num_pairs\":N,\"split\":\"...\"}\n"
        << "  POST /registry/<group>/upload?filename=<name>  (raw body = file bytes)\n"
        << "  GET  /admin/config\n"
        << "  PUT  /admin/config\n"
        << "  GET  /health\n";
}

int main(int argc, char* argv[]) {
    // Single pass: collect raw CLI values without applying any precedence yet.
    // Precedence (config.registry.conf < persisted admin overrides < this run's
    // CLI flags) is resolved explicitly below — see CLAUDE.md "Daemon admin
    // config API".
    std::optional<std::string> cli_config_path;
    std::optional<int> cli_port;
    std::optional<std::string> cli_data_dir;
    std::optional<int> cli_ftp_ttl;
    std::optional<int> cli_ftp_max_sessions;
    std::optional<bool> cli_admin_enabled;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            cli_config_path = argv[++i];
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            cli_port = std::stoi(argv[++i]);
        } else if (arg == "--data-dir" && i + 1 < argc) {
            cli_data_dir = argv[++i];
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
            cli_ftp_ttl = std::stoi(argv[++i]);
        } else if (arg == "--ftp-secret" && i + 1 < argc) {
            ftp_server_secret = argv[++i];
        } else if (arg == "--ftps") {
            ftps_enabled = true;
        } else if (arg == "--ftp-cert" && i + 1 < argc) {
            ftp_cert_file = argv[++i];
        } else if (arg == "--ftp-key" && i + 1 < argc) {
            ftp_key_file = argv[++i];
        } else if (arg == "--ftp-max-sessions" && i + 1 < argc) {
            cli_ftp_max_sessions = std::stoi(argv[++i]);
        } else if (arg == "--admin-enabled" && i + 1 < argc) {
            std::string v = argv[++i];
            for (auto& c : v)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            cli_admin_enabled = (v == "true" || v == "1" || v == "yes" || v == "on");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    const std::string config_path = adai::ConfigLoader::discover_config_path(
        cli_config_path.value_or(""), "config.registry.conf");
    adai::ServiceConfig file_config = adai::ConfigLoader::load(config_path);

    int port = cli_port.value_or(file_config.registry_listen_port);
    data_dir = cli_data_dir.value_or(file_config.registry_data_dir);
    // ftp_server_port/pasv range/cert/key/ftps_enabled intentionally keep their
    // CLI-only defaults above — they're immutable-at-runtime listener settings
    // (see kImmutableKeys in handle_admin_put_config) and not worth threading
    // through the file for the same reason port/data_dir aren't either. Only
    // the two admin-mutable FTP settings get the full file/DB/CLI treatment:
    int ftp_ttl_value = file_config.ftp_token_ttl_minutes;
    ftp_max_sessions = file_config.ftp_max_sessions_per_run;

    // Overlay persisted admin overrides on top of the file defaults.
    try {
        fs::create_directories(data_dir);
        g_config_store = std::make_unique<adai::DaemonConfigStore>(data_dir + "/daemon_config.db");
        const auto overrides = g_config_store->load_all();
        if (auto it = overrides.find("ftp_token_ttl_minutes"); it != overrides.end()) {
            try {
                ftp_ttl_value = std::stoi(it->second);
            } catch (...) {
            }
        }
        if (auto it = overrides.find("ftp_max_sessions_per_run"); it != overrides.end()) {
            try {
                ftp_max_sessions = std::stoi(it->second);
            } catch (...) {
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: daemon_config.db unavailable (" << e.what()
                  << "); admin config changes won't persist across restarts\n";
    }

    try {
        std::lock_guard<std::mutex> lock(g_session_store_mtx);
        g_session_store =
            std::make_unique<adai::DaemonConfigStore>(data_dir + "/session_counters.db");
    } catch (const std::exception& e) {
        std::cerr << "Warning: session_counters.db unavailable (" << e.what()
                  << "); POST /registry/<group>/session/next will fail\n";
    }

    // This run's explicit CLI flags win over everything, including persisted
    // admin overrides.
    if (cli_ftp_ttl)
        ftp_ttl_value = *cli_ftp_ttl;
    if (cli_ftp_max_sessions)
        ftp_max_sessions = *cli_ftp_max_sessions;
    ftp_token_ttl_min.store(ftp_ttl_value);
    admin_enabled = cli_admin_enabled.value_or(true);

    Logger::init(Logger::Level::INFO, "registry_server");

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Start FTP server if enabled (Phase 3: pass security params)
    if (ftp_enabled) {
        g_ftp_server = std::make_unique<FtpDataServer>(
            data_dir, ftp_port, ftp_pasv_min, ftp_pasv_max, ftp_advertise_ip, ftp_server_secret,
            ftps_enabled, ftp_cert_file, ftp_key_file, ftp_max_sessions);
        g_ftp_server->start();
    }

    httplib::Server svr;
    g_server = &svr;

    // Route all /registry/<group>/... paths
    svr.Get(R"(/registry/([^/]+)/queue)", [](const httplib::Request& r, httplib::Response& res) {
        handle_queue(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/acquire)", [](const httplib::Request& r, httplib::Response& res) {
        handle_acquire(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/session/next)",
             [](const httplib::Request& r, httplib::Response& res) {
                 handle_session_next(r, res, r.matches[1]);
             });
    svr.Post(R"(/registry/([^/]+)/release)", [](const httplib::Request& r, httplib::Response& res) {
        handle_release(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/assign)", [](const httplib::Request& r, httplib::Response& res) {
        handle_assign(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/unassign)",
             [](const httplib::Request& r, httplib::Response& res) {
                 handle_unassign(r, res, r.matches[1]);
             });
    svr.Post(R"(/registry/([^/]+)/delete)", [](const httplib::Request& r, httplib::Response& res) {
        handle_delete(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/trained)", [](const httplib::Request& r, httplib::Response& res) {
        handle_trained(r, res, r.matches[1]);
    });
    svr.Get(R"(/registry/([^/]+)/registry)", [](const httplib::Request& r, httplib::Response& res) {
        handle_registry(r, res, r.matches[1]);
    });
    svr.Get(R"(/registry/([^/]+)/runs)", [](const httplib::Request& r, httplib::Response& res) {
        handle_runs(r, res, r.matches[1]);
    });
    svr.Get(R"(/registry/([^/]+)/history)", [](const httplib::Request& r, httplib::Response& res) {
        handle_history(r, res, r.matches[1]);
    });
    svr.Post(R"(/registry/([^/]+)/pending/add)",
             [](const httplib::Request& r, httplib::Response& res) {
                 handle_pending_add(r, res, r.matches[1]);
             });
    svr.Post(R"(/registry/([^/]+)/fetch/gutenberg)",
             [](const httplib::Request& r, httplib::Response& res) {
                 handle_fetch_gutenberg(r, res, r.matches[1]);
             });
    svr.Post(R"(/registry/([^/]+)/fetch/huggingface)",
             [](const httplib::Request& r, httplib::Response& res) {
                 handle_fetch_huggingface(r, res, r.matches[1]);
             });
    svr.Post(R"(/registry/([^/]+)/upload)", [](const httplib::Request& r, httplib::Response& res) {
        handle_upload(r, res, r.matches[1]);
    });
    svr.Get("/admin/config", handle_admin_get_config);
    svr.Put("/admin/config", handle_admin_put_config);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    Logger::info("registry_server listening on port {}", port);
    Logger::info("Data directory: {}", data_dir);
    if (ftp_enabled) {
        Logger::info("FTP data server enabled on port {} (PASV {}–{})", ftp_port, ftp_pasv_min,
                     ftp_pasv_max);
    }

    svr.listen("0.0.0.0", port);

    if (g_ftp_server)
        g_ftp_server->stop();
    Logger::info("registry_server stopped");
    return 0;
}
