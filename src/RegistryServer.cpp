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
#include <atomic>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
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
static int ftp_token_ttl_min = 30;
static std::string ftp_advertise_ip;  // set from --ftp-ip; falls back to empty
// Phase 3: security hardening options
static std::string ftp_server_secret;  // HMAC key (empty = random passwords)
static bool ftps_enabled = false;      // FTPS (explicit TLS via AUTH TLS)
static std::string ftp_cert_file;      // PEM cert (empty = self-signed)
static std::string ftp_key_file;       // PEM key  (empty = self-signed)
static int ftp_max_sessions = 4;       // max concurrent sessions per run_id

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

static std::vector<std::string> json_string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> result;
    const std::string needle = "\"" + key + "\":[";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return result;
    auto cur = pos + needle.size();
    while (cur < body.size()) {
        cur = body.find('"', cur);
        if (cur == std::string::npos)
            break;
        ++cur;
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
             << json_escape(entries[i].run_id) << "\"}";
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
        if (fs::exists(file_path) && fs::is_regular_file(file_path)) {
            size_bytes = static_cast<std::size_t>(fs::file_size(file_path));
            const auto ftime = fs::last_write_time(file_path);
            std::ostringstream cs;
            cs << size_bytes << "_" << static_cast<long long>(ftime.time_since_epoch().count());
            checksum = cs.str();
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

// POST /registry/<group>/trained  {"run_id":"...","files":[...],"samples":[...],"model_id":"..."}
static void handle_trained(const httplib::Request& req, httplib::Response& res,
                           const std::string& group) {
    const std::string run_id = json_string(req.body, "run_id");
    const std::string model_id = json_string(req.body, "model_id");
    const std::vector<std::string> files = json_string_array(req.body, "files");
    const std::vector<int> samples = json_int_array(req.body, "samples");

    auto& gs = get_group(group);
    std::lock_guard<std::mutex> lock(gs.mtx);

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
            // Server has no filesystem access to compute a real checksum;
            // use a placeholder so the space-separated flat-file format
            // keeps its column alignment when loaded back by LocalTransport.
            dv.checksum = "REMOTE";
            dv.num_samples = (i < samples.size()) ? samples[i] : 0;
            dv.trained = true;
            dv.model_id = model_id;
            reg.push_back(std::move(dv));
            ++trained;
        }
    }
    gs.transport->save_registry(reg);

    // Remove trained files from pending
    std::vector<PendingEntry> pending;
    gs.transport->load_pending(pending);
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
             << "," << "\"trained\":" << (reg[i].trained ? "true" : "false") << "}";
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
static bool add_pending_path_locked(GroupState& gs, const std::string& path) {
    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);
    for (const auto& e : entries) {
        if (e.path == path) {
            return false;
        }
    }
    entries.push_back({path, {}, {}});
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

    if (!add_pending_path_locked(gs, path)) {
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

// Model names flow straight into a cursor-file path (see *_cursor_path below),
// so unlike dataset_id/split this is deliberately stricter — no '/' at all, since
// a model name has no legitimate reason to contain one and it doubles as
// path-traversal protection. Empty is allowed (buckets into a shared cursor).
static const std::regex kSafeModelName(R"(^[A-Za-z0-9._-]*$)");

static bool is_safe_model_name(const std::string& s) {
    return std::regex_match(s, kSafeModelName);
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
    add_pending_path_locked(gs, path);

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
    add_pending_path_locked(gs, path);

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
    add_pending_path_locked(gs, path);

    std::ostringstream json;
    json << "{\"added\":true,\"path\":\"" << json_escape(path) << "\"}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] upload: '{}' ({} bytes)", group, path, req.body.size());
}

// ============================================================================
// Usage / main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout
        << "ADAI Registry Server — distributed dataset queue coordination\n"
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
    svr.Post(R"(/registry/([^/]+)/release)", [](const httplib::Request& r, httplib::Response& res) {
        handle_release(r, res, r.matches[1]);
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
