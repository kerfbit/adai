// @adai-status: beta        (TD-035 partially resolved — argv/config parsing extracted and tested; the full request-handler-isolated unit test this item calls for still needs main() extracted into a reusable class; TD-040 fully resolved, see below)
// @adai-version: 0.20.0
// @adai-reviewed: 2026-10-25

// TD-040 is fully resolved: handle_acquire()'s FTP-token path-confinement gap is fixed (see the
// fix and its comment there), ftp_detail::random_hex() (FtpDataServer.hpp) is hardened to
// RAND_bytes(), handle_pending_add() below logs a defense-in-depth warning for an out-of-tree
// path, and tests/registry_ftp_confinement_test.cpp is a permanent regression test for the fix.
// See TECHNICAL_DEBT_RESOLVED.md for the full writeup.

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
 * State is persisted to flat files in --data-dir/<group>/[<kind>/] using the same format
 * as LocalTransport, so manual inspection and recovery are always possible.
 *
 * Endpoints (every path below also accepts an optional kind segment right after <group> —
 * /registry/<group>/<encoder|decoder|world_model|chatbot>/<action> — TD-202; omitting it
 * targets the legacy shared pool exactly as before. A kind value outside that fixed set
 * simply doesn't match any route and 404s, rather than reaching a handler):
 *   GET  /registry/<group>[/<kind>]/queue     — all pending entries with run assignments
 *   POST /registry/<group>[/<kind>]/acquire   — atomically claim up to N files for run_id
 *   POST /registry/<group>[/<kind>]/release   — return reserved files to the unassigned pool
 *   POST /registry/<group>[/<kind>]/assign    — set model_name on N pending entries, by path,
 *                                       count, or all (Phase 14; count added Phase 16)
 *   POST /registry/<group>[/<kind>]/unassign  — clear model_name back to unassigned (Phase 16)
 *   POST /registry/<group>[/<kind>]/delete    — purge entries from pending and/or registry by
 *                                       path, optionally unlinking the file (Phase 16)
 *   POST /registry/<group>[/<kind>]/trained   — mark files trained and remove from queue
 *   POST /registry/<group>[/<kind>]/requeue   — move trained entries back to the pending queue
 *                                       (listed paths or all), e.g. before retraining
 *   GET  /registry/<group>[/<kind>]/registry  — full data registry
 *   GET  /registry/<group>[/<kind>]/runs      — files currently assigned per run_id
 *   POST /registry/<group>[/<kind>]/fetch/gutenberg   — download a Gutenberg book server-side (Phase 11)
 *   POST /registry/<group>[/<kind>]/fetch/huggingface — download a HuggingFace dataset server-side (Phase 11)
 *   POST /registry/<group>[/<kind>]/upload?filename=  — upload a local file's bytes server-side (Phase 11)
 *   GET  /health                     — liveness check
 *
 * Phase 11 extension: the three endpoints above make the registry itself
 * responsible for getting dataset bytes onto its own data_dir (instead of
 * expecting the caller to already have them locally), so the existing FTP
 * delivery pipeline always has something real to serve.
 */

#include <httplib.h>
#include <climits>
#include "nlohmann/json.hpp"
#include "BrowserWriteGuard.hpp"
#include "HttpListenSocket.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
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
#include "RegistryServerArgs.hpp"
#include "DataFetcher.hpp"
#include "FtpDataServer.hpp"
#include "Logger.hpp"
#include "PortableTime.hpp"
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

// TD-202: kind partitions a group's pending pool into its own physical sub-directory — one per
// trainable piece (encoder/decoder/world_model/chatbot, same vocabulary as MNS's own
// ModelRecord::kind, TD-196) — so e.g. --objective=lejepa's world-model pretraining data never
// leaks into plain chatbot fine-tuning's own pool, or vice versa. Empty kind reproduces today's
// exact single-pool-per-group behavior byte-for-byte; this is what keeps every already-deployed
// group working untouched after this upgrade. Shared by get_group() and every path builder below
// that (like the Gutenberg/HuggingFace fetch-cursor and dataset-output paths) doesn't go through
// get_group() at all but still needs to land inside the same kind-scoped sub-directory.
// TD-490: a group name is one path component under data_dir. httplib decodes the URL before the
// route regex, so "/registry/../..." and "/registry/%2e%2e/..." both reached handlers as "..",
// which resolved to data_dir's parent (an unauthenticated delete unlinked a file there), and "."
// to data_dir itself. Allowed: [A-Za-z0-9_-][A-Za-z0-9._-]*, at most 128 characters; it can't
// start with '.', so "." and ".." are out.
static bool valid_group_name(const std::string& group) {
    if (group.empty() || group.size() > 128) {
        return false;
    }
    auto ok = [](char c, bool first) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '-' || (!first && c == '.');
    };
    if (!ok(group[0], true)) {
        return false;
    }
    for (size_t i = 1; i < group.size(); ++i) {
        if (!ok(group[i], false)) {
            return false;
        }
    }
    return true;
}

// Every route checks the group first; a bad one gets 400 and never reaches a handler.
static bool group_ok(const std::string& group, httplib::Response& res) {
    if (valid_group_name(group)) {
        return true;
    }
    res.status = 400;
    res.set_content("{\"error\":\"invalid group name\"}", "application/json");
    Logger::warn("registry: rejected invalid group name '{}'", group);
    return false;
}

static std::string group_dir(const std::string& group, const std::string& kind) {
    // TD-490: defense in depth behind the route check, for any path that builds a directory.
    if (!valid_group_name(group)) {
        throw std::invalid_argument("invalid registry group name");
    }
    return kind.empty() ? data_dir + "/" + group : data_dir + "/" + group + "/" + kind;
}

// TODO: See TD-495 in TECHNICAL_DEBT.md - every handler ignores LocalTransport's save_*() result, so a failed
// (truncating) write is reported as success.
static GroupState& get_group(const std::string& group, const std::string& kind) {
    std::lock_guard<std::mutex> lock(groups_mtx);
    const std::string map_key = kind.empty() ? group : group + "/" + kind;
    auto& gs = groups[map_key];
    if (!gs.transport) {
        const std::string dir = group_dir(group, kind);
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

// TD-494: claim leases. A claim older than the lease (claimed_unix + lease) may be taken by
// another run's acquire; POST …/renew restarts it. 0 = claims never expire. A claim made before
// leases existed has no claimed_unix; its lease counts from this server's start, so upgrading
// never releases anything immediately.
static std::atomic<long long> g_claim_lease_seconds{30 * 60};
static long long g_server_start_unix = 0;

// TD-494: fetches that may run at once, so slow downloads can't occupy every request thread.
static std::atomic<int> g_max_concurrent_fetches{2};
static std::atomic<int> g_fetches_running{0};

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

// TD-494: request bodies are parsed with the vendored nlohmann/json. The old helpers matched
// substrings of compact JSON only, so {"run_id": "r2"} (a space after the colon) answered
// "400 run_id required", escapes like \" or é were never decoded, a key name appearing inside a
// value matched as a key, and a bad element of an integer array silently became 0. The helpers keep
// their signatures and read top-level keys; a body that isn't a JSON object yields the defaults.
static const nlohmann::json& parsed_body(const std::string& body) {
    // Handlers read several fields from one body; parse it once per request thread.
    thread_local std::string last_body;
    thread_local nlohmann::json last_parsed = nlohmann::json::object();
    thread_local bool has_last = false;
    if (!has_last || body != last_body) {
        last_parsed = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
        if (!last_parsed.is_object())
            last_parsed = nlohmann::json::object();
        last_body = body;
        has_last = true;
    }
    return last_parsed;
}

static const nlohmann::json* json_field(const std::string& body, const std::string& key) {
    const auto& j = parsed_body(body);
    const auto it = j.find(key);
    return it == j.end() ? nullptr : &*it;
}

static std::string json_string(const std::string& body, const std::string& key) {
    const auto* v = json_field(body, key);
    return (v && v->is_string()) ? v->get<std::string>() : std::string();
}

// An integer in int range; anything else (missing, a string, a fraction, out of range) gives @p def.
static int json_int(const std::string& body, const std::string& key, int def = 0) {
    const auto* v = json_field(body, key);
    if (!v || !v->is_number_integer())
        return def;
    const auto n = v->get<long long>();
    return (n < INT_MIN || n > INT_MAX) ? def : static_cast<int>(n);
}

static bool json_bool(const std::string& body, const std::string& key, bool def = false) {
    const auto* v = json_field(body, key);
    return (v && v->is_boolean()) ? v->get<bool>() : def;
}

// Presence check — distinguishes "key not sent" from "sent with its zero/default value".
static bool json_has_key(const std::string& body, const std::string& key) {
    return json_field(body, key) != nullptr;
}

// The string elements of an array (non-string elements are skipped).
static std::vector<std::string> json_string_array(const std::string& body, const std::string& key) {
    std::vector<std::string> result;
    const auto* v = json_field(body, key);
    if (!v || !v->is_array())
        return result;
    for (const auto& e : *v)
        if (e.is_string())
            result.push_back(e.get<std::string>());
    return result;
}

// An array of ints, or empty if any element isn't an integer in int range. These arrays are
// aligned with another one (samples/segment_starts/segment_counts with files), so dropping or
// zeroing a bad element would misalign or falsify them; empty means "not sent" to every caller.
static std::vector<int> json_int_array(const std::string& body, const std::string& key) {
    std::vector<int> result;
    const auto* v = json_field(body, key);
    if (!v || !v->is_array())
        return result;
    for (const auto& e : *v) {
        if (!e.is_number_integer())
            return {};
        const auto n = e.get<long long>();
        if (n < INT_MIN || n > INT_MAX)
            return {};
        result.push_back(static_cast<int>(n));
    }
    return result;
}

// TD-205: "<key>":[{"path":"...","segment_start":N,"segment_count":N},...]; objects without a
// string path are skipped, missing ranges default to -1 (whole file).
static std::vector<SegmentTarget> json_segment_targets(const std::string& body,
                                                        const std::string& key) {
    std::vector<SegmentTarget> result;
    const auto* v = json_field(body, key);
    if (!v || !v->is_array())
        return result;
    for (const auto& obj : *v) {
        if (!obj.is_object() || !obj.contains("path") || !obj["path"].is_string())
            continue;
        SegmentTarget t;
        t.path = obj["path"].get<std::string>();
        const auto int_or = [&](const char* k) {
            const auto it = obj.find(k);
            return (it != obj.end() && it->is_number_integer()) ? it->get<int>() : -1;
        };
        t.segment_start = int_or("segment_start");
        t.segment_count = int_or("segment_count");
        if (!t.path.empty())
            result.push_back(std::move(t));
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
    adai::gmtime_utc(&t, &tm_buf);
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

// TD-040: true iff `path` resolves under `root` — both sides canonicalized,
// then reject if the relative path starts with "..". Same pattern
// handle_delete() and handle_acquire() each already apply inline against
// their own respective roots (data_dir/group and plain data_dir); factored
// out here specifically for handle_pending_add()'s defense-in-depth warning
// below, without touching either of those two already-verified call sites.
static bool path_resolves_under(const std::string& path, const fs::path& root) {
    try {
        const auto canon = fs::weakly_canonical(path);
        // `root` must be canonicalized too, not just `path`: data_dir defaults
        // to (and is commonly configured as) a *relative* path ("registry_sessions").
        // lexically_relative() is purely lexical — comparing an absolute `canon`
        // against a relative, uncanonicalized `root` returns an empty path
        // regardless of actual containment (mismatched root-name/absoluteness),
        // which would make every legitimate in-tree path look out-of-tree here.
        const auto canon_root = fs::weakly_canonical(root);
        const auto rel = canon.lexically_relative(canon_root);
        return !rel.empty() && rel.string().compare(0, 2, "..") != 0;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Route handlers
// ============================================================================

// GET /registry/<group>/queue
static void handle_queue(const httplib::Request& req, httplib::Response& res,
                         const std::string& group, const std::string& kind) {
    auto& gs = get_group(group, kind);
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
             << "," << "\"checksum\":\"" << json_escape(entries[i].checksum) << "\","
             << "\"segment_start\":" << entries[i].segment_start << ","
             << "\"segment_count\":" << entries[i].segment_count << ","
             << "\"claimed_unix\":" << entries[i].claimed_unix << "}";  // TD-494
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
                           const std::string& group, const std::string& kind) {
    const std::string run_id = json_string(req.body, "run_id");
    const std::string model_name = json_string(req.body, "model_name");
    const int max_files = json_int(req.body, "max_files", 0);

    if (run_id.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"run_id required\"}", "application/json");
        return;
    }

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    // Assignment-aware: an entry is claimable iff unassigned or assigned to
    // this exact model_name — never another model's assigned entry. See
    // RegistryTransport::acquire's doc comment (same rule, LocalTransport side).
    const auto eligible = [&model_name](const PendingEntry& e) {
        return e.model_name.empty() || e.model_name == model_name;
    };

    // TD-040: when FTP delivery is active, an entry must also resolve under
    // data_dir before it's claimable — otherwise the per-file loop below
    // would mint a real, remotely-servable FTP token whose allowed_path
    // escapes data_dir (lexically_relative() on an out-of-tree path produces
    // a "../../etc/passwd"-style string; cmd_retr's exact-match check has no
    // problem serving it). Reachable today since handle_pending_add() accepts
    // any path with zero validation. Mirrors handle_delete()'s existing
    // containment check (weakly_canonical both sides, then reject if the
    // relative path starts with ".."), applied here against the same
    // data_root handle_acquire's own ftp_path computation already uses below
    // (plain data_dir, not data_dir/group — pending entries aren't assumed to
    // live under a per-group subdirectory; handle_pending_add() accepts any
    // absolute path).
    //
    // Checked at claim time, not filtered out of the response afterward: an
    // out-of-tree entry is left unclaimed entirely (run_id untouched) rather
    // than claimed-then-released, so it doesn't consume a max_files slot a
    // deliverable file could have used instead, and a legacy/non-FTP acquire
    // (or an operator fixing/deleting the offending entry) can still reach it
    // later. Non-FTP (legacy) acquires are unaffected — those hand the raw
    // registry_path back to the same local trainer process that already has
    // full filesystem access, not a remote FTP client, so there's no trust
    // boundary being crossed for that path.
    const bool ftp_active = ftp_enabled && g_ftp_server != nullptr;
    fs::path data_root;
    if (ftp_active) {
        try {
            data_root = fs::weakly_canonical(fs::path(data_dir));
        } catch (...) {
            data_root.clear();
        }
    }
    const auto ftp_deliverable = [&](const PendingEntry& e) {
        if (!ftp_active)
            return true;
        try {
            const auto canon = fs::weakly_canonical(e.path);
            const auto rel = canon.lexically_relative(data_root);
            return !rel.empty() && rel.string().compare(0, 2, "..") != 0;
        } catch (...) {
            return false;
        }
    };

    const int limit = (max_files > 0) ? max_files : static_cast<int>(entries.size());
    std::vector<std::string> acquired;
    // TD-205: parallel arrays, pushed in lockstep with `acquired`, so both response shapes
    // below can carry each claimed entry's segment range (whole-file entries are -1/-1).
    std::vector<int> acquired_segment_start;
    std::vector<int> acquired_segment_count;
    std::vector<std::string> ftp_rejected;
    // TD-494: a claim whose lease ran out (its run crashed or stopped renewing) is claimable again.
    const long long now_unix = static_cast<long long>(std::time(nullptr));
    const long long lease = g_claim_lease_seconds.load();
    const auto claim_expired = [&](const PendingEntry& e) {
        if (lease <= 0 || e.run_id.empty())
            return false;
        const long long since = e.claimed_unix > 0 ? e.claimed_unix : g_server_start_unix;
        return now_unix - since >= lease;
    };
    for (auto& e : entries) {
        // Unclaimed, OR already claimed by this exact run_id — see the
        // matching comment in LocalTransport::acquire() (RegistryTransport.cpp) — OR (TD-494)
        // claimed by a run whose lease has expired.
        const bool expired = e.run_id != run_id && claim_expired(e);
        if ((e.run_id.empty() || e.run_id == run_id || expired) && eligible(e) &&
            static_cast<int>(acquired.size()) < limit) {
            if (!ftp_deliverable(e)) {
                ftp_rejected.push_back(e.path);
                continue;
            }
            if (expired)
                Logger::warn("[{}] acquire: run '{}' takes '{}' from run '{}', whose claim lease "
                             "expired",
                             group, run_id, e.path, e.run_id);
            e.run_id = run_id;
            e.claimed_unix = now_unix;
            acquired.push_back(e.path);
            acquired_segment_start.push_back(e.segment_start);
            acquired_segment_count.push_back(e.segment_count);
        }
    }

    for (const auto& p : ftp_rejected) {
        Logger::warn(
            "[{}] acquire: refusing to claim '{}' for FTP delivery — outside data_dir", group, p);
    }

    if (!acquired.empty()) {
        gs.transport->save_pending(entries);
    }

    if (!ftp_enabled || !g_ftp_server) {
        // Legacy response format. TD-205: "segment_starts"/"segment_counts" are new, purely
        // additive parallel arrays (same order/length as "acquired", -1 = whole file) — an
        // older client that only knows "acquired" simply ignores the extra keys.
        std::ostringstream json;
        json << "{\"acquired\":[";
        for (std::size_t i = 0; i < acquired.size(); ++i) {
            if (i)
                json << ',';
            json << '"' << json_escape(acquired[i]) << '"';
        }
        json << "],\"segment_starts\":[";
        for (std::size_t i = 0; i < acquired_segment_start.size(); ++i) {
            if (i)
                json << ',';
            json << acquired_segment_start[i];
        }
        json << "],\"segment_counts\":[";
        for (std::size_t i = 0; i < acquired_segment_count.size(); ++i) {
            if (i)
                json << ',';
            json << acquired_segment_count[i];
        }
        json << "],\"claim_lease_seconds\":" << lease << "}";  // TD-494
        res.set_content(json.str(), "application/json");
        Logger::info("[{}] acquire: run='{}' claimed {} files", group, run_id, acquired.size());
        return;
    }

    // Phase 10: extended response with per-file FTP tokens
    // ftp_path is the registry path made relative to data_dir. data_root was
    // already computed above (ftp_active is true in this branch by
    // construction — we only get here when ftp_enabled && g_ftp_server).
    std::ostringstream json;
    json << "{\"run_id\":\"" << json_escape(run_id) << "\"," << "\"ftp_server_host\":\""
         << json_escape(ftp_advertise_ip) << "\"," << "\"ftp_server_port\":" << ftp_port << ","
         << "\"ftps_enabled\":" << (ftps_enabled ? "true" : "false") << ","
         << "\"claim_lease_seconds\":" << lease << ","  // TD-494
         << "\"files\":[";

    for (std::size_t i = 0; i < acquired.size(); ++i) {
        if (i)
            json << ',';

        const fs::path file_path(acquired[i]);
        std::string ftp_path;
        std::size_t size_bytes = 0;
        std::string checksum;

        // Compute ftp_path relative to data_dir. TD-040 fixed: every entry in
        // `acquired` has already passed the canonicalized containment check
        // (ftp_deliverable(), above, computed against this same data_root)
        // before it was ever claimed — an out-of-tree entry never reaches
        // this loop at all. This lexically_relative() call (deliberately not
        // re-canonicalizing — that's already been done) is purely computing
        // the relative-path *string* to embed in the token, not re-validating
        // containment.
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
             << json_escape(tok.token_expires_utc) << "\"," << "\"segment_start\":"
             << acquired_segment_start[i] << "," << "\"segment_count\":"
             << acquired_segment_count[i] << "}";
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
                           const std::string& group, const std::string& kind) {
    const std::string run_id = json_string(req.body, "run_id");
    const std::vector<std::string> files = json_string_array(req.body, "files");
    // TD-205: explicit segment targets, additive alongside "files" — see json_segment_targets().
    const std::vector<SegmentTarget> segments = json_segment_targets(req.body, "segments");

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    // TD-205: whole-file "files" match only whole-file entries; a segment needs its own
    // `segments` target. TD-333: both lists apply in one request (this used to be either/or, so
    // a batch with both kinds released only its segments).
    int released = 0;
    const std::set<std::string> to_release(files.begin(), files.end());
    for (auto& e : entries) {
        if (!run_id.empty() && e.run_id != run_id)
            continue;
        bool match = e.segment_start < 0 && to_release.count(e.path) > 0;
        for (std::size_t i = 0; !match && i < segments.size(); ++i)
            match = segment_matches(e, segments[i].path, segments[i].segment_start,
                                    segments[i].segment_count);
        if (match) {
            e.run_id.clear();
            e.claimed_unix = 0;
            ++released;
        }
    }

    gs.transport->save_pending(entries);

    std::ostringstream json;
    json << "{\"released\":" << released << "}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] release: run='{}' released {} files", group, run_id, released);
}

// TD-494: runs @p fetch only if fewer than g_max_concurrent_fetches fetches are running, else
// answers 429 with Retry-After. A fetch downloads on the request thread (often for minutes), and
// httplib's pool is small (8 threads on small hosts), so unlimited fetches could leave no thread
// for trainers' acquire/trained calls.
template <typename Fetch>
static void with_fetch_slot(httplib::Response& res, const std::string& group, Fetch&& fetch) {
    if (g_fetches_running.fetch_add(1) >= g_max_concurrent_fetches.load()) {
        g_fetches_running.fetch_sub(1);
        res.status = 429;
        res.set_header("Retry-After", "30");
        res.set_content("{\"added\":false,\"reason\":\"busy\",\"max_concurrent_fetches\":" +
                            std::to_string(g_max_concurrent_fetches.load()) + "}",
                        "application/json");
        Logger::warn("[{}] fetch refused: {} fetches already running", group,
                     g_max_concurrent_fetches.load());
        return;
    }
    struct Release {
        ~Release() { g_fetches_running.fetch_sub(1); }
    } release;
    fetch();
}

// POST /registry/<group>[/<kind>]/renew  {"run_id":"..."}
// TD-494: restarts the lease of every entry @p run_id has claimed. Trainers call it every third of
// the lease while they work (DatasetRegistry's claim renewer). "renewed":0 tells a trainer its
// claims are gone: its lease ran out and another run took them.
static void handle_renew(const httplib::Request& req, httplib::Response& res,
                         const std::string& group, const std::string& kind) {
    const std::string run_id = json_string(req.body, "run_id");
    if (run_id.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"run_id required\"}", "application/json");
        return;
    }
    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);
    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);
    const long long now_unix = static_cast<long long>(std::time(nullptr));
    int renewed = 0;
    for (auto& e : entries) {
        if (e.run_id == run_id) {
            e.claimed_unix = now_unix;
            ++renewed;
        }
    }
    if (renewed > 0)
        gs.transport->save_pending(entries);
    res.set_content("{\"renewed\":" + std::to_string(renewed) +
                        ",\"claim_lease_seconds\":" + std::to_string(g_claim_lease_seconds.load()) +
                        "}",
                    "application/json");
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
                          const std::string& group, const std::string& kind) {
    const std::string model_name = json_string(req.body, "model_name");
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const int count = json_int(req.body, "count", 0);
    // TD-205: explicit segment targets, highest priority — see json_segment_targets().
    const std::vector<SegmentTarget> segments = json_segment_targets(req.body, "segments");

    if (!is_safe_model_name(model_name) || model_name.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"non-empty model_name matching [A-Za-z0-9._-]+ required\"}",
                        "application/json");
        Logger::warn("[{}] assign: rejected model_name='{}'", group, model_name);
        return;
    }

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    std::vector<std::string> assigned_paths;
    std::vector<SegmentTarget> assigned_segments;
    if (!segments.empty()) {
        for (const auto& t : segments) {
            for (auto& e : entries) {
                if (segment_matches(e, t.path, t.segment_start, t.segment_count)) {
                    e.model_name = model_name;
                    assigned_segments.push_back(t);
                    break;
                }
            }
        }
    } else {
        const bool by_paths = !paths.empty();
        const bool by_count = !by_paths && count > 0;
        const bool assign_all = !by_paths && !by_count;
        const std::set<std::string> target_paths(paths.begin(), paths.end());
        for (auto& e : entries) {
            if (by_count && static_cast<int>(assigned_paths.size()) >= count) {
                break;
            }
            // TD-205: by_paths/by_count only ever match whole-file entries; assign_all (the
            // bulk default) still applies to segments too — see handle_unassign's own note.
            const bool matches = by_paths ? (e.segment_start < 0 && target_paths.count(e.path) > 0)
                                : by_count ? (e.segment_start < 0 && e.model_name.empty())
                                           : assign_all;
            if (matches) {
                e.model_name = model_name;
                assigned_paths.push_back(e.path);
            }
        }
    }

    const std::size_t assigned_total = assigned_paths.size() + assigned_segments.size();
    if (assigned_total > 0) {
        gs.transport->save_pending(entries);
    }

    std::ostringstream json;
    json << "{\"assigned\":" << assigned_total << ",\"paths\":[";
    for (std::size_t i = 0; i < assigned_paths.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "\"" << json_escape(assigned_paths[i]) << "\"";
    }
    json << "],\"segments\":[";
    for (std::size_t i = 0; i < assigned_segments.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "{\"path\":\"" << json_escape(assigned_segments[i].path) << "\","
             << "\"segment_start\":" << assigned_segments[i].segment_start << ","
             << "\"segment_count\":" << assigned_segments[i].segment_count << "}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] assign: model='{}' assigned {} file(s) (count={})", group, model_name,
                assigned_total, count);
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
                            const std::string& group, const std::string& kind) {
    const std::string model_name = json_string(req.body, "model_name");
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const bool force = json_bool(req.body, "force", false);
    // TD-205: explicit segment targets, highest priority — see json_segment_targets().
    const std::vector<SegmentTarget> segments = json_segment_targets(req.body, "segments");

    if (paths.empty() && segments.empty() && model_name.empty()) {
        res.status = 400;
        res.set_content(
            "{\"error\":\"paths, segments, or model_name (or a combination) required\"}",
            "application/json");
        Logger::warn("[{}] unassign: rejected — paths, segments, and model_name all empty",
                     group);
        return;
    }
    if (!model_name.empty() && !is_safe_model_name(model_name)) {
        res.status = 400;
        res.set_content("{\"error\":\"model_name must match [A-Za-z0-9._-]+\"}",
                        "application/json");
        Logger::warn("[{}] unassign: rejected model_name='{}'", group, model_name);
        return;
    }

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);

    std::vector<std::string> unassigned_paths;
    std::vector<SegmentTarget> unassigned_segments;
    int skipped = 0;
    if (!segments.empty()) {
        for (const auto& t : segments) {
            for (auto& e : entries) {
                if (segment_matches(e, t.path, t.segment_start, t.segment_count) &&
                    (model_name.empty() || e.model_name == model_name)) {
                    if (!e.run_id.empty() && !force) {
                        ++skipped;
                    } else {
                        e.model_name.clear();
                        unassigned_segments.push_back(t);
                    }
                    break;
                }
            }
        }
    } else {
        // TD-205: bulk-by-model (paths empty) intentionally matches whole-file AND segment
        // entries alike; explicit `paths` matches whole-file entries only — see
        // handle_assign's own note.
        const bool bulk_by_model = paths.empty();
        const std::set<std::string> target_paths(paths.begin(), paths.end());
        for (auto& e : entries) {
            const bool matches = bulk_by_model
                                    ? e.model_name == model_name
                                    : (e.segment_start < 0 && target_paths.count(e.path) > 0 &&
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
    }

    const std::size_t unassigned_total = unassigned_paths.size() + unassigned_segments.size();
    if (unassigned_total > 0) {
        gs.transport->save_pending(entries);
    }

    std::ostringstream json;
    json << "{\"unassigned\":" << unassigned_total << ",\"skipped\":" << skipped
        << ",\"paths\":[";
    for (std::size_t i = 0; i < unassigned_paths.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "\"" << json_escape(unassigned_paths[i]) << "\"";
    }
    json << "],\"segments\":[";
    for (std::size_t i = 0; i < unassigned_segments.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "{\"path\":\"" << json_escape(unassigned_segments[i].path) << "\","
             << "\"segment_start\":" << unassigned_segments[i].segment_start << ","
             << "\"segment_count\":" << unassigned_segments[i].segment_count << "}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] unassign: model='{}' unassigned {} file(s), {} skipped (force={})", group,
                model_name, unassigned_total, skipped, force);
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
                          const std::string& group, const std::string& kind) {
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const bool force = json_bool(req.body, "force", false);
    const bool delete_files = json_bool(req.body, "delete_files", false);
    // TD-205: explicit segment targets, additive alongside "paths" — see json_segment_targets().
    const std::vector<SegmentTarget> segments = json_segment_targets(req.body, "segments");

    if (paths.empty() && segments.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"paths or segments required\"}", "application/json");
        Logger::warn("[{}] delete: rejected — paths and segments both empty", group);
        return;
    }

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);

    std::vector<PendingEntry> pending;
    gs.transport->load_pending(pending);
    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);

    fs::path group_root;
    bool have_root = false;
    try {
        // TD-490: group_dir() rejects "..", "." and any other name that isn't one plain path
        // component, so this root is always inside data_dir.
        group_root = fs::weakly_canonical(fs::path(group_dir(group, "")));
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

    // TD-205: one combined worklist — a whole-file path becomes a target with segment_start=-1,
    // so it and an explicit segment target share the exact same per-item logic below.
    std::vector<SegmentTarget> targets;
    targets.reserve(paths.size() + segments.size());
    for (const auto& p : paths)
        targets.push_back({p, -1, -1});
    for (const auto& s : segments)
        targets.push_back(s);

    for (const auto& t : targets) {
        bool pending_blocked = false;
        bool found_anywhere = false;

        auto pit = std::find_if(pending.begin(), pending.end(), [&](const PendingEntry& e) {
            return segment_matches(e, t.path, t.segment_start, t.segment_count);
        });
        if (pit != pending.end()) {
            found_anywhere = true;
            if (!pit->run_id.empty() && !force) {
                pending_blocked = true;
            } else {
                pending.erase(pit);
                pending_changed = true;
            }
        }

        // TD-333: a whole-file target removes every trained record of the path; a segment
        // target removes exactly that segment's record.
        {
            const auto before = reg.size();
            reg.erase(std::remove_if(reg.begin(), reg.end(),
                                     [&](const DataVersion& dv) {
                                         return t.segment_start < 0
                                                    ? dv.data_file == t.path
                                                    : trained_matches(dv, t.path, t.segment_start,
                                                                      t.segment_count);
                                     }),
                      reg.end());
            if (reg.size() != before) {
                found_anywhere = true;
                registry_changed = true;
            }
        }

        std::string status;
        bool file_deleted = false;
        if (pending_blocked) {
            status = "skipped_active_run";
            ++skipped;
        } else if (found_anywhere) {
            status = "deleted";
            ++deleted;
            // TD-205: never unlink the physical file for a segment target — other segments
            // (or a legacy whole-file consumer) may still reference the same physical path.
            if (delete_files && t.segment_start < 0 && have_root) {
                try {
                    if (fs::exists(t.path)) {
                        const auto canon = fs::weakly_canonical(t.path);
                        const auto rel = canon.lexically_relative(group_root);
                        const bool contained =
                            !rel.empty() && rel.string().compare(0, 2, "..") != 0;
                        if (contained) {
                            std::error_code ec;
                            fs::remove(canon, ec);
                            file_deleted = !ec;
                        } else {
                            Logger::warn(
                                "[{}] delete: refusing to unlink '{}' — outside group data_dir",
                                group, t.path);
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
        details << "{\"path\":\"" << json_escape(t.path) << "\",\"status\":\"" << status
               << "\",\"file_deleted\":" << (file_deleted ? "true" : "false")
               << ",\"segment_start\":" << t.segment_start
               << ",\"segment_count\":" << t.segment_count << "}";
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

// POST /registry/<group>/requeue  {"paths":[...],"all":bool,"model_name":"..."}
//
// Moves trained DataVersion entries back to the pending queue, each as the unit it was trained
// as (the whole file, or that TD-205 segment, TD-333), so the next passes train on them again (e.g. before retraining a model from scratch). Either `paths`
// or `all:true` is required. `model_name`, if non-empty, assigns the re-queued entries to that
// model. A path already pending only loses its trained record ("already_pending").
static void handle_requeue(const httplib::Request& req, httplib::Response& res,
                           const std::string& group, const std::string& kind) {
    const std::vector<std::string> paths = json_string_array(req.body, "paths");
    const bool all = json_bool(req.body, "all", false);
    const std::string model_name = json_string(req.body, "model_name");
    if (paths.empty() && !all) {
        res.status = 400;
        res.set_content("{\"error\":\"paths or all:true required\"}", "application/json");
        Logger::warn("[{}] requeue: rejected — no paths and all=false", group);
        return;
    }

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);
    const RequeueResult result = gs.transport->requeue_trained(paths, all, model_name);

    std::ostringstream json;
    json << "{\"requeued\":" << result.requeued
         << ",\"already_pending\":" << result.already_pending
         << ",\"not_found\":" << result.not_found << ",\"details\":[";
    for (std::size_t i = 0; i < result.details.size(); ++i) {
        if (i)
            json << ",";
        json << "{\"path\":\"" << json_escape(result.details[i].path) << "\",\"status\":\""
             << result.details[i].status << "\"}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] requeue: {} requeued, {} already pending, {} not found (all={}, model='{}')",
                 group, result.requeued, result.already_pending, result.not_found, all,
                 model_name);
}

// POST /registry/<group>/trained  {"run_id":"...","files":[...],"samples":[...],"model_id":"..."}
static void handle_trained(const httplib::Request& req, httplib::Response& res,
                           const std::string& group, const std::string& kind) {
    const std::string run_id = json_string(req.body, "run_id");
    const std::string model_id = json_string(req.body, "model_id");
    const std::vector<std::string> files = json_string_array(req.body, "files");
    const std::vector<int> samples = json_int_array(req.body, "samples");

    auto& gs = get_group(group, kind);
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

    // TD-333: optional per-file segment ranges, aligned with "files" (exact mode). Without them
    // every entry is treated as the whole file and the pre-TD-333 path-wide removal applies.
    const std::vector<int> seg_starts = json_int_array(req.body, "segment_starts");
    const std::vector<int> seg_counts = json_int_array(req.body, "segment_counts");
    const bool exact = seg_starts.size() == files.size() && seg_counts.size() == files.size() &&
                       !files.empty();
    auto seg_start_of = [&](std::size_t i) { return exact ? seg_starts[i] : -1; };
    auto seg_count_of = [&](std::size_t i) { return exact ? seg_counts[i] : -1; };

    // Build new DataVersion entries
    std::vector<DataVersion> reg;
    gs.transport->load_registry(reg);
    auto recorded = [&](const std::string& path, int start, int count) {
        return std::any_of(reg.begin(), reg.end(), [&](const DataVersion& dv) {
            return trained_matches(dv, path, start, count);
        });
    };

    // TD-493: a unit is recorded only when it is pending and claimed by the calling run_id. It used
    // to be recorded for any caller, while only the pending removal checked the claim: another
    // run's report recorded r1's claimed file, and r1's own report then recorded nothing because
    // the path was already there. A claimed unit is now recorded even if an earlier pass recorded
    // it too (re-queued or re-uploaded and trained again), so the registry keeps each pass's sample
    // count. An empty run_id is an explicit force (as for release): it records any unit not already
    // recorded, the old behavior.
    auto claimed_by_caller = [&](std::size_t i) {
        return std::any_of(pending.begin(), pending.end(), [&](const PendingEntry& e) {
            if (e.run_id != run_id)
                return false;
            return exact ? segment_matches(e, files[i], seg_starts[i], seg_counts[i])
                         : e.path == files[i];
        });
    };
    std::vector<std::size_t> rejected;
    std::set<std::tuple<std::string, int, int>> recorded_now;
    int trained = 0;
    for (std::size_t i = 0; i < files.size(); ++i) {
        const auto unit = std::make_tuple(files[i], seg_start_of(i), seg_count_of(i));
        if (recorded_now.count(unit))
            continue;  // listed twice in one request
        const bool eligible = run_id.empty() ? !recorded(files[i], seg_start_of(i), seg_count_of(i))
                                             : claimed_by_caller(i);
        if (!eligible) {
            if (!run_id.empty())
                rejected.push_back(i);
            continue;
        }
        recorded_now.insert(unit);
        {
            DataVersion dv;
            dv.data_file = files[i];
            dv.num_samples = (i < samples.size()) ? samples[i] : 0;
            dv.trained = true;
            dv.model_id = model_id;
            dv.segment_start = seg_start_of(i);
            dv.segment_count = seg_count_of(i);

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

    // Remove trained files from pending (exact mode: exactly the trained units).
    const std::set<std::string> trained_set(files.begin(), files.end());
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](const PendingEntry& e) {
                                     if (!run_id.empty() && e.run_id != run_id)
                                         return false;
                                     if (!exact)
                                         return trained_set.count(e.path) > 0;
                                     for (std::size_t i = 0; i < files.size(); ++i)
                                         if (segment_matches(e, files[i], seg_starts[i],
                                                             seg_counts[i]))
                                             return true;
                                     return false;
                                 }),
                  pending.end());
    gs.transport->save_pending(pending);

    std::ostringstream json;
    json << "{\"trained\":" << trained << ",\"rejected\":[";
    for (std::size_t k = 0; k < rejected.size(); ++k)
        json << (k ? "," : "") << "\"" << json_escape(files[rejected[k]]) << "\"";
    json << "]}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] trained: run='{}' committed {} units", group, run_id, trained);
    if (!rejected.empty())
        Logger::warn("[{}] trained: run='{}' reported {} unit(s) it doesn't hold a claim on; not "
                     "recorded (first: '{}')",
                     group, run_id, rejected.size(), files[rejected.front()]);
}

// GET /registry/<group>/registry
static void handle_registry(const httplib::Request& req, httplib::Response& res,
                            const std::string& group, const std::string& kind) {
    auto& gs = get_group(group, kind);
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
             << json_escape(reg[i].source) << "\",\"segment_start\":" << reg[i].segment_start
             << ",\"segment_count\":" << reg[i].segment_count << "}";
    }
    json << "]}";
    res.set_content(json.str(), "application/json");
}

// GET /registry/<group>/runs
static void handle_runs(const httplib::Request& req, httplib::Response& res,
                        const std::string& group, const std::string& kind) {
    auto& gs = get_group(group, kind);
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
                           const std::string& group, const std::string& kind) {
    const std::string filter_id = req.has_param("model_id") ? req.get_param_value("model_id") : "";

    auto& gs = get_group(group, kind);
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
                                    const std::string& source, int known_num_entries = -1,
                                    int segment_start = -1, int segment_count = -1) {
    std::vector<PendingEntry> entries;
    gs.transport->load_pending(entries);
    for (const auto& e : entries) {
        if (segment_matches(e, path, segment_start, segment_count)) {
            return false;
        }
    }

    PendingEntry entry;
    entry.path = path;
    entry.source = source;
    entry.added_utc = utc_now_string();
    entry.segment_start = segment_start;
    entry.segment_count = segment_count;
    if (const auto stat = stat_local_file(path)) {
        entry.size_bytes = stat->size_bytes;
        entry.checksum = stat->checksum;
    }
    // TD-205: a segment's own num_entries is its segment_count (the pair count it actually
    // covers), never the whole physical file's line count — count_jsonl_entries(path) would
    // otherwise report the WHOLE file's size for what is only a slice of it.
    entry.num_entries = (segment_start >= 0) ? segment_count
                       : (known_num_entries >= 0) ? known_num_entries
                                                   : count_jsonl_entries(path);

    entries.push_back(std::move(entry));
    gs.transport->save_pending(entries);
    return true;
}

// POST /registry/<group>/pending/add  {"path":"..."}
// TD-040: deliberately still accepts any path with no *rejection* for one
// outside data_dir — handle_acquire() is now the actual enforcement point
// (an out-of-tree entry is simply never claimed for FTP delivery, see its own
// TD-040 comment), and rejecting here would also block legitimate use of this
// endpoint for files outside data_dir in non-FTP (direct-filesystem-path)
// deployments, which never routes through FtpDataServer at all. This does add
// the defense-in-depth half of that decision: a same-request warning log so
// an operator gets immediate visibility into an add that handle_acquire()
// will later have to reject for FTP purposes, instead of only discovering it
// from handle_acquire()'s own warning (or not at all, if FTP happens to be
// disabled at add-time and enabled later).
static void handle_pending_add(const httplib::Request& req, httplib::Response& res,
                               const std::string& group, const std::string& kind) {
    const std::string path = json_string(req.body, "path");
    if (path.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"path required\"}", "application/json");
        return;
    }
    // TD-205: optional row-range segment — absent/-1 means "the whole file", exactly as before.
    const int segment_start = json_int(req.body, "segment_start", -1);
    const int segment_count = json_int(req.body, "segment_count", -1);

    if (!path_resolves_under(path, fs::path(data_dir))) {
        Logger::warn(
            "[{}] pending/add: '{}' does not resolve under data_dir — will be accepted for "
            "direct-filesystem-path use, but handle_acquire() will refuse to mint an FTP token "
            "for it if --ftp-enabled",
            group, path);
    }

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);

    if (!add_pending_path_locked(gs, path, "manual", -1, segment_start, segment_count)) {
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
// before it ever reaches DataFetcher. TD-310: the old allow-list ([A-Za-z0-9._-/]+) let ".."
// through, and `split` went into paths unflattened, so "x/../../ESCAPED" wrote outside the group's
// data directory. A component is now [A-Za-z0-9._-]+ and never "." or "..".
static const std::regex kSafeHfComponent(R"(^[A-Za-z0-9._\-]+$)");

static bool is_safe_hf_component(const std::string& s) {
    return !s.empty() && s != "." && s != ".." && std::regex_match(s, kSafeHfComponent);
}

// dataset_id: "name" or "org/name" (one or more components joined by '/').
static bool is_safe_hf_dataset_id(const std::string& s) {
    if (s.empty() || s.front() == '/' || s.back() == '/') {
        return false;
    }
    size_t start = 0;
    while (start <= s.size()) {
        const size_t slash = s.find('/', start);
        const size_t end = slash == std::string::npos ? s.size() : slash;
        if (!is_safe_hf_component(s.substr(start, end - start))) {
            return false;
        }
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
    return true;
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

static std::string hf_cursor_path(const std::string& group, const std::string& kind,
                                  const std::string& dataset_id, const std::string& split,
                                  const std::string& model_name) {
    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    return group_dir(group, kind) + "/fetch_cursors/hf_" + flatten_for_path(dataset_id) + "__" +
          flatten_for_path(split) + "__" + safe_model + ".txt";
}

static std::string gutenberg_cursor_path(const std::string& group, const std::string& kind,
                                         int book_id, const std::string& model_name) {
    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    return group_dir(group, kind) + "/fetch_cursors/gutenberg_" + std::to_string(book_id) + "__" +
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

// TD-491: one lock per cursor file, held from reading the cursor to queuing the slice and advancing
// it. Without it, eight concurrent fetch/huggingface calls all read offset 0, wrote the same file at
// once and queued it once, while each caller was told its 1,000 pairs were added.
static std::mutex& fetch_cursor_mutex(const std::string& cursor_path) {
    static std::mutex map_mtx;
    static std::map<std::string, std::unique_ptr<std::mutex>> mutexes;
    std::lock_guard<std::mutex> lock(map_mtx);
    auto& m = mutexes[cursor_path];
    if (!m)
        m = std::make_unique<std::mutex>();
    return *m;
}

// TD-491: "<stem>_training.jsonl", or "<stem>_<n>_training.jsonl" when that file already exists.
// After the cursor wraps, the earlier "_row<offset>" slice may still be pending, claimed or trained,
// and rewriting it would change data under a claim or a recorded checksum.
static std::string unique_slice_path(const std::string& stem) {
    std::string path = stem + "_training.jsonl";
    for (int n = 2; fs::exists(path); ++n)
        path = stem + "_" + std::to_string(n) + "_training.jsonl";
    return path;
}

// TD-491: queues a freshly written slice; on failure removes it and answers 409 added:false.
static bool queue_fetched_slice(httplib::Response& res, const std::string& group,
                                const std::string& kind, const std::string& path,
                                const std::string& source, int pairs_written) {
    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);
    if (add_pending_path_locked(gs, path, source, pairs_written))
        return true;
    fs::remove(path);
    res.status = 409;
    res.set_content("{\"added\":false,\"reason\":\"already_queued\",\"path\":\"" +
                        json_escape(path) + "\"}",
                    "application/json");
    Logger::warn("[{}] fetch/{}: '{}' was already queued; nothing added", group, source, path);
    return false;
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
                                   const std::string& group, const std::string& kind) {
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

    const std::string out_dir = group_dir(group, kind) + "/datasets";
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

    const std::string cursor_path = gutenberg_cursor_path(group, kind, book_id, model_name);
    std::lock_guard<std::mutex> cursor_lock(fetch_cursor_mutex(cursor_path));  // TD-491
    const int offset = read_cursor(cursor_path);

    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    const std::string path = unique_slice_path(out_dir + "/gutenberg_" + std::to_string(book_id) +
                                               "_" + safe_model + "_row" + std::to_string(offset));

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
    if (!queue_fetched_slice(res, group, kind, path, "gutenberg", pairs_written))
        return;
    write_cursor(cursor_path, next_offset);

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
                                     const std::string& group, const std::string& kind) {
    const std::string dataset_id = json_string(req.body, "dataset_id");
    const int num_pairs = json_int(req.body, "num_pairs", 500);
    std::string split = json_string(req.body, "split");
    if (split.empty())
        split = "train";
    const std::string input_field = json_string(req.body, "input_field");
    const std::string output_field = json_string(req.body, "output_field");
    const std::string model_name = json_string(req.body, "model_name");

    if (!is_safe_hf_dataset_id(dataset_id) || !is_safe_hf_component(split)) {
        res.status = 400;
        res.set_content("{\"error\":\"dataset_id must be name or org/name and split one name, each "
                        "[A-Za-z0-9._-]+ and never . or ..\"}",
                        "application/json");
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

    const std::string out_dir = group_dir(group, kind) + "/datasets";
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

    const std::string cursor_path = hf_cursor_path(group, kind, dataset_id, split, model_name);
    std::lock_guard<std::mutex> cursor_lock(fetch_cursor_mutex(cursor_path));  // TD-491
    const int offset = read_cursor(cursor_path);

    const std::string safe_id = flatten_for_path(dataset_id);
    const std::string safe_model = model_name.empty() ? "_unassigned" : flatten_for_path(model_name);
    const std::string path = unique_slice_path(out_dir + "/" + safe_id + "_" +
                                               flatten_for_path(split) + "_" + safe_model + "_row" +
                                               std::to_string(offset));

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
    if (!queue_fetched_slice(res, group, kind, path, "huggingface", pairs_written))
        return;
    write_cursor(cursor_path, next_offset);

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
                          const std::string& group, const std::string& kind) {
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

    const std::string dir = group_dir(group, kind) + "/uploads";
    fs::create_directories(dir);
    const std::string path = dir + "/" + filename;
    const bool overwrite = req.has_param("overwrite") && req.get_param_value("overwrite") == "true";

    // TD-492: the upload used to truncate and rewrite the file whatever its queue state, and report
    // added:true. Replacing the bytes under a claim or a trained record means a trainer trains (or
    // downloads over FTP) data that isn't what was queued, and the recorded checksum stops matching.
    // The body goes to a temporary file first; the target is only replaced, under the group lock,
    // when nothing depends on its current content.
    static std::atomic<unsigned long> upload_seq{0};
    const std::string tmp_path = path + ".upload-" + std::to_string(upload_seq.fetch_add(1)) + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open() || !(out << req.body) || !out.good()) {
            out.close();
            fs::remove(tmp_path);
            res.status = 500;
            res.set_content("{\"added\":false,\"reason\":\"write_failed\"}", "application/json");
            Logger::error("[{}] upload: failed to write '{}'", group, tmp_path);
            return;
        }
    }

    auto refuse = [&](const char* reason, const std::string& detail) {
        fs::remove(tmp_path);
        res.status = 409;
        res.set_content("{\"added\":false,\"reason\":\"" + std::string(reason) + "\",\"path\":\"" +
                            json_escape(path) + "\"}",
                        "application/json");
        Logger::warn("[{}] upload: refused '{}': {}", group, path, detail);
    };

    auto& gs = get_group(group, kind);
    std::lock_guard<std::mutex> lock(gs.mtx);
    std::vector<PendingEntry> pending;
    gs.transport->load_pending(pending);
    int queued_units = 0;
    bool queued_whole = false;
    for (const auto& e : pending) {
        if (e.path != path)
            continue;
        if (!e.run_id.empty())
            return refuse("claimed", "claimed by run '" + e.run_id + "'");
        ++queued_units;
        queued_whole = queued_whole || e.segment_start < 0;
    }
    std::vector<DataVersion> trained;
    gs.transport->load_registry(trained);
    for (const auto& r : trained) {
        if (r.data_file == path && !overwrite)
            return refuse("trained", "already trained; pass overwrite=true to upload a new version");
    }
    if (queued_units > 0 && !overwrite)
        return refuse("already_queued", "already queued; pass overwrite=true to replace it");
    if (queued_units > 0 && (queued_units > 1 || !queued_whole))
        return refuse("segmented", "queued as segments, whose row ranges a new version would break");

    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
        fs::remove(tmp_path);
        res.status = 500;
        res.set_content("{\"added\":false,\"reason\":\"write_failed\"}", "application/json");
        Logger::error("[{}] upload: failed to move '{}' into place: {}", group, path, ec.message());
        return;
    }

    bool replaced = false;
    if (queued_units > 0) {
        // overwrite=true on an unclaimed whole-file entry: refresh what it says about the file.
        for (auto& e : pending) {
            if (e.path != path)
                continue;
            if (const auto stat = stat_local_file(path)) {
                e.size_bytes = stat->size_bytes;
                e.checksum = stat->checksum;
            }
            e.num_entries = count_jsonl_entries(path);
            e.added_utc = utc_now_string();
        }
        gs.transport->save_pending(pending);
        replaced = true;
    } else {
        add_pending_path_locked(gs, path, "upload");
    }

    std::ostringstream json;
    json << "{\"added\":" << (replaced ? "false" : "true")
         << ",\"replaced\":" << (replaced ? "true" : "false") << ",\"path\":\"" << json_escape(path)
         << "\"}";
    res.set_content(json.str(), "application/json");
    Logger::info("[{}] upload: '{}' ({} bytes{})", group, path, req.body.size(),
                 replaced ? ", replaced the queued version" : "");
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
        // TODO: See TD-306 in TECHNICAL_DEBT.md - no range check (negative values accepted and persisted)
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
        // TODO: See TD-305 in TECHNICAL_DEBT.md - per-key set() with no transaction; a failure leaves memory and DB inconsistent
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
                                const std::string& /*group*/, const std::string& /*kind*/) {
    const std::string model_name = json_string(req.body, "model_name");
    const std::string run_id = json_string(req.body, "run_id");
    if (run_id.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"run_id required\"}", "application/json");
        return;
    }
    // TD-494: the counter key is "<model_name>|<run_id>", so a '|' in either could make two pairs
    // share one counter ("a|b" + "c" and "a" + "b|c").
    if (!is_safe_model_name(model_name) || run_id.find('|') != std::string::npos) {
        res.status = 400;
        res.set_content("{\"error\":\"model_name must match [A-Za-z0-9._-]* and run_id must not "
                        "contain '|'\"}",
                        "application/json");
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
    // TODO: See TD-304 in TECHNICAL_DEBT.md - read-modify-write is atomic only within this process (TD-308: full-table scan)
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
        << "  --claim-lease-minutes N  Claim lease without renewal; 0 = never expire (default: 30)\n"
        << "  --max-upload-mb N     Largest request body (uploads), MiB (default: 2048)\n"
        << "  --max-concurrent-fetches N  Gutenberg/HuggingFace fetches at once (default: 2)\n"
        << "  --help                Show this message\n\n"
        << "Endpoints per group:\n"
        << "  (TD-202: every path below also accepts an optional kind segment right after\n"
        << "   <group> — /registry/<group>/<encoder|decoder|world_model|chatbot>/<action> —\n"
        << "   to target that trainable piece's own sub-pool instead of the legacy shared one;\n"
        << "   any other kind value 404s.)\n"
        << "  GET  /registry/<group>/queue\n"
        << "  POST /registry/<group>/acquire  "
           "{\"run_id\":\"...\",\"max_files\":N,\"model_name\":\"...\"}\n"
        << "  POST /registry/<group>/session/next {\"model_name\":\"...\",\"run_id\":\"...\"}\n"
        << "  POST /registry/<group>/release  {\"run_id\":\"...\",\"files\":[...]}\n"
        << "  POST /registry/<group>/renew    {\"run_id\":\"...\"}  (restart the run's claim lease)\n"
        << "  POST /registry/<group>/assign   "
           "{\"model_name\":\"...\",\"paths\":[...],\"count\":N}\n"
        << "  POST /registry/<group>/unassign "
           "{\"model_name\":\"...\",\"paths\":[...],\"force\":bool}\n"
        << "  POST /registry/<group>/delete   "
           "{\"paths\":[...],\"force\":bool,\"delete_files\":bool}\n"
        << "  POST /registry/<group>/trained  "
           "{\"run_id\":\"...\",\"files\":[...],\"samples\":[...]}\n"
        << "  POST /registry/<group>/requeue  "
           "{\"paths\":[...],\"all\":bool,\"model_name\":\"...\"}\n"
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
    // Argument parsing and config-precedence resolution themselves live in
    // RegistryServerArgs.{hpp,cpp} (TD-035) — everything from here down still runs against this
    // file's own shared globals exactly as before.
    adai::RegistryServerArgs cli = adai::parse_registry_server_args(argc, argv);
    if (cli.help) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string config_path = adai::ConfigLoader::discover_config_path(
        cli.config_path.value_or(""), "config.registry.conf");
    adai::ServiceConfig file_config = adai::ConfigLoader::load(config_path);

    // TODO: See TD-284 in TECHNICAL_DEBT.md - config.registry.conf ships FTPS_ENABLED/FTP_DATA_SERVER_SECRET/
    // FTP_SERVER_PORT/PASV keys that are ignored here; FTPS/secret set in the file silently do nothing.
    // ftp_server_port/pasv range/cert/key/ftps_enabled are CLI-only listener settings — applied
    // directly from `cli` with no file/admin-override layer, same as before.
    // TODO: See TD-489 in TECHNICAL_DEBT.md - FTP/FTPS flags without --ftp-enabled are ignored without a warning.
    ftp_enabled = cli.ftp_enabled;
    ftp_port = cli.ftp_port;
    ftp_advertise_ip = cli.ftp_advertise_ip;
    ftp_pasv_min = cli.ftp_pasv_min;
    ftp_pasv_max = cli.ftp_pasv_max;
    ftp_server_secret = cli.ftp_server_secret;
    ftps_enabled = cli.ftps_enabled;
    ftp_cert_file = cli.ftp_cert_file;
    ftp_key_file = cli.ftp_key_file;

    int port = 0;
    // Overlay persisted admin overrides on top of the file defaults.
    try {
        data_dir = cli.data_dir.value_or(file_config.registry_data_dir);
        fs::create_directories(data_dir);
        g_config_store = std::make_unique<adai::DaemonConfigStore>(data_dir + "/daemon_config.db");
        const auto overrides = g_config_store->load_all();
        const auto effective = adai::resolve_registry_server_config(cli, file_config, overrides);
        port = effective.port;
        data_dir = effective.data_dir;
        ftp_token_ttl_min.store(effective.ftp_ttl_minutes);
        ftp_max_sessions = effective.ftp_max_sessions;
        admin_enabled = effective.admin_enabled;
    } catch (const std::exception& e) {
        std::cerr << "Warning: daemon_config.db unavailable (" << e.what()
                  << "); admin config changes won't persist across restarts\n";
        // No persisted overrides available — resolve with an empty override map instead, so
        // file/CLI precedence still applies even though the DB itself couldn't be opened.
        const auto effective = adai::resolve_registry_server_config(cli, file_config, {});
        port = effective.port;
        data_dir = effective.data_dir;
        ftp_token_ttl_min.store(effective.ftp_ttl_minutes);
        ftp_max_sessions = effective.ftp_max_sessions;
        admin_enabled = effective.admin_enabled;
    }
    // TD-494: claim lease, upload cap and fetch concurrency (file < CLI; not admin-mutable).
    const auto td494 = adai::resolve_registry_server_config(cli, file_config, {});
    g_claim_lease_seconds.store(static_cast<long long>(td494.claim_lease_minutes) * 60);
    g_max_concurrent_fetches.store(td494.max_concurrent_fetches);
    g_server_start_unix = static_cast<long long>(std::time(nullptr));
    const std::size_t max_body_bytes = static_cast<std::size_t>(td494.max_upload_mb) * 1024 * 1024;

    try {
        std::lock_guard<std::mutex> lock(g_session_store_mtx);
        g_session_store =
            std::make_unique<adai::DaemonConfigStore>(data_dir + "/session_counters.db");
    } catch (const std::exception& e) {
        std::cerr << "Warning: session_counters.db unavailable (" << e.what()
                  << "); POST /registry/<group>/session/next will fail\n";
    }

    Logger::init(Logger::Level::INFO, "registry_server");

    // TODO: See TD-283 in TECHNICAL_DEBT.md - no SIGHUP handler, so SIGHUP terminates this process
    // (CLAUDE.md said incremental_trainer hot-reloads on it).
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
    // TD-550: refuse writes a web page could make from an operator's browser (an unlisted Origin, or a
    // preflight-free Content-Type); see BrowserWriteGuard.hpp.
    adai::install_browser_write_guard(svr, "registry_server");
    adai::use_exclusive_listen_port(svr);  // TD-551
    // TD-494: bodies (uploads) are held in memory; larger ones get 413.
    svr.set_payload_max_length(max_body_bytes);

    // Route all /registry/<group>/... paths. TD-202: every pattern gains an optional kind
    // segment right after the group capture — (?:/(encoder|decoder|world_model|chatbot))? — so
    // one regex handles both today's 2-segment legacy URL (kind unmatched, r.matches[2] == "")
    // and the new 3-segment kind-scoped URL. [^/]+ can never cross a '/', so a bare group name
    // (e.g. "foo") and a kind-qualified one (e.g. "foo/encoder") never collide regardless of
    // segment count. A mistyped/unknown kind value simply fails to match any route (404) rather
    // than reaching a handler — the closed alternation *is* the validation, so no handler needs
    // its own redundant kind check.
#define ADAI_KIND_SEGMENT R"((?:/(encoder|decoder|world_model|chatbot))?)"
    svr.Get(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/queue)",
            [](const httplib::Request& r, httplib::Response& res) {
                if (group_ok(r.matches[1], res)) {
                    handle_queue(r, res, r.matches[1], r.matches[2]);
                }
            });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/acquire)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_acquire(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/session/next)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_session_next(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/release)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_release(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/renew)",  // TD-494
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_renew(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/assign)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_assign(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/unassign)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_unassign(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/delete)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_delete(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/requeue)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_requeue(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/trained)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_trained(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Get(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/registry)",
            [](const httplib::Request& r, httplib::Response& res) {
                if (group_ok(r.matches[1], res)) {
                    handle_registry(r, res, r.matches[1], r.matches[2]);
                }
            });
    svr.Get(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/runs)",
            [](const httplib::Request& r, httplib::Response& res) {
                if (group_ok(r.matches[1], res)) {
                    handle_runs(r, res, r.matches[1], r.matches[2]);
                }
            });
    svr.Get(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/history)",
            [](const httplib::Request& r, httplib::Response& res) {
                if (group_ok(r.matches[1], res)) {
                    handle_history(r, res, r.matches[1], r.matches[2]);
                }
            });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/pending/add)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_pending_add(r, res, r.matches[1], r.matches[2]);
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/fetch/gutenberg)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     with_fetch_slot(res, r.matches[1], [&] {
                         handle_fetch_gutenberg(r, res, r.matches[1], r.matches[2]);
                     });
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/fetch/huggingface)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     with_fetch_slot(res, r.matches[1], [&] {
                         handle_fetch_huggingface(r, res, r.matches[1], r.matches[2]);
                     });
                 }
             });
    svr.Post(R"(/registry/([^/]+))" ADAI_KIND_SEGMENT R"(/upload)",
             [](const httplib::Request& r, httplib::Response& res) {
                 if (group_ok(r.matches[1], res)) {
                     handle_upload(r, res, r.matches[1], r.matches[2]);
                 }
             });
#undef ADAI_KIND_SEGMENT
    svr.Get("/admin/config", handle_admin_get_config);
    svr.Put("/admin/config", handle_admin_put_config);

    // TD-494: without session_counters.db every session/next answers 503, so say so here too. Still
    // 200: everything else (acquire, trained, FTP) works, and clients treat a failed /health as down.
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        bool sessions_ok = false;
        {
            std::lock_guard<std::mutex> lock(g_session_store_mtx);
            sessions_ok = g_session_store != nullptr;
        }
        res.set_content(sessions_ok ? "{\"status\":\"ok\",\"session_counters\":true}"
                                    : "{\"status\":\"degraded\",\"session_counters\":false}",
                        "application/json");
    });

    Logger::info("registry_server listening on port {}", port);
    Logger::info("Data directory: {}", data_dir);
    if (ftp_enabled) {
        Logger::info("FTP data server enabled on port {} (PASV {}–{})", ftp_port, ftp_pasv_min,
                     ftp_pasv_max);
    }

    // TD-494/TD-551: a failed listen() (port in use, now that the port is exclusive) used to log
    // "stopped" and exit 0, so restart-on-failure never fired. A stop requested before listen() is
    // checked first; one arriving in the narrow window just before listen() is still lost.
    const bool listened = !shutdown_requested.load() && svr.listen("0.0.0.0", port);

    if (g_ftp_server)
        g_ftp_server->stop();
    if (!listened && !shutdown_requested.load()) {
        Logger::error("registry_server could not listen on port {} (already in use?)", port);
        return 1;
    }
    Logger::info("registry_server stopped");
    return 0;
}
