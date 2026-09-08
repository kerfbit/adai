// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "ModelNameService.hpp"
#include <httplib.h>
#include <sqlite3.h>
#include "DaemonConfigStore.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include "Logger.hpp"

namespace fs = std::filesystem;
using adai::Logger;

// ============================================================================
// ServerImpl (hides httplib from the header)
// ============================================================================

class adai::ModelNameService::ServerImpl {
   public:
    httplib::Server server;
    sqlite3* db{nullptr};
    ~ServerImpl() {
        if (db) {
            sqlite3_close(db);
            db = nullptr;
        }
    }
};

// ============================================================================
// Internal JSON / format helpers
// ============================================================================

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
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
                if (c < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Find the value of a string field in a flat JSON body.
// Does NOT handle values that span nested objects — use extract_object() first.
std::string json_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return {};
    auto i = pos + needle.size();
    std::string out;
    while (i < body.size()) {
        if (body[i] == '\\' && i + 1 < body.size()) {
            ++i;
            switch (body[i]) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                default:
                    out += body[i];
                    break;
            }
        } else if (body[i] == '"') {
            break;
        } else {
            out += body[i];
        }
        ++i;
    }
    return out;
}

/// Presence check for string-valued keys (distinguishes "not sent" from "sent as "").
bool json_has_string_key(const std::string& body, const std::string& key) {
    return body.find("\"" + key + "\":\"") != std::string::npos;
}

// "N" -> "0N" for N < 10, otherwise unchanged, e.g. "run-01" ... "run-10", "run-11", ...
std::string zero_pad2(int n) {
    const std::string s = std::to_string(n);
    return s.size() < 2 ? "0" + s : s;
}

int json_int(const std::string& body, const std::string& key, int def = 0) {
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

double json_double(const std::string& body, const std::string& key, double def = 0.0) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    try {
        return std::stod(body.substr(pos + needle.size()));
    } catch (...) {
        return def;
    }
}

// Bare-literal boolean (true/false, no quotes).
bool json_bool(const std::string& body, const std::string& key, bool def = false) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    const auto start = pos + needle.size();
    if (body.compare(start, 4, "true") == 0)
        return true;
    if (body.compare(start, 5, "false") == 0)
        return false;
    return def;
}

// Extract a sub-object {"..."} for a given key using depth-counting.
// Returns "{}" when the key is absent or malformed.
std::string extract_object(const std::string& s, const std::string& key) {
    const std::string needle = "\"" + key + "\":{";
    const auto pos = s.find(needle);
    if (pos == std::string::npos)
        return "{}";
    const auto start = pos + needle.size() - 1;  // points to the opening '{'
    int depth = 0;
    bool in_str = false;
    for (auto i = start; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') {
                ++i;
            } else if (c == '"') {
                in_str = false;
            }
        } else {
            if (c == '"') {
                in_str = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                if (--depth == 0)
                    return s.substr(start, i - start + 1);
            }
        }
    }
    return "{}";
}

// Extract an array [...] for a given key using depth-counting.
// Returns "[]" when absent.
std::string extract_array(const std::string& s, const std::string& key) {
    const std::string needle = "\"" + key + "\":[";
    const auto pos = s.find(needle);
    if (pos == std::string::npos)
        return "[]";
    const auto start = pos + needle.size() - 1;  // points to '['
    int depth = 0;
    bool in_str = false;
    for (auto i = start; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') {
                ++i;
            } else if (c == '"') {
                in_str = false;
            }
        } else {
            if (c == '"') {
                in_str = true;
            } else if (c == '[') {
                ++depth;
            } else if (c == ']') {
                if (--depth == 0)
                    return s.substr(start, i - start + 1);
            }
        }
    }
    return "[]";
}

// Split "[{...},{...}]" into a vector of individual "{...}" strings.
std::vector<std::string> split_array_objects(const std::string& arr) {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < arr.size()) {
        const auto obj_start = arr.find('{', i);
        if (obj_start == std::string::npos)
            break;
        int depth = 0;
        bool in_str = false;
        size_t j = obj_start;
        for (; j < arr.size(); ++j) {
            const char c = arr[j];
            if (in_str) {
                if (c == '\\') {
                    ++j;
                } else if (c == '"') {
                    in_str = false;
                }
            } else {
                if (c == '"') {
                    in_str = true;
                } else if (c == '{') {
                    ++depth;
                } else if (c == '}') {
                    if (--depth == 0) {
                        result.push_back(arr.substr(obj_start, j - obj_start + 1));
                        i = j + 1;
                        break;
                    }
                }
            }
        }
        if (depth != 0)
            break;
    }
    return result;
}

// Parse a {"key":"val",...} string into a string map.
std::map<std::string, std::string> parse_string_map(const std::string& obj) {
    std::map<std::string, std::string> result;
    size_t i = 0;
    // Find first '{' (handles leading content)
    while (i < obj.size() && obj[i] != '{')
        ++i;
    ++i;  // skip '{'
    while (i < obj.size() && obj[i] != '}') {
        while (i < obj.size() && (obj[i] == ',' || obj[i] == ' '))
            ++i;
        if (i >= obj.size() || obj[i] == '}')
            break;
        if (obj[i] != '"') {
            ++i;
            continue;
        }
        ++i;
        std::string key;
        while (i < obj.size() && obj[i] != '"') {
            if (obj[i] == '\\' && i + 1 < obj.size()) {
                ++i;
                key += obj[i];
            } else
                key += obj[i];
            ++i;
        }
        if (i < obj.size())
            ++i;  // skip closing '"'
        while (i < obj.size() && obj[i] != '"')
            ++i;
        if (i >= obj.size())
            break;
        ++i;  // skip opening '"'
        std::string val;
        while (i < obj.size() && obj[i] != '"') {
            if (obj[i] == '\\' && i + 1 < obj.size()) {
                ++i;
                val += obj[i];
            } else
                val += obj[i];
            ++i;
        }
        if (i < obj.size())
            ++i;  // skip closing '"'
        if (!key.empty())
            result[key] = val;
    }
    return result;
}

// UUID v4 generator.
std::string generate_uuid() {
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_int_distribution<unsigned> dist(0, 255);
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    unsigned char b[16];
    for (auto& x : b)
        x = static_cast<unsigned char>(dist(rng));
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    char buf[37];
    std::snprintf(buf, sizeof(buf),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0],
                  b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
                  b[14], b[15]);
    return buf;
}

std::string utc_now() {
    const auto t = std::time(nullptr);
    struct tm tm_utc {};
    gmtime_r(&t, &tm_utc);
    char buf[24];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
}

// Validate model_name / role: [a-z0-9][a-z0-9\-\.]{1,127}
bool valid_name(const std::string& name) {
    if (name.size() < 2 || name.size() > 128)
        return false;
    const char c0 = name[0];
    if (!((c0 >= 'a' && c0 <= 'z') || (c0 >= '0' && c0 <= '9')))
        return false;
    for (size_t i = 1; i < name.size(); ++i) {
        const char c = name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
            return false;
    }
    return true;
}

}  // namespace

// ============================================================================
// Serialization helpers (adai namespace so they can see the structs)
// ============================================================================

namespace adai {

static std::string serialize_artifact(const ArtifactLocation& a) {
    std::ostringstream j;
    j << "{\"host\":\"" << json_escape(a.host) << "\"" << ",\"path\":\"" << json_escape(a.path)
      << "\"" << ",\"checksum\":\"" << json_escape(a.checksum) << "\"" << ",\"format\":\""
      << json_escape(a.format) << "\"}";
    return j.str();
}

static ArtifactLocation parse_artifact(const std::string& obj) {
    ArtifactLocation a;
    a.host = json_string(obj, "host");
    a.path = json_string(obj, "path");
    a.checksum = json_string(obj, "checksum");
    a.format = json_string(obj, "format");
    if (a.format.empty())
        a.format = "adai-native";
    return a;
}

static std::string serialize_record(const ModelRecord& r) {
    std::ostringstream j;
    j << "{\"model_id\":\"" << json_escape(r.model_id) << "\"" << ",\"model_name\":\""
      << json_escape(r.model_name) << "\"" << ",\"role\":\"" << json_escape(r.role) << "\""
      << ",\"run_group\":\"" << json_escape(r.run_group) << "\"" << ",\"state\":\""
      << json_escape(r.state) << "\"" << ",\"run_id\":\""
      << json_escape(r.run_id) << "\"" << ",\"created_utc\":\"" << json_escape(r.created_utc)
      << "\"" << ",\"updated_utc\":\"" << json_escape(r.updated_utc) << "\""
      << ",\"artifact\":" << serialize_artifact(r.artifact) << ",\"arch\":{"
      << "\"d_model\":" << r.d_model << ",\"num_heads\":" << r.num_heads << ",\"d_ff\":" << r.d_ff
      << ",\"num_encoder_layers\":" << r.num_encoder_layers
      << ",\"num_decoder_layers\":" << r.num_decoder_layers
      << ",\"max_seq_length\":" << r.max_seq_length << "}"
      << ",\"current_run_number\":" << r.current_run_number << ",\"run_started_utc\":\""
      << json_escape(r.run_started_utc) << "\"" << ",\"progress\":{\"session_id\":\""
      << json_escape(r.progress_session_id) << "\"" << ",\"epoch\":" << r.progress_epoch
      << ",\"loss\":" << r.progress_loss << ",\"best_loss\":" << r.progress_best_loss
      << ",\"updated_utc\":\"" << json_escape(r.progress_updated_utc) << "\"}"
      << ",\"training_history\":[";
    for (size_t i = 0; i < r.training_history.size(); ++i) {
        if (i)
            j << ',';
        const auto& h = r.training_history[i];
        j << "{\"run_id\":\"" << json_escape(h.run_id) << "\"" << ",\"metrics_session_key\":\""
          << json_escape(h.metrics_session_key) << "\"" << ",\"dataset_group\":\""
          << json_escape(h.dataset_group) << "\"" << ",\"epochs\":" << h.epochs
          << ",\"final_loss\":" << h.final_loss << ",\"started_utc\":\""
          << json_escape(h.started_utc) << "\"" << ",\"finished_utc\":\""
          << json_escape(h.finished_utc) << "\"" << ",\"incomplete\":"
          << (h.incomplete ? "true" : "false") << "}";
    }
    j << "],\"tags\":{";
    bool first = true;
    for (const auto& [k, v] : r.tags) {
        if (!first)
            j << ',';
        first = false;
        j << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
    }
    j << "}}";
    return j.str();
}

static ModelRecord parse_record(const std::string& line) {
    ModelRecord r;
    r.model_id = json_string(line, "model_id");
    r.model_name = json_string(line, "model_name");
    r.role = json_string(line, "role");
    r.run_group = json_string(line, "run_group");
    r.state = json_string(line, "state");
    r.run_id = json_string(line, "run_id");
    r.created_utc = json_string(line, "created_utc");
    r.updated_utc = json_string(line, "updated_utc");
    if (r.state.empty())
        r.state = "initializing";

    r.artifact = parse_artifact(extract_object(line, "artifact"));

    const auto arch = extract_object(line, "arch");
    r.d_model = static_cast<size_t>(json_int(arch, "d_model"));
    r.num_heads = static_cast<size_t>(json_int(arch, "num_heads"));
    r.d_ff = static_cast<size_t>(json_int(arch, "d_ff"));
    r.num_encoder_layers = static_cast<size_t>(json_int(arch, "num_encoder_layers"));
    r.num_decoder_layers = static_cast<size_t>(json_int(arch, "num_decoder_layers"));
    r.max_seq_length = static_cast<size_t>(json_int(arch, "max_seq_length"));

    r.current_run_number = json_int(line, "current_run_number");
    r.run_started_utc = json_string(line, "run_started_utc");
    const auto progress = extract_object(line, "progress");
    r.progress_session_id = json_string(progress, "session_id");
    r.progress_epoch = json_int(progress, "epoch");
    r.progress_loss = json_double(progress, "loss");
    r.progress_best_loss = json_double(progress, "best_loss");
    r.progress_updated_utc = json_string(progress, "updated_utc");

    for (const auto& obj : split_array_objects(extract_array(line, "training_history"))) {
        TrainingHistoryEntry h;
        h.run_id = json_string(obj, "run_id");
        h.metrics_session_key = json_string(obj, "metrics_session_key");
        h.dataset_group = json_string(obj, "dataset_group");
        h.epochs = json_int(obj, "epochs");
        h.final_loss = json_double(obj, "final_loss");
        h.started_utc = json_string(obj, "started_utc");
        h.finished_utc = json_string(obj, "finished_utc");
        h.incomplete = json_bool(obj, "incomplete", false);
        r.training_history.push_back(std::move(h));
    }

    r.tags = parse_string_map(extract_object(line, "tags"));
    return r;
}

// Compact resolve response (no full record — just what callers need to load weights).
static std::string resolve_json(const ModelRecord& r) {
    std::ostringstream j;
    j << "{\"model_id\":\"" << json_escape(r.model_id) << "\"" << ",\"model_name\":\""
      << json_escape(r.model_name) << "\"" << ",\"run_group\":\"" << json_escape(r.run_group)
      << "\"" << ",\"state\":\"" << json_escape(r.state) << "\""
      << ",\"artifact\":" << serialize_artifact(r.artifact) << "}";
    return j.str();
}

}  // namespace adai

// ============================================================================
// Constructor — register HTTP routes
// ============================================================================

adai::ModelNameService::ModelNameService(std::string data_dir, int port)
    : data_dir_(std::move(data_dir)), port_(port), server_impl_(std::make_unique<ServerImpl>()) {
    auto& svr = server_impl_->server;

    svr.Post("/models", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = handle_register(req.body);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/models", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string sf = req.has_param("state") ? req.get_param_value("state") : "";
        const std::string rf = req.has_param("role") ? req.get_param_value("role") : "";
        int limit = 50;
        if (req.has_param("limit")) {
            try {
                limit = std::stoi(req.get_param_value("limit"));
            } catch (...) {
            }
        }
        auto [status, body] = handle_list(sf, rf, limit);
        res.status = status;
        res.set_content(body, "application/json");
    });

    // Order matters: /resolve must be registered before /{name} (more specific first).
    svr.Get(R"(/models/([^/]+)/resolve)",
            [this](const httplib::Request& req, httplib::Response& res) {
                auto [status, body] = handle_resolve(std::string(req.matches[1]));
                res.status = status;
                res.set_content(body, "application/json");
            });

    svr.Get(R"(/models/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = handle_get(std::string(req.matches[1]));
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Put(
        R"(/models/([^/]+)/state)", [this](const httplib::Request& req, httplib::Response& res) {
            auto [status, body] = handle_state_transition(std::string(req.matches[1]), req.body);
            res.status = status;
            res.set_content(body, "application/json");
        });

    svr.Put(
        R"(/models/([^/]+)/progress)", [this](const httplib::Request& req, httplib::Response& res) {
            auto [status, body] = handle_progress_update(std::string(req.matches[1]), req.body);
            res.status = status;
            res.set_content(body, "application/json");
        });

    svr.Put(R"(/models/([^/]+)/run_group)",
            [this](const httplib::Request& req, httplib::Response& res) {
                auto [status, body] = handle_update_run_group(std::string(req.matches[1]), req.body);
                res.status = status;
                res.set_content(body, "application/json");
            });

    svr.Delete(R"(/models/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = handle_delete(std::string(req.matches[1]));
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/roles", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_list_roles();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get(R"(/roles/([^/]+)/production)",
            [this](const httplib::Request& req, httplib::Response& res) {
                auto [status, body] = handle_resolve_role(std::string(req.matches[1]));
                res.status = status;
                res.set_content(body, "application/json");
            });

    svr.Put(R"(/roles/([^/]+)/production)",
            [this](const httplib::Request& req, httplib::Response& res) {
                auto [status, body] = handle_promote(std::string(req.matches[1]), req.body);
                res.status = status;
                res.set_content(body, "application/json");
            });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_health();
        res.status = status;
        res.set_content(body, "application/json");
    });

    // Phase 2: dataset history proxy (registered after /resolve to keep specificity ordering)
    svr.Get(R"(/models/([^/]+)/datasets)",
            [this](const httplib::Request& req, httplib::Response& res) {
                auto [status, body] = handle_datasets(std::string(req.matches[1]));
                res.status = status;
                res.set_content(body, "application/json");
            });

    // Admin config: live-mutable subset of this daemon's settings (registry_url,
    // registry_group). port/data_dir are never exposed here — see CLAUDE.md
    // "Daemon admin config API".
    svr.Get("/admin/config", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_admin_get_config();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Put("/admin/config", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = handle_admin_put_config(req.body);
        res.status = status;
        res.set_content(body, "application/json");
    });
}

void adai::ModelNameService::set_registry(const std::string& url, const std::string& group) {
    registry_url_ = url;
    registry_group_ = group.empty() ? "default" : group;
}

void adai::ModelNameService::set_admin_enabled(bool enabled) {
    admin_enabled_ = enabled;
}

adai::ModelNameService::~ModelNameService() {
    stop();
}

// ============================================================================
// start / stop
// ============================================================================

bool adai::ModelNameService::start() {
    fs::create_directories(data_dir_);
    init_db();
    load_from_disk();

    // Opened here (not read from) so PUT /admin/config can persist future
    // changes. The caller (mns_server's main()) is responsible for reading any
    // already-persisted overrides *before* constructing this service and
    // passing the resolved values to set_registry() — that's what preserves
    // file < persisted-override < this-run's-CLI-flags precedence; doing the
    // overlay here instead would let a stale DB value silently beat an explicit
    // CLI flag on this run. See CLAUDE.md "Daemon admin config API".
    try {
        config_store_ = std::make_unique<DaemonConfigStore>(data_dir_ + "/daemon_config.db");
    } catch (const std::exception& e) {
        Logger::warn("ModelNameService: daemon_config.db unavailable ({}); admin config changes "
                     "won't persist across restarts",
                     e.what());
    }

    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
    Logger::info("ModelNameService listening on port {} (data_dir={})", port_, data_dir_);
    const bool ok = server_impl_->server.listen("0.0.0.0", port_);
    running_ = false;
    return ok;
}

void adai::ModelNameService::stop() {
    server_impl_->server.stop();
    running_ = false;
}

bool adai::ModelNameService::is_running() const {
    return running_.load();
}
int adai::ModelNameService::get_port() const {
    return port_;
}

// ============================================================================
// Persistence — SQLite backend (Phase 2)
// ============================================================================

namespace adai {

static std::string tags_to_json(const std::map<std::string, std::string>& tags) {
    std::string j = "{";
    bool first = true;
    for (const auto& [k, v] : tags) {
        if (!first)
            j += ',';
        first = false;
        j += '"' + json_escape(k) + "\":\"" + json_escape(v) + '"';
    }
    j += '}';
    return j;
}

}  // namespace adai

void adai::ModelNameService::init_db() {
    const std::string db_path = data_dir_ + "/models.db";
    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
        Logger::error("ModelNameService: cannot open SQLite DB {}: {}", db_path,
                      sqlite3_errmsg(db));
        if (db)
            sqlite3_close(db);
        return;
    }
    server_impl_->db = db;
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);

    const char* schema = R"(
CREATE TABLE IF NOT EXISTS models (
    model_id          TEXT PRIMARY KEY,
    model_name        TEXT UNIQUE NOT NULL,
    role              TEXT DEFAULT '',
    state             TEXT NOT NULL DEFAULT 'initializing',
    run_id            TEXT DEFAULT '',
    created_utc       TEXT NOT NULL,
    updated_utc       TEXT NOT NULL,
    artifact_host     TEXT DEFAULT '',
    artifact_path     TEXT DEFAULT '',
    artifact_checksum TEXT DEFAULT '',
    artifact_format   TEXT DEFAULT 'adai-native',
    d_model           INTEGER DEFAULT 0,
    num_heads         INTEGER DEFAULT 0,
    d_ff              INTEGER DEFAULT 0,
    num_encoder_layers INTEGER DEFAULT 0,
    num_decoder_layers INTEGER DEFAULT 0,
    max_seq_length    INTEGER DEFAULT 0,
    tags_json         TEXT DEFAULT '{}',
    current_run_number   INTEGER DEFAULT 0,
    run_started_utc      TEXT DEFAULT '',
    progress_session_id  TEXT DEFAULT '',
    progress_epoch       INTEGER DEFAULT 0,
    progress_loss        REAL DEFAULT 0.0,
    progress_best_loss   REAL DEFAULT 0.0,
    progress_updated_utc TEXT DEFAULT '',
    run_group            TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS training_history (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name         TEXT NOT NULL,
    run_id             TEXT DEFAULT '',
    metrics_session_key TEXT DEFAULT '',
    dataset_group      TEXT DEFAULT '',
    epochs             INTEGER DEFAULT 0,
    final_loss         REAL DEFAULT 0.0,
    started_utc        TEXT DEFAULT '',
    finished_utc       TEXT DEFAULT '',
    incomplete         INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS roles (
    role       TEXT PRIMARY KEY,
    model_name TEXT NOT NULL
);
)";
    char* errmsg = nullptr;
    if (sqlite3_exec(db, schema, nullptr, nullptr, &errmsg) != SQLITE_OK) {
        Logger::error("ModelNameService: schema creation failed: {}", errmsg);
        sqlite3_free(errmsg);
        return;
    }

    // Migration guard: CREATE TABLE IF NOT EXISTS above doesn't touch a
    // pre-existing models.db from before run-numbering/progress-tracking was
    // added, so add any missing columns explicitly (idempotent — skipped if
    // already present).
    const auto add_column_if_missing = [&](const char* table, const char* column,
                                           const char* decl) {
        bool exists = false;
        const std::string pragma = std::string("PRAGMA table_info(") + table + ")";
        sqlite3_stmt* pst = nullptr;
        if (sqlite3_prepare_v2(db, pragma.c_str(), -1, &pst, nullptr) == SQLITE_OK) {
            while (sqlite3_step(pst) == SQLITE_ROW) {
                const char* name = reinterpret_cast<const char*>(sqlite3_column_text(pst, 1));
                if (name && std::string(name) == column) {
                    exists = true;
                    break;
                }
            }
            sqlite3_finalize(pst);
        }
        if (!exists) {
            const std::string alter =
                std::string("ALTER TABLE ") + table + " ADD COLUMN " + column + " " + decl;
            sqlite3_exec(db, alter.c_str(), nullptr, nullptr, nullptr);
        }
    };
    add_column_if_missing("models", "current_run_number", "INTEGER DEFAULT 0");
    add_column_if_missing("models", "run_started_utc", "TEXT DEFAULT ''");
    add_column_if_missing("models", "progress_session_id", "TEXT DEFAULT ''");
    add_column_if_missing("models", "progress_epoch", "INTEGER DEFAULT 0");
    add_column_if_missing("models", "progress_loss", "REAL DEFAULT 0.0");
    add_column_if_missing("models", "progress_best_loss", "REAL DEFAULT 0.0");
    add_column_if_missing("models", "progress_updated_utc", "TEXT DEFAULT ''");
    // Must stay last — ALTER TABLE ADD COLUMN always appends physically at the
    // end regardless of where it's written in the CREATE TABLE literal above,
    // so migration call order here has to match that literal's column order
    // (run_group is the last column there) for a fresh vs. migrated DB to end
    // up with the same physical layout — required for persist_model's
    // positional `INSERT ... VALUES (?,?,...)` to bind correctly either way.
    add_column_if_missing("models", "run_group", "TEXT DEFAULT ''");
    add_column_if_missing("training_history", "incomplete", "INTEGER DEFAULT 0");

    // Migrate legacy JSONL files on first run (no rows yet).
    int count = 0;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM models", -1, &st, nullptr) == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW)
            count = sqlite3_column_int(st, 0);
        sqlite3_finalize(st);
    }
    if (count == 0)
        migrate_from_jsonl();

    Logger::info("ModelNameService: SQLite DB ready ({})", db_path);
}

void adai::ModelNameService::migrate_from_jsonl() {
    sqlite3* db = server_impl_->db;
    if (!db)
        return;

    const std::string models_path = data_dir_ + "/models.jsonl";
    std::ifstream f(models_path);
    if (!f.is_open())
        return;

    Logger::info("ModelNameService: migrating {} to SQLite", models_path);
    sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    std::string line;
    int imported = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        try {
            ModelRecord r = parse_record(line);
            if (r.model_name.empty())
                continue;

            const std::string sql =
                "INSERT OR REPLACE INTO models VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
            sqlite3_stmt* st = nullptr;
            if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK)
                continue;
            sqlite3_bind_text(st, 1, r.model_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 2, r.model_name.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 3, r.role.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 4, r.state.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 5, r.run_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 6, r.created_utc.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 7, r.updated_utc.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 8, r.artifact.host.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 9, r.artifact.path.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 10, r.artifact.checksum.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 11, r.artifact.format.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(st, 12, static_cast<sqlite3_int64>(r.d_model));
            sqlite3_bind_int64(st, 13, static_cast<sqlite3_int64>(r.num_heads));
            sqlite3_bind_int64(st, 14, static_cast<sqlite3_int64>(r.d_ff));
            sqlite3_bind_int64(st, 15, static_cast<sqlite3_int64>(r.num_encoder_layers));
            sqlite3_bind_int64(st, 16, static_cast<sqlite3_int64>(r.num_decoder_layers));
            sqlite3_bind_int64(st, 17, static_cast<sqlite3_int64>(r.max_seq_length));
            const std::string tags_j = tags_to_json(r.tags);
            sqlite3_bind_text(st, 18, tags_j.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(st, 19, r.current_run_number);
            sqlite3_bind_text(st, 20, r.run_started_utc.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 21, r.progress_session_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(st, 22, r.progress_epoch);
            sqlite3_bind_double(st, 23, r.progress_loss);
            sqlite3_bind_double(st, 24, r.progress_best_loss);
            sqlite3_bind_text(st, 25, r.progress_updated_utc.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(st);
            sqlite3_finalize(st);

            // History rows
            for (const auto& h : r.training_history) {
                const char* hsql =
                    "INSERT INTO training_history "
                    "(model_name,run_id,metrics_session_key,dataset_group,epochs,final_loss,"
                    "started_utc,finished_utc,incomplete) "
                    "VALUES (?,?,?,?,?,?,?,?,?)";
                sqlite3_stmt* hs = nullptr;
                if (sqlite3_prepare_v2(db, hsql, -1, &hs, nullptr) == SQLITE_OK) {
                    sqlite3_bind_text(hs, 1, r.model_name.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(hs, 2, h.run_id.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(hs, 3, h.metrics_session_key.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(hs, 4, h.dataset_group.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_int(hs, 5, h.epochs);
                    sqlite3_bind_double(hs, 6, h.final_loss);
                    sqlite3_bind_text(hs, 7, h.started_utc.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_text(hs, 8, h.finished_utc.c_str(), -1, SQLITE_TRANSIENT);
                    sqlite3_bind_int(hs, 9, h.incomplete ? 1 : 0);
                    sqlite3_step(hs);
                    sqlite3_finalize(hs);
                }
            }
            ++imported;
        } catch (...) {
        }
    }

    // Migrate roles.json
    const std::string roles_path = data_dir_ + "/roles.json";
    std::ifstream rf(roles_path);
    if (rf.is_open()) {
        const std::string content(std::istreambuf_iterator<char>(rf),
                                  std::istreambuf_iterator<char>{});
        const auto loaded_roles = parse_string_map(content);
        for (const auto& [role, name] : loaded_roles) {
            sqlite3_stmt* rs = nullptr;
            if (sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO roles VALUES (?,?)", -1, &rs,
                                   nullptr) == SQLITE_OK) {
                sqlite3_bind_text(rs, 1, role.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(rs, 2, name.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_step(rs);
                sqlite3_finalize(rs);
            }
        }
    }

    sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
    Logger::info("ModelNameService: migrated {} records from JSONL to SQLite", imported);
}

void adai::ModelNameService::load_from_disk() {
    sqlite3* db = server_impl_->db;
    if (!db) {
        Logger::warn("ModelNameService: no DB handle; starting empty");
        return;
    }

    // Load models
    sqlite3_stmt* st = nullptr;
    const char* sql =
        "SELECT model_id,model_name,role,state,run_id,created_utc,updated_utc,"
        "artifact_host,artifact_path,artifact_checksum,artifact_format,"
        "d_model,num_heads,d_ff,num_encoder_layers,num_decoder_layers,max_seq_length,tags_json,"
        "current_run_number,run_started_utc,progress_session_id,progress_epoch,progress_loss,"
        "progress_best_loss,progress_updated_utc,run_group "
        "FROM models";
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
        Logger::error("ModelNameService: load query failed: {}", sqlite3_errmsg(db));
        return;
    }
    size_t loaded = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
        ModelRecord r;
        auto col_text = [&](int i) -> std::string {
            const char* p = reinterpret_cast<const char*>(sqlite3_column_text(st, i));
            return p ? p : "";
        };
        r.model_id = col_text(0);
        r.model_name = col_text(1);
        r.role = col_text(2);
        r.state = col_text(3);
        r.run_id = col_text(4);
        r.created_utc = col_text(5);
        r.updated_utc = col_text(6);
        r.artifact.host = col_text(7);
        r.artifact.path = col_text(8);
        r.artifact.checksum = col_text(9);
        r.artifact.format = col_text(10);
        r.d_model = static_cast<size_t>(sqlite3_column_int64(st, 11));
        r.num_heads = static_cast<size_t>(sqlite3_column_int64(st, 12));
        r.d_ff = static_cast<size_t>(sqlite3_column_int64(st, 13));
        r.num_encoder_layers = static_cast<size_t>(sqlite3_column_int64(st, 14));
        r.num_decoder_layers = static_cast<size_t>(sqlite3_column_int64(st, 15));
        r.max_seq_length = static_cast<size_t>(sqlite3_column_int64(st, 16));
        const std::string tags_json = col_text(17);
        const auto tm = parse_string_map(tags_json);
        r.tags = std::map<std::string, std::string>(tm.begin(), tm.end());
        r.current_run_number = sqlite3_column_int(st, 18);
        r.run_started_utc = col_text(19);
        r.progress_session_id = col_text(20);
        r.progress_epoch = sqlite3_column_int(st, 21);
        r.progress_loss = sqlite3_column_double(st, 22);
        r.progress_best_loss = sqlite3_column_double(st, 23);
        r.progress_updated_utc = col_text(24);
        r.run_group = col_text(25);
        if (r.state.empty())
            r.state = "initializing";
        if (r.artifact.format.empty())
            r.artifact.format = "adai-native";
        if (!r.model_name.empty()) {
            models_[r.model_name] = std::move(r);
            ++loaded;
        }
    }
    sqlite3_finalize(st);

    // Load training_history for all models
    const char* hsql =
        "SELECT "
        "model_name,run_id,metrics_session_key,dataset_group,epochs,final_loss,started_utc,"
        "finished_utc,incomplete "
        "FROM training_history ORDER BY id ASC";
    sqlite3_stmt* hs = nullptr;
    if (sqlite3_prepare_v2(db, hsql, -1, &hs, nullptr) == SQLITE_OK) {
        while (sqlite3_step(hs) == SQLITE_ROW) {
            auto col_text = [&](int i) -> std::string {
                const char* p = reinterpret_cast<const char*>(sqlite3_column_text(hs, i));
                return p ? p : "";
            };
            const std::string mn = col_text(0);
            auto it = models_.find(mn);
            if (it == models_.end())
                continue;
            TrainingHistoryEntry h;
            h.run_id = col_text(1);
            h.metrics_session_key = col_text(2);
            h.dataset_group = col_text(3);
            h.epochs = sqlite3_column_int(hs, 4);
            h.final_loss = sqlite3_column_double(hs, 5);
            h.started_utc = col_text(6);
            h.finished_utc = col_text(7);
            h.incomplete = sqlite3_column_int(hs, 8) != 0;
            it->second.training_history.push_back(std::move(h));
        }
        sqlite3_finalize(hs);
    }

    // Load roles
    sqlite3_stmt* rs = nullptr;
    if (sqlite3_prepare_v2(db, "SELECT role,model_name FROM roles", -1, &rs, nullptr) ==
        SQLITE_OK) {
        while (sqlite3_step(rs) == SQLITE_ROW) {
            const char* role = reinterpret_cast<const char*>(sqlite3_column_text(rs, 0));
            const char* name = reinterpret_cast<const char*>(sqlite3_column_text(rs, 1));
            if (role && name)
                roles_[role] = name;
        }
        sqlite3_finalize(rs);
    }

    Logger::info("ModelNameService: loaded {} models, {} roles from SQLite", loaded, roles_.size());
}

void adai::ModelNameService::persist_model(const ModelRecord& rec) {
    sqlite3* db = server_impl_->db;
    if (!db)
        return;

    const char* sql =
        "INSERT OR REPLACE INTO models VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
        Logger::warn("ModelNameService: persist_model prepare failed: {}", sqlite3_errmsg(db));
        return;
    }
    sqlite3_bind_text(st, 1, rec.model_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, rec.model_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 3, rec.role.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 4, rec.state.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 5, rec.run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 6, rec.created_utc.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 7, rec.updated_utc.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 8, rec.artifact.host.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 9, rec.artifact.path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 10, rec.artifact.checksum.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 11, rec.artifact.format.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 12, static_cast<sqlite3_int64>(rec.d_model));
    sqlite3_bind_int64(st, 13, static_cast<sqlite3_int64>(rec.num_heads));
    sqlite3_bind_int64(st, 14, static_cast<sqlite3_int64>(rec.d_ff));
    sqlite3_bind_int64(st, 15, static_cast<sqlite3_int64>(rec.num_encoder_layers));
    sqlite3_bind_int64(st, 16, static_cast<sqlite3_int64>(rec.num_decoder_layers));
    sqlite3_bind_int64(st, 17, static_cast<sqlite3_int64>(rec.max_seq_length));
    const std::string tags_j = tags_to_json(rec.tags);
    sqlite3_bind_text(st, 18, tags_j.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(st, 19, rec.current_run_number);
    sqlite3_bind_text(st, 20, rec.run_started_utc.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 21, rec.progress_session_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(st, 22, rec.progress_epoch);
    sqlite3_bind_double(st, 23, rec.progress_loss);
    sqlite3_bind_double(st, 24, rec.progress_best_loss);
    sqlite3_bind_text(st, 25, rec.progress_updated_utc.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 26, rec.run_group.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st) != SQLITE_DONE) {
        Logger::warn("ModelNameService: persist_model step failed: {}", sqlite3_errmsg(db));
    }
    sqlite3_finalize(st);

    // Rewrite training_history rows for this model (delete + insert is safe at our scale)
    sqlite3_stmt* del = nullptr;
    if (sqlite3_prepare_v2(db, "DELETE FROM training_history WHERE model_name=?", -1, &del,
                           nullptr) == SQLITE_OK) {
        sqlite3_bind_text(del, 1, rec.model_name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_step(del);
        sqlite3_finalize(del);
    }
    const char* hsql =
        "INSERT INTO training_history "
        "(model_name,run_id,metrics_session_key,dataset_group,epochs,final_loss,started_utc,"
        "finished_utc,incomplete) "
        "VALUES (?,?,?,?,?,?,?,?,?)";
    for (const auto& h : rec.training_history) {
        sqlite3_stmt* hs = nullptr;
        if (sqlite3_prepare_v2(db, hsql, -1, &hs, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(hs, 1, rec.model_name.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(hs, 2, h.run_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(hs, 3, h.metrics_session_key.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(hs, 4, h.dataset_group.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(hs, 5, h.epochs);
            sqlite3_bind_double(hs, 6, h.final_loss);
            sqlite3_bind_text(hs, 7, h.started_utc.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(hs, 8, h.finished_utc.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(hs, 9, h.incomplete ? 1 : 0);
            sqlite3_step(hs);
            sqlite3_finalize(hs);
        }
    }
}

void adai::ModelNameService::persist_roles() {
    sqlite3* db = server_impl_->db;
    if (!db)
        return;
    sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "DELETE FROM roles", nullptr, nullptr, nullptr);
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db, "INSERT INTO roles VALUES (?,?)", -1, &st, nullptr) == SQLITE_OK) {
        for (const auto& [role, name] : roles_) {
            sqlite3_reset(st);
            sqlite3_bind_text(st, 1, role.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 2, name.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(st);
        }
        sqlite3_finalize(st);
    }
    sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
}

void adai::ModelNameService::rewrite_models_jsonl() {
    // Phase 2: deletion is handled in handle_delete via SQLite directly.
    // This method is kept for interface compatibility but is a no-op; the
    // in-memory map was already updated before this call.
}

// ============================================================================
// Handler: POST /models
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_register(const std::string& body) {
    const std::string model_name = json_string(body, "model_name");
    if (model_name.empty())
        return {400, "{\"error\":\"model_name required\"}"};
    if (!valid_name(model_name))
        return {400,
                "{\"error\":\"invalid model_name: must match [a-z0-9][a-z0-9\\-\\.]{1,127}\"}"};

    std::unique_lock lock(mutex_);
    if (models_.count(model_name))
        return {409, "{\"error\":\"model_name already registered\"}"};

    ModelRecord r;
    r.model_id = generate_uuid();
    r.model_name = model_name;
    r.role = json_string(body, "role");
    r.run_group = json_string(body, "run_group");  // optional; empty = not yet migrated to MNS-sourced group
    r.state = "initializing";
    r.created_utc = utc_now();
    r.updated_utc = r.created_utc;

    const auto arch = extract_object(body, "arch");
    if (arch != "{}") {
        r.d_model = static_cast<size_t>(json_int(arch, "d_model"));
        r.num_heads = static_cast<size_t>(json_int(arch, "num_heads"));
        r.d_ff = static_cast<size_t>(json_int(arch, "d_ff"));
        r.num_encoder_layers = static_cast<size_t>(json_int(arch, "num_encoder_layers"));
        r.num_decoder_layers = static_cast<size_t>(json_int(arch, "num_decoder_layers"));
        r.max_seq_length = static_cast<size_t>(json_int(arch, "max_seq_length"));
    }
    r.tags = parse_string_map(extract_object(body, "tags"));

    models_[model_name] = r;
    persist_model(r);

    std::ostringstream resp;
    resp << "{\"model_id\":\"" << json_escape(r.model_id) << "\",\"state\":\"initializing\"}";
    Logger::info("ModelNameService: registered '{}' (id={})", model_name, r.model_id);
    return {201, resp.str()};
}

// ============================================================================
// Handler: GET /models
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_list(const std::string& state_filter,
                                                                const std::string& role_filter,
                                                                int limit) {
    std::shared_lock lock(mutex_);
    std::ostringstream j;
    j << "{\"models\":[";
    bool first = true;
    int count = 0;
    for (const auto& [name, r] : models_) {
        if (!state_filter.empty() && r.state != state_filter)
            continue;
        if (!role_filter.empty() && r.role != role_filter)
            continue;
        if (count >= limit)
            break;
        if (!first)
            j << ',';
        first = false;
        j << serialize_record(r);
        ++count;
    }
    j << "]}";
    return {200, j.str()};
}

// ============================================================================
// Handler: GET /models/{name}
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_get(const std::string& name) {
    std::shared_lock lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};
    return {200, serialize_record(it->second)};
}

// ============================================================================
// Handler: GET /models/{name}/resolve
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_resolve(const std::string& name) {
    std::shared_lock lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};
    if (it->second.state == "initializing")
        return {409, "{\"error\":\"model has no artifact yet (state: initializing)\"}"};
    return {200, resolve_json(it->second)};
}

// ============================================================================
// Handler: PUT /models/{name}/state
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_state_transition(
    const std::string& name, const std::string& body) {
    const std::string new_state = json_string(body, "state");
    if (new_state.empty())
        return {400, "{\"error\":\"state required\"}"};

    std::unique_lock lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};

    ModelRecord& r = it->second;
    const std::string cur = r.state;

    // State machine: validate the requested transition
    bool valid = false;
    if (new_state == "training") {
        // Can start training from initializing (first run), candidate (retrain),
        // or training itself — the client no longer asserts its own run_id (see
        // below), so a second "training" call while already "training" is
        // always accepted and treated as "the previous run was abandoned/
        // crashed" rather than a locking conflict.
        valid = (cur == "initializing" || cur == "candidate" || cur == "training");
    } else if (new_state == "candidate") {
        // Normal completion of training; also allows imported models (initializing→candidate)
        // and revival of retired models
        valid = (cur == "training" || cur == "initializing" || cur == "retired");
    } else if (new_state == "retired") {
        valid = (cur != "retired");
    } else {
        return {400,
                "{\"error\":\"unknown target state; valid values: training, candidate, retired\"}"};
    }

    if (!valid) {
        std::ostringstream err;
        err << "{\"error\":\"invalid transition: " << json_escape(cur) << " -> "
            << json_escape(new_state) << "\"}";
        return {409, err.str()};
    }

    const std::string run_id = json_string(body, "run_id");

    if (new_state == "training") {
        // Client no longer asserts run_id — MNS allocates it (see CLAUDE.md
        // "Configuration": MNS/registry are the definitive standard). "training"
        // -> "training" means a previous run never reached "candidate" (killed
        // or crashed); archive its last-known progress before reallocating.
        if (cur == "training" && (r.progress_epoch > 0 || !r.progress_session_id.empty())) {
            TrainingHistoryEntry h;
            h.run_id = r.run_id;
            h.metrics_session_key = r.progress_session_id;
            h.epochs = r.progress_epoch;
            h.final_loss = r.progress_loss;
            h.started_utc = r.run_started_utc;
            h.finished_utc = utc_now();
            h.incomplete = true;
            r.training_history.push_back(std::move(h));
        }
        r.progress_session_id.clear();
        r.progress_epoch = 0;
        r.progress_loss = 0.0;
        r.progress_best_loss = 0.0;
        r.progress_updated_utc.clear();

        const bool new_run = json_bool(body, "new_run", false);
        if (r.current_run_number == 0) {
            r.current_run_number = 1;  // bootstrap: first-ever training, regardless of new_run
        } else if (new_run) {
            r.current_run_number += 1;  // retrain: fresh run
        }
        // else: continuing (train/resume) — keep current_run_number as-is
        r.run_id = "run-" + zero_pad2(r.current_run_number);
        r.run_started_utc = utc_now();
    } else if (new_state == "candidate") {
        // Attach new artifact location if provided
        const auto art_obj = extract_object(body, "artifact");
        if (art_obj != "{}")
            r.artifact = parse_artifact(art_obj);

        // Append training history entry
        const auto summary = extract_object(body, "training_summary");
        TrainingHistoryEntry h;
        h.run_id = run_id.empty() ? r.run_id : run_id;
        h.metrics_session_key = json_string(body, "metrics_session_key");
        h.dataset_group = json_string(summary, "dataset_group");
        h.epochs = json_int(summary, "epochs");
        h.final_loss = json_double(summary, "final_loss");
        h.started_utc = json_string(summary, "started_utc");
        h.finished_utc = json_string(summary, "finished_utc");
        if (h.finished_utc.empty())
            h.finished_utc = utc_now();
        if (!h.run_id.empty())
            r.training_history.push_back(std::move(h));

        r.run_id.clear();

        // The run reached candidate normally — the live progress snapshot is
        // now redundant with the training_history entry just appended.
        r.progress_session_id.clear();
        r.progress_epoch = 0;
        r.progress_loss = 0.0;
        r.progress_best_loss = 0.0;
        r.progress_updated_utc.clear();
    }

    r.state = new_state;
    r.updated_utc = utc_now();
    persist_model(r);
    Logger::info("ModelNameService: '{}' state {} -> {}", name, cur, new_state);
    return {200, serialize_record(r)};
}

// ============================================================================
// Handler: PUT /models/{name}/progress
//
// Pushed after every training epoch (see IncrementalTrainer's epoch callback)
// so a killed/crashed trainer still leaves an accurate last-known state —
// see CLAUDE.md "Configuration" / handle_state_transition's "training" branch,
// which archives this snapshot into training_history (incomplete=true) if a
// new run starts before the current one ever reaches "candidate".
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_progress_update(
    const std::string& name, const std::string& body) {
    std::unique_lock lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};

    ModelRecord& r = it->second;
    if (r.state != "training")
        return {409, "{\"error\":\"model is not currently training\"}"};

    const std::string run_id = json_string(body, "run_id");
    if (run_id.empty() || run_id != r.run_id) {
        return {409,
                "{\"error\":\"run_id does not match the active run; this trainer's run has been "
                "superseded\"}"};
    }

    r.progress_session_id = json_string(body, "session_id");
    r.progress_epoch = json_int(body, "epoch", r.progress_epoch);
    r.progress_loss = json_double(body, "loss", r.progress_loss);
    r.progress_best_loss = json_double(body, "best_loss", r.progress_best_loss);
    r.progress_updated_utc = utc_now();
    r.updated_utc = r.progress_updated_utc;
    persist_model(r);
    return {200, "{\"status\":\"ok\"}"};
}

// ============================================================================
// Handler: PUT /models/{name}/run_group
//
// Deliberately not gated by state (unlike handle_progress_update, which
// requires "training") — run_group is static routing metadata, valid to set
// on a model in any state, including one that's already trained/production.
// No run_id/ownership check either, since it isn't tied to a specific run.
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_update_run_group(
    const std::string& name, const std::string& body) {
    std::unique_lock lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};

    ModelRecord& r = it->second;
    r.run_group = json_string(body, "run_group");
    r.updated_utc = utc_now();
    persist_model(r);
    Logger::info("ModelNameService: run_group for '{}' set to '{}'", name, r.run_group);
    return {200, "{\"status\":\"ok\",\"run_group\":\"" + json_escape(r.run_group) + "\"}"};
}

// ============================================================================
// Handler: DELETE /models/{name}
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_delete(const std::string& name) {
    std::unique_lock lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};

    const std::string& s = it->second.state;
    if (s == "training" || s == "candidate" || s == "production")
        return {409, "{\"error\":\"can only delete models in initializing or retired state\"}"};

    models_.erase(it);
    for (auto rit = roles_.begin(); rit != roles_.end();) {
        if (rit->second == name)
            rit = roles_.erase(rit);
        else
            ++rit;
    }

    // SQLite hard-delete
    sqlite3* db = server_impl_->db;
    if (db) {
        sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);
        sqlite3_stmt* sd = nullptr;
        if (sqlite3_prepare_v2(db, "DELETE FROM models WHERE model_name=?", -1, &sd, nullptr) ==
            SQLITE_OK) {
            sqlite3_bind_text(sd, 1, name.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(sd);
            sqlite3_finalize(sd);
        }
        sqlite3_stmt* shd = nullptr;
        if (sqlite3_prepare_v2(db, "DELETE FROM training_history WHERE model_name=?", -1, &shd,
                               nullptr) == SQLITE_OK) {
            sqlite3_bind_text(shd, 1, name.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(shd);
            sqlite3_finalize(shd);
        }
        sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
    }
    persist_roles();

    std::ostringstream j;
    j << "{\"deleted\":true,\"model_name\":\"" << json_escape(name) << "\"}";
    Logger::info("ModelNameService: deleted '{}'", name);
    return {200, j.str()};
}

// ============================================================================
// Handler: GET /roles
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_list_roles() {
    std::shared_lock lock(mutex_);
    std::ostringstream j;
    j << "{\"roles\":[";
    bool first = true;
    for (const auto& [role, model_name] : roles_) {
        if (!first)
            j << ',';
        first = false;
        j << "{\"role\":\"" << json_escape(role) << "\",\"production_model\":\""
          << json_escape(model_name) << "\"}";
    }
    j << "]}";
    return {200, j.str()};
}

// ============================================================================
// Handler: GET /roles/{role}/production
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_resolve_role(const std::string& role) {
    std::shared_lock lock(mutex_);
    const auto rit = roles_.find(role);
    if (rit == roles_.end())
        return {404, "{\"error\":\"no production model for role\"}"};
    const auto mit = models_.find(rit->second);
    if (mit == models_.end())
        return {404, "{\"error\":\"production model record not found\"}"};
    return {200, resolve_json(mit->second)};
}

// ============================================================================
// Handler: PUT /roles/{role}/production
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_promote(const std::string& role,
                                                                   const std::string& body) {
    if (!valid_name(role))
        return {400, "{\"error\":\"invalid role name\"}"};
    const std::string model_name = json_string(body, "model_name");
    if (model_name.empty())
        return {400, "{\"error\":\"model_name required\"}"};

    std::unique_lock lock(mutex_);
    const auto it = models_.find(model_name);
    if (it == models_.end())
        return {404, "{\"error\":\"model not found\"}"};
    if (it->second.state != "candidate")
        return {409, "{\"error\":\"model must be in candidate state to promote to production\"}"};

    // Retire the previous production model for this role (if any)
    std::string retired_name;
    const auto rit = roles_.find(role);
    if (rit != roles_.end() && !rit->second.empty() && rit->second != model_name) {
        const auto prev = models_.find(rit->second);
        if (prev != models_.end() && prev->second.state == "production") {
            retired_name = rit->second;
            prev->second.state = "retired";
            prev->second.updated_utc = utc_now();
            persist_model(prev->second);
        }
    }

    it->second.state = "production";
    it->second.role = role;
    it->second.updated_utc = utc_now();
    roles_[role] = model_name;
    persist_model(it->second);
    persist_roles();

    std::ostringstream j;
    j << "{\"promoted\":\"" << json_escape(model_name) << "\",\"retired\":\""
      << json_escape(retired_name) << "\",\"role\":\"" << json_escape(role) << "\"}";
    Logger::info("ModelNameService: promoted '{}' to role='{}' (retired='{}')", model_name, role,
                 retired_name);
    return {200, j.str()};
}

// ============================================================================
// Handler: GET /health
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_health() {
    const auto now = std::chrono::steady_clock::now();
    const auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    size_t count;
    {
        std::shared_lock lock(mutex_);
        count = models_.size();
    }
    std::ostringstream j;
    j << "{\"status\":\"ok\",\"model_count\":" << count << ",\"uptime_seconds\":" << uptime << "}";
    return {200, j.str()};
}

// ============================================================================
// Handler: GET /models/{name}/datasets  (Phase 2 — proxies to registry /history)
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_datasets(const std::string& name) {
    if (registry_url_.empty())
        return {
            501,
            "{\"error\":\"registry_url not configured; start mns_server with --registry-url\"}"};

    std::string model_id;
    {
        std::shared_lock lock(mutex_);
        const auto it = models_.find(name);
        if (it == models_.end())
            return {404, "{\"error\":\"model not found\"}"};
        model_id = it->second.model_id;
    }

    // Parse the registry URL for httplib
    std::string host;
    int port = 80;
    std::string path_prefix;
    {
        std::string s = registry_url_;
        if (s.rfind("http://", 0) == 0)
            s = s.substr(7);
        else if (s.rfind("https://", 0) == 0) {
            s = s.substr(8);
            port = 443;
        }
        const auto slash = s.find('/');
        std::string authority;
        if (slash != std::string::npos) {
            authority = s.substr(0, slash);
            path_prefix = s.substr(slash);
        } else {
            authority = s;
        }
        while (!path_prefix.empty() && path_prefix.back() == '/')
            path_prefix.pop_back();
        const auto colon = authority.find(':');
        if (colon != std::string::npos) {
            host = authority.substr(0, colon);
            try {
                port = std::stoi(authority.substr(colon + 1));
            } catch (...) {
            }
        } else {
            host = authority;
        }
    }

    const std::string full_path =
        path_prefix + "/registry/" + registry_group_ + "/history?model_id=" + model_id;

    try {
        httplib::Client c(host, port);
        c.set_connection_timeout(5, 0);
        c.set_read_timeout(5, 0);
        auto res = c.Get(full_path);
        if (!res)
            return {502, "{\"error\":\"registry server unreachable\"}"};
        return {res->status, res->body};
    } catch (const std::exception& e) {
        return {502, std::string("{\"error\":\"registry proxy failed: ") + json_escape(e.what()) +
                         "\"}"};
    }
}

// ============================================================================
// Handler: GET/PUT /admin/config
//
// Only registry_url/registry_group are live-mutable — port/data_dir are baked
// into the already-bound listener socket and already-opened SQLite handles,
// so changing them here would require a restart anyway; they stay file/CLI-only.
// PUT persists accepted keys to daemon_config.db (DaemonConfigStore), which
// overlays config.mns.conf on the next restart. See CLAUDE.md "Daemon admin
// config API".
// ============================================================================

std::pair<int, std::string> adai::ModelNameService::handle_admin_get_config() {
    std::ostringstream j;
    j << "{\"port\":" << port_ << ",\"data_dir\":\"" << json_escape(data_dir_) << "\""
      << ",\"registry_url\":\"" << json_escape(registry_url_) << "\""
      << ",\"registry_group\":\"" << json_escape(registry_group_) << "\""
      << ",\"admin_enabled\":" << (admin_enabled_ ? "true" : "false") << "}";
    return {200, j.str()};
}

std::pair<int, std::string> adai::ModelNameService::handle_admin_put_config(
    const std::string& body) {
    if (!admin_enabled_) {
        return {403, "{\"error\":\"admin config mutation disabled (--admin-enabled=false)\"}"};
    }
    if (body.find("\"port\"") != std::string::npos ||
        body.find("\"data_dir\"") != std::string::npos) {
        return {400,
                "{\"error\":\"port and data_dir are immutable at runtime; set them via "
                "config.mns.conf or --port/--data-dir and restart\"}"};
    }

    bool changed = false;
    if (json_has_string_key(body, "registry_url")) {
        registry_url_ = json_string(body, "registry_url");
        changed = true;
    }
    if (json_has_string_key(body, "registry_group")) {
        const std::string group = json_string(body, "registry_group");
        registry_group_ = group.empty() ? "default" : group;
        changed = true;
    }
    if (!changed) {
        return {400,
                "{\"error\":\"no recognized mutable keys in body (registry_url, "
                "registry_group)\"}"};
    }

    if (config_store_) {
        config_store_->set("registry_url", registry_url_);
        config_store_->set("registry_group", registry_group_);
    }
    Logger::info("ModelNameService: admin config updated (registry_url={}, registry_group={})",
                registry_url_, registry_group_);
    return handle_admin_get_config();
}
