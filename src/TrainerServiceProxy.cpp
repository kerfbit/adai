// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-14

#include "TrainerServiceProxy.hpp"
#include <httplib.h>
#include <sstream>
#include "DaemonConfigStore.hpp"
#include "Logger.hpp"

// ============================================================================
// ServerImpl (hides httplib from the header — same pimpl pattern as
// TrainerAdminAPI/ModelNameService/RegistryServer)
// ============================================================================

class adai::TrainerServiceProxy::ServerImpl {
   public:
    httplib::Server server;
};

// ============================================================================
// Internal JSON helpers — deliberately duplicated per-daemon rather than shared, matching
// TrainerAdminAPI.cpp's own comment on this: there is no shared JSON-parsing header in this
// codebase's daemons (see CLAUDE.md).
// ============================================================================

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
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

bool json_bool(const std::string& body, const std::string& key, bool def = false) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    const auto start = pos + needle.size();
    return body.compare(start, 4, "true") == 0;
}

bool json_has_key(const std::string& body, const std::string& key) {
    return body.find("\"" + key + "\"") != std::string::npos;
}

// A proxied request whose connection itself failed (child mid-start or mid-exit) gets this,
// distinct from the "no child at all" idle default below — same HTTP status, different message,
// so an operator can tell the two situations apart from the response body alone.
constexpr int kChildUnreachableStatus = 503;
std::string child_unreachable_json() {
    return "{\"error\":\"training pass is starting or exiting; try again shortly\"}";
}

}  // namespace

// ============================================================================
// Construction / route registration
// ============================================================================

adai::TrainerServiceProxy::TrainerServiceProxy(std::string host, int port,
                                               std::string child_admin_dir)
    : host_(std::move(host)),
      port_(port),
      child_admin_dir_(std::move(child_admin_dir)),
      server_impl_(std::make_unique<ServerImpl>()) {
    auto& svr = server_impl_->server;

    // Never proxied — the supervisor itself is "healthy" as long as it can answer at all,
    // regardless of whether a child happens to be running right now.
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/admin/config", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] =
            child_port_.load() != 0 ? proxy_get("/admin/config") : handle_get_config_idle();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Put("/admin/config", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = child_port_.load() != 0 ? proxy_put("/admin/config", req.body)
                                                       : handle_put_config_idle(req.body);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/status", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] =
            child_port_.load() != 0 ? proxy_get("/admin/status") : std::pair{200, idle_status_json()};
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/logs", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] =
            child_port_.load() != 0 ? proxy_get("/admin/logs") : std::pair{200, std::string("{\"entries\":[]}")};
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/checkpoint", [this](const httplib::Request& req, httplib::Response& res) {
        std::string path = "/admin/checkpoint";
        if (req.has_param("wait_ms")) {
            path += "?wait_ms=" + req.get_param_value("wait_ms");
        }
        auto [status, body] = child_port_.load() != 0
            ? proxy_post(path, "")
            : std::pair{409, std::string("{\"error\":\"no active training pass; nothing to checkpoint\"}")};
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/pause", [this](const httplib::Request&, httplib::Response& res) {
        // Pausing while idle is meaningless (no pass to drain) but harmless to accept —
        // mirrors TrainerAdminAPI::handle_pause() never rejecting based on phase either.
        auto [status, body] = child_port_.load() != 0
            ? proxy_post("/admin/pause", "")
            : std::pair{202, std::string("{\"paused\":true}")};
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/resume", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = child_port_.load() != 0
            ? proxy_post("/admin/resume", "")
            : std::pair{202, std::string("{\"paused\":false}")};
        res.status = status;
        res.set_content(body, "application/json");
    });
}

adai::TrainerServiceProxy::~TrainerServiceProxy() {
    stop();
}

void adai::TrainerServiceProxy::set_child_port(int port) {
    child_port_ = port;
}

bool adai::TrainerServiceProxy::start() {
    running_ = true;
    Logger::info("Trainer service admin API listening on {}:{}", host_, port_);
    const bool ok = server_impl_->server.listen(host_, port_);
    running_ = false;
    return ok;
}

void adai::TrainerServiceProxy::stop() {
    if (server_impl_) {
        server_impl_->server.stop();
    }
    running_ = false;
}

bool adai::TrainerServiceProxy::is_running() const {
    return running_.load();
}

// ============================================================================
// Proxying to the current child
// ============================================================================

std::pair<int, std::string> adai::TrainerServiceProxy::proxy_get(const std::string& path) {
    const int port = child_port_.load();
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);  // 1s — a live child's admin API answers near-instantly
    cli.set_read_timeout(5, 0);        // 5s — generous for e.g. a large /admin/logs response
    auto res = cli.Get(path);
    if (!res) {
        return {kChildUnreachableStatus, child_unreachable_json()};
    }
    return {res->status, res->body};
}

std::pair<int, std::string> adai::TrainerServiceProxy::proxy_post(const std::string& path,
                                                                  const std::string& body) {
    const int port = child_port_.load();
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(5, 0);
    auto res = cli.Post(path, body, "application/json");
    if (!res) {
        return {kChildUnreachableStatus, child_unreachable_json()};
    }
    return {res->status, res->body};
}

std::pair<int, std::string> adai::TrainerServiceProxy::proxy_put(const std::string& path,
                                                                 const std::string& body) {
    const int port = child_port_.load();
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(5, 0);
    auto res = cli.Put(path, body, "application/json");
    if (!res) {
        return {kChildUnreachableStatus, child_unreachable_json()};
    }
    return {res->status, res->body};
}

// ============================================================================
// Idle fallbacks (no child running at all)
// ============================================================================

std::string adai::TrainerServiceProxy::idle_status_json() {
    // Same shape TrainerAdminAPI::handle_status() reports, with TrainerControlState's own
    // just-constructed defaults for every field — this IS what a fresh TrainerControlState would
    // report before any pass ever touched it, so no separate "idle schema" is being invented here.
    return "{\"phase\":\"idle\""
          ",\"paused\":false"
          ",\"run_id\":\"\""
          ",\"session_id\":\"\""
          ",\"model_name\":\"\""
          ",\"current_epoch\":0"
          ",\"total_epochs\":0"
          ",\"samples_trained_this_pass\":0"
          ",\"last_loss\":0"
          ",\"best_loss\":0"
          ",\"checkpoints_written\":0"
          ",\"last_checkpoint_path\":\"\""
          ",\"last_checkpoint_time_unix\":0}";
}

std::pair<int, std::string> adai::TrainerServiceProxy::handle_get_config_idle() {
    // Same defaults as TrainerControlState's own field initializers (TrainerControlState.hpp),
    // overlaid with whatever a previous pass (or a previous idle PUT, see below) persisted to the
    // shared daemon_config.db — mirrors TrainerAdminAPI::start()'s own overlay-on-construction
    // logic, just performed directly here instead of via a live TrainerControlState.
    bool auto_save_enabled = true;
    int auto_save_every_samples = 1000;
    int auto_save_every_minutes = 30;
    int max_sessions_to_keep = 50;

    try {
        DaemonConfigStore store(child_admin_dir_ + "/daemon_config.db");
        const auto overrides = store.load_all();
        if (auto it = overrides.find("auto_save_enabled"); it != overrides.end()) {
            auto_save_enabled = (it->second == "true" || it->second == "1");
        }
        if (auto it = overrides.find("auto_save_every_samples"); it != overrides.end()) {
            try {
                auto_save_every_samples = std::stoi(it->second);
            } catch (...) {
            }
        }
        if (auto it = overrides.find("auto_save_every_minutes"); it != overrides.end()) {
            try {
                auto_save_every_minutes = std::stoi(it->second);
            } catch (...) {
            }
        }
        if (auto it = overrides.find("max_sessions_to_keep"); it != overrides.end()) {
            try {
                max_sessions_to_keep = std::stoi(it->second);
            } catch (...) {
            }
        }
    } catch (const std::exception& e) {
        Logger::warn("TrainerServiceProxy: daemon_config.db unavailable while idle ({})", e.what());
    }

    std::ostringstream j;
    j << "{\"auto_save_enabled\":" << (auto_save_enabled ? "true" : "false")
      << ",\"auto_save_every_samples\":" << auto_save_every_samples
      << ",\"auto_save_every_minutes\":" << auto_save_every_minutes
      << ",\"max_sessions_to_keep\":" << max_sessions_to_keep << "}";
    return {200, j.str()};
}

std::pair<int, std::string> adai::TrainerServiceProxy::handle_put_config_idle(
    const std::string& body) {
    // Deliberately mirrors TrainerAdminAPI::handle_put_config()'s validation for the same four
    // mutable keys — no shared validator exists to call into (see the JSON-helper comment above),
    // and this set is small and stable enough that keeping the two in sync by hand is reasonable.
    static const char* kImmutableKeys[] = {"port", "host", "dir", "enabled"};
    for (const auto* key : kImmutableKeys) {
        if (json_has_key(body, key)) {
            return {400, std::string("{\"error\":\"'") + key +
                            "' is immutable at runtime; set it via config.trainer.conf "
                            "(TRAINER_ADMIN_*) and restart\"}"};
        }
    }

    try {
        DaemonConfigStore store(child_admin_dir_ + "/daemon_config.db");
        bool changed = false;
        if (json_has_key(body, "auto_save_enabled")) {
            store.set("auto_save_enabled", json_bool(body, "auto_save_enabled", true) ? "true" : "false");
            changed = true;
        }
        if (json_has_key(body, "auto_save_every_samples")) {
            const int v = json_int(body, "auto_save_every_samples", -1);
            if (v < 0) {
                return {400, "{\"error\":\"'auto_save_every_samples' must be >= 0 (0 disables the "
                            "sample-count trigger)\"}"};
            }
            store.set("auto_save_every_samples", std::to_string(v));
            changed = true;
        }
        if (json_has_key(body, "auto_save_every_minutes")) {
            const int v = json_int(body, "auto_save_every_minutes", -1);
            if (v < 0) {
                return {400, "{\"error\":\"'auto_save_every_minutes' must be >= 0 (0 disables the "
                            "time-based trigger)\"}"};
            }
            store.set("auto_save_every_minutes", std::to_string(v));
            changed = true;
        }
        if (json_has_key(body, "max_sessions_to_keep")) {
            const int v = json_int(body, "max_sessions_to_keep", -1);
            if (v < 1) {
                return {400, "{\"error\":\"'max_sessions_to_keep' must be >= 1\"}"};
            }
            store.set("max_sessions_to_keep", std::to_string(v));
            changed = true;
        }
        if (!changed) {
            return {400,
                    "{\"error\":\"no recognized mutable keys in body (auto_save_enabled, "
                    "auto_save_every_samples, auto_save_every_minutes, max_sessions_to_keep)\"}"};
        }
    } catch (const std::exception& e) {
        return {500, std::string("{\"error\":\"daemon_config.db unavailable while idle (") +
                        json_escape(e.what()) + ")\"}"};
    }

    return handle_get_config_idle();
}
