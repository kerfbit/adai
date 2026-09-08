// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "TrainerAdminAPI.hpp"
#include <httplib.h>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include "DaemonConfigStore.hpp"
#include "Logger.hpp"

namespace fs = std::filesystem;

// ============================================================================
// ServerImpl (hides httplib from the header — same pimpl pattern as
// ModelNameService/RegistryServer)
// ============================================================================

class adai::TrainerAdminAPI::ServerImpl {
   public:
    httplib::Server server;
};

// ============================================================================
// Internal JSON helpers — deliberately duplicated per-daemon rather than
// shared (matches the existing convention in ModelNameService.cpp/
// RegistryServer.cpp; there is no shared JSON-parsing header in this
// codebase's daemons — see CLAUDE.md).
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

// Bare-literal boolean (true/false, no quotes).
bool json_bool(const std::string& body, const std::string& key, bool def = false) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return def;
    const auto start = pos + needle.size();
    return body.compare(start, 4, "true") == 0;
}

// Presence check — distinguishes "key not sent" from "sent with its
// zero/default value".
bool json_has_key(const std::string& body, const std::string& key) {
    return body.find("\"" + key + "\"") != std::string::npos;
}

}  // namespace

// ============================================================================
// Construction / route registration
// ============================================================================

adai::TrainerAdminAPI::TrainerAdminAPI(std::shared_ptr<TrainerControlState> control,
                                       std::string host, int port, std::string config_store_dir)
    : control_(std::move(control)),
      host_(std::move(host)),
      port_(port),
      config_store_dir_(std::move(config_store_dir)),
      server_impl_(std::make_unique<ServerImpl>()) {
    auto& svr = server_impl_->server;

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_health();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/config", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_get_config();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Put("/admin/config", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = handle_put_config(req.body);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/status", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_status();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/logs", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_logs();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/checkpoint", [this](const httplib::Request& req, httplib::Response& res) {
        int wait_ms = 0;
        if (req.has_param("wait_ms")) {
            try {
                wait_ms = std::stoi(req.get_param_value("wait_ms"));
            } catch (...) {
            }
        }
        auto [status, body] = handle_checkpoint(wait_ms);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/pause", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_pause();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/resume", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] = handle_resume();
        res.status = status;
        res.set_content(body, "application/json");
    });
}

adai::TrainerAdminAPI::~TrainerAdminAPI() {
    stop();
}

bool adai::TrainerAdminAPI::start() {
    if (!config_store_dir_.empty()) {
        try {
            fs::create_directories(config_store_dir_);
            config_store_ = std::make_unique<DaemonConfigStore>(config_store_dir_ +
                                                                 "/daemon_config.db");
            // Overlay any persisted tunables onto the control state — file/
            // constructor-seeded defaults < persisted admin overrides, same
            // precedence every other daemon's admin config follows.
            const auto overrides = config_store_->load_all();
            if (auto it = overrides.find("auto_save_enabled"); it != overrides.end()) {
                control_->auto_save_enabled = (it->second == "true" || it->second == "1");
            }
            if (auto it = overrides.find("auto_save_every_samples"); it != overrides.end()) {
                try {
                    control_->auto_save_every_samples = std::stoi(it->second);
                } catch (...) {
                }
            }
            if (auto it = overrides.find("auto_save_every_minutes"); it != overrides.end()) {
                try {
                    control_->auto_save_every_minutes = std::stoi(it->second);
                } catch (...) {
                }
            }
            if (auto it = overrides.find("max_sessions_to_keep"); it != overrides.end()) {
                try {
                    control_->max_sessions_to_keep = std::stoi(it->second);
                } catch (...) {
                }
            }
        } catch (const std::exception& e) {
            control_->log(TrainerLogLevel::Warn,
                          std::string("daemon_config.db unavailable (") + e.what() +
                              "); admin config changes won't persist across restarts");
        }
    }

    running_ = true;
    control_->log(TrainerLogLevel::Info, "Admin API listening on " + host_ + ":" + std::to_string(port_));
    const bool ok = server_impl_->server.listen(host_, port_);
    running_ = false;
    return ok;
}

void adai::TrainerAdminAPI::stop() {
    if (server_impl_) {
        server_impl_->server.stop();
    }
    running_ = false;
}

bool adai::TrainerAdminAPI::is_running() const {
    return running_.load();
}

int adai::TrainerAdminAPI::get_port() const {
    return port_;
}

// ============================================================================
// Handlers
// ============================================================================

std::pair<int, std::string> adai::TrainerAdminAPI::handle_health() {
    return {200, "{\"status\":\"ok\"}"};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_get_config() {
    std::ostringstream j;
    j << "{\"auto_save_enabled\":" << (control_->auto_save_enabled.load() ? "true" : "false")
      << ",\"auto_save_every_samples\":" << control_->auto_save_every_samples.load()
      << ",\"auto_save_every_minutes\":" << control_->auto_save_every_minutes.load()
      << ",\"max_sessions_to_keep\":" << control_->max_sessions_to_keep.load() << "}";
    return {200, j.str()};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_put_config(const std::string& body) {
    // port/host/dir are baked into the already-bound listener socket and the
    // already-opened daemon_config.db handle, so changing them here would
    // require a restart anyway; they stay file/CLI-only (TRAINER_ADMIN_PORT/
    // TRAINER_ADMIN_HOST/TRAINER_ADMIN_DIR in config.trainer.conf).
    static const char* kImmutableKeys[] = {"port", "host", "dir", "enabled"};
    for (const auto* key : kImmutableKeys) {
        if (json_has_key(body, key)) {
            return {400, std::string("{\"error\":\"'") + key +
                            "' is immutable at runtime; set it via config.trainer.conf "
                            "(TRAINER_ADMIN_*) and restart\"}"};
        }
    }

    bool changed = false;
    if (json_has_key(body, "auto_save_enabled")) {
        control_->auto_save_enabled = json_bool(body, "auto_save_enabled", true);
        changed = true;
    }
    if (json_has_key(body, "auto_save_every_samples")) {
        const int v = json_int(body, "auto_save_every_samples", -1);
        if (v < 0) {
            return {400, "{\"error\":\"'auto_save_every_samples' must be >= 0 (0 disables the "
                        "sample-count trigger)\"}"};
        }
        control_->auto_save_every_samples = v;
        changed = true;
    }
    if (json_has_key(body, "auto_save_every_minutes")) {
        const int v = json_int(body, "auto_save_every_minutes", -1);
        if (v < 0) {
            return {400, "{\"error\":\"'auto_save_every_minutes' must be >= 0 (0 disables the "
                        "time-based trigger)\"}"};
        }
        control_->auto_save_every_minutes = v;
        changed = true;
    }
    if (json_has_key(body, "max_sessions_to_keep")) {
        const int v = json_int(body, "max_sessions_to_keep", -1);
        if (v < 1) {
            return {400, "{\"error\":\"'max_sessions_to_keep' must be >= 1\"}"};
        }
        control_->max_sessions_to_keep = v;
        changed = true;
    }

    if (!changed) {
        return {400,
                "{\"error\":\"no recognized mutable keys in body (auto_save_enabled, "
                "auto_save_every_samples, auto_save_every_minutes, max_sessions_to_keep)\"}"};
    }

    if (config_store_) {
        config_store_->set("auto_save_enabled",
                           control_->auto_save_enabled.load() ? "true" : "false");
        config_store_->set("auto_save_every_samples",
                           std::to_string(control_->auto_save_every_samples.load()));
        config_store_->set("auto_save_every_minutes",
                           std::to_string(control_->auto_save_every_minutes.load()));
        config_store_->set("max_sessions_to_keep",
                           std::to_string(control_->max_sessions_to_keep.load()));
    }
    {
        std::ostringstream msg;
        msg << "Admin config updated via PUT /admin/config (auto_save_enabled="
            << (control_->auto_save_enabled.load() ? "true" : "false")
            << ", auto_save_every_samples=" << control_->auto_save_every_samples.load()
            << ", auto_save_every_minutes=" << control_->auto_save_every_minutes.load()
            << ", max_sessions_to_keep=" << control_->max_sessions_to_keep.load() << ")";
        control_->log(TrainerLogLevel::Info, msg.str());
    }
    return handle_get_config();
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_status() {
    std::ostringstream j;
    j << "{\"phase\":\"" << to_string(control_->phase.load()) << "\""
      << ",\"paused\":" << (control_->paused.load() ? "true" : "false")
      << ",\"run_id\":\"" << json_escape(control_->get_run_id()) << "\""
      << ",\"session_id\":\"" << json_escape(control_->get_session_id()) << "\""
      << ",\"model_name\":\"" << json_escape(control_->get_model_name()) << "\""
      << ",\"current_epoch\":" << control_->current_epoch.load()
      << ",\"total_epochs\":" << control_->total_epochs.load()
      << ",\"samples_trained_this_pass\":" << control_->samples_trained_this_pass.load()
      << ",\"last_loss\":" << control_->last_loss.load()
      << ",\"best_loss\":" << control_->best_loss.load()
      << ",\"checkpoints_written\":" << control_->checkpoints_written.load()
      << ",\"last_checkpoint_path\":\"" << json_escape(control_->get_last_checkpoint_path()) << "\""
      << ",\"last_checkpoint_time_unix\":" << control_->last_checkpoint_time_unix.load() << "}";
    return {200, j.str()};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_checkpoint(int wait_ms) {
    // Idle means no active pass at all — nothing would ever consume the
    // flag, so reject rather than leave it to fire as a surprise on some
    // unrelated future pass's very first sample. Any other phase (including
    // LoadingData/Tokenizing, where it's a no-op until the training loop
    // proper starts — see CLAUDE.md) accepts the request.
    if (control_->phase.load() == TrainerPhase::Idle) {
        control_->log(TrainerLogLevel::Warn,
                      "Checkpoint requested via admin API while idle — rejected (nothing to checkpoint)");
        return {409, "{\"error\":\"no active training pass; nothing to checkpoint\"}"};
    }

    control_->log(TrainerLogLevel::Info, "Checkpoint requested via admin API");
    const long long before = control_->checkpoints_written.load();
    control_->checkpoint_requested = true;

    if (wait_ms > 0) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (control_->checkpoints_written.load() > before) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    const bool completed = control_->checkpoints_written.load() > before;
    std::ostringstream j;
    j << "{\"requested\":true,\"completed\":" << (completed ? "true" : "false")
      << ",\"checkpoints_written\":" << control_->checkpoints_written.load()
      << ",\"checkpoint_path\":\"" << json_escape(control_->get_last_checkpoint_path()) << "\"}";
    return {202, j.str()};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_pause() {
    control_->paused = true;
    control_->log(TrainerLogLevel::Warn, "Pause requested via admin API");
    return {202, "{\"paused\":true}"};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_resume() {
    control_->paused = false;
    control_->wake();
    control_->log(TrainerLogLevel::Info, "Resume requested via admin API");
    return {202, "{\"paused\":false}"};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_logs() {
    std::ostringstream j;
    j << "{\"entries\":[";
    bool first = true;
    for (const auto& entry : control_->recent_logs()) {
        if (!first) {
            j << ",";
        }
        first = false;
        j << "{\"id\":" << entry.id << ",\"timestamp_unix_ms\":" << entry.timestamp_unix_ms
          << ",\"level\":\"" << to_string(entry.level) << "\""
          << ",\"message\":\"" << json_escape(entry.message) << "\"}";
    }
    j << "]}";
    return {200, j.str()};
}
