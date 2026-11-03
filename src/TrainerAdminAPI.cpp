// @adai-status: stable
// @adai-version: 1.6.1
// @adai-reviewed: 2026-11-03

#include "TrainerAdminAPI.hpp"
#include <httplib.h>
#include "BrowserWriteGuard.hpp"
#include "HttpListenSocket.hpp"
#include <charconv>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <vector>
#include "DaemonConfigStore.hpp"
#include "TrainerAdminConfig.hpp"
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
// Internal JSON output helpers. Request bodies are parsed by the shared
// parse_trainer_admin_config() (TrainerAdminConfig.hpp, TD-519).
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
    // TD-550: loopback-only, but a browser on the same host can still reach it.
    // TD-550: refuse writes a web page could make from an operator's browser (an unlisted Origin, or a
    // preflight-free Content-Type); see BrowserWriteGuard.hpp.
    install_browser_write_guard(svr, "TrainerAdminAPI");
    use_exclusive_listen_port(svr);  // TD-551

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

    svr.Get("/admin/logs", [this](const httplib::Request& req, httplib::Response& res) {
        // TD-522: ?limit=N returns the newest N entries (0 to TrainerControlState::kMaxLogEntries).
        std::size_t limit = TrainerControlState::kMaxLogEntries;
        if (req.has_param("limit")) {
            const std::string v = req.get_param_value("limit");
            const auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), limit);
            if (v.empty() || ec != std::errc() || ptr != v.data() + v.size() ||
                limit > TrainerControlState::kMaxLogEntries) {
                res.status = 400;
                res.set_content("{\"error\":\"limit must be a whole number from 0 to " +
                                    std::to_string(TrainerControlState::kMaxLogEntries) + "\"}",
                                "application/json");
                return;
            }
        }
        auto [status, body] = handle_logs(limit);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/checkpoint", [this](const httplib::Request& req, httplib::Response& res) {
        // TD-521: a whole number up to kMaxCheckpointWaitMs (each wait holds a worker thread), else 400.
        int wait_ms = 0;
        if (req.has_param("wait_ms")) {
            const std::string v = req.get_param_value("wait_ms");
            const auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), wait_ms);
            if (v.empty() || ec != std::errc() || ptr != v.data() + v.size() || wait_ms < 0 ||
                wait_ms > kMaxCheckpointWaitMs) {
                res.status = 400;
                res.set_content("{\"error\":\"wait_ms must be a whole number from 0 to " +
                                    std::to_string(kMaxCheckpointWaitMs) + "\"}",
                                "application/json");
                return;
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
            // TD-521: overrides get the PUT checks; an invalid one is logged and the file value kept.
            std::vector<std::string> rejected;
            const auto u = parse_trainer_admin_overrides(config_store_->load_all(), &rejected);
            for (const auto& r : rejected) {
                control_->log(TrainerLogLevel::Warn, "Ignoring invalid saved admin setting " + r);
            }
            if (u.auto_save_enabled) control_->auto_save_enabled = *u.auto_save_enabled;
            if (u.auto_save_every_samples) control_->auto_save_every_samples = *u.auto_save_every_samples;
            if (u.auto_save_every_minutes) control_->auto_save_every_minutes = *u.auto_save_every_minutes;
            if (u.max_sessions_to_keep) control_->max_sessions_to_keep = *u.max_sessions_to_keep;
        } catch (const std::exception& e) {
            // TD-307: the store can open and still fail to read; then PUTs are still saved.
            control_->log(TrainerLogLevel::Warn,
                          std::string("saved admin settings not loaded from daemon_config.db (") + e.what() +
                              "); using the configured values" +
                              (config_store_ ? "" : "; admin config changes won't persist across restarts"));
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
    // TD-519: the whole body is parsed as JSON and validated before anything is applied, so a
    // spaced `"auto_save_enabled": true` reads as true and a rejected PUT changes nothing.
    const auto parsed = parse_trainer_admin_config(body);
    if (!parsed.error.empty()) {
        return {400, "{\"error\":\"" + json_escape(parsed.error) + "\"}"};
    }
    const auto& u = parsed.update;
    // TD-305: save first, in one transaction, and apply only once that worked. Keys were applied
    // in memory and then saved one set() at a time, so a failed save left the live pass and
    // daemon_config.db disagreeing (and the exception escaped the handler).
    std::lock_guard<std::mutex> lock(put_config_mutex_);
    const bool enabled = u.auto_save_enabled.value_or(control_->auto_save_enabled.load());
    const int samples = u.auto_save_every_samples.value_or(control_->auto_save_every_samples.load());
    const int minutes = u.auto_save_every_minutes.value_or(control_->auto_save_every_minutes.load());
    const int keep = u.max_sessions_to_keep.value_or(control_->max_sessions_to_keep.load());
    if (config_store_) {
        try {
            config_store_->set_many({{"auto_save_enabled", enabled ? "true" : "false"},
                                     {"auto_save_every_samples", std::to_string(samples)},
                                     {"auto_save_every_minutes", std::to_string(minutes)},
                                     {"max_sessions_to_keep", std::to_string(keep)}});
        } catch (const std::exception& e) {
            control_->log(TrainerLogLevel::Error,
                          std::string("PUT /admin/config: could not save daemon_config.db: ") + e.what());
            return {500, "{\"error\":\"could not save the admin config (" + json_escape(e.what()) +
                             "); nothing was changed\"}"};
        }
    }
    control_->auto_save_enabled = enabled;
    control_->auto_save_every_samples = samples;
    control_->auto_save_every_minutes = minutes;
    control_->max_sessions_to_keep = keep;
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
    // TD-520: a non-finite loss is null (a raw nan/inf is invalid JSON and lost the whole status), and
    // loss_diverged tells a diverged pass apart from one that hasn't reported a loss yet.
    const auto number_or_null = [](double v) {
        std::ostringstream n;
        n.precision(17);
        if (std::isfinite(v)) n << v; else n << "null";
        return n.str();
    };
    const double last_loss = control_->last_loss.load();
    const bool loss_diverged =
        control_->samples_trained_this_pass.load() > 0 && !std::isfinite(last_loss);
    std::ostringstream j;
    j << "{\"phase\":\"" << to_string(control_->phase.load()) << "\""
      << ",\"paused\":" << (control_->paused.load() ? "true" : "false")
      << ",\"run_id\":\"" << json_escape(control_->get_run_id()) << "\""
      << ",\"session_id\":\"" << json_escape(control_->get_session_id()) << "\""
      << ",\"model_name\":\"" << json_escape(control_->get_model_name()) << "\""
      << ",\"current_epoch\":" << control_->current_epoch.load()
      << ",\"total_epochs\":" << control_->total_epochs.load()
      << ",\"samples_trained_this_pass\":" << control_->samples_trained_this_pass.load()
      << ",\"last_loss\":" << number_or_null(last_loss)
      << ",\"best_loss\":" << number_or_null(control_->best_loss.load())
      << ",\"loss_diverged\":" << (loss_diverged ? "true" : "false")
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
    // TD-521: completed means a checkpoint written for this request (its ticket served), not any
    // checkpoint; an auto-save landing in the wait used to count.
    const long long ticket = control_->request_checkpoint();
    const auto served = [&] { return control_->checkpoint_requests_served.load() >= ticket; };

    if (wait_ms > 0) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        while (!served() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    const bool completed = served();
    std::ostringstream j;
    j << "{\"requested\":true,\"completed\":" << (completed ? "true" : "false")
      << ",\"checkpoints_written\":" << control_->checkpoints_written.load()
      << ",\"checkpoint_path\":\"" << json_escape(control_->get_last_checkpoint_path()) << "\"}";
    return {202, j.str()};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_pause() {
    control_->request_pause();  // TD-522: reports `pausing` until the pass drains
    control_->log(TrainerLogLevel::Warn, "Pause requested via admin API");
    return {202, "{\"paused\":true}"};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_resume() {
    control_->clear_pause();
    control_->log(TrainerLogLevel::Info, "Resume requested via admin API");
    return {202, "{\"paused\":false}"};
}

std::pair<int, std::string> adai::TrainerAdminAPI::handle_logs(std::size_t limit) {
    std::ostringstream j;
    j << "{\"entries\":[";
    bool first = true;
    for (const auto& entry : control_->recent_logs(limit)) {
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
