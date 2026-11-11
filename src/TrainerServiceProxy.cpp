// @adai-status: beta
// @adai-version: 0.8.0
// @adai-reviewed: 2026-11-11

#include "TrainerServiceProxy.hpp"
#include <httplib.h>
#include <charconv>
#include <chrono>
#include <map>
#include <optional>
#include <sstream>
#include <vector>
#include "BrowserWriteGuard.hpp"
#include "DaemonConfigStore.hpp"
#include "HttpListenSocket.hpp"
#include "Logger.hpp"
#include "TrainerAdminConfig.hpp"
#include "nlohmann/json.hpp"

// ============================================================================
// ServerImpl (hides httplib from the header — same pimpl pattern as
// TrainerAdminAPI/ModelNameService/RegistryServer)
// ============================================================================

class adai::TrainerServiceProxy::ServerImpl {
   public:
    httplib::Server server;
};

// ============================================================================
// Internal JSON output helpers. Request bodies are parsed by the shared
// parse_trainer_admin_config() (TrainerAdminConfig.hpp, TD-519).
// ============================================================================

namespace {

// TD-571: {"error": message} through nlohmann/json, so any character in the message (an exception's
// text, a database path) stays valid JSON; a hand-written escape covered five characters.
std::string error_json(const std::string& message) {
    return nlohmann::json{{"error", message}}.dump(-1, ' ', false,
                                                   nlohmann::json::error_handler_t::replace);
}

// A proxied request whose connection itself failed (child mid-start or mid-exit) gets this,
// distinct from the "no child at all" idle default below — same HTTP status, different message,
// so an operator can tell the two situations apart from the response body alone.
constexpr int kChildUnreachableStatus = 503;
// TD-525: the whole value must be a decimal number in [0, max] (also used for /admin/logs' limit, TD-522).
std::optional<int> parse_wait_ms(const std::string& value, int max) {
    int out = 0;
    const char* end = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(value.data(), end, out);
    if (value.empty() || ec != std::errc() || ptr != end || out < 0 || out > max) return std::nullopt;
    return out;
}

std::string child_unreachable_json() {
    return "{\"error\":\"training pass is starting or exiting; try again shortly\"}";
}

// TD-525: the child took the request but didn't answer in time, so it may have acted on it (an
// accepted checkpoint used to come back as "starting or exiting", inviting a second request).
constexpr int kChildTimeoutStatus = 504;
std::string child_timeout_json(int read_timeout_ms) {
    return "{\"error\":\"training pass did not answer within " + std::to_string((read_timeout_ms + 999) / 1000) +
           " s; the request may still have been applied\"}";
}

}  // namespace

// ============================================================================
// Construction / route registration
// ============================================================================

adai::TrainerServiceProxy::TrainerServiceProxy(std::string host, int port,
                                               std::string child_admin_dir,
                                               std::shared_ptr<TrainerServiceControlState> control,
                                               TrainerAdminDefaults idle_defaults)
    : host_(std::move(host)),
      port_(port),
      child_admin_dir_(std::move(child_admin_dir)),
      control_(std::move(control)),
      idle_defaults_(idle_defaults),
      server_impl_(std::make_unique<ServerImpl>()) {
    auto& svr = server_impl_->server;
    // TD-550: refuse writes a web page could make from an operator's browser (an unlisted Origin, or a
    // preflight-free Content-Type); see BrowserWriteGuard.hpp.
    install_browser_write_guard(svr, "trainer_service");
    use_exclusive_listen_port(svr);  // TD-551

    // Never proxied — the supervisor itself is "healthy" as long as it can answer at all,
    // regardless of whether a child happens to be running right now.
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/admin/config", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] =
            child_port_.load() != 0 ? proxy("GET", "/admin/config", "") : handle_get_config_idle();
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Put("/admin/config", [this](const httplib::Request& req, httplib::Response& res) {
        auto [status, body] = child_port_.load() != 0 ? proxy("PUT", "/admin/config", req.body)
                                                       : handle_put_config_idle(req.body);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/status", [this](const httplib::Request&, httplib::Response& res) {
        auto [status, body] =
            child_port_.load() != 0 ? proxy("GET", "/admin/status", "") : std::pair{200, idle_status_json()};
        // TD-571: on every answer, a 503/504 included (it used to be a 200 only), so the service's
        // own figures, seconds_since_pass_progress above all, show while a pass's admin API isn't
        // answering.
        body = with_supervisor_fields(body);
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Get("/admin/logs", [this](const httplib::Request& req, httplib::Response& res) {
        // TD-522: ?limit=N is checked here and passed on rebuilt from the number, like wait_ms.
        std::string path = "/admin/logs";
        if (req.has_param("limit")) {
            const auto limit = parse_wait_ms(req.get_param_value("limit"), kMaxAdminLogEntries);
            if (!limit) {
                res.status = 400;
                res.set_content("{\"error\":\"limit must be a whole number from 0 to " +
                                    std::to_string(kMaxAdminLogEntries) + "\"}",
                                "application/json");
                return;
            }
            path += "?limit=" + std::to_string(*limit);
        }
        auto [status, body] =
            child_port_.load() != 0 ? proxy("GET", path, "") : std::pair{200, std::string("{\"entries\":[]}")};
        res.status = status;
        res.set_content(body, "application/json");
    });

    svr.Post("/admin/checkpoint", [this](const httplib::Request& req, httplib::Response& res) {
        // TD-525: wait_ms is parsed here and the proxy waits that long plus 5 s for the child's answer
        // (a fixed 5 s used to turn an accepted checkpoint into a 503), and the path is rebuilt from the number.
        int wait_ms = 0;
        if (req.has_param("wait_ms")) {
            const auto parsed = parse_wait_ms(req.get_param_value("wait_ms"), kMaxCheckpointWaitMs);
            if (!parsed) {
                res.status = 400;
                res.set_content("{\"error\":\"wait_ms must be a whole number from 0 to " +
                                    std::to_string(kMaxCheckpointWaitMs) + "\"}",
                                "application/json");
                return;
            }
            wait_ms = *parsed;
        }
        std::string path = "/admin/checkpoint";
        if (wait_ms > 0) path += "?wait_ms=" + std::to_string(wait_ms);
        auto [status, body] = child_port_.load() != 0
            ? proxy("POST", path, "", wait_ms + 5000)
            : std::pair{409, std::string("{\"error\":\"no active training pass; nothing to checkpoint\"}")};
        res.status = status;
        res.set_content(body, "application/json");
    });

    // TD-173: pause/resume always mutate the supervisor's own persistent control state — that's
    // the real, service-level effect (stop/resume launching further passes at all) — and
    // additionally, best-effort, proxy to a live child so an in-flight pass drains/resumes
    // promptly too. Before this fix these two branches only ever did the proxy-or-synthesize half
    // and never touched anything that outlived the current pass — see TECHNICAL_DEBT.md TD-173.
    svr.Post("/admin/pause", [this](const httplib::Request&, httplib::Response& res) {
        control_->paused = true;
        // Also wake — without this, a pause issued while the supervisor is mid-way through its
        // 45s "nothing pending" idle sleep wouldn't be noticed until that sleep times out
        // naturally, taking up to 45s to actually take effect. Waking makes the loop re-check
        // `paused` (now true) immediately and park in the dedicated paused-wait instead. Safe
        // here — a real thread, not a signal handler.
        control_->wake();
        if (child_port_.load() != 0) {
            // Best-effort; the response isn't surfaced to the caller, but a failure is logged (TD-525).
            if (const auto [status, body] = proxy("POST", "/admin/pause", ""); status >= 400) {
                Logger::warn("trainer_service: couldn't pause the running pass ({}: {}); it stops at its end", status,
                             body);
            }
        }
        res.status = 202;
        res.set_content("{\"paused\":true}", "application/json");
    });

    svr.Post("/admin/resume", [this](const httplib::Request&, httplib::Response& res) {
        control_->paused = false;
        control_->wake();  // safe here — a real thread, not a signal handler
        if (child_port_.load() != 0) {
            // Best-effort; the response isn't surfaced to the caller, but a failure is logged (TD-525).
            if (const auto [status, body] = proxy("POST", "/admin/resume", ""); status >= 400) {
                Logger::warn("trainer_service: couldn't resume the running pass ({}: {})", status, body);
            }
        }
        res.status = 202;
        res.set_content("{\"paused\":false}", "application/json");
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

std::pair<int, std::string> adai::TrainerServiceProxy::proxy(const std::string& method, const std::string& path,
                                                             const std::string& body, int read_timeout_ms) {
    const int port = child_port_.load();
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);  // 1s — a live child's admin API answers near-instantly
    cli.set_read_timeout(std::chrono::milliseconds(read_timeout_ms));  // 5 s default, generous for /admin/logs
    httplib::Result res = method == "GET"   ? cli.Get(path)
                          : method == "PUT" ? cli.Put(path, body, "application/json")
                                            : cli.Post(path, body, "application/json");
    if (!res) {
        const auto err = res.error();
        if (err == httplib::Error::Connection || err == httplib::Error::ConnectionTimeout) {
            return {kChildUnreachableStatus, child_unreachable_json()};
        }
        Logger::warn("trainer_service: {} {} to the running pass failed: {}", method, path, httplib::to_string(err));
        return {kChildTimeoutStatus, child_timeout_json(read_timeout_ms)};
    }
    return {res->status, res->body};
}

// ============================================================================
// Idle fallbacks (no child running at all)
// ============================================================================

std::string adai::TrainerServiceProxy::idle_status_json() const {
    // Same shape TrainerAdminAPI::handle_status() reports, with TrainerControlState's own
    // just-constructed defaults for every field — this IS what a fresh TrainerControlState would
    // report before any pass ever touched it, so no separate "idle schema" is being invented here.
    // TD-173: `paused` is the one field that must NOT be a hardcoded default — it's the real,
    // persistent supervisor-level pause state; every other field stays a true default (no pass
    // means no epoch/loss/etc. regardless of pause state).
    std::ostringstream j;
    j << "{\"phase\":\"idle\""
      << ",\"paused\":" << (control_->paused.load() ? "true" : "false")
      << ",\"run_id\":\"\""
         ",\"session_id\":\"\""
         ",\"model_name\":\"\""
         ",\"current_epoch\":0"
         ",\"total_epochs\":0"
         ",\"samples_trained_this_pass\":0"
         ",\"last_loss\":null"  // TD-520: no pass, no loss
         ",\"best_loss\":null"
         ",\"loss_diverged\":false"
         ",\"checkpoints_written\":0"
         ",\"last_checkpoint_path\":\"\""
         ",\"last_checkpoint_time_unix\":0}";
    return j.str();
}

std::string adai::TrainerServiceProxy::with_supervisor_fields(const std::string& body) const {
    // Defensive: only append onto a well-formed JSON object ending in '}' — both call sites
    // (idle_status_json() and a live child's real /admin/status 200 response) always produce
    // exactly that shape, but this guards against silently corrupting anything unexpected rather
    // than assuming.
    if (body.empty() || body.back() != '}') {
        return body;
    }

    const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    const auto started = control_->service_started_unix.load();
    const long long uptime_seconds = started > 0 ? static_cast<long long>(now) - started : 0;

    // TD-569: total_passes_stalled and seconds_since_pass_progress (the running pass's seconds
    // without progress) come after last_failure_unix.
    std::ostringstream extra;
    extra << body.substr(0, body.size() - 1)
          << ",\"supervisor_paused\":" << (control_->paused.load() ? "true" : "false")
          << ",\"total_passes_launched\":" << control_->total_passes_launched.load()
          << ",\"total_passes_did_work\":" << control_->total_passes_did_work.load()
          << ",\"total_passes_crashed\":" << control_->total_passes_crashed.load()
          << ",\"total_passes_failed\":" << control_->total_passes_failed.load()
          << ",\"consecutive_failures\":" << control_->consecutive_failures.load()
          << ",\"last_failure_unix\":" << control_->last_failure_unix.load()
          << ",\"total_passes_stalled\":" << control_->total_passes_stalled.load()
          << ",\"seconds_since_pass_progress\":" << control_->seconds_since_pass_progress.load()
          << ",\"last_exit_code\":" << control_->last_exit_code.load()
          << ",\"service_uptime_seconds\":" << uptime_seconds << "}";
    return extra.str();
}

std::pair<int, std::string> adai::TrainerServiceProxy::handle_get_config_idle() {
    // The loaded config.trainer.conf values (what a pass seeds its TrainerControlState from),
    // overlaid with whatever a previous pass (or a previous idle PUT, see below) persisted to the
    // shared daemon_config.db — mirrors TrainerAdminAPI::start()'s own overlay-on-construction
    // logic, just performed directly here instead of via a live TrainerControlState.
    // TD-525: the config file's values (idle_defaults_), as the next pass seeds from, not hardcoded ones.
    bool auto_save_enabled = idle_defaults_.auto_save_enabled;
    int auto_save_every_samples = idle_defaults_.auto_save_every_samples;
    int auto_save_every_minutes = idle_defaults_.auto_save_every_minutes;
    int max_sessions_to_keep = idle_defaults_.max_sessions_to_keep;

    try {
        DaemonConfigStore store(child_admin_dir_ + "/daemon_config.db");
        // TD-521: the same checks a pass applies when it loads them; a bad saved value is skipped.
        std::vector<std::string> rejected;
        const auto u = parse_trainer_admin_overrides(store.load_all(), &rejected);
        for (const auto& r : rejected) {
            Logger::warn("TrainerServiceProxy: ignoring invalid saved admin setting {}", r);
        }
        if (u.auto_save_enabled) auto_save_enabled = *u.auto_save_enabled;
        if (u.auto_save_every_samples) auto_save_every_samples = *u.auto_save_every_samples;
        if (u.auto_save_every_minutes) auto_save_every_minutes = *u.auto_save_every_minutes;
        if (u.max_sessions_to_keep) max_sessions_to_keep = *u.max_sessions_to_keep;
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
    // TD-519: same parser as TrainerAdminAPI::handle_put_config(), so a live pass and an idle
    // service accept and reject exactly the same bodies; nothing is written for a rejected one.
    const auto parsed = parse_trainer_admin_config(body);
    if (!parsed.error.empty()) {
        return {400, error_json(parsed.error)};
    }
    const auto& u = parsed.update;
    // TD-305: every key in one transaction, so a failure stores none of them (one set() per key
    // could leave the first keys stored and the rest not).
    std::map<std::string, std::string> values;
    if (u.auto_save_enabled) values["auto_save_enabled"] = *u.auto_save_enabled ? "true" : "false";
    if (u.auto_save_every_samples) values["auto_save_every_samples"] = std::to_string(*u.auto_save_every_samples);
    if (u.auto_save_every_minutes) values["auto_save_every_minutes"] = std::to_string(*u.auto_save_every_minutes);
    if (u.max_sessions_to_keep) values["max_sessions_to_keep"] = std::to_string(*u.max_sessions_to_keep);
    try {
        std::lock_guard<std::mutex> lock(idle_config_mutex_);
        DaemonConfigStore store(child_admin_dir_ + "/daemon_config.db");
        store.set_many(values);
    } catch (const std::exception& e) {
        return {500, error_json(std::string("could not save the admin config while idle (") +
                                e.what() + "); nothing was changed")};
    }

    return handle_get_config_idle();
}
