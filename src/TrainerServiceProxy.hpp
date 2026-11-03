#pragma once

// @adai-status: experimental
// @adai-version: 0.3.1
// @adai-reviewed: 2026-11-02

// TD-172: trainer_service's own admin HTTP listener. Unlike TrainerAdminAPI (which answers every
// request from an in-process TrainerControlState it owns directly), this class is a thin reverse
// proxy: while a single-pass incremental_trainer child is alive, every /admin/* request is
// forwarded verbatim to that child's own TrainerAdminAPI (started via `--admin-port`, see
// IncrementalTrainingTool.cpp's `resume` command) and its response relayed back unchanged. While
// no child is running, GET/PUT /admin/config are answered directly against the shared
// daemon_config.db overlay (so a config change made while idle still takes effect on the next
// child launch) and every other endpoint answers with an idle-shaped default — mirroring exactly
// what TrainerAdminAPI itself would report with phase == Idle. See TECHNICAL_DEBT.md TD-172 for
// the design rationale (loopback HTTP chosen over a file-based channel: an already-established
// pattern in this codebase, and more portable).
//
// TD-173: pause/resume are no longer pure proxy-or-synthesize — they always mutate the
// process-lifetime TrainerServiceControlState (so pause actually stops the supervisor from
// launching another pass, not just draining the current one) in addition to best-effort proxying
// to a live child (so an in-flight pass still drains promptly). See TrainerServiceControlState.hpp
// and TrainerServiceMain.cpp's main loop for the other half of this fix.

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include "TrainerAdminConfig.hpp"
#include "TrainerServiceControlState.hpp"

namespace adai {

/// TD-525: what a pass starts from before daemon_config.db overrides apply, i.e. the loaded
/// config.trainer.conf values. The idle GET /admin/config reports these with the same overlay, so it
/// matches what the next pass will use.
struct TrainerAdminDefaults {
    bool auto_save_enabled = true;
    int auto_save_every_samples = 1000;
    int auto_save_every_minutes = 30;
    int max_sessions_to_keep = 50;
};

class TrainerServiceProxy {
   public:
    /**
     * @param host             Bind host for this listener (the supervisor's own public-facing
     *                         admin port — TRAINER_ADMIN_HOST/PORT in config.trainer.conf).
     * @param port             Bind port for this listener.
     * @param child_admin_dir  Same directory `incremental_trainer resume --admin-port` passes to
     *                         its own TrainerAdminAPI as config_store_dir (TRAINER_ADMIN_DIR) —
     *                         reused here, while idle, to read/write the same daemon_config.db
     *                         overlay directly.
     * @param control          The supervisor's own process-lifetime control state (TD-173) —
     *                         pause/resume/status read and write this directly, in addition to
     *                         whatever proxying to a live child also happens.
     * @param idle_defaults    TD-525: the config file's tunables, reported while idle under the
     *                         daemon_config.db overlay.
     */
    TrainerServiceProxy(std::string host, int port, std::string child_admin_dir,
                        std::shared_ptr<TrainerServiceControlState> control,
                        TrainerAdminDefaults idle_defaults = {});

    /// TD-525: the longest checkpoint wait_ms the proxy forwards, the pass's own limit (TD-521, 60 s);
    /// above it, or not a whole number, is a 400.
    static constexpr int kMaxCheckpointWaitMs = adai::kMaxCheckpointWaitMs;
    ~TrainerServiceProxy();

    TrainerServiceProxy(const TrainerServiceProxy&) = delete;
    TrainerServiceProxy& operator=(const TrainerServiceProxy&) = delete;

    /// Called by the supervisor loop whenever the currently-tracked child starts (its assigned
    /// --admin-port) or exits (0). Safe to call from a different thread than start()/stop() run
    /// on — the current child port is read fresh on every incoming request.
    void set_child_port(int port);

    /// Blocking — like TrainerAdminAPI::start(): binds host:port and serves until stop() is
    /// called. Returns false if the listener failed to bind.
    bool start();

    /// Stops the HTTP server. Safe to call multiple times / before start().
    void stop();
    bool is_running() const;

   private:
    /// TD-525: forwards one request to the current child. A refused connection (the pass is starting
    /// or exiting) is a 503; no answer within the read timeout is a 504, since the child may already
    /// have acted on it.
    std::pair<int, std::string> proxy(const std::string& method, const std::string& path, const std::string& body,
                                      int read_timeout_ms = 5000);

    std::pair<int, std::string> handle_get_config_idle();
    std::pair<int, std::string> handle_put_config_idle(const std::string& body);
    std::string idle_status_json() const;
    /// Appends TrainerServiceControlState's own observability fields as additional top-level
    /// JSON keys onto an existing well-formed status object (additive only — every existing key
    /// stays exactly where it was, so clients parsing the pre-TD-173 shape are unaffected). `body`
    /// must end in '}' (both idle_status_json() and a live child's real /admin/status response
    /// always do); returns `body` unchanged otherwise rather than risk mangling an unexpected
    /// shape (e.g. a proxied error response).
    std::string with_supervisor_fields(const std::string& body) const;

    std::string host_;
    int port_;
    std::string child_admin_dir_;
    std::shared_ptr<TrainerServiceControlState> control_;
    TrainerAdminDefaults idle_defaults_;
    std::atomic<int> child_port_{0};
    std::atomic<bool> running_{false};

    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};

}  // namespace adai
