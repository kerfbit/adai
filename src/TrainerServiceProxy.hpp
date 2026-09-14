#pragma once

// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-14

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

#include <atomic>
#include <memory>
#include <string>
#include <utility>

namespace adai {

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
     */
    TrainerServiceProxy(std::string host, int port, std::string child_admin_dir);
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
    std::pair<int, std::string> proxy_get(const std::string& path);
    std::pair<int, std::string> proxy_post(const std::string& path, const std::string& body);
    std::pair<int, std::string> proxy_put(const std::string& path, const std::string& body);

    std::pair<int, std::string> handle_get_config_idle();
    std::pair<int, std::string> handle_put_config_idle(const std::string& body);
    static std::string idle_status_json();

    std::string host_;
    int port_;
    std::string child_admin_dir_;
    std::atomic<int> child_port_{0};
    std::atomic<bool> running_{false};

    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};

}  // namespace adai
