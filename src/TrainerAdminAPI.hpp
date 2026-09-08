#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include "TrainerControlState.hpp"

namespace adai {

class DaemonConfigStore;

/**
 * @brief Always-on HTTP admin daemon for `incremental_trainer serve`.
 *
 * Constructed and started once at `serve` startup and stays alive for the
 * process's entire lifetime — independent of any single training pass. This
 * is what makes the admin API genuinely always-on rather than reachable only
 * while a `resume` process instance happens to be alive between systemd
 * restart cycles (see the "incremental_trainer admin control daemon
 * (always-on)" plan). Talks only to an in-process TrainerControlState — no
 * IPC, no companion CLI (curl is the documented interface, matching the
 * precedent of the other three daemons' /admin/config).
 *
 * Endpoints:
 *   GET  /health
 *   GET  /admin/config
 *   PUT  /admin/config             — auto_save_enabled/auto_save_every_samples/
 *                                     auto_save_every_minutes/max_sessions_to_keep
 *   GET  /admin/status             — phase, run/session identity, progress, paused
 *   GET  /admin/logs               — recent entries from TrainerControlState's log
 *                                     ring buffer (see TrainerControlState::log()) —
 *                                     the same messages that land in the daemon's own
 *                                     log file/journal, not a separate UI-only stream
 *   POST /admin/checkpoint[?wait_ms=N]
 *   POST /admin/pause
 *   POST /admin/resume
 *
 * Not exposed over HTTP at all: full process shutdown — `systemctl stop`/
 * SIGTERM stays the sole mechanism (see CLAUDE.md).
 */
class TrainerAdminAPI {
   public:
    /**
     * @param control          Shared in-process state; also handed to every
     *                         IncrementalTrainer the `serve` loop constructs
     *                         via set_control_state().
     * @param host             Bind host (default config: "127.0.0.1").
     * @param port             Bind port (default config: 8084).
     * @param config_store_dir Directory for this daemon's daemon_config.db
     *                         overlay (TRAINER_ADMIN_DIR) — separate from
     *                         session_dir/training_sessions and from any
     *                         other daemon's own daemon_config.db.
     */
    TrainerAdminAPI(std::shared_ptr<TrainerControlState> control, std::string host, int port,
                    std::string config_store_dir);
    ~TrainerAdminAPI();

    TrainerAdminAPI(const TrainerAdminAPI&) = delete;
    TrainerAdminAPI& operator=(const TrainerAdminAPI&) = delete;

    /// Blocking — opens config_store_dir/daemon_config.db, overlays any
    /// persisted tunables onto `control`, then calls the underlying
    /// httplib::Server's listen(), which blocks until stop() is called.
    /// Returns false if the listener failed to bind.
    bool start();

    /// Stops the HTTP server. Safe to call multiple times / before start().
    void stop();
    bool is_running() const;
    int get_port() const;

   private:
    std::pair<int, std::string> handle_health();
    std::pair<int, std::string> handle_get_config();
    std::pair<int, std::string> handle_put_config(const std::string& body);
    std::pair<int, std::string> handle_status();
    std::pair<int, std::string> handle_logs();
    std::pair<int, std::string> handle_checkpoint(int wait_ms);
    std::pair<int, std::string> handle_pause();
    std::pair<int, std::string> handle_resume();

    std::shared_ptr<TrainerControlState> control_;
    std::string host_;
    int port_;
    std::string config_store_dir_;
    std::unique_ptr<DaemonConfigStore> config_store_;
    std::atomic<bool> running_{false};

    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};

}  // namespace adai
