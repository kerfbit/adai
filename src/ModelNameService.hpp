#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "DaemonConfigStore.hpp"

namespace adai {

// ============================================================================
// Shared data structures
// ============================================================================

struct ArtifactLocation {
    std::string host;                    ///< Hostname where weight file lives; empty = localhost
    std::string path;                    ///< Absolute path on host
    std::string checksum;                ///< Opaque checksum token (e.g. "8388608_1718890000")
    std::string format = "adai-native";  ///< "adai-native" | "safetensors" | "gguf"
};

/// Model architecture parameters — mirrors the arch fields on ModelRecord.
/// Returned by ModelNameClient::get_architecture() so clients (chatbot_api_server,
/// incremental_trainer) can treat MNS as the authoritative source for these
/// values instead of their own local config.conf fallback.
struct ModelArchitecture {
    size_t d_model = 0;
    size_t num_heads = 0;
    size_t d_ff = 0;
    size_t num_encoder_layers = 0;
    size_t num_decoder_layers = 0;
    size_t max_seq_length = 0;
};

struct TrainingHistoryEntry {
    std::string run_id;
    std::string metrics_session_key;
    std::string dataset_group;
    int epochs = 0;
    double final_loss = 0.0;
    std::string started_utc;
    std::string finished_utc;
    /// True when synthesized from a crashed/killed/superseded run's last-known
    /// progress snapshot rather than a normal candidate transition — see
    /// ModelNameService::handle_state_transition's "training" branch.
    bool incomplete = false;
};

struct ModelRecord {
    std::string model_id;
    std::string model_name;
    std::string role;
    std::string state =
        "initializing";  // initializing | training | candidate | production | retired
    std::string run_id;  // set while state == "training"
    std::string created_utc;
    std::string updated_utc;

    ArtifactLocation artifact;

    // Architecture metadata (mirrors ServiceConfig model parameters)
    size_t d_model = 0;
    size_t num_heads = 0;
    size_t d_ff = 0;
    size_t num_encoder_layers = 0;
    size_t num_decoder_layers = 0;
    size_t max_seq_length = 0;

    // Run numbering (see handle_state_transition's "training" branch):
    // current_run_number increments only when a caller requests new_run=true
    // (a retrain) and never on plain continuation (train/resume); run_id is
    // derived from it as "run-01", "run-02", etc. 0 = never trained.
    int current_run_number = 0;
    std::string run_started_utc;  // stamped whenever run_id is (re)allocated

    // Live progress snapshot for the run currently marked "training" — updated
    // by PUT /models/{name}/progress after every epoch, so a killed/crashed
    // trainer still leaves an accurate last-known state. Cleared on a normal
    // candidate transition (the real record moves to training_history) or
    // archived into training_history (incomplete=true) if superseded by a new
    // set_training call before ever reaching candidate.
    std::string progress_session_id;  // e.g. "session-03", from registry_server
    int progress_epoch = 0;
    double progress_loss = 0.0;
    double progress_best_loss = 0.0;
    std::string progress_updated_utc;

    std::vector<TrainingHistoryEntry> training_history;
    std::map<std::string, std::string> tags;
};

// ============================================================================
// ModelNameService — HTTP daemon
// ============================================================================

/**
 * @brief Model Name Service — authoritative registry for model identity.
 *
 * Assigns stable UUIDs to models, tracks lifecycle state
 * (initializing → training → candidate → production → retired),
 * and resolves a role (e.g. "chatbot") to the current production artifact.
 *
 * Storage: Phase 1 uses a JSONL flat file (models.jsonl) plus an atomic
 * roles.json.  The latest record per model_name wins on reload.
 *
 * Thread safety: all public methods are thread-safe.  State changes hold
 * an exclusive write lock; reads share a shared lock.
 */
class ModelNameService {
   public:
    explicit ModelNameService(std::string data_dir, int port = 8083);
    ~ModelNameService();

    ModelNameService(const ModelNameService&) = delete;
    ModelNameService& operator=(const ModelNameService&) = delete;

    bool start();  ///< Blocking — returns when stop() is called
    void stop();
    bool is_running() const;
    int get_port() const;

    /// Configure registry server proxy for GET /models/{name}/datasets.
    void set_registry(const std::string& url, const std::string& group = "default");

    /// Gate PUT /admin/config (default: enabled). GET /admin/config always works.
    void set_admin_enabled(bool enabled);

   private:
    // ── HTTP handlers (return {status_code, json_body}) ─────────────────────
    std::pair<int, std::string> handle_register(const std::string& body);
    std::pair<int, std::string> handle_list(const std::string& state_filter,
                                            const std::string& role_filter, int limit);
    std::pair<int, std::string> handle_get(const std::string& name);
    std::pair<int, std::string> handle_resolve(const std::string& name);
    std::pair<int, std::string> handle_state_transition(const std::string& name,
                                                        const std::string& body);
    std::pair<int, std::string> handle_delete(const std::string& name);
    std::pair<int, std::string> handle_list_roles();
    std::pair<int, std::string> handle_resolve_role(const std::string& role);
    std::pair<int, std::string> handle_promote(const std::string& role, const std::string& body);
    std::pair<int, std::string> handle_health();
    std::pair<int, std::string> handle_datasets(const std::string& name);
    std::pair<int, std::string> handle_admin_get_config();
    std::pair<int, std::string> handle_admin_put_config(const std::string& body);
    std::pair<int, std::string> handle_progress_update(const std::string& name,
                                                        const std::string& body);

    // ── Persistence ──────────────────────────────────────────────────────────
    void load_from_disk();
    void persist_model(const ModelRecord& rec);
    void persist_roles();
    void rewrite_models_jsonl();  ///< Hard-delete (Phase 1 compat); delegates to SQLite DELETE
    void init_db();               ///< Open/create models.db and run schema migrations
    void migrate_from_jsonl();    ///< Import models.jsonl + roles.json on first SQLite run

    // ── State ────────────────────────────────────────────────────────────────
    std::string data_dir_;
    int port_;
    std::string registry_url_;
    std::string registry_group_ = "default";
    bool admin_enabled_ = true;

    // Persists live-mutable admin/config overrides (registry_url, registry_group)
    // in <data_dir_>/daemon_config.db, separate from models.db. Opened in start().
    std::unique_ptr<DaemonConfigStore> config_store_;

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, ModelRecord> models_;  ///< key: model_name
    std::unordered_map<std::string, std::string> roles_;   ///< role -> model_name

    std::atomic<bool> running_{false};
    std::chrono::steady_clock::time_point start_time_;

    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};

}  // namespace adai
