#pragma once

// @adai-status: stable        (TD-197 — validate_world_model_link() declared, shared by handle_register()/handle_link_world_model())
// @adai-version: 1.1.1
// @adai-reviewed: 2026-09-19


#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
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

/// TD-196: the "connection standard" between a chatbot's encoder/decoder and its (optional)
/// world model + hippocampal memory — everything that used to live only in local
/// config.trainer.conf/config.chatbot.conf, kept consistent across the two files purely by
/// operator discipline. Split by which `kind` (below) actually populates each group:
///   - encoder_name/decoder_name: chatbot-kind only, set at register time, immutable (like
///     architecture — swapping either represents a genuinely different, likely
///     checkpoint-incompatible model).
///   - world_model_name and everything else in the middle group: chatbot-kind only, mutable via
///     POST /models/{name}/link-world-model (a pluggable, swappable-for-a-compatible-one
///     attachment, not part of the chatbot's own fixed identity — same "doesn't break checkpoint
///     compatibility" reasoning run_group already gets its own mutable update path for).
///   - sigreg_lambda/sigreg_num_sketches: world_model-kind only, set at register time.
/// A field group irrelevant to a given record's own `kind` is simply left at its default.
struct ModelConnection {
    std::string encoder_name;
    std::string decoder_name;

    std::string world_model_name;  // empty = no world model attached
    size_t world_model_inject_every_n_layers = 0;
    bool hippocampal_memory_enabled = false;
    size_t hippocampal_memory_capacity = 512;
    float hippocampal_repetition_alpha = 0.0f;
    float hippocampal_repetition_decay = 0.95f;
    float hippocampal_cross_reference_alpha = 0.0f;
    float hippocampal_association_decay = 0.95f;

    float sigreg_lambda = 1.0f;
    size_t sigreg_num_sketches = 64;
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
    // TD-196: one of "encoder" | "decoder" | "world_model" | "chatbot" — which of the 6 inline
    // architecture fields below actually apply, and how `connection` (below) is interpreted:
    //   - "encoder"/"decoder": d_model/num_heads/d_ff/max_seq_length + this kind's own layer
    //     count (num_encoder_layers for "encoder", num_decoder_layers for "decoder" — the other
    //     layer-count field must be 0); connection unused.
    //   - "world_model": same 4 shared fields + num_encoder_layers as its own (standalone
    //     encoder) layer count; connection.sigreg_lambda/sigreg_num_sketches also apply.
    //   - "chatbot": EITHER inline architecture exactly as before this field existed (legacy —
    //     the backward-compatibility path, not a special case) OR, when connection.encoder_name/
    //     decoder_name are both set, the 6 inline fields are unused and get_architecture()
    //     composes them from the linked encoder/decoder records instead; connection's own
    //     world_model_name/injection/hippocampal fields apply either way.
    // Defaults to "chatbot" so every pre-existing row (from before this field existed) keeps
    // behaving exactly as it did before. See handle_register()'s own per-kind validation.
    std::string kind = "chatbot";
    // Dataset-registry run_group this model's trainer should use — a per-model
    // value, set at register time and updatable afterward via
    // PUT /models/{name}/run_group (handle_update_run_group) — unlike
    // architecture, changing it doesn't affect checkpoint compatibility, so it
    // isn't register-only. Empty means "not yet migrated to MNS-sourced
    // run_group" — clients fall back to their own local RUN_GROUP config /
    // SESSION_DIR-basename derivation (see DatasetRegistry.cpp's
    // build_transport()). Distinct from the unrelated daemon-wide
    // registry_group_ member below (admin-mutable, used only by the
    // /models/{name}/datasets proxy) — do not conflate the two.
    std::string run_group;
    std::string state =
        "initializing";  // initializing | training | candidate | production | retired
    std::string run_id;  // set while state == "training"
    std::string created_utc;
    std::string updated_utc;

    ArtifactLocation artifact;

    // Architecture metadata (mirrors ServiceConfig model parameters) — interpreted per `kind`,
    // see its own doc comment above.
    size_t d_model = 0;
    size_t num_heads = 0;
    size_t d_ff = 0;
    size_t num_encoder_layers = 0;
    size_t num_decoder_layers = 0;
    size_t max_seq_length = 0;

    // TD-196: the connection standard between this record and whatever else it's paired with —
    // see ModelConnection's own doc comment. Empty/default for "encoder"/"decoder"-kind records.
    ModelConnection connection;

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
                                            const std::string& role_filter, int limit,
                                            const std::string& kind_filter = "");
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
    // Unlike architecture (immutable — changing it would break checkpoint
    // compatibility), run_group is pure dataset-routing metadata with no such
    // constraint, so it gets a real update path rather than register-only.
    std::pair<int, std::string> handle_update_run_group(const std::string& name,
                                                         const std::string& body);
    // TD-196: attach/replace/detach a chatbot-kind record's world-model connection. Unlike
    // encoder_name/decoder_name (immutable, set at register time — see ModelConnection's own
    // doc comment), the world model itself and its injection/hippocampal parameters are mutable,
    // same "doesn't break checkpoint compatibility" reasoning run_group's own update path already
    // rests on. Validates the linked world_model's own d_model against this record's resolved
    // d_model (via encoder_name for a new-style chatbot, or the inline field for a legacy one)
    // whenever world_model_inject_every_n_layers > 0 — 409 Conflict on mismatch.
    std::pair<int, std::string> handle_link_world_model(const std::string& name,
                                                        const std::string& body);

    // TD-196: shared world-model-link validation used by both handle_register() (a chatbot
    // registering with connection.world_model_name already set in its body — found, during
    // review, to have been silently skipping this check entirely) and handle_link_world_model()
    // (POST .../link-world-model). Requires `models_` already locked by the caller (both do).
    // Returns an {status, json_body} error pair on failure, std::nullopt on success.
    std::optional<std::pair<int, std::string>> validate_world_model_link(
        const ModelRecord& r, const std::string& world_model_name,
        size_t inject_every_n_layers) const;

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
