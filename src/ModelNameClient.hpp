#pragma once

// @adai-status: beta        (tested only indirectly via mns_manager_gui_test.cpp)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07


#include <map>
#include <optional>
#include <string>
#include <vector>
#include "Config.hpp"
#include "ModelNameService.hpp"  // ArtifactLocation, ModelRecord

namespace adai {

// Result type returned by resolve operations.
struct ResolvedModel {
    std::string model_id;
    std::string model_name;
    std::string state;
    ArtifactLocation artifact;
    // Dataset-registry run_group for this model, per its MNS record — empty
    // means the model hasn't been registered with one (or predates this
    // field), in which case callers should keep their own local RUN_GROUP
    // config / SESSION_DIR-basename fallback instead of overwriting it.
    std::string run_group;
};

// Lightweight summary returned by list_models().
struct ModelSummary {
    std::string model_name;
    std::string state;
    std::string role;
    std::string updated_utc;
};

/**
 * @brief Synchronous HTTP client for the ModelNameService daemon.
 *
 * All methods block until the server responds or timeout_ms elapses.
 * A std::runtime_error is thrown on non-2xx responses or network failure
 * (consistent with RegistryTransport's error-propagation contract).
 *
 * Usage pattern:
 *   ModelNameClient mns("http://192.168.1.19:8083");
 *   auto id = mns.register_model("adai-chatbot-v3", "chatbot", config);
 *   mns.set_training("adai-chatbot-v3", run_id_, session_key_);
 *   // ... training loop ...
 *   mns.set_candidate("adai-chatbot-v3", run_id_, artifact, summary);
 */
class ModelNameClient {
   public:
    explicit ModelNameClient(std::string server_url, int timeout_ms = 5000);

    // Register a new model; returns the assigned UUID.
    std::string register_model(const std::string& model_name, const std::string& role,
                               const ServiceConfig& arch,
                               const std::map<std::string, std::string>& tags = {});

    // Transition model to "training" state. MNS allocates and returns the
    // run_id (definitive standard — see CLAUDE.md "Configuration"): "run-01"
    // the first time this model ever trains, incrementing only when
    // new_run=true (a retrain); a plain continuation (train/resume) passes
    // new_run=false and gets back the model's current run_id unchanged. If a
    // previous run was still marked "training" (crashed/killed), its last
    // pushed progress is archived into training_history (incomplete=true)
    // before this call's run_id is allocated.
    std::string set_training(const std::string& model_name, bool new_run,
                             const std::string& metrics_session_key = "");

    // Transition model to "candidate" state (releases training lock, attaches artifact).
    void set_candidate(const std::string& model_name, const std::string& run_id,
                       const ArtifactLocation& artifact,
                       const std::map<std::string, std::string>& training_summary = {});

    // Push epoch/loss progress for the currently-active run — call after every
    // epoch so a killed/crashed trainer still leaves an accurate last-known
    // state in MNS. Throws on a stale/superseded run_id (409) or network
    // failure; callers should treat failures as non-fatal (log and continue).
    void push_progress(const std::string& model_name, const std::string& run_id,
                       const std::string& session_id, int epoch, double loss, double best_loss);

    // Resolve a model by name; throws if not found or in initializing state.
    ResolvedModel resolve_model(const std::string& model_name);

    // Fetch a registered model's authoritative architecture. Returns std::nullopt
    // if the model isn't registered (404); throws on other request failures.
    std::optional<ModelArchitecture> get_architecture(const std::string& model_name);

    // Resolve the production model for a role; throws if no production model.
    ResolvedModel resolve_role(const std::string& role);

    // List models, optionally filtered by state and/or role.
    std::vector<ModelSummary> list_models(const std::string& state_filter = "",
                                          const std::string& role_filter = "", int limit = 50);

    // Promote a candidate model to production for a role.
    void promote(const std::string& role, const std::string& model_name);

    // Set/update a registered model's dataset-registry run_group. Unlike
    // architecture (register-time-only, immutable), this is safe to call at
    // any time — run_group doesn't affect checkpoint compatibility. Throws if
    // the model isn't registered (404) or on network failure.
    void update_run_group(const std::string& model_name, const std::string& run_group);

   private:
    struct ParsedUrl {
        std::string host = "localhost";
        int port = 8083;
        std::string base_path;
        static ParsedUrl from(const std::string& url);
    };

    // Returns HTTP status code; body is set to the response body.
    // Throws std::runtime_error on persistent connection failure (returns 0).
    int http_post(const std::string& path, const std::string& body, std::string& out) const;
    int http_get(const std::string& path, std::string& out) const;
    int http_put(const std::string& path, const std::string& body, std::string& out) const;

    static void check_status(int status, const std::string& out, const std::string& op);

    std::string server_url_;
    int timeout_ms_;
    ParsedUrl parsed_;
};

}  // namespace adai
