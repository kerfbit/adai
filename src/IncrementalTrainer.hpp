#pragma once

#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "ChatbotTrainer.hpp"
#include "Config.hpp"
#include "DatasetRegistry.hpp"
#include "EncoderDecoderModel.hpp"
#include "Logger.hpp"
#include "IMetricsReporter.hpp"
#include "MetricsPushClient.hpp"

// Forward declaration — full type in ModelNameClient.hpp (included by .cpp)
namespace adai { class ModelNameClient; }


/**
 * @brief Training session information
 */
struct TrainingSession {
    int session_id = 0;
    int samples_trained = 0;
    int epochs_completed = 0;
    float final_loss = 0.0f;
    float final_validation_loss = 0.0f;
    std::string checkpoint_path;
    std::chrono::system_clock::time_point timestamp;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;

    // Per-epoch metrics (TD-009)
    std::vector<float> per_epoch_losses;             ///< training loss per epoch
    std::vector<float> per_epoch_validation_losses;  ///< validation loss per epoch
    std::vector<float> per_epoch_learning_rates;     ///< learning rate at end of each epoch
    std::vector<double> training_time_per_epoch;     ///< wall-clock seconds per epoch
    std::vector<float> per_epoch_perplexities;       ///< training perplexity per epoch (exp(loss))
    std::vector<float> per_epoch_validation_perplexities;  ///< validation perplexity per epoch
};


/**
 * @brief Incremental training configuration
 */
struct IncrementalConfig {
    TrainingConfig base_config;  // Base training configuration

    // Session management
    std::string session_dir = "training_sessions";
    int max_sessions_to_keep = 50;

    // Auto-save settings
    bool auto_save_enabled = true;
    int auto_save_every_samples = 1000;
    int auto_save_every_minutes = 30;

    // Checkpointing
    bool save_incremental_checkpoints = true;
    std::string checkpoint_dir = "checkpoints";

    // Checkpoint symlink management (TD-005)
    bool enable_checkpoint_symlinks = true;  // Create latest/best symlinks
    std::string latest_symlink_name =
        "latest_checkpoint.bin";                            // Name for latest checkpoint symlink
    std::string best_symlink_name = "best_checkpoint.bin";  // Name for best checkpoint symlink

    // Metrics push configuration (TD-021)
    std::string metrics_server_url;          // URL of metrics API daemon; empty = no push
    std::string metrics_session_label;       // Human-readable label; auto-derived when empty
    int metrics_push_timeout_ms = 1000;      // HTTP push timeout in milliseconds
    int metrics_heartbeat_interval_ms = 30000; // Idle heartbeat interval in milliseconds

    // Model Name Service configuration
    std::string mns_server_url;   // URL of ModelNameService daemon; empty = MNS disabled
    std::string mns_model_name;   // MNS model name; empty = MNS disabled
};

/**
 * @brief Incremental Training Manager
 *
 * Manages ongoing training sessions with:
 * - Session-based training history
 * - Data versioning and tracking
 * - Automatic checkpointing
 * - Incremental data addition
 * - Resume capability
 */
class IncrementalTrainer {
   public:
    /**
     * @brief Primary constructor — loads all settings from a config.conf file.
     *
     * This is the required entry point.  The file must set at least VOCAB_PATH;
     * MODEL_PATH, architecture, and training hyper-parameters are read from the
     * same file (and can be overridden via environment variables).
     *
     * @param config_file_path  Path to config.conf (empty = search system default).
     * @throws std::runtime_error if VOCAB_PATH is not set in the config.
     */
    explicit IncrementalTrainer(const std::string& config_file_path);
    ~IncrementalTrainer();

    /**
     * @brief Explicit-paths constructor (low-level).
     * @param vocab_path   Path to vocabulary file.
     * @param model_path   Path to model checkpoint.
     */
    IncrementalTrainer(std::string vocab_path, const std::string& model_path);

    /**
     * @brief Explicit-paths constructor with pre-built configuration.
     * @param vocab_path   Path to vocabulary file.
     * @param model_path   Path to model checkpoint.
     * @param cfg          Configuration settings; architecture is applied before
     *                     the model is constructed so no defaults are baked in.
     */
    IncrementalTrainer(std::string vocab_path, const std::string& model_path,
                       IncrementalConfig cfg);

    /**
     * @brief Build an IncrementalConfig from a parsed ServiceConfig.
     *
     * Translates all model-architecture and training hyper-parameter fields from
     * ServiceConfig into IncrementalConfig.  Call this before constructing an
     * IncrementalTrainer when you already have a ServiceConfig in hand.
     */
    static IncrementalConfig make_incremental_config(const adai::ServiceConfig& svc);

    // Configuration
    void set_config(const IncrementalConfig& cfg);
    IncrementalConfig& get_config();

    /**
     * @brief Tear down the current model and rebuild it from config.base_config.
     *
     * Call this after set_config() when you want to train from scratch with a
     * different architecture (e.g. switching from 512-dim/6-layer to
     * 768-dim/24-layer).  Any previously loaded weights are discarded.
     */
    void reset_model_for_config();

    // Training — file-list API (TD-028 Phase 3)
    bool train_on_files(const std::vector<std::string>& files, int num_epochs);
    bool retrain_on_files(const std::vector<std::string>& files, int num_epochs);

    // Session management
    bool resume_last_session();
    bool load_session_history();
    bool save_session_history();
    TrainingSession get_current_session();
    std::vector<TrainingSession> get_session_history() const;
    void cleanup_old_sessions();

    /**
     * @brief Remove dead/broken/crashed session artifacts left on disk.
     *
     * Two passes:
     *  1. Deletes orphaned `session_<N>_best.bin` / `auto_save_session_<N>.bin`
     *     files for any N other than the current in-progress session, and any
     *     `session_<N>_checkpoint.bin` not referenced by a session_history
     *     entry — all of these can only exist because a run crashed before
     *     reaching finalize_session(), which is what normally deletes a
     *     session's superseded snapshot / registers its checkpoint.
     *  2. Drops session_history entries that fail is_sane_checkpoint_candidate
     *     (zero samples trained, non-finite/non-positive loss, or a missing
     *     checkpoint file) and deletes their files, then persists the
     *     cleaned history.
     *
     * Safe to call repeatedly; a directory with no dead artifacts is a no-op.
     */
    void cleanup_dead_sessions();

    // Model operations
    bool save_model(const std::string& path);
    bool load_model(const std::string& path);
    std::string get_latest_checkpoint();

    /**
     * @brief Hard-reset: erase all checkpoints, session history, and optionally
     *        the data registry, then rebuild the model from the current config.
     *
     * The old model file is renamed to <model_path>.bak rather than deleted so
     * it can be recovered manually if needed.
     *
     * @param keep_data_registry  When true the data-registry file is preserved
     *        but all entries are marked untrained so every data file will be
     *        picked up on the next train/retrain run.  When false the registry
     *        is deleted entirely.
     * @return true on success
     */
    bool reset_all(bool keep_data_registry = false);

    // Status and reporting
    void print_training_summary() const;
    int get_total_samples_trained() const;
    float get_total_training_time_hours() const;

    /// Returns the push-session key set at the start of the most recent training
    /// run (TD-021).  Empty when no metrics_server_url was configured or before
    /// the first training run.
    std::string get_metrics_session_key() const { return active_session_key_; }

   private:
    // Training components
    std::string vocab_path_;  ///< Path to vocabulary file (for architecture reinit)
    std::string model_path_;  ///< Path to main model file (for reset)
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    IncrementalConfig config;
    DatasetConfig     dataset_config_;  ///< Data-specific config (TD-028 Phase 2)

    // Vocabulary auto-build state
    int vocab_build_size_ = 0;        ///< 0 = auto-size via recommend_vocab_size(); >0 = explicit size
    bool pending_vocab_build_ = false; ///< True until vocab is built on first training run

    // Session tracking
    std::vector<TrainingSession> session_history;
    int current_session_id;

    // Auto-save state
    std::chrono::system_clock::time_point last_save_time;
    int samples_since_last_save;

    // Best checkpoint tracking (TD-005)
    float best_validation_loss;
    std::string best_checkpoint_path;

    // Metrics reporter (TD-021)
    std::unique_ptr<IMetricsReporter> metrics_reporter_;    ///< active reporter (Null or Push)
    MetricsPushClient* push_client_{nullptr};               ///< non-owning alias when reporter is MetricsPushClient
    std::string active_session_key_;                        ///< key for the in-flight push session

    // Model Name Service client (Phase 2)
    std::unique_ptr<adai::ModelNameClient> mns_client_;     ///< null when MNS disabled

    // Helper methods

    /**
     * @brief Single entry point for model construction.
     *
     * Reads vocab_path_ and config.base_config to build a fresh
     * EncoderDecoderModel and transfer a new tokenizer into it.
     * Every code path that creates or re-creates the model calls this
     * method — there is no other place that instantiates EncoderDecoderModel.
     */
    void build_model();

    /**
     * @brief Build and save the vocabulary from a set of conversation pairs.
     *
     * Extracts all input and response texts, runs BPE with vocab_build_size_
     * target tokens, and writes the vocab file to vocab_path_.  Clears
     * pending_vocab_build_ on success.
     *
     * @return true if the vocab file was written successfully, false otherwise.
     */
    bool bootstrap_vocab(const std::vector<ConversationPair>& pairs);

    bool initialize_session();
    bool finalize_session(int samples_trained, int epochs_completed, float final_loss,
                          float final_val_loss);
    bool should_auto_save();
    void perform_auto_save();
    std::string generate_session_checkpoint_path();
    std::string get_session_dir() const;
    void ensure_directories_exist();
    static int load_conversation_pairs(const std::string& filepath,
                                       std::vector<ConversationPair>& pairs);

    // Remove a saved model and all its sidecar files (.config, .vocab, .encoder, .decoder,
    // .lm_head)
    static void remove_model_files(const std::string& base_path);

    // TD-005 best-checkpoint selection guard: rejects sessions whose recorded
    // final_validation_loss can't be trusted as a real "best" candidate —
    // zero-sample sessions (loss left at its zero-initialized default),
    // non-finite/non-positive losses (NaN/Inf from a diverged or crashed run,
    // or a stray exact 0.0), and checkpoints missing from disk.
    static bool is_sane_checkpoint_candidate(const TrainingSession& session);

    // Symlink management helpers (TD-005)
    void update_checkpoint_symlinks(const std::string& checkpoint_path);
    void update_best_checkpoint(float validation_loss, const std::string& checkpoint_path);
    std::string get_best_checkpoint_path() const;
    static bool is_windows_platform();
    bool create_or_update_symlink(const std::string& target, const std::string& link_path);
    static bool remove_symlink_if_exists(const std::string& link_path);

    static std::string format_duration(double seconds);

    // Shared training execution: starts the metrics session, wires up callbacks,
    // runs trainer.train(), saves a checkpoint, and finalizes the session.
    // Called by train_on_files and retrain_on_files after data is loaded and
    // tokenized.  metrics_sample_count feeds start_session(); finalize_sample_count
    // feeds finalize_session() (they differ in the retrain path).
    bool run_training(ChatbotTrainer& trainer, int num_epochs,
                      int metrics_sample_count, int finalize_sample_count,
                      bool enable_best_model_snapshot);

};
