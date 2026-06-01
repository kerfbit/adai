#pragma once

#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "BPETokenizer.hpp"
#include "ChatbotTrainer.hpp"
#include "Config.hpp"
#include "EncoderDecoderModel.hpp"
#include "Logger.hpp"
#include "IMetricsReporter.hpp"
#include "MetricsPushClient.hpp"


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
 * @brief Data file version tracking
 */
struct DataVersion {
    std::string data_file;
    std::string checksum;
    int num_samples = 0;
    std::chrono::system_clock::time_point added_time;
    bool trained = false;
};

/**
 * @brief Incremental training configuration
 */
struct IncrementalConfig {
    TrainingConfig base_config;  // Base training configuration

    // Session management
    std::string session_dir = "training_sessions";
    int max_sessions_to_keep = 50;

    // Data management
    std::string data_registry_file = "data_registry.txt";
    bool cache_tokenized_data = false;
    std::string tokenized_cache_dir = "tokenized_cache";

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
 * - Project Gutenberg integration
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

    // Data management
    bool add_new_data(const std::string& data_file);
    bool add_new_data_batch(const std::vector<std::string>& data_files);
    void clear_pending_data();
    std::vector<std::string> get_pending_data_files() const;
    std::vector<std::string> get_trained_data_files() const;

    // Training
    bool train_incremental(int num_epochs);
    bool train_on_new_data_only(int num_epochs);
    bool train_full_retrain(int num_epochs);

    // Session management
    bool resume_last_session();
    bool load_session_history();
    bool save_session_history();
    TrainingSession get_current_session();
    std::vector<TrainingSession> get_session_history() const;
    void cleanup_old_sessions();

    // Data registry
    bool load_data_registry();
    bool save_data_registry();
    bool is_data_trained(const std::string& data_file);
    static std::string compute_data_checksum(const std::string& data_file);

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
    void print_session_history();
    void print_data_registry();
    int get_total_samples_trained() const;
    float get_total_training_time_hours() const;

    /// Returns the push-session key set at the start of the most recent training
    /// run (TD-021).  Empty when no metrics_server_url was configured or before
    /// the first training run.
    std::string get_metrics_session_key() const { return active_session_key_; }

    // Project Gutenberg integration
    bool add_gutenberg_book(int book_id, int num_pairs = 500);
    bool add_gutenberg_books(const std::vector<int>& book_ids, int num_pairs_per_book = 300);

    // HuggingFace Datasets integration
    /**
     * @brief Download a dataset from the HuggingFace Datasets server and add it to
     *        the pending training queue.
     *
     * Uses the HuggingFace datasets-server API (no Python / huggingface_hub required).
     * Rows are fetched as JSON in chunks of 100 and converted to the INPUT:/RESPONSE:
     * training format.
     *
     * Field auto-detection tries common pairs (instruction/output, question/answer, …).
     * For dialog-array datasets (e.g. daily_dialog) consecutive turns are paired.
     * Set the HF_TOKEN environment variable to access gated datasets.
     *
     * @param dataset_id   HuggingFace dataset identifier, e.g. "daily_dialog" or
     *                     "tatsu-lab/alpaca".  Slashes are allowed.
     * @param num_pairs    Maximum number of training pairs to extract (default 500).
     * @param split        Dataset split to use (default "train").
     * @param input_field  JSON field name for the input text.  Empty = auto-detect.
     * @param output_field JSON field name for the output text.  Empty = auto-detect.
     * @return true if the data was downloaded and added successfully.
     */
    bool add_huggingface_dataset(const std::string& dataset_id, int num_pairs = 500,
                                 const std::string& split = "train",
                                 const std::string& input_field = "",
                                 const std::string& output_field = "");

   private:
    // Training components
    std::string vocab_path_;  ///< Path to vocabulary file (for architecture reinit)
    std::string model_path_;  ///< Path to main model file (for reset)
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    IncrementalConfig config;

    // Session tracking
    std::vector<TrainingSession> session_history;
    int current_session_id;

    // Data tracking
    std::vector<DataVersion> data_registry;
    std::set<std::string> trained_data_files;
    std::vector<std::string> pending_data_files;

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

    // TD-009: Dashboard / timing state
    mutable int dashboard_lines_drawn_;  ///< lines drawn by last display_dashboard() call
    std::chrono::steady_clock::time_point
        session_start_time_steady_;  ///< steady-clock start of current session
    std::chrono::steady_clock::time_point
        epoch_start_time_steady_;  ///< steady-clock start of current epoch

    // Per-sample progress state (updated by sample callback, read by display_dashboard)
    mutable int current_sample_in_epoch_;  ///< 1-based sample index within the current epoch (0 =
                                           ///< not started)
    mutable int total_samples_in_epoch_;   ///< total training samples loaded for this run
    mutable float running_sample_loss_;    ///< running-average loss so far within the current epoch
    mutable float current_item_loss_;      ///< loss of the most recent optimizer step
    mutable float current_item_grad_norm_;  ///< gradient norm of the most recent optimizer step
    mutable float current_item_lr_;         ///< learning rate at the most recent optimizer step

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

    bool initialize_session();
    bool finalize_session(int samples_trained, int epochs_completed, float final_loss,
                          float final_val_loss);
    bool should_auto_save();
    void perform_auto_save();
    std::string generate_session_checkpoint_path();
    std::string get_session_dir() const;
    void ensure_directories_exist();
    bool save_pending_data_list();
    bool load_pending_data_list();
    static int load_conversation_pairs(const std::string& filepath,
                                       std::vector<ConversationPair>& pairs);

    // Remove a saved model and all its sidecar files (.config, .vocab, .encoder, .decoder,
    // .lm_head)
    static void remove_model_files(const std::string& base_path);

    // Symlink management helpers (TD-005)
    void update_checkpoint_symlinks(const std::string& checkpoint_path);
    void update_best_checkpoint(float validation_loss, const std::string& checkpoint_path);
    std::string get_best_checkpoint_path() const;
    static bool is_windows_platform();
    bool create_or_update_symlink(const std::string& target, const std::string& link_path);
    static bool remove_symlink_if_exists(const std::string& link_path);

    // TD-009: Real-time dashboard helpers
    void display_dashboard(const TrainingSession& session, int current_epoch, int total_epochs,
                           bool is_final) const;
    static std::string format_duration(double seconds);
    static std::string progress_bar(int current, int total, int bar_width = 42);

    // Project Gutenberg helpers
    static std::string get_gutenberg_url(int book_id);
    static bool download_file(const std::string& url, const std::string& output_path);
    static bool download_gutenberg_book(int book_id, const std::string& output_dir);
    static bool download_gutenberg_books(const std::vector<int>& book_ids,
                                         const std::string& output_dir);
    static std::string clean_gutenberg_text(const std::string& raw_text);
    static std::vector<std::string> extract_sentences(const std::string& text);
    static std::string generate_question_from_sentence(const std::string& sentence);
    static std::vector<std::pair<std::string, std::string>> create_qa_pairs_from_text(
        const std::vector<std::string>& sentences, int max_pairs);
    static bool convert_gutenberg_to_training_data(const std::string& text_file,
                                                   const std::string& output_file, int max_pairs);

    // HuggingFace helpers
    static bool download_hf_rows(const std::string& dataset_id, const std::string& split,
                                 int offset, int length, const std::string& output_path);
    static bool convert_hf_to_training_data(const std::string& rows_dir,
                                            const std::string& output_file,
                                            const std::string& input_field,
                                            const std::string& output_field, int max_pairs);
};
