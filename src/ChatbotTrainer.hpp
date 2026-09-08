#pragma once

// @adai-status: beta        (capped by TD-039 — large, actively evolving core trainer)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-08


#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "BPETokenizer.hpp"
#include "EncoderDecoderModel.hpp"
#include "IMetricsReporter.hpp"
#include "Optimizer.hpp"
#include "TrainingSampleMeta.hpp"

// Parallel optimizations (Priority 1-5)
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Training data pair (input, target response) with optional metadata.
 */
struct ConversationPair {
    std::string input;
    std::string response;
    SampleMeta meta;

    ConversationPair(std::string in, std::string resp)
        : input(std::move(in)), response(std::move(resp)) {}
    ConversationPair(std::string in, std::string resp, SampleMeta m)
        : input(std::move(in)), response(std::move(resp)), meta(std::move(m)) {}
};

/**
 * @brief Pre-tokenized training data for efficient training
 */
struct TokenizedPair {
    std::vector<int> input_tokens;
    std::vector<int> target_tokens;
    std::string input_text;   // Keep for debugging/logging
    std::string target_text;  // Keep for debugging/logging

    TokenizedPair() = default;  // Required for pre-sized parallel fill
    TokenizedPair(const std::vector<int>& in_tok, const std::vector<int>& tgt_tok,
                  std::string in_txt, std::string tgt_txt)
        : input_tokens(in_tok),
          target_tokens(tgt_tok),
          input_text(std::move(in_txt)),
          target_text(std::move(tgt_txt)) {}
};

/**
 * @brief Learning rate scheduling strategy
 */
enum class LRSchedule : std::uint8_t {
    CONSTANT,          // No scheduling
    LINEAR_WARMUP,     // Linear warmup then constant
    COSINE_DECAY,      // Cosine annealing decay
    WARMUP_COSINE,     // Linear warmup + cosine decay (recommended)
    STEP_DECAY,        // Step-wise decay at intervals
    EXPONENTIAL_DECAY  // Exponential decay
};

/**
 * @brief Logging verbosity levels
 */
enum class LogLevel : std::uint8_t {
    SILENT = 0,   // No output except errors
    NORMAL = 1,   // Basic progress and results
    VERBOSE = 2,  // Detailed progress (default)
    DEBUG = 3     // Debug information
};

/**
 * @brief Training configuration
 */
struct TrainingConfig {
    // Model architecture
    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int num_encoder_layers = 6;
    int num_decoder_layers = 6;
    int max_seq_length = 512;

    // Training parameters
    int num_epochs = 10;
    float learning_rate = 0.001f;         // Initial/base learning rate
    int batch_size = 1;                   // Batch size (samples per gradient accumulation)
    int gradient_accumulation_steps = 1;  // Accumulate gradients over N steps
    int validation_split = 10;            // Use 1/10 of data for validation

    // Learning rate scheduling
    LRSchedule lr_schedule = LRSchedule::WARMUP_COSINE;
    int warmup_steps = 0;             // Warmup steps (0 = auto: 10% of total)
    float min_learning_rate = 1e-6f;  // Minimum LR for decay schedules
    float lr_decay_factor = 0.1f;     // Decay factor for step/exponential
    int lr_decay_steps = 0;           // Steps between decays (0 = auto: per epoch)

    // Optimizer settings
    OptimizerType optimizer_type = OptimizerType::ADAMW;  // Optimizer algorithm
    float adam_beta1 = 0.9f;                              // Adam first moment decay
    float adam_beta2 = 0.999f;                            // Adam second moment decay
    float weight_decay = 0.01f;                           // L2 regularization / weight decay
    float gradient_clip_norm = 1.0f;                      // Maximum gradient norm (0 = no clipping)

    // Adaptive gradient clipping (TD-017)
    bool adaptive_gradient_clip = false;    // Master switch; false = legacy fixed-clip behavior
    float gradient_clip_min = 0.1f;         // Hard floor — threshold never drops below this
    float gradient_clip_max = 5.0f;         // Hard ceiling — threshold never rises above this
    float gradient_clip_ema_decay = 0.05f;  // EMA smoothing factor α
    float gradient_clip_headroom = 2.0f;    // Threshold = ema_norm × headroom
    int gradient_clip_warmup_steps = 100;   // Steps before adaptive logic activates
    float gradient_clip_spike_k = 5.0f;     // Outlier: norms > k×ema are not fed into EMA

    // Early stopping
    bool enable_early_stopping = false;
    int patience = 5;                  // Epochs to wait for improvement
    float min_delta = 1e-4f;           // Minimum change to qualify as improvement
    bool restore_best_weights = true;  // Restore best model after early stop

    // Logging
    int log_every = 10;                      // Log every N samples
    LogLevel log_level = LogLevel::VERBOSE;  // Logging verbosity
    bool verbose = true;                     // Deprecated: use log_level instead

    // Generation quality metrics (BLEU/ROUGE)
    // Disabled by default; each evaluation generates model responses which is expensive.
    bool enable_generation_quality_metrics =
        false;                                // Compute BLEU/ROUGE during each validation pass
    int generation_quality_sample_size = 10;  // Max validation samples used for scoring
    int generation_quality_max_tokens = 50;   // max_length passed to generate_response()
    /// Minimum sample size to use the async parallel-scoring path (TD-023).
    /// Below this threshold the synchronous path is used (no extra memory cost).
    int generation_quality_async_threshold = 50;

    // Quality score backfill into SampleMeta
    bool enable_loss_quality_backfill = false;  // Write exp(-loss) into meta.quality each epoch
    bool enable_generation_quality_backfill =
        false;  // Overwrite meta.quality with BLEU4 after training (slow)
    int generation_backfill_max_tokens =
        50;  // max_length passed to generate_response() during backfill

    // Outlier detection thresholds (TD-021, moved from MetricsServiceConfig)
    float loss_outlier_z_threshold = 3.0f;      // Flag sample if loss > epoch_mean + N×std
    float grad_norm_outlier_threshold = 10.0f;  // Flag sample if grad_norm exceeds this value
    int max_abnormal_samples = 1000;            // Cap on flagged samples reported per training run

    // Tokenizer mode: ASCII (default, byte-level) or UNICODE (UTF-8 code-point-level)
    TokenizerMode tokenizer_mode = TokenizerMode::ASCII;

    // On-disk tokenized-data cache: preprocess_data() skips its BPE-encode loops
    // entirely on a cache hit (see ChatbotTrainer.cpp). Mirrors
    // DatasetConfig::cache_tokenized_data/tokenized_cache_dir (DatasetRegistry.hpp)
    // — IncrementalTrainer copies those two here and computes tokenized_cache_key
    // fresh before each ChatbotTrainer construction, so this class itself stays
    // unaware of DatasetRegistry/file checksums.
    bool cache_tokenized_data = false;
    std::string tokenized_cache_dir = "tokenized_cache";
    std::string tokenized_cache_key;  // empty = cache effectively disabled for this run
};

/**
 * @brief Chatbot model trainer
 *
 * Provides comprehensive training pipeline for transformer-based chatbot models:
 * - Vocabulary management (load or build BPE tokenizer)
 * - Data loading and preprocessing
 * - Advanced training with gradient accumulation
 * - Multiple learning rate scheduling strategies
 * - Proper validation (inference-only)
 * - Checkpointing and resume capability
 * - Early stopping
 * - Metrics tracking (loss, perplexity, accuracy)
 * - Configurable logging levels
 */
/**
 * @brief Per-epoch callback invoked at the end of each training epoch (TD-009)
 *
 * @param epoch     0-based epoch index
 * @param total     total number of epochs requested
 * @param loss      training loss for this epoch
 * @param val_loss  validation loss for this epoch (0.0 if no validation data)
 * @param lr        current learning rate at end of this epoch
 */
using EpochCallback =
    std::function<void(int epoch, int total, float loss, float val_loss, float lr)>;

/**
 * @brief Per-sample callback invoked after each optimizer step inside an epoch.
 *
 * @param sample         1-based sample index within the current epoch
 * @param total_samples  total number of training samples in the epoch
 * @param running_loss   running-average training loss so far this epoch
 * @param step_loss      average loss for this specific optimizer step
 * @param grad_norm      gradient norm for this optimizer step
 */
using SampleCallback = std::function<void(int sample, int total_samples, float running_loss,
                                          float step_loss, float grad_norm, float lr)>;

/**
 * @brief Callback invoked whenever a new best validation loss is recorded and
 *        the best-model snapshot has been flushed to disk.
 *
 * @param epoch    1-based epoch number that achieved the new best
 * @param val_loss new best validation loss
 */
using BestModelCallback = std::function<void(int epoch, float val_loss)>;

class ChatbotTrainer {
   private:
    // Allow the tokenized-data cache's unit tests direct access to
    // load_tokenized_cache()/save_tokenized_cache() and the tokenized_*/
    // training_data/validation_data members — a full train() integration
    // test would need a real model+tokenizer and actual forward/backward
    // passes just to exercise preprocess_data(), which this codebase's other
    // ChatbotTrainer tests already avoid (see chatbottrainer_test.cpp).
    friend class ChatbotTrainerCacheTest;

    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<EncoderDecoderModel> model;
    std::unique_ptr<Optimizer> optimizer;
    TrainingConfig config;

    std::vector<ConversationPair> training_data;
    std::vector<ConversationPair> validation_data;

    // Pre-tokenized data for efficient training
    std::vector<TokenizedPair> tokenized_training_data;
    std::vector<TokenizedPair> tokenized_validation_data;
    std::vector<int> training_indices;  // For shuffling

    // Training statistics
    std::vector<float> training_losses;
    std::vector<float> validation_losses;
    std::vector<float> learning_rates;
    std::vector<float> gradient_norms;
    std::vector<float> training_perplexities;
    std::vector<float> validation_perplexities;
    std::vector<float> training_accuracies;
    std::vector<float> validation_accuracies;
    float best_validation_loss;
    int best_epoch{0};

    // Learning rate scheduling state
    int global_step{0};
    int total_training_steps{0};
    float current_learning_rate;

    // Gradient accumulation state
    int accumulation_step{0};
    float accumulated_loss{0.0f};

    // Early stopping state
    int epochs_without_improvement{0};
    std::string best_model_path;
    bool early_stopped{false};

    // TD-009: Per-epoch monitoring callback
    EpochCallback epoch_callback_;

    // Per-sample monitoring callback
    SampleCallback sample_callback_;

    // Fired whenever a new best validation loss is saved to disk
    BestModelCallback best_model_callback_;

    IMetricsReporter* metrics_reporter_{nullptr};
    int abnormal_sample_count_ = 0;  ///< flagged samples reported this training run (TD-021)

    // Cooperative abort (incremental_trainer's `serve` admin API — /admin/pause).
    // Not owned; caller (IncrementalTrainer) guarantees the pointee outlives
    // this ChatbotTrainer instance. Checked only at optimizer-step boundaries
    // (accumulation_step == 0) in train_epoch()'s per-sample loop — never
    // mid-accumulation-window, so a pause never leaves a half-applied gradient
    // update. nullptr (the default) means abort is never requested.
    const std::atomic<bool>* abort_flag_{nullptr};
    bool was_aborted_{false};

    // TD-023: background generation-quality scoring thread
    std::optional<std::thread> generation_quality_thread_;

    // Private helper methods
    void validate_and_correct_config();
    void split_data();
    void preprocess_data();
    void shuffle_training_data();
    /**
     * @brief Loads tokenized_training_data/tokenized_validation_data from
     * @p cache_path if present. Returns false (leaving both untouched) on any
     * missing file, read error, or sample-count mismatch against the current
     * training_data/validation_data sizes — callers must fall back to
     * re-tokenizing on a false return, never treat it as fatal.
     */
    bool load_tokenized_cache(const std::string& cache_path);
    /**
     * @brief Serializes the current tokenized_training_data/
     * tokenized_validation_data to @p cache_path. Best-effort — logs and
     * returns on any write failure rather than throwing; a failed cache write
     * must never fail training itself.
     */
    void save_tokenized_cache(const std::string& cache_path) const;
    /**
     * @brief Single entry point for EncoderDecoderModel construction.
     * Reads vocab size from the current tokenizer and all architecture
     * dimensions from config.  Every code path that creates the model
     * must call this — there is no other direct make_unique<EncoderDecoderModel>()
     * call in this class.
     * @pre tokenizer must be initialised and not yet transferred to the model.
     */
    void build_model();
    void initialize_model();
    float calculate_learning_rate(int step);
    void update_learning_rate();
    std::string get_schedule_name();
    float train_epoch(int epoch);
    float validate();
    /**
     * @brief Compute BLEU-4 and ROUGE-1/2/L on a random subset of validation pairs.
     *
     * Calls model->generate_response() in eval mode and scores the outputs
     * against reference targets.  Results are pushed to metrics_reporter_.
     * No-op when metrics_reporter_ is null or config option is disabled.
     */
    void compute_generation_quality_metrics();
    /**
     * @brief Join `generation_quality_thread_` if it is joinable (TD-023).
     *
     * Called at the top of each `compute_generation_quality_metrics()` call and
     * inside the destructor / `release_model()` to guarantee the scoring thread
     * has finished before weights are modified or released.
     */
    void join_generation_quality_thread();
    /**
     * @brief Overwrite meta.quality on all training and validation samples using
     *        per-sample generation BLEU4. Called once after the final epoch when
     *        config.enable_generation_quality_backfill is true.
     *
     * Runs in eval mode; expensive for large datasets.
     */
    void backfill_generation_quality();

   public:
    /**
     * @brief Construct trainer with configuration
     */
    ChatbotTrainer(const TrainingConfig& cfg);

    /**
     * @brief Destructor — joins any outstanding generation quality thread (TD-023)
     */
    ~ChatbotTrainer();
    ChatbotTrainer(const ChatbotTrainer&) = delete;
    ChatbotTrainer& operator=(const ChatbotTrainer&) = delete;
    ChatbotTrainer(ChatbotTrainer&&) = delete;
    ChatbotTrainer& operator=(ChatbotTrainer&&) = delete;

    /**
     * @brief Load tokenizer from vocabulary file
     */
    bool load_tokenizer(const std::string& vocab_path);

    /**
     * @brief Build vocabulary from text corpus
     */
    bool build_vocabulary(const std::vector<std::string>& texts, int vocab_size = 5000,
                          const std::string& save_path = "vocab.txt");

    /**
     * @brief Load conversation pairs from file
     */
    bool load_conversation_data(const std::string& filepath);

    /**
     * @brief Train the model and return success status (for incremental training)
     */
    bool train(int num_epochs);

    // Model and tokenizer ownership transfer (for incremental training)
    void set_tokenizer(std::unique_ptr<BPETokenizer> tok);
    void set_model(std::unique_ptr<EncoderDecoderModel> mdl);
    std::unique_ptr<EncoderDecoderModel> release_model();
    BPETokenizer* release_tokenizer();

    // Data management
    void add_training_pair(const std::string& input, const std::string& response);
    void add_validation_pair(const std::string& input, const std::string& response);

    // Access final metrics
    float get_final_training_loss() const;
    float get_final_validation_loss() const;

    // Metrics and logging helpers (public for testing)

    /**
     * @brief Log message based on log level
     */
    void log(LogLevel level, const std::string& message, const std::string& color = "");

    /**
     * @brief Calculate perplexity from loss
     */
    static float calculate_perplexity(float loss);

    /**
     * @brief Truncate text to at most max_chars characters, keeping the TAIL
     *        (end of the string) rather than the head. Used by preprocess_data()
     *        for encoder input text, so a truncated input stays adjacent to
     *        where the paired decoder target begins (training pairs are a
     *        document split at its midpoint) instead of keeping the document's
     *        opening text, arbitrarily far from what the target continues.
     *        Breaks at a valid UTF-8 character boundary.
     */
    static std::string truncate_text_tail(const std::string& s, size_t max_chars);

    /**
     * @brief Truncate a token-id sequence to at most max_len ids, keeping the
     *        TAIL rather than the head. Token-level counterpart to
     *        truncate_text_tail(), applied after BPE encoding.
     */
    static std::vector<int> truncate_tokens_tail(std::vector<int> ids, int max_len);

    /**
     * @brief Calculate token-level accuracy (fraction of positions where predictions match targets)
     */
    static float calculate_accuracy(const std::vector<int>& predictions,
                                    const std::vector<int>& targets);

    // Getters for testing — also how IncrementalTrainer integration consumes per-epoch metrics
    // (TD-004, absorbed into and completed by TD-009).
    const std::vector<float>& get_training_losses() const {
        return training_losses;
    }
    const std::vector<float>& get_validation_losses() const {
        return validation_losses;
    }
    const std::vector<float>& get_training_perplexities() const {
        return training_perplexities;
    }
    const std::vector<float>& get_validation_perplexities() const {
        return validation_perplexities;
    }
    const std::vector<float>& get_learning_rates() const {
        return learning_rates;
    }
    const std::vector<float>& get_gradient_norms() const {
        return gradient_norms;
    }
    float get_best_validation_loss() const {
        return best_validation_loss;
    }
    int get_best_epoch() const {
        return best_epoch;
    }
    bool was_early_stopped() const {
        return early_stopped;
    }
    int get_global_step() const {
        return global_step;
    }
    float get_current_learning_rate() const {
        return current_learning_rate;
    }
    const TrainingConfig& get_config() const {
        return config;
    }

    /**
     * @brief Register a callback invoked at the end of each training epoch (TD-009)
     * @param cb Callback function; pass {} or nullptr to clear.
     */
    void set_epoch_callback(EpochCallback cb);

    /**
     * @brief Register a callback invoked after each optimizer step (per sample).
     * @param cb Callback function; pass {} or nullptr to clear.
     */
    void set_sample_callback(SampleCallback cb);

    /**
     * @brief Register a callback invoked whenever a new best validation loss is
     *        found and the model snapshot has been flushed to disk.
     * @param cb Callback function; pass {} or nullptr to clear.
     */
    void set_best_model_callback(BestModelCallback cb);

    /**
     * @brief Save the current in-memory model weights to @p path.
     *        Intended for use inside a BestModelCallback to persist the snapshot
     *        to an additional location.
     */
    void save_to(const std::string& path);

    /**
     * @brief Set the metrics reporter for in-epoch reporting (TD-021).
     * @param reporter Pointer to IMetricsReporter (not owned, may be nullptr to disable)
     */
    void set_metrics_reporter(IMetricsReporter* reporter);

    /**
     * @brief Register a cooperative-abort flag (incremental_trainer `serve`'s
     *        /admin/pause). When *flag becomes true, train_epoch() breaks out
     *        of its per-sample loop at the next optimizer-step boundary
     *        (accumulation_step == 0) — never mid-accumulation-window — and
     *        train()'s epoch loop stops starting further epochs. Pass nullptr
     *        (the default) to disable; not owned, caller must outlive this
     *        ChatbotTrainer.
     */
    void set_abort_flag(const std::atomic<bool>* flag) {
        abort_flag_ = flag;
    }

    /**
     * @brief True if the most recent train() call ended early because
     *        abort_flag_ was set, rather than completing all epochs or
     *        stopping via early-stopping. Reset to false at the start of
     *        each train() call.
     */
    bool was_aborted() const {
        return was_aborted_;
    }

    // For testing data management
    size_t get_training_data_size() const {
        return training_data.size();
    }
    size_t get_validation_data_size() const {
        return validation_data.size();
    }
    size_t get_tokenized_training_size() const {
        return tokenized_training_data.size();
    }
    size_t get_tokenized_validation_size() const {
        return tokenized_validation_data.size();
    }
};
