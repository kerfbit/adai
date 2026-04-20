#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include "BPETokenizer.hpp"
#include "EncoderDecoderModel.hpp"
#include "Optimizer.hpp"
#include "TrainingMetricsService.hpp"

// Parallel optimizations (Priority 1-5)
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Training data pair (input, target response)
 */
struct ConversationPair {
    std::string input;
    std::string response;

    ConversationPair(const std::string& in, const std::string& resp) : input(in), response(resp) {}
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
                  const std::string& in_txt, const std::string& tgt_txt)
        : input_tokens(in_tok), target_tokens(tgt_tok), input_text(in_txt), target_text(tgt_txt) {}
};

/**
 * @brief Learning rate scheduling strategy
 */
enum class LRSchedule {
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
enum class LogLevel {
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
    float learning_rate = 0.001f;     // Initial/base learning rate
    int batch_size = 1;               // Batch size (samples per gradient accumulation)
    int gradient_accumulation_steps = 1;  // Accumulate gradients over N steps
    int validation_split = 10;        // Use 1/10 of data for validation

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

    // Checkpointing
    bool save_checkpoints = true;
    int checkpoint_every = 1;  // Save every N epochs
    bool keep_all_checkpoints = false;  // Keep all epoch checkpoints (default: keep only best)
    std::string resume_from_checkpoint;  // Path to checkpoint to resume from (empty = no resume)

    // Early stopping
    bool enable_early_stopping = false;
    int patience = 5;                  // Epochs to wait for improvement
    float min_delta = 1e-4f;           // Minimum change to qualify as improvement
    bool restore_best_weights = true;  // Restore best model after early stop

    // Logging
    int log_every = 10;     // Log every N samples
    LogLevel log_level = LogLevel::VERBOSE;  // Logging verbosity
    bool verbose = true;    // Deprecated: use log_level instead

    // Generation quality metrics (BLEU/ROUGE)
    // Disabled by default; each evaluation generates model responses which is expensive.
    bool enable_generation_quality_metrics = false;  // Compute BLEU/ROUGE during each validation pass
    int  generation_quality_sample_size    = 10;     // Max validation samples used for scoring
    int  generation_quality_max_tokens     = 50;     // max_length passed to generate_response()
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
using EpochCallback = std::function<void(int epoch, int total, float loss, float val_loss, float lr)>;

/**
 * @brief Per-sample callback invoked after each optimizer step inside an epoch.
 *
 * @param sample         1-based sample index within the current epoch
 * @param total_samples  total number of training samples in the epoch
 * @param running_loss   running-average training loss so far this epoch
 * @param step_loss      average loss for this specific optimizer step
 * @param grad_norm      gradient norm for this optimizer step
 */
using SampleCallback = std::function<void(int sample, int total_samples, float running_loss, float step_loss, float grad_norm, float lr)>;

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
    int best_epoch;

    // Learning rate scheduling state
    int global_step;
    int total_training_steps;
    float current_learning_rate;

    // Gradient accumulation state
    int accumulation_step;
    float accumulated_loss;

    // Early stopping state
    int epochs_without_improvement;
    std::string best_model_path;
    bool early_stopped;

    // Resume state
    int start_epoch;

    // TD-009: Per-epoch monitoring callback
    EpochCallback epoch_callback_;

    // Per-sample monitoring callback
    SampleCallback sample_callback_;

    // Fired whenever a new best validation loss is saved to disk
    BestModelCallback best_model_callback_;

    // TrainingMetricsService integration
    TrainingMetricsService* metrics_service_;

    // Private helper methods
    void validate_and_correct_config();
    void split_data();
    void preprocess_data();
    void shuffle_training_data();
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
     * against reference targets.  Results are pushed to metrics_service_.
     * No-op when metrics_service_ is null or config option is disabled.
     */
    void compute_generation_quality_metrics();
    bool should_early_stop();
    void restore_best_model();
    void save_checkpoint(const std::string& filepath, int epoch);
    void finalize_model(const std::string& output_path);
    bool load_checkpoint(const std::string& filepath);
    void print_training_summary(long duration);

   public:
    /**
     * @brief Construct trainer with configuration
     */
    ChatbotTrainer(const TrainingConfig& cfg);

    /**
     * @brief Destructor (smart pointers handle cleanup)
     */
    ~ChatbotTrainer() = default;

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
     * @brief Train the model
     */
    void train(const std::string& output_model_path);
    
    /**
     * @brief Train the model and return success status (for incremental training)
     */
    bool train(int num_epochs);

    /**
     * @brief Test generation with trained model
     */
    void test_generation(const std::vector<std::string>& test_prompts);
    
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
    float calculate_perplexity(float loss);

    /**
     * @brief Calculate token-level accuracy (placeholder)
     */
    float calculate_accuracy(const std::vector<int>& predictions, const std::vector<int>& targets);

    // Getters for testing
    // TODO: TD-004 - These getters expose per-epoch metrics for IncrementalTrainer integration
    const std::vector<float>& get_training_losses() const { return training_losses; }
    const std::vector<float>& get_validation_losses() const { return validation_losses; }
    const std::vector<float>& get_training_perplexities() const { return training_perplexities; }
    const std::vector<float>& get_validation_perplexities() const { return validation_perplexities; }
    const std::vector<float>& get_learning_rates() const { return learning_rates; }
    const std::vector<float>& get_gradient_norms() const { return gradient_norms; }
    float get_best_validation_loss() const { return best_validation_loss; }
    int get_best_epoch() const { return best_epoch; }
    bool was_early_stopped() const { return early_stopped; }
    int get_global_step() const { return global_step; }
    float get_current_learning_rate() const { return current_learning_rate; }
    const TrainingConfig& get_config() const { return config; }
    
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
     * @brief Set the TrainingMetricsService for metrics reporting.
     * @param service Pointer to TrainingMetricsService (not owned, can be nullptr to disable)
     */
    void set_metrics_service(TrainingMetricsService* service);

    // For testing data management
    size_t get_training_data_size() const { return training_data.size(); }
    size_t get_validation_data_size() const { return validation_data.size(); }
    size_t get_tokenized_training_size() const { return tokenized_training_data.size(); }
    size_t get_tokenized_validation_size() const { return tokenized_validation_data.size(); }
};
