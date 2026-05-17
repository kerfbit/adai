#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/**
 * @brief Real-time training metrics snapshot
 */
struct TrainingMetricsSnapshot {
    // Session info
    int session_id = 0;
    bool is_training = false;
    std::chrono::system_clock::time_point session_start_time;
    std::chrono::system_clock::time_point last_update_time;

    // Current epoch info
    int current_epoch = 0;
    int total_epochs = 0;
    int current_sample = 0;
    int total_samples = 0;

    // Real-time metrics
    float current_loss = 0.0f;
    float current_validation_loss = 0.0f;
    float current_learning_rate = 0.0f;
    float current_gradient_norm = 0.0f;
    float current_perplexity = 0.0f;
    float current_validation_perplexity =
        0.0f;                                   ///< TD-015: validation perplexity for current epoch
    float current_validation_accuracy = -1.0f;  ///< TD-015: validation accuracy (-1 = not computed)

    // Running averages
    float running_loss = 0.0f;
    float running_validation_loss = 0.0f;

    // Per-epoch history
    std::vector<float> epoch_losses;
    std::vector<float> epoch_validation_losses;
    std::vector<float> epoch_learning_rates;
    std::vector<float> epoch_perplexities;
    std::vector<float> epoch_validation_perplexities;  ///< TD-015: per-epoch validation perplexity
    std::vector<float>
        epoch_validation_accuracies;  ///< TD-015: per-epoch validation accuracy (-1 = not computed)
    std::vector<double> epoch_durations;
    std::vector<float> epoch_gradient_norms;

    // Cumulative stats
    int total_samples_trained = 0;
    double total_training_time_seconds = 0.0;
    float best_validation_loss = std::numeric_limits<float>::max();
    int best_epoch = 0;

    // Throughput metrics
    float samples_per_second = 0.0f;
    float estimated_time_remaining_seconds = 0.0f;

    // Advanced metrics (TD-013)
    float gradient_variance = 0.0f;    ///< Variance of per-step gradient norms within current epoch
    float compute_time_ratio = 0.0f;   ///< Fraction of epoch wall time spent in forward+backward
    float weight_update_ratio = 0.0f;  ///< Avg (lr * ||g||) / ||w|| ratio across optimizer steps
    float activation_saturation_ratio = -1.0f;  ///< Avg fraction of near-zero post-GELU units
                                                ///< across all FF layers (-1 = not computed)
    float attention_entropy = -1.0f;  ///< Avg per-token attention entropy across all self-attention
                                      ///< layers (-1 = not computed)

    // Generation quality metrics (BLEU/ROUGE, -1 = not computed)
    float current_bleu4 = -1.0f;   ///< Corpus BLEU-4 score for the current validation epoch
    float current_rouge1 = -1.0f;  ///< Macro-avg ROUGE-1 F1 for the current validation epoch
    float current_rouge2 = -1.0f;  ///< Macro-avg ROUGE-2 F1 for the current validation epoch
    float current_rougeL = -1.0f;  ///< Macro-avg ROUGE-L F1 for the current validation epoch

    // Per-epoch generation quality history (-1 entries = not computed for that epoch)
    std::vector<float> epoch_bleu4;
    std::vector<float> epoch_rouge1;
    std::vector<float> epoch_rouge2;
    std::vector<float> epoch_rougeL;

    // Batch padding efficiency (-1 = not computed)
    /// Avg fraction of non-padding tokens across all gradient-accumulation windows in the current
    /// epoch. 1.0 = all tokens are real (no wasted padding); lower values indicate sequence-length
    /// mismatch within accumulation windows. Trivially 1.0 when gradient_accumulation_steps == 1.
    float current_padding_efficiency = -1.0f;
    std::vector<float> epoch_padding_efficiencies;  ///< Per-epoch history (-1 = not computed)

    // Adaptive gradient clipping (TD-017; -1 / 0 = not used / not computed)
    float current_adaptive_clip_threshold =
        -1.0f;  ///< Effective clip threshold for the latest optimizer step
    int current_adaptive_clip_spikes = 0;  ///< Cumulative spike count for the current epoch
    std::vector<float> epoch_adaptive_clip_thresholds;  ///< Per-epoch average effective threshold
                                                        ///< (-1 = fixed-clip mode)
};

/**
 * @brief Abnormal training sample flagged by outlier detection (TD-013)
 */
struct AbnormalSample {
    int epoch = 0;            ///< 1-based epoch number
    int sample_id = 0;        ///< 1-based sample index within the epoch
    float loss = 0.0f;        ///< Loss value that triggered the flag
    float grad_norm = 0.0f;   ///< Gradient norm that triggered the flag
    std::string reason;       ///< Human-readable reason (e.g. "loss_outlier", "grad_norm_outlier")
    std::string input_text;   ///< Input text of the offending sample
    std::string target_text;  ///< Target text of the offending sample
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Persistent metrics record for historical tracking
 */
struct PersistentMetricsRecord {
    std::chrono::system_clock::time_point timestamp;
    int session_id;
    int epoch;
    int sample;
    float loss;
    float validation_loss;
    float learning_rate;
    float gradient_norm;
    float perplexity;
};

/**
 * @brief Configuration for metrics service
 */
struct MetricsServiceConfig {
    // Persistence settings
    bool enable_persistence = true;
    std::string metrics_file = "training_sessions/metrics.jsonl";  // JSON Lines format
    std::string summary_file = "training_sessions/metrics_summary.json";
    int persist_every_samples = 100;  // Write to disk every N samples
    int persist_every_seconds = 30;   // Or every N seconds

    // Data retention
    int max_records_in_memory = 10000;
    int max_records_on_disk = 100000;
    bool compress_old_records = false;

    // Monitoring
    bool enable_prometheus_format = false;
    std::string prometheus_file = "training_sessions/metrics.prom";

    // Push to external metrics API daemon
    bool enable_push = false;
    std::string push_url = "http://localhost:8081";  // URL of metrics API daemon
    int push_timeout_ms = 1000;                      // HTTP request timeout

    // Outlier detection (TD-013)
    std::string abnormal_samples_file = "training_sessions/abnormal_samples.json";
    float loss_outlier_z_threshold = 3.0f;  // Flag sample if loss > epoch_mean + N*epoch_std
    float grad_norm_outlier_threshold =
        10.0f;                        // Flag sample if grad_norm exceeds this absolute value
    int max_abnormal_samples = 1000;  // Max outlier records kept in memory

    // Generation quality metrics (BLEU/ROUGE) — disabled by default because generation is expensive
    bool enable_generation_quality =
        false;  // When true, generate responses during validation and score them
    int generation_quality_sample_size = 10;  // Number of validation samples used for scoring
};

/**
 * @brief Training Metrics Service - Pollable daemon for tracking training progress
 *
 * This service provides:
 * - Thread-safe real-time metrics access
 * - Persistent metrics storage
 * - Historical metrics querying
 * - Multiple output formats (JSON, Prometheus, CSV)
 * - Non-blocking polling interface
 *
 * Usage:
 *   MetricsService service(config);
 *   service.start_session(session_id);
 *
 *   // In training loop:
 *   service.update_sample_metrics(sample, loss, grad_norm, lr);
 *   service.update_epoch_metrics(epoch, loss, val_loss, lr);
 *
 *   // From monitoring thread:
 *   auto snapshot = service.get_current_snapshot();
 *   std::string json = service.to_json();
 */
class TrainingMetricsService {
   public:
    /**
     * @brief Construct metrics service with configuration
     */
    explicit TrainingMetricsService(const MetricsServiceConfig& config = MetricsServiceConfig());

    /**
     * @brief Destructor - ensures all metrics are persisted
     */
    ~TrainingMetricsService();

    // Session lifecycle
    void start_session(int session_id, int total_epochs = 0, int total_samples = 0);
    void end_session();
    bool is_session_active() const;

    // Epoch lifecycle
    void start_epoch(int epoch, int total_samples = 0);
    void end_epoch(int epoch, float loss, float validation_loss, float learning_rate,
                   float perplexity = 0.0f, float gradient_norm = 0.0f,
                   double epoch_time_seconds = 0.0);

    // Real-time updates (called from training callbacks)
    void update_sample_metrics(int sample, float loss, float gradient_norm, float learning_rate);
    /**
     * @brief Update validation metrics for the current epoch (TD-015)
     * @param validation_loss  Average loss on the validation set
     * @param validation_accuracy  Token-level accuracy (-1.0 = not computed)
     * @param validation_perplexity  Perplexity (0 = auto-derived from loss)
     */
    void update_validation_metrics(float validation_loss, float validation_accuracy = -1.0f,
                                   float validation_perplexity = 0.0f);
    void update_best_metrics(float validation_loss, int epoch);

    // Advanced epoch-level diagnostics (TD-013)
    void update_advanced_epoch_metrics(float gradient_variance, float compute_time_ratio,
                                       float weight_update_ratio);

    // Adaptive gradient clipping (TD-017)
    /// Called once per optimizer step when adaptive clipping is active.
    void update_adaptive_clip_metrics(float effective_clip_threshold, int cumulative_spike_count);
    /// Called once per epoch with the epoch-average effective threshold and total spike count.
    void update_adaptive_clip_epoch(float avg_clip_threshold, int total_spike_count);

    /**
     * @brief Update generation quality metrics for the current epoch.
     *
     * Stores corpus BLEU-4 and macro-averaged ROUGE-1/2/L F1 scores computed
     * by running generate_response() on a sample of validation pairs.
     * Values outside [0, 1] (e.g. -1) are accepted as "not computed" markers.
     *
     * @param bleu4   Corpus BLEU-4 score (0–1, or -1 if not available)
     * @param rouge1  ROUGE-1 F1 (0–1, or -1 if not available)
     * @param rouge2  ROUGE-2 F1 (0–1, or -1 if not available)
     * @param rougeL  ROUGE-L F1 (0–1, or -1 if not available)
     */
    void update_generation_quality_metrics(float bleu4, float rouge1, float rouge2, float rougeL);

    /**
     * @brief Update batch padding efficiency for the current epoch.
     *
     * @param efficiency Average fraction of non-padding tokens across all gradient-accumulation
     *                   windows processed so far this epoch (0–1, or -1 = not computed).
     */
    void update_padding_efficiency(float efficiency);

    /**
     * @brief Update epoch-average activation saturation ratio (TD-013)
     * @param ratio Fraction of near-zero post-GELU units averaged across all
     *              FeedForward layers and forward passes in the epoch.
     *              Pass -1.0f to indicate "not computed".
     */
    void update_activation_saturation(float ratio);

    /// Update average per-token attention entropy for the current epoch.
    /// Pass -1.0f to mark "not computed".
    void update_attention_entropy(float entropy);

    // Outlier / abnormal-sample tracking (TD-013)
    void flag_abnormal_sample(const AbnormalSample& sample);
    std::vector<AbnormalSample> get_abnormal_samples() const;

    // Polling interface (thread-safe, non-blocking)
    TrainingMetricsSnapshot get_current_snapshot() const;
    std::string to_json() const;
    std::string to_json_summary() const;
    std::string to_prometheus() const;
    std::string to_csv_header() const;
    std::string to_csv_row() const;

    // Historical queries
    std::vector<PersistentMetricsRecord> get_history(int max_records = 1000) const;
    std::vector<PersistentMetricsRecord> get_session_history(int session_id) const;
    std::vector<float> get_epoch_losses() const;
    std::vector<float> get_epoch_validation_losses() const;

    // Persistence control
    void flush_to_disk();
    void clear_history();

    // Configuration
    void set_config(const MetricsServiceConfig& config);
    MetricsServiceConfig get_config() const;

   private:
    // Thread-safe state
    mutable std::mutex mutex_;
    std::atomic<bool> is_training_;
    std::atomic<int> current_session_id_;

    // Current metrics
    TrainingMetricsSnapshot current_snapshot_;

    // Historical records
    std::vector<PersistentMetricsRecord> history_;

    // Configuration
    MetricsServiceConfig config_;

    // Persistence state
    std::chrono::system_clock::time_point last_persist_time_;
    int samples_since_last_persist_;

    // Timing helpers
    std::chrono::steady_clock::time_point session_start_steady_;
    std::chrono::steady_clock::time_point epoch_start_steady_;

    // Private helpers
    void restore_from_summary();  // Restore snapshot from persisted summary file on startup
    void persist_metrics();
    void persist_summary();
    void persist_prometheus();
    void persist_summary_with_data(const std::string& json_data);
    void persist_prometheus_with_data(const std::string& prometheus_data);
    std::string to_json_summary_internal() const;  // No locking - caller must hold lock
    std::string to_prometheus_internal() const;    // No locking - caller must hold lock
    void add_record(const PersistentMetricsRecord& record);
    void trim_history();
    void update_throughput_metrics();
    std::string escape_json(const std::string& s) const;
    std::string format_timestamp(const std::chrono::system_clock::time_point& tp) const;

    // HTTP push to external metrics API daemon
    void push_to_api(const std::string& endpoint, const std::string& json_body);
    std::string build_push_url(const std::string& endpoint) const;

    // Outlier storage & persistence (TD-013)
    std::vector<AbnormalSample> abnormal_samples_;  // in-memory list of flagged samples
    void persist_abnormal_samples();                // write all to abnormal_samples_file
};

/**
 * @brief Global metrics service instance (optional singleton access)
 */
class GlobalMetricsService {
   public:
    static TrainingMetricsService& instance();
    static void initialize(const MetricsServiceConfig& config);
    static void shutdown();

   private:
    static std::unique_ptr<TrainingMetricsService> instance_;
    static std::mutex instance_mutex_;
};
