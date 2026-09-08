#pragma once

// @adai-status: beta        (capped by TD-039 — large, actively evolving)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-08


#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
// TD-021: AbnormalSample and IMetricsReporter are defined here.
// TrainingMetricsService does not implement IMetricsReporter; it is the
// server-side storage class.  The include makes AbnormalSample available to
// all translation units that include TrainingMetricsService.hpp.
#include "IMetricsReporter.hpp"
#include "MetricsDatabase.hpp"

/**
 * @brief Real-time training metrics snapshot
 */
struct TrainingMetricsSnapshot {
    // Session info
    int session_id = 0;
    bool is_training = false;
    std::string label;            ///< Human-readable session label (TD-021)
    std::string config_snapshot;  ///< Compact JSON config at session start (TD-021)
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

    // Per-layer gradient norms (TD-013 extension) — one entry per encoder/decoder
    // layer, in layer order, snapshotted once per epoch. Lets the dashboard show
    // whether gradients are shrinking uniformly across layers (healthy) or
    // specifically in early layers (vanishing-gradient signature), instead of
    // only the whole-model aggregate norm. Empty until the first epoch reports.
    std::vector<float> encoder_layer_grad_norms;
    std::vector<float> decoder_layer_grad_norms;

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

    // Stale-state detection (TD-019; computed on demand in get_current_snapshot() / to_json())
    bool is_stale = false;  ///< True when is_training && seconds since last ingest > threshold
    double seconds_since_last_update =
        0.0;  ///< Wall-clock seconds since the most recent metrics ingest
    bool effective_is_training =
        false;  ///< is_training && !is_stale — safe liveness signal for dashboards/health checks
};

// AbnormalSample is defined in IMetricsReporter.hpp (TD-021).
// It remains accessible here via the transitive include above.

/**
 * @brief Persistent metrics record for historical tracking
 */
struct PersistentMetricsRecord {
    std::chrono::system_clock::time_point timestamp;
    int session_id = 0;
    int epoch = 0;
    int sample = 0;
    float loss = 0.0f;
    float validation_loss = 0.0f;
    float learning_rate = 0.0f;
    float gradient_norm = 0.0f;
    float perplexity = 0.0f;

    // TD-013 diagnostics — populated only on the epoch-end record (created in
    // end_epoch()), left at their defaults on per-sample records. Unlike
    // gradient_variance (see insert_gradient_variance_sample()), these don't
    // need per-optimizer-step granularity, so they piggyback on the existing
    // epoch-boundary persistence instead of a dedicated table.
    float compute_time_ratio = 0.0f;
    float weight_update_ratio = 0.0f;
    float activation_saturation_ratio = -1.0f;
    float attention_entropy = -1.0f;
    float padding_efficiency = -1.0f;
    // JSON string {"encoder":[...],"decoder":[...]} of per-layer gradient
    // norms, or empty if not reported this epoch. Stored pre-serialized
    // (rather than as vectors) since this is the exact form persisted to and
    // read back from the DB's TEXT column.
    std::string layer_gradient_norms_json;
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
    std::string push_url = "http://localhost:8081";  // Base URL of metrics API daemon
    std::string session_key;     // Optional session key for /api/sessions/{key} push routing
    int push_timeout_ms = 1000;  // HTTP request timeout

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

    // Staleness detection (TD-019)
    /// Seconds without an ingest before is_stale is set on snapshot reads (default: 60).
    int staleness_threshold_seconds = 60;
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
    explicit TrainingMetricsService(MetricsServiceConfig config = MetricsServiceConfig());

    /**
     * @brief Destructor - ensures all metrics are persisted
     */
    ~TrainingMetricsService();
    TrainingMetricsService(const TrainingMetricsService&) = delete;
    TrainingMetricsService& operator=(const TrainingMetricsService&) = delete;
    TrainingMetricsService(TrainingMetricsService&&) = delete;
    TrainingMetricsService& operator=(TrainingMetricsService&&) = delete;

    // Session lifecycle
    //
    // reset_best: when false (default), best_validation_loss/best_epoch carry
    // forward from the previous session on this service if the architecture
    // (d_model/heads/d_ff/layers) is unchanged — intentional so incremental
    // training doesn't lose track of "best ever" between calls. Pass true for
    // a session that starts over from scratch (e.g. a full retrain) so the
    // dashboard doesn't show an unrelated prior session's best.
    void start_session(int session_id, int total_epochs = 0, int total_samples = 0,
                       const std::string& label = "", const std::string& config_snapshot = "",
                       bool reset_best = false);
    void end_session();
    bool is_session_active() const;
    void heartbeat();

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

    /// Report per-layer gradient norms for the current epoch (TD-013 extension).
    /// @param encoder_layer_norms One entry per encoder layer, in layer order
    /// @param decoder_layer_norms One entry per decoder layer, in layer order
    void update_layer_gradient_norms(const std::vector<float>& encoder_layer_norms,
                                     const std::vector<float>& decoder_layer_norms);

    // Outlier / abnormal-sample tracking (TD-013)
    void flag_abnormal_sample(const AbnormalSample& sample);
    std::vector<AbnormalSample> get_abnormal_samples() const;

    // Polling interface (thread-safe, non-blocking)
    TrainingMetricsSnapshot get_current_snapshot() const;
    std::string to_json() const;
    std::string to_json_summary() const;
    std::string to_prometheus(const std::string& session_key = "") const;
    static std::string to_csv_header();
    std::string to_csv_row() const;

    // Historical queries
    std::vector<PersistentMetricsRecord> get_history(int max_records = 1000) const;
    std::vector<PersistentMetricsRecord> get_session_history(int session_id) const;
    std::vector<float> get_epoch_losses() const;
    std::vector<float> get_epoch_validation_losses() const;

    // Persistence control
    void flush_to_disk();
    // Clears per-sample history AND resets best_validation_loss/best_epoch
    // (persisting the reset immediately) — the explicit "force clear" for a
    // session, since start_session()'s cross-session carryover otherwise keeps
    // resurrecting whatever best was last recorded for this key.
    void clear_history();

    // Configuration
    void set_config(const MetricsServiceConfig& config);
    MetricsServiceConfig get_config() const;

    // Database persistence (TD-020)
    void set_database(IMetricsDatabase* db, const std::string& session_key);
    IMetricsDatabase* get_database() const {
        return db_;
    }

   private:
    // Thread-safe state
    mutable std::mutex mutex_;
    std::atomic<bool> is_training_;
    // Resolved by TD-018: multi-session support comes from MetricsSessionRegistry owning one
    // TrainingMetricsService instance per session key, not from this class tracking multiple
    // sessions internally — so a single current_session_id_ per instance is correct as-is.
    std::atomic<int> current_session_id_;

    // Current metrics
    TrainingMetricsSnapshot current_snapshot_;

    // Historical records — a rolling in-memory window (capped at
    // config_.max_records_in_memory by trim_history()) used to serve recent-history
    // queries; NOT a "pending persist" queue. persisted_up_to_ tracks how much of it
    // persist_metrics() has already written out, so periodic persistence only writes
    // newly-added records instead of re-writing the whole window every time.
    std::vector<PersistentMetricsRecord> history_;
    std::size_t persisted_up_to_{0};

    // Configuration
    MetricsServiceConfig config_;

    // Persistence state
    std::chrono::system_clock::time_point last_persist_time_;
    int samples_since_last_persist_{0};

    // Timing helpers
    std::chrono::steady_clock::time_point session_start_steady_;
    std::chrono::steady_clock::time_point epoch_start_steady_;

    // Validation-gap staleness extension
    bool awaiting_validation_ =
        false;  ///< Set on last training sample of an epoch; cleared by end_epoch/start_epoch
    double last_epoch_training_duration_seconds_ =
        0.0;  ///< Training-phase wall time of the most recent epoch

    // Raw in-epoch sample index as of the last update_sample_metrics() call.
    // update_sample_metrics() is only invoked once per gradient-accumulation
    // window, not once per raw sample, so `sample - last_sample_in_epoch_`
    // gives the number of raw samples represented by this call. Reset at each
    // start_epoch() alongside current_sample.
    int last_sample_in_epoch_ = 0;

    // Private helpers
    void restore_from_summary();  // Restore snapshot from persisted summary file on startup
    void persist_metrics();
    void persist_summary();
    void persist_prometheus();
    void persist_summary_with_data(const std::string& json_data);
    void persist_prometheus_with_data(const std::string& prometheus_data);
    std::string to_json_summary_internal() const;  // No locking - caller must hold lock
    std::string label_;            // Session label stored across session lifetime (TD-021)
    std::string config_snapshot_;  // Compact config JSON stored across session lifetime (TD-021)

    std::string to_prometheus_internal(
        const std::string& session_key = "") const;  // No locking - caller must hold lock
    void add_record(const PersistentMetricsRecord& record);
    void trim_history();
    void update_throughput_metrics();
    static std::string escape_json(const std::string& s);
    static std::string format_timestamp(const std::chrono::system_clock::time_point& tp);

    // HTTP push to external metrics API daemon
    void push_to_api(const std::string& endpoint, const std::string& json_body);
    std::string build_push_url(const std::string& endpoint) const;

    // Outlier storage & persistence (TD-013)
    std::vector<AbnormalSample> abnormal_samples_;  // in-memory list of flagged samples
    void persist_abnormal_samples();                // write all to abnormal_samples_file

    // SQL database persistence (TD-020)
    IMetricsDatabase* db_ = nullptr;
    std::string session_key_;
    SessionRecord build_session_record() const;
    // Monotonic step counter for insert_gradient_variance_sample() — this is a
    // server-side ordinal (one per update_advanced_epoch_metrics() call), not
    // the trainer's own sample/step index, which never crosses the wire.
    int gradient_variance_step_counter_{0};
};

class MetricsSessionRegistry;

/**
 * @brief Global metrics service instance (optional singleton access)
 *
 * Resolved by TD-018: this is now a thin proxy through the "0-default" slot of a
 * MetricsSessionRegistry, kept deliberately (not replaced) so the pre-existing
 * instance().start_session(id, ...) call-site API didn't need to change everywhere
 * it's used.
 */
class GlobalMetricsService {
   public:
    static TrainingMetricsService& instance();
    static void initialize(const MetricsServiceConfig& config);
    static void shutdown();

   private:
    static std::unique_ptr<MetricsSessionRegistry> registry_;
    static std::mutex instance_mutex_;
};
