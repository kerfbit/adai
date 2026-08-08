#pragma once

#include <chrono>
#include <string>
#include <vector>

/**
 * @brief Abnormal training sample flagged by outlier detection (TD-013)
 *
 * Canonical definition lives here (TD-021).  Moved from TrainingMetricsService.hpp
 * so that ChatbotTrainer can construct an AbnormalSample without depending on
 * the server-side header.  TrainingMetricsService.hpp now includes this file
 * and drops its own copy.
 */
struct AbnormalSample {
    int epoch = 0;           ///< 1-based epoch number
    int sample_id = 0;       ///< 1-based sample index within the epoch
    float loss = 0.0f;       ///< Loss value that triggered the flag
    float grad_norm = 0.0f;  ///< Gradient norm that triggered the flag
    std::string reason;      ///< e.g. "loss_outlier", "grad_norm_outlier"
    std::string input_text;
    std::string target_text;
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Abstract metrics reporter interface (TD-021)
 *
 * ChatbotTrainer holds an IMetricsReporter* and calls methods on it during
 * training.  The two concrete implementations are:
 *
 *   - MetricsPushClient   — non-blocking HTTP push to the metrics API server.
 *     A single background thread drains a bounded priority queue; all methods
 *     return in O(1) time.  Epoch/Session events are never dropped; per-sample
 *     events are lossy under backpressure.
 *
 *   - NullMetricsReporter — no-op used when metrics_server_url is empty.
 *     No push thread is started and no TrainingMetricsService is instantiated
 *     anywhere in the trainer process.
 *
 * Session lifecycle (start_session / end_session) is deliberately excluded
 * from this interface.  Those methods are called by IncrementalTrainer directly
 * on the concrete MetricsPushClient before/after invoking trainer.train().
 * ChatbotTrainer is only responsible for in-epoch and per-sample reporting.
 */
class IMetricsReporter {
   public:
    virtual ~IMetricsReporter() = default;

    // ── Epoch lifecycle ───────────────────────────────────────────────────────

    /// Signal the start of a new epoch.
    /// @param epoch        1-based epoch number
    /// @param total_samples Number of training samples for this epoch
    virtual void start_epoch(int epoch, int total_samples) = 0;

    /// Signal the end of an epoch with aggregate metrics.
    /// @param epoch              1-based epoch number
    /// @param loss               Average training loss for the epoch
    /// @param validation_loss    Average validation loss (0 if not computed)
    /// @param learning_rate      Learning rate at the end of the epoch
    /// @param perplexity         Training perplexity (0 = auto-derive from loss)
    /// @param gradient_norm      Average gradient norm for the epoch
    /// @param epoch_time_seconds Wall-clock seconds spent in this epoch
    virtual void end_epoch(int epoch, float loss, float validation_loss, float learning_rate,
                           float perplexity = 0.0f, float gradient_norm = 0.0f,
                           double epoch_time_seconds = 0.0) = 0;

    // ── Per-sample metrics ────────────────────────────────────────────────────

    /// Update real-time per-sample metrics after each optimizer step.
    /// @param sample        1-based sample index within the current epoch
    /// @param loss          Loss for this step (after gradient accumulation averaging)
    /// @param gradient_norm Raw gradient norm before clipping
    /// @param learning_rate Current learning rate
    virtual void update_sample_metrics(int sample, float loss, float gradient_norm,
                                       float learning_rate) = 0;

    // ── Validation metrics (TD-015) ───────────────────────────────────────────

    /// Update validation metrics for the current epoch.
    /// @param validation_loss       Average loss on the validation set
    /// @param validation_accuracy   Token-level accuracy (-1 = not computed)
    /// @param validation_perplexity Perplexity (0 = auto-derive from loss)
    virtual void update_validation_metrics(float validation_loss, float validation_accuracy,
                                           float validation_perplexity) = 0;

    /// Update the best-so-far validation metrics.
    /// @param validation_loss Best validation loss observed
    /// @param epoch           1-based epoch at which the best was achieved
    virtual void update_best_metrics(float validation_loss, int epoch) = 0;

    // ── Advanced epoch diagnostics (TD-013) ──────────────────────────────────

    /// Update advanced diagnostic accumulators (called once per optimizer step
    /// with running values, and once at epoch end with the final averages).
    /// @param gradient_variance  Variance of per-step gradient norms
    /// @param compute_time_ratio Fraction of wall time spent in forward+backward
    /// @param weight_update_ratio Average (lr * ||g||) / ||w|| across optimizer steps
    virtual void update_advanced_epoch_metrics(float gradient_variance, float compute_time_ratio,
                                               float weight_update_ratio) = 0;

    /// Flag a training sample detected as an outlier by the Welford z-score or
    /// gradient-norm threshold logic.
    virtual void flag_abnormal_sample(const AbnormalSample& sample) = 0;

    // ── Adaptive gradient clipping (TD-017) ──────────────────────────────────

    /// Called once per optimizer step when adaptive clipping is active.
    /// @param effective_clip_threshold Effective clip value applied this step
    /// @param cumulative_spike_count   Running spike count for the epoch so far
    virtual void update_adaptive_clip_metrics(float effective_clip_threshold,
                                              int cumulative_spike_count) = 0;

    /// Called once at epoch end with aggregate adaptive-clip statistics.
    /// @param avg_clip_threshold Epoch-average effective clip threshold
    /// @param total_spike_count  Total spike count for the epoch
    virtual void update_adaptive_clip_epoch(float avg_clip_threshold, int total_spike_count) = 0;

    // ── Activation / attention diagnostics (TD-013) ──────────────────────────

    /// Update epoch-average activation saturation ratio.
    /// @param ratio Fraction of near-zero post-GELU units (-1 = not computed)
    virtual void update_activation_saturation(float ratio) = 0;

    /// Update epoch-average per-token attention entropy.
    /// @param entropy Average Shannon entropy of the softmax distribution (-1 = not computed)
    virtual void update_attention_entropy(float entropy) = 0;

    /// Report per-layer gradient norms, once per epoch — the direct way to see
    /// whether gradients are shrinking uniformly (healthy convergence) or
    /// specifically in early layers (the classic vanishing-gradient signature),
    /// instead of only inferring it from the whole-model aggregate norm.
    /// @param encoder_layer_norms One entry per encoder layer, in layer order
    /// @param decoder_layer_norms One entry per decoder layer, in layer order
    virtual void update_layer_gradient_norms(const std::vector<float>& encoder_layer_norms,
                                             const std::vector<float>& decoder_layer_norms) = 0;

    // ── Batch padding efficiency ──────────────────────────────────────────────

    /// Update epoch-average batch padding efficiency.
    /// @param efficiency Fraction of non-padding tokens across accumulation windows
    ///                   (-1 = not computed; trivially 1.0 when accum steps == 1)
    virtual void update_padding_efficiency(float efficiency) = 0;

    // ── Generation quality metrics (TD-016) ──────────────────────────────────

    /// Update BLEU/ROUGE generation quality scores for the current epoch.
    /// All values are in [0, 1] or -1 when not computed.
    /// @param bleu4   Corpus BLEU-4 score
    /// @param rouge1  Macro-avg ROUGE-1 F1
    /// @param rouge2  Macro-avg ROUGE-2 F1
    /// @param rougeL  Macro-avg ROUGE-L F1
    virtual void update_generation_quality_metrics(float bleu4, float rouge1, float rouge2,
                                                   float rougeL) = 0;
};

/**
 * @brief No-op metrics reporter (TD-021)
 *
 * Used by IncrementalTrainer when metrics_server_url is empty or metrics
 * reporting is disabled.  All methods are no-ops; no push thread is started.
 */
class NullMetricsReporter final : public IMetricsReporter {
   public:
    void start_epoch(int /*epoch*/, int /*total_samples*/) override {}
    void end_epoch(int /*epoch*/, float /*loss*/, float /*validation_loss*/,
                   float /*learning_rate*/, float /*perplexity*/, float /*gradient_norm*/,
                   double /*epoch_time_seconds*/) override {}
    void update_sample_metrics(int /*sample*/, float /*loss*/, float /*gradient_norm*/,
                               float /*learning_rate*/) override {}
    void update_validation_metrics(float /*validation_loss*/, float /*validation_accuracy*/,
                                   float /*validation_perplexity*/) override {}
    void update_best_metrics(float /*validation_loss*/, int /*epoch*/) override {}
    void update_advanced_epoch_metrics(float /*gradient_variance*/, float /*compute_time_ratio*/,
                                       float /*weight_update_ratio*/) override {}
    void flag_abnormal_sample(const AbnormalSample& /*sample*/) override {}
    void update_adaptive_clip_metrics(float /*effective_clip_threshold*/,
                                      int /*cumulative_spike_count*/) override {}
    void update_adaptive_clip_epoch(float /*avg_clip_threshold*/,
                                    int /*total_spike_count*/) override {}
    void update_activation_saturation(float /*ratio*/) override {}
    void update_attention_entropy(float /*entropy*/) override {}
    void update_layer_gradient_norms(const std::vector<float>& /*encoder_layer_norms*/,
                                     const std::vector<float>& /*decoder_layer_norms*/) override {}
    void update_padding_efficiency(float /*efficiency*/) override {}
    void update_generation_quality_metrics(float /*bleu4*/, float /*rouge1*/, float /*rouge2*/,
                                           float /*rougeL*/) override {}
};
