#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include "IMetricsReporter.hpp"

// ============================================================================
// TD-021: MetricsPushClient — HTTP push client for the trainer process
// ============================================================================

/**
 * @brief Priority level for push events.
 *
 * Under queue backpressure:
 *   Sample  — dropped (per-sample data is intentionally lossy).
 *   Epoch   — the oldest Sample entry is evicted to make room; never dropped.
 *   Session — same eviction policy as Epoch; never dropped.
 */
enum class EventPriority { Sample, Epoch, Session };

/**
 * @brief A single queued HTTP POST event.
 */
struct PushEvent {
    EventPriority priority;
    std::string endpoint;  ///< Relative path appended to session_base_url_
    std::string body;      ///< JSON payload
};

/**
 * @brief HTTP-only metrics reporter for the trainer process (TD-021).
 *
 * Implements IMetricsReporter by enqueueing every call as a PushEvent that a
 * single background thread drains via HTTP POST to the metrics API server.
 * All IMetricsReporter methods return in O(1) time — the training loop is
 * never blocked by network I/O or server latency.
 *
 * Session lifecycle (start_session / end_session) is separate from
 * IMetricsReporter; IncrementalTrainer calls these directly on the concrete
 * MetricsPushClient object before/after invoking trainer.train().
 * ChatbotTrainer is only responsible for in-epoch and per-sample reporting.
 *
 * start_session() is a blocking synchronous call so that IncrementalTrainer
 * can inspect the HTTP status code and implement a 409-conflict retry loop
 * (see TD-021 §4.6).
 *
 * end_session() enqueues the session-end POST, signals the push thread to
 * stop, and joins it — guaranteeing no threads outlive this object.
 *
 * Several "state-only" IMetricsReporter methods (update_activation_saturation,
 * update_attention_entropy, update_padding_efficiency,
 * update_adaptive_clip_metrics, update_adaptive_clip_epoch) buffer their
 * values locally.  These values are folded into the end_epoch payload so the
 * server receives a single, coherent epoch summary.
 */
class MetricsPushClient final : public IMetricsReporter {
   public:
    /**
     * @param session_base_url  Session-scoped URL prefix, e.g.
     *                          "http://host:8081/api/sessions/1-devbox1234".
     *                          All POSTs are sent to "{session_base_url}{endpoint}".
     * @param timeout_ms        Per-request HTTP timeout in milliseconds.
     * @param max_queue_depth   Maximum number of events in the queue.  Sample
     *                          events are dropped when the queue is full;
     *                          Epoch/Session events evict the oldest Sample entry.
     */
    explicit MetricsPushClient(std::string session_base_url, int timeout_ms = 1000,
                               size_t max_queue_depth = 1024);
    ~MetricsPushClient() override;

    MetricsPushClient(const MetricsPushClient&) = delete;
    MetricsPushClient& operator=(const MetricsPushClient&) = delete;
    MetricsPushClient(MetricsPushClient&&) = delete;
    MetricsPushClient& operator=(MetricsPushClient&&) = delete;

    // ── Session lifecycle (called by IncrementalTrainer, not via IMetricsReporter*) ──

    /**
     * @brief Synchronously start a metrics session on the server.
     *
     * Sends POST {session_base_url}/start and returns the HTTP status code.
     * Returns 0 on connection failure or when session_base_url is empty.
     * The caller (IncrementalTrainer) uses the return value to detect 409
     * conflicts and retry with a different session key.
     *
     * @param session_id       Integer session identifier
     * @param total_epochs     Total epochs planned for this run
     * @param total_samples    Total training samples per epoch
     * @param label            Human-readable session label (may be empty)
     * @param config_snapshot  JSON object string of key training config fields
     *                         (may be empty; passed verbatim as the "config" field)
     * @return HTTP status code, or 0 on connection/timeout failure
     */
    int start_session(int session_id, int total_epochs, int total_samples,
                      const std::string& label = "",
                      const std::string& config_snapshot = "");

    /**
     * @brief Enqueue a session-end event and flush the queue.
     *
     * Enqueues a Session-priority POST to /end, sets the stop flag, and blocks
     * until the push thread has processed all queued events and exited.
     * Safe to call from the destructor if the caller forgets to call it
     * explicitly (idempotent after the first call).
     */
    void end_session();

    // ── IMetricsReporter ─────────────────────────────────────────────────────

    void start_epoch(int epoch, int total_samples) override;
    void end_epoch(int epoch, float loss, float validation_loss, float learning_rate,
                   float perplexity = 0.0f, float gradient_norm = 0.0f,
                   double epoch_time_seconds = 0.0) override;
    void update_sample_metrics(int sample, float loss, float gradient_norm,
                               float learning_rate) override;
    void update_validation_metrics(float validation_loss, float validation_accuracy,
                                   float validation_perplexity) override;
    void update_best_metrics(float validation_loss, int epoch) override;
    void update_advanced_epoch_metrics(float gradient_variance, float compute_time_ratio,
                                       float weight_update_ratio) override;
    void flag_abnormal_sample(const AbnormalSample& sample) override;
    void update_adaptive_clip_metrics(float effective_clip_threshold,
                                      int cumulative_spike_count) override;
    void update_adaptive_clip_epoch(float avg_clip_threshold, int total_spike_count) override;
    void update_activation_saturation(float ratio) override;
    void update_attention_entropy(float entropy) override;
    void update_padding_efficiency(float efficiency) override;
    void update_generation_quality_metrics(float bleu4, float rouge1, float rouge2,
                                           float rougeL) override;

   private:
    std::string session_base_url_;
    int timeout_ms_;
    size_t max_queue_depth_;

    std::deque<PushEvent> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread push_thread_;
    std::atomic<bool> stop_{false};
    bool overflow_warned_{false};  ///< True while a Sample-drop overflow episode is active

    // ── Buffered per-epoch state ──────────────────────────────────────────────
    // Updated by training-loop-thread methods and folded into the end_epoch
    // payload.  No separate lock needed: all writers and the reader (end_epoch)
    // run on the same training thread.
    float buf_gradient_variance_{0.0f};
    float buf_compute_time_ratio_{0.0f};
    float buf_weight_update_ratio_{0.0f};
    float buf_activation_saturation_{-1.0f};
    float buf_attention_entropy_{-1.0f};
    float buf_padding_efficiency_{-1.0f};
    float buf_adaptive_clip_avg_{-1.0f};
    int buf_adaptive_clip_spikes_{0};

    void push_loop();
    void enqueue(PushEvent event);

    /**
     * @brief Attempt a single HTTP POST with up to 3 retries on 5xx / network error.
     *
     * Back-off delays between attempts: 0 ms, 200 ms, 1000 ms.
     * HTTP 4xx errors (including 409) are returned immediately without retry —
     * the caller is responsible for higher-level retry logic.
     *
     * @param endpoint  Relative path appended to session_base_url_, e.g. "/start".
     * @param body      JSON request body.
     * @return HTTP status code, or 0 on persistent connection failure.
     */
    int attempt_post(const std::string& endpoint, const std::string& body) const;
};
