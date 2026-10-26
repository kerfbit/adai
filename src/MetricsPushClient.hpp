#pragma once

// @adai-status: stable
// @adai-version: 1.5.0
// @adai-reviewed: 2026-10-25


#include <atomic>
#include <chrono>
#include <array>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <set>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "IMetricsReporter.hpp"

// ============================================================================
// Metrics session key/label helpers
//
// Shared by every MetricsPushClient caller — originally lived in
// IncrementalTrainer.cpp's anonymous namespace (chatbot path only) until a
// second training loop (run_lejepa_training_pass, IncrementalTrainingTool.cpp)
// needed the identical session-key/label derivation and would otherwise have
// had to duplicate it. Header-only/inline since multiple translation units
// include this file.
// ============================================================================

inline std::string trim_trailing_slashes(std::string value) {
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

inline std::string sanitize_session_key(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch) || ch == '-' || ch == '_') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('-');
        }
    }
    return out;
}

// TD-444: up to 32 characters. The old 8-character cut made ai-machine and ai-machine2 share keys.
inline std::string detect_hostname_fragment() {
    std::string host = "host";
#ifdef _WIN32
    if (const char* env_host = std::getenv("COMPUTERNAME")) {
        host = env_host;
    }
#else
    std::array<char, 256> buffer{};
    if (gethostname(buffer.data(), buffer.size() - 1) == 0) {
        host = buffer.data();
    }
#endif

    host = sanitize_session_key(host);
    if (host.empty()) {
        host = "host";
    }
    if (host.size() > 32) {
        host = host.substr(0, 32);
    }
    return host;
}

inline std::string derive_metrics_session_key(int session_id) {
    const std::string host = detect_hostname_fragment();
    return std::to_string(session_id) + "-" + host;
}

// TD-444: METRICS_SERVER_HEADERS, e.g. "CF-Access-Client-Id: abc; CF-Access-Client-Secret: xyz",
// split on ';' into "Name: value" lines. Entries without a ':' are skipped.
inline std::vector<std::string> parse_metrics_server_headers(const std::string& spec) {
    std::vector<std::string> headers;
    std::string::size_type start = 0;
    while (start <= spec.size()) {
        auto end = spec.find(';', start);
        if (end == std::string::npos)
            end = spec.size();
        std::string item = spec.substr(start, end - start);
        const auto first = item.find_first_not_of(" \t");
        const auto last = item.find_last_not_of(" \t");
        item = first == std::string::npos ? "" : item.substr(first, last - first + 1);
        if (item.find(':') != std::string::npos && item.front() != ':')
            headers.push_back(item);
        start = end + 1;
    }
    return headers;
}

inline std::string build_metrics_session_push_base(const std::string& metrics_server_url,
                                                    const std::string& session_key) {
    const std::string base = trim_trailing_slashes(metrics_server_url);
    return base + "/api/sessions/" + session_key;
}

/**
 * @brief Auto-derive a human-readable session label when none is configured.
 *
 * Format: "#{id}: {stem} ({host}, {date})"
 *   id    — numeric session ID
 *   stem  — filename stem of the model file (e.g. "chatbot_model")
 *   host  — sanitised hostname (first 32 characters)
 *   date  — ISO-8601 date of the training run (YYYY-MM-DD)
 */
inline std::string derive_metrics_session_label(int session_id, const std::string& model_path) {
    // Stem: filename without extension
    std::string stem = std::filesystem::path(model_path).stem().string();
    if (stem.empty())
        stem = "model";

    // Host (first 32 chars, sanitised)
    const std::string host = detect_hostname_fragment();

    // Date: YYYY-MM-DD
    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char date_buf[16];
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &tm_buf);

    std::ostringstream label;
    label << "#" << session_id << ": " << stem << " (" << host << ", " << date_buf << ")";
    return label.str();
}

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
     * @param extra_headers     TD-444: "Name: value" headers sent with every request (e.g. a
     *                          Cloudflare Access service token; see parse_metrics_server_headers()).
     * @param ca_file           TD-444: CA bundle for verifying an https:// server; empty = the
     *                          system store.
     *
     * https:// URLs need the libcurl transport (built when libcurl is found); without it,
     * start_session() refuses them with an error instead of sending plain HTTP to port 443.
     */
    explicit MetricsPushClient(std::string session_base_url, int timeout_ms = 1000,
                               size_t max_queue_depth = 1024, int heartbeat_interval_ms = 30000,
                               std::vector<std::string> extra_headers = {},
                               std::string ca_file = {});
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
     * @param reset_best       When true, tells the server not to carry forward
     *                         best_validation_loss/best_epoch from a prior session
     *                         with the same architecture — for a session that
     *                         starts over from scratch (e.g. a full retrain).
     * @return HTTP status code, or 0 on connection/timeout failure
     */
    int start_session(int session_id, int total_epochs, int total_samples,
                      const std::string& label = "", const std::string& config_snapshot = "",
                      bool reset_best = false);

    /**
     * @brief Enqueue a session-end event and flush the queue.
     *
     * Enqueues a Session-priority POST to /end, sets the stop flag, and blocks
     * until the push thread has processed all queued events and exited.
     * Safe to call from the destructor if the caller forgets to call it
     * explicitly (idempotent after the first call).
     */
    void end_session();
    /// TD-442: the most end_session() (or the destructor) spends sending queued events once it
    /// stops (default 10 s). While draining, Sample events are dropped and nothing is retried; at
    /// the deadline the rest is dropped and /end gets one attempt.
    void set_drain_timeout_ms(int ms) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        drain_timeout_ms_ = ms < 0 ? 0 : ms;
    }

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
    void update_layer_gradient_norms(const std::vector<float>& encoder_layer_norms,
                                     const std::vector<float>& decoder_layer_norms) override;
    void update_padding_efficiency(float efficiency) override;
    void update_generation_quality_metrics(float bleu4, float rouge1, float rouge2, float rougeL,
                                           float bleu1 = -1.0f, float bleu2 = -1.0f) override;
    void update_lejepa_metrics(float predictor_loss, float sigreg_loss) override;
    void update_lejepa_advanced_metrics(float masking_ratio, float predictor_target_cosine_sim,
                                        float sigreg_variance_mean,
                                        float sigreg_variance_stddev) override;

   private:
    std::string session_base_url_;
    int timeout_ms_;
    std::vector<std::string> extra_headers_;
    std::string ca_file_;
    // TD-444: one connection reused by every request (keep-alive), behind its own mutex since
    // start_session() posts from the caller's thread while the push thread runs. Defined in the .cpp.
    struct Transport;
    std::unique_ptr<Transport> transport_;
    // TD-444: rate-limited warnings for rejected or failed requests (push thread only).
    std::chrono::steady_clock::time_point last_failure_warn_{};
    int failures_since_warn_{0};
    void note_failure(const std::string& endpoint, int status);
    // TD-444: the heartbeat is sent only once start_session() succeeded.
    std::atomic<bool> session_started_{false};
    size_t max_queue_depth_;
    int heartbeat_interval_ms_;

    std::deque<PushEvent> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread push_thread_;
    std::atomic<bool> stop_{false};
    bool overflow_warned_{false};  ///< True while a Sample-drop overflow episode is active
    // TD-555 (push thread only): log a lost session once per outage, and a route the server
    // doesn't have once per endpoint, instead of once per event.
    bool session_loss_logged_{false};
    // TD-442: how long end_session()/the destructor may spend sending what's queued once stopping;
    // the deadline is set when stopping begins (guarded by queue_mutex_).
    int drain_timeout_ms_{10000};
    std::chrono::steady_clock::time_point drain_deadline_{};
    std::set<std::string> missing_route_logged_;

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
    float buf_predictor_loss_{-1.0f};  ///< TD-178: LeJEPA predictor loss
    float buf_sigreg_loss_{-1.0f};     ///< TD-178: LeJEPA SIGReg loss
    float buf_masking_ratio_{-1.0f};                  ///< LeJEPA advanced: masking ratio
    float buf_predictor_target_cosine_sim_{-1.0f};    ///< LeJEPA advanced: cosine similarity
    float buf_sigreg_variance_mean_{-1.0f};           ///< LeJEPA advanced: SIGReg variance mean
    float buf_sigreg_variance_stddev_{-1.0f};         ///< LeJEPA advanced: SIGReg variance stddev

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
    /// @p response_body, when non-null, receives the body of the final response (TD-555).
    /// @p retry false (TD-442: while draining) makes a single attempt with no back-off.
    int attempt_post(const std::string& endpoint, const std::string& body,
                     std::string* response_body = nullptr, bool retry = true) const;
    /// TD-442: sets stop_ and the drain deadline; call with queue_mutex_ held.
    void begin_stop_locked();

    // Saved verbatim from the last start_session() call so push_loop can
    // re-register the session automatically after a 404 (server restart /
    // session eviction). TD-444: written by start_session() while the push thread may be
    // running, so guarded by reconnect_mutex_ (the push thread reads a copy).
    std::string reconnect_body_;
    mutable std::mutex reconnect_mutex_;
    std::string reconnect_body() const;
};
