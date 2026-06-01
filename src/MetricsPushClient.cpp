#include "MetricsPushClient.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <thread>
#include "Logger.hpp"

// HTTP client (optional — guarded at compile time, same as TrainingMetricsService.cpp)
#ifdef BUILD_METRICS_API_SERVER
#include <httplib.h>
#endif

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

/**
 * @brief Parsed components of an "http://host[:port]/path..." URL.
 *
 * Used to split the session_base_url into the parts expected by httplib::Client
 * (host + port) and httplib::Client::Post (path).
 */
struct ParsedUrl {
    std::string host{"localhost"};
    int port{8081};
    std::string base_path;  ///< Path component, stripped of trailing slashes

    static ParsedUrl from(const std::string& url) {
        ParsedUrl result;
        std::string s = url;

        // Strip scheme
        const std::string http_prefix = "http://";
        const std::string https_prefix = "https://";
        if (s.rfind(http_prefix, 0) == 0) {
            s = s.substr(http_prefix.size());
        } else if (s.rfind(https_prefix, 0) == 0) {
            s = s.substr(https_prefix.size());
            result.port = 443;
        }

        // Split authority from path
        const auto slash_pos = s.find('/');
        std::string authority;
        if (slash_pos != std::string::npos) {
            authority = s.substr(0, slash_pos);
            result.base_path = s.substr(slash_pos);
        } else {
            authority = s;
        }

        // Strip trailing slashes from base_path
        while (!result.base_path.empty() && result.base_path.back() == '/') {
            result.base_path.pop_back();
        }

        // Split host:port
        const auto colon_pos = authority.find(':');
        if (colon_pos != std::string::npos) {
            result.host = authority.substr(0, colon_pos);
            try {
                result.port = std::stoi(authority.substr(colon_pos + 1));
            } catch (...) {
                // Keep default port
            }
        } else {
            result.host = authority;
        }

        return result;
    }
};

/**
 * @brief Minimal JSON string escaping: handles the characters that must be
 *        escaped inside a JSON string literal.
 */
std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

}  // namespace

// ============================================================================
// MetricsPushClient — Construction / Destruction
// ============================================================================

MetricsPushClient::MetricsPushClient(std::string session_base_url, int timeout_ms,
                                     size_t max_queue_depth)
    : session_base_url_(std::move(session_base_url)),
      timeout_ms_(timeout_ms),
      max_queue_depth_(max_queue_depth) {
    push_thread_ = std::thread(&MetricsPushClient::push_loop, this);
}

MetricsPushClient::~MetricsPushClient() {
    // Guard against the case where end_session() was not called (e.g. after
    // an exception that unwinds the stack past the IncrementalTrainer).
    if (push_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        queue_cv_.notify_one();
        push_thread_.join();
    }
}

// ============================================================================
// Session lifecycle
// ============================================================================

int MetricsPushClient::start_session(int session_id, int total_epochs, int total_samples,
                                     const std::string& label,
                                     const std::string& config_snapshot) {
    std::ostringstream json;
    json << "{\"session_id\":" << session_id
         << ",\"total_epochs\":" << total_epochs
         << ",\"total_samples\":" << total_samples;
    if (!label.empty()) {
        json << ",\"label\":\"" << escape_json_string(label) << "\"";
    }
    if (!config_snapshot.empty()) {
        // config_snapshot is already a JSON object string — embed verbatim
        json << ",\"config\":" << config_snapshot;
    }
    json << "}";

    const int rc = attempt_post("/start", json.str());
    if (rc == 0) {
        adai::Logger::warn("MetricsPushClient: start_session — connection failed");
    } else {
        adai::Logger::info("MetricsPushClient: start_session HTTP {}", rc);
    }
    return rc;
}

void MetricsPushClient::end_session() {
    enqueue({EventPriority::Session, "/end", "{}"});
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_one();
    if (push_thread_.joinable()) {
        push_thread_.join();
    }
}

// ============================================================================
// IMetricsReporter — Epoch lifecycle
// ============================================================================

void MetricsPushClient::start_epoch(int epoch, int total_samples) {
    // Reset buffered per-epoch state for the epoch that is about to begin.
    buf_gradient_variance_ = 0.0f;
    buf_compute_time_ratio_ = 0.0f;
    buf_weight_update_ratio_ = 0.0f;
    buf_activation_saturation_ = -1.0f;
    buf_attention_entropy_ = -1.0f;
    buf_padding_efficiency_ = -1.0f;
    buf_adaptive_clip_avg_ = -1.0f;
    buf_adaptive_clip_spikes_ = 0;

    std::ostringstream json;
    json << "{\"epoch\":" << epoch << ",\"total_samples\":" << total_samples << "}";
    enqueue({EventPriority::Epoch, "/epoch/start", json.str()});
}

void MetricsPushClient::end_epoch(int epoch, float loss, float validation_loss,
                                  float learning_rate, float perplexity, float gradient_norm,
                                  double epoch_time_seconds) {
    const float stored_perplexity = (perplexity > 0.0f) ? perplexity : std::exp(loss);

    std::ostringstream json;
    json << std::fixed;
    json << "{\"epoch\":" << epoch
         << ",\"loss\":" << loss
         << ",\"validation_loss\":" << validation_loss
         << ",\"learning_rate\":" << learning_rate
         << ",\"perplexity\":" << stored_perplexity
         << ",\"gradient_norm\":" << gradient_norm
         << ",\"epoch_time\":" << epoch_time_seconds
         << ",\"gradient_variance\":" << buf_gradient_variance_
         << ",\"compute_time_ratio\":" << buf_compute_time_ratio_
         << ",\"weight_update_ratio\":" << buf_weight_update_ratio_
         << ",\"activation_saturation_ratio\":" << buf_activation_saturation_
         << ",\"attention_entropy\":" << buf_attention_entropy_
         << ",\"current_padding_efficiency\":" << buf_padding_efficiency_
         << "}";
    enqueue({EventPriority::Epoch, "/epoch/end", json.str()});
}

// ============================================================================
// IMetricsReporter — Per-sample metrics
// ============================================================================

void MetricsPushClient::update_sample_metrics(int sample, float loss, float gradient_norm,
                                              float learning_rate) {
    std::ostringstream json;
    json << "{\"sample\":" << sample
         << ",\"loss\":" << loss
         << ",\"gradient_norm\":" << gradient_norm
         << ",\"learning_rate\":" << learning_rate
         << "}";
    enqueue({EventPriority::Sample, "/metrics/sample", json.str()});
}

// ============================================================================
// IMetricsReporter — Validation metrics
// ============================================================================

void MetricsPushClient::update_validation_metrics(float validation_loss, float validation_accuracy,
                                                  float validation_perplexity) {
    std::ostringstream json;
    json << "{\"validation_loss\":" << validation_loss
         << ",\"validation_accuracy\":" << validation_accuracy
         << ",\"validation_perplexity\":" << validation_perplexity
         << "}";
    enqueue({EventPriority::Epoch, "/metrics/validation", json.str()});
}

void MetricsPushClient::update_best_metrics(float validation_loss, int epoch) {
    std::ostringstream json;
    json << "{\"validation_loss\":" << validation_loss << ",\"epoch\":" << epoch << "}";
    enqueue({EventPriority::Epoch, "/metrics/best", json.str()});
}

// ============================================================================
// IMetricsReporter — Advanced epoch diagnostics (TD-013)
// ============================================================================

void MetricsPushClient::update_advanced_epoch_metrics(float gradient_variance,
                                                      float compute_time_ratio,
                                                      float weight_update_ratio) {
    // Update buffered state so the epoch-final values reach end_epoch().
    buf_gradient_variance_ = gradient_variance;
    buf_compute_time_ratio_ = compute_time_ratio;
    buf_weight_update_ratio_ = weight_update_ratio;

    // Also push immediately for per-step dashboard updates (Sample priority —
    // these events are dropped under queue backpressure).
    std::ostringstream json;
    json << "{\"gradient_variance\":" << gradient_variance
         << ",\"compute_time_ratio\":" << compute_time_ratio
         << ",\"weight_update_ratio\":" << weight_update_ratio
         << "}";
    enqueue({EventPriority::Sample, "/metrics/advanced", json.str()});
}

void MetricsPushClient::flag_abnormal_sample(const AbnormalSample& sample) {
    std::ostringstream json;
    json << "{\"epoch\":" << sample.epoch
         << ",\"sample_id\":" << sample.sample_id
         << ",\"loss\":" << sample.loss
         << ",\"grad_norm\":" << sample.grad_norm
         << ",\"reason\":\"" << escape_json_string(sample.reason) << "\""
         << ",\"input_text\":\"" << escape_json_string(sample.input_text) << "\""
         << ",\"target_text\":\"" << escape_json_string(sample.target_text) << "\""
         << "}";
    enqueue({EventPriority::Epoch, "/metrics/abnormal", json.str()});
}

// ============================================================================
// IMetricsReporter — Adaptive gradient clipping (TD-017)
// ============================================================================

void MetricsPushClient::update_adaptive_clip_metrics(float effective_clip_threshold,
                                                     int cumulative_spike_count) {
    // State-only: buffer for inclusion in end_epoch payload.
    buf_adaptive_clip_avg_ = effective_clip_threshold;
    buf_adaptive_clip_spikes_ = cumulative_spike_count;
}

void MetricsPushClient::update_adaptive_clip_epoch(float avg_clip_threshold,
                                                   int total_spike_count) {
    buf_adaptive_clip_avg_ = avg_clip_threshold;
    buf_adaptive_clip_spikes_ = total_spike_count;
}

// ============================================================================
// IMetricsReporter — Activation / attention diagnostics (TD-013)
// ============================================================================

void MetricsPushClient::update_activation_saturation(float ratio) {
    buf_activation_saturation_ = ratio;
}

void MetricsPushClient::update_attention_entropy(float entropy) {
    buf_attention_entropy_ = entropy;
}

void MetricsPushClient::update_padding_efficiency(float efficiency) {
    buf_padding_efficiency_ = efficiency;
}

// ============================================================================
// IMetricsReporter — Generation quality metrics (TD-016)
// ============================================================================

void MetricsPushClient::update_generation_quality_metrics(float bleu4, float rouge1, float rouge2,
                                                          float rougeL) {
    std::ostringstream json;
    json << std::fixed;
    json << "{\"bleu4\":" << bleu4
         << ",\"rouge1\":" << rouge1
         << ",\"rouge2\":" << rouge2
         << ",\"rougeL\":" << rougeL
         << "}";
    enqueue({EventPriority::Epoch, "/metrics/generation-quality", json.str()});
}

// ============================================================================
// Queue management
// ============================================================================

void MetricsPushClient::enqueue(PushEvent event) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (queue_.size() >= max_queue_depth_) {
        if (event.priority == EventPriority::Sample) {
            // Drop incoming Sample event under backpressure; warn once per episode.
            if (!overflow_warned_) {
                adai::Logger::warn(
                    "MetricsPushClient: push queue full ({} events) — "
                    "dropping per-sample metrics until queue drains",
                    max_queue_depth_);
                overflow_warned_ = true;
            }
            return;
        }

        // Epoch / Session: evict the oldest Sample-priority entry to make room.
        // If no Sample entry exists the queue grows beyond max_queue_depth_ —
        // Epoch and Session events are never dropped.
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            if (it->priority == EventPriority::Sample) {
                queue_.erase(it);
                break;
            }
        }
    }

    queue_.push_back(std::move(event));
    queue_cv_.notify_one();

    // Clear overflow warning once the queue is below capacity again.
    if (queue_.size() < max_queue_depth_) {
        overflow_warned_ = false;
    }
}

// ============================================================================
// Background push thread
// ============================================================================

void MetricsPushClient::push_loop() {
    while (true) {
        PushEvent event;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !queue_.empty() || stop_.load(); });

            if (queue_.empty()) {
                // stop_ is set and the queue has been fully drained — exit cleanly.
                return;
            }

            event = std::move(queue_.front());
            queue_.pop_front();
        }

        attempt_post(event.endpoint, event.body);
    }
}

// ============================================================================
// HTTP — conditional on BUILD_METRICS_API_SERVER
// ============================================================================

#ifdef BUILD_METRICS_API_SERVER

int MetricsPushClient::attempt_post(const std::string& endpoint,
                                    const std::string& body) const {
    if (session_base_url_.empty()) {
        return 0;
    }

    const std::string full_url = session_base_url_ + endpoint;
    const ParsedUrl parsed = ParsedUrl::from(session_base_url_);
    const std::string path = parsed.base_path + endpoint;

    static constexpr int kBackoffMs[] = {0, 200, 1000};
    static constexpr int kMaxAttempts = 3;

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (attempt > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
        }

        try {
            httplib::Client client(parsed.host, parsed.port);
            client.set_connection_timeout(0, static_cast<long>(timeout_ms_) * 1000);
            client.set_read_timeout(timeout_ms_ / 1000,
                                    static_cast<long>(timeout_ms_ % 1000) * 1000);
            client.set_write_timeout(timeout_ms_ / 1000,
                                     static_cast<long>(timeout_ms_ % 1000) * 1000);

            auto res = client.Post(path, body, "application/json");

            if (!res) {
                adai::Logger::debug(
                    "MetricsPushClient: POST {} — no response (attempt {}/{})",
                    full_url, attempt + 1, kMaxAttempts);
                continue;  // Network error → retry
            }

            const int status = res->status;
            if (status >= 500) {
                adai::Logger::debug(
                    "MetricsPushClient: POST {} returned HTTP {} (attempt {}/{})",
                    full_url, status, attempt + 1, kMaxAttempts);
                continue;  // Server error → retry
            }

            // 2xx or 4xx — return immediately (callers handle 409 at their level)
            return status;

        } catch (const std::exception& e) {
            adai::Logger::debug(
                "MetricsPushClient: POST {} exception: {} (attempt {}/{})",
                full_url, e.what(), attempt + 1, kMaxAttempts);
        }
    }

    return 0;  // All attempts failed
}

#else  // BUILD_METRICS_API_SERVER not defined

// NOLINTBEGIN(readability-convert-member-functions-to-static)
int MetricsPushClient::attempt_post(const std::string& /*endpoint*/,
                                    const std::string& /*body*/) const {
    return 0;  // No-op when httplib is not available
}
// NOLINTEND(readability-convert-member-functions-to-static)

#endif  // BUILD_METRICS_API_SERVER
