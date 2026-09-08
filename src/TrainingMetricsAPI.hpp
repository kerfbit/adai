#pragma once

// @adai-status: beta        (large, actively evolving)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07


#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include "DaemonConfigStore.hpp"
#include "MetricsSessionRegistry.hpp"
#include "ModelNameClient.hpp"

/**
 * @brief REST API for TrainingMetricsService - Provides HTTP endpoints for polling training metrics
 *
 * This class wraps TrainingMetricsService and exposes its functionality via HTTP endpoints.
 * Follows the same design pattern as ChatbotAPI for consistency.
 *
 * Features:
 * - Thread-safe polling interface
 * - Multiple output formats (JSON, Prometheus, CSV)
 * - Historical metrics access
 * - Session status monitoring
 * - Non-blocking, optimized for frequent polling
 *
 * Endpoints:
 * - TODO(TD-018): add session-scoped routes under /api/sessions/{key}/... and keep
 *   legacy flat routes as compatibility aliases for 0-default.
 * - GET  /api/metrics/current    - Current training snapshot (JSON)
 * - GET  /api/metrics/summary    - Aggregated metrics summary (JSON)
 * - GET  /api/metrics/history    - Historical metrics records (JSON)
 * - GET  /api/metrics/prometheus - Prometheus format metrics (legacy, alias for 0-default)
 * - GET  /api/metrics/prometheus/aggregate - Concatenated Prometheus output for all live sessions
 * (TD-021)
 * - GET  /api/metrics/csv        - CSV format (header + current row)
 * - GET  /api/metrics/abnormal   - TD-013: Outlier samples (JSON)
 * - GET  /api/metrics/generation-quality - BLEU/ROUGE generation quality scores (JSON)
 * - GET  /api/metrics/padding-efficiency  - Batch padding efficiency history (JSON)
 * - GET  /api/session/status     - Session status (active, epoch, progress)
 * - GET  /api/session/epochs     - Per-epoch metrics (losses, validation)
 * - POST /api/session/start      - Start new training session
 * - POST /api/session/end        - End current training session
 * - POST /api/epoch/start        - Start new epoch
 * - POST /api/epoch/end          - End current epoch
 * - POST /api/metrics/sample     - Update sample metrics
 * - POST /api/metrics/validation - Update validation metrics
 * - POST /api/metrics/best       - Update best metrics
 * - POST /api/control/flush      - Force flush metrics to disk
 * - POST /api/control/clear      - Clear historical metrics
 * - GET  /health                 - API health check
 *
 * Usage:
 *   auto registry = std::make_shared<MetricsSessionRegistry>(config);
 *   TrainingMetricsAPI api(registry, 8081);
 *   api.start();  // Blocking - runs server on port 8081
 */
class TrainingMetricsAPI {
   public:
    /**
     * @brief Construct the metrics REST API
     * @param session_registry Shared pointer to the metrics session registry
     * @param port Port number to listen on (default: 8081)
     * @param allow_control Enable control endpoints (flush, clear) - default: true
     * @param name_service_url URL of MNS daemon for /api/models (empty = disabled)
     */
    /**
     * @param admin_config_db_dir Directory for this daemon's daemon_config.db (admin-mutable
     *        settings persisted via PUT /admin/config); empty = admin changes don't persist
     *        across restarts (GET/PUT /admin/config still work in-memory).
     */
    explicit TrainingMetricsAPI(std::shared_ptr<MetricsSessionRegistry> session_registry,
                                int port = 8081, bool allow_control = true,
                                const std::string& name_service_url = "",
                                const std::string& admin_config_db_dir = "");

    /**
     * @brief Destructor - ensures server is stopped
     */
    ~TrainingMetricsAPI();
    TrainingMetricsAPI(const TrainingMetricsAPI&) = delete;
    TrainingMetricsAPI& operator=(const TrainingMetricsAPI&) = delete;
    TrainingMetricsAPI(TrainingMetricsAPI&&) = delete;
    TrainingMetricsAPI& operator=(TrainingMetricsAPI&&) = delete;

    /**
     * @brief Start the HTTP server (blocking)
     * @return true if server started successfully
     */
    bool start();

    /**
     * @brief Stop the HTTP server
     */
    void stop();

    /**
     * @brief Check if server is running
     * @return true if server is running
     */
    bool is_running() const {
        return running_;
    }

    /**
     * @brief Get the port number
     * @return Port number the server is configured to use
     */
    int get_port() const {
        return port_;
    }

   private:
    // HTTP endpoint handlers (return JSON/plain text)
    std::string handle_current_metrics(const std::string& session_key);
    std::string handle_metrics_summary(const std::string& session_key);
    std::string handle_metrics_history(const std::string& session_key,
                                       const std::string& query_params);
    std::string handle_prometheus_metrics(const std::string& session_key);
    std::string handle_csv_metrics(const std::string& session_key);
    std::string handle_session_status(const std::string& session_key);
    std::string handle_epoch_metrics(const std::string& session_key);
    std::string handle_abnormal_samples(const std::string& session_key);  // TD-013: outlier samples
    std::string handle_generation_quality_metrics(const std::string& session_key);  // BLEU/ROUGE
    std::string handle_padding_efficiency_metrics(const std::string& session_key);  // Batch padding
    std::string handle_sessions_list();
    std::string handle_sessions_list_filtered(const std::string& query_params);
    std::string handle_metrics_aggregate();
    std::string handle_db_history(const std::string& session_key, const std::string& query_params);
    std::string handle_metrics_compare(const std::string& query_params);
    std::string handle_metrics_export(const std::string& session_key,
                                      const std::string& query_params);
    std::string handle_prometheus_aggregate();  ///< TD-021: per-session labelled Prometheus output
    std::string handle_models_list();
    std::string handle_flush_control(const std::string& session_key);
    std::string handle_clear_control(const std::string& session_key);
    std::string handle_health_check();
    std::string handle_admin_get_config();
    std::pair<int, std::string> handle_admin_put_config(const std::string& body);

    // POST endpoint handlers for receiving metrics updates
    std::string handle_post_session_start(const std::string& session_key, const std::string& body);
    std::string handle_post_session_end(const std::string& session_key);
    std::string handle_post_heartbeat(const std::string& session_key);
    std::string handle_post_epoch_start(const std::string& session_key, const std::string& body);
    std::string handle_post_epoch_end(const std::string& session_key, const std::string& body);
    std::string handle_post_sample_metrics(const std::string& session_key, const std::string& body);
    std::string handle_post_validation_metrics(const std::string& session_key,
                                               const std::string& body);
    std::string handle_post_best_metrics(const std::string& session_key, const std::string& body);
    std::string handle_post_advanced_metrics(const std::string& session_key,
                                             const std::string& body);
    std::string handle_post_layer_gradient_norms(const std::string& session_key,
                                                 const std::string& body);
    std::string handle_post_generation_quality_metrics(const std::string& session_key,
                                                       const std::string& body);  // TD-016

    // Helper functions
    static std::string create_error_response(const std::string& error_message);
    static std::string escape_json(const std::string& s);
    static int parse_query_param_int(const std::string& query, const std::string& param,
                                     int default_value);
    static std::string parse_query_param_string(const std::string& query, const std::string& param,
                                                const std::string& default_value = "");
    static bool is_valid_session_key(const std::string& key);
    std::shared_ptr<TrainingMetricsService> resolve_session_service(const std::string& session_key,
                                                                    bool create_if_missing) const;

    // Members
    std::shared_ptr<MetricsSessionRegistry> session_registry_;
    int port_;
    bool allow_control_;
    std::string name_service_url_;
    std::atomic<bool> running_;
    std::unique_ptr<adai::DaemonConfigStore> config_store_;

    // HTTP server implementation (forward declaration to avoid including httplib.h in header)
    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};
