#pragma once

#include "TrainingMetricsService.hpp"
#include <memory>
#include <atomic>
#include <string>

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
 * - GET  /api/metrics/current    - Current training snapshot (JSON)
 * - GET  /api/metrics/summary    - Aggregated metrics summary (JSON)
 * - GET  /api/metrics/history    - Historical metrics records (JSON)
 * - GET  /api/metrics/prometheus - Prometheus format metrics
 * - GET  /api/metrics/csv        - CSV format (header + current row)
 * - GET  /api/metrics/abnormal   - TD-013: Outlier samples (JSON)
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
 *   auto metrics_service = std::make_shared<TrainingMetricsService>(config);
 *   TrainingMetricsAPI api(metrics_service, 8081);
 *   api.start();  // Blocking - runs server on port 8081
 */
class TrainingMetricsAPI {
public:
    /**
     * @brief Construct the metrics REST API
     * @param metrics_service Shared pointer to the metrics service to expose
     * @param port Port number to listen on (default: 8081)
     * @param allow_control Enable control endpoints (flush, clear) - default: true
     */
    explicit TrainingMetricsAPI(std::shared_ptr<TrainingMetricsService> metrics_service,
                                int port = 8081,
                                bool allow_control = true);
    
    /**
     * @brief Destructor - ensures server is stopped
     */
    ~TrainingMetricsAPI();
    
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
    bool is_running() const { return running_; }
    
    /**
     * @brief Get the port number
     * @return Port number the server is configured to use
     */
    int get_port() const { return port_; }

private:
    // HTTP endpoint handlers (return JSON/plain text)
    std::string handle_current_metrics();
    std::string handle_metrics_summary();
    std::string handle_metrics_history(const std::string& query_params);
    std::string handle_prometheus_metrics();
    std::string handle_csv_metrics();
    std::string handle_session_status();
    std::string handle_epoch_metrics();
    std::string handle_abnormal_samples();  // TD-013: outlier samples
    std::string handle_flush_control();
    std::string handle_clear_control();
    std::string handle_health_check();
    
    // POST endpoint handlers for receiving metrics updates
    std::string handle_post_session_start(const std::string& body);
    std::string handle_post_session_end();
    std::string handle_post_epoch_start(const std::string& body);
    std::string handle_post_epoch_end(const std::string& body);
    std::string handle_post_sample_metrics(const std::string& body);
    std::string handle_post_validation_metrics(const std::string& body);
    std::string handle_post_best_metrics(const std::string& body);
    
    // Helper functions
    std::string create_error_response(const std::string& error_message) const;
    std::string escape_json(const std::string& s) const;
    int parse_query_param_int(const std::string& query, const std::string& param, int default_value) const;
    
    // Members
    std::shared_ptr<TrainingMetricsService> metrics_service_;
    int port_;
    bool allow_control_;
    std::atomic<bool> running_;
    
    // HTTP server implementation (forward declaration to avoid including httplib.h in header)
    class ServerImpl;
    std::unique_ptr<ServerImpl> server_impl_;
};
