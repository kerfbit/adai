#include "TrainingMetricsAPI.hpp"
#include <httplib.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

// ServerImpl using cpp-httplib
class TrainingMetricsAPI::ServerImpl {
public:
    httplib::Server server;
};

TrainingMetricsAPI::TrainingMetricsAPI(std::shared_ptr<TrainingMetricsService> metrics_service,
                                       int port,
                                       bool allow_control)
    : metrics_service_(metrics_service),
      port_(port),
      allow_control_(allow_control),
      running_(false),
      server_impl_(std::make_unique<ServerImpl>()) {
    
    // Set up HTTP endpoints
    
    // GET /api/metrics/current - Current snapshot
    server_impl_->server.Get("/api/metrics/current", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_current_metrics();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // GET /api/metrics/summary - Summary
    server_impl_->server.Get("/api/metrics/summary", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_metrics_summary();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // GET /api/metrics/history - Historical records
    server_impl_->server.Get("/api/metrics/history", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            // Build query string from params
            std::string query_params;
            if (req.has_param("max_records")) {
                query_params = "max_records=" + req.get_param_value("max_records");
            }
            if (req.has_param("session_id")) {
                if (!query_params.empty()) query_params += "&";
                query_params += "session_id=" + req.get_param_value("session_id");
            }
            
            std::string response = handle_metrics_history(query_params);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // GET /api/metrics/prometheus - Prometheus format
    server_impl_->server.Get("/api/metrics/prometheus", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_prometheus_metrics();
            res.set_content(response, "text/plain");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // GET /api/metrics/csv - CSV format
    server_impl_->server.Get("/api/metrics/csv", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_csv_metrics();
            res.set_content(response, "text/csv");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // GET /api/session/status - Session status
    server_impl_->server.Get("/api/session/status", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_session_status();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // GET /api/session/epochs - Epoch metrics
    server_impl_->server.Get("/api/session/epochs", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_epoch_metrics();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    // GET /api/metrics/abnormal - Outlier samples (TD-013)
    server_impl_->server.Get("/api/metrics/abnormal", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_abnormal_samples();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    // GET /api/metrics/generation-quality - BLEU/ROUGE generation quality scores
    server_impl_->server.Get("/api/metrics/generation-quality", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_generation_quality_metrics();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    // GET /api/metrics/padding-efficiency - Batch padding efficiency history
    server_impl_->server.Get("/api/metrics/padding-efficiency", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_padding_efficiency_metrics();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });
    
    // POST endpoints for receiving metrics updates from trainers
    // POST /api/session/start - Start new training session
    server_impl_->server.Post("/api/session/start", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_session_start(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // POST /api/session/end - End current training session
    server_impl_->server.Post("/api/session/end", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_post_session_end();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // POST /api/epoch/start - Start new epoch
    server_impl_->server.Post("/api/epoch/start", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_epoch_start(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // POST /api/epoch/end - End current epoch
    server_impl_->server.Post("/api/epoch/end", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_epoch_end(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // POST /api/metrics/sample - Update sample metrics
    server_impl_->server.Post("/api/metrics/sample", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_sample_metrics(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // POST /api/metrics/validation - Update validation metrics
    server_impl_->server.Post("/api/metrics/validation", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_validation_metrics(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // POST /api/metrics/best - Update best metrics
    server_impl_->server.Post("/api/metrics/best", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_best_metrics(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });

    // POST /api/metrics/advanced - Update advanced diagnostic metrics (TD-013)
    server_impl_->server.Post("/api/metrics/advanced", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_advanced_metrics(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });

    // POST /api/metrics/generation-quality - Update BLEU/ROUGE scores (TD-016)
    server_impl_->server.Post("/api/metrics/generation-quality", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response = handle_post_generation_quality_metrics(req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });
    
    // Control endpoints (if enabled)
    if (allow_control_) {
        // POST /api/control/flush - Flush to disk
        server_impl_->server.Post("/api/control/flush", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_flush_control();
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });
        
        // POST /api/control/clear - Clear history
        server_impl_->server.Post("/api/control/clear", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_clear_control();
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });
    }
    
    // GET /health - Health check
    server_impl_->server.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        std::string response = handle_health_check();
        res.set_content(response, "application/json");
        res.status = 200;
    });
    
    // Add CORS headers to all responses
    server_impl_->server.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
    
    // Handle OPTIONS requests for CORS preflight
    server_impl_->server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });
}

TrainingMetricsAPI::~TrainingMetricsAPI() {
    if (running_) {
        stop();
    }
}

bool TrainingMetricsAPI::start() {
    if (running_) {
        return false;
    }
    
    running_ = true;
    
    // This is blocking - will run until stop() is called
    bool success = server_impl_->server.listen("0.0.0.0", port_);
    
    running_ = false;
    return success;
}

void TrainingMetricsAPI::stop() {
    if (!running_) {
        return;
    }
    
    server_impl_->server.stop();
    running_ = false;
}

// ============================================================================
// HTTP Endpoint Handlers
// ============================================================================

std::string TrainingMetricsAPI::handle_current_metrics() {
    return metrics_service_->to_json();
}

std::string TrainingMetricsAPI::handle_metrics_summary() {
    return metrics_service_->to_json_summary();
}

std::string TrainingMetricsAPI::handle_metrics_history(const std::string& query_params) {
    // Parse query parameters
    int max_records = parse_query_param_int(query_params, "max_records", 1000);
    int session_id = parse_query_param_int(query_params, "session_id", -1);
    
    // Get history
    std::vector<PersistentMetricsRecord> history;
    if (session_id >= 0) {
        history = metrics_service_->get_session_history(session_id);
    } else {
        history = metrics_service_->get_history(max_records);
    }
    
    // Convert to JSON array
    std::ostringstream json;
    json << "{\"records\":[";
    
    for (size_t i = 0; i < history.size(); ++i) {
        const auto& record = history[i];
        
        if (i > 0) json << ",";
        json << "{";
        json << "\"timestamp\":\"" << std::chrono::system_clock::to_time_t(record.timestamp) << "\",";
        json << "\"session_id\":" << record.session_id << ",";
        json << "\"epoch\":" << record.epoch << ",";
        json << "\"sample\":" << record.sample << ",";
        json << "\"loss\":" << record.loss << ",";
        json << "\"validation_loss\":" << record.validation_loss << ",";
        json << "\"learning_rate\":" << record.learning_rate << ",";
        json << "\"gradient_norm\":" << record.gradient_norm << ",";
        json << "\"perplexity\":" << record.perplexity;
        json << "}";
    }
    
    json << "],\"count\":" << history.size() << "}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_prometheus_metrics() {
    return metrics_service_->to_prometheus();
}

std::string TrainingMetricsAPI::handle_csv_metrics() {
    std::ostringstream csv;
    csv << metrics_service_->to_csv_header() << "\n";
    csv << metrics_service_->to_csv_row();
    return csv.str();
}

std::string TrainingMetricsAPI::handle_session_status() {
    auto snapshot = metrics_service_->get_current_snapshot();
    
    std::ostringstream json;
    json << "{";
    json << "\"is_training\":" << (snapshot.is_training ? "true" : "false") << ",";
    json << "\"session_id\":" << snapshot.session_id << ",";
    json << "\"current_epoch\":" << snapshot.current_epoch << ",";
    json << "\"total_epochs\":" << snapshot.total_epochs << ",";
    json << "\"current_sample\":" << snapshot.current_sample << ",";
    json << "\"total_samples\":" << snapshot.total_samples << ",";
    json << "\"progress_percent\":" << std::fixed << std::setprecision(2);
    
    if (snapshot.total_samples > 0) {
        json << (100.0f * snapshot.current_sample / snapshot.total_samples);
    } else {
        json << 0.0f;
    }
    
    json << ",";
    json << "\"samples_per_second\":" << snapshot.samples_per_second << ",";
    json << "\"estimated_time_remaining_seconds\":" << snapshot.estimated_time_remaining_seconds;
    json << "}";
    
    return json.str();
}

std::string TrainingMetricsAPI::handle_epoch_metrics() {
    auto snapshot = metrics_service_->get_current_snapshot();
    
    std::ostringstream json;
    json << "{";
    json << "\"current_epoch\":" << snapshot.current_epoch << ",";
    json << "\"total_epochs\":" << snapshot.total_epochs << ",";
    json << "\"epoch_losses\":[";
    
    for (size_t i = 0; i < snapshot.epoch_losses.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_losses[i];
    }
    
    json << "],\"epoch_validation_losses\":[";
    
    for (size_t i = 0; i < snapshot.epoch_validation_losses.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_validation_losses[i];
    }
    
    json << "],\"epoch_learning_rates\":[";
    
    for (size_t i = 0; i < snapshot.epoch_learning_rates.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_learning_rates[i];
    }
    
    json << "],\"epoch_perplexities\":[";
    
    for (size_t i = 0; i < snapshot.epoch_perplexities.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_perplexities[i];
    }
    
    json << "],\"epoch_durations\":[";
    
    for (size_t i = 0; i < snapshot.epoch_durations.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_durations[i];
    }
    
    json << "],\"epoch_gradient_norms\":[";
    
    for (size_t i = 0; i < snapshot.epoch_gradient_norms.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_gradient_norms[i];
    }
    
    // TD-015: per-epoch validation perplexity and accuracy
    json << "],\"epoch_validation_perplexities\":[";
    for (size_t i = 0; i < snapshot.epoch_validation_perplexities.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_validation_perplexities[i];
    }

    json << "],\"epoch_validation_accuracies\":[";
    for (size_t i = 0; i < snapshot.epoch_validation_accuracies.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_validation_accuracies[i];
    }

    json << "],\"best_validation_loss\":" << snapshot.best_validation_loss << ",";
    json << "\"best_epoch\":" << snapshot.best_epoch;
    json << "}";
    
    return json.str();
}

std::string TrainingMetricsAPI::handle_abnormal_samples() {
    auto samples = metrics_service_->get_abnormal_samples();

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\"abnormal_samples\":[";

    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        if (i > 0) json << ",";
        json << "{";
        json << "\"epoch\":"      << s.epoch      << ",";
        json << "\"sample_id\":"  << s.sample_id  << ",";
        json << "\"loss\":"       << s.loss        << ",";
        json << "\"grad_norm\":"  << s.grad_norm   << ",";
        json << "\"reason\":\""   << escape_json(s.reason)      << "\",";
        json << "\"input_text\":\""  << escape_json(s.input_text)  << "\",";
        json << "\"target_text\":\"" << escape_json(s.target_text) << "\",";
        json << "\"timestamp\":" << std::chrono::system_clock::to_time_t(s.timestamp);
        json << "}";
    }

    json << "],\"count\":" << samples.size() << "}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_generation_quality_metrics() {
    auto snapshot = metrics_service_->get_current_snapshot();
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{";
    json << "\"current_bleu4\":"  << snapshot.current_bleu4  << ",";
    json << "\"current_rouge1\":" << snapshot.current_rouge1 << ",";
    json << "\"current_rouge2\":" << snapshot.current_rouge2 << ",";
    json << "\"current_rougeL\":" << snapshot.current_rougeL << ",";

    // Per-epoch history arrays
    json << "\"epoch_bleu4\":[";
    for (size_t i = 0; i < snapshot.epoch_bleu4.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_bleu4[i];
    }
    json << "],\"epoch_rouge1\":[";
    for (size_t i = 0; i < snapshot.epoch_rouge1.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_rouge1[i];
    }
    json << "],\"epoch_rouge2\":[";
    for (size_t i = 0; i < snapshot.epoch_rouge2.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_rouge2[i];
    }
    json << "],\"epoch_rougeL\":[";
    for (size_t i = 0; i < snapshot.epoch_rougeL.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_rougeL[i];
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_padding_efficiency_metrics() {
    auto snapshot = metrics_service_->get_current_snapshot();
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{";
    json << "\"current_padding_efficiency\":" << snapshot.current_padding_efficiency << ",";
    json << "\"epoch_padding_efficiencies\":[";
    for (size_t i = 0; i < snapshot.epoch_padding_efficiencies.size(); ++i) {
        if (i > 0) json << ",";
        json << snapshot.epoch_padding_efficiencies[i];
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_flush_control() {
    metrics_service_->flush_to_disk();
    return "{\"status\":\"success\",\"message\":\"Metrics flushed to disk\"}";
}

std::string TrainingMetricsAPI::handle_clear_control() {
    metrics_service_->clear_history();
    return "{\"status\":\"success\",\"message\":\"Metrics history cleared\"}";
}

std::string TrainingMetricsAPI::handle_health_check() {
    bool is_active = metrics_service_->is_session_active();
    
    std::ostringstream json;
    json << "{";
    json << "\"status\":\"ok\",";
    json << "\"service\":\"TrainingMetricsAPI\",";
    json << "\"is_training\":" << (is_active ? "true" : "false");
    json << "}";
    
    return json.str();
}

// ============================================================================
// Helper Functions
// ============================================================================

std::string TrainingMetricsAPI::create_error_response(const std::string& error_message) const {
    std::ostringstream json;
    json << "{";
    json << "\"error\":\"" << escape_json(error_message) << "\"";
    json << "}";
    return json.str();
}

std::string TrainingMetricsAPI::escape_json(const std::string& s) const {
    std::ostringstream escaped;
    for (char c : s) {
        switch (c) {
            case '"':  escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b";  break;
            case '\f': escaped << "\\f";  break;
            case '\n': escaped << "\\n";  break;
            case '\r': escaped << "\\r";  break;
            case '\t': escaped << "\\t";  break;
            default:
                if (c < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    escaped << c;
                }
        }
    }
    return escaped.str();
}

int TrainingMetricsAPI::parse_query_param_int(const std::string& query, const std::string& param, int default_value) const {
    std::string search_key = param + "=";
    size_t pos = query.find(search_key);
    
    if (pos == std::string::npos) {
        return default_value;
    }
    
    pos += search_key.length();
    size_t end_pos = query.find('&', pos);
    
    std::string value_str;
    if (end_pos == std::string::npos) {
        value_str = query.substr(pos);
    } else {
        value_str = query.substr(pos, end_pos - pos);
    }
    
    try {
        return std::stoi(value_str);
    } catch (...) {
        return default_value;
    }
}

// POST handler implementations for receiving metrics updates

std::string TrainingMetricsAPI::handle_post_session_start(const std::string& body) {
    // Parse JSON body: {"session_id": int, "total_epochs": int, "total_samples": int}
    int session_id = 0;
    int total_epochs = 0;
    int total_samples = 0;
    
    // Simple JSON parsing (extract integer values)
    size_t pos = body.find("\"session_id\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            session_id = std::stoi(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"total_epochs\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            total_epochs = std::stoi(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"total_samples\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            total_samples = std::stoi(body.substr(pos + 1));
        }
    }
    
    metrics_service_->start_session(session_id, total_epochs, total_samples);
    
    return "{\"status\":\"ok\",\"message\":\"Session started\"}";
}

std::string TrainingMetricsAPI::handle_post_session_end() {
    metrics_service_->end_session();
    return "{\"status\":\"ok\",\"message\":\"Session ended\"}";
}

std::string TrainingMetricsAPI::handle_post_epoch_start(const std::string& body) {
    // Parse JSON body: {"epoch": int, "total_samples": int}
    int epoch = 0;
    int total_samples = 0;
    
    size_t pos = body.find("\"epoch\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            epoch = std::stoi(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"total_samples\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            total_samples = std::stoi(body.substr(pos + 1));
        }
    }
    
    metrics_service_->start_epoch(epoch, total_samples);
    
    return "{\"status\":\"ok\",\"message\":\"Epoch started\"}";
}

std::string TrainingMetricsAPI::handle_post_epoch_end(const std::string& body) {
    // Parse JSON body: {"epoch": int, "loss": float, "validation_loss": float, "learning_rate": float, "perplexity": float, "gradient_norm": float}
    int epoch = 0;
    float loss = 0.0f;
    float validation_loss = 0.0f;
    float learning_rate = 0.0f;
    float perplexity = 0.0f;
    float gradient_norm = 0.0f;
    
    size_t pos = body.find("\"epoch\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            epoch = std::stoi(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"loss\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            loss = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"validation_loss\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            validation_loss = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"learning_rate\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            learning_rate = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"perplexity\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            perplexity = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"gradient_norm\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            gradient_norm = std::stof(body.substr(pos + 1));
        }
    }

    float gradient_variance = 0.0f;
    float compute_time_ratio = 0.0f;
    float weight_update_ratio = 0.0f;

    pos = body.find("\"gradient_variance\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            gradient_variance = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"compute_time_ratio\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            compute_time_ratio = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"weight_update_ratio\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            weight_update_ratio = std::stof(body.substr(pos + 1));
        }
    }

    float activation_saturation_ratio = -1.0f;
    float attention_entropy = -1.0f;
    float current_padding_efficiency = -1.0f;
    double epoch_time = 0.0;

    pos = body.find("\"epoch_time\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            epoch_time = std::stod(body.substr(pos + 1));
        }
    }

    pos = body.find("\"activation_saturation_ratio\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            activation_saturation_ratio = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"attention_entropy\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            attention_entropy = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"current_padding_efficiency\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            current_padding_efficiency = std::stof(body.substr(pos + 1));
        }
    }

    metrics_service_->end_epoch(epoch, loss, validation_loss, learning_rate, perplexity, gradient_norm, epoch_time);
    if (gradient_variance != 0.0f || compute_time_ratio != 0.0f || weight_update_ratio != 0.0f) {
        metrics_service_->update_advanced_epoch_metrics(gradient_variance, compute_time_ratio, weight_update_ratio);
    }
    if (activation_saturation_ratio >= 0.0f) {
        metrics_service_->update_activation_saturation(activation_saturation_ratio);
    }
    if (attention_entropy >= 0.0f) {
        metrics_service_->update_attention_entropy(attention_entropy);
    }
    if (current_padding_efficiency >= 0.0f) {
        metrics_service_->update_padding_efficiency(current_padding_efficiency);
    }

    return "{\"status\":\"ok\",\"message\":\"Epoch ended\"}";
}

std::string TrainingMetricsAPI::handle_post_sample_metrics(const std::string& body) {
    // Parse JSON body: {"sample": int, "loss": float, "gradient_norm": float, "learning_rate": float}
    int sample = 0;
    float loss = 0.0f;
    float gradient_norm = 0.0f;
    float learning_rate = 0.0f;
    
    size_t pos = body.find("\"sample\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            sample = std::stoi(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"loss\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            loss = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"gradient_norm\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            gradient_norm = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"learning_rate\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            learning_rate = std::stof(body.substr(pos + 1));
        }
    }
    
    metrics_service_->update_sample_metrics(sample, loss, gradient_norm, learning_rate);
    
    return "{\"status\":\"ok\"}";
}

std::string TrainingMetricsAPI::handle_post_validation_metrics(const std::string& body) {
    // Parse JSON body: {"validation_loss": float, "validation_accuracy": float, "validation_perplexity": float}
    float validation_loss = 0.0f;
    float validation_accuracy = -1.0f;  // TD-015: optional
    float validation_perplexity = 0.0f; // TD-015: optional (0 = auto-derive from loss)
    
    size_t pos = body.find("\"validation_loss\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            validation_loss = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"validation_accuracy\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            validation_accuracy = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"validation_perplexity\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            validation_perplexity = std::stof(body.substr(pos + 1));
        }
    }
    
    metrics_service_->update_validation_metrics(validation_loss, validation_accuracy, validation_perplexity);
    
    return "{\"status\":\"ok\"}";
}

std::string TrainingMetricsAPI::handle_post_best_metrics(const std::string& body) {
    // Parse JSON body: {"validation_loss": float, "epoch": int}
    float validation_loss = 0.0f;
    int epoch = 0;
    
    size_t pos = body.find("\"validation_loss\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            validation_loss = std::stof(body.substr(pos + 1));
        }
    }
    
    pos = body.find("\"epoch\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            epoch = std::stoi(body.substr(pos + 1));
        }
    }
    
    metrics_service_->update_best_metrics(validation_loss, epoch);
    
    return "{\"status\":\"ok\"}";
}

std::string TrainingMetricsAPI::handle_post_generation_quality_metrics(const std::string& body) {
    // Parse JSON body: {"bleu4": float, "rouge1": float, "rouge2": float, "rougeL": float}
    float bleu4  = -1.0f;
    float rouge1 = -1.0f;
    float rouge2 = -1.0f;
    float rougeL = -1.0f;

    size_t pos = body.find("\"bleu4\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) bleu4 = std::stof(body.substr(pos + 1));
    }
    pos = body.find("\"rouge1\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) rouge1 = std::stof(body.substr(pos + 1));
    }
    pos = body.find("\"rouge2\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) rouge2 = std::stof(body.substr(pos + 1));
    }
    pos = body.find("\"rougeL\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) rougeL = std::stof(body.substr(pos + 1));
    }

    metrics_service_->update_generation_quality_metrics(bleu4, rouge1, rouge2, rougeL);
    return "{\"status\":\"ok\"}";
}

std::string TrainingMetricsAPI::handle_post_advanced_metrics(const std::string& body) {
    // Parse JSON body: {"gradient_variance": float, "compute_time_ratio": float, "weight_update_ratio": float}
    float gradient_variance  = 0.0f;
    float compute_time_ratio = 0.0f;
    float weight_update_ratio = 0.0f;

    size_t pos = body.find("\"gradient_variance\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            gradient_variance = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"compute_time_ratio\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            compute_time_ratio = std::stof(body.substr(pos + 1));
        }
    }

    pos = body.find("\"weight_update_ratio\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            weight_update_ratio = std::stof(body.substr(pos + 1));
        }
    }

    metrics_service_->update_advanced_epoch_metrics(gradient_variance, compute_time_ratio, weight_update_ratio);

    return "{\"status\":\"ok\"}";
}

