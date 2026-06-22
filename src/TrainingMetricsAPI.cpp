#include "TrainingMetricsAPI.hpp"
#include <httplib.h>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <utility>

// ServerImpl using cpp-httplib
class TrainingMetricsAPI::ServerImpl {
   public:
    httplib::Server server;
};

namespace {

// Shared by handle_health_check() and handle_metrics_aggregate() so both endpoints
// agree on what counts as a live session.
constexpr int kHealthStalenessThreshold = 60;

class ApiRequestError : public std::runtime_error {
   public:
    ApiRequestError(int status_code, const std::string& message)
        : std::runtime_error(message), status_code_(status_code) {}

    int status_code() const {
        return status_code_;
    }

   private:
    int status_code_;
};

void set_legacy_deprecation_headers(httplib::Response& res,
                                    const std::string& replacement_path) {
    res.set_header("Deprecation", "true");
    res.set_header("Link", replacement_path);
}

}  // namespace

TrainingMetricsAPI::TrainingMetricsAPI(
    std::shared_ptr<MetricsSessionRegistry> session_registry, int port, bool allow_control,
    const std::string& name_service_url)
    : session_registry_(std::move(session_registry)),
      port_(port),
      allow_control_(allow_control),
      name_service_url_(name_service_url),
      running_(false),
      server_impl_(std::make_unique<ServerImpl>()) {
    const std::string key_pattern = "([A-Za-z0-9][A-Za-z0-9_-]{0,63})";

    // Set up session-scoped HTTP endpoints
    server_impl_->server.Get("/api/sessions", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_sessions_list();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    server_impl_->server.Get("/api/metrics/aggregate",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_metrics_aggregate();
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    // TD-021: aggregate Prometheus output for all live sessions, each labelled with session= key
    server_impl_->server.Get("/api/metrics/prometheus/aggregate",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_prometheus_aggregate();
                                     res.set_content(response, "text/plain");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/current",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                const std::string& matched_key = req.matches[1];
                adai::Logger::info("[metrics/current] matched key='{}' len={}",
                                   matched_key, matched_key.size());
                std::string response = handle_current_metrics(matched_key);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/summary",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_metrics_summary(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/history",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // Build query string from params
                std::string query_params;
                if (req.has_param("max_records")) {
                    query_params = "max_records=" + req.get_param_value("max_records");
                }
                if (req.has_param("session_id")) {
                    if (!query_params.empty()) {
                        query_params += "&";
                    }
                    query_params += "session_id=" + req.get_param_value("session_id");
                }

                std::string response = handle_metrics_history(req.matches[1], query_params);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/prometheus",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_prometheus_metrics(req.matches[1]);
                res.set_content(response, "text/plain");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/csv",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_csv_metrics(req.matches[1]);
                res.set_content(response, "text/csv");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/status",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_session_status(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/epochs",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_epoch_metrics(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/abnormal",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_abnormal_samples(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/generation-quality",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_generation_quality_metrics(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get("/api/sessions/" + key_pattern + "/metrics/padding-efficiency",
                             [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_padding_efficiency_metrics(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/start",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_session_start(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/end",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_session_end(req.matches[1]);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/epoch/start",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_epoch_start(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/epoch/end",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_epoch_end(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/metrics/sample",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_sample_metrics(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/metrics/validation",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response =
                    handle_post_validation_metrics(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/metrics/best",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_best_metrics(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/metrics/advanced",
                              [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_advanced_metrics(req.matches[1], req.body);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/sessions/" + key_pattern + "/metrics/generation-quality",
                              [this](const httplib::Request& req, httplib::Response& res) {
        try {
            std::string response =
                handle_post_generation_quality_metrics(req.matches[1], req.body);
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const ApiRequestError& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = e.status_code();
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });

    // Control endpoints (if enabled)
    if (allow_control_) {
        server_impl_->server.Post("/api/sessions/" + key_pattern + "/control/flush",
                                  [this](const httplib::Request& req, httplib::Response& res) {
                try {
                    std::string response = handle_flush_control(req.matches[1]);
                    res.set_content(response, "application/json");
                    res.status = 200;
                } catch (const ApiRequestError& e) {
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = e.status_code();
                } catch (const std::exception& e) {
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = 500;
                }
            });

        server_impl_->server.Post("/api/sessions/" + key_pattern + "/control/clear",
                                  [this](const httplib::Request& req, httplib::Response& res) {
                try {
                    std::string response = handle_clear_control(req.matches[1]);
                    res.set_content(response, "application/json");
                    res.status = 200;
                } catch (const ApiRequestError& e) {
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = e.status_code();
                } catch (const std::exception& e) {
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = 500;
                }
            });
    }

    // Legacy aliases for the 0-default session
    server_impl_->server.Get("/api/metrics/current",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_current_metrics("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/metrics/current");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/summary",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_metrics_summary("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/metrics/summary");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/history",
                             [this](const httplib::Request& req, httplib::Response& res) {
                                 try {
                                     std::string query_params;
                                     if (req.has_param("max_records")) {
                                         query_params =
                                             "max_records=" + req.get_param_value("max_records");
                                     }
                                     if (req.has_param("session_id")) {
                                         if (!query_params.empty()) {
                                             query_params += "&";
                                         }
                                         query_params +=
                                             "session_id=" + req.get_param_value("session_id");
                                     }

                                     std::string response =
                                         handle_metrics_history("0-default", query_params);
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/metrics/history");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/prometheus",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_prometheus_metrics("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/metrics/prometheus");
                                     res.set_content(response, "text/plain");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/csv",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_csv_metrics("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/metrics/csv");
                                     res.set_content(response, "text/csv");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/abnormal",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_abnormal_samples("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/metrics/abnormal");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/generation-quality",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response =
                                         handle_generation_quality_metrics("0-default");
                                     set_legacy_deprecation_headers(
                                         res,
                                         "/api/sessions/0-default/metrics/generation-quality");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/metrics/padding-efficiency",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response =
                                         handle_padding_efficiency_metrics("0-default");
                                     set_legacy_deprecation_headers(
                                         res,
                                         "/api/sessions/0-default/metrics/padding-efficiency");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/session/status",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_session_status("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/status");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Get("/api/session/epochs",
                             [this](const httplib::Request&, httplib::Response& res) {
                                 try {
                                     std::string response = handle_epoch_metrics("0-default");
                                     set_legacy_deprecation_headers(
                                         res, "/api/sessions/0-default/epochs");
                                     res.set_content(response, "application/json");
                                     res.status = 200;
                                 } catch (const std::exception& e) {
                                     res.set_content(create_error_response(e.what()),
                                                     "application/json");
                                     res.status = 500;
                                 }
                             });

    server_impl_->server.Post("/api/session/start",
                              [this](const httplib::Request& req, httplib::Response& res) {
                                  try {
                                      std::string response =
                                          handle_post_session_start("0-default", req.body);
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/start");
                                      res.set_content(response, "application/json");
                                      res.status = 200;
                                  } catch (const ApiRequestError& e) {
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/start");
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = e.status_code();
                                  } catch (const std::exception& e) {
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = 400;
                                  }
                              });

    server_impl_->server.Post("/api/session/end",
                              [this](const httplib::Request&, httplib::Response& res) {
                                  try {
                                      std::string response = handle_post_session_end("0-default");
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/end");
                                      res.set_content(response, "application/json");
                                      res.status = 200;
                                  } catch (const std::exception& e) {
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = 400;
                                  }
                              });

    server_impl_->server.Post("/api/epoch/start",
                              [this](const httplib::Request& req, httplib::Response& res) {
                                  try {
                                      std::string response =
                                          handle_post_epoch_start("0-default", req.body);
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/epoch/start");
                                      res.set_content(response, "application/json");
                                      res.status = 200;
                                  } catch (const std::exception& e) {
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = 400;
                                  }
                              });

    server_impl_->server.Post("/api/epoch/end",
                              [this](const httplib::Request& req, httplib::Response& res) {
                                  try {
                                      std::string response =
                                          handle_post_epoch_end("0-default", req.body);
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/epoch/end");
                                      res.set_content(response, "application/json");
                                      res.status = 200;
                                  } catch (const std::exception& e) {
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = 400;
                                  }
                              });

    server_impl_->server.Post("/api/metrics/sample",
                              [this](const httplib::Request& req, httplib::Response& res) {
                                  try {
                                      std::string response =
                                          handle_post_sample_metrics("0-default", req.body);
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/metrics/sample");
                                      res.set_content(response, "application/json");
                                      res.status = 200;
                                  } catch (const std::exception& e) {
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = 400;
                                  }
                              });

    server_impl_->server.Post(
        "/api/metrics/validation",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_validation_metrics("0-default", req.body);
                set_legacy_deprecation_headers(res,
                                               "/api/sessions/0-default/metrics/validation");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/metrics/best",
                              [this](const httplib::Request& req, httplib::Response& res) {
                                  try {
                                      std::string response =
                                          handle_post_best_metrics("0-default", req.body);
                                      set_legacy_deprecation_headers(
                                          res, "/api/sessions/0-default/metrics/best");
                                      res.set_content(response, "application/json");
                                      res.status = 200;
                                  } catch (const std::exception& e) {
                                      res.set_content(create_error_response(e.what()),
                                                      "application/json");
                                      res.status = 400;
                                  }
                              });

    server_impl_->server.Post(
        "/api/metrics/advanced",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_advanced_metrics("0-default", req.body);
                set_legacy_deprecation_headers(res,
                                               "/api/sessions/0-default/metrics/advanced");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/metrics/generation-quality",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response =
                    handle_post_generation_quality_metrics("0-default", req.body);
                set_legacy_deprecation_headers(
                    res, "/api/sessions/0-default/metrics/generation-quality");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    if (allow_control_) {
        server_impl_->server.Post("/api/control/flush",
                                  [this](const httplib::Request&, httplib::Response& res) {
                                      try {
                                          std::string response =
                                              handle_flush_control("0-default");
                                          set_legacy_deprecation_headers(
                                              res, "/api/sessions/0-default/control/flush");
                                          res.set_content(response, "application/json");
                                          res.status = 200;
                                      } catch (const std::exception& e) {
                                          res.set_content(create_error_response(e.what()),
                                                          "application/json");
                                          res.status = 500;
                                      }
                                  });

        server_impl_->server.Post("/api/control/clear",
                                  [this](const httplib::Request&, httplib::Response& res) {
                                      try {
                                          std::string response =
                                              handle_clear_control("0-default");
                                          set_legacy_deprecation_headers(
                                              res, "/api/sessions/0-default/control/clear");
                                          res.set_content(response, "application/json");
                                          res.status = 200;
                                      } catch (const std::exception& e) {
                                          res.set_content(create_error_response(e.what()),
                                                          "application/json");
                                          res.status = 500;
                                      }
                                  });
    }

    // GET /api/models - Registered model names from name service
    server_impl_->server.Get("/api/models", [this](const httplib::Request&, httplib::Response& res) {
        try {
            std::string response = handle_models_list();
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 502;
        }
    });

    // GET /health - Health check
    server_impl_->server.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        std::string response = handle_health_check();
        res.set_content(response, "application/json");
        res.status = 200;
    });

    // Add CORS headers to all responses
    server_impl_->server.set_post_routing_handler(
        [](const httplib::Request&, httplib::Response& res) {
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

std::string TrainingMetricsAPI::handle_current_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    return service->to_json();
}

std::string TrainingMetricsAPI::handle_metrics_summary(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    return service->to_json_summary();
}

std::string TrainingMetricsAPI::handle_metrics_history(const std::string& session_key,
                                                       const std::string& query_params) {
    auto service = resolve_session_service(session_key, false);
    // Parse query parameters
    int max_records = parse_query_param_int(query_params, "max_records", 1000);
    int session_id = parse_query_param_int(query_params, "session_id", -1);

    // Get history
    std::vector<PersistentMetricsRecord> history;
    if (session_id >= 0) {
        history = service->get_session_history(session_id);
    } else {
        history = service->get_history(max_records);
    }

    // Convert to JSON array
    std::ostringstream json;
    json << "{\"records\":[";

    for (size_t i = 0; i < history.size(); ++i) {
        const auto& record = history[i];

        if (i > 0) {
            json << ",";
        }
        json << "{";
        json << R"("timestamp":")" << std::chrono::system_clock::to_time_t(record.timestamp)
             << "\",";
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

std::string TrainingMetricsAPI::handle_prometheus_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    return service->to_prometheus(session_key);
}

std::string TrainingMetricsAPI::handle_csv_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    std::ostringstream csv;
    csv << TrainingMetricsService::to_csv_header() << "\n";
    csv << service->to_csv_row();
    return csv.str();
}

std::string TrainingMetricsAPI::handle_session_status(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    auto snapshot = service->get_current_snapshot();

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
        json << (100.0f * static_cast<float>(snapshot.current_sample) /
                 static_cast<float>(snapshot.total_samples));
    } else {
        json << 0.0f;
    }

    json << ",";
    json << "\"samples_per_second\":" << snapshot.samples_per_second << ",";
    json << "\"estimated_time_remaining_seconds\":" << snapshot.estimated_time_remaining_seconds
         << ",";
    json << "\"is_stale\":" << (snapshot.is_stale ? "true" : "false") << ",";
    json << "\"seconds_since_last_update\":" << std::fixed << std::setprecision(1)
         << snapshot.seconds_since_last_update << ",";
    json << "\"effective_is_training\":"
         << (snapshot.effective_is_training ? "true" : "false");
    json << "}";

    return json.str();
}

std::string TrainingMetricsAPI::handle_epoch_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    auto snapshot = service->get_current_snapshot();

    std::ostringstream json;
    json << "{";
    json << "\"current_epoch\":" << snapshot.current_epoch << ",";
    json << "\"total_epochs\":" << snapshot.total_epochs << ",";
    json << "\"epoch_losses\":[";

    for (size_t i = 0; i < snapshot.epoch_losses.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_losses[i];
    }

    json << "],\"epoch_validation_losses\":[";

    for (size_t i = 0; i < snapshot.epoch_validation_losses.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_validation_losses[i];
    }

    json << "],\"epoch_learning_rates\":[";

    for (size_t i = 0; i < snapshot.epoch_learning_rates.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_learning_rates[i];
    }

    json << "],\"epoch_perplexities\":[";

    for (size_t i = 0; i < snapshot.epoch_perplexities.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_perplexities[i];
    }

    json << "],\"epoch_durations\":[";

    for (size_t i = 0; i < snapshot.epoch_durations.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_durations[i];
    }

    json << "],\"epoch_gradient_norms\":[";

    for (size_t i = 0; i < snapshot.epoch_gradient_norms.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_gradient_norms[i];
    }

    // TD-017: per-epoch average adaptive clip threshold (-1 entries = fixed-clip mode)
    json << "],\"epoch_adaptive_clip_thresholds\":[";
    for (size_t i = 0; i < snapshot.epoch_adaptive_clip_thresholds.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_adaptive_clip_thresholds[i];
    }

    // TD-015: per-epoch validation perplexity and accuracy
    json << "],\"epoch_validation_perplexities\":[";
    for (size_t i = 0; i < snapshot.epoch_validation_perplexities.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_validation_perplexities[i];
    }

    json << "],\"epoch_validation_accuracies\":[";
    for (size_t i = 0; i < snapshot.epoch_validation_accuracies.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_validation_accuracies[i];
    }

    json << "],\"best_validation_loss\":" << snapshot.best_validation_loss << ",";
    json << "\"best_epoch\":" << snapshot.best_epoch;
    json << "}";

    return json.str();
}

std::string TrainingMetricsAPI::handle_abnormal_samples(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    auto samples = service->get_abnormal_samples();

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\"abnormal_samples\":[";

    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        if (i > 0) {
            json << ",";
        }
        json << "{";
        json << "\"epoch\":" << s.epoch << ",";
        json << "\"sample_id\":" << s.sample_id << ",";
        json << "\"loss\":" << s.loss << ",";
        json << "\"grad_norm\":" << s.grad_norm << ",";
        json << R"("reason":")" << escape_json(s.reason) << "\",";
        json << R"("input_text":")" << escape_json(s.input_text) << "\",";
        json << R"("target_text":")" << escape_json(s.target_text) << "\",";
        json << "\"timestamp\":" << std::chrono::system_clock::to_time_t(s.timestamp);
        json << "}";
    }

    json << "],\"count\":" << samples.size() << "}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_generation_quality_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    auto snapshot = service->get_current_snapshot();
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{";
    json << "\"current_bleu4\":" << snapshot.current_bleu4 << ",";
    json << "\"current_rouge1\":" << snapshot.current_rouge1 << ",";
    json << "\"current_rouge2\":" << snapshot.current_rouge2 << ",";
    json << "\"current_rougeL\":" << snapshot.current_rougeL << ",";

    // Per-epoch history arrays
    json << "\"epoch_bleu4\":[";
    for (size_t i = 0; i < snapshot.epoch_bleu4.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_bleu4[i];
    }
    json << "],\"epoch_rouge1\":[";
    for (size_t i = 0; i < snapshot.epoch_rouge1.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_rouge1[i];
    }
    json << "],\"epoch_rouge2\":[";
    for (size_t i = 0; i < snapshot.epoch_rouge2.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_rouge2[i];
    }
    json << "],\"epoch_rougeL\":[";
    for (size_t i = 0; i < snapshot.epoch_rougeL.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_rougeL[i];
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_padding_efficiency_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    auto snapshot = service->get_current_snapshot();
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{";
    json << "\"current_padding_efficiency\":" << snapshot.current_padding_efficiency << ",";
    json << "\"epoch_padding_efficiencies\":[";
    for (size_t i = 0; i < snapshot.epoch_padding_efficiencies.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_padding_efficiencies[i];
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_sessions_list() {
    auto sessions = session_registry_->list_sessions();
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    size_t live = 0;

    json << "{\"sessions\":[";
    for (size_t i = 0; i < sessions.size(); ++i) {
        const auto& session = sessions[i];
        if (i > 0) {
            json << ",";
        }
        if (session.is_training) {
            ++live;
        }
        json << "{";
        json << R"("key":")" << escape_json(session.key) << "\",";
        json << "\"session_id\":" << session.session_id << ",";
        json << "\"is_training\":" << (session.is_training ? "true" : "false") << ",";
        json << "\"current_epoch\":" << session.current_epoch << ",";
        json << "\"total_epochs\":" << session.total_epochs << ",";
        json << "\"current_loss\":" << session.current_loss << ",";
        json << "\"best_validation_loss\":" << session.best_validation_loss << ",";
        json << "\"session_start_time\":"
             << std::chrono::system_clock::to_time_t(session.session_start_time) << ",";
        json << "\"last_update_time\":"
             << std::chrono::system_clock::to_time_t(session.last_update_time) << ",";
        json << R"("metrics_url":")" << escape_json(session.metrics_url) << "\"";
        json << "}";
    }
    json << "],\"total\":" << sessions.size() << ",\"live\":" << live << "}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_prometheus_aggregate() {
    // Snapshot session summaries under the registry's shared lock (released on return from
    // list_sessions()), then call to_prometheus(key) on each service without holding any
    // registry lock — per-service mutex inside to_prometheus() handles thread safety.
    auto summaries = session_registry_->list_sessions();
    std::ostringstream out;
    for (const auto& summary : summaries) {
        auto maybe_service = session_registry_->get_session(summary.key);
        if (!maybe_service) {
            continue;  // session evicted between list and get — skip
        }
        out << (*maybe_service)->to_prometheus(summary.key);
    }
    return out.str();
}

std::string TrainingMetricsAPI::handle_metrics_aggregate() {
    auto sessions = session_registry_->list_sessions();

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);

    size_t live_sessions = 0;
    json << "{\"live_sessions\":";

    std::ostringstream sessions_json;
    sessions_json << "\"sessions\":[";
    bool first = true;
    auto now = std::chrono::system_clock::now();
    for (const auto& session : sessions) {
        if (!session.is_training) {
            continue;
        }
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                        now - session.last_update_time)
                        .count();
        if (secs > kHealthStalenessThreshold) {
            continue;
        }
        ++live_sessions;
        if (!first) {
            sessions_json << ",";
        }
        first = false;

        sessions_json << "{";
        sessions_json << R"("key":")" << escape_json(session.key) << "\",";
        sessions_json << "\"epoch\":" << session.current_epoch << ",";
        sessions_json << "\"loss\":" << session.current_loss << ",";
        sessions_json << "\"validation_loss\":" << session.current_validation_loss;
        sessions_json << "}";
    }
    sessions_json << "]";

    json << live_sessions << "," << sessions_json.str() << "}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_flush_control(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    service->flush_to_disk();
    return R"({"status":"success","message":"Metrics flushed to disk"})";
}

std::string TrainingMetricsAPI::handle_clear_control(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    service->clear_history();
    return R"({"status":"success","message":"Metrics history cleared"})";
}

std::string TrainingMetricsAPI::handle_health_check() {
    auto sessions = session_registry_->list_sessions();

    // TD-019: Compute per-session staleness to determine effective liveness.
    // A session is "effectively active" only when is_training is true AND
    // it has received a recent metrics ingest (not stale).
    bool is_active = false;
    bool any_stale = false;
    for (const auto& summary : sessions) {
        if (!summary.is_training) {
            continue;
        }
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now() - summary.last_update_time)
                        .count();
        // Session-level staleness uses the per-service configured threshold (which may extend
        // during validation); the aggregate/health view uses the flat kHealthStalenessThreshold.
        if (secs > kHealthStalenessThreshold) {
            any_stale = true;
        } else {
            is_active = true;
        }
    }

    std::ostringstream json;
    json << "{";
    json << R"("status":"ok",)";
    json << R"("service":"TrainingMetricsAPI",)";
    json << "\"is_training\":" << (is_active ? "true" : "false") << ",";
    json << "\"any_stale\":" << (any_stale ? "true" : "false");
    json << "}";

    return json.str();
}

std::string TrainingMetricsAPI::handle_models_list() {
    if (name_service_url_.empty()) {
        return R"({"error":"name service not configured","models":[]})";
    }

    // Parse host:port from the configured URL
    std::string host = "localhost";
    int port = 8083;
    std::string url = name_service_url_;
    if (url.rfind("http://", 0) == 0) url = url.substr(7);
    else if (url.rfind("https://", 0) == 0) url = url.substr(8);
    auto colon = url.find(':');
    if (colon != std::string::npos) {
        host = url.substr(0, colon);
        auto slash = url.find('/', colon);
        std::string port_str = (slash != std::string::npos)
            ? url.substr(colon + 1, slash - colon - 1)
            : url.substr(colon + 1);
        try { port = std::stoi(port_str); } catch (...) {}
    } else {
        auto slash = url.find('/');
        host = (slash != std::string::npos) ? url.substr(0, slash) : url;
    }

    httplib::Client http(host, port);
    http.set_connection_timeout(5, 0);
    http.set_read_timeout(5, 0);
    auto res = http.Get("/models");
    if (!res || res->status != 200) {
        return R"({"error":"name service unavailable","models":[]})";
    }
    return res->body;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::string TrainingMetricsAPI::create_error_response(const std::string& error_message) {
    // Pre-built JSON bodies (e.g. structured 503 capacity errors) are passed through as-is.
    if (!error_message.empty() && error_message.front() == '{') {
        return error_message;
    }
    std::ostringstream json;
    json << "{";
    json << R"("error":")" << escape_json(error_message) << "\"";
    json << "}";
    return json.str();
}

std::string TrainingMetricsAPI::escape_json(const std::string& s) {
    std::ostringstream escaped;
    for (char c : s) {
        switch (c) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\b':
                escaped << "\\b";
                break;
            case '\f':
                escaped << "\\f";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                if (c < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(c);
                } else {
                    escaped << c;
                }
        }
    }
    return escaped.str();
}

int TrainingMetricsAPI::parse_query_param_int(const std::string& query, const std::string& param,
                                              int default_value) {
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

std::string TrainingMetricsAPI::handle_post_session_start(const std::string& session_key,
                                                          const std::string& body) {
    auto service = resolve_session_service(session_key, true);

    const auto snapshot = service->get_current_snapshot();
    if (snapshot.is_training && !snapshot.is_stale) {
        throw ApiRequestError(409,
                              "session already active for key: " + session_key);
    }
    if (snapshot.is_stale) {
        adai::Logger::warn("[session/start] key='{}' is stale — allowing restart", session_key);
    }

    // Parse JSON body: {"session_id": int, "total_epochs": int, "total_samples": int,
    //                   "label": string (optional), "config": JSON object (optional)}
    int session_id = 0;
    int total_epochs = 0;
    int total_samples = 0;
    std::string label;
    std::string config_snapshot;

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

    // Extract optional "label" string field
    pos = body.find("\"label\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            size_t q1 = body.find('"', pos + 1);
            if (q1 != std::string::npos) {
                size_t q2 = body.find('"', q1 + 1);
                if (q2 != std::string::npos) {
                    label = body.substr(q1 + 1, q2 - q1 - 1);
                }
            }
        }
    }

    // Extract optional "config" JSON object field by brace-matching
    pos = body.find("\"config\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            size_t brace_start = body.find('{', pos + 1);
            if (brace_start != std::string::npos) {
                int depth = 1;
                size_t i = brace_start + 1;
                while (i < body.size() && depth > 0) {
                    if (body[i] == '{') ++depth;
                    else if (body[i] == '}') --depth;
                    ++i;
                }
                if (depth == 0) {
                    config_snapshot = body.substr(brace_start, i - brace_start);
                }
            }
        }
    }

    // Extract optional "model_id" string field (Phase 2 — MNS UUID injected into config_snapshot)
    std::string model_id;
    pos = body.find("\"model_id\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            size_t q1 = body.find('"', pos + 1);
            if (q1 != std::string::npos) {
                size_t q2 = body.find('"', q1 + 1);
                if (q2 != std::string::npos) {
                    model_id = body.substr(q1 + 1, q2 - q1 - 1);
                }
            }
        }
    }
    if (!model_id.empty()) {
        if (config_snapshot.size() >= 2 && config_snapshot.back() == '}') {
            const bool is_empty_obj = (config_snapshot == "{}");
            config_snapshot.pop_back();
            if (!is_empty_obj) config_snapshot += ',';
            config_snapshot += "\"model_id\":\"" + model_id + "\"}";
        } else {
            config_snapshot = "{\"model_id\":\"" + model_id + "\"}";
        }
    }

    service->start_session(session_id, total_epochs, total_samples, label, config_snapshot);

    return R"({"status":"ok","message":"Session started"})";
}

std::string TrainingMetricsAPI::handle_post_session_end(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    service->end_session();
    return R"({"status":"ok","message":"Session ended"})";
}

std::string TrainingMetricsAPI::handle_post_epoch_start(const std::string& session_key,
                                                        const std::string& body) {
    auto service = resolve_session_service(session_key, false);
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

    service->start_epoch(epoch, total_samples);

    return R"({"status":"ok","message":"Epoch started"})";
}

std::string TrainingMetricsAPI::handle_post_epoch_end(const std::string& session_key,
                                                      const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // Parse JSON body: {"epoch": int, "loss": float, "validation_loss": float, "learning_rate":
    // float, "perplexity": float, "gradient_norm": float}
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

    service->end_epoch(epoch, loss, validation_loss, learning_rate, perplexity, gradient_norm,
                       epoch_time);
    if (gradient_variance != 0.0f || compute_time_ratio != 0.0f || weight_update_ratio != 0.0f) {
        service->update_advanced_epoch_metrics(gradient_variance, compute_time_ratio,
                                               weight_update_ratio);
    }
    if (activation_saturation_ratio >= 0.0f) {
        service->update_activation_saturation(activation_saturation_ratio);
    }
    if (attention_entropy >= 0.0f) {
        service->update_attention_entropy(attention_entropy);
    }
    if (current_padding_efficiency >= 0.0f) {
        service->update_padding_efficiency(current_padding_efficiency);
    }

    return R"({"status":"ok","message":"Epoch ended"})";
}

std::string TrainingMetricsAPI::handle_post_sample_metrics(const std::string& session_key,
                                                           const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // Parse JSON body: {"sample": int, "loss": float, "gradient_norm": float, "learning_rate":
    // float}
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

    service->update_sample_metrics(sample, loss, gradient_norm, learning_rate);

    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_validation_metrics(const std::string& session_key,
                                                               const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // Parse JSON body: {"validation_loss": float, "validation_accuracy": float,
    // "validation_perplexity": float}
    float validation_loss = 0.0f;
    float validation_accuracy = -1.0f;   // TD-015: optional
    float validation_perplexity = 0.0f;  // TD-015: optional (0 = auto-derive from loss)

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

    service->update_validation_metrics(validation_loss, validation_accuracy, validation_perplexity);

    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_best_metrics(const std::string& session_key,
                                                         const std::string& body) {
    auto service = resolve_session_service(session_key, false);
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

    service->update_best_metrics(validation_loss, epoch);

    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_generation_quality_metrics(
    const std::string& session_key, const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // Parse JSON body: {"bleu4": float, "rouge1": float, "rouge2": float, "rougeL": float}
    float bleu4 = -1.0f;
    float rouge1 = -1.0f;
    float rouge2 = -1.0f;
    float rougeL = -1.0f;

    size_t pos = body.find("\"bleu4\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            bleu4 = std::stof(body.substr(pos + 1));
        }
    }
    pos = body.find("\"rouge1\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            rouge1 = std::stof(body.substr(pos + 1));
        }
    }
    pos = body.find("\"rouge2\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            rouge2 = std::stof(body.substr(pos + 1));
        }
    }
    pos = body.find("\"rougeL\"");
    if (pos != std::string::npos) {
        pos = body.find(':', pos);
        if (pos != std::string::npos) {
            rougeL = std::stof(body.substr(pos + 1));
        }
    }

    service->update_generation_quality_metrics(bleu4, rouge1, rouge2, rougeL);
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_advanced_metrics(const std::string& session_key,
                                                             const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // Parse JSON body: {"gradient_variance": float, "compute_time_ratio": float,
    // "weight_update_ratio": float}
    float gradient_variance = 0.0f;
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

    service->update_advanced_epoch_metrics(gradient_variance, compute_time_ratio,
                                           weight_update_ratio);

    return R"({"status":"ok"})";
}

std::shared_ptr<TrainingMetricsService> TrainingMetricsAPI::resolve_session_service(
    const std::string& session_key, bool create_if_missing) const {
    if (!is_valid_session_key(session_key)) {
        throw ApiRequestError(400, "Invalid session key format");
    }

    if (create_if_missing) {
        auto service = session_registry_->create_or_get_session(session_key);
        if (!service) {
            std::ostringstream body;
            body << R"({"error":"metrics_server_full","max_live_sessions":)"
                 << session_registry_->max_live_sessions() << "}";
            throw ApiRequestError(503, body.str());
        }
        return service;
    }

    auto service = session_registry_->get_session(session_key);
    if (!service.has_value()) {
        throw ApiRequestError(404, "Unknown session key: " + session_key);
    }

    return service.value();
}

bool TrainingMetricsAPI::is_valid_session_key(const std::string& key) {
    if (key.empty() || key.size() > 64) {
        return false;
    }

    if (!std::isalnum(static_cast<unsigned char>(key.front()))) {
        return false;
    }

    return std::all_of(key.begin(), key.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '-';
    });
}
