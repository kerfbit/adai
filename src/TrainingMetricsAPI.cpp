// @adai-status: beta        (capped by TD-039 — large, actively evolving)
// @adai-version: 0.15.2
// @adai-reviewed: 2026-10-25

#include "TrainingMetricsAPI.hpp"
#include "BrowserWriteGuard.hpp"
#include "HttpListenSocket.hpp"
#include "PortableTime.hpp"
#include "nlohmann/json.hpp"
#include <httplib.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

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

void set_legacy_deprecation_headers(httplib::Response& res, const std::string& replacement_path) {
    res.set_header("Deprecation", "true");
    res.set_header("Link", replacement_path);
}

// TD-527: a pushed request body, parsed once with nlohmann/json. Fields used to be found by
// substring search (a key inside a nested object or a string value could match) and read with
// stoi/stof (a malformed number came back as a 400 saying only "stof"). Absent fields take the
// caller's default; a field of the wrong type is a 400 naming the field. JSON null in a float field
// is NaN (TD-443: what MetricsPushClient sends for NaN/infinity).
class PushBody {
   public:
    explicit PushBody(const std::string& body) {
        if (body.find_first_not_of(" \t\r\n") == std::string::npos) {
            j_ = nlohmann::json::object();
            return;
        }
        j_ = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
        if (j_.is_discarded())
            throw ApiRequestError(400, "request body is not valid JSON");
        if (!j_.is_object())
            throw ApiRequestError(400, "request body must be a JSON object");
    }

    int get_int(const char* key, int fallback) const {
        const auto* v = find(key);
        if (!v || v->is_null())
            return fallback;
        if (v->is_number_integer() &&
            v->get<long long>() >= std::numeric_limits<int>::min() &&
            v->get<long long>() <= std::numeric_limits<int>::max())
            return static_cast<int>(v->get<long long>());
        if (v->is_number_float() && std::floor(v->get<double>()) == v->get<double>() &&
            std::fabs(v->get<double>()) <= std::numeric_limits<int>::max())
            return static_cast<int>(v->get<double>());
        throw bad_field(key, "an integer");
    }

    double get_double(const char* key, double fallback) const {
        const auto* v = find(key);
        if (!v)
            return fallback;
        if (v->is_null())
            return std::numeric_limits<double>::quiet_NaN();
        if (!v->is_number())
            throw bad_field(key, "a number");
        return v->get<double>();
    }

    float get_float(const char* key, float fallback) const {
        return static_cast<float>(get_double(key, fallback));
    }

    bool get_bool(const char* key, bool fallback) const {
        const auto* v = find(key);
        if (!v || v->is_null())
            return fallback;
        if (!v->is_boolean())
            throw bad_field(key, "true or false");
        return v->get<bool>();
    }

    std::string get_string(const char* key) const {
        const auto* v = find(key);
        if (!v || v->is_null())
            return {};
        if (!v->is_string())
            throw bad_field(key, "a string");
        return v->get<std::string>();
    }

    // A JSON object field, re-serialized; empty when absent or null.
    nlohmann::json get_object(const char* key) const {
        const auto* v = find(key);
        if (!v || v->is_null())
            return nlohmann::json();
        if (!v->is_object())
            throw bad_field(key, "an object");
        return *v;
    }

    // An array of numbers; null elements are NaN, kept in place so later indices don't shift.
    std::vector<float> get_float_array(const char* key) const {
        std::vector<float> out;
        const auto* v = find(key);
        if (!v || v->is_null())
            return out;
        if (!v->is_array())
            throw bad_field(key, "an array of numbers");
        for (const auto& e : *v) {
            if (e.is_null())
                out.push_back(std::numeric_limits<float>::quiet_NaN());
            else if (e.is_number())
                out.push_back(e.get<float>());
            else
                throw bad_field(key, "an array of numbers");
        }
        return out;
    }

   private:
    const nlohmann::json* find(const char* key) const {
        const auto it = j_.find(key);
        return it == j_.end() ? nullptr : &*it;
    }
    static ApiRequestError bad_field(const char* key, const char* expected) {
        return ApiRequestError(400, std::string("field '") + key + "' must be " + expected);
    }

    nlohmann::json j_;
};

// TD-526: METRICS_API_CORS_ALLOWED_ORIGINS entries, trimmed, without a trailing '/' (browsers
// send "https://host:port" with no path).
std::vector<std::string> parse_origin_list(const std::string& list) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= list.size()) {
        std::size_t comma = list.find(',', start);
        if (comma == std::string::npos)
            comma = list.size();
        std::string item = list.substr(start, comma - start);
        const auto first = item.find_first_not_of(" \t");
        const auto last = item.find_last_not_of(" \t");
        item = first == std::string::npos ? "" : item.substr(first, last - first + 1);
        while (item.size() > 1 && item.back() == '/')
            item.pop_back();
        if (!item.empty())
            out.push_back(item);
        start = comma + 1;
    }
    return out;
}

}  // namespace

// TD-443: JSON has no NaN or infinity, but this file and TrainingMetricsService build JSON with
// operator<<, which writes them as nan/inf: a diverged run's /epochs read
// "epoch_losses":[nan,2.5], which JSON parsers reject, exactly when the dashboard is needed. Every
// JSON response passes through here (post-routing), so a bare nan/-nan/inf/-inf/infinity value
// outside a string becomes null, whichever endpoint produced it.
std::string sanitize_non_finite_json(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool in_string = false;
    char prev_sig = 0;  // last non-space character outside strings
    for (std::size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (in_string) {
            out += c;
            if (c == '\\' && i + 1 < in.size()) {
                out += in[++i];
            } else if (c == '"') {
                in_string = false;
                prev_sig = '"';
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            out += c;
            continue;
        }
        if ((prev_sig == ':' || prev_sig == '[' || prev_sig == ',') &&
            (c == '-' || c == '+' || c == 'n' || c == 'N' || c == 'i' || c == 'I')) {
            std::size_t j = i + ((c == '-' || c == '+') ? 1 : 0);
            std::size_t k = j;
            while (k < in.size() && std::isalpha(static_cast<unsigned char>(in[k])))
                ++k;
            std::string word = in.substr(j, k - j);
            std::transform(word.begin(), word.end(), word.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if ((word == "nan" || word == "inf" || word == "infinity") &&
                (k >= in.size() || !std::isalnum(static_cast<unsigned char>(in[k])))) {
                out += "null";
                prev_sig = 'l';
                i = k - 1;
                continue;
            }
        }
        out += c;
        if (!std::isspace(static_cast<unsigned char>(c)))
            prev_sig = c;
    }
    return out;
}

TrainingMetricsAPI::TrainingMetricsAPI(std::shared_ptr<MetricsSessionRegistry> session_registry,
                                       int port, bool allow_control,
                                       const std::string& name_service_url,
                                       const std::string& admin_config_db_dir,
                                       const std::string& cors_allowed_origins)
    : session_registry_(std::move(session_registry)),
      port_(port),
      allow_control_(allow_control),
      name_service_url_(name_service_url),
      running_(false),
      server_impl_(std::make_unique<ServerImpl>()) {
    for (const std::string& origin : parse_origin_list(cors_allowed_origins)) {
        if (origin == "*")
            cors_any_origin_reads_ = true;
        else
            cors_origins_.insert(origin);
    }
    if (!admin_config_db_dir.empty()) {
        try {
            config_store_ = std::make_unique<adai::DaemonConfigStore>(admin_config_db_dir +
                                                                      "/daemon_config.db");
        } catch (const std::exception& e) {
            adai::Logger::warn(
                "TrainingMetricsAPI: daemon_config.db unavailable ({}); admin config changes "
                "won't persist across restarts",
                e.what());
        }
    }
    const std::string key_pattern = "([A-Za-z0-9][A-Za-z0-9_-]{0,63})";

    // Set up session-scoped HTTP endpoints
    server_impl_->server.Get(
        "/api/sessions", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string query_params;
                for (const auto& param : {"status", "from"}) {
                    if (req.has_param(param)) {
                        if (!query_params.empty())
                            query_params += "&";
                        query_params += std::string(param) + "=" + req.get_param_value(param);
                    }
                }
                std::string response = query_params.empty()
                                           ? handle_sessions_list()
                                           : handle_sessions_list_filtered(query_params);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    // TD-020: cross-session metric comparison from DB
    server_impl_->server.Get(
        "/api/metrics/compare", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string query_params;
                if (req.has_param("keys"))
                    query_params += "keys=" + req.get_param_value("keys");
                if (req.has_param("metric")) {
                    if (!query_params.empty())
                        query_params += "&";
                    query_params += "metric=" + req.get_param_value("metric");
                }
                std::string response = handle_metrics_compare(query_params);
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/aggregate", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_metrics_aggregate();
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    // TD-021: aggregate Prometheus output for all live sessions, each labelled with session= key
    server_impl_->server.Get("/api/metrics/prometheus/aggregate", [this](const httplib::Request&,
                                                                         httplib::Response& res) {
        try {
            std::string response = handle_prometheus_aggregate();
            res.set_content(response, "text/plain");
            res.status = 200;
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 500;
        }
    });

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/current",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                const std::string& matched_key = req.matches[1];
                // TD-527: debug, not info: dashboards poll this continuously (one line per poll
                // crowded the log and sped up its rotation).
                adai::Logger::debug("[metrics/current] key='{}'", matched_key);
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/summary",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/history",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/prometheus",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/csv",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/status",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/epochs",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/abnormal",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/generation-quality",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/padding-efficiency",
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

    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/lejepa",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_lejepa_metrics(req.matches[1]);
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

    // TD-020: DB-backed time-range history query
    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/db-history",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string query_params;
                for (const auto& param : {"from", "to", "limit"}) {
                    if (req.has_param(param)) {
                        if (!query_params.empty())
                            query_params += "&";
                        query_params += std::string(param) + "=" + req.get_param_value(param);
                    }
                }
                std::string response = handle_db_history(req.matches[1], query_params);
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

    // TD-440: readers for the per-step gradient variance and per-epoch generation quality tables
    // (written since TD-013/TD-019, never readable before).
    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/db-gradient-variance",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string query_params;
                if (req.has_param("limit"))
                    query_params = "limit=" + req.get_param_value("limit");
                res.set_content(handle_db_gradient_variance(req.matches[1], query_params),
                                "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });
    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/db-generation-quality",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                res.set_content(handle_db_generation_quality(req.matches[1]), "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    // TD-020: Full unbounded history export from DB
    server_impl_->server.Get(
        "/api/sessions/" + key_pattern + "/metrics/export",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string query_params;
                if (req.has_param("format")) {
                    query_params = "format=" + req.get_param_value("format");
                }
                std::string response = handle_metrics_export(req.matches[1], query_params);
                std::string format =
                    req.has_param("format") ? req.get_param_value("format") : "json";
                std::string content_type = (format == "csv") ? "text/csv" : "application/json";
                res.set_content(response, content_type);
                res.status = 200;
            } catch (const ApiRequestError& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/start",
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/end",
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/heartbeat",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_heartbeat(req.matches[1]);
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/epoch/start",
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/epoch/end",
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/sample",
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/validation",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_validation_metrics(req.matches[1], req.body);
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/best",
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

    // TD-555: the trainer pushes flagged samples here. The path was GET-only, so every push got a
    // 404 that MetricsPushClient took for a lost session, and the samples were never stored.
    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/abnormal",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_abnormal_sample(req.matches[1], req.body);
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/advanced",
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/layer-gradients",
        [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_layer_gradient_norms(req.matches[1], req.body);
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

    server_impl_->server.Post(
        "/api/sessions/" + key_pattern + "/metrics/generation-quality",
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
        server_impl_->server.Post(
            "/api/sessions/" + key_pattern + "/control/flush",
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

        server_impl_->server.Post(
            "/api/sessions/" + key_pattern + "/control/clear",
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
    server_impl_->server.Get(
        "/api/metrics/current", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_current_metrics("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/current");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/current");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/summary", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_metrics_summary("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/summary");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/summary");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/history", [this](const httplib::Request& req, httplib::Response& res) {
            try {
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

                std::string response = handle_metrics_history("0-default", query_params);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/history");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/history");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/prometheus", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_prometheus_metrics("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/prometheus");
                res.set_content(response, "text/plain");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/prometheus");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/csv", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_csv_metrics("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/csv");
                res.set_content(response, "text/csv");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/csv");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/abnormal", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_abnormal_samples("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/abnormal");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/abnormal");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/generation-quality", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_generation_quality_metrics("0-default");
                set_legacy_deprecation_headers(
                    res, "/api/sessions/0-default/metrics/generation-quality");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(
                    res, "/api/sessions/0-default/metrics/generation-quality");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/padding-efficiency", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_padding_efficiency_metrics("0-default");
                set_legacy_deprecation_headers(
                    res, "/api/sessions/0-default/metrics/padding-efficiency");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(
                    res, "/api/sessions/0-default/metrics/padding-efficiency");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/metrics/lejepa", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_lejepa_metrics("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/lejepa");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/lejepa");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/session/status", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_session_status("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/status");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/status");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Get(
        "/api/session/epochs", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_epoch_metrics("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/epochs");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/epochs");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 500;
            }
        });

    server_impl_->server.Post(
        "/api/session/start", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_session_start("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/start");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/start");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/session/end", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::string response = handle_post_session_end("0-default");
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/end");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/end");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/epoch/start", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_epoch_start("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/epoch/start");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/epoch/start");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/epoch/end", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_epoch_end("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/epoch/end");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/epoch/end");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/metrics/sample", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_sample_metrics("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/sample");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/sample");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/metrics/validation", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_validation_metrics("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/validation");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/validation");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/metrics/best", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_best_metrics("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/best");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/best");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post(
        "/api/metrics/advanced", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string response = handle_post_advanced_metrics("0-default", req.body);
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/advanced");
                res.set_content(response, "application/json");
                res.status = 200;
            } catch (const ApiRequestError& e) {
                set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/advanced");
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = e.status_code();
            } catch (const std::exception& e) {
                res.set_content(create_error_response(e.what()), "application/json");
                res.status = 400;
            }
        });

    server_impl_->server.Post("/api/metrics/generation-quality", [this](const httplib::Request& req,
                                                                        httplib::Response& res) {
        try {
            std::string response = handle_post_generation_quality_metrics("0-default", req.body);
            set_legacy_deprecation_headers(res,
                                           "/api/sessions/0-default/metrics/generation-quality");
            res.set_content(response, "application/json");
            res.status = 200;
        } catch (const ApiRequestError& e) {
            set_legacy_deprecation_headers(res, "/api/sessions/0-default/metrics/generation-quality");
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = e.status_code();
        } catch (const std::exception& e) {
            res.set_content(create_error_response(e.what()), "application/json");
            res.status = 400;
        }
    });

    if (allow_control_) {
        server_impl_->server.Post(
            "/api/control/flush", [this](const httplib::Request&, httplib::Response& res) {
                try {
                    std::string response = handle_flush_control("0-default");
                    set_legacy_deprecation_headers(res, "/api/sessions/0-default/control/flush");
                    res.set_content(response, "application/json");
                    res.status = 200;
                } catch (const ApiRequestError& e) {
                    set_legacy_deprecation_headers(res, "/api/sessions/0-default/control/flush");
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = e.status_code();
                } catch (const std::exception& e) {
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = 500;
                }
            });

        server_impl_->server.Post(
            "/api/control/clear", [this](const httplib::Request&, httplib::Response& res) {
                try {
                    std::string response = handle_clear_control("0-default");
                    set_legacy_deprecation_headers(res, "/api/sessions/0-default/control/clear");
                    res.set_content(response, "application/json");
                    res.status = 200;
                } catch (const ApiRequestError& e) {
                    set_legacy_deprecation_headers(res, "/api/sessions/0-default/control/clear");
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = e.status_code();
                } catch (const std::exception& e) {
                    res.set_content(create_error_response(e.what()), "application/json");
                    res.status = 500;
                }
            });
    }

    // Admin config: live-mutable subset of this daemon's settings. Gated behind
    // the same allow_control_ flag as the control/flush and control/clear
    // endpoints above. port, storage_backend, db_path, db_url, db_pool_size, and
    // the metrics/summary/prometheus file paths are never exposed here — they're
    // baked into an already-bound listener socket or an already-opened DB
    // handle. See CLAUDE.md "Daemon admin config API".
    if (allow_control_) {
        server_impl_->server.Get(
            "/admin/config", [this](const httplib::Request&, httplib::Response& res) {
                res.set_content(handle_admin_get_config(), "application/json");
                res.status = 200;
            });

        server_impl_->server.Put(
            "/admin/config", [this](const httplib::Request& req, httplib::Response& res) {
                auto [status, body] = handle_admin_put_config(req.body);
                res.status = status;
                res.set_content(body, "application/json");
            });
    }

    // GET /api/models - Registered model names from name service
    server_impl_->server.Get(
        "/api/models", [this](const httplib::Request&, httplib::Response& res) {
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

    // TD-526: CORS. This API has no authentication, and it used to answer every request with
    // "Access-Control-Allow-Origin: *" and parse any body whatever its Content-Type, so any web page
    // open in an operator's browser could clear or inject metrics with a preflight-free text/plain
    // POST and read every response, abnormal-sample training text included. Browsers always send
    // Origin on cross-origin requests and on POSTs; the trainer, the Android app and curl don't.
    //   - Writes carrying an Origin that isn't explicitly listed are refused (403), and writes with
    //     a preflight-free Content-Type are refused (415), so a browser can't send one unasked.
    //   - Responses carry CORS headers only for a listed origin ("*" grants reads, never writes).
    // The write guard is shared with the other daemons (TD-550, BrowserWriteGuard.hpp).
    adai::install_browser_write_guard(server_impl_->server, "TrainingMetricsAPI", cors_origins_);
    adai::use_exclusive_listen_port(server_impl_->server);  // TD-551

    server_impl_->server.set_post_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
            // TD-443: no response may carry nan/inf as a JSON value. httplib has already set
            // Content-Length from the body by now, so a changed body (null is longer than nan/inf)
            // needs it reset, or clients read a truncated reply. Compressed bodies are left alone.
            if (res.get_header_value("Content-Type").rfind("application/json", 0) == 0 &&
                !res.has_header("Content-Encoding")) {
                std::string sanitized = sanitize_non_finite_json(res.body);
                if (sanitized != res.body) {
                    res.body = std::move(sanitized);
                    const auto range = res.headers.equal_range("Content-Length");
                    res.headers.erase(range.first, range.second);
                    res.set_header("Content-Length", std::to_string(res.body.size()));
                }
            }
            const std::string origin = req.get_header_value("Origin");
            if (origin.empty() || req.method == "OPTIONS")
                return;
            res.set_header("Vary", "Origin");
            if (cors_origins_.count(origin) != 0)
                res.set_header("Access-Control-Allow-Origin", origin);
            else if (cors_any_origin_reads_ && !adai::is_write_method(req.method))
                res.set_header("Access-Control-Allow-Origin", "*");
        });

    // Preflight: only a listed origin is told it may send writes.
    server_impl_->server.Options(".*", [this](const httplib::Request& req,
                                              httplib::Response& res) {
        const std::string origin = req.get_header_value("Origin");
        res.set_header("Vary", "Origin");
        if (origin.empty() || cors_origins_.count(origin) == 0) {
            res.status = 403;
            return;
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });
}

TrainingMetricsAPI::~TrainingMetricsAPI() {
    if (running_) {
        stop();
    }
}

// TD-528: httplib's stop() acts only once listen() is running, so a stop() that came before (a
// SIGTERM right after the signal handlers were installed) used to be lost and the server ran forever.
// start() and stop() now hand off through two flags: start() marks itself starting before it checks
// for a stop request, and stop() requests the stop before it checks for a start in progress, so at
// least one of them sees the other.
bool TrainingMetricsAPI::start() {
    if (running_ || starting_.exchange(true)) {
        return false;
    }
    struct ClearStarting {
        std::atomic<bool>& flag;
        ~ClearStarting() { flag = false; }
    } clear_starting{starting_};

    if (stop_requested_) {
        return false;
    }
    if (!server_impl_->server.bind_to_port("0.0.0.0", port_)) {
        return false;
    }
    running_ = true;

    // Blocking: runs until stop() is called.
    bool success = server_impl_->server.listen_after_bind();

    running_ = false;
    return success;
}

void TrainingMetricsAPI::stop() {
    stop_requested_ = true;
    // A start() in progress: wait (briefly) until it is listening, when httplib's stop() works, or
    // until it has given up after seeing stop_requested_.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (starting_ && !server_impl_->server.is_running() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    server_impl_->server.stop();
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
    // TD-527: capped at the in-memory history size (it was uncapped).
    int max_records = parse_query_param_int(query_params, "max_records", 1000);
    max_records = std::max(1, std::min(max_records, kMaxHistoryRecords));
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
        // "timestamp" stays whole seconds (dashboards read it as is); TD-527 adds milliseconds.
        json << R"("timestamp":")" << std::chrono::system_clock::to_time_t(record.timestamp)
             << "\",";
        json << "\"timestamp_ms\":"
             << std::chrono::duration_cast<std::chrono::milliseconds>(
                    record.timestamp.time_since_epoch())
                    .count()
             << ",";
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
    // TD-527: only progress_percent is rounded to 2 decimals. std::fixed used to stay set for the
    // fields after it, rounding samples_per_second (and the ETA) to 2 decimals too.
    std::ostringstream progress;
    progress << std::fixed << std::setprecision(2)
             << (snapshot.total_samples > 0
                     ? 100.0f * static_cast<float>(snapshot.current_sample) /
                           static_cast<float>(snapshot.total_samples)
                     : 0.0f);
    json << "\"progress_percent\":" << progress.str() << ",";
    json << "\"samples_per_second\":" << snapshot.samples_per_second << ",";
    json << "\"estimated_time_remaining_seconds\":" << snapshot.estimated_time_remaining_seconds
         << ",";
    json << "\"is_stale\":" << (snapshot.is_stale ? "true" : "false") << ",";
    json << "\"seconds_since_last_update\":" << std::fixed << std::setprecision(1)
         << snapshot.seconds_since_last_update << ",";
    json << "\"effective_is_training\":" << (snapshot.effective_is_training ? "true" : "false");
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

std::string TrainingMetricsAPI::handle_lejepa_metrics(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    auto snapshot = service->get_current_snapshot();
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{";
    json << "\"current_predictor_loss\":" << snapshot.current_predictor_loss << ",";
    json << "\"current_sigreg_loss\":" << snapshot.current_sigreg_loss << ",";
    json << "\"epoch_predictor_losses\":[";
    for (size_t i = 0; i < snapshot.epoch_predictor_losses.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_predictor_losses[i];
    }
    json << "],\"epoch_sigreg_losses\":[";
    for (size_t i = 0; i < snapshot.epoch_sigreg_losses.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_sigreg_losses[i];
    }
    json << "],";
    json << "\"current_masking_ratio\":" << snapshot.current_masking_ratio << ",";
    json << "\"current_predictor_target_cosine_sim\":"
         << snapshot.current_predictor_target_cosine_sim << ",";
    json << "\"current_sigreg_variance_mean\":" << snapshot.current_sigreg_variance_mean << ",";
    json << "\"current_sigreg_variance_stddev\":" << snapshot.current_sigreg_variance_stddev
         << ",";
    json << "\"epoch_masking_ratios\":[";
    for (size_t i = 0; i < snapshot.epoch_masking_ratios.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_masking_ratios[i];
    }
    json << "],\"epoch_predictor_target_cosine_sims\":[";
    for (size_t i = 0; i < snapshot.epoch_predictor_target_cosine_sims.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_predictor_target_cosine_sims[i];
    }
    json << "],\"epoch_sigreg_variance_means\":[";
    for (size_t i = 0; i < snapshot.epoch_sigreg_variance_means.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_sigreg_variance_means[i];
    }
    json << "],\"epoch_sigreg_variance_stddevs\":[";
    for (size_t i = 0; i < snapshot.epoch_sigreg_variance_stddevs.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << snapshot.epoch_sigreg_variance_stddevs[i];
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
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(now - session.last_update_time)
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

namespace {

bool json_has_key(const std::string& body, const std::string& key) {
    return body.find("\"" + key + "\"") != std::string::npos;
}

std::optional<long long> json_int_value(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return std::nullopt;
    try {
        return std::stoll(body.substr(pos + needle.size()));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> json_bool_value(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return std::nullopt;
    const auto start = pos + needle.size();
    if (body.compare(start, 4, "true") == 0)
        return true;
    if (body.compare(start, 5, "false") == 0)
        return false;
    return std::nullopt;
}

}  // namespace

std::string TrainingMetricsAPI::handle_admin_get_config() {
    const auto cfg = session_registry_->base_metrics_config();
    std::ostringstream j;
    j << "{\"max_live_sessions\":" << session_registry_->max_live_sessions()
      << ",\"completed_ttl_seconds\":" << session_registry_->completed_ttl_seconds()
      << ",\"sweep_interval_seconds\":" << session_registry_->sweep_interval_seconds()
      << ",\"persist_every_samples\":" << cfg.persist_every_samples
      << ",\"persist_every_seconds\":" << cfg.persist_every_seconds
      << ",\"max_records_in_memory\":" << cfg.max_records_in_memory
      << ",\"max_records_on_disk\":" << cfg.max_records_on_disk
      << ",\"retention_days\":" << cfg.retention_days
      << ",\"metrics_file_max_mb\":" << cfg.metrics_file_max_mb
      << ",\"metrics_file_backups\":" << cfg.metrics_file_backups
      << ",\"enable_prometheus\":" << (cfg.enable_prometheus_format ? "true" : "false")
      << ",\"staleness_threshold_seconds\":" << cfg.staleness_threshold_seconds
      << ",\"allow_control\":" << (allow_control_ ? "true" : "false") << "}";
    return j.str();
}

std::pair<int, std::string> TrainingMetricsAPI::handle_admin_put_config(const std::string& body) {
    static const char* kImmutableKeys[] = {"port",       "storage_backend", "db_path",
                                           "db_url",     "db_pool_size",    "metrics_file",
                                           "summary_file", "prometheus_file", "allow_control"};
    for (const auto* key : kImmutableKeys) {
        if (json_has_key(body, key)) {
            return {400,
                    std::string("{\"error\":\"'") + key +
                        "' is immutable at runtime; set it via config.metrics.conf or the "
                        "matching --flag and restart\"}"};
        }
    }

    bool changed = false;

    // TODO: See TD-306 in TECHNICAL_DEBT.md - no range checks on these tunables (negative max_live_sessions becomes a huge size_t)
    if (auto v = json_int_value(body, "max_live_sessions")) {
        session_registry_->set_max_live_sessions(static_cast<size_t>(*v));
        changed = true;
    }
    if (auto v = json_int_value(body, "completed_ttl_seconds")) {
        session_registry_->set_completed_ttl_seconds(static_cast<int>(*v));
        changed = true;
    }
    if (auto v = json_int_value(body, "sweep_interval_seconds")) {
        session_registry_->set_sweep_interval_seconds(static_cast<int>(*v));
        changed = true;
    }
    if (auto v = json_int_value(body, "persist_every_samples")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.persist_every_samples = n; });
        changed = true;
    }
    if (auto v = json_int_value(body, "persist_every_seconds")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.persist_every_seconds = n; });
        changed = true;
    }
    if (auto v = json_int_value(body, "max_records_in_memory")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.max_records_in_memory = n; });
        changed = true;
    }
    if (auto v = json_int_value(body, "max_records_on_disk")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.max_records_on_disk = n; });
        changed = true;
    }
    // TD-440: retention settings, applied to future persists and the next retention sweep.
    if (auto v = json_int_value(body, "retention_days")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.retention_days = n; });
        changed = true;
    }
    if (auto v = json_int_value(body, "metrics_file_max_mb")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.metrics_file_max_mb = n; });
        changed = true;
    }
    if (auto v = json_int_value(body, "metrics_file_backups")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.metrics_file_backups = n; });
        changed = true;
    }
    if (auto v = json_bool_value(body, "enable_prometheus")) {
        const bool b = *v;
        session_registry_->update_metrics_config(
            [b](MetricsServiceConfig& c) { c.enable_prometheus_format = b; });
        changed = true;
    }
    if (auto v = json_int_value(body, "staleness_threshold_seconds")) {
        const int n = static_cast<int>(*v);
        session_registry_->update_metrics_config(
            [n](MetricsServiceConfig& c) { c.staleness_threshold_seconds = n; });
        changed = true;
    }

    if (!changed) {
        return {400,
                "{\"error\":\"no recognized mutable keys in body (max_live_sessions, "
                "completed_ttl_seconds, sweep_interval_seconds, persist_every_samples, "
                "persist_every_seconds, max_records_in_memory, max_records_on_disk, "
                "retention_days, metrics_file_max_mb, metrics_file_backups, "
                "enable_prometheus, staleness_threshold_seconds)\"}"};
    }

    if (config_store_) {
        const auto cfg = session_registry_->base_metrics_config();
        // TODO: See TD-305 in TECHNICAL_DEBT.md - per-key set() with no transaction; memory already updated
        config_store_->set("max_live_sessions",
                           std::to_string(session_registry_->max_live_sessions()));
        config_store_->set("completed_ttl_seconds",
                           std::to_string(session_registry_->completed_ttl_seconds()));
        config_store_->set("sweep_interval_seconds",
                           std::to_string(session_registry_->sweep_interval_seconds()));
        config_store_->set("persist_every_samples", std::to_string(cfg.persist_every_samples));
        config_store_->set("persist_every_seconds", std::to_string(cfg.persist_every_seconds));
        config_store_->set("max_records_in_memory", std::to_string(cfg.max_records_in_memory));
        config_store_->set("max_records_on_disk", std::to_string(cfg.max_records_on_disk));
        config_store_->set("retention_days", std::to_string(cfg.retention_days));
        config_store_->set("metrics_file_max_mb", std::to_string(cfg.metrics_file_max_mb));
        config_store_->set("metrics_file_backups", std::to_string(cfg.metrics_file_backups));
        config_store_->set("enable_prometheus", cfg.enable_prometheus_format ? "true" : "false");
        config_store_->set("staleness_threshold_seconds",
                           std::to_string(cfg.staleness_threshold_seconds));
    }

    return {200, handle_admin_get_config()};
}

std::string TrainingMetricsAPI::handle_models_list() {
    if (name_service_url_.empty()) {
        return R"({"error":"name service not configured","models":[]})";
    }

    // Parse host:port from the configured URL
    std::string host = "localhost";
    int port = 8083;
    std::string url = name_service_url_;
    if (url.rfind("http://", 0) == 0)
        url = url.substr(7);
    else if (url.rfind("https://", 0) == 0)
        url = url.substr(8);
    auto colon = url.find(':');
    if (colon != std::string::npos) {
        host = url.substr(0, colon);
        auto slash = url.find('/', colon);
        std::string port_str = (slash != std::string::npos)
                                   ? url.substr(colon + 1, slash - colon - 1)
                                   : url.substr(colon + 1);
        try {
            port = std::stoi(port_str);
        } catch (...) {
        }
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
    if (!is_valid_session_key(session_key)) {
        throw ApiRequestError(400, "Invalid session key format");
    }

    // Body: {"session_id": int, "total_epochs": int, "total_samples": int, "label": string,
    //        "config": object, "model_id": string, "reset_best": bool}, every field optional.
    const PushBody b(body);
    const int session_id = b.get_int("session_id", 0);
    const int total_epochs = b.get_int("total_epochs", 0);
    const int total_samples = b.get_int("total_samples", 0);
    const std::string label = b.get_string("label");
    nlohmann::json config = b.get_object("config");
    // Phase 2: an MNS model UUID is folded into the config snapshot.
    const std::string model_id = b.get_string("model_id");
    if (!model_id.empty()) {
        if (!config.is_object())
            config = nlohmann::json::object();
        config["model_id"] = model_id;
    }
    const std::string config_snapshot = config.is_object() ? config.dump() : std::string();
    // A fresh retrain sets reset_best so best_validation_loss/best_epoch don't carry forward from a
    // prior session.
    const bool reset_best = b.get_bool("reset_best", false);

    // The body is parsed before the key is claimed below: a claimed session refuses other starts
    // until it has started (MetricsSessionRegistry::SessionEntry::claimed), so a malformed body
    // must not leave a claim behind.
    // The conflict check and the reclaim-if-stale decision must happen atomically under one
    // registry lock (see MetricsSessionRegistry::start_session_or_conflict) — otherwise a
    // session that is still genuinely training can be silently archived and replaced out from
    // under itself before a separate "is it already active?" check gets to run.
    auto outcome = session_registry_->start_session_or_conflict(session_key);
    if (outcome.conflict) {
        throw ApiRequestError(409, "session already active for key: " + session_key);
    }
    auto service = outcome.service;
    if (!service) {
        std::ostringstream body_err;
        body_err << R"({"error":"metrics_server_full","max_live_sessions":)"
                 << session_registry_->max_live_sessions() << "}";
        throw ApiRequestError(503, body_err.str());
    }

    service->start_session(session_id, total_epochs, total_samples, label, config_snapshot,
                           reset_best);

    return R"({"status":"ok","message":"Session started"})";
}

std::string TrainingMetricsAPI::handle_post_session_end(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    service->end_session();
    return R"({"status":"ok","message":"Session ended"})";
}

std::string TrainingMetricsAPI::handle_post_heartbeat(const std::string& session_key) {
    auto service = resolve_session_service(session_key, false);
    service->heartbeat();
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_epoch_start(const std::string& session_key,
                                                        const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    const PushBody b(body);  // {"epoch": int, "total_samples": int}
    service->start_epoch(b.get_int("epoch", 0), b.get_int("total_samples", 0));
    return R"({"status":"ok","message":"Epoch started"})";
}

std::string TrainingMetricsAPI::handle_post_epoch_end(const std::string& session_key,
                                                      const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // Body: {"epoch", "loss", "validation_loss", "learning_rate", "perplexity", "gradient_norm",
    // "epoch_time"} plus the optional per-epoch diagnostics below (-1 = not sent).
    const PushBody b(body);
    const int epoch = b.get_int("epoch", 0);
    const float loss = b.get_float("loss", 0.0f);
    const float validation_loss = b.get_float("validation_loss", 0.0f);
    const float learning_rate = b.get_float("learning_rate", 0.0f);
    const float perplexity = b.get_float("perplexity", 0.0f);
    const float gradient_norm = b.get_float("gradient_norm", 0.0f);
    const double epoch_time = b.get_double("epoch_time", 0.0);
    const float gradient_variance = b.get_float("gradient_variance", 0.0f);
    const float compute_time_ratio = b.get_float("compute_time_ratio", 0.0f);
    const float weight_update_ratio = b.get_float("weight_update_ratio", 0.0f);
    const float activation_saturation_ratio = b.get_float("activation_saturation_ratio", -1.0f);
    const float attention_entropy = b.get_float("attention_entropy", -1.0f);
    const float current_padding_efficiency = b.get_float("current_padding_efficiency", -1.0f);
    // TD-178: LeJEPA predictor/SIGReg losses, only for --objective=lejepa passes, sent as a pair.
    const float predictor_loss_field = b.get_float("predictor_loss", -1.0f);
    const float sigreg_loss_field = b.get_float("sigreg_loss", -1.0f);
    // LeJEPA advanced diagnostics, sent once update_lejepa_advanced_metrics() ran this epoch.
    const float masking_ratio_field = b.get_float("masking_ratio", -1.0f);
    const float predictor_target_cosine_sim_field =
        b.get_float("predictor_target_cosine_sim", -1.0f);
    const float sigreg_variance_mean_field = b.get_float("sigreg_variance_mean", -1.0f);
    const float sigreg_variance_stddev_field = b.get_float("sigreg_variance_stddev", -1.0f);
    // TD-443: adaptive gradient clipping's epoch average and spike count.
    const float adaptive_clip_threshold = b.get_float("adaptive_clip_threshold", -1.0f);
    const int adaptive_clip_spikes = b.get_int("adaptive_clip_spikes", 0);

    // These optional per-epoch fields must be applied BEFORE end_epoch() — end_epoch() both
    // (a) pushes the CURRENT snapshot value into that field's history vector and (b) reads it
    // into the PersistentMetricsRecord it persists, so calling it first would push/persist last
    // epoch's stale value instead of the one just POSTed in this same request body (caught by
    // TrainingMetricsAPIRoutesTest.LejepaMetricsRoundTripThroughEpochEndAndGet).
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
    // Gate on predictor_loss alone (non-negative; -1 = "not sent"): the two always arrive together,
    // and since TD-503 sigreg_loss is an unbiased estimate that can be slightly negative, so a
    // sigreg_loss >= 0 check would drop real epochs.
    if (predictor_loss_field >= 0.0f) {
        service->update_lejepa_metrics(predictor_loss_field, sigreg_loss_field);
    }
    // Gate on masking_ratio alone: it is strictly positive whenever computed (span_len >= 1), unlike
    // predictor_target_cosine_sim, whose valid range [-1, 1] includes the -1 sentinel. All four are
    // sent together from one train_step(), so masking_ratio's presence means the whole bundle's.
    if (masking_ratio_field >= 0.0f) {
        service->update_lejepa_advanced_metrics(masking_ratio_field,
                                                predictor_target_cosine_sim_field,
                                                sigreg_variance_mean_field,
                                                sigreg_variance_stddev_field);
    }
    if (adaptive_clip_threshold >= 0.0f) {
        service->update_adaptive_clip_epoch(adaptive_clip_threshold, adaptive_clip_spikes);
    }

    service->end_epoch(epoch, loss, validation_loss, learning_rate, perplexity, gradient_norm,
                       epoch_time);

    return R"({"status":"ok","message":"Epoch ended"})";
}

std::string TrainingMetricsAPI::handle_post_sample_metrics(const std::string& session_key,
                                                           const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // {"sample": int, "loss": float, "gradient_norm": float, "learning_rate": float}
    const PushBody b(body);
    service->update_sample_metrics(b.get_int("sample", 0), b.get_float("loss", 0.0f),
                                   b.get_float("gradient_norm", 0.0f),
                                   b.get_float("learning_rate", 0.0f));
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_validation_metrics(const std::string& session_key,
                                                               const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // {"validation_loss": float, "validation_accuracy": float (TD-015: optional, -1 = not
    // computed), "validation_perplexity": float (optional, 0 = derive from the loss)}
    const PushBody b(body);
    service->update_validation_metrics(b.get_float("validation_loss", 0.0f),
                                       b.get_float("validation_accuracy", -1.0f),
                                       b.get_float("validation_perplexity", 0.0f));
    return R"({"status":"ok"})";
}

namespace {

// TD-555/TD-399: abnormal-sample texts are whole training inputs/targets; keep what is stored and
// served bounded. Cut on a UTF-8 character boundary and mark the cut.
std::string truncate_sample_text(std::string text) {
    constexpr std::size_t kMaxBytes = 4096;
    if (text.size() <= kMaxBytes)
        return text;
    std::size_t cut = kMaxBytes;
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80)
        --cut;
    text.resize(cut);
    text += "\u2026";
    return text;
}

}  // namespace

std::string TrainingMetricsAPI::handle_post_abnormal_sample(const std::string& session_key,
                                                            const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    const PushBody b(body);
    AbnormalSample sample;
    sample.epoch = b.get_int("epoch", 0);
    sample.sample_id = b.get_int("sample_id", 0);
    sample.loss = b.get_float("loss", 0.0f);
    sample.grad_norm = b.get_float("grad_norm", 0.0f);
    sample.reason = b.get_string("reason").substr(0, 128);
    sample.input_text = truncate_sample_text(b.get_string("input_text"));
    sample.target_text = truncate_sample_text(b.get_string("target_text"));
    sample.timestamp = std::chrono::system_clock::now();
    service->flag_abnormal_sample(sample);
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_best_metrics(const std::string& session_key,
                                                         const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    const PushBody b(body);  // {"validation_loss": float, "epoch": int}
    service->update_best_metrics(b.get_float("validation_loss", 0.0f), b.get_int("epoch", 0));
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_generation_quality_metrics(
    const std::string& session_key, const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // {"bleu4", "rouge1", "rouge2", "rougeL", "bleu1", "bleu2"}: floats, -1 when absent (TD-379:
    // bleu1/bleu2 are optional)
    const PushBody b(body);
    service->update_generation_quality_metrics(
        b.get_float("bleu4", -1.0f), b.get_float("rouge1", -1.0f), b.get_float("rouge2", -1.0f),
        b.get_float("rougeL", -1.0f), b.get_float("bleu1", -1.0f), b.get_float("bleu2", -1.0f));
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_advanced_metrics(const std::string& session_key,
                                                             const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // {"gradient_variance": float, "compute_time_ratio": float, "weight_update_ratio": float}
    const PushBody b(body);
    service->update_advanced_epoch_metrics(b.get_float("gradient_variance", 0.0f),
                                           b.get_float("compute_time_ratio", 0.0f),
                                           b.get_float("weight_update_ratio", 0.0f));
    return R"({"status":"ok"})";
}

std::string TrainingMetricsAPI::handle_post_layer_gradient_norms(const std::string& session_key,
                                                                  const std::string& body) {
    auto service = resolve_session_service(session_key, false);
    // {"encoder_layer_grad_norms": [float, ...], "decoder_layer_grad_norms": [float, ...]}
    const PushBody b(body);
    service->update_layer_gradient_norms(b.get_float_array("encoder_layer_grad_norms"),
                                         b.get_float_array("decoder_layer_grad_norms"));
    return R"({"status":"ok"})";
}

// ============================================================================
// TD-020: DB-backed Query Endpoints
// ============================================================================

std::string TrainingMetricsAPI::parse_query_param_string(const std::string& query,
                                                         const std::string& param,
                                                         const std::string& default_value) {
    std::string search_key = param + "=";
    size_t pos = query.find(search_key);
    if (pos == std::string::npos)
        return default_value;
    pos += search_key.length();
    size_t end_pos = query.find('&', pos);
    return end_pos == std::string::npos ? query.substr(pos) : query.substr(pos, end_pos - pos);
}

static std::chrono::system_clock::time_point parse_iso8601(const std::string& s) {
    if (s.empty())
        return {};
    std::tm tm{};
    int ms = 0;
    // TODO: See TD-477 in TECHNICAL_DEBT.md - parse failure falls through to 1899; atoi() misreads fractions (.12 -> 12 ms); offsets ignored
    if (auto* end = adai::strptime_utc(s.c_str(), 'T', &tm)) {
        if (*end == '.')
            ms = std::atoi(end + 1);
    }
    auto tp = std::chrono::system_clock::from_time_t(adai::timegm_utc(&tm));
    tp += std::chrono::milliseconds(ms);
    return tp;
}

static std::string format_iso8601(const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    adai::gmtime_utc(&time_t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
    oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z";
    return oss.str();
}

std::string TrainingMetricsAPI::handle_db_history(const std::string& session_key,
                                                  const std::string& query_params) {
    auto* db = session_registry_->get_database();
    if (!db) {
        return create_error_response("No database backend configured");
    }

    std::optional<std::chrono::system_clock::time_point> from, to;
    auto from_str = parse_query_param_string(query_params, "from");
    auto to_str = parse_query_param_string(query_params, "to");
    int limit = parse_query_param_int(query_params, "limit", 0);

    if (!from_str.empty())
        from = parse_iso8601(from_str);
    if (!to_str.empty())
        to = parse_iso8601(to_str);

    auto records = db->query_history(session_key, from, to, limit);

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\"session_key\":\"" << escape_json(session_key) << "\",";
    json << "\"count\":" << records.size() << ",";
    json << "\"records\":[";
    for (size_t i = 0; i < records.size(); ++i) {
        if (i > 0)
            json << ",";
        const auto& r = records[i];
        json << "{\"timestamp\":\"" << format_iso8601(r.timestamp) << "\",";
        json << "\"epoch\":" << r.epoch << ",";
        json << "\"sample\":" << r.sample << ",";
        json << "\"loss\":" << r.loss << ",";
        json << "\"validation_loss\":" << r.validation_loss << ",";
        json << "\"learning_rate\":" << r.learning_rate << ",";
        json << "\"gradient_norm\":" << r.gradient_norm << ",";
        json << "\"perplexity\":" << r.perplexity << "}";
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_db_gradient_variance(const std::string& session_key,
                                                            const std::string& query_params) {
    auto* db = session_registry_->get_database();
    if (!db) {
        return create_error_response("No database backend configured");
    }
    const int limit = parse_query_param_int(query_params, "limit", 0);
    const auto records = db->query_gradient_variance(session_key, limit);

    std::ostringstream json;
    json << std::setprecision(std::numeric_limits<float>::max_digits10);
    json << "{\"session_key\":\"" << escape_json(session_key) << "\",";
    json << "\"count\":" << records.size() << ",\"records\":[";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        json << (i > 0 ? "," : "") << "{\"timestamp\":\"" << format_iso8601(r.timestamp)
             << "\",\"step\":" << r.step << ",\"epoch\":" << r.epoch
             << ",\"value\":" << r.value << "}";
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_db_generation_quality(const std::string& session_key) {
    auto* db = session_registry_->get_database();
    if (!db) {
        return create_error_response("No database backend configured");
    }
    const auto records = db->query_generation_quality(session_key);

    std::ostringstream json;
    json << std::setprecision(std::numeric_limits<float>::max_digits10);
    json << "{\"session_key\":\"" << escape_json(session_key) << "\",";
    json << "\"count\":" << records.size() << ",\"records\":[";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        json << (i > 0 ? "," : "") << "{\"timestamp\":\"" << format_iso8601(r.timestamp)
             << "\",\"epoch\":" << r.epoch << ",\"bleu1\":" << r.bleu1
             << ",\"bleu2\":" << r.bleu2 << ",\"bleu4\":" << r.bleu4
             << ",\"rouge1\":" << r.rouge1 << ",\"rouge2\":" << r.rouge2
             << ",\"rougeL\":" << r.rougeL << "}";
    }
    json << "]}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_metrics_compare(const std::string& query_params) {
    auto* db = session_registry_->get_database();
    if (!db) {
        return create_error_response("No database backend configured");
    }

    auto keys_str = parse_query_param_string(query_params, "keys");
    auto metric = parse_query_param_string(query_params, "metric", "loss");

    if (keys_str.empty()) {
        return create_error_response("Missing required parameter: keys");
    }

    // Parse comma-separated keys
    std::vector<std::string> keys;
    std::istringstream ks(keys_str);
    std::string k;
    while (std::getline(ks, k, ',')) {
        if (!k.empty())
            keys.push_back(k);
    }
    // TD-441: compare used to read every listed session's whole history per request.
    if (keys.size() > kMaxCompareSessions) {
        return create_error_response("At most " + std::to_string(kMaxCompareSessions) +
                                     " sessions can be compared at once");
    }

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\"metric\":\"" << escape_json(metric) << "\",\"sessions\":{";

    for (size_t i = 0; i < keys.size(); ++i) {
        if (i > 0)
            json << ",";
        json << "\"" << escape_json(keys[i]) << "\":[";

        // TD-441: the newest kMaxReadRows rows per session (the comparison keeps each epoch's last
        // value, so the newest rows are the ones that matter).
        auto records = db->query_history(keys[i], std::nullopt, std::nullopt, kMaxReadRows);
        std::map<int, float> epoch_values;
        for (const auto& r : records) {
            float val = 0.0f;
            if (metric == "loss")
                val = r.loss;
            else if (metric == "validation_loss")
                val = r.validation_loss;
            else if (metric == "learning_rate")
                val = r.learning_rate;
            else if (metric == "gradient_norm")
                val = r.gradient_norm;
            else if (metric == "perplexity")
                val = r.perplexity;
            else
                val = r.loss;
            epoch_values[r.epoch] = val;
        }

        bool first = true;
        for (const auto& [epoch, val] : epoch_values) {
            if (!first)
                json << ",";
            first = false;
            json << val;
        }
        json << "]";
    }
    json << "}}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_sessions_list_filtered(const std::string& query_params) {
    auto status_str = parse_query_param_string(query_params, "status");
    auto from_str = parse_query_param_string(query_params, "from");

    // If no DB-specific filters, delegate to standard list
    if (status_str.empty() && from_str.empty()) {
        return handle_sessions_list();
    }

    auto* db = session_registry_->get_database();
    if (!db) {
        return handle_sessions_list();
    }

    std::optional<bool> is_training_filter;
    if (status_str == "completed")
        is_training_filter = false;
    else if (status_str == "active" || status_str == "training")
        is_training_filter = true;

    auto db_sessions = db->list_sessions(is_training_filter);

    std::optional<std::chrono::system_clock::time_point> from_tp;
    if (!from_str.empty())
        from_tp = parse_iso8601(from_str);

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\"sessions\":[";
    size_t count = 0;
    for (const auto& rec : db_sessions) {
        if (from_tp && rec.created_at < *from_tp)
            continue;
        if (count > 0)
            json << ",";
        json << "{\"key\":\"" << escape_json(rec.key) << "\",";
        json << "\"session_id\":" << rec.session_id << ",";
        json << "\"label\":\"" << escape_json(rec.label) << "\",";
        json << "\"is_training\":" << (rec.is_training ? "true" : "false") << ",";
        json << "\"total_epochs\":" << rec.total_epochs << ",";
        json << "\"total_samples\":" << rec.total_samples << ",";
        json << "\"best_validation_loss\":" << rec.best_validation_loss << ",";
        json << "\"best_epoch\":" << rec.best_epoch << ",";
        json << "\"created_at\":\"" << format_iso8601(rec.created_at) << "\",";
        json << "\"last_update_at\":\"" << format_iso8601(rec.last_update_at) << "\"";
        if (rec.ended_at) {
            json << ",\"ended_at\":\"" << format_iso8601(*rec.ended_at) << "\"";
        }
        json << "}";
        ++count;
    }
    json << "],\"total\":" << count << "}";
    return json.str();
}

std::string TrainingMetricsAPI::handle_metrics_export(const std::string& session_key,
                                                      const std::string& query_params) {
    auto* db = session_registry_->get_database();
    if (!db) {
        return create_error_response("No database backend configured");
    }

    auto format = parse_query_param_string(query_params, "format", "json");
    // TD-441: bounded to the newest kMaxReadRows rows (it read the whole history per request). One
    // extra row is fetched to tell whether anything was left out; JSON reports it as "truncated".
    auto records = db->query_history(session_key, std::nullopt, std::nullopt, kMaxReadRows + 1);
    const bool truncated = records.size() > static_cast<size_t>(kMaxReadRows);
    if (truncated)
        records.erase(records.begin());

    if (format == "csv") {
        std::ostringstream csv;
        csv << "timestamp,epoch,sample,loss,validation_loss,learning_rate,gradient_norm,"
               "perplexity\n";
        csv << std::fixed << std::setprecision(6);
        for (const auto& r : records) {
            csv << format_iso8601(r.timestamp) << "," << r.epoch << "," << r.sample << "," << r.loss
                << "," << r.validation_loss << "," << r.learning_rate << "," << r.gradient_norm
                << "," << r.perplexity << "\n";
        }
        return csv.str();
    }

    // JSON format
    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\"session_key\":\"" << escape_json(session_key) << "\",";
    json << "\"count\":" << records.size() << ",";
    json << "\"truncated\":" << (truncated ? "true" : "false") << ",";
    json << "\"records\":[";
    for (size_t i = 0; i < records.size(); ++i) {
        if (i > 0)
            json << ",";
        const auto& r = records[i];
        json << "{\"timestamp\":\"" << format_iso8601(r.timestamp) << "\",";
        json << "\"epoch\":" << r.epoch << ",";
        json << "\"sample\":" << r.sample << ",";
        json << "\"loss\":" << r.loss << ",";
        json << "\"validation_loss\":" << r.validation_loss << ",";
        json << "\"learning_rate\":" << r.learning_rate << ",";
        json << "\"gradient_norm\":" << r.gradient_norm << ",";
        json << "\"perplexity\":" << r.perplexity << "}";
    }
    json << "]}";
    return json.str();
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

    return std::all_of(key.begin(), key.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; });
}
