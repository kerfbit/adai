// @adai-status: beta        (tested only indirectly via mns_manager_gui_test.cpp)
// @adai-version: 0.8.0
// @adai-reviewed: 2026-09-07

#include "ModelNameClient.hpp"
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "Logger.hpp"

#ifdef BUILD_MNS_SERVER
#include <httplib.h>
#endif

// ============================================================================
// URL parsing (same pattern as MetricsPushClient::ParsedUrl)
// ============================================================================

adai::ModelNameClient::ParsedUrl adai::ModelNameClient::ParsedUrl::from(const std::string& url) {
    ParsedUrl r;
    std::string s = url;

    // Strip scheme
    const std::string http_prefix = "http://";
    const std::string https_prefix = "https://";
    if (s.rfind(http_prefix, 0) == 0) {
        s = s.substr(http_prefix.size());
    } else if (s.rfind(https_prefix, 0) == 0) {
        s = s.substr(https_prefix.size());
        r.port = 443;
    }

    const auto slash = s.find('/');
    std::string authority;
    if (slash != std::string::npos) {
        authority = s.substr(0, slash);
        r.base_path = s.substr(slash);
    } else {
        authority = s;
    }
    while (!r.base_path.empty() && r.base_path.back() == '/')
        r.base_path.pop_back();

    const auto colon = authority.find(':');
    if (colon != std::string::npos) {
        r.host = authority.substr(0, colon);
        try {
            r.port = std::stoi(authority.substr(colon + 1));
        } catch (...) {
        }
    } else {
        r.host = authority;
    }
    return r;
}

// ============================================================================
// Construction
// ============================================================================

adai::ModelNameClient::ModelNameClient(std::string server_url, int timeout_ms)
    : server_url_(std::move(server_url)),
      timeout_ms_(timeout_ms),
      parsed_(ParsedUrl::from(server_url_)) {}

// ============================================================================
// Internal HTTP helpers
// ============================================================================

namespace {

std::string json_escape_client(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += static_cast<char>(c);
                break;
        }
    }
    return out;
}

std::string json_string_client(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return {};
    const auto start = pos + needle.size();
    const auto end = body.find('"', start);
    if (end == std::string::npos)
        return {};
    return body.substr(start, end - start);
}

size_t json_int_client(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const auto pos = body.find(needle);
    if (pos == std::string::npos)
        return 0;
    try {
        return static_cast<size_t>(std::stoull(body.substr(pos + needle.size())));
    } catch (...) {
        return 0;
    }
}

}  // namespace

#ifdef BUILD_MNS_SERVER

static constexpr int kBackoffMs[] = {0, 200, 1000};
static constexpr int kMaxAttempts = 3;

static httplib::Client make_client(const std::string& host, int port, int timeout_ms) {
    httplib::Client c(host, port);
    c.set_connection_timeout(0, static_cast<long>(timeout_ms) * 1000);
    c.set_read_timeout(timeout_ms / 1000, static_cast<long>(timeout_ms % 1000) * 1000);
    c.set_write_timeout(timeout_ms / 1000, static_cast<long>(timeout_ms % 1000) * 1000);
    return c;
}

int adai::ModelNameClient::http_post(const std::string& path, const std::string& body,
                                     std::string& out) const {
    const std::string full_path = parsed_.base_path + path;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
        try {
            auto c = make_client(parsed_.host, parsed_.port, timeout_ms_);
            auto res = c.Post(full_path, body, "application/json");
            if (!res)
                continue;
            out = res->body;
            if (res->status >= 500)
                continue;
            return res->status;
        } catch (const std::exception& e) {
            adai::Logger::debug("ModelNameClient: POST {} exception: {}", path, e.what());
        }
    }
    return 0;
}

int adai::ModelNameClient::http_get(const std::string& path, std::string& out) const {
    const std::string full_path = parsed_.base_path + path;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
        try {
            auto c = make_client(parsed_.host, parsed_.port, timeout_ms_);
            auto res = c.Get(full_path);
            if (!res)
                continue;
            out = res->body;
            if (res->status >= 500)
                continue;
            return res->status;
        } catch (const std::exception& e) {
            adai::Logger::debug("ModelNameClient: GET {} exception: {}", path, e.what());
        }
    }
    return 0;
}

int adai::ModelNameClient::http_put(const std::string& path, const std::string& body,
                                    std::string& out) const {
    const std::string full_path = parsed_.base_path + path;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (attempt > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
        try {
            auto c = make_client(parsed_.host, parsed_.port, timeout_ms_);
            auto res = c.Put(full_path, body, "application/json");
            if (!res)
                continue;
            out = res->body;
            if (res->status >= 500)
                continue;
            return res->status;
        } catch (const std::exception& e) {
            adai::Logger::debug("ModelNameClient: PUT {} exception: {}", path, e.what());
        }
    }
    return 0;
}

#else  // BUILD_MNS_SERVER not defined — no-op stubs

int adai::ModelNameClient::http_post(const std::string&, const std::string&, std::string&) const {
    return 0;
}
int adai::ModelNameClient::http_get(const std::string&, std::string&) const {
    return 0;
}
int adai::ModelNameClient::http_put(const std::string&, const std::string&, std::string&) const {
    return 0;
}

#endif  // BUILD_MNS_SERVER

// ============================================================================
// Error helper
// ============================================================================

void adai::ModelNameClient::check_status(int status, const std::string& out,
                                         const std::string& op) {
    if (status == 0)
        throw std::runtime_error("ModelNameClient: " + op + " — connection failed");
    if (status < 200 || status >= 300)
        throw std::runtime_error("ModelNameClient: " + op + " returned HTTP " +
                                 std::to_string(status) + ": " + out);
}

// ============================================================================
// Public API
// ============================================================================

std::string adai::ModelNameClient::register_model(const std::string& model_name,
                                                  const std::string& role,
                                                  const ServiceConfig& arch,
                                                  const std::map<std::string, std::string>& tags) {
    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape_client(model_name) << "\"" << ",\"role\":\""
         << json_escape_client(role) << "\"" << ",\"run_group\":\""
         << json_escape_client(arch.run_group) << "\"" << ",\"arch\":{" << "\"d_model\":" << arch.d_model
         << ",\"num_heads\":" << arch.num_heads << ",\"d_ff\":" << arch.d_ff
         << ",\"num_encoder_layers\":" << arch.num_encoder_layers
         << ",\"num_decoder_layers\":" << arch.num_decoder_layers
         << ",\"max_seq_length\":" << arch.max_seq_length << "},\"tags\":{";
    bool first = true;
    for (const auto& [k, v] : tags) {
        if (!first)
            body << ',';
        first = false;
        body << '"' << json_escape_client(k) << "\":\"" << json_escape_client(v) << '"';
    }
    body << "}}";

    std::string out;
    const int status = http_post("/models", body.str(), out);
    check_status(status, out, "register_model");
    return json_string_client(out, "model_id");
}

std::string adai::ModelNameClient::set_training(const std::string& model_name, bool new_run,
                                                const std::string& metrics_session_key) {
    std::ostringstream body;
    body << "{\"state\":\"training\"" << ",\"new_run\":" << (new_run ? "true" : "false");
    if (!metrics_session_key.empty())
        body << ",\"metrics_session_key\":\"" << json_escape_client(metrics_session_key) << "\"";
    body << "}";

    std::string out;
    const int status = http_put("/models/" + model_name + "/state", body.str(), out);
    check_status(status, out, "set_training(" + model_name + ")");
    return json_string_client(out, "run_id");
}

void adai::ModelNameClient::set_candidate(
    const std::string& model_name, const std::string& run_id, const ArtifactLocation& artifact,
    const std::map<std::string, std::string>& training_summary) {
    std::ostringstream body;
    body << "{\"state\":\"candidate\"" << ",\"run_id\":\"" << json_escape_client(run_id) << "\""
         << ",\"artifact\":{" << "\"host\":\"" << json_escape_client(artifact.host) << "\""
         << ",\"path\":\"" << json_escape_client(artifact.path) << "\"" << ",\"checksum\":\""
         << json_escape_client(artifact.checksum) << "\"" << ",\"format\":\""
         << json_escape_client(artifact.format) << "\"" << "}";
    if (!training_summary.empty()) {
        body << ",\"training_summary\":{";
        bool first = true;
        for (const auto& [k, v] : training_summary) {
            if (!first)
                body << ',';
            first = false;
            body << '"' << json_escape_client(k) << "\":\"" << json_escape_client(v) << '"';
        }
        body << "}";
    }
    body << "}";

    std::string out;
    const int status = http_put("/models/" + model_name + "/state", body.str(), out);
    check_status(status, out, "set_candidate(" + model_name + ")");
}

void adai::ModelNameClient::push_progress(const std::string& model_name, const std::string& run_id,
                                          const std::string& session_id, int epoch, double loss,
                                          double best_loss) {
    std::ostringstream body;
    body << "{\"run_id\":\"" << json_escape_client(run_id) << "\"" << ",\"session_id\":\""
         << json_escape_client(session_id) << "\"" << ",\"epoch\":" << epoch << ",\"loss\":" << loss
         << ",\"best_loss\":" << best_loss << "}";

    std::string out;
    const int status = http_put("/models/" + model_name + "/progress", body.str(), out);
    check_status(status, out, "push_progress(" + model_name + ")");
}

std::vector<adai::ModelSummary> adai::ModelNameClient::list_models(const std::string& state_filter,
                                                                   const std::string& role_filter,
                                                                   int limit) {
    std::string path = "/models?limit=" + std::to_string(limit);
    if (!state_filter.empty())
        path += "&state=" + state_filter;
    if (!role_filter.empty())
        path += "&role=" + role_filter;

    std::string out;
    const int status = http_get(path, out);
    check_status(status, out, "list_models");

    std::vector<ModelSummary> result;
    const std::string needle = "{\"model_id\":";
    size_t pos = 0;
    while ((pos = out.find(needle, pos)) != std::string::npos) {
        size_t end = out.find("}}", pos);
        if (end == std::string::npos)
            break;
        end += 2;
        std::string record = out.substr(pos, end - pos);
        ModelSummary ms;
        ms.model_name = json_string_client(record, "model_name");
        ms.state = json_string_client(record, "state");
        ms.role = json_string_client(record, "role");
        ms.updated_utc = json_string_client(record, "updated_utc");
        result.push_back(std::move(ms));
        pos = end;
    }
    return result;
}

adai::ResolvedModel adai::ModelNameClient::resolve_model(const std::string& model_name) {
    std::string out;
    const int status = http_get("/models/" + model_name + "/resolve", out);
    check_status(status, out, "resolve_model(" + model_name + ")");

    ResolvedModel rm;
    rm.model_id = json_string_client(out, "model_id");
    rm.model_name = json_string_client(out, "model_name");
    rm.state = json_string_client(out, "state");
    rm.run_group = json_string_client(out, "run_group");
    rm.artifact.host = json_string_client(out, "host");
    rm.artifact.path = json_string_client(out, "path");
    rm.artifact.checksum = json_string_client(out, "checksum");
    rm.artifact.format = json_string_client(out, "format");
    return rm;
}

std::optional<adai::ModelArchitecture> adai::ModelNameClient::get_architecture(
    const std::string& model_name) {
    std::string out;
    const int status = http_get("/models/" + model_name, out);
    if (status == 404) {
        return std::nullopt;
    }
    check_status(status, out, "get_architecture(" + model_name + ")");

    const auto arch_pos = out.find("\"arch\":{");
    if (arch_pos == std::string::npos) {
        return std::nullopt;
    }
    const std::string arch = out.substr(arch_pos);

    ModelArchitecture a;
    a.d_model = json_int_client(arch, "d_model");
    a.num_heads = json_int_client(arch, "num_heads");
    a.d_ff = json_int_client(arch, "d_ff");
    a.num_encoder_layers = json_int_client(arch, "num_encoder_layers");
    a.num_decoder_layers = json_int_client(arch, "num_decoder_layers");
    a.max_seq_length = json_int_client(arch, "max_seq_length");
    return a;
}

adai::ResolvedModel adai::ModelNameClient::resolve_role(const std::string& role) {
    std::string out;
    const int status = http_get("/roles/" + role + "/production", out);
    check_status(status, out, "resolve_role(" + role + ")");

    ResolvedModel rm;
    rm.model_id = json_string_client(out, "model_id");
    rm.model_name = json_string_client(out, "model_name");
    rm.state = json_string_client(out, "state");
    rm.run_group = json_string_client(out, "run_group");
    rm.artifact.host = json_string_client(out, "host");
    rm.artifact.path = json_string_client(out, "path");
    rm.artifact.checksum = json_string_client(out, "checksum");
    rm.artifact.format = json_string_client(out, "format");
    return rm;
}

void adai::ModelNameClient::promote(const std::string& role, const std::string& model_name) {
    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape_client(model_name) << "\"}";

    std::string out;
    const int status = http_put("/roles/" + role + "/production", body.str(), out);
    check_status(status, out, "promote(" + model_name + " -> " + role + ")");
}

void adai::ModelNameClient::update_run_group(const std::string& model_name,
                                             const std::string& run_group) {
    std::ostringstream body;
    body << "{\"run_group\":\"" << json_escape_client(run_group) << "\"}";

    std::string out;
    const int status = http_put("/models/" + model_name + "/run_group", body.str(), out);
    check_status(status, out, "update_run_group(" + model_name + ")");
}
