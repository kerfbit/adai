// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-11

#include "MnsCliCommands.hpp"
#include <sstream>

namespace adai {

ParsedUrl parse_url(const std::string& raw) {
    ParsedUrl p;
    std::string s = raw;
    if (s.rfind("http://", 0) == 0)
        s = s.substr(7);
    if (s.rfind("https://", 0) == 0)
        s = s.substr(8);
    auto slash = s.find('/');
    if (slash != std::string::npos)
        s = s.substr(0, slash);
    auto colon = s.find(':');
    if (colon != std::string::npos) {
        p.host = s.substr(0, colon);
        try {
            p.port = std::stoi(s.substr(colon + 1));
        } catch (...) {
        }
    } else {
        p.host = s;
    }
    return p;
}

std::string json_escape(const std::string& s) {
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

namespace {
MnsCliRequest make_error(std::string message) {
    MnsCliRequest r;
    r.error = true;
    r.error_message = std::move(message);
    return r;
}
}  // namespace

MnsCliRequest build_list_request(const std::vector<std::string>& args) {
    std::string state, role;
    int limit = 0;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--state" && i + 1 < args.size()) {
            state = args[++i];
        } else if (args[i] == "--role" && i + 1 < args.size()) {
            role = args[++i];
        } else if (args[i] == "--limit" && i + 1 < args.size()) {
            limit = std::stoi(args[++i]);
        }
    }
    std::string path = "/models";
    std::string sep = "?";
    if (!state.empty()) {
        path += sep + "state=" + state;
        sep = "&";
    }
    if (!role.empty()) {
        path += sep + "role=" + role;
        sep = "&";
    }
    if (limit > 0) {
        path += sep + "limit=" + std::to_string(limit);
    }
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = path;
    return r;
}

MnsCliRequest build_get_request(const std::vector<std::string>& args) {
    if (args.empty())
        return make_error("Usage: get <name>");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/models/" + args[0];
    return r;
}

MnsCliRequest build_register_request(const std::vector<std::string>& args,
                                     const ServiceConfig& cfg) {
    if (args.size() < 2)
        return make_error("Usage: register <name> <role> [options...]");
    const std::string& name = args[0];
    const std::string& role = args[1];

    size_t d_model = cfg.d_model;
    size_t num_heads = cfg.num_heads;
    size_t d_ff = cfg.d_ff;
    size_t enc_layers = cfg.num_encoder_layers;
    size_t dec_layers = cfg.num_decoder_layers;
    size_t max_seq = cfg.max_seq_length;
    std::string run_group = cfg.run_group;
    std::map<std::string, std::string> tags;

    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--d-model" && i + 1 < args.size())
            d_model = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--num-heads" && i + 1 < args.size())
            num_heads = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--d-ff" && i + 1 < args.size())
            d_ff = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--encoder-layers" && i + 1 < args.size())
            enc_layers = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--decoder-layers" && i + 1 < args.size())
            dec_layers = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--max-seq-length" && i + 1 < args.size())
            max_seq = static_cast<size_t>(std::stoul(args[++i]));
        else if (args[i] == "--run-group" && i + 1 < args.size())
            run_group = args[++i];
        else if (args[i] == "--tag" && i + 1 < args.size()) {
            const auto& kv = args[++i];
            auto eq = kv.find('=');
            if (eq != std::string::npos)
                tags[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(name) << "\"" << ",\"role\":\"" << json_escape(role)
         << "\"" << ",\"run_group\":\"" << json_escape(run_group) << "\""
         << ",\"arch\":{" << "\"d_model\":" << d_model << ",\"num_heads\":" << num_heads
         << ",\"d_ff\":" << d_ff << ",\"num_encoder_layers\":" << enc_layers
         << ",\"num_decoder_layers\":" << dec_layers << ",\"max_seq_length\":" << max_seq << "}";
    if (!tags.empty()) {
        body << ",\"tags\":{";
        bool first = true;
        for (const auto& [k, v] : tags) {
            if (!first)
                body << ',';
            first = false;
            body << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
        }
        body << "}";
    }
    body << "}";

    MnsCliRequest r;
    r.method = HttpMethod::Post;
    r.path = "/models";
    r.body = body.str();
    return r;
}

MnsCliRequest build_update_run_group_request(const std::vector<std::string>& args) {
    if (args.size() < 2 || args[1] != "--run-group" || args.size() < 3)
        return make_error("Usage: update <name> --run-group <value>");
    std::ostringstream body;
    body << "{\"run_group\":\"" << json_escape(args[2]) << "\"}";
    MnsCliRequest r;
    r.method = HttpMethod::Put;
    r.path = "/models/" + args[0] + "/run_group";
    r.body = body.str();
    return r;
}

MnsCliRequest build_resolve_request(const std::vector<std::string>& args) {
    if (args.empty())
        return make_error("Usage: resolve <name>");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/models/" + args[0] + "/resolve";
    return r;
}

MnsCliRequest build_set_training_request(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error(
            "Usage: set-training <name> [--new-run] [session-key]\n"
            "  MNS allocates run_id (definitive standard); --new-run requests a fresh\n"
            "  run (equivalent to 'retrain'), omitted means continue the current run.\n"
            "  The allocated run_id is printed in the response's \"run_id\" field.");
    }
    const std::string& name = args[0];
    bool new_run = false;
    std::string session_key;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--new-run")
            new_run = true;
        else
            session_key = args[i];
    }

    std::ostringstream body;
    body << "{\"state\":\"training\"" << ",\"new_run\":" << (new_run ? "true" : "false");
    if (!session_key.empty())
        body << ",\"metrics_session_key\":\"" << json_escape(session_key) << "\"";
    body << "}";

    MnsCliRequest r;
    r.method = HttpMethod::Put;
    r.path = "/models/" + name + "/state";
    r.body = body.str();
    return r;
}

MnsCliRequest build_set_candidate_request(const std::vector<std::string>& args) {
    if (args.size() < 2)
        return make_error("Usage: set-candidate <name> <run-id> [options...]");
    const std::string& name = args[0];
    const std::string& run_id = args[1];

    std::string art_path, art_host, art_checksum, art_format = "adai-native";
    std::map<std::string, std::string> summary;

    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--artifact-path" && i + 1 < args.size())
            art_path = args[++i];
        else if (args[i] == "--artifact-host" && i + 1 < args.size())
            art_host = args[++i];
        else if (args[i] == "--artifact-checksum" && i + 1 < args.size())
            art_checksum = args[++i];
        else if (args[i] == "--artifact-format" && i + 1 < args.size())
            art_format = args[++i];
        else if (args[i] == "--summary" && i + 1 < args.size()) {
            const auto& kv = args[++i];
            auto eq = kv.find('=');
            if (eq != std::string::npos)
                summary[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }

    std::ostringstream body;
    body << "{\"state\":\"candidate\"" << ",\"run_id\":\"" << json_escape(run_id) << "\""
         << ",\"artifact\":{" << "\"host\":\"" << json_escape(art_host) << "\"" << ",\"path\":\""
         << json_escape(art_path) << "\"" << ",\"checksum\":\"" << json_escape(art_checksum) << "\""
         << ",\"format\":\"" << json_escape(art_format) << "\"" << "}";
    if (!summary.empty()) {
        body << ",\"training_summary\":{";
        bool first = true;
        for (const auto& [k, v] : summary) {
            if (!first)
                body << ',';
            first = false;
            body << '"' << json_escape(k) << "\":\"" << json_escape(v) << '"';
        }
        body << "}";
    }
    body << "}";

    MnsCliRequest r;
    r.method = HttpMethod::Put;
    r.path = "/models/" + name + "/state";
    r.body = body.str();
    return r;
}

MnsCliRequest build_delete_request(const std::vector<std::string>& args) {
    if (args.empty())
        return make_error("Usage: delete <name>");
    MnsCliRequest r;
    r.method = HttpMethod::Delete;
    r.path = "/models/" + args[0];
    return r;
}

MnsCliRequest build_roles_request() {
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/roles";
    return r;
}

MnsCliRequest build_resolve_role_request(const std::vector<std::string>& args) {
    if (args.empty())
        return make_error("Usage: resolve-role <role>");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/roles/" + args[0] + "/production";
    return r;
}

MnsCliRequest build_promote_request(const std::vector<std::string>& args) {
    if (args.size() < 2)
        return make_error("Usage: promote <role> <model-name>");
    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(args[1]) << "\"}";
    MnsCliRequest r;
    r.method = HttpMethod::Put;
    r.path = "/roles/" + args[0] + "/production";
    r.body = body.str();
    return r;
}

MnsCliRequest build_health_request() {
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/health";
    return r;
}

}  // namespace adai
