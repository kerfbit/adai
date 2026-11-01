// @adai-status: experimental
// @adai-version: 0.2.0
// @adai-reviewed: 2026-11-01

#include "MnsCliCommands.hpp"
#include <functional>
#include <limits>
#include <sstream>

namespace adai {

ParsedUrl parse_url(const std::string& raw) {
    ParsedUrl p;
    std::string s = raw;
    if (s.rfind("http://", 0) == 0)
        s = s.substr(7);
    // TODO: See TD-451 in TECHNICAL_DEBT.md - https:// is accepted but keeps port 8083 and mns_cli connects with plain HTTP
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

// TODO: See TD-449 in TECHNICAL_DEBT.md - other control characters (< 0x20) pass through raw, producing invalid JSON
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

// TD-448: every builder reads its options through parse_options(). They used to skip anything they
// didn't recognise (`register m chat --dmodel 256` registered the config's d_model, permanently),
// drop a value flag at the end of the line, and parse numbers with stoul/stoi/stof (an exception
// aborted mns_cli; -1 wrapped to 2^64-1; 512x became 512).

// Applies a flag's value; returns an error message, or nullopt on success.
using ValueSetter = std::function<std::optional<std::string>(const std::string& flag, const std::string& value)>;

// Largest count accepted: the server stores these as int.
constexpr size_t kMaxCount = static_cast<size_t>(std::numeric_limits<int>::max());

std::string invalid(const std::string& flag, const std::string& value, const std::string& expected) {
    return "Invalid value for " + flag + ": '" + value + "' (expected " + expected + ")";
}

ValueSetter count_into(size_t& field, size_t min) {
    return [&field, min](const std::string& flag, const std::string& value) -> std::optional<std::string> {
        const auto n = parse_strict_size(value);
        if (!n || *n < min || *n > kMaxCount) {
            return invalid(flag, value, "a whole number >= " + std::to_string(min));
        }
        field = *n;
        return std::nullopt;
    };
}

ValueSetter float_into(float& field, float min, float max) {
    return [&field, min, max](const std::string& flag, const std::string& value) -> std::optional<std::string> {
        const auto f = parse_strict_float(value);
        if (!f || *f < min || *f > max) {
            std::ostringstream expected;
            expected << "a number from " << min;
            if (max < std::numeric_limits<float>::max()) expected << " to " << max;
            else expected << " up";
            return invalid(flag, value, expected.str());
        }
        field = *f;
        return std::nullopt;
    };
}

ValueSetter string_into(std::string& field) {
    return [&field](const std::string&, const std::string& value) -> std::optional<std::string> {
        field = value;
        return std::nullopt;
    };
}

// --tag/--summary key=value; a value without '=' used to be dropped silently.
ValueSetter key_value_into(std::map<std::string, std::string>& map) {
    return [&map](const std::string& flag, const std::string& value) -> std::optional<std::string> {
        const auto eq = value.find('=');
        if (eq == std::string::npos || eq == 0) return invalid(flag, value, "key=value");
        map[value.substr(0, eq)] = value.substr(eq + 1);
        return std::nullopt;
    };
}

// Reads args[start..]. A flag in value_flags takes the next argument (missing, or another "--"
// flag, is an error); a flag in switches sets its bool. Any other "-"-prefixed argument is an
// unknown option. Other arguments go to `positionals`, or are an error when it's null.
std::optional<std::string> parse_options(const std::vector<std::string>& args, size_t start,
                                         const std::map<std::string, ValueSetter>& value_flags,
                                         const std::map<std::string, bool*>& switches = {},
                                         std::vector<std::string>* positionals = nullptr) {
    for (size_t i = start; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (auto it = value_flags.find(arg); it != value_flags.end()) {
            if (i + 1 >= args.size() || args[i + 1].rfind("--", 0) == 0) return "Missing value for " + arg;
            if (auto error = it->second(arg, args[++i])) return error;
        } else if (auto sw = switches.find(arg); sw != switches.end()) {
            *sw->second = true;
        } else if (arg.size() > 1 && arg[0] == '-') {
            return "Unknown option: " + arg;
        } else if (positionals) {
            positionals->push_back(arg);
        } else {
            return "Unexpected argument: " + arg;
        }
    }
    return std::nullopt;
}
}  // namespace

// TODO: See TD-449 in TECHNICAL_DEBT.md - path segments and query values aren't URL-encoded
MnsCliRequest build_list_request(const std::vector<std::string>& args) {
    std::string state, role, kind;
    size_t limit = 0;  // 0 = no limit
    if (auto error = parse_options(args, 0,
                                   {{"--state", string_into(state)},
                                    {"--role", string_into(role)},
                                    {"--kind", string_into(kind)},
                                    {"--limit", count_into(limit, 0)}})) {
        return make_error(*error + "\nUsage: list [--state S] [--role R] [--kind K] [--limit N]");
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
    if (!kind.empty()) {
        path += sep + "kind=" + kind;
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

    std::string kind = "chatbot";
    size_t d_model = cfg.d_model;
    size_t num_heads = cfg.num_heads;
    size_t d_ff = cfg.d_ff;
    size_t enc_layers = cfg.num_encoder_layers;
    size_t dec_layers = cfg.num_decoder_layers;
    size_t max_seq = cfg.max_seq_length;
    std::string run_group = cfg.run_group;
    std::map<std::string, std::string> tags;

    // TD-196: --num-layers is the single layer-count flag for standalone encoder/decoder/
    // world_model records (own-kind field selected below); --encoder-layers/--decoder-layers
    // stay as-is for legacy bundled chatbot registration.
    bool num_layers_set = false;
    size_t num_layers = 0;
    std::string encoder_name, decoder_name;
    float sigreg_lambda = 1.0f;
    size_t sigreg_num_sketches = 64;

    auto num_layers_into = count_into(num_layers, 1);
    if (auto error = parse_options(
            args, 2,
            {{"--kind", string_into(kind)},
             {"--d-model", count_into(d_model, 1)},
             {"--num-heads", count_into(num_heads, 1)},
             {"--d-ff", count_into(d_ff, 1)},
             {"--num-layers",
              [&](const std::string& flag, const std::string& value) {
                  num_layers_set = true;
                  return num_layers_into(flag, value);
              }},
             {"--encoder-layers", count_into(enc_layers, 0)},
             {"--decoder-layers", count_into(dec_layers, 0)},
             {"--max-seq-length", count_into(max_seq, 1)},
             {"--run-group", string_into(run_group)},
             {"--encoder", string_into(encoder_name)},
             {"--decoder", string_into(decoder_name)},
             {"--sigreg-lambda", float_into(sigreg_lambda, 0.0f, std::numeric_limits<float>::max())},
             {"--sigreg-num-sketches", count_into(sigreg_num_sketches, 1)},
             {"--tag", key_value_into(tags)}})) {
        return make_error(*error + "\nUsage: register <name> <role> [options...] (see --help)");
    }

    if (kind != "encoder" && kind != "decoder" && kind != "world_model" && kind != "chatbot")
        return make_error("--kind must be one of: encoder, decoder, world_model, chatbot");

    const bool has_encoder_decoder = !encoder_name.empty() || !decoder_name.empty();
    if (kind != "chatbot" && has_encoder_decoder)
        return make_error("--encoder/--decoder are only valid with --kind chatbot");
    if (kind == "chatbot" && has_encoder_decoder &&
        (d_model != cfg.d_model || num_heads != cfg.num_heads || d_ff != cfg.d_ff ||
         enc_layers != cfg.num_encoder_layers || dec_layers != cfg.num_decoder_layers ||
         max_seq != cfg.max_seq_length)) {
        return make_error(
            "--encoder/--decoder (new-style linked chatbot) cannot be combined with "
            "--d-model/--num-heads/--d-ff/--encoder-layers/--decoder-layers/--max-seq-length "
            "(legacy inline architecture)");
    }

    // TODO: See TD-449 in TECHNICAL_DEBT.md - without --num-layers both config layer counts are sent; the server rejects it
    if (kind == "encoder" || kind == "decoder" || kind == "world_model") {
        if (num_layers_set) {
            if (kind == "decoder") {
                dec_layers = num_layers;
                enc_layers = 0;
            } else {
                enc_layers = num_layers;
                dec_layers = 0;
            }
        }
    }

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(name) << "\"" << ",\"role\":\"" << json_escape(role)
         << "\"" << ",\"kind\":\"" << json_escape(kind) << "\""
         << ",\"run_group\":\"" << json_escape(run_group) << "\"";
    if (kind != "chatbot" || !has_encoder_decoder) {
        body << ",\"arch\":{" << "\"d_model\":" << d_model << ",\"num_heads\":" << num_heads
             << ",\"d_ff\":" << d_ff << ",\"num_encoder_layers\":" << enc_layers
             << ",\"num_decoder_layers\":" << dec_layers << ",\"max_seq_length\":" << max_seq
             << "}";
    }
    if (has_encoder_decoder || kind == "world_model") {
        body << ",\"connection\":{";
        bool first = true;
        if (has_encoder_decoder) {
            body << "\"encoder_name\":\"" << json_escape(encoder_name) << "\""
                 << ",\"decoder_name\":\"" << json_escape(decoder_name) << "\"";
            first = false;
        }
        if (kind == "world_model") {
            if (!first)
                body << ',';
            body << "\"sigreg_lambda\":" << sigreg_lambda
                 << ",\"sigreg_num_sketches\":" << sigreg_num_sketches;
        }
        body << "}";
    }
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

MnsCliRequest build_link_world_model_request(const std::vector<std::string>& args) {
    if (args.empty()) {
        return make_error(
            "Usage: link-world-model <chatbot-name> --world-model <name> "
            "[--inject-every-n-layers N] [--hippocampal-enabled] [--hippocampal-capacity N] "
            "[--hippocampal-repetition-alpha F] [--hippocampal-repetition-decay F] "
            "[--hippocampal-cross-reference-alpha F] [--hippocampal-association-decay F]\n"
            "  Pass --world-model \"\" (empty) to detach.");
    }
    const std::string& name = args[0];
    std::string world_model_name;
    bool world_model_set = false;
    size_t inject_every_n_layers = 0;
    bool hippocampal_enabled = false;
    size_t hippocampal_capacity = 512;
    float repetition_alpha = 0.0f;
    float repetition_decay = 0.95f;
    float cross_reference_alpha = 0.0f;
    float association_decay = 0.95f;

    if (auto error = parse_options(
            args, 1,
            {{"--world-model",
              [&](const std::string&, const std::string& value) -> std::optional<std::string> {
                  world_model_name = value;
                  world_model_set = true;
                  return std::nullopt;
              }},
             {"--inject-every-n-layers", count_into(inject_every_n_layers, 0)},
             {"--hippocampal-capacity", count_into(hippocampal_capacity, 1)},
             {"--hippocampal-repetition-alpha", float_into(repetition_alpha, 0.0f, std::numeric_limits<float>::max())},
             {"--hippocampal-repetition-decay", float_into(repetition_decay, 0.0f, 1.0f)},
             {"--hippocampal-cross-reference-alpha",
              float_into(cross_reference_alpha, 0.0f, std::numeric_limits<float>::max())},
             {"--hippocampal-association-decay", float_into(association_decay, 0.0f, 1.0f)}},
            {{"--hippocampal-enabled", &hippocampal_enabled}})) {
        return make_error(*error + "\nUsage: link-world-model <chatbot-name> --world-model <name> [...]");
    }

    if (!world_model_set)
        return make_error("Usage: link-world-model <chatbot-name> --world-model <name> [...]");

    // TODO: See TD-449 in TECHNICAL_DEBT.md - always sends all eight fields; omitted flags reset hippocampal settings on relink
    std::ostringstream body;
    body << "{\"world_model_name\":\"" << json_escape(world_model_name) << "\""
         << ",\"world_model_inject_every_n_layers\":" << inject_every_n_layers
         << ",\"hippocampal_memory_enabled\":" << (hippocampal_enabled ? "true" : "false")
         << ",\"hippocampal_memory_capacity\":" << hippocampal_capacity
         << ",\"hippocampal_repetition_alpha\":" << repetition_alpha
         << ",\"hippocampal_repetition_decay\":" << repetition_decay
         << ",\"hippocampal_cross_reference_alpha\":" << cross_reference_alpha
         << ",\"hippocampal_association_decay\":" << association_decay << "}";

    MnsCliRequest r;
    r.method = HttpMethod::Post;
    r.path = "/models/" + name + "/link-world-model";
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
    // TD-448: at most one session key, and no unknown options; any stray argument (e.g. the typo
    // --new_run) used to become the session key, continuing the run with a bogus metrics link.
    std::vector<std::string> positionals;
    if (auto error = parse_options(args, 1, {}, {{"--new-run", &new_run}}, &positionals)) {
        return make_error(*error + "\nUsage: set-training <name> [--new-run] [session-key]");
    }
    if (positionals.size() > 1) {
        return make_error("Unexpected argument: " + positionals[1] +
                          " (set-training takes at most one session key)\nUsage: set-training <name> [--new-run] [session-key]");
    }
    const std::string session_key = positionals.empty() ? "" : positionals[0];

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

    if (auto error = parse_options(args, 2,
                                   {{"--artifact-path", string_into(art_path)},
                                    {"--artifact-host", string_into(art_host)},
                                    {"--artifact-checksum", string_into(art_checksum)},
                                    {"--artifact-format", string_into(art_format)},
                                    {"--summary", key_value_into(summary)}})) {
        return make_error(*error + "\nUsage: set-candidate <name> <run-id> [options...] (see --help)");
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
