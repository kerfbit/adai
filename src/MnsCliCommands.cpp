// @adai-status: beta
// @adai-version: 0.6.0
// @adai-reviewed: 2026-11-10

#include "MnsCliCommands.hpp"
#include <fstream>
#include <functional>
#include <limits>
#include <cctype>
#include <set>
#include <sstream>

namespace adai {

// TD-451: https:// used to be stripped while keeping port 8083, and mns_cli then connected with
// plain HTTP; a bad port was ignored (8083 kept) and any other scheme was read as a host name.
ParsedUrl parse_url(const std::string& raw) {
    ParsedUrl p;
    std::string s = raw;
    const auto scheme_end = s.find("://");
    if (scheme_end != std::string::npos) {
        const std::string scheme = s.substr(0, scheme_end);
        if (scheme == "https") {
            p.https = true;
            p.port = 443;
        } else if (scheme != "http") {
            p.error = "Unsupported URL scheme '" + scheme + "://' in " + raw + " (use http:// or https://)";
            return p;
        }
        s = s.substr(scheme_end + 3);
    }
    auto slash = s.find('/');
    if (slash != std::string::npos) {
        p.base_path = s.substr(slash);
        while (!p.base_path.empty() && p.base_path.back() == '/') p.base_path.pop_back();
        s = s.substr(0, slash);
    }
    std::string port_text;
    if (!s.empty() && s.front() == '[') {  // [IPv6]:port
        const auto close = s.find(']');
        if (close == std::string::npos) {
            p.error = "Invalid server URL: " + raw;
            return p;
        }
        p.host = s.substr(0, close + 1);
        if (close + 1 < s.size()) {
            if (s[close + 1] != ':') {
                p.error = "Invalid server URL: " + raw;
                return p;
            }
            port_text = s.substr(close + 2);
            if (port_text.empty()) port_text = "-";  // "[::1]:" has no port
        }
    } else if (const auto colon = s.find(':'); colon != std::string::npos) {
        p.host = s.substr(0, colon);
        port_text = s.substr(colon + 1);
        if (port_text.empty()) port_text = "-";
    } else {
        p.host = s;
    }
    if (p.host.empty()) {
        p.error = "Invalid server URL (no host): " + raw;
        return p;
    }
    if (!port_text.empty()) {
        const auto port = parse_strict_int(port_text);
        if (!port || *port < 1 || *port > 65535) {
            p.error = "Invalid port in server URL " + raw + " (expected 1-65535)";
            return p;
        }
        p.port = *port;
    }
    return p;
}

std::string ParsedUrl::url_for(const std::string& path) const {
    return std::string(https ? "https://" : "http://") + host + ":" + std::to_string(port) + base_path + path;
}

// TD-449: every control character is escaped (other ones below 0x20 used to pass through raw,
// producing invalid JSON).
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
                if (c < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    out += "\\u00";
                    out += kHex[c >> 4];
                    out += kHex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

std::string url_encode(const std::string& s) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '.' || c == '_' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += kHex[c >> 4];
            out += kHex[c & 0xF];
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

// TD-449: names and query values are URL-encoded here and in every path below (`get a/state` used to
// request /models/a/state, and `list --role "a b&c"` built /models?role=a b&c).
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
        path += sep + "state=" + url_encode(state);
        sep = "&";
    }
    if (!role.empty()) {
        path += sep + "role=" + url_encode(role);
        sep = "&";
    }
    if (!kind.empty()) {
        path += sep + "kind=" + url_encode(kind);
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

// TD-451: the commands below take exact positional arguments. Extra ones used to be ignored, which
// would have silently dropped a global option written after the command (`delete m --url X` would
// delete m on the default server).
namespace {
std::optional<std::string> exact_args(const std::vector<std::string>& args, size_t count) {
    std::vector<std::string> positionals;
    if (auto error = parse_options(args, 0, {}, {}, &positionals)) return error;
    if (positionals.size() > count) return "Unexpected argument: " + positionals[count];
    if (positionals.size() < count) return std::string("Missing argument");
    return std::nullopt;
}
}  // namespace

MnsCliRequest build_get_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 1))
        return make_error(*error + "\nUsage: get <name>");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/models/" + url_encode(args[0]);
    return r;
}

MnsCliRequest build_register_request(const std::vector<std::string>& args,
                                     const ServiceConfig& cfg) {
    if (args.size() < 2)
        return make_error("Usage: register <name> <role> [options...]");
    const std::string& name = args[0];
    const std::string& role = args[1];

    // TD-450: the architecture comes from the flags, else from --from-config (the trainer's own
    // config file), never from mns_cli's config. It used to default to mns_cli's config.mns.conf
    // (512/8), so a flagless register permanently gave the model an architecture the trainer then
    // adopted in place of its own config's 768/12. `cfg` now supplies only the run_group default.
    std::string kind = "chatbot";
    std::optional<size_t> d_model, num_heads, d_ff, enc_layers, dec_layers, max_seq;
    std::string from_config;
    std::string run_group = cfg.run_group;
    std::map<std::string, std::string> tags;

    // TD-196: --num-layers is the single layer-count flag for standalone encoder/decoder/
    // world_model records (own-kind field selected below); --encoder-layers/--decoder-layers
    // stay as-is for legacy bundled chatbot registration.
    std::optional<size_t> num_layers;
    std::string encoder_name, decoder_name;
    float sigreg_lambda = 1.0f;
    size_t sigreg_num_sketches = 64;
    bool sigreg_given = false;  // TD-449
    const auto sigreg = [&sigreg_given](ValueSetter setter) -> ValueSetter {
        return [&sigreg_given, setter](const std::string& flag, const std::string& value) {
            sigreg_given = true;
            return setter(flag, value);
        };
    };

    auto optional_count = [](std::optional<size_t>& field, size_t min) -> ValueSetter {
        return [&field, min](const std::string& flag, const std::string& value) -> std::optional<std::string> {
            size_t n = 0;
            if (auto error = count_into(n, min)(flag, value)) return error;
            field = n;
            return std::nullopt;
        };
    };
    if (auto error = parse_options(
            args, 2,
            {{"--kind", string_into(kind)},
             {"--d-model", optional_count(d_model, 1)},
             {"--num-heads", optional_count(num_heads, 1)},
             {"--d-ff", optional_count(d_ff, 1)},
             {"--num-layers", optional_count(num_layers, 1)},
             {"--encoder-layers", optional_count(enc_layers, 0)},
             {"--decoder-layers", optional_count(dec_layers, 0)},
             {"--max-seq-length", optional_count(max_seq, 1)},
             {"--from-config", string_into(from_config)},
             {"--run-group", string_into(run_group)},
             {"--encoder", string_into(encoder_name)},
             {"--decoder", string_into(decoder_name)},
             {"--sigreg-lambda", sigreg(float_into(sigreg_lambda, 0.0f, std::numeric_limits<float>::max()))},
             {"--sigreg-num-sketches", sigreg(count_into(sigreg_num_sketches, 1))},
             {"--tag", key_value_into(tags)}})) {
        return make_error(*error + "\nUsage: register <name> <role> [options...] (see --help)");
    }

    if (kind != "encoder" && kind != "decoder" && kind != "world_model" && kind != "chatbot")
        return make_error("--kind must be one of: encoder, decoder, world_model, chatbot");

    // TD-449: these were silently dropped for any other kind.
    if (sigreg_given && kind != "world_model")
        return make_error("--sigreg-lambda/--sigreg-num-sketches are only valid with --kind world_model");

    const bool standalone = kind != "chatbot";
    const bool has_encoder_decoder = !encoder_name.empty() || !decoder_name.empty();
    if (standalone && has_encoder_decoder)
        return make_error("--encoder/--decoder are only valid with --kind chatbot");
    const bool any_arch_flag = d_model || num_heads || d_ff || enc_layers || dec_layers || max_seq || num_layers;
    if (has_encoder_decoder && (any_arch_flag || !from_config.empty())) {
        return make_error(
            "--encoder/--decoder (new-style linked chatbot) cannot be combined with "
            "--d-model/--num-heads/--d-ff/--encoder-layers/--decoder-layers/--max-seq-length/--from-config "
            "(legacy inline architecture)");
    }
    if (num_layers && !standalone)
        return make_error("--num-layers is for --kind encoder/decoder/world_model; a chatbot takes "
                          "--encoder-layers and --decoder-layers");

    MnsCliRequest r;
    if (!has_encoder_decoder) {
        // A standalone kind has one layer count of its own (TD-196): num_encoder_layers for
        // encoder/world_model, num_decoder_layers for decoder; the other stays 0 unless a flag
        // sets it (the server then rejects it). TD-449: without --num-layers it used to send both
        // of the config's counts, which the server rejected.
        std::optional<size_t>& own_layers = kind == "decoder" ? dec_layers : enc_layers;
        std::optional<size_t>& other_layers = kind == "decoder" ? enc_layers : dec_layers;
        if (standalone && num_layers) own_layers = num_layers;

        std::string source = "the command line";
        if (!from_config.empty()) {
            if (!std::ifstream(from_config)) return make_error("--from-config: can't read " + from_config);
            const ServiceConfig file = ConfigLoader::load(from_config);
            if (!d_model) d_model = file.d_model;
            if (!num_heads) num_heads = file.num_heads;
            if (!d_ff) d_ff = file.d_ff;
            if (!max_seq) max_seq = file.max_seq_length;
            if (standalone) {
                if (!own_layers) own_layers = kind == "decoder" ? file.num_decoder_layers : file.num_encoder_layers;
            } else {
                if (!enc_layers) enc_layers = file.num_encoder_layers;
                if (!dec_layers) dec_layers = file.num_decoder_layers;
            }
            source = from_config + (any_arch_flag ? " and the command line" : "");
        }
        if (standalone && !other_layers) other_layers = 0;

        std::vector<std::string> missing;
        if (!d_model) missing.push_back("--d-model");
        if (!num_heads) missing.push_back("--num-heads");
        if (!d_ff) missing.push_back("--d-ff");
        if (standalone) {
            if (!own_layers) missing.push_back("--num-layers");
        } else {
            if (!enc_layers) missing.push_back("--encoder-layers");
            if (!dec_layers) missing.push_back("--decoder-layers");
        }
        if (!max_seq) missing.push_back("--max-seq-length");
        if (!missing.empty()) {
            std::string list;
            for (const auto& flag : missing) list += (list.empty() ? "" : ", ") + flag;
            return make_error("register needs the whole architecture, which can't be changed afterwards. Missing: " +
                              list +
                              ". Pass them, or --from-config <file> to take the rest from the trainer's config "
                              "(e.g. /etc/adai/config.trainer.conf).");
        }
        std::ostringstream notice;
        notice << "Registering " << name << " (" << kind << ") with d_model=" << *d_model
               << " num_heads=" << *num_heads << " d_ff=" << *d_ff << " num_encoder_layers=" << *enc_layers
               << " num_decoder_layers=" << *dec_layers << " max_seq_length=" << *max_seq << " (from " << source
               << ")";
        r.notice = notice.str();
    }

    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(name) << "\"" << ",\"role\":\"" << json_escape(role)
         << "\"" << ",\"kind\":\"" << json_escape(kind) << "\""
         << ",\"run_group\":\"" << json_escape(run_group) << "\"";
    if (!has_encoder_decoder) {
        body << ",\"arch\":{" << "\"d_model\":" << *d_model << ",\"num_heads\":" << *num_heads
             << ",\"d_ff\":" << *d_ff << ",\"num_encoder_layers\":" << *enc_layers
             << ",\"num_decoder_layers\":" << *dec_layers << ",\"max_seq_length\":" << *max_seq
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

    r.method = HttpMethod::Post;
    r.path = "/models";
    r.body = body.str();
    return r;
}

MnsCliRequest build_link_world_model_request(const std::vector<std::string>& args) {
    static const char* kUsage =
        "Usage: link-world-model <chatbot-name> --world-model <name> "
        "[--inject-every-n-layers N] [--hippocampal-enabled | --hippocampal-disabled] [--hippocampal-capacity N] "
        "[--hippocampal-repetition-alpha F] [--hippocampal-repetition-decay F] "
        "[--hippocampal-cross-reference-alpha F] [--hippocampal-association-decay F]\n"
        "  Settings not given keep their current values. Pass --world-model \"\" (empty) to detach.";
    if (args.empty()) return make_error(kUsage);
    const std::string& name = args[0];
    std::string world_model_name;
    bool world_model_set = false;
    size_t inject_every_n_layers = 0;
    bool hippocampal_enabled = false;
    bool hippocampal_disabled = false;
    size_t hippocampal_capacity = 512;
    float repetition_alpha = 0.0f;
    float repetition_decay = 0.95f;
    float cross_reference_alpha = 0.0f;
    float association_decay = 0.95f;

    // TD-449: only the settings given are sent, with "merge":true so mns_server keeps the rest. All eight
    // used to be sent, defaults for the omitted ones, so re-running it to change --inject-every-n-layers
    // turned hippocampal memory off and reset its tuning.
    std::set<std::string> given;
    const auto mark = [&given](const char* key, ValueSetter setter) -> ValueSetter {
        return [&given, key, setter](const std::string& flag, const std::string& value) {
            given.insert(key);
            return setter(flag, value);
        };
    };
    if (auto error = parse_options(
            args, 1,
            {{"--world-model",
              [&](const std::string&, const std::string& value) -> std::optional<std::string> {
                  world_model_name = value;
                  world_model_set = true;
                  return std::nullopt;
              }},
             {"--inject-every-n-layers",
              mark("world_model_inject_every_n_layers", count_into(inject_every_n_layers, 0))},
             {"--hippocampal-capacity", mark("hippocampal_memory_capacity", count_into(hippocampal_capacity, 1))},
             {"--hippocampal-repetition-alpha",
              mark("hippocampal_repetition_alpha",
                   float_into(repetition_alpha, 0.0f, std::numeric_limits<float>::max()))},
             {"--hippocampal-repetition-decay",
              mark("hippocampal_repetition_decay", float_into(repetition_decay, 0.0f, 1.0f))},
             {"--hippocampal-cross-reference-alpha",
              mark("hippocampal_cross_reference_alpha",
                   float_into(cross_reference_alpha, 0.0f, std::numeric_limits<float>::max()))},
             {"--hippocampal-association-decay",
              mark("hippocampal_association_decay", float_into(association_decay, 0.0f, 1.0f))}},
            {{"--hippocampal-enabled", &hippocampal_enabled}, {"--hippocampal-disabled", &hippocampal_disabled}})) {
        return make_error(*error + "\n" + kUsage);
    }

    if (!world_model_set) return make_error(kUsage);
    if (hippocampal_enabled && hippocampal_disabled)
        return make_error(std::string("--hippocampal-enabled and --hippocampal-disabled are exclusive\n") + kUsage);

    std::ostringstream body;
    body << "{\"world_model_name\":\"" << json_escape(world_model_name) << "\",\"merge\":true";
    if (given.count("world_model_inject_every_n_layers"))
        body << ",\"world_model_inject_every_n_layers\":" << inject_every_n_layers;
    if (hippocampal_enabled || hippocampal_disabled)
        body << ",\"hippocampal_memory_enabled\":" << (hippocampal_enabled ? "true" : "false");
    if (given.count("hippocampal_memory_capacity"))
        body << ",\"hippocampal_memory_capacity\":" << hippocampal_capacity;
    if (given.count("hippocampal_repetition_alpha"))
        body << ",\"hippocampal_repetition_alpha\":" << repetition_alpha;
    if (given.count("hippocampal_repetition_decay"))
        body << ",\"hippocampal_repetition_decay\":" << repetition_decay;
    if (given.count("hippocampal_cross_reference_alpha"))
        body << ",\"hippocampal_cross_reference_alpha\":" << cross_reference_alpha;
    if (given.count("hippocampal_association_decay"))
        body << ",\"hippocampal_association_decay\":" << association_decay;
    body << "}";

    MnsCliRequest r;
    r.method = HttpMethod::Post;
    r.path = "/models/" + url_encode(name) + "/link-world-model";
    r.body = body.str();
    return r;
}

MnsCliRequest build_update_run_group_request(const std::vector<std::string>& args) {
    // TD-449/TD-451: options in any order, and nothing else (it used to need exactly
    // `<name> --run-group <value>` and ignore anything after).
    std::vector<std::string> positionals;
    std::optional<std::string> run_group;
    auto error = parse_options(args, 0,
                               {{"--run-group",
                                 [&](const std::string&, const std::string& value) -> std::optional<std::string> {
                                     run_group = value;
                                     return std::nullopt;
                                 }}},
                               {}, &positionals);
    if (!error && positionals.size() > 1) error = "Unexpected argument: " + positionals[1];
    if (!error && (positionals.empty() || !run_group)) error = std::string("Missing argument");
    if (error) return make_error(*error + "\nUsage: update <name> --run-group <value>");
    std::ostringstream body;
    body << "{\"run_group\":\"" << json_escape(*run_group) << "\"}";
    MnsCliRequest r;
    r.method = HttpMethod::Put;
    r.path = "/models/" + url_encode(positionals[0]) + "/run_group";
    r.body = body.str();
    return r;
}

MnsCliRequest build_resolve_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 1))
        return make_error(*error + "\nUsage: resolve <name>");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/models/" + url_encode(args[0]) + "/resolve";
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
    r.path = "/models/" + url_encode(name) + "/state";
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
    bool revive = false;  // TD-460

    if (auto error = parse_options(args, 2,
                                   {{"--artifact-path", string_into(art_path)},
                                    {"--artifact-host", string_into(art_host)},
                                    {"--artifact-checksum", string_into(art_checksum)},
                                    {"--artifact-format", string_into(art_format)},
                                    {"--summary", key_value_into(summary)}},
                                   {{"--revive", &revive}})) {
        return make_error(*error + "\nUsage: set-candidate <name> <run-id> [options...] (see --help)");
    }

    std::ostringstream body;
    body << "{\"state\":\"candidate\"" << (revive ? ",\"revive\":true" : "") << ",\"run_id\":\""
         << json_escape(run_id) << "\""
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
    r.path = "/models/" + url_encode(name) + "/state";
    r.body = body.str();
    return r;
}

MnsCliRequest build_delete_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 1))
        return make_error(*error + "\nUsage: delete <name>");
    MnsCliRequest r;
    r.method = HttpMethod::Delete;
    r.path = "/models/" + url_encode(args[0]);
    return r;
}

MnsCliRequest build_roles_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 0))
        return make_error(*error + "\nUsage: roles");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/roles";
    return r;
}

MnsCliRequest build_resolve_role_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 1))
        return make_error(*error + "\nUsage: resolve-role <role>");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/roles/" + url_encode(args[0]) + "/production";
    return r;
}

MnsCliRequest build_promote_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 2))
        return make_error(*error + "\nUsage: promote <role> <model-name>");
    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(args[1]) << "\"}";
    MnsCliRequest r;
    r.method = HttpMethod::Put;
    r.path = "/roles/" + url_encode(args[0]) + "/production";
    r.body = body.str();
    return r;
}

MnsCliRequest build_health_request(const std::vector<std::string>& args) {
    if (auto error = exact_args(args, 0))
        return make_error(*error + "\nUsage: health");
    MnsCliRequest r;
    r.method = HttpMethod::Get;
    r.path = "/health";
    return r;
}

}  // namespace adai
