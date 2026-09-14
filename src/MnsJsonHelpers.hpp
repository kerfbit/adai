#pragma once

// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-09-13

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mns_gui {

// TD-068 (fixed): find the closing quote of a JSON string literal starting
// just after the opening quote at `start`, honoring backslash-escapes (so a
// value like "he said \"hi\"" isn't truncated at the escaped quote).
// Returns std::string::npos if the string is unterminated.
inline size_t find_string_end(const std::string& body, size_t start) {
    for (size_t i = start; i < body.size(); ++i) {
        if (body[i] == '\\') {
            ++i;  // skip the escaped character (safe even if it's the last char)
            continue;
        }
        if (body[i] == '"') {
            return i;
        }
    }
    return std::string::npos;
}

inline std::string json_value(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    auto pos = body.find(needle);
    if (pos == std::string::npos)
        return {};
    pos += needle.size();
    while (pos < body.size() && body[pos] == ' ')
        ++pos;
    if (pos >= body.size())
        return {};
    if (body[pos] == '"') {
        auto end = find_string_end(body, pos + 1);
        return (end != std::string::npos) ? body.substr(pos + 1, end - pos - 1) : "";
    }
    auto end = body.find_first_of(",}] \n", pos);
    return (end != std::string::npos) ? body.substr(pos, end - pos) : body.substr(pos);
}

// TD-068 (fixed): this used to count '{'/'}' unconditionally, so a string
// value anywhere in an object (e.g. a "tags" entry, a label, an error
// message) containing a literal '{' or '}' character desynchronized the
// depth counter — corrupting that object and every object parsed after it
// for the rest of the array (reproduced: a single stray '}' inside one
// object's string value truncated that object and turned the next real
// object into a bogus empty "{}"). Now skips over string content (honoring
// backslash-escapes) the same way json_pretty() below already does, so only
// structural braces are counted.
inline std::vector<std::string> json_array_objects(const std::string& body,
                                                   const std::string& key) {
    std::vector<std::string> result;
    std::string needle = "\"" + key + "\":[";
    auto pos = body.find(needle);
    if (pos == std::string::npos) {
        pos = body.find('[');
        if (pos == std::string::npos)
            return result;
    } else {
        pos += needle.size();
    }

    int depth = 0;
    size_t obj_start = 0;
    for (size_t i = pos; i < body.size(); ++i) {
        if (body[i] == '"') {
            auto end = find_string_end(body, i + 1);
            if (end == std::string::npos)
                break;  // unterminated string — malformed JSON, stop parsing
            i = end;
            continue;
        }
        if (body[i] == '{') {
            if (depth == 0)
                obj_start = i;
            ++depth;
        } else if (body[i] == '}') {
            --depth;
            if (depth == 0) {
                result.push_back(body.substr(obj_start, i - obj_start + 1));
            }
        } else if (body[i] == ']' && depth == 0) {
            break;
        }
    }
    return result;
}

inline std::string json_escape(const std::string& s) {
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
            default:
                out += static_cast<char>(c);
                break;
        }
    }
    return out;
}

inline std::string json_pretty(const std::string& s) {
    std::string out;
    int indent = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"') {
            // TD-102 (fixed): used to toggle an in_string flag by checking only
            // whether the *immediately preceding* character was a backslash —
            // for a real closing quote that happens to follow an escaped
            // backslash in the JSON text (`\\"`, i.e. the string's own content
            // ends in a literal backslash, e.g. a Windows-style artifact path
            // like "C:\\models\\"), that preceding character IS a backslash,
            // so the check misclassified the true closing quote as escaped and
            // never toggled out of "in string" — garbling every character
            // after it for the rest of the output. json_array_objects() above
            // already had (and was fixed for, TD-068) the identical class of
            // bug; this reuses that fix's escape-aware find_string_end()
            // scanner instead of a naive single-character lookback.
            out += c;
            auto end = find_string_end(s, i + 1);
            if (end == std::string::npos) {
                out += s.substr(i + 1);
                break;
            }
            out += s.substr(i + 1, end - i);  // string content + its closing quote
            i = end;
            continue;
        }
        switch (c) {
            case '{':
            case '[':
                out += c;
                out += '\n';
                ++indent;
                out += std::string(indent * 2, ' ');
                break;
            case '}':
            case ']':
                out += '\n';
                --indent;
                out += std::string(indent * 2, ' ');
                out += c;
                break;
            case ',':
                out += c;
                out += '\n';
                out += std::string(indent * 2, ' ');
                break;
            case ':':
                out += ": ";
                break;
            default:
                out += c;
        }
    }
    return out;
}

// TD-037: parses MnsManagerGUI's "Register" tab tags field (comma-separated `key=value` pairs,
// arbitrary surrounding whitespace around each key/value) into ordered pairs — extracted
// verbatim from onRegisterModel() so it's testable without a QMainWindow/QLineEdit. An entry with
// no '=' is silently skipped (matches the original widget code's behavior).
inline std::vector<std::pair<std::string, std::string>> parse_tags(const std::string& tags_str) {
    std::vector<std::pair<std::string, std::string>> tags;
    if (tags_str.empty())
        return tags;

    std::istringstream ss(tags_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto eq = tok.find('=');
        if (eq == std::string::npos)
            continue;
        std::string k = tok.substr(0, eq);
        std::string v = tok.substr(eq + 1);
        while (!k.empty() && k.front() == ' ')
            k.erase(k.begin());
        while (!k.empty() && k.back() == ' ')
            k.pop_back();
        while (!v.empty() && v.front() == ' ')
            v.erase(v.begin());
        while (!v.empty() && v.back() == ' ')
            v.pop_back();
        tags.emplace_back(std::move(k), std::move(v));
    }
    return tags;
}

// TD-037: the register-model request body's architecture sub-object — one field per
// MnsManagerGUI "Register" tab spin box, extracted so the whole body can be built (and tested)
// without any of those widgets existing.
struct RegisterModelArch {
    int d_model = 0;
    int num_heads = 0;
    int d_ff = 0;
    int num_encoder_layers = 0;
    int num_decoder_layers = 0;
    int max_seq_length = 0;
};

// TD-037: builds POST /models's JSON body — extracted verbatim from onRegisterModel() (the
// tags-parsing loop now lives in parse_tags() above).
inline std::string build_register_model_body(const std::string& name, const std::string& role,
                                             const RegisterModelArch& arch,
                                             const std::string& tags_str) {
    std::ostringstream body;
    body << "{\"model_name\":\"" << json_escape(name) << "\"" << ",\"role\":\"" << json_escape(role)
         << "\"" << ",\"arch\":{" << "\"d_model\":" << arch.d_model
         << ",\"num_heads\":" << arch.num_heads << ",\"d_ff\":" << arch.d_ff
         << ",\"num_encoder_layers\":" << arch.num_encoder_layers
         << ",\"num_decoder_layers\":" << arch.num_decoder_layers
         << ",\"max_seq_length\":" << arch.max_seq_length << "}";

    auto tags = parse_tags(tags_str);
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
    return body.str();
}

// TD-037: builds PUT /models/{name}/state's JSON body for the "candidate" transition — extracted
// verbatim from onSetCandidate(). run_id/artifact_path are both optional (either may be empty).
inline std::string build_set_candidate_body(const std::string& run_id,
                                            const std::string& artifact_path) {
    std::ostringstream body;
    body << "{\"state\":\"candidate\"";
    if (!run_id.empty())
        body << ",\"run_id\":\"" << json_escape(run_id) << "\"";
    if (!artifact_path.empty()) {
        body << ",\"artifact\":{\"path\":\"" << json_escape(artifact_path)
             << "\",\"host\":\"\",\"checksum\":\"\",\"format\":\"adai-native\"}";
    }
    body << "}";
    return body.str();
}

struct ParsedUrl {
    std::string host = "localhost";
    int port = 8083;

    static ParsedUrl from(const std::string& url) {
        ParsedUrl p;
        std::string s = url;
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
};

}  // namespace mns_gui
