#pragma once

#include <string>
#include <vector>

namespace mns_gui {

inline std::string json_value(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    while (pos < body.size() && body[pos] == ' ') ++pos;
    if (pos >= body.size()) return {};
    if (body[pos] == '"') {
        auto end = body.find('"', pos + 1);
        return (end != std::string::npos) ? body.substr(pos + 1, end - pos - 1) : "";
    }
    auto end = body.find_first_of(",}] \n", pos);
    return (end != std::string::npos) ? body.substr(pos, end - pos) : body.substr(pos);
}

inline std::vector<std::string> json_array_objects(const std::string& body,
                                                    const std::string& key) {
    std::vector<std::string> result;
    std::string needle = "\"" + key + "\":[";
    auto pos = body.find(needle);
    if (pos == std::string::npos) {
        pos = body.find('[');
        if (pos == std::string::npos) return result;
    } else {
        pos += needle.size();
    }

    int depth = 0;
    size_t obj_start = 0;
    for (size_t i = pos; i < body.size(); ++i) {
        if (body[i] == '{') {
            if (depth == 0) obj_start = i;
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
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += static_cast<char>(c); break;
        }
    }
    return out;
}

inline std::string json_pretty(const std::string& s) {
    std::string out;
    int indent = 0;
    bool in_string = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_string = !in_string;
        if (in_string) { out += c; continue; }
        switch (c) {
            case '{': case '[':
                out += c;
                out += '\n';
                ++indent;
                out += std::string(indent * 2, ' ');
                break;
            case '}': case ']':
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

struct ParsedUrl {
    std::string host = "localhost";
    int         port = 8083;

    static ParsedUrl from(const std::string& url) {
        ParsedUrl p;
        std::string s = url;
        if (s.rfind("http://", 0) == 0) s = s.substr(7);
        if (s.rfind("https://", 0) == 0) s = s.substr(8);
        auto slash = s.find('/');
        if (slash != std::string::npos) s = s.substr(0, slash);
        auto colon = s.find(':');
        if (colon != std::string::npos) {
            p.host = s.substr(0, colon);
            try { p.port = std::stoi(s.substr(colon + 1)); } catch (...) {}
        } else {
            p.host = s;
        }
        return p;
    }
};

}  // namespace mns_gui
