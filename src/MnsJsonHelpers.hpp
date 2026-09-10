#pragma once

// @adai-status: stable
// @adai-version: 1.0.1
// @adai-reviewed: 2026-09-10


#include <string>
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
