#pragma once

// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-17


#include <cctype>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Optional metadata attached to a training sample.
 *
 * All fields are optional.  Unset strings are empty; unset numerics use
 * sentinel values: quality < 0 means "not set", token_count < 0 means
 * "not computed", weight defaults to 1.0 (normal sampling).
 */
struct SampleMeta {
    std::string domain;     ///< Thematic domain: "fiction", "qa", "code", "dialogue", …
    std::string task_type;  ///< Task category: "chat", "instruction", "summarization", …
    std::string language;   ///< BCP-47 language code, e.g. "en", "fr"
    std::string split;      ///< Intended split: "train", "val", "test"
    float quality = -1.0f;  ///< Quality score [0, 1]; negative = not set
    float weight = 1.0f;    ///< Sampling weight (1.0 = unweighted)
    int token_count = -1;   ///< Pre-computed token count; negative = not computed
};

// ── Internal helpers ──────────────────────────────────────────────────────────
namespace tsm_detail {

// TODO: See TD-533 in TECHNICAL_DEBT.md - other control characters are written raw (invalid JSON for standard parsers).
inline std::string json_esc(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':
                o += "\\\"";
                break;
            case '\\':
                o += "\\\\";
                break;
            case '\n':
                o += "\\n";
                break;
            case '\r':
                o += "\\r";
                break;
            case '\t':
                o += "\\t";
                break;
            default:
                o += c;
        }
    }
    return o;
}

// TD-531/TD-532: a small JSON object reader for one JSONL line. It returns only top-level
// members (nested objects/arrays are skipped, so {"meta":{"input":"x"},"input":"y"} reads "y"),
// decodes every JSON escape including \uXXXX and surrogate pairs to UTF-8, and is deliberately
// lenient where strict parsers aren't: raw control characters and invalid UTF-8 inside strings
// pass through unchanged (sample_to_jsonl() writes raw control characters, TD-533), and a
// malformed line keeps the members read before the error. A repeated key keeps its last value,
// as Python's json and nlohmann/json do.
struct JsonMember {
    bool is_string = false;  ///< true: `value` is the decoded string; false: the raw token
    std::string value;       ///< number/true/false/null token; empty for a skipped object/array
};

inline void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/// Reads 4 hex digits at s[pos]; returns false if they aren't all hex.
inline bool read_hex4(const std::string& s, size_t pos, unsigned& out) {
    if (pos + 4 > s.size())
        return false;
    out = 0;
    for (size_t i = pos; i < pos + 4; ++i) {
        const char c = s[i];
        out <<= 4;
        if (c >= '0' && c <= '9')
            out |= static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f')
            out |= static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            out |= static_cast<unsigned>(c - 'A' + 10);
        else
            return false;
    }
    return true;
}

/// Parses the string starting at s[pos] == '"'; on success sets pos past the closing quote.
inline bool parse_string(const std::string& s, size_t& pos, std::string& out) {
    out.clear();
    for (size_t i = pos + 1; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '"') {
            pos = i + 1;
            return true;
        }
        if (c != '\\') {
            out += c;
            continue;
        }
        if (++i >= s.size())
            return false;
        switch (s[i]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                unsigned cp = 0;
                if (!read_hex4(s, i + 1, cp)) {
                    out += "\\u";  // not a valid escape: keep it literally
                    break;
                }
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    unsigned lo = 0;
                    if (i + 2 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u' &&
                        read_hex4(s, i + 3, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        i += 6;
                    } else {
                        cp = 0xFFFD;  // unpaired high surrogate
                    }
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    cp = 0xFFFD;  // unpaired low surrogate
                }
                append_utf8(out, cp);
                break;
            }
            default:
                out += s[i];  // unknown escape: keep the character, as before
                break;
        }
    }
    return false;  // unterminated
}

inline void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])))
        ++pos;
}

/// Skips a nested object/array starting at s[pos] ('{' or '['), honouring strings.
inline bool skip_container(const std::string& s, size_t& pos) {
    int depth = 0;
    std::string scratch;
    while (pos < s.size()) {
        const char c = s[pos];
        if (c == '"') {
            if (!parse_string(s, pos, scratch))
                return false;
            continue;
        }
        if (c == '{' || c == '[') {
            ++depth;
        } else if (c == '}' || c == ']') {
            if (--depth == 0) {
                ++pos;
                return true;
            }
        }
        ++pos;
    }
    return false;
}

/// Top-level members of the JSON object on `line`, in order (later duplicates override earlier
/// ones on lookup, see find_member()).
inline std::vector<std::pair<std::string, JsonMember>> parse_object_members(
    const std::string& line) {
    std::vector<std::pair<std::string, JsonMember>> members;
    size_t pos = 0;
    skip_ws(line, pos);
    if (pos >= line.size() || line[pos] != '{')
        return members;
    ++pos;
    std::string key;
    while (true) {
        skip_ws(line, pos);
        if (pos >= line.size() || line[pos] == '}')
            return members;
        if (line[pos] != '"' || !parse_string(line, pos, key))
            return members;
        skip_ws(line, pos);
        if (pos >= line.size() || line[pos] != ':')
            return members;
        ++pos;
        skip_ws(line, pos);
        if (pos >= line.size())
            return members;
        JsonMember m;
        const char c = line[pos];
        if (c == '"') {
            if (!parse_string(line, pos, m.value))
                return members;
            m.is_string = true;
        } else if (c == '{' || c == '[') {
            if (!skip_container(line, pos))
                return members;
        } else {
            const size_t begin = pos;
            while (pos < line.size() && line[pos] != ',' && line[pos] != '}' &&
                   !std::isspace(static_cast<unsigned char>(line[pos])))
                ++pos;
            m.value = line.substr(begin, pos - begin);
        }
        members.emplace_back(key, std::move(m));
        skip_ws(line, pos);
        if (pos < line.size() && line[pos] == ',')
            ++pos;
        else
            return members;
    }
}

inline const JsonMember* find_member(
    const std::vector<std::pair<std::string, JsonMember>>& members, const std::string& key) {
    for (auto it = members.rbegin(); it != members.rend(); ++it)
        if (it->first == key)
            return &it->second;
    return nullptr;
}

inline std::string member_str(const std::vector<std::pair<std::string, JsonMember>>& members,
                              const std::string& key) {
    const JsonMember* m = find_member(members, key);
    return (m && m->is_string) ? m->value : std::string();
}

inline float member_flt(const std::vector<std::pair<std::string, JsonMember>>& members,
                        const std::string& key, float def) {
    const JsonMember* m = find_member(members, key);
    if (!m || m->is_string || m->value.empty())
        return def;
    try {
        return std::stof(m->value);
    } catch (...) {
        return def;
    }
}

inline int member_inum(const std::vector<std::pair<std::string, JsonMember>>& members,
                       const std::string& key, int def) {
    const JsonMember* m = find_member(members, key);
    if (!m || m->is_string || m->value.empty())
        return def;
    try {
        return std::stoi(m->value);
    } catch (...) {
        return def;
    }
}

}  // namespace tsm_detail

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * @brief Serialize an input/response pair (plus optional metadata) to a JSONL line.
 *
 * The returned string does NOT include a trailing newline.
 *
 * @code
 *   SampleMeta m; m.task_type = "qa"; m.language = "en"; m.quality = 0.9f;
 *   out << sample_to_jsonl("What is X?", "X is …", m) << '\n';
 * @endcode
 */
inline std::string sample_to_jsonl(const std::string& input, const std::string& response,
                                   const SampleMeta& meta = {}) {
    using tsm_detail::json_esc;
    std::ostringstream o;
    o << "{\"input\":\"" << json_esc(input) << "\",\"response\":\"" << json_esc(response) << "\"";
    if (!meta.domain.empty())
        o << ",\"domain\":\"" << json_esc(meta.domain) << "\"";
    if (!meta.task_type.empty())
        o << ",\"task_type\":\"" << json_esc(meta.task_type) << "\"";
    if (!meta.language.empty())
        o << ",\"language\":\"" << json_esc(meta.language) << "\"";
    if (!meta.split.empty())
        o << ",\"split\":\"" << json_esc(meta.split) << "\"";
    if (meta.quality >= 0.0f)
        o << ",\"quality\":" << meta.quality;
    if (meta.weight != 1.0f)
        o << ",\"weight\":" << meta.weight;
    if (meta.token_count >= 0)
        o << ",\"token_count\":" << meta.token_count;
    o << "}";
    return o.str();
}

/**
 * @brief Parse a JSONL line into input, response, and optional metadata.
 *
 * @return true if the line contained a valid "input" field; false otherwise.
 */
inline bool parse_jsonl_sample(const std::string& line, std::string& input, std::string& response,
                               SampleMeta& meta) {
    using tsm_detail::member_flt;
    using tsm_detail::member_inum;
    using tsm_detail::member_str;
    const auto members = tsm_detail::parse_object_members(line);
    input = member_str(members, "input");
    response = member_str(members, "response");
    if (input.empty())
        return false;
    meta.domain = member_str(members, "domain");
    meta.task_type = member_str(members, "task_type");
    meta.language = member_str(members, "language");
    meta.split = member_str(members, "split");
    meta.quality = member_flt(members, "quality", -1.0f);
    meta.weight = member_flt(members, "weight", 1.0f);
    meta.token_count = member_inum(members, "token_count", -1);
    return true;
}
