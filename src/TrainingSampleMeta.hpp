#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


#include <cctype>
#include <sstream>
#include <string>

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

inline std::string json_str(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos)
        return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos)
        return "";
    size_t vpos = colon + 1;
    while (vpos < json.size() && std::isspace(static_cast<unsigned char>(json[vpos])))
        ++vpos;
    if (vpos >= json.size() || json[vpos] != '"')
        return "";
    std::string raw;
    for (++vpos; vpos < json.size() && json[vpos] != '"'; ++vpos) {
        if (json[vpos] == '\\' && vpos + 1 < json.size()) {
            ++vpos;
            switch (json[vpos]) {
                case 'n':
                    raw += '\n';
                    break;
                case 't':
                    raw += '\t';
                    break;
                case 'r':
                    raw += '\r';
                    break;
                default:
                    raw += json[vpos];
                    break;
            }
        } else {
            raw += json[vpos];
        }
    }
    return raw;
}

inline float json_flt(const std::string& json, const std::string& key, float def) {
    const std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos)
        return def;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos)
        return def;
    size_t vpos = colon + 1;
    while (vpos < json.size() && std::isspace(static_cast<unsigned char>(json[vpos])))
        ++vpos;
    if (vpos >= json.size())
        return def;
    try {
        return std::stof(json.substr(vpos));
    } catch (...) {
        return def;
    }
}

inline int json_inum(const std::string& json, const std::string& key, int def) {
    const std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos)
        return def;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos)
        return def;
    size_t vpos = colon + 1;
    while (vpos < json.size() && std::isspace(static_cast<unsigned char>(json[vpos])))
        ++vpos;
    if (vpos >= json.size())
        return def;
    try {
        return std::stoi(json.substr(vpos));
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
    using tsm_detail::json_flt;
    using tsm_detail::json_inum;
    using tsm_detail::json_str;
    input = json_str(line, "input");
    response = json_str(line, "response");
    if (input.empty())
        return false;
    meta.domain = json_str(line, "domain");
    meta.task_type = json_str(line, "task_type");
    meta.language = json_str(line, "language");
    meta.split = json_str(line, "split");
    meta.quality = json_flt(line, "quality", -1.0f);
    meta.weight = json_flt(line, "weight", 1.0f);
    meta.token_count = json_inum(line, "token_count", -1);
    return true;
}
