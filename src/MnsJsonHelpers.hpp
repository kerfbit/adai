#pragma once

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-10

#include <deque>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "nlohmann/json.hpp"

// The MNS Manager GUI's JSON helpers. TD-452: they parse with the vendored nlohmann/json (an
// ordered_json, so a pretty-printed record keeps the server's field order) instead of scanning the
// text, so values are unescaped, any spacing is accepted, an object value comes back whole, a
// missing array is empty rather than whichever array came first, and escaping covers every control
// character. Malformed input gives empty results, never an exception.
namespace mns_gui {

namespace detail {

inline nlohmann::ordered_json parse_or_discard(const std::string& body) {
    return nlohmann::ordered_json::parse(body, nullptr, /*allow_exceptions=*/false);
}

// The value of `key` nearest the top of `doc` (breadth-first), or nullptr if there is none. Nearest
// first, so a record's own "run_id" wins over one in its training_history.
inline const nlohmann::ordered_json* find_key(const nlohmann::ordered_json& doc,
                                              const std::string& key) {
    std::deque<const nlohmann::ordered_json*> queue{&doc};
    while (!queue.empty()) {
        const nlohmann::ordered_json* node = queue.front();
        queue.pop_front();
        if (node->is_object()) {
            auto it = node->find(key);
            if (it != node->end()) {
                return &*it;
            }
        }
        if (node->is_structured()) {
            for (const auto& child : *node) {
                queue.push_back(&child);
            }
        }
    }
    return nullptr;
}

inline std::string compact(const nlohmann::ordered_json& value) {
    return value.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

}  // namespace detail

/// The value of `key` (the one nearest the top of the document): a string unescaped, anything else
/// as compact JSON (`512`, `true`, `null`, `{"host":"h"}`). Empty when the key is missing or the
/// body isn't JSON.
inline std::string json_value(const std::string& body, const std::string& key) {
    const auto doc = detail::parse_or_discard(body);
    const auto* value = detail::find_key(doc, key);
    if (!value) {
        return {};
    }
    return value->is_string() ? value->get<std::string>() : detail::compact(*value);
}

/// The elements of the array under `key`, each as compact JSON. An empty `key` means the body
/// itself is the array. Empty when there is no such array or the body isn't JSON.
inline std::vector<std::string> json_array_objects(const std::string& body,
                                                   const std::string& key) {
    std::vector<std::string> result;
    const auto doc = detail::parse_or_discard(body);
    const auto* array = key.empty() ? &doc : detail::find_key(doc, key);
    if (!array || !array->is_array()) {
        return result;
    }
    for (const auto& element : *array) {
        result.push_back(detail::compact(element));
    }
    return result;
}

/// `s` escaped for use inside a JSON string literal (without the surrounding quotes): quotes,
/// backslashes and every control character; invalid UTF-8 becomes U+FFFD.
inline std::string json_escape(const std::string& s) {
    const std::string quoted = detail::compact(nlohmann::ordered_json(s));
    return quoted.substr(1, quoted.size() - 2);
}

/// `s` indented by two spaces per level, in the server's field order; text that isn't JSON is
/// returned unchanged.
inline std::string json_pretty(const std::string& s) {
    const auto doc = detail::parse_or_discard(s);
    if (doc.is_discarded()) {
        return s;
    }
    return doc.dump(2, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

// TD-037: parses MnsManagerGUI's "Register" tab tags field (comma-separated `key=value` pairs,
// arbitrary surrounding whitespace around each key/value) into ordered pairs. An entry with no '='
// is skipped.
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

// TD-454: builds PUT /models/{name}/state's JSON body for the "training" transition. MNS allocates
// the run ID (run-NN), so the body carries none; new_run=true starts the next run instead of
// continuing the current one (the first run ever is run-01 either way).
inline std::string build_set_training_body(bool new_run) {
    return new_run ? R"({"state":"training","new_run":true})" : R"({"state":"training"})";
}

// TD-037: builds PUT /models/{name}/state's JSON body for the "candidate" transition.
// run_id/artifact_path are both optional (either may be empty).
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

}  // namespace mns_gui
