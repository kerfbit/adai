// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-10-24

#include "BrowserWriteGuard.hpp"

#include <algorithm>
#include <cctype>

namespace adai {

namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos)
        return "";
    const auto last = s.find_last_not_of(" \t");
    return s.substr(first, last - first + 1);
}

}  // namespace

bool is_write_method(const std::string& method) {
    return method == "POST" || method == "PUT" || method == "PATCH" || method == "DELETE";
}

bool is_cors_simple_content_type(const std::string& content_type) {
    std::string media = trim(content_type.substr(0, content_type.find(';')));
    std::transform(media.begin(), media.end(), media.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return media == "text/plain" || media == "application/x-www-form-urlencoded" ||
           media == "multipart/form-data";
}

std::optional<BrowserWriteRefusal> browser_write_refusal(
    const std::string& method, const std::string& origin, const std::string& content_type,
    const std::set<std::string>& allowed_origins) {
    if (!is_write_method(method))
        return std::nullopt;
    if (!origin.empty() && allowed_origins.count(origin) == 0)
        return BrowserWriteRefusal{403, "cross-origin writes are not allowed from '" + origin + "'"};
    if (is_cors_simple_content_type(content_type))
        return BrowserWriteRefusal{415, "request bodies must not be sent as '" +
                                            trim(content_type.substr(0, content_type.find(';'))) +
                                            "'; use application/json"};
    return std::nullopt;
}

std::string browser_write_refusal_json(const BrowserWriteRefusal& refusal) {
    std::string out = "{\"error\":\"";
    for (char ch : refusal.message) {
        if (ch == '"' || ch == '\\')
            out += '\\';
        if (static_cast<unsigned char>(ch) >= 0x20)
            out += ch;
    }
    out += "\"}";
    return out;
}

}  // namespace adai
