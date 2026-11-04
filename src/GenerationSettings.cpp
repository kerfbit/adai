// @adai-status: experimental
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-03

#include "GenerationSettings.hpp"
#include <algorithm>
#include <cctype>

namespace adai {

const std::vector<std::string>& generation_strategy_names() {
    static const std::vector<std::string> names = {"greedy", "beam", "temperature", "top_k", "nucleus"};
    return names;
}

std::string canonical_generation_strategy(std::string_view name) {
    std::string lower(name);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const auto& known : generation_strategy_names()) {
        if (lower == known) return known;
    }
    if (lower == "sampling") return "temperature";
    if (lower == "top-k" || lower == "topk") return "top_k";
    if (lower == "top_p" || lower == "top-p") return "nucleus";
    return "";
}

}  // namespace adai
