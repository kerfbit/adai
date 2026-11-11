#pragma once

// @adai-status: beta
// @adai-version: 0.1.0
// @adai-reviewed: 2026-11-10


#include <string>
#include <string_view>
#include <vector>

namespace adai {

/**
 * @brief TD-254: the generation settings' allowed ranges and strategy names, in one place.
 *
 * ConfigLoader::validate() checks the server's defaults (MAX_GEN_LENGTH, TEMPERATURE, …) against these,
 * ChatbotAPI checks the same settings sent with a /chat or /chat/session request, and the `chatbot` CLI
 * checks `/set` values before sending them. Kept free of other dependencies so the lightweight `chatbot`
 * client can compile it in.
 */
struct GenerationLimits {
    static constexpr int kMinMaxLength = 1;
    static constexpr int kMaxMaxLength = 4096;
    static constexpr float kMinTemperature = 0.0f;
    static constexpr float kMaxTemperature = 2.0f;
    static constexpr float kMinTopP = 0.0f;
    static constexpr float kMaxTopP = 1.0f;
    static constexpr int kMinTopK = 1;
    static constexpr int kMaxTopK = 1000;
    static constexpr int kMinBeamWidth = 1;
    static constexpr int kMaxBeamWidth = 16;
};

/// The server's strategy names: greedy, beam, temperature, top_k, nucleus.
const std::vector<std::string>& generation_strategy_names();

/// The server's name for a strategy, matched in any case; also accepts the `chatbot` CLI's old names
/// (`sampling` for temperature, `top-k`/`topk` for top_k) and `top_p`/`top-p` for nucleus. Empty when the
/// name isn't a strategy.
std::string canonical_generation_strategy(std::string_view name);

}  // namespace adai
