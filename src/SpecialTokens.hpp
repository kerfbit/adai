#ifndef SPECIAL_TOKENS_HPP
#define SPECIAL_TOKENS_HPP

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

/**
 * @file SpecialTokens.hpp
 * @brief Centralized header-only special token handling for ADAI project
 *
 * This header consolidates all special token definitions, constants, and utility
 * functions used throughout the ADAI codebase. It ensures consistency in special
 * token handling across tokenizers, models, generators, and data processing components.
 *
 * Special Token Standard:
 * ----------------------
 * Token   | ID | Purpose                      | Usage
 * --------|----|-----------------------------|------------------------------------
 * <pad>   | 0  | Padding sequences           | Fill sequences to uniform length
 * <unk>   | 1  | Unknown tokens              | Replace out-of-vocabulary tokens
 * <bos>   | 2  | Beginning of sequence       | Mark sequence start
 * <eos>   | 3  | End of sequence             | Mark sequence end, stop generation
 *
 * @note These IDs are standardized across the entire codebase and must not be changed
 *       without updating all vocabulary files and retrained models.
 */

namespace adai {

// ============================================================================
// Special Token ID Constants
// ============================================================================

/**
 * @brief Standard special token IDs used throughout the ADAI system
 *
 * These constants define the canonical token IDs for special tokens.
 * DO NOT modify these values without retraining models and rebuilding vocabularies.
 */
namespace SpecialTokenIDs {
constexpr int PAD = 0;  ///< Padding token ID - used to fill sequences to uniform length
constexpr int UNK = 1;  ///< Unknown token ID - used for out-of-vocabulary tokens
constexpr int BOS = 2;  ///< Beginning of sequence - marks the start of a sequence
constexpr int EOS = 3;  ///< End of sequence - marks the end of a sequence and stops generation
}  // namespace SpecialTokenIDs

/**
 * @brief Standard special token string representations
 */
namespace SpecialTokenStrings {
constexpr const char* PAD = "<pad>";  ///< Padding token string
constexpr const char* UNK = "<unk>";  ///< Unknown token string
constexpr const char* BOS = "<bos>";  ///< Beginning of sequence token string
constexpr const char* EOS = "<eos>";  ///< End of sequence token string
}  // namespace SpecialTokenStrings

// ============================================================================
// Special Token Configuration Structure
// ============================================================================

/**
 * @brief Configuration structure for special token IDs
 *
 * This structure encapsulates all special token IDs in a single object
 * for easy passing between components and serialization.
 *
 * Usage example:
 * @code
 * SpecialTokenConfig config;  // Uses default IDs
 * auto bos = config.get_bos_token_id();
 *
 * // Or create from existing values
 * SpecialTokenConfig custom_config(0, 1, 2, 3);
 * @endcode
 */
struct SpecialTokenConfig {
    int pad_token_id;  ///< ID for padding token
    int unk_token_id;  ///< ID for unknown token
    int bos_token_id;  ///< ID for beginning of sequence token
    int eos_token_id;  ///< ID for end of sequence token

    /**
     * @brief Default constructor with standard token IDs
     */
    SpecialTokenConfig()
        : pad_token_id(SpecialTokenIDs::PAD),
          unk_token_id(SpecialTokenIDs::UNK),
          bos_token_id(SpecialTokenIDs::BOS),
          eos_token_id(SpecialTokenIDs::EOS) {}

    /**
     * @brief Constructor with custom token IDs
     * @param pad_id Padding token ID
     * @param unk_id Unknown token ID
     * @param bos_id Beginning of sequence token ID
     * @param eos_id End of sequence token ID
     */
    SpecialTokenConfig(int pad_id, int unk_id, int bos_id, int eos_id)
        : pad_token_id(pad_id), unk_token_id(unk_id), bos_token_id(bos_id), eos_token_id(eos_id) {}

    // Getter methods for compatibility with existing code
    int get_pad_token_id() const {
        return pad_token_id;
    }
    int get_unk_token_id() const {
        return unk_token_id;
    }
    int get_bos_token_id() const {
        return bos_token_id;
    }
    int get_eos_token_id() const {
        return eos_token_id;
    }

    /**
     * @brief Validate that token IDs are non-negative and unique
     * @throws std::invalid_argument if validation fails
     */
    void validate() const {
        if (pad_token_id < 0 || unk_token_id < 0 || bos_token_id < 0 || eos_token_id < 0) {
            throw std::invalid_argument("Special token IDs must be non-negative");
        }

        std::unordered_set<int> ids = {pad_token_id, unk_token_id, bos_token_id, eos_token_id};
        if (ids.size() != 4) {
            throw std::invalid_argument("Special token IDs must be unique");
        }
    }
};

// ============================================================================
// Special Token Utility Functions
// ============================================================================

/**
 * @brief Check if a token ID corresponds to any special token
 * @param token_id The token ID to check
 * @param config Special token configuration (uses defaults if not provided)
 * @return true if the token is a special token, false otherwise
 */
inline bool is_special_token(int token_id,
                             const SpecialTokenConfig& config = SpecialTokenConfig()) {
    return token_id == config.pad_token_id || token_id == config.unk_token_id ||
           token_id == config.bos_token_id || token_id == config.eos_token_id;
}

/**
 * @brief Check if a token string corresponds to any special token
 * @param token_str The token string to check
 * @return true if the token is a special token string, false otherwise
 */
inline bool is_special_token_string(const std::string& token_str) {
    return token_str == SpecialTokenStrings::PAD || token_str == SpecialTokenStrings::UNK ||
           token_str == SpecialTokenStrings::BOS || token_str == SpecialTokenStrings::EOS;
}

/**
 * @brief Check if a token ID is a stop token (EOS or PAD)
 *
 * Stop tokens indicate that generation should terminate or a sequence has ended.
 * This is commonly used in generation loops to determine when to stop.
 *
 * @param token_id The token ID to check
 * @param config Special token configuration (uses defaults if not provided)
 * @return true if the token is EOS or PAD, false otherwise
 */
inline bool is_stop_token(int token_id, const SpecialTokenConfig& config = SpecialTokenConfig()) {
    return token_id == config.eos_token_id || token_id == config.pad_token_id;
}

/**
 * @brief Get the string representation of a special token ID
 * @param token_id The special token ID
 * @param config Special token configuration (uses defaults if not provided)
 * @return The string representation (e.g., "<bos>")
 * @throws std::invalid_argument if token_id is not a special token
 */
inline std::string get_special_token_string(
    int token_id, const SpecialTokenConfig& config = SpecialTokenConfig()) {
    if (token_id == config.pad_token_id)
        return SpecialTokenStrings::PAD;
    if (token_id == config.unk_token_id)
        return SpecialTokenStrings::UNK;
    if (token_id == config.bos_token_id)
        return SpecialTokenStrings::BOS;
    if (token_id == config.eos_token_id)
        return SpecialTokenStrings::EOS;

    throw std::invalid_argument("Token ID " + std::to_string(token_id) + " is not a special token");
}

/**
 * @brief Get the token ID for a special token string
 * @param token_str The special token string (e.g., "<bos>")
 * @param config Special token configuration (uses defaults if not provided)
 * @return The token ID
 * @throws std::invalid_argument if token_str is not a special token string
 */
inline int get_special_token_id(const std::string& token_str,
                                const SpecialTokenConfig& config = SpecialTokenConfig()) {
    if (token_str == SpecialTokenStrings::PAD)
        return config.pad_token_id;
    if (token_str == SpecialTokenStrings::UNK)
        return config.unk_token_id;
    if (token_str == SpecialTokenStrings::BOS)
        return config.bos_token_id;
    if (token_str == SpecialTokenStrings::EOS)
        return config.eos_token_id;

    throw std::invalid_argument("Token string '" + token_str + "' is not a special token");
}

/**
 * @brief Create a set of special token strings
 * @return An unordered set containing all special token strings
 */
inline std::unordered_set<std::string> create_special_token_set() {
    return {SpecialTokenStrings::PAD, SpecialTokenStrings::UNK, SpecialTokenStrings::BOS,
            SpecialTokenStrings::EOS};
}

/**
 * @brief Create a map from special token strings to their IDs
 * @param config Special token configuration (uses defaults if not provided)
 * @return An unordered map of token string -> token ID
 */
inline std::unordered_map<std::string, int> create_special_token_map(
    const SpecialTokenConfig& config = SpecialTokenConfig()) {
    return {{SpecialTokenStrings::PAD, config.pad_token_id},
            {SpecialTokenStrings::UNK, config.unk_token_id},
            {SpecialTokenStrings::BOS, config.bos_token_id},
            {SpecialTokenStrings::EOS, config.eos_token_id}};
}

/**
 * @brief Create an inverse map from special token IDs to their strings
 * @param config Special token configuration (uses defaults if not provided)
 * @return An unordered map of token ID -> token string
 */
inline std::unordered_map<int, std::string> create_inverse_special_token_map(
    const SpecialTokenConfig& config = SpecialTokenConfig()) {
    return {{config.pad_token_id, SpecialTokenStrings::PAD},
            {config.unk_token_id, SpecialTokenStrings::UNK},
            {config.bos_token_id, SpecialTokenStrings::BOS},
            {config.eos_token_id, SpecialTokenStrings::EOS}};
}

// ============================================================================
// Usage Pattern Documentation
// ============================================================================

/**
 * @example Encoding with special tokens
 *
 * When encoding text for encoder-decoder models:
 *
 * Encoder input (typically no special tokens):
 * @code
 * auto input_tokens = tokenizer->encode(input_text, false);
 * @endcode
 *
 * Decoder target (with special tokens for teacher forcing):
 * @code
 * auto target_tokens = tokenizer->encode(target_text, true);
 * // Result: [BOS, token1, token2, ..., EOS]
 * @endcode
 */

/**
 * @example Decoding with special token filtering
 *
 * When decoding generated token IDs to text:
 *
 * Skip special tokens (typical for output):
 * @code
 * std::string output = tokenizer->decode(generated_tokens, true);
 * // Special tokens like <bos>, <eos>, <pad> are removed
 * @endcode
 *
 * Keep special tokens (for debugging):
 * @code
 * std::string debug_output = tokenizer->decode(generated_tokens, false);
 * // Output includes <bos>, <eos>, etc.
 * @endcode
 */

/**
 * @example Generation termination
 *
 * When generating text, check for stop tokens:
 * @code
 * SpecialTokenConfig config;
 * for (int step = 0; step < max_steps; ++step) {
 *     int next_token = sample_from_logits(logits);
 *     generated_tokens.push_back(next_token);
 *
 *     // Stop if we hit EOS or PAD
 *     if (is_stop_token(next_token, config)) {
 *         break;
 *     }
 * }
 * @endcode
 */

/**
 * @example Synchronizing special tokens between components
 *
 * When setting up a model with a tokenizer:
 * @code
 * BPETokenizer* tokenizer = new BPETokenizer();
 * model->set_tokenizer(tokenizer);
 *
 * // Synchronize special token IDs from tokenizer to model/generator config
 * TextGenerator::GenerationConfig gen_config;
 * gen_config.bos_token_id = tokenizer->get_bos_token_id();
 * gen_config.eos_token_id = tokenizer->get_eos_token_id();
 * gen_config.pad_token_id = tokenizer->get_pad_token_id();
 * gen_config.unk_token_id = tokenizer->get_unk_token_id();
 * model->set_generation_config(gen_config);
 * @endcode
 */

/**
 * @example Vocabulary file format
 *
 * Special tokens must be defined in vocabulary files:
 * @code
 * # BPE Tokenizer Vocabulary v1.0
 * VOCAB_SIZE 10000
 * SPECIAL_TOKENS
 * pad_token_id 0
 * unk_token_id 1
 * bos_token_id 2
 * eos_token_id 3
 * VOCAB
 * <pad>    0
 * <unk>    1
 * <bos>    2
 * <eos>    3
 * token1   4
 * ...
 * @endcode
 */

}  // namespace adai

#endif  // SPECIAL_TOKENS_HPP
