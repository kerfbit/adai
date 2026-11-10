#ifndef SPECIAL_TOKENS_HPP
#define SPECIAL_TOKENS_HPP

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-09


#include <string>

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
 * <system>    | 4 | System turn (TD-295)      | Written only by the prompt formatter, as an ID
 * <user>      | 5 | User turn (TD-295)        | Same
 * <assistant> | 6 | Assistant turn (TD-295)   | Same; also ends a prompt to cue the reply
 * <reserved_N>| 7-19 | Unassigned (TD-565)     | Held for future special tokens; never encoded, decoded or generated
 *
 * TD-565: ids 0-19 are reserved for special tokens in every new vocabulary, and ordinary tokens start
 * at 20, so a later special token takes a reserved id without moving ordinary tokens or resizing a
 * trained model's embedding and output layers. An unassigned id has a placeholder entry,
 * reserved_slot_token(id), that behaves as an unknown: text never encodes to it, decode() drops it and
 * generation masks it.
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
constexpr int SYSTEM_TURN = 4;     ///< TD-295: starts a system turn
constexpr int USER_TURN = 5;       ///< TD-295: starts a user turn
constexpr int ASSISTANT_TURN = 6;  ///< TD-295: starts an assistant turn
constexpr int EOS = 3;  ///< End of sequence - marks the end of a sequence and stops generation
constexpr int FIRST_UNASSIGNED = 7;  ///< TD-565: first reserved id with no special token yet
constexpr int RESERVED_SLOTS = 20;   ///< TD-565: ids 0-19 are special; ordinary tokens start here
}  // namespace SpecialTokenIDs

/// TD-565: the placeholder entry for an unassigned reserved id, e.g. "<reserved_7>".
inline std::string reserved_slot_token(int id) {
    return "<reserved_" + std::to_string(id) + ">";
}

/**
 * @brief Standard special token string representations
 */
namespace SpecialTokenStrings {
constexpr const char* PAD = "<pad>";  ///< Padding token string
constexpr const char* UNK = "<unk>";  ///< Unknown token string
constexpr const char* BOS = "<bos>";  ///< Beginning of sequence token string
constexpr const char* SYSTEM_TURN = "<system>";        ///< TD-295
constexpr const char* USER_TURN = "<user>";            ///< TD-295
constexpr const char* ASSISTANT_TURN = "<assistant>";  ///< TD-295
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
};

// ============================================================================
// Special Token Utility Functions
// ============================================================================

// TD-505: the id/string lookup and set/map helpers that used to be here were unused outside their
// tests and knew only ids 0-3, not the turn tokens or reserved ids (TD-295, TD-565). BPETokenizer is
// the source of truth: decode() skips every special token, first_ordinary_id() bounds them, and
// load_vocab() refuses a file whose ids 0-3 aren't the standard ones.

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
 * Decoder training target (EOS kept, leading BOS dropped, TD-361; the model's teacher-forcing
 * decoder input [BOS] + target[:-1] supplies the BOS):
 * @code
 * auto target_tokens = EncoderDecoderModel::strip_leading_bos(
 *     tokenizer->encode(target_text, true), tokenizer->get_bos_token_id());
 * // Result: [token1, token2, ..., EOS]
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
 * The current format (v4, TD-565), as save_vocab() writes it; tokens are escaped (\s for a space)
 * and separated from their ids by a tab. load_vocab() requires ids 0-3 as below (TD-505).
 * @code
 * # BPE Tokenizer Vocabulary v4.0
 * TOKENIZER_MODE UNICODE
 * NORMALIZATION CASED
 * WHITESPACE PRESERVE
 * VOCAB_SIZE 10000
 * RESERVED_SPECIAL_IDS 20
 * SPECIAL_TOKENS
 * pad_token_id 0
 * unk_token_id 1
 * bos_token_id 2
 * eos_token_id 3
 * VOCAB
 * <pad>         0
 * <unk>         1
 * <bos>         2
 * <eos>         3
 * <system>      4
 * <user>        5
 * <assistant>   6
 * <reserved_7>  7
 * ...
 * <reserved_19> 19
 * \s            20
 * ...
 * BPE_MERGES 9980
 * t  h
 * ...
 * @endcode
 */

}  // namespace adai

#endif  // SPECIAL_TOKENS_HPP
