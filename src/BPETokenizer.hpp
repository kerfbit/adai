
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "SpecialTokens.hpp"

// Custom exception types for BPE Tokenizer errors

/**
 * @brief Exception thrown when input string is empty or invalid
 */
class TokenizerInputError : public std::invalid_argument {
   public:
    explicit TokenizerInputError(const std::string& message)
        : std::invalid_argument("Tokenizer Input Error: " + message) {}
};

/**
 * @brief Exception thrown when UTF-8 encoding is invalid
 */
class TokenizerEncodingError : public std::runtime_error {
   public:
    explicit TokenizerEncodingError(const std::string& message)
        : std::runtime_error("Tokenizer Encoding Error: " + message) {}
};

/**
 * @brief Exception thrown when vocabulary file is malformed
 */
class VocabularyFileError : public std::runtime_error {
   public:
    explicit VocabularyFileError(const std::string& message)
        : std::runtime_error("Vocabulary File Error: " + message) {}
};

/**
 * @brief Exception thrown when token IDs are out of range
 */
class TokenIDError : public std::out_of_range {
   public:
    explicit TokenIDError(const std::string& message)
        : std::out_of_range("Token ID Error: " + message) {}
};

/**
 * @brief Controls whether the tokenizer operates on raw bytes or UTF-8 code points.
 *
 * ASCII   – byte-level mode; each byte is one unit. Non-ASCII input passes through
 *           but multi-byte sequences are split into individual raw-byte tokens.
 * UNICODE – code-point mode; multi-byte UTF-8 sequences are kept intact as a
 *           single atomic unit, so BPE can learn meaningful merges across all scripts.
 */
enum class TokenizerMode {
    ASCII,
    UNICODE,
};

class BPETokenizer {
   private:
    TokenizerMode mode;

    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inverse_vocab;
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    std::unordered_set<std::string> special_tokens;

    // Special token IDs (using standard definitions from SpecialTokens.hpp)
    int pad_token_id = adai::SpecialTokenIDs::PAD;
    int unk_token_id = adai::SpecialTokenIDs::UNK;
    int bos_token_id = adai::SpecialTokenIDs::BOS;
    int eos_token_id = adai::SpecialTokenIDs::EOS;

    // Regex pattern for pre-tokenization (chosen at construction time based on mode)
    std::regex token_pattern;

   public:
    /**
     * @brief Construct a BPETokenizer.
     * @param mode  ASCII (default) for byte-level tokenization; UNICODE for
     *              UTF-8 code-point-level tokenization.
     */
    explicit BPETokenizer(TokenizerMode mode = TokenizerMode::ASCII);

    // Query the active tokenization mode
    TokenizerMode get_mode() const { return mode; }
    bool is_unicode_mode() const { return mode == TokenizerMode::UNICODE; }

    // Build vocabulary from text corpus
    void build_vocab(const std::vector<std::string>& texts, int vocab_size = 10000,
                     int frequency_threshold = 1);

    // Build BPE merge rules
    void build_bpe_merges(const std::vector<std::string>& texts, int num_merges);

    // Pre-tokenization step
    std::vector<std::string> pre_tokenize(const std::string& text);

    // Find most frequent adjacent pair
    static std::pair<std::string, std::string> get_most_frequent_pair(
        const std::vector<std::vector<std::string>>& word_tokens);

    // Apply merge to word tokens
    static void merge_tokens(std::vector<std::vector<std::string>>& word_tokens,
                             const std::string& first, const std::string& second);

    // Apply BPE encoding to a single word
    std::vector<std::string> apply_bpe(const std::string& word);

    // Tokenize text into subword tokens
    std::vector<std::string> tokenize(const std::string& text);

    // Convert tokens to IDs
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true);

    // Convert IDs back to text
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true);

    // Get vocabulary size
    size_t get_vocab_size() const;

    // Get special token IDs
    int get_bos_token_id() const { return bos_token_id; }
    int get_eos_token_id() const { return eos_token_id; }
    int get_pad_token_id() const { return pad_token_id; }
    int get_unk_token_id() const { return unk_token_id; }

    // Save vocabulary to file
    void save_vocab(const std::string& filename) const;

    // Load vocabulary from file (restores the saved TokenizerMode automatically)
    void load_vocab(const std::string& filename);

    // Print vocabulary statistics
    void print_vocab_stats() const;

    // Get top-k most frequent tokens (for debugging)
    std::vector<std::pair<std::string, int>> get_top_tokens(int k = 10) const;

    /**
     * @brief Recommend a vocabulary size based on corpus characteristics and model architecture.
     *
     * Combines five factors:
     *   1. Dataset size   — larger corpora support larger, more useful vocabularies.
     *   2. Script family  — CJK characters are already atomic (needs smaller vocab);
     *                       Arabic/agglutinative scripts need larger vocab for morphology.
     *   3. Architecture   — embedding table (V × d_model) is capped at ≤30% of total params.
     *   4. Sequence length — tighter context budgets benefit from lower token fertility
     *                        (more words per sequence), which means a larger vocabulary.
     *   5. Tokenizer mode — Unicode mode with multibyte text gets a small upward adjustment
     *                        to ensure adequate code-point coverage.
     *
     * @param texts              Training corpus texts.
     * @param d_model            Model embedding dimension.
     * @param num_encoder_layers Number of encoder layers.
     * @param num_decoder_layers Number of decoder layers.
     * @param max_seq_length     Maximum sequence length the model will process.
     * @param mode               ASCII or UNICODE tokenizer mode.
     * @return Recommended vocabulary size (multiple of 500, minimum 2000).
     */
    static int recommend_vocab_size(const std::vector<std::string>& texts,
                                    int d_model,
                                    int num_encoder_layers,
                                    int num_decoder_layers,
                                    int max_seq_length,
                                    TokenizerMode mode = TokenizerMode::ASCII);

    /**
     * @brief Measure token fertility (average BPE tokens per whitespace word).
     *
     * A value in [1.2, 1.8] is generally healthy; above 2.5 indicates the vocabulary
     * is too small for the corpus; below 1.1 indicates possible over-fitting or a
     * very restricted vocabulary relative to the text.
     *
     * @param texts       Texts to evaluate (a random sample suffices).
     * @param sample_limit Maximum number of texts to evaluate (default 500).
     * @return Fertility ratio, or 0 if the tokenizer has no vocabulary.
     */
    float measure_fertility(const std::vector<std::string>& texts, int sample_limit = 500) const;

   private:
    // Split a UTF-8 string into one std::string per code point.
    // In ASCII mode the tokenizer calls this as well, but each "code point" is a
    // single byte, so the behaviour is identical to the old char loop.
    static std::vector<std::string> utf8_split_codepoints(const std::string& str);

    // UTF-8 validation helper
    static bool is_valid_utf8(const std::string& text);

    // Validate input string
    static void validate_input(const std::string& text, const std::string& context);
};
