
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

class BPETokenizer {
   private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inverse_vocab;
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    std::unordered_set<std::string> special_tokens;

    // Special token IDs
    int pad_token_id = 0;
    int unk_token_id = 1;
    int bos_token_id = 2;
    int eos_token_id = 3;

    // Regex pattern for pre-tokenization
    std::regex token_pattern;

   public:
    BPETokenizer()
        : token_pattern(
              R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|\s+(?!\S)|\s+)") {
        // Initialize special tokens
        vocab["<pad>"] = pad_token_id;
        vocab["<unk>"] = unk_token_id;
        vocab["<bos>"] = bos_token_id;
        vocab["<eos>"] = eos_token_id;

        inverse_vocab[pad_token_id] = "<pad>";
        inverse_vocab[unk_token_id] = "<unk>";
        inverse_vocab[bos_token_id] = "<bos>";
        inverse_vocab[eos_token_id] = "<eos>";

        special_tokens.insert("<pad>");
        special_tokens.insert("<unk>");
        special_tokens.insert("<bos>");
        special_tokens.insert("<eos>");
    }

    // Build vocabulary from text corpus
    void build_vocab(const std::vector<std::string>& texts, int vocab_size = 10000,
                     int frequency_threshold = 1);

    // Build BPE merge rules
    void build_bpe_merges(const std::vector<std::string>& texts, int num_merges);

    // Pre-tokenization step
    std::vector<std::string> pre_tokenize(const std::string& text);

    // Find most frequent adjacent pair
    std::pair<std::string, std::string> get_most_frequent_pair(
        const std::vector<std::vector<std::string>>& word_tokens);

    // Apply merge to word tokens
    void merge_tokens(std::vector<std::vector<std::string>>& word_tokens, const std::string& first,
                      const std::string& second);

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

    // Load vocabulary from file
    void load_vocab(const std::string& filename);

    // Print vocabulary statistics
    void print_vocab_stats() const;

    // Get top-k most frequent tokens (for debugging)
    std::vector<std::pair<std::string, int>> get_top_tokens(int k = 10) const;

   private:
    // UTF-8 validation helper
    bool is_valid_utf8(const std::string& text) const;
    
    // Validate input string
    void validate_input(const std::string& text, const std::string& context) const;
};
