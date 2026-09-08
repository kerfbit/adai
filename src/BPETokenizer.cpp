// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07

#include "BPETokenizer.hpp"
#include <cmath>
#include <cstdint>
#include <iomanip>

// Pre-tokenization regex patterns
// ASCII: every byte is one token unit; only ASCII letter/digit classes used.
// UNICODE: \x80-\xFF included in letter/non-space classes so multi-byte UTF-8
//          sequences cluster together before BPE splits them into code points.
static const char* ASCII_PATTERN =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|\s+(?!\S)|\s+)";
static const char* UNICODE_PATTERN =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?[\x80-\xFFa-zA-Z]+| ?[0-9]+| ?[^ \t\r\na-zA-Z0-9\x80-\xFF]+|\s+(?!\S)|\s+)";

// ============================================================================
// Constructor
// ============================================================================

BPETokenizer::BPETokenizer(TokenizerMode mode_)
    : mode(mode_),
      token_pattern(mode_ == TokenizerMode::UNICODE ? UNICODE_PATTERN : ASCII_PATTERN) {
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

// ============================================================================
// Private helpers
// ============================================================================

// Split a UTF-8 string into one std::string per code point.
// Invalid start bytes are treated as single-byte units to avoid dropping data.
std::vector<std::string> BPETokenizer::utf8_split_codepoints(const std::string& str) {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < str.size()) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t bytes = 1;
        if ((c & 0x80) == 0)
            bytes = 1;
        else if ((c & 0xE0) == 0xC0)
            bytes = 2;
        else if ((c & 0xF0) == 0xE0)
            bytes = 3;
        else if ((c & 0xF8) == 0xF0)
            bytes = 4;
        if (i + bytes > str.size())
            bytes = 1;  // guard against truncated sequences
        result.push_back(str.substr(i, bytes));
        i += bytes;
    }
    return result;
}

// UTF-8 validation helper
bool BPETokenizer::is_valid_utf8(const std::string& text) {
    size_t i = 0;
    while (i < text.length()) {
        unsigned char c = text[i];
        int bytes = 0;

        if ((c & 0x80) == 0) {
            bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            bytes = 3;
        } else if ((c & 0xF8) == 0xF0) {
            bytes = 4;
        } else {
            return false;
        }

        if (i + bytes > text.length()) {
            return false;
        }

        for (int j = 1; j < bytes; j++) {
            if ((text[i + j] & 0xC0) != 0x80) {
                return false;
            }
        }

        i += bytes;
    }
    return true;
}

// Validate input string
void BPETokenizer::validate_input(const std::string& text, const std::string& context) {
    if (text.empty()) {
        throw TokenizerInputError(context + ": Input text is empty");
    }

    if (!is_valid_utf8(text)) {
        throw TokenizerEncodingError(context + ": Input text contains invalid UTF-8 sequences");
    }
}

// ============================================================================
// Vocabulary building
// ============================================================================

// frequency_threshold: minimum frequency for a character/codepoint to be added to vocab
void BPETokenizer::build_vocab(const std::vector<std::string>& texts, int vocab_size,
                               int frequency_threshold) {
    std::cout << "\n[BPE Tokenizer] Building vocabulary ("
              << (mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII") << " mode)..." << '\n';
    std::cout << "[1/3] Counting character frequencies..." << std::flush;

    // Count frequency of each atomic unit (byte in ASCII mode, code point in Unicode mode)
    std::unordered_map<std::string, int> char_freq;
    size_t total_chars = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        if (mode == TokenizerMode::UNICODE) {
            for (const auto& cp : utf8_split_codepoints(texts[i])) {
                char_freq[cp]++;
                total_chars++;
            }
        } else {
            for (char c : texts[i]) {
                char_freq[std::string(1, c)]++;
                total_chars++;
            }
        }

        if ((i + 1) % 100 == 0 || i == texts.size() - 1) {
            std::cout << "\r[1/3] Counting character frequencies... " << (i + 1) << "/"
                      << texts.size() << " texts processed" << std::flush;
        }
    }
    std::cout << " (" << total_chars << " total characters)" << '\n';

    std::cout << "[2/3] Building base vocabulary..." << std::flush;
    int current_id = 4;  // Start after special tokens
    int added_chars = 0;
    for (const auto& pair : char_freq) {
        if (pair.second >= frequency_threshold && current_id < vocab_size) {
            vocab[pair.first] = current_id;
            inverse_vocab[current_id] = pair.first;
            current_id++;
            added_chars++;
        }
    }
    std::cout << " Added " << added_chars << " characters" << '\n';

    int num_merges = std::max(0, vocab_size - current_id);
    std::cout << "[3/3] Learning BPE merges (target: " << num_merges << " merges)..." << '\n';
    build_bpe_merges(texts, num_merges);
    std::cout << "\n[BPE Tokenizer] Vocabulary built successfully! Final size: " << vocab.size()
              << " tokens" << '\n';
}

// ============================================================================
// BPE merge learning
// ============================================================================

void BPETokenizer::build_bpe_merges(const std::vector<std::string>& texts, int num_merges) {
    std::cout << "    Tokenizing text corpus..." << std::flush;
    std::vector<std::vector<std::string>> word_tokens;
    for (const auto& text : texts) {
        auto tokens = pre_tokenize(text);
        for (const auto& token : tokens) {
            std::vector<std::string> units;
            if (mode == TokenizerMode::UNICODE) {
                units = utf8_split_codepoints(token);
            } else {
                for (char c : token) {
                    units.emplace_back(1, c);
                }
            }
            if (!units.empty()) {
                word_tokens.push_back(units);
            }
        }
    }
    std::cout << " " << word_tokens.size() << " word tokens" << '\n';

    for (int i = 0; i < num_merges; i++) {
        auto best_pair = get_most_frequent_pair(word_tokens);
        if (best_pair.first.empty()) {
            std::cout << "\r    Merge " << i << "/" << num_merges << " - No more pairs to merge"
                      << std::string(50, ' ') << '\n';
            break;
        }

        bpe_merges.push_back(best_pair);

        std::string merged = best_pair.first + best_pair.second;
        int new_id = static_cast<int>(vocab.size());
        vocab[merged] = new_id;
        inverse_vocab[new_id] = merged;

        merge_tokens(word_tokens, best_pair.first, best_pair.second);

        if ((i + 1) % 10 == 0 || i == num_merges - 1) {
            float progress = (float)(i + 1) / static_cast<float>(num_merges) * 100.0f;
            std::cout << "\r    Merge " << (i + 1) << "/" << num_merges << " (" << std::fixed
                      << std::setprecision(1) << progress << "%)" << " - Latest: '"
                      << best_pair.first << "' + '" << best_pair.second << "' → '" << merged << "'"
                      << std::string(20, ' ') << std::flush;
        } else if ((i + 1) % 100 == 0) {
            std::cout << "\r    Processed " << (i + 1) << "/" << num_merges << " merges..."
                      << std::string(50, ' ') << std::flush;
        }
    }
    std::cout << '\n';
}

// ============================================================================
// Pre-tokenization
// ============================================================================

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) {
    if (!text.empty() && !is_valid_utf8(text)) {
        throw TokenizerEncodingError("pre_tokenize(): Input text contains invalid UTF-8 sequences");
    }

    std::string lower_text = text;
    if (mode == TokenizerMode::UNICODE) {
        // Only lowercase ASCII bytes; touching continuation bytes corrupts multi-byte sequences
        for (auto& c : lower_text) {
            if ((unsigned char)c < 0x80) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
    } else {
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                       [](unsigned char c) { return std::tolower(c); });
    }

    std::vector<std::string> tokens;
    std::sregex_iterator iter(lower_text.begin(), lower_text.end(), token_pattern);
    std::sregex_iterator end;

    size_t match_count = 0;
    size_t text_length = lower_text.length();
    size_t last_pos = 0;

    // Hoisted out of the loop: constructing a std::regex is expensive (builds
    // an NFA + internal category tables), and this pattern is fixed — doing
    // it once per token match instead of once per pre_tokenize() call made
    // large inputs pathologically slow (worse under ASAN, where every one of
    // std::regex's internal allocations also gets instrumented).
    static const std::regex kWhitespaceRun("\\s+");
    for (; iter != end; ++iter) {
        std::string token = iter->str();
        token = std::regex_replace(token, kWhitespaceRun, " ");
        if (!token.empty()) {
            tokens.push_back(token);
        }

        match_count++;
        last_pos = iter->position() + iter->length();

        if (match_count % 1000 == 0) {
            float progress = text_length > 0 ? static_cast<float>(last_pos) /
                                                   static_cast<float>(text_length) * 100.0f
                                             : 0.0f;
            std::cout << "\r    Pre-tokenizing... " << match_count << " tokens (" << std::fixed
                      << std::setprecision(1) << progress << "% of text)" << std::flush;
        }
    }

    if (match_count >= 1000) {
        std::cout << "\r    Pre-tokenizing... " << match_count << " tokens - Complete"
                  << std::string(30, ' ') << '\n';
    }

    return tokens;
}

// ============================================================================
// BPE application
// ============================================================================

std::pair<std::string, std::string> BPETokenizer::get_most_frequent_pair(
    const std::vector<std::vector<std::string>>& word_tokens) {
    std::unordered_map<std::string, int> pair_counts;

    for (const auto& tokens : word_tokens) {
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            std::string pair_key = tokens[i] + "|||" + tokens[i + 1];
            pair_counts[pair_key]++;
        }
    }

    if (pair_counts.empty()) {
        return {"", ""};
    }

    auto max_pair =
        std::max_element(pair_counts.begin(), pair_counts.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });

    size_t sep_pos = max_pair->first.find("|||");
    return {max_pair->first.substr(0, sep_pos), max_pair->first.substr(sep_pos + 3)};
}

void BPETokenizer::merge_tokens(std::vector<std::vector<std::string>>& word_tokens,
                                const std::string& first, const std::string& second) {
    for (auto& tokens : word_tokens) {
        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i < tokens.size() - 1 && tokens[i] == first && tokens[i + 1] == second) {
                new_tokens.push_back(first + second);
                i++;
            } else {
                new_tokens.push_back(tokens[i]);
            }
        }
        tokens = new_tokens;
    }
}

std::vector<std::string> BPETokenizer::apply_bpe(const std::string& word) {
    if (word.empty()) {
        return {};
    }

    // Per-thread memoization, keyed by (tokenizer instance, word). Natural
    // text is extremely word-repetitive (Zipf's law — "the"/"a"/"and"/etc.
    // dominate real-world token counts), and without this every occurrence
    // reran the full O(bpe_merges.size()) merge scan below from scratch —
    // this dominated preprocessing time on real text (worse still under
    // ASAN, which instruments every one of the loop's vector allocations).
    // thread_local avoids any synchronization: preprocess_data() already
    // calls tokenize()/encode()/apply_bpe() concurrently from many OpenMP
    // worker threads on one shared BPETokenizer instance, so a plain shared
    // cache would race; a thread_local one needs none, at the cost of not
    // sharing hits across threads. Safe only because bpe_merges is immutable
    // for the lifetime of any concurrent tokenize() calls on a given
    // instance — its only mutators (load_vocab(), train_bpe()) always run
    // single-threaded, before parallel preprocessing ever starts.
    thread_local std::unordered_map<const BPETokenizer*,
                                    std::unordered_map<std::string, std::vector<std::string>>>
        tls_cache;
    auto& cache = tls_cache[this];
    if (auto it = cache.find(word); it != cache.end()) {
        return it->second;
    }

    std::vector<std::string> tokens;
    if (mode == TokenizerMode::UNICODE) {
        tokens = utf8_split_codepoints(word);
    } else {
        for (char c : word) {
            tokens.emplace_back(1, c);
        }
    }

    for (const auto& merge : bpe_merges) {
        std::vector<std::string> new_tokens;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i < tokens.size() - 1 && tokens[i] == merge.first &&
                tokens[i + 1] == merge.second) {
                new_tokens.push_back(merge.first + merge.second);
                i++;
            } else {
                new_tokens.push_back(tokens[i]);
            }
        }
        tokens = new_tokens;
    }

    cache.emplace(word, tokens);
    return tokens;
}

// ============================================================================
// Encode / Decode
// ============================================================================

std::vector<std::string> BPETokenizer::tokenize(const std::string& text) {
    std::vector<std::string> result;
    auto pre_tokens = pre_tokenize(text);

    for (const auto& token : pre_tokens) {
        auto bpe_tokens = apply_bpe(token);
        for (const auto& bpe_token : bpe_tokens) {
            result.push_back(bpe_token);
        }
    }

    return result;
}

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_special_tokens) {
    validate_input(text, "encode()");

    std::vector<int> ids;

    if (add_special_tokens) {
        ids.push_back(bos_token_id);
    }

    auto tokens = tokenize(text);
    for (const auto& token : tokens) {
        if (vocab.find(token) != vocab.end()) {
            ids.push_back(vocab[token]);
        } else {
            ids.push_back(unk_token_id);
        }
    }

    if (add_special_tokens) {
        ids.push_back(eos_token_id);
    }

    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) {
    if (ids.empty()) {
        throw TokenizerInputError("decode(): Input token ID vector is empty");
    }

    std::string result;

    for (int id : ids) {
        if (id < 0) {
            throw TokenIDError("decode(): Token ID " + std::to_string(id) + " is negative");
        }

        if (inverse_vocab.find(id) != inverse_vocab.end()) {
            std::string token = inverse_vocab[id];
            if (skip_special_tokens && special_tokens.count(token)) {
                continue;
            }
            result += token;
        } else {
            std::cerr << "Warning: Unknown token ID " << id << " encountered during decoding"
                      << '\n';
        }
    }

    return result;
}

// ============================================================================
// Vocabulary I/O
// ============================================================================

size_t BPETokenizer::get_vocab_size() const {
    return vocab.size();
}

void BPETokenizer::save_vocab(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << '\n';
        return;
    }

    file << "# BPE Tokenizer Vocabulary v1.0\n";
    file << "TOKENIZER_MODE " << (mode == TokenizerMode::UNICODE ? "UNICODE" : "ASCII") << "\n";
    file << "VOCAB_SIZE " << vocab.size() << "\n";

    file << "SPECIAL_TOKENS\n";
    file << "pad_token_id " << pad_token_id << "\n";
    file << "unk_token_id " << unk_token_id << "\n";
    file << "bos_token_id " << bos_token_id << "\n";
    file << "eos_token_id " << eos_token_id << "\n";

    auto escape = [](const std::string& s) {
        std::string result;
        for (char c : s) {
            if (c == '\n')
                result += "\\n";
            else if (c == '\t')
                result += "\\t";
            else if (c == '\r')
                result += "\\r";
            else if (c == '\\')
                result += "\\\\";
            else if (c == ' ')
                result += "\\s";
            else
                result += c;
        }
        return result;
    };

    file << "VOCAB\n";
    for (const auto& pair : vocab) {
        file << escape(pair.first) << "\t" << pair.second << "\n";
    }

    file << "BPE_MERGES " << bpe_merges.size() << "\n";
    for (const auto& merge : bpe_merges) {
        file << escape(merge.first) << "\t" << escape(merge.second) << "\n";
    }

    file.close();
    std::cout << "[BPE Tokenizer] Vocabulary saved to " << filename << '\n';
    std::cout << "  - Mode: " << (mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII") << '\n';
    std::cout << "  - Vocabulary size: " << vocab.size() << '\n';
    std::cout << "  - BPE merges: " << bpe_merges.size() << '\n';
}

void BPETokenizer::load_vocab(const std::string& filename) {
    if (filename.empty()) {
        throw VocabularyFileError("Vocabulary filename is empty");
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw VocabularyFileError("Could not open vocabulary file: " + filename);
    }

    std::string line;
    vocab.clear();
    inverse_vocab.clear();
    bpe_merges.clear();
    special_tokens.clear();

    auto unescape = [](const std::string& s) {
        std::string result;
        for (size_t i = 0; i < s.length(); i++) {
            if (s[i] == '\\' && i + 1 < s.length()) {
                char next = s[i + 1];
                if (next == 'n') {
                    result += '\n';
                    i++;
                } else if (next == 't') {
                    result += '\t';
                    i++;
                } else if (next == 'r') {
                    result += '\r';
                    i++;
                } else if (next == '\\') {
                    result += '\\';
                    i++;
                } else if (next == 's') {
                    result += ' ';
                    i++;
                } else {
                    result += s[i];
                }
            } else {
                result += s[i];
            }
        }
        return result;
    };

    std::string section;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line.find("TOKENIZER_MODE") == 0) {
            std::string mode_str = line.substr(15);  // skip "TOKENIZER_MODE "
            mode = (mode_str == "UNICODE") ? TokenizerMode::UNICODE : TokenizerMode::ASCII;
            token_pattern =
                std::regex(mode == TokenizerMode::UNICODE ? UNICODE_PATTERN : ASCII_PATTERN);
            continue;
        }
        if (line.find("VOCAB_SIZE") == 0) {
            continue;
        }
        if (line == "SPECIAL_TOKENS") {
            section = "SPECIAL_TOKENS";
            continue;
        }
        if (line == "VOCAB") {
            section = "VOCAB";
            continue;
        }
        if (line.find("BPE_MERGES") == 0) {
            section = "BPE_MERGES";
            continue;
        }

        if (section == "SPECIAL_TOKENS") {
            size_t space_pos = line.find(' ');
            if (space_pos == std::string::npos) {
                throw VocabularyFileError(std::string("Malformed SPECIAL_TOKENS line in ")
                                              .append(filename)
                                              .append(": ")
                                              .append(line));
            }

            std::string key = line.substr(0, space_pos);
            std::string value_str = line.substr(space_pos + 1);

            try {
                int value = std::stoi(value_str);

                if (key == "pad_token_id")
                    pad_token_id = value;
                else if (key == "unk_token_id")
                    unk_token_id = value;
                else if (key == "bos_token_id")
                    bos_token_id = value;
                else if (key == "eos_token_id")
                    eos_token_id = value;
                else {
                    std::cerr << "Warning: Unknown special token key '" << key << "' in "
                              << filename << '\n';
                }
            } catch (const std::invalid_argument&) {
                throw VocabularyFileError(std::string("Invalid integer value for special token in ")
                                              .append(filename)
                                              .append(": ")
                                              .append(value_str));
            } catch (const std::out_of_range&) {
                throw VocabularyFileError(std::string("Special token ID out of range in ")
                                              .append(filename)
                                              .append(": ")
                                              .append(value_str));
            }
        } else if (section == "VOCAB") {
            size_t tab_pos = line.find('\t');
            if (tab_pos == std::string::npos) {
                throw VocabularyFileError(
                    std::string("Malformed VOCAB line (missing tab separator) in ")
                        .append(filename)
                        .append(": ")
                        .append(line));
            }

            std::string token = unescape(line.substr(0, tab_pos));
            std::string id_str = line.substr(tab_pos + 1);

            try {
                int id = std::stoi(id_str);
                if (id < 0) {
                    throw VocabularyFileError(std::string("Negative token ID in ")
                                                  .append(filename)
                                                  .append(": ")
                                                  .append(std::to_string(id)));
                }

                vocab[token] = id;
                inverse_vocab[id] = token;

                if (id == pad_token_id || id == unk_token_id || id == bos_token_id ||
                    id == eos_token_id) {
                    special_tokens.insert(token);
                }
            } catch (const std::invalid_argument&) {
                throw VocabularyFileError(std::string("Invalid token ID in ")
                                              .append(filename)
                                              .append(": ")
                                              .append(id_str));
            } catch (const std::out_of_range&) {
                throw VocabularyFileError(std::string("Token ID out of range in ")
                                              .append(filename)
                                              .append(": ")
                                              .append(id_str));
            }
        } else if (section == "BPE_MERGES") {
            size_t tab_pos = line.find('\t');
            if (tab_pos == std::string::npos) {
                throw VocabularyFileError(
                    std::string("Malformed BPE_MERGES line (missing tab separator) in ")
                        .append(filename)
                        .append(": ")
                        .append(line));
            }

            std::string first = unescape(line.substr(0, tab_pos));
            std::string second = unescape(line.substr(tab_pos + 1));

            if (first.empty() || second.empty()) {
                throw VocabularyFileError("Empty merge token in " + filename);
            }

            bpe_merges.emplace_back(first, second);
        }
    }

    file.close();

    if (vocab.empty()) {
        throw VocabularyFileError("Loaded vocabulary is empty from file: " + filename);
    }

    if (vocab.find("<pad>") == vocab.end() || vocab.find("<unk>") == vocab.end() ||
        vocab.find("<bos>") == vocab.end() || vocab.find("<eos>") == vocab.end()) {
        throw VocabularyFileError("Missing required special tokens in " + filename);
    }

    // Ensure special_tokens set is fully populated
    special_tokens.insert("<pad>");
    special_tokens.insert("<unk>");
    special_tokens.insert("<bos>");
    special_tokens.insert("<eos>");

    inverse_vocab[pad_token_id] = "<pad>";
    inverse_vocab[unk_token_id] = "<unk>";
    inverse_vocab[bos_token_id] = "<bos>";
    inverse_vocab[eos_token_id] = "<eos>";

    std::cout << "[BPE Tokenizer] Vocabulary loaded from " << filename << '\n';
    std::cout << "  - Mode: " << (mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII") << '\n';
    std::cout << "  - Vocabulary size: " << vocab.size() << '\n';
    std::cout << "  - BPE merges: " << bpe_merges.size() << '\n';
    std::cout << "  - Special tokens: " << special_tokens.size() << '\n';
}

// ============================================================================
// Diagnostics
// ============================================================================

void BPETokenizer::print_vocab_stats() const {
    std::cout << "Mode: " << (mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII") << '\n';
    std::cout << "Vocabulary size: " << vocab.size() << '\n';
    std::cout << "Number of BPE merges: " << bpe_merges.size() << '\n';
    std::cout << "Special tokens: " << special_tokens.size() << '\n';
}

std::vector<std::pair<std::string, int>> BPETokenizer::get_top_tokens(int k) const {
    std::vector<std::pair<std::string, int>> sorted_vocab(vocab.begin(), vocab.end());
    std::sort(sorted_vocab.begin(), sorted_vocab.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    int actual_k = std::min(k, (int)sorted_vocab.size());
    return std::vector<std::pair<std::string, int>>(sorted_vocab.begin(),
                                                    sorted_vocab.begin() + actual_k);
}

// ============================================================================
// recommend_vocab_size() — data-driven vocabulary size recommendation.
// ============================================================================

// Decode one UTF-8 codepoint starting at p (len bytes remaining).
// Returns the codepoint value and sets out_bytes to the sequence length.
static uint32_t decode_codepoint(const unsigned char* p, size_t len, size_t& out_bytes) {
    const unsigned char c = *p;
    if ((c & 0x80) == 0) {
        out_bytes = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && len >= 2) {
        out_bytes = 2;
        return ((c & 0x1F) << 6) | (p[1] & 0x3F);
    }
    if ((c & 0xF0) == 0xE0 && len >= 3) {
        out_bytes = 3;
        return ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0 && len >= 4) {
        out_bytes = 4;
        return ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    }
    out_bytes = 1;
    return c;
}

/*static*/
int BPETokenizer::recommend_vocab_size(const std::vector<std::string>& texts, int d_model,
                                       int num_encoder_layers, int num_decoder_layers,
                                       int max_seq_length, TokenizerMode mode) {
    // ── 1. Corpus analysis ───────────────────────────────────────────────────
    size_t total_chars = 0;     // total UTF-8 code points
    size_t total_words = 0;     // whitespace-delimited word tokens
    size_t cjk_count = 0;       // CJK / Hangul / Kana code points
    size_t arabic_count = 0;    // Arabic / Hebrew code points
    size_t cyrillic_count = 0;  // Cyrillic code points
    size_t mb_other_count = 0;  // other multibyte (Extended Latin, Devanagari, …)

    for (const auto& text : texts) {
        bool in_word = false;
        const auto* p = reinterpret_cast<const unsigned char*>(text.data());
        const auto* end = p + text.size();
        while (p < end) {
            size_t bytes = 1;
            uint32_t cp = decode_codepoint(p, static_cast<size_t>(end - p), bytes);
            p += bytes;
            ++total_chars;

            const bool is_space = (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r');
            if (!is_space && !in_word) {
                ++total_words;
                in_word = true;
            } else if (is_space) {
                in_word = false;
            }

            // Script categorisation (Unicode block ranges)
            if (cp >= 0x4E00 && cp <= 0x9FFF)
                ++cjk_count;  // CJK Unified Ideographs
            else if (cp >= 0x3040 && cp <= 0x30FF)
                ++cjk_count;  // Hiragana / Katakana
            else if (cp >= 0xAC00 && cp <= 0xD7AF)
                ++cjk_count;  // Hangul Syllables
            else if (cp >= 0x3400 && cp <= 0x4DBF)
                ++cjk_count;  // CJK Extension A
            else if (cp >= 0x20000 && cp <= 0x2A6DF)
                ++cjk_count;  // CJK Extension B
            else if (cp >= 0x0600 && cp <= 0x06FF)
                ++arabic_count;  // Arabic
            else if (cp >= 0x0590 && cp <= 0x05FF)
                ++arabic_count;  // Hebrew
            else if (cp >= 0xFB1D && cp <= 0xFDFF)
                ++arabic_count;  // Arabic / Hebrew presentation forms
            else if (cp >= 0x0400 && cp <= 0x04FF)
                ++cyrillic_count;
            else if (cp >= 0x0500 && cp <= 0x052F)
                ++cyrillic_count;  // Cyrillic Supplement
            else if (cp >= 0x0080 && cp <= 0x024F)
                ++mb_other_count;  // Extended Latin
            else if (cp >= 0x0900 && cp <= 0x097F)
                ++mb_other_count;  // Devanagari
            else if (cp >= 0x0E00 && cp <= 0x0E7F)
                ++mb_other_count;  // Thai
        }
    }

    if (total_chars == 0)
        return 2000;

    // ── 2. Dataset-size base ─────────────────────────────────────────────────
    // Anchored to known models: ~1k words → ~3k vocab, ~100k → ~14k, ~1M → ~44k.
    const double words = static_cast<double>(std::max(total_words, size_t(100)));
    const double dataset_base = std::min(2000.0 + 3500.0 * std::sqrt(words / 1000.0), 64000.0);

    // ── 3. Script multiplier ─────────────────────────────────────────────────
    const double cjk_frac = static_cast<double>(cjk_count) / total_chars;
    const double arabic_frac = static_cast<double>(arabic_count) / total_chars;
    const double cyrillic_frac = static_cast<double>(cyrillic_count) / total_chars;
    const double mb_frac =
        static_cast<double>(cjk_count + arabic_count + cyrillic_count + mb_other_count) /
        total_chars;

    double script_mult = 1.0;
    if (cjk_frac > 0.3) {
        // CJK characters are already atomic semantic units; BPE needs fewer merges.
        script_mult = 0.55;
    } else if (arabic_frac > 0.2) {
        // Arabic (and Hebrew) morphology is extremely rich — many prefix/suffix combos.
        script_mult = 1.5;
    } else if (cyrillic_frac > 0.3) {
        // Cyrillic languages are moderately inflected.
        script_mult = 1.2;
    } else if (mb_frac > 0.3) {
        // Heavily mixed scripts: must cover multiple character inventories.
        script_mult = 1.25;
    } else if (mb_frac > 0.1) {
        script_mult = 1.1;
    }

    // ── 4. Architecture embedding-budget cap ─────────────────────────────────
    // Embedding table (input only) = V × d_model.  Keep ≤ 30% of total params.
    // Approximate non-embedding params ≈ (12 × L_enc + 16 × L_dec) × d_model².
    // Solving for V:  V × d ≤ 0.3 × (V × d + arch)
    //                 0.7 × V × d ≤ 0.3 × arch
    //                 V ≤ (3/7) × (12 × L_enc + 16 × L_dec) × d_model
    const double arch_mult = 12.0 * num_encoder_layers + 16.0 * num_decoder_layers;
    const int arch_cap = static_cast<int>((3.0 / 7.0) * arch_mult * d_model);

    // ── 5. Sequence-length adjustment ────────────────────────────────────────
    // Short context → want denser tokens (fewer tokens per word) → larger vocab.
    // Long  context → compression less critical → smaller vocab is fine.
    double seq_mult = 1.0;
    if (max_seq_length <= 128)
        seq_mult = 1.40;
    else if (max_seq_length <= 256)
        seq_mult = 1.20;
    else if (max_seq_length >= 2048)
        seq_mult = 0.85;
    else if (max_seq_length >= 1024)
        seq_mult = 0.92;

    // ── 6. Unicode-mode coverage bump ────────────────────────────────────────
    // Unicode mode needs slightly larger vocab to adequately cover multibyte
    // code-point merges beyond what byte-level naturally handles.
    const double mode_mult = (mode == TokenizerMode::UNICODE && mb_frac > 0.05) ? 1.1 : 1.0;

    // ── 7. Combine, cap, and round ───────────────────────────────────────────
    double recommended = dataset_base * script_mult * seq_mult * mode_mult;
    recommended = std::min(recommended, static_cast<double>(arch_cap));
    recommended = std::max(recommended, 2000.0);

    // Round to nearest 500 for clean, memorable sizes.
    const int rounded = static_cast<int>(std::round(recommended / 500.0) * 500.0);
    return std::max(rounded, 2000);
}

// ============================================================================
// measure_fertility() — tokens-per-word ratio on a text sample.
// ============================================================================
float BPETokenizer::measure_fertility(const std::vector<std::string>& texts,
                                      int sample_limit) const {
    if (vocab.empty())
        return 0.0f;

    // encode() is non-const (validates and mutates nothing, but the signature isn't marked const).
    // Use a shallow copy that shares our vocab/merge tables via the same header-constructed state.
    BPETokenizer probe(mode);
    probe.vocab = vocab;
    probe.inverse_vocab = inverse_vocab;
    probe.bpe_merges = bpe_merges;
    probe.special_tokens = special_tokens;
    probe.pad_token_id = pad_token_id;
    probe.unk_token_id = unk_token_id;
    probe.bos_token_id = bos_token_id;
    probe.eos_token_id = eos_token_id;

    size_t total_tokens = 0;
    size_t total_words = 0;
    int sampled = 0;

    for (const auto& text : texts) {
        if (sampled >= sample_limit)
            break;
        if (text.empty())
            continue;

        // Count whitespace-delimited words
        bool in_word = false;
        for (unsigned char c : text) {
            const bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
            if (!is_space && !in_word) {
                ++total_words;
                in_word = true;
            } else if (is_space) {
                in_word = false;
            }
        }

        try {
            total_tokens += probe.encode(text, /*add_special_tokens=*/false).size();
        } catch (...) {
            // skip texts that fail validation
        }
        ++sampled;
    }

    return (total_words == 0) ? 0.0f
                              : static_cast<float>(total_tokens) / static_cast<float>(total_words);
}
