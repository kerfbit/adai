// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-11-10

// TD-225: progress, warnings and save/load summaries go through adai::Logger; this file writes
// nothing to std::cout or std::cerr.

#include "BPETokenizer.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iterator>
#include <sstream>
#include <utility>
#include "Logger.hpp"

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

std::uint64_t BPETokenizer::new_cache_key() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

BPETokenizer::BPETokenizer(TokenizerMode mode_)
    : mode(mode_),
      token_pattern(mode_ == TokenizerMode::UNICODE ? UNICODE_PATTERN : ASCII_PATTERN),
      cache_key_(new_cache_key()) {
    for (const auto& [token, id] :
         {std::pair<const char*, int>{adai::SpecialTokenStrings::PAD, pad_token_id},
          {adai::SpecialTokenStrings::UNK, unk_token_id},
          {adai::SpecialTokenStrings::BOS, bos_token_id},
          {adai::SpecialTokenStrings::EOS, eos_token_id}}) {
        vocab[token] = id;
        inverse_vocab[id] = token;
        special_tokens.insert(token);
    }
}

void BPETokenizer::set_normalization(TextNormalization normalization) {
    normalization_ = normalization;
    cache_key_ = new_cache_key();  // TD-222: cached splits were made under the old setting
}

const char* BPETokenizer::normalization_name(TextNormalization normalization) {
    return normalization == TextNormalization::CASED ? "cased" : "lowercase";
}

void BPETokenizer::set_whitespace(WhitespaceHandling whitespace) {
    whitespace_ = whitespace;
    cache_key_ = new_cache_key();
}

const char* BPETokenizer::whitespace_name(WhitespaceHandling whitespace) {
    return whitespace == WhitespaceHandling::PRESERVE ? "preserve" : "collapse";
}

std::vector<int> BPETokenizer::encode_conversation(
    const std::vector<ConversationTurn>& turns) const {
    if (!has_turn_tokens_) {
        throw std::logic_error("encode_conversation(): this vocabulary has no turn tokens");
    }
    std::vector<int> ids;
    for (const auto& turn : turns) {
        ids.push_back(turn.role == "system"      ? adai::SpecialTokenIDs::SYSTEM_TURN
                      : turn.role == "assistant" ? adai::SpecialTokenIDs::ASSISTANT_TURN
                                                 : adai::SpecialTokenIDs::USER_TURN);
        if (!turn.content.empty()) {  // encode() refuses empty text
            const std::vector<int> content = encode(turn.content, false);
            ids.insert(ids.end(), content.begin(), content.end());
        }
    }
    ids.push_back(adai::SpecialTokenIDs::ASSISTANT_TURN);
    return ids;
}

std::vector<int> BPETokenizer::encode_prompt(const std::string& text) const {
    if (!has_turn_tokens_) {
        return encode(text, false);
    }
    return encode_conversation({{"user", text}});
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
// TD-225: RFC 3629. Each lead byte's first continuation byte has its own range, which rules out
// overlong forms (C0, C1, E0 80-9F, F0 80-8F), UTF-16 surrogates (ED A0-BF) and code points past
// U+10FFFF (F4 90-BF, F5-FF). The check used to accept all of them, and lead bytes up to F7.
bool BPETokenizer::is_valid_utf8(const std::string& text) {
    const auto* p = reinterpret_cast<const unsigned char*>(text.data());
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        if (c < 0x80) {
            ++i;
            continue;
        }
        size_t bytes = 0;
        unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the first continuation byte
        if (c >= 0xC2 && c <= 0xDF) {
            bytes = 2;
        } else if (c >= 0xE0 && c <= 0xEF) {
            bytes = 3;
            if (c == 0xE0)
                lo = 0xA0;
            if (c == 0xED)
                hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            bytes = 4;
            if (c == 0xF0)
                lo = 0x90;
            if (c == 0xF4)
                hi = 0x8F;
        } else {
            return false;  // a continuation byte, C0/C1, or F5-FF
        }
        if (i + bytes > n || p[i + 1] < lo || p[i + 1] > hi) {
            return false;
        }
        for (size_t j = 2; j < bytes; ++j) {
            if ((p[i + j] & 0xC0) != 0x80) {
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
    // TD-223: a second build used to restart ids at 4 and append to the first build's merges,
    // giving one id to two tokens. Start from a new tokenizer with this one's settings instead.
    if (vocab.size() > 4 || !bpe_merges.empty()) {
        BPETokenizer fresh(mode);
        fresh.normalization_ = normalization_;
        fresh.whitespace_ = whitespace_;
        *this = std::move(fresh);
    }
    adai::Logger::info("[BPE Tokenizer] Building vocabulary ({} mode) from {} texts",
                       mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII", texts.size());

    // Count frequency of each atomic unit (byte in ASCII mode, code point in Unicode mode)
    std::unordered_map<std::string, int> char_freq;
    size_t total_chars = 0;
    // TD-560: a LOWERCASE vocabulary counts the text as pre_tokenize() will see it; it used to give
    // capital letters base ids that encoding never produces.
    auto as_seen = [this](const std::string& s) {
        if (normalization_ == TextNormalization::CASED) {
            return s;
        }
        std::string lowered = s;
        for (auto& c : lowered) {
            if (static_cast<unsigned char>(c) < 0x80) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
        return lowered;
    };
    for (size_t i = 0; i < texts.size(); i++) {
        const std::string text = as_seen(texts[i]);
        if (mode == TokenizerMode::UNICODE) {
            for (const auto& cp : utf8_split_codepoints(text)) {
                char_freq[cp]++;
                total_chars++;
            }
        } else {
            for (char c : text) {
                char_freq[std::string(1, c)]++;
                total_chars++;
            }
        }
    }
    adai::Logger::info("[BPE Tokenizer] [1/3] {} characters, {} distinct", total_chars,
                       char_freq.size());

    int current_id = 4;  // Start after special tokens
    int added_chars = 0;
    // TD-295: the three turn tokens take ids 4-6 in every new vocabulary. TD-565: ids 7-19 are held
    // for future special tokens, each with a placeholder entry, so ordinary tokens start at 20. A
    // vocabulary with no room past them keeps the old layout.
    if (total_chars > 0 && vocab_size > adai::SpecialTokenIDs::RESERVED_SLOTS) {
        for (const char* turn :
             {adai::SpecialTokenStrings::SYSTEM_TURN, adai::SpecialTokenStrings::USER_TURN,
              adai::SpecialTokenStrings::ASSISTANT_TURN}) {
            vocab[turn] = current_id;
            inverse_vocab[current_id] = turn;
            special_tokens.insert(turn);
            current_id++;
        }
        has_turn_tokens_ = true;
        for (; current_id < adai::SpecialTokenIDs::RESERVED_SLOTS; ++current_id) {
            const std::string slot = adai::reserved_slot_token(current_id);
            vocab[slot] = current_id;
            inverse_vocab[current_id] = slot;
            special_tokens.insert(slot);
        }
        first_ordinary_id_ = current_id;
    }
    // TD-224: whitespace always has a token, or a corpus read line by line (vocab_builder's plain
    // and legacy formats) would turn every newline in later text into <unk>.
    if (whitespace_ == WhitespaceHandling::PRESERVE && total_chars > 0) {
        for (const char* ws : {" ", "\n", "\t"}) {
            char_freq.erase(ws);
            if (current_id < vocab_size) {
                vocab[ws] = current_id;
                inverse_vocab[current_id] = ws;
                current_id++;
                added_chars++;
            }
        }
    }
    // TD-225: base units take ids by frequency, most frequent first, ties by the unit's bytes. They
    // used to follow unordered_map iteration order, so one corpus could give different ids under
    // different standard libraries.
    std::vector<std::pair<std::string, int>> units(char_freq.begin(), char_freq.end());
    std::sort(units.begin(), units.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    for (const auto& pair : units) {
        if (pair.second >= frequency_threshold && current_id < vocab_size) {
            vocab[pair.first] = current_id;
            inverse_vocab[current_id] = pair.first;
            current_id++;
            added_chars++;
        }
    }
    adai::Logger::info("[BPE Tokenizer] [2/3] {} base units added", added_chars);

    int num_merges = std::max(0, vocab_size - current_id);
    adai::Logger::info("[BPE Tokenizer] [3/3] Learning up to {} merges", num_merges);
    build_bpe_merges(texts, num_merges);
    adai::Logger::info("[BPE Tokenizer] Vocabulary built: {} tokens, {} merges", vocab.size(),
                       bpe_merges.size());
}

// ============================================================================
// BPE merge learning
// ============================================================================

void BPETokenizer::build_bpe_merges(const std::vector<std::string>& texts, int num_merges) {
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
    adai::Logger::debug("[BPE Tokenizer] {} words to learn merges from", word_tokens.size());

    // TD-225: progress at every tenth of the target, at debug, instead of a console line per merge.
    const int progress_step = std::max(1, num_merges / 10);
    for (int i = 0; i < num_merges; i++) {
        auto best_pair = get_most_frequent_pair(word_tokens);
        if (best_pair.first.empty()) {
            adai::Logger::info("[BPE Tokenizer] No more pairs to merge after {} of {} merges", i,
                               num_merges);
            break;
        }

        bpe_merges.push_back(best_pair);
        cache_key_ = new_cache_key();  // TD-222: the merges changed

        std::string merged = best_pair.first + best_pair.second;
        int new_id = static_cast<int>(vocab.size());
        vocab[merged] = new_id;
        inverse_vocab[new_id] = merged;

        merge_tokens(word_tokens, best_pair.first, best_pair.second);

        if ((i + 1) % progress_step == 0 || i == num_merges - 1) {
            adai::Logger::debug("[BPE Tokenizer] Merge {}/{}", i + 1, num_merges);
        }
    }
}

// ============================================================================
// Pre-tokenization
// ============================================================================

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    if (!text.empty() && !is_valid_utf8(text)) {
        throw TokenizerEncodingError("pre_tokenize(): Input text contains invalid UTF-8 sequences");
    }

    // TD-560: a CASED vocabulary keeps letter case; only a LOWERCASE one (every v1 file)
    // lowercases.
    std::string lower_text = text;
    if (normalization_ == TextNormalization::CASED) {
        // keep the text as written
    } else if (mode == TokenizerMode::UNICODE) {
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

    // Hoisted out of the loop: constructing a std::regex is expensive (builds
    // an NFA + internal category tables), and this pattern is fixed — doing
    // it once per token match instead of once per pre_tokenize() call made
    // large inputs pathologically slow (worse under ASAN, where every one of
    // std::regex's internal allocations also gets instrumented).
    static const std::regex kWhitespaceRun("\\s+");
    for (; iter != end; ++iter) {
        std::string token = iter->str();
        if (whitespace_ == WhitespaceHandling::COLLAPSE) {  // TD-224: every pre-v3 vocabulary
            token = std::regex_replace(token, kWhitespaceRun, " ");
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    // TD-225: no progress output. It printed for any text of 1,000 or more pieces, including long
    // chat messages on the server, interleaved across preprocessing threads.

    return tokens;
}

// ============================================================================
// BPE application
// ============================================================================

namespace {
// TD-220: pair counts are keyed by the pair itself. They used to be keyed by
// first + "|||" + second and split back at the first "|||", which mis-split any pair
// containing '|' (an empty first token stopped merge learning; a wrong non-empty pair was
// re-recorded as a merge rule every round).
struct StringPairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        const size_t h1 = std::hash<std::string>{}(p.first);
        const size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};
}  // namespace

std::pair<std::string, std::string> BPETokenizer::get_most_frequent_pair(
    const std::vector<std::vector<std::string>>& word_tokens) {
    std::unordered_map<std::pair<std::string, std::string>, int, StringPairHash> pair_counts;

    for (const auto& tokens : word_tokens) {
        // `i + 1 < tokens.size()` rather than `i < tokens.size() - 1` (TD-058,
        // fixed): the latter underflows to SIZE_MAX for an empty `tokens`
        // entry, since tokens.size() is unsigned — every internal caller
        // happens to only ever push non-empty entries, but this is a public
        // static method and callers outside this file have no such guarantee.
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            pair_counts[{tokens[i], tokens[i + 1]}]++;
        }
    }

    if (pair_counts.empty()) {
        return {"", ""};
    }

    // Highest count wins; ties go to the lexicographically smallest pair, so the learned merges
    // don't depend on hash-table iteration order.
    auto best = pair_counts.begin();
    for (auto it = std::next(pair_counts.begin()); it != pair_counts.end(); ++it) {
        if (it->second > best->second || (it->second == best->second && it->first < best->first))
            best = it;
    }
    return best->first;
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

std::vector<std::string> BPETokenizer::apply_bpe(const std::string& word) const {
    if (word.empty()) {
        return {};
    }

    // Per-thread memoization, keyed by (cache_key_, word). Natural text is extremely
    // word-repetitive (Zipf's law), and without this every occurrence reran the full
    // O(bpe_merges.size()) merge scan below from scratch, which dominated preprocessing time.
    // thread_local avoids any synchronization: preprocess_data() calls tokenize()/encode()/
    // apply_bpe() concurrently from many OpenMP worker threads on one shared BPETokenizer, so a
    // shared cache would race. Safe because the merges don't change while encoding runs: their
    // mutators (build_vocab()/build_bpe_merges()/load_vocab()) run single-threaded first.
    //
    // TD-222: keyed by cache_key_, not the tokenizer's address, so a reload or a new tokenizer at a
    // reused address starts empty instead of inheriting another state's results. Each thread keeps
    // the caches of its last kCachedStates states (RAG alternates between two tokenizers) and drops
    // a cache once it holds kMaxCachedWords words, so memory stays bounded.
    constexpr std::size_t kCachedStates = 4;
    constexpr std::size_t kMaxCachedWords = 1 << 16;
    using WordCache = std::unordered_map<std::string, std::vector<std::string>>;
    thread_local std::vector<std::pair<std::uint64_t, WordCache>> tls_caches;  // most recent last
    WordCache* cache = nullptr;
    for (auto it = tls_caches.begin(); it != tls_caches.end(); ++it) {
        if (it->first == cache_key_) {
            if (std::next(it) != tls_caches.end()) std::rotate(it, std::next(it), tls_caches.end());
            cache = &tls_caches.back().second;
            break;
        }
    }
    if (!cache) {
        if (tls_caches.size() >= kCachedStates) tls_caches.erase(tls_caches.begin());
        tls_caches.emplace_back(cache_key_, WordCache{});
        cache = &tls_caches.back().second;
    }
    if (auto it = cache->find(word); it != cache->end()) {
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

    if (cache->size() >= kMaxCachedWords) cache->clear();
    cache->emplace(word, tokens);
    return tokens;
}

// ============================================================================
// Encode / Decode
// ============================================================================

std::vector<std::string> BPETokenizer::tokenize(const std::string& text) const {
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

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_special_tokens) const {
    validate_input(text, "encode()");

    std::vector<int> ids;

    if (add_special_tokens) {
        ids.push_back(bos_token_id);
    }

    auto tokens = tokenize(text);
    for (const auto& token : tokens) {
        // TD-565: text never encodes to a special id, assigned or reserved.
        const auto it = vocab.find(token);
        ids.push_back(it != vocab.end() && it->second >= first_ordinary_id_ ? it->second
                                                                            : unk_token_id);
    }

    if (add_special_tokens) {
        ids.push_back(eos_token_id);
    }

    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    if (ids.empty()) {
        throw TokenizerInputError("decode(): Input token ID vector is empty");
    }

    std::string result;
    // TD-225: ids with no token are counted and reported in one warning per call. They used to get
    // one std::cerr line each, which a log never showed, so out-of-range model output was cut
    // short without a trace.
    size_t unknown = 0;
    int first_unknown = 0;

    for (int id : ids) {
        if (id < 0) {
            throw TokenIDError("decode(): Token ID " + std::to_string(id) + " is negative");
        }

        const auto it = inverse_vocab.find(id);
        if (it != inverse_vocab.end()) {
            const std::string& token = it->second;
            if (skip_special_tokens && special_tokens.count(token)) {
                continue;
            }
            result += token;
        } else {
            if (unknown++ == 0)
                first_unknown = id;
        }
    }
    if (unknown > 0) {
        adai::Logger::warn(
            "[BPE Tokenizer] decode(): left out {} of {} ids with no token (first: {}; vocabulary "
            "size {})",
            unknown, ids.size(), first_unknown, vocab.size());
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
    // TD-221: write to a temporary file next to the target and rename it into place, throwing
    // VocabularyFileError on any failure, so callers (every EncoderDecoderModel::save_model()
    // checkpoint, vocab_builder, the trainer's vocab bootstrap) can't believe a failed save
    // succeeded, and a failed save never leaves a truncated vocab where a good one used to be.
    if (filename.empty()) {
        throw VocabularyFileError("Vocabulary filename is empty");
    }
    const std::string tmp_filename = filename + ".tmp";
    std::ofstream file(tmp_filename, std::ios::trunc);
    if (!file.is_open()) {
        throw VocabularyFileError("Could not open vocabulary file for writing: " + tmp_filename);
    }

    // TD-560: v2 adds NORMALIZATION. A file without it is v1, which always lowercased.
    // TD-224: v3 adds WHITESPACE. A file without it collapsed every whitespace run.
    // TD-565: v4 adds RESERVED_SPECIAL_IDS, the ids below the first ordinary token.
    file << "# BPE Tokenizer Vocabulary v4.0\n";
    file << "TOKENIZER_MODE " << (mode == TokenizerMode::UNICODE ? "UNICODE" : "ASCII") << "\n";
    file << "NORMALIZATION " << (normalization_ == TextNormalization::CASED ? "CASED" : "LOWERCASE")
         << "\n";
    file << "WHITESPACE " << (whitespace_ == WhitespaceHandling::PRESERVE ? "PRESERVE" : "COLLAPSE")
         << "\n";
    file << "VOCAB_SIZE " << vocab.size() << "\n";
    file << "RESERVED_SPECIAL_IDS " << first_ordinary_id_ << "\n";

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

    // TD-225: in id order, so one vocabulary is written as the same file everywhere.
    std::vector<std::pair<std::string, int>> by_id(vocab.begin(), vocab.end());
    std::sort(by_id.begin(), by_id.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    file << "VOCAB\n";
    for (const auto& pair : by_id) {
        file << escape(pair.first) << "\t" << pair.second << "\n";
    }

    file << "BPE_MERGES " << bpe_merges.size() << "\n";
    for (const auto& merge : bpe_merges) {
        file << escape(merge.first) << "\t" << escape(merge.second) << "\n";
    }

    file.flush();
    const bool write_ok = static_cast<bool>(file);
    file.close();
    if (!write_ok || file.fail()) {
        std::remove(tmp_filename.c_str());
        throw VocabularyFileError("Failed writing vocabulary file: " + tmp_filename);
    }
    // std::filesystem::rename replaces an existing target on every platform; std::rename()
    // fails on Windows when the target exists, so every save after the first would throw there.
    std::error_code rename_ec;
    std::filesystem::rename(tmp_filename, filename, rename_ec);
    if (rename_ec) {
        std::remove(tmp_filename.c_str());
        throw VocabularyFileError("Could not move vocabulary file into place: " + filename + " (" +
                                  rename_ec.message() + ")");
    }
    adai::Logger::info(
        "[BPE Tokenizer] Vocabulary saved to {} ({}, {}, whitespace {}, {} tokens, {} "
        "merges)",
        filename, mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII",
        normalization_name(normalization_), whitespace_name(whitespace_), vocab.size(),
        bpe_merges.size());
}

void BPETokenizer::load_vocab(const std::string& filename) {
    // TD-223: parse into a new tokenizer and take it only if the whole file loads, so a failure
    // leaves this one unchanged (it used to clear every table first).
    BPETokenizer loaded;
    loaded.read_vocab_file(filename);
    *this = std::move(loaded);
}

void BPETokenizer::read_vocab_file(const std::string& filename) {
    if (filename.empty()) {
        throw VocabularyFileError("Vocabulary filename is empty");
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw VocabularyFileError("Could not open vocabulary file: " + filename);
    }

    cache_key_ = new_cache_key();  // TD-222: results cached for the old merges no longer apply
    // TD-560: a v1 file has no NORMALIZATION line and was always lowercased.
    normalization_ = TextNormalization::LOWERCASE;
    whitespace_ = WhitespaceHandling::COLLAPSE;  // TD-224: likewise for a file before v3
    // TD-564: a file from before the TOKENIZER_MODE line was always built in ASCII mode; it used to
    // keep the constructor's mode, which is now Unicode.
    mode = TokenizerMode::ASCII;
    token_pattern = std::regex(ASCII_PATTERN);
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
    int reserved_ids = -1;  // TD-565: RESERVED_SPECIAL_IDS, absent before v4
    while (std::getline(file, line)) {
        // TD-563: a vocabulary written by a newer format is refused rather than misread (an older
        // binary used to skip header lines it didn't know, so it would lowercase a cased
        // vocabulary).
        static const std::string kFormatComment = "# BPE Tokenizer Vocabulary v";
        if (line.rfind(kFormatComment, 0) == 0) {
            const int major = std::atoi(line.c_str() + kFormatComment.size());
            if (major > kVocabFormatMajor) {
                throw VocabularyFileError(
                    filename + " is vocabulary format v" + line.substr(kFormatComment.size()) +
                    ", newer than this build reads (v" + std::to_string(kVocabFormatMajor) +
                    "); update the binary");
            }
            continue;
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("NORMALIZATION", 0) == 0) {  // TD-560
            const std::string value = line.size() > 14 ? line.substr(14) : std::string();
            if (value == "CASED") {
                normalization_ = TextNormalization::CASED;
            } else if (value == "LOWERCASE") {
                normalization_ = TextNormalization::LOWERCASE;
            } else {
                throw VocabularyFileError("Unknown NORMALIZATION '" + value + "' in " + filename);
            }
            continue;
        }

        if (line.rfind("WHITESPACE", 0) == 0) {  // TD-224
            const std::string value = line.size() > 11 ? line.substr(11) : std::string();
            if (value == "PRESERVE") {
                whitespace_ = WhitespaceHandling::PRESERVE;
            } else if (value == "COLLAPSE") {
                whitespace_ = WhitespaceHandling::COLLAPSE;
            } else {
                throw VocabularyFileError("Unknown WHITESPACE '" + value + "' in " + filename);
            }
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
        if (line.rfind("RESERVED_SPECIAL_IDS ", 0) == 0) {  // TD-565
            try {
                reserved_ids = std::stoi(line.substr(21));
            } catch (const std::exception&) {
                reserved_ids = -1;
            }
            if (reserved_ids != adai::SpecialTokenIDs::SYSTEM_TURN &&
                reserved_ids != adai::SpecialTokenIDs::FIRST_UNASSIGNED &&
                reserved_ids != adai::SpecialTokenIDs::RESERVED_SLOTS) {
                throw VocabularyFileError("Unsupported RESERVED_SPECIAL_IDS '" + line.substr(21) +
                                          "' in " + filename);
            }
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
        if (section.empty()) {
            // TD-563: an unknown header line belongs to a format this build doesn't understand.
            throw VocabularyFileError("Unknown vocabulary header line in " + filename + ": " +
                                      line);
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
                    adai::Logger::warn("[BPE Tokenizer] Unknown special token key '{}' in {}", key,
                                       filename);
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

    // TD-295: only tokens spelled and numbered exactly so; older vocabularies use these ids for
    // ordinary tokens.
    const auto is_at = [this](const char* token, int id) {
        const auto it = vocab.find(token);
        return it != vocab.end() && it->second == id;
    };
    has_turn_tokens_ =
        is_at(adai::SpecialTokenStrings::SYSTEM_TURN, adai::SpecialTokenIDs::SYSTEM_TURN) &&
        is_at(adai::SpecialTokenStrings::USER_TURN, adai::SpecialTokenIDs::USER_TURN) &&
        is_at(adai::SpecialTokenStrings::ASSISTANT_TURN, adai::SpecialTokenIDs::ASSISTANT_TURN);
    if (has_turn_tokens_) {
        special_tokens.insert(adai::SpecialTokenStrings::SYSTEM_TURN);
        special_tokens.insert(adai::SpecialTokenStrings::USER_TURN);
        special_tokens.insert(adai::SpecialTokenStrings::ASSISTANT_TURN);
    }
    // TD-565: before v4 the layout follows from the turn tokens: ordinary tokens from 7 with them,
    // from 4 without. A v4 file states it, and every id below it must be present and is special.
    if (reserved_ids < 0) {
        first_ordinary_id_ = has_turn_tokens_ ? adai::SpecialTokenIDs::FIRST_UNASSIGNED
                                              : adai::SpecialTokenIDs::SYSTEM_TURN;
    } else {
        if (reserved_ids > adai::SpecialTokenIDs::SYSTEM_TURN && !has_turn_tokens_) {
            throw VocabularyFileError("RESERVED_SPECIAL_IDS " + std::to_string(reserved_ids) +
                                      " without turn tokens at ids 4-6 in " + filename);
        }
        for (int id = adai::SpecialTokenIDs::FIRST_UNASSIGNED; id < reserved_ids; ++id) {
            const auto it = inverse_vocab.find(id);
            if (it == inverse_vocab.end()) {
                throw VocabularyFileError("Reserved special id " + std::to_string(id) +
                                          " has no entry in " + filename);
            }
            special_tokens.insert(it->second);
        }
        first_ordinary_id_ = reserved_ids;
    }

    if (vocab.empty()) {
        throw VocabularyFileError("Loaded vocabulary is empty from file: " + filename);
    }

    // TD-505: the four ids are the standard 0-3, which the batching, padding and generation code
    // uses directly; a file with other ids used to load (BOS = EOS made every target start with the
    // stop token).
    const struct {
        const char* key;
        int id;
        int standard;
        const char* token;
    } kStandard[] = {
        {"pad_token_id", pad_token_id, adai::SpecialTokenIDs::PAD, adai::SpecialTokenStrings::PAD},
        {"unk_token_id", unk_token_id, adai::SpecialTokenIDs::UNK, adai::SpecialTokenStrings::UNK},
        {"bos_token_id", bos_token_id, adai::SpecialTokenIDs::BOS, adai::SpecialTokenStrings::BOS},
        {"eos_token_id", eos_token_id, adai::SpecialTokenIDs::EOS, adai::SpecialTokenStrings::EOS},
    };
    for (const auto& special : kStandard) {
        if (special.id != special.standard) {
            throw VocabularyFileError(std::string(special.key) + " is " +
                                      std::to_string(special.id) + " in " + filename +
                                      "; it must be " + std::to_string(special.standard));
        }
        const auto it = vocab.find(special.token);
        if (it == vocab.end() || it->second != special.standard) {
            throw VocabularyFileError(std::string(special.token) + " must be id " +
                                      std::to_string(special.standard) + " in " + filename);
        }
        special_tokens.insert(special.token);
    }

    adai::Logger::info(
        "[BPE Tokenizer] Vocabulary loaded from {} ({}, {} tokens, {} merges, {} "
        "special tokens)",
        filename, mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII", vocab.size(),
        bpe_merges.size(), special_tokens.size());
}

// ============================================================================
// Diagnostics
// ============================================================================

std::string BPETokenizer::vocab_stats() const {
    std::ostringstream out;
    out << "Mode: " << (mode == TokenizerMode::UNICODE ? "Unicode" : "ASCII") << '\n';
    out << "Normalization: " << normalization_name(normalization_) << '\n';
    out << "Whitespace: " << whitespace_name(whitespace_) << '\n';
    out << "Vocabulary size: " << vocab.size() << '\n';
    out << "Number of BPE merges: " << bpe_merges.size() << '\n';
    out << "Special tokens: " << special_tokens.size() << '\n';
    return out.str();
}

std::vector<std::pair<std::string, int>> BPETokenizer::get_tokens_by_id(int k) const {
    std::vector<std::pair<std::string, int>> sorted_vocab(vocab.begin(), vocab.end());
    std::sort(sorted_vocab.begin(), sorted_vocab.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    int actual_k = std::clamp(k, 0, static_cast<int>(sorted_vocab.size()));
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
                                       int max_seq_length, TokenizerMode mode,
                                       TextNormalization normalization) {
    // ── 1. Corpus analysis ───────────────────────────────────────────────────
    size_t total_chars = 0;     // total UTF-8 code points
    size_t total_words = 0;     // whitespace-delimited word tokens
    size_t capitalized_words = 0;  // TD-562: words with an ASCII capital letter
    size_t cjk_count = 0;       // CJK / Hangul / Kana code points
    size_t arabic_count = 0;    // Arabic / Hebrew code points
    size_t cyrillic_count = 0;  // Cyrillic code points
    size_t mb_other_count = 0;  // other multibyte (Extended Latin, Devanagari, …)

    for (const auto& text : texts) {
        bool in_word = false;
        bool word_has_capital = false;
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
                word_has_capital = false;
            } else if (is_space) {
                in_word = false;
            }
            if (in_word && !word_has_capital && cp >= 'A' && cp <= 'Z') {
                word_has_capital = true;
                ++capitalized_words;
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

    // ── 6b. Letter case (TD-562) ─────────────────────────────────────────────
    // Keeping case gives each capitalized word its own forms ("The"/"the"): on English prose the
    // distinct words grew by about the share of capitalized words (23% for 23%).
    const double cap_frac =
        total_words > 0 ? static_cast<double>(capitalized_words) / static_cast<double>(total_words)
                        : 0.0;
    const double case_mult =
        normalization == TextNormalization::CASED ? 1.0 + std::min(cap_frac, 0.3) : 1.0;

    // ── 7. Combine, cap, and round ───────────────────────────────────────────
    double recommended = dataset_base * script_mult * seq_mult * mode_mult * case_mult;
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

    // TD-225: encode() is const, so this no longer copies the whole tokenizer into a probe.
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
            total_tokens += encode(text, /*add_special_tokens=*/false).size();
        } catch (...) {
            // skip texts that fail validation
        }
        ++sampled;
    }

    return (total_words == 0) ? 0.0f
                              : static_cast<float>(total_tokens) / static_cast<float>(total_words);
}
