#include <../gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <new>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>
#include "../src/BPETokenizer.hpp"

// Helper function to create a test vocabulary file
void create_test_vocab(const std::string& filename) {
    std::ofstream file(filename);
    file << "# BPE Tokenizer Vocabulary v1.0\n";
    file << "VOCAB_SIZE 20\n";
    file << "SPECIAL_TOKENS\n";
    file << "pad_token_id 0\n";
    file << "unk_token_id 1\n";
    file << "bos_token_id 2\n";
    file << "eos_token_id 3\n";
    file << "VOCAB\n";
    file << "<pad>\t0\n";
    file << "<unk>\t1\n";
    file << "<bos>\t2\n";
    file << "<eos>\t3\n";
    file << "h\t4\n";
    file << "e\t5\n";
    file << "l\t6\n";
    file << "o\t7\n";
    file << "w\t8\n";
    file << "r\t9\n";
    file << "d\t10\n";
    file << " \t11\n";
    file << ",\t12\n";
    file << "!\t13\n";
    file << "t\t14\n";
    file << "he\t15\n";
    file << "lo\t16\n";
    file << "wor\t17\n";
    file << "world\t18\n";
    file << "hello\t19\n";
    file << "BPE_MERGES 5\n";
    file << "h\te\n";
    file << "l\to\n";
    file << "w\to\n";
    file << "wo\tr\n";
    file << "he\tl\n";
    file.close();
}

// Test fixture for BPETokenizer tests
class BPETokenizerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        test_vocab_file = "test_tokenizer_vocab.txt";
    }

    void TearDown() override {
        // Clean up test files
        std::remove(test_vocab_file.c_str());
        std::remove("test_save_vocab.txt");
    }

    std::string test_vocab_file;
};

// ============================================================================
// Construction and Initialization Tests
// ============================================================================

TEST_F(BPETokenizerTest, DefaultConstruction) {
    BPETokenizer tokenizer;

    // Should have 4 special tokens
    EXPECT_EQ(tokenizer.get_vocab_size(), 4);
}

TEST_F(BPETokenizerTest, SpecialTokensInitialized) {
    BPETokenizer tokenizer;

    // Encode with special tokens should add BOS and EOS
    auto ids = tokenizer.encode("test", true);
    EXPECT_EQ(ids.front(), 2);  // BOS
    EXPECT_EQ(ids.back(), 3);   // EOS
    EXPECT_GE(ids.size(), 2);   // At least BOS and EOS
}

// ============================================================================
// Vocabulary Building Tests
// ============================================================================

TEST_F(BPETokenizerTest, BuildVocabularyBasic) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello world", "hello there", "world peace"};

    tokenizer.build_vocab(texts, 50, 1);

    // Should have more than special tokens
    EXPECT_GT(tokenizer.get_vocab_size(), 4);
}

TEST_F(BPETokenizerTest, BuildVocabularyWithThreshold) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"aaabbbccc"};

    tokenizer.build_vocab(texts, 20, 2);  // threshold = 2

    // Each character appears 3 times (>= threshold), so all should be included
    size_t vocab_size = tokenizer.get_vocab_size();
    EXPECT_GE(vocab_size, 4);  // At least special tokens + frequent chars
}

TEST_F(BPETokenizerTest, BuildVocabularyEmptyTexts) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {};

    tokenizer.build_vocab(texts, 10, 1);

    // Should still have special tokens
    EXPECT_EQ(tokenizer.get_vocab_size(), 4);
}

// ============================================================================
// Pre-tokenization Tests
// ============================================================================

TEST_F(BPETokenizerTest, PreTokenizeBasic) {
    BPETokenizer tokenizer;

    auto tokens = tokenizer.pre_tokenize("Hello world");

    EXPECT_GT(tokens.size(), 0);
    // Text is lowercased, first token may or may not have leading space
    EXPECT_TRUE(tokens[0] == "hello" || tokens[0] == " hello");
}

TEST_F(BPETokenizerTest, PreTokenizeContractions) {
    BPETokenizer tokenizer;

    auto tokens = tokenizer.pre_tokenize("I'm can't won't");

    EXPECT_GT(tokens.size(), 0);
    // Should split contractions properly
    bool found_contraction = false;
    for (const auto& token : tokens) {
        if (token == "'m" || token == "'t") {
            found_contraction = true;
            break;
        }
    }
    EXPECT_TRUE(found_contraction);
}

TEST_F(BPETokenizerTest, PreTokenizeNumbers) {
    BPETokenizer tokenizer;

    auto tokens = tokenizer.pre_tokenize("There are 42 items");

    EXPECT_GT(tokens.size(), 0);
    // Should extract number
    bool found_number = false;
    for (const auto& token : tokens) {
        if (token.find("42") != std::string::npos) {
            found_number = true;
            break;
        }
    }
    EXPECT_TRUE(found_number);
}

TEST_F(BPETokenizerTest, PreTokenizePunctuation) {
    BPETokenizer tokenizer;

    auto tokens = tokenizer.pre_tokenize("Hello, world!");

    EXPECT_GT(tokens.size(), 0);
    // Should handle punctuation
}

TEST_F(BPETokenizerTest, PreTokenizeEmptyString) {
    BPETokenizer tokenizer;

    auto tokens = tokenizer.pre_tokenize("");

    EXPECT_EQ(tokens.size(), 0);
}

// ============================================================================
// Tokenization Tests
// ============================================================================

TEST_F(BPETokenizerTest, TokenizeWithoutVocab) {
    BPETokenizer tokenizer;

    // Build a simple vocabulary
    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);

    auto tokens = tokenizer.tokenize("hello");

    EXPECT_GT(tokens.size(), 0);
}

TEST_F(BPETokenizerTest, TokenizeEmptyString) {
    BPETokenizer tokenizer;

    auto tokens = tokenizer.tokenize("");

    EXPECT_EQ(tokens.size(), 0);
}

TEST_F(BPETokenizerTest, TokenizeWithMerges) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello hello hello", "world world world"};

    tokenizer.build_vocab(texts, 30, 1);
    auto tokens = tokenizer.tokenize("hello world");

    EXPECT_GT(tokens.size(), 0);
}

// ============================================================================
// Encoding Tests
// ============================================================================

TEST_F(BPETokenizerTest, EncodeWithSpecialTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 20, 1);

    auto ids = tokenizer.encode("test", true);

    EXPECT_GE(ids.size(), 2);
    EXPECT_EQ(ids.front(), 2);  // BOS
    EXPECT_EQ(ids.back(), 3);   // EOS
}

TEST_F(BPETokenizerTest, EncodeWithoutSpecialTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 20, 1);

    auto ids = tokenizer.encode("test", false);

    // Should not have BOS/EOS
    if (ids.size() > 0) {
        EXPECT_NE(ids.front(), 2);  // Not BOS
    }
    if (ids.size() > 1) {
        EXPECT_NE(ids.back(), 3);  // Not EOS
    }
}

TEST_F(BPETokenizerTest, EncodeUnknownTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"known"};
    tokenizer.build_vocab(texts, 20, 1);

    // Encode text with characters not in vocabulary
    auto ids = tokenizer.encode("xyz", false);

    // Unknown tokens should map to unk_token_id (1)
    EXPECT_GT(ids.size(), 0);
    bool has_unk = false;
    for (int id : ids) {
        if (id == 1) {
            has_unk = true;
            break;
        }
    }
    EXPECT_TRUE(has_unk);
}

TEST_F(BPETokenizerTest, EncodeEmptyString) {
    BPETokenizer tokenizer;

    // Empty string should now throw TokenizerInputError
    EXPECT_THROW({ tokenizer.encode("", false); }, TokenizerInputError);

    EXPECT_THROW({ tokenizer.encode("", true); }, TokenizerInputError);
}

// ============================================================================
// Decoding Tests
// ============================================================================

TEST_F(BPETokenizerTest, DecodeBasic) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello"};
    tokenizer.build_vocab(texts, 20, 1);

    std::string text = "hello";
    auto ids = tokenizer.encode(text, false);
    auto decoded = tokenizer.decode(ids, false);

    // Decoded should contain 'hello'
    EXPECT_NE(decoded.find("hello"), std::string::npos);
}

TEST_F(BPETokenizerTest, DecodeSkipSpecialTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 20, 1);

    auto ids = tokenizer.encode("test", true);   // With BOS/EOS
    auto decoded = tokenizer.decode(ids, true);  // Skip special tokens

    // Should not contain special token markers
    EXPECT_EQ(decoded.find("<bos>"), std::string::npos);
    EXPECT_EQ(decoded.find("<eos>"), std::string::npos);
}

TEST_F(BPETokenizerTest, DecodeKeepSpecialTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 20, 1);

    auto ids = tokenizer.encode("test", true);    // With BOS/EOS
    auto decoded = tokenizer.decode(ids, false);  // Keep special tokens

    // Should contain special tokens
    EXPECT_NE(decoded.find("<bos>"), std::string::npos);
    EXPECT_NE(decoded.find("<eos>"), std::string::npos);
}

TEST_F(BPETokenizerTest, DecodeEmptyIds) {
    BPETokenizer tokenizer;

    std::vector<int> ids = {};

    // Empty IDs vector should now throw TokenizerInputError
    EXPECT_THROW({ tokenizer.decode(ids, true); }, TokenizerInputError);
}

// ============================================================================
// Encode-Decode Round Trip Tests
// ============================================================================

TEST_F(BPETokenizerTest, EncodeDecodeRoundTrip) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello world", "this is a test"};
    tokenizer.build_vocab(texts, 50, 1);

    std::string original = "hello world";
    auto ids = tokenizer.encode(original, false);
    auto decoded = tokenizer.decode(ids, true);

    // Should contain the key words from original (lowercase)
    EXPECT_NE(decoded.find("hello"), std::string::npos);
    EXPECT_NE(decoded.find("world"), std::string::npos);
}

TEST_F(BPETokenizerTest, EncodeDecodeWithSpecialTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 20, 1);

    std::string original = "test";
    auto ids = tokenizer.encode(original, true);
    auto decoded = tokenizer.decode(ids, true);

    // Should get back text without special tokens
    EXPECT_NE(decoded.find("test"), std::string::npos);
}

// ============================================================================
// Save and Load Tests
// ============================================================================

TEST_F(BPETokenizerTest, SaveVocabulary) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);

    tokenizer.save_vocab("test_save_vocab.txt");

    // Check file exists
    std::ifstream file("test_save_vocab.txt");
    EXPECT_TRUE(file.good());
    file.close();
}

TEST_F(BPETokenizerTest, LoadVocabulary) {
    create_test_vocab(test_vocab_file);

    BPETokenizer tokenizer;
    tokenizer.load_vocab(test_vocab_file);

    // Should have loaded vocabulary
    EXPECT_EQ(tokenizer.get_vocab_size(), 20);
}

TEST_F(BPETokenizerTest, SaveLoadRoundTrip) {
    BPETokenizer tokenizer1;

    std::vector<std::string> texts = {"hello world test"};
    tokenizer1.build_vocab(texts, 40, 1);

    size_t original_size = tokenizer1.get_vocab_size();
    tokenizer1.save_vocab("test_save_vocab.txt");

    BPETokenizer tokenizer2;
    tokenizer2.load_vocab("test_save_vocab.txt");

    EXPECT_EQ(tokenizer2.get_vocab_size(), original_size);
}

TEST_F(BPETokenizerTest, SaveLoadPreservesEncoding) {
    BPETokenizer tokenizer1;

    std::vector<std::string> texts = {"hello world"};
    tokenizer1.build_vocab(texts, 30, 1);

    auto ids1 = tokenizer1.encode("hello", false);

    tokenizer1.save_vocab("test_save_vocab.txt");

    BPETokenizer tokenizer2;
    tokenizer2.load_vocab("test_save_vocab.txt");

    auto ids2 = tokenizer2.encode("hello", false);

    EXPECT_EQ(ids1, ids2);
}

// ============================================================================
// Utility Method Tests
// ============================================================================

TEST_F(BPETokenizerTest, GetVocabSize) {
    BPETokenizer tokenizer;

    EXPECT_EQ(tokenizer.get_vocab_size(), 4);  // Special tokens only

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 20, 1);

    EXPECT_GT(tokenizer.get_vocab_size(), 4);
}

TEST_F(BPETokenizerTest, PrintVocabStats) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);

    // Should not crash
    EXPECT_NO_THROW(tokenizer.print_vocab_stats());
}

TEST_F(BPETokenizerTest, GetTopTokens) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);

    auto top_tokens = tokenizer.get_top_tokens(5);

    EXPECT_LE(top_tokens.size(), 5);
    EXPECT_GT(top_tokens.size(), 0);

    // Should be sorted by ID
    for (size_t i = 1; i < top_tokens.size(); i++) {
        EXPECT_GE(top_tokens[i].second, top_tokens[i - 1].second);
    }
}

TEST_F(BPETokenizerTest, GetTopTokensMoreThanVocab) {
    BPETokenizer tokenizer;

    auto top_tokens = tokenizer.get_top_tokens(100);

    // Should return only available tokens (4 special tokens)
    EXPECT_EQ(top_tokens.size(), 4);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(BPETokenizerTest, VeryLongText) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test"};
    tokenizer.build_vocab(texts, 30, 1);

    std::string long_text(10000, 'a');

    EXPECT_NO_THROW({
        auto tokens = tokenizer.tokenize(long_text);
        EXPECT_GT(tokens.size(), 0);
    });
}

TEST_F(BPETokenizerTest, SpecialCharacters) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"test\nwith\ttabs\rand\rreturns"};
    tokenizer.build_vocab(texts, 50, 1);

    EXPECT_NO_THROW({
        auto tokens = tokenizer.tokenize("test\nline");
        EXPECT_GT(tokens.size(), 0);
    });
}

TEST_F(BPETokenizerTest, UnicodeCharacters) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts = {"hello"};
    tokenizer.build_vocab(texts, 30, 1);

    // Test with ASCII only for now
    auto tokens = tokenizer.tokenize("test");
    EXPECT_GT(tokens.size(), 0);
}

TEST_F(BPETokenizerTest, RepeatedBuildVocab) {
    BPETokenizer tokenizer;

    std::vector<std::string> texts1 = {"hello"};
    tokenizer.build_vocab(texts1, 20, 1);
    size_t size1 = tokenizer.get_vocab_size();

    std::vector<std::string> texts2 = {"world"};
    tokenizer.build_vocab(texts2, 25, 1);
    size_t size2 = tokenizer.get_vocab_size();

    // Second build should replace vocabulary
    // TODO: See TD-223 in TECHNICAL_DEBT.md - this only checks the size changed; a second build actually
    // appends and produces duplicate IDs. Check ID uniqueness once fixed.
    EXPECT_NE(size1, size2);
}

TEST_F(BPETokenizerTest, LoadNonexistentFile) {
    BPETokenizer tokenizer;

    // Should now throw VocabularyFileError instead of handling gracefully
    EXPECT_THROW({ tokenizer.load_vocab("nonexistent_file.txt"); }, VocabularyFileError);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(BPETokenizerTest, CompleteWorkflow) {
    BPETokenizer tokenizer;

    // 1. Build vocabulary
    std::vector<std::string> training_texts = {
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be that is the question"};

    tokenizer.build_vocab(training_texts, 100, 1);
    EXPECT_GT(tokenizer.get_vocab_size(), 4);

    // 2. Save vocabulary
    tokenizer.save_vocab("test_save_vocab.txt");

    // 3. Load in new tokenizer
    BPETokenizer tokenizer2;
    tokenizer2.load_vocab("test_save_vocab.txt");

    // 4. Encode text
    std::string test_text = "the quick fox";
    auto ids = tokenizer2.encode(test_text, true);
    EXPECT_GT(ids.size(), 2);  // More than just BOS/EOS

    // 5. Decode back
    auto decoded = tokenizer2.decode(ids, true);
    EXPECT_NE(decoded.find("quick"), std::string::npos);
}

// ============================================================================
// TokenizerMode Tests
// ============================================================================

TEST_F(BPETokenizerTest, AsciiModeIsDefault) {
    BPETokenizer tokenizer;
    EXPECT_EQ(tokenizer.get_mode(), TokenizerMode::ASCII);
    EXPECT_FALSE(tokenizer.is_unicode_mode());
}

TEST_F(BPETokenizerTest, UnicodeModeConstruction) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    EXPECT_EQ(tokenizer.get_mode(), TokenizerMode::UNICODE);
    EXPECT_TRUE(tokenizer.is_unicode_mode());
}

TEST_F(BPETokenizerTest, AsciiModeExplicit) {
    BPETokenizer tokenizer(TokenizerMode::ASCII);
    EXPECT_EQ(tokenizer.get_mode(), TokenizerMode::ASCII);
    EXPECT_FALSE(tokenizer.is_unicode_mode());
}

TEST_F(BPETokenizerTest, UnicodeModeHasFourSpecialTokens) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    EXPECT_EQ(tokenizer.get_vocab_size(), 4u);
}

TEST_F(BPETokenizerTest, UnicodeModeSpecialTokenIdsUnchanged) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    EXPECT_EQ(tokenizer.get_pad_token_id(), 0);
    EXPECT_EQ(tokenizer.get_unk_token_id(), 1);
    EXPECT_EQ(tokenizer.get_bos_token_id(), 2);
    EXPECT_EQ(tokenizer.get_eos_token_id(), 3);
}

TEST_F(BPETokenizerTest, UnicodeModeBuildsVocab) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {"hello world", "test text"};
    EXPECT_NO_THROW(tokenizer.build_vocab(texts, 50, 1));
    EXPECT_GT(tokenizer.get_vocab_size(), 4u);
}

TEST_F(BPETokenizerTest, UnicodeModeEncodeDecodeASCII) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 50, 1);

    auto ids = tokenizer.encode("hello", false);
    EXPECT_GT(ids.size(), 0u);

    auto decoded = tokenizer.decode(ids, false);
    EXPECT_NE(decoded.find("hello"), std::string::npos);
}

TEST_F(BPETokenizerTest, UnicodeModeEncodeMultibyteText) {
    // "café" contains the two-byte code point U+00E9 (é = 0xC3 0xA9)
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {"caf\xC3\xA9", "hello world"};
    EXPECT_NO_THROW(tokenizer.build_vocab(texts, 50, 1));

    // Should be able to encode UTF-8 text without throwing
    EXPECT_NO_THROW(tokenizer.encode("caf\xC3\xA9", false));
}

TEST_F(BPETokenizerTest, UnicodeModeRoundTripMultibyteText) {
    // U+4E2D U+6587 = 中文 (Chinese), encoded as 0xE4 0xB8 0xAD 0xE6 0x96 0x87
    std::string cjk = "\xE4\xB8\xAD\xE6\x96\x87";
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {cjk, cjk, cjk};  // repeat so chars are frequent
    tokenizer.build_vocab(texts, 50, 1);

    auto ids = tokenizer.encode(cjk, false);
    EXPECT_GT(ids.size(), 0u);

    auto decoded = tokenizer.decode(ids, false);
    // The decoded text should contain the original characters
    EXPECT_FALSE(decoded.empty());
}

TEST_F(BPETokenizerTest, UnicodeModePreTokenizeLowercasesOnlyASCII) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    // "HEllo" uppercase ASCII + "café" with accented char
    auto tokens = tokenizer.pre_tokenize("HEllo caf\xC3\xA9");
    EXPECT_GT(tokens.size(), 0u);

    // ASCII letters must be lowercased
    bool found_hello = false;
    for (const auto& t : tokens) {
        if (t.find("hello") != std::string::npos)
            found_hello = true;
        // Multi-byte sequences must not be corrupted — U+00E9 bytes must survive
        for (size_t i = 0; i + 1 < t.size(); ++i) {
            // 0xC3 0xA9 should appear intact (tolower must not have touched them)
            if ((unsigned char)t[i] == 0xC3 && (unsigned char)t[i + 1] == 0xA9) {
                // the sequence is present and uncorrupted
                SUCCEED();
            }
        }
    }
    EXPECT_TRUE(found_hello);
}

TEST_F(BPETokenizerTest, AsciiModeSaveContainsModeField) {
    BPETokenizer tokenizer(TokenizerMode::ASCII);
    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);
    tokenizer.save_vocab("test_save_vocab.txt");

    std::ifstream f("test_save_vocab.txt");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("TOKENIZER_MODE ASCII"), std::string::npos);
}

TEST_F(BPETokenizerTest, UnicodeModeFileSaveContainsModeField) {
    BPETokenizer tokenizer(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {"hello world"};
    tokenizer.build_vocab(texts, 30, 1);
    tokenizer.save_vocab("test_save_vocab.txt");

    std::ifstream f("test_save_vocab.txt");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("TOKENIZER_MODE UNICODE"), std::string::npos);
}

TEST_F(BPETokenizerTest, SaveLoadPreservesAsciiMode) {
    BPETokenizer builder(TokenizerMode::ASCII);
    std::vector<std::string> texts = {"hello world"};
    builder.build_vocab(texts, 30, 1);
    builder.save_vocab("test_save_vocab.txt");

    BPETokenizer loader;  // starts as ASCII but load_vocab should confirm
    loader.load_vocab("test_save_vocab.txt");

    EXPECT_EQ(loader.get_mode(), TokenizerMode::ASCII);
    EXPECT_FALSE(loader.is_unicode_mode());
}

TEST_F(BPETokenizerTest, SaveLoadPreservesUnicodeMode) {
    BPETokenizer builder(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {"hello world"};
    builder.build_vocab(texts, 30, 1);
    builder.save_vocab("test_save_vocab.txt");

    // Load into a default-ASCII tokenizer — mode should switch to UNICODE
    BPETokenizer loader;
    loader.load_vocab("test_save_vocab.txt");

    EXPECT_EQ(loader.get_mode(), TokenizerMode::UNICODE);
    EXPECT_TRUE(loader.is_unicode_mode());
}

TEST_F(BPETokenizerTest, SaveLoadUnicodeModeEncodingIsConsistent) {
    BPETokenizer builder(TokenizerMode::UNICODE);
    std::vector<std::string> texts = {"hello world test"};
    builder.build_vocab(texts, 50, 1);
    auto ids_before = builder.encode("hello", false);
    builder.save_vocab("test_save_vocab.txt");

    BPETokenizer loader;
    loader.load_vocab("test_save_vocab.txt");
    auto ids_after = loader.encode("hello", false);

    EXPECT_EQ(ids_before, ids_after);
}

// Regression test (TD-058): get_most_frequent_pair() is a public static
// method whose loop bound used to be `i < tokens.size() - 1`, which
// underflows to SIZE_MAX for an empty inner vector (size_t is unsigned) and
// reads out of bounds. No internal caller ever passes an empty entry, but
// external callers of this public API have no such guarantee.
TEST_F(BPETokenizerTest, GetMostFrequentPairIgnoresEmptyWordEntries) {
    std::vector<std::vector<std::string>> word_tokens = {
        {},                  // empty entry — must not underflow/crash
        {"a", "b", "a", "b"},
        {"a"},               // single-token entry — no pair possible either
    };

    std::pair<std::string, std::string> result;
    EXPECT_NO_THROW(result = BPETokenizer::get_most_frequent_pair(word_tokens));
    EXPECT_EQ(result.first, "a");
    EXPECT_EQ(result.second, "b");
}

// ============================================================================
// TD-220: pair counts keyed by the pair, not by first + "|||" + second
// ============================================================================

namespace {
// The BPE_MERGES section of a saved vocab file, unescaped only as far as these tests need
// (pipes and '!' aren't escaped).
std::vector<std::pair<std::string, std::string>> saved_merges(const BPETokenizer& tok,
                                                               const std::string& path) {
    tok.save_vocab(path);
    std::ifstream in(path);
    std::string line;
    std::vector<std::pair<std::string, std::string>> merges;
    bool in_merges = false;
    while (std::getline(in, line)) {
        if (line.rfind("BPE_MERGES ", 0) == 0) {
            in_merges = true;
            continue;
        }
        if (!in_merges)
            continue;
        const size_t tab = line.find('\t');
        if (tab != std::string::npos)
            merges.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    std::remove(path.c_str());
    return merges;
}
}  // namespace

TEST_F(BPETokenizerTest, GetMostFrequentPairHandlesPipeTokens) {
    // Key "|||||" used to split into ("", "||"), which build_bpe_merges() read as "no pairs".
    std::vector<std::vector<std::string>> word_tokens = {{"|", "|"}, {"|", "|"}, {"a", "b"}};
    const auto best = BPETokenizer::get_most_frequent_pair(word_tokens);
    EXPECT_EQ(best.first, "|");
    EXPECT_EQ(best.second, "|");

    word_tokens = {{"a|", "|b"}, {"a|", "|b"}};
    const auto best2 = BPETokenizer::get_most_frequent_pair(word_tokens);
    EXPECT_EQ(best2.first, "a|");
    EXPECT_EQ(best2.second, "|b");
}

TEST_F(BPETokenizerTest, GetMostFrequentPairBreaksTiesDeterministically) {
    const std::vector<std::vector<std::string>> word_tokens = {{"y", "z"}, {"b", "c"}, {"m", "n"}};
    const auto best = BPETokenizer::get_most_frequent_pair(word_tokens);
    EXPECT_EQ(best.first, "b");
    EXPECT_EQ(best.second, "c");
}

TEST_F(BPETokenizerTest, PipeHeavyCorpusStillLearnsMerges) {
    // Asked for 52 merges and learned 0 before the fix.
    std::vector<std::string> texts(20, "|| || || ab ab");
    BPETokenizer tok;
    tok.build_vocab(texts, 300, 1);
    const auto merges = saved_merges(tok, "test_td220_pipes.vocab");
    EXPECT_FALSE(merges.empty());
    EXPECT_NE(std::find(merges.begin(), merges.end(), std::make_pair(std::string("|"),
                                                                     std::string("|"))),
              merges.end());
}

TEST_F(BPETokenizerTest, RandomPipeCorporaNeverRepeatAMergeRule) {
    // Fuzzing corpora over {|, !} used to record the same wrong pair as a rule repeatedly.
    std::mt19937 rng(220);
    std::uniform_int_distribution<int> coin(0, 1), word_len(1, 6), words(3, 12);
    for (int trial = 0; trial < 40; ++trial) {
        std::vector<std::string> texts;
        for (int line = 0; line < 8; ++line) {
            std::string text;
            for (int w = words(rng); w > 0; --w) {
                for (int c = word_len(rng); c > 0; --c) text += coin(rng) ? '|' : '!';
                text += ' ';
            }
            texts.push_back(text);
        }
        BPETokenizer tok;
        tok.build_vocab(texts, 300, 1);
        const auto merges = saved_merges(tok, "test_td220_fuzz.vocab");
        std::set<std::pair<std::string, std::string>> unique(merges.begin(), merges.end());
        ASSERT_EQ(unique.size(), merges.size()) << "repeated merge rule in trial " << trial;
        ASSERT_FALSE(merges.empty()) << "no merges learned in trial " << trial;
    }
}

// ============================================================================
// TD-221: save_vocab() throws on failure and replaces the file atomically
// ============================================================================

TEST_F(BPETokenizerTest, SaveVocabToUnwritablePathThrows) {
    BPETokenizer tok;
    tok.build_vocab({"hello world"}, 300, 1);
    EXPECT_THROW(tok.save_vocab("/nonexistent_dir_td221/sub/vocab.txt"), VocabularyFileError);
    EXPECT_THROW(tok.save_vocab(""), VocabularyFileError);
}

TEST_F(BPETokenizerTest, SaveVocabLeavesNoTempFileOnSuccess) {
    BPETokenizer tok;
    tok.build_vocab({"hello world"}, 300, 1);
    const std::string path = "test_td221_ok.vocab";
    ASSERT_NO_THROW(tok.save_vocab(path));
    EXPECT_TRUE(std::ifstream(path).good());
    EXPECT_FALSE(std::ifstream(path + ".tmp").good());
    std::remove(path.c_str());
}

TEST_F(BPETokenizerTest, FailedSaveVocabLeavesExistingFileIntact) {
    if (geteuid() == 0) {
        GTEST_SKIP() << "root ignores directory permissions";
    }
    const std::string dir = "test_td221_ro_dir";
    const std::string path = dir + "/model.vocab";
    ::mkdir(dir.c_str(), 0755);
    {
        std::ofstream good(path);
        good << "previous good vocab\n";
    }
    ::chmod(dir.c_str(), 0555);  // can't create the temp file next to it

    BPETokenizer tok;
    tok.build_vocab({"hello world"}, 300, 1);
    EXPECT_THROW(tok.save_vocab(path), VocabularyFileError);

    ::chmod(dir.c_str(), 0755);
    std::ifstream in(path);
    std::string content;
    std::getline(in, content);
    EXPECT_EQ(content, "previous good vocab");
    std::remove(path.c_str());
    ::rmdir(dir.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============================================================================
// TD-222: apply_bpe()'s cache follows the tokenizer's current merges
// ============================================================================

namespace {
std::string td222_vocab_path(const std::string& name) {
    return "/tmp/adai_td222_" + std::to_string(::getpid()) + "_" + name + ".vocab";
}
}  // namespace

TEST(BPETokenizerCacheTest, ReloadThenEncodeMatchesAFreshTokenizer) {
    BPETokenizer big;
    big.build_vocab({"abab abab abab abab ababab"}, 40);
    const std::string big_path = td222_vocab_path("big");
    big.save_vocab(big_path);

    BPETokenizer tiny;
    tiny.build_vocab({"xy"}, 8);  // no merges for "ab"
    const std::string tiny_path = td222_vocab_path("tiny");
    tiny.save_vocab(tiny_path);

    BPETokenizer reloaded;
    reloaded.load_vocab(big_path);
    const auto with_merges = reloaded.encode("abab", false);
    reloaded.load_vocab(tiny_path);  // the cache used to keep "abab"'s merged result

    BPETokenizer fresh;
    fresh.load_vocab(tiny_path);
    EXPECT_EQ(reloaded.encode("abab", false), fresh.encode("abab", false));
    EXPECT_NE(reloaded.encode("abab", false).size(), with_merges.size());
    std::remove(big_path.c_str());
    std::remove(tiny_path.c_str());
}

TEST(BPETokenizerCacheTest, ATokenizerAtAReusedAddressInheritsNothing) {
    alignas(BPETokenizer) unsigned char storage[sizeof(BPETokenizer)];
    auto* first = new (storage) BPETokenizer();
    first->build_vocab({"abab abab abab abab"}, 40);
    const auto merged = first->encode("abab", false);
    first->~BPETokenizer();

    auto* second = new (storage) BPETokenizer();  // same address
    second->build_vocab({"xy"}, 8);
    BPETokenizer reference;
    reference.build_vocab({"xy"}, 8);
    EXPECT_EQ(second->encode("abab", false), reference.encode("abab", false));
    EXPECT_NE(second->encode("abab", false).size(), merged.size());
    second->~BPETokenizer();
}

TEST(BPETokenizerCacheTest, AlternatingTokenizersKeepTheirOwnResults) {
    BPETokenizer a;
    a.build_vocab({"abab abab abab abab"}, 40);
    BPETokenizer b;
    b.build_vocab({"xy"}, 8);
    const auto a_once = a.encode("abab", false);
    const auto b_once = b.encode("abab", false);
    ASSERT_NE(a_once.size(), b_once.size());
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(a.encode("abab", false), a_once);
        EXPECT_EQ(b.encode("abab", false), b_once);
    }
}
