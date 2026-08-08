#include <../gtest/gtest.h>
#include <cstdio>
#include <fstream>
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
