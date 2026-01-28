#include <gtest/gtest.h>
#include "BPETokenizer.hpp"
#include <fstream>

/**
 * @file tokenizer_error_handling_test.cpp
 * @brief Test suite for BPE Tokenizer error handling improvements (TD-002)
 * 
 * Tests custom exception types, input validation, UTF-8 validation,
 * and vocabulary file format validation.
 */

// Test fixture for tokenizer error handling
class TokenizerErrorHandlingTest : public ::testing::Test {
protected:
    BPETokenizer tokenizer;
    std::string test_vocab_file = "test_vocab_error.txt";
    
    void SetUp() override {
        // Initialize tokenizer with basic vocab for some tests
        std::vector<std::string> sample_texts = {"hello world", "test text"};
        tokenizer.build_vocab(sample_texts, 100);
    }
    
    void TearDown() override {
        // Clean up test files
        std::remove(test_vocab_file.c_str());
    }
    
    // Helper to create a malformed vocab file
    void create_malformed_vocab_file(const std::string& content) {
        std::ofstream file(test_vocab_file);
        file << content;
        file.close();
    }
};

// ============================================================================
// Input Validation Tests
// ============================================================================

TEST_F(TokenizerErrorHandlingTest, EncodeEmptyStringThrowsException) {
    EXPECT_THROW({
        tokenizer.encode("");
    }, TokenizerInputError);
}

TEST_F(TokenizerErrorHandlingTest, EncodeEmptyStringExceptionMessage) {
    try {
        tokenizer.encode("");
        FAIL() << "Expected TokenizerInputError";
    } catch (const TokenizerInputError& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Input text is empty") != std::string::npos);
        EXPECT_TRUE(msg.find("encode()") != std::string::npos);
    }
}

TEST_F(TokenizerErrorHandlingTest, DecodeEmptyVectorThrowsException) {
    std::vector<int> empty_ids;
    EXPECT_THROW({
        tokenizer.decode(empty_ids);
    }, TokenizerInputError);
}

TEST_F(TokenizerErrorHandlingTest, DecodeEmptyVectorExceptionMessage) {
    std::vector<int> empty_ids;
    try {
        tokenizer.decode(empty_ids);
        FAIL() << "Expected TokenizerInputError";
    } catch (const TokenizerInputError& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("token ID vector is empty") != std::string::npos);
        EXPECT_TRUE(msg.find("decode()") != std::string::npos);
    }
}

// ============================================================================
// UTF-8 Validation Tests
// ============================================================================

TEST_F(TokenizerErrorHandlingTest, EncodeInvalidUTF8ThrowsException) {
    // Invalid UTF-8 sequence: continuation byte without start byte
    std::string invalid_utf8 = "hello\x80world";
    EXPECT_THROW({
        tokenizer.encode(invalid_utf8);
    }, TokenizerEncodingError);
}

TEST_F(TokenizerErrorHandlingTest, EncodeInvalidUTF8ExceptionMessage) {
    // Invalid UTF-8: incomplete multi-byte sequence
    std::string invalid_utf8 = "test\xF0\x90";  // Start of 4-byte but incomplete
    try {
        tokenizer.encode(invalid_utf8);
        FAIL() << "Expected TokenizerEncodingError";
    } catch (const TokenizerEncodingError& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("invalid UTF-8") != std::string::npos);
    }
}

TEST_F(TokenizerErrorHandlingTest, EncodeValidUTF8Succeeds) {
    // Valid UTF-8 with multi-byte characters
    std::string valid_utf8 = "Hello 世界 🌍";
    EXPECT_NO_THROW({
        tokenizer.encode(valid_utf8);
    });
}

TEST_F(TokenizerErrorHandlingTest, EncodeASCIISucceeds) {
    std::string ascii_text = "Hello World 123!";
    EXPECT_NO_THROW({
        auto ids = tokenizer.encode(ascii_text);
        EXPECT_GT(ids.size(), 0);
    });
}

// ============================================================================
// Token ID Validation Tests
// ============================================================================

TEST_F(TokenizerErrorHandlingTest, DecodeNegativeTokenIDThrowsException) {
    std::vector<int> invalid_ids = {1, 2, -5, 3};
    EXPECT_THROW({
        tokenizer.decode(invalid_ids);
    }, TokenIDError);
}

TEST_F(TokenizerErrorHandlingTest, DecodeNegativeTokenIDExceptionMessage) {
    std::vector<int> invalid_ids = {-10};
    try {
        tokenizer.decode(invalid_ids);
        FAIL() << "Expected TokenIDError";
    } catch (const TokenIDError& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Token ID") != std::string::npos);
        EXPECT_TRUE(msg.find("negative") != std::string::npos);
    }
}

TEST_F(TokenizerErrorHandlingTest, DecodeUnknownTokenIDWarning) {
    // This should warn but not throw
    std::vector<int> ids_with_unknown = {2, 99999, 3};  // 99999 likely doesn't exist
    
    // Redirect stderr to capture warning
    testing::internal::CaptureStderr();
    
    std::string result;
    EXPECT_NO_THROW({
        result = tokenizer.decode(ids_with_unknown);
    });
    
    std::string stderr_output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(stderr_output.find("Warning: Unknown token ID") != std::string::npos);
}

// ============================================================================
// Vocabulary File Validation Tests
// ============================================================================

TEST_F(TokenizerErrorHandlingTest, LoadVocabEmptyFilenameThrowsException) {
    EXPECT_THROW({
        tokenizer.load_vocab("");
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabNonexistentFileThrowsException) {
    EXPECT_THROW({
        tokenizer.load_vocab("nonexistent_file_12345.txt");
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabNonexistentFileExceptionMessage) {
    try {
        tokenizer.load_vocab("nonexistent_file_12345.txt");
        FAIL() << "Expected VocabularyFileError";
    } catch (const VocabularyFileError& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Could not open") != std::string::npos);
    }
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabMalformedSpecialTokensThrowsException) {
    // Missing space separator in special tokens
    create_malformed_vocab_file(
        "VOCAB_SIZE 10\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id0\n"  // Missing space
        "VOCAB\n"
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabInvalidSpecialTokenIDThrowsException) {
    // Invalid integer for special token ID
    create_malformed_vocab_file(
        "VOCAB_SIZE 10\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id invalid\n"
        "VOCAB\n"
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabMalformedVocabLineThrowsException) {
    // Missing tab separator in vocab line
    create_malformed_vocab_file(
        "VOCAB_SIZE 4\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id 0\n"
        "VOCAB\n"
        "<pad> 0\n"  // Missing tab, using space
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabInvalidTokenIDThrowsException) {
    // Invalid token ID (not an integer)
    create_malformed_vocab_file(
        "VOCAB_SIZE 4\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id 0\n"
        "VOCAB\n"
        "<pad>\tnotanumber\n"
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabNegativeTokenIDThrowsException) {
    // Negative token ID
    create_malformed_vocab_file(
        "VOCAB_SIZE 4\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id 0\n"
        "VOCAB\n"
        "<pad>\t-5\n"
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabMalformedBPEMergeThrowsException) {
    // Missing tab in BPE merge
    create_malformed_vocab_file(
        "VOCAB_SIZE 4\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id 0\n"
        "unk_token_id 1\n"
        "bos_token_id 2\n"
        "eos_token_id 3\n"
        "VOCAB\n"
        "<pad>\t0\n"
        "<unk>\t1\n"
        "<bos>\t2\n"
        "<eos>\t3\n"
        "BPE_MERGES 1\n"
        "ab\n"  // Missing second part
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabEmptyVocabularyThrowsException) {
    // Vocabulary section exists but is empty
    create_malformed_vocab_file(
        "VOCAB_SIZE 0\n"
        "VOCAB\n"
        "BPE_MERGES 0\n"
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabMissingSpecialTokensThrowsException) {
    // Missing required special tokens
    create_malformed_vocab_file(
        "VOCAB_SIZE 2\n"
        "VOCAB\n"
        "hello\t0\n"
        "world\t1\n"
        "BPE_MERGES 0\n"
    );
    
    EXPECT_THROW({
        tokenizer.load_vocab(test_vocab_file);
    }, VocabularyFileError);
}

TEST_F(TokenizerErrorHandlingTest, LoadVocabValidFileSucceeds) {
    // Create a valid vocabulary file
    create_malformed_vocab_file(
        "VOCAB_SIZE 4\n"
        "SPECIAL_TOKENS\n"
        "pad_token_id 0\n"
        "unk_token_id 1\n"
        "bos_token_id 2\n"
        "eos_token_id 3\n"
        "VOCAB\n"
        "<pad>\t0\n"
        "<unk>\t1\n"
        "<bos>\t2\n"
        "<eos>\t3\n"
        "BPE_MERGES 0\n"
    );
    
    EXPECT_NO_THROW({
        tokenizer.load_vocab(test_vocab_file);
        EXPECT_EQ(tokenizer.get_vocab_size(), 4);
    });
}

// ============================================================================
// Exception Type Hierarchy Tests
// ============================================================================

TEST_F(TokenizerErrorHandlingTest, TokenizerInputErrorIsInvalidArgument) {
    try {
        tokenizer.encode("");
        FAIL() << "Expected exception";
    } catch (const std::invalid_argument& e) {
        // Should catch as base class
        SUCCEED();
    }
}

TEST_F(TokenizerErrorHandlingTest, TokenizerEncodingErrorIsRuntimeError) {
    try {
        tokenizer.encode("invalid\x80utf8");
        FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
        // Should catch as base class
        SUCCEED();
    }
}

TEST_F(TokenizerErrorHandlingTest, VocabularyFileErrorIsRuntimeError) {
    try {
        tokenizer.load_vocab("nonexistent.txt");
        FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
        // Should catch as base class
        SUCCEED();
    }
}

TEST_F(TokenizerErrorHandlingTest, TokenIDErrorIsOutOfRange) {
    try {
        std::vector<int> ids = {-1};
        tokenizer.decode(ids);
        FAIL() << "Expected exception";
    } catch (const std::out_of_range& e) {
        // Should catch as base class
        SUCCEED();
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
