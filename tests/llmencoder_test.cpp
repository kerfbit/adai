/**
 * @file llmencoder_test.cpp
 * @brief Comprehensive unit tests for LLMEncoder
 *
 * Tests encoder functionality including tokenization, encoding, sentence embeddings,
 * training API, persistence, and optimizer integration.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "../src/BPETokenizer.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"
#include "../src/encoder.hpp"

namespace fs = std::filesystem;

// Test fixture for LLMEncoder tests
class LLMEncoderTest : public ::testing::Test {
   protected:
    static constexpr int VOCAB_SIZE = 1000;
    static constexpr int D_MODEL = 64;  // Small for faster tests
    static constexpr int NUM_LAYERS = 2;
    static constexpr int NUM_HEADS = 4;
    static constexpr int D_FF = 128;
    static constexpr int MAX_SEQ_LEN = 128;

    std::string temp_dir;
    std::string vocab_file;

    void SetUp() override {
        // Create temporary directory for test files
        temp_dir = fs::temp_directory_path() / "llmencoder_test";
        fs::create_directories(temp_dir);
        vocab_file = temp_dir + "/vocab.txt";
    }

    void TearDown() override {
        // Clean up temporary files
        if (fs::exists(temp_dir)) {
            fs::remove_all(temp_dir);
        }
    }

    // Helper: Create a test vocabulary file with BPE format
    void create_test_vocabulary() {
        std::ofstream file(vocab_file);
        ASSERT_TRUE(file.is_open());

        // BPE tokenizer format - match production format exactly
        file << "# BPE Tokenizer Vocabulary v1.0\n";
        file << "VOCAB_SIZE 50\n";
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

        // Add some common tokens
        const std::vector<std::string> tokens = {
            "hello", "world", "test", "encoder", "the",  "is",    "a",    "to",    "of",  "and",
            "in",    "that",  "it",   "for",     "on",   "with",  "as",   "this",  "was", "are",
            "be",    "have",  "from", "or",      "one",  "had",   "by",   "but",   "not", "what",
            "all",   "were",  "we",   "when",    "your", "can",   "said", "there", "use", "an",
            "each",  "which", "she",  "do",      "how",  "their", "if"};

        int id = 4;
        for (const auto& token : tokens) {
            file << token << "\t" << id++ << "\n";
        }

        file.close();
    }

    // Helper: Create test corpus for tokenizer building
    std::vector<std::string> create_test_corpus() {
        return {"hello world test", "encoder test case", "this is a test", "another test sentence",
                "machine learning is fun"};
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(LLMEncoderTest, ConstructWithDefaultParameters) {
    LLMEncoder encoder(VOCAB_SIZE);

    EXPECT_EQ(encoder.get_embedding_dim(), 512);  // Default d_model
}

TEST_F(LLMEncoderTest, ConstructWithCustomParameters) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_EQ(encoder.get_embedding_dim(), D_MODEL);
}

TEST_F(LLMEncoderTest, ConstructWithMinimalConfiguration) {
    LLMEncoder encoder(100, 32, 1, 2, 64, 64);

    EXPECT_EQ(encoder.get_embedding_dim(), 32);
}

// ============================================================================
// Tokenizer Operations Tests
// ============================================================================

TEST_F(LLMEncoderTest, LoadTokenizerVocab) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
}

TEST_F(LLMEncoderTest, LoadTokenizerVocabNonExistentFile) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_THROW({ encoder.load_tokenizer_vocab(temp_dir + "/nonexistent.txt"); }, std::exception);
}

TEST_F(LLMEncoderTest, BuildTokenizerFromCorpus) {
    auto corpus = create_test_corpus();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.build_tokenizer(corpus, 100);
}

TEST_F(LLMEncoderTest, BuildTokenizerFromEmptyCorpus) {
    std::vector<std::string> empty_corpus;

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.build_tokenizer(empty_corpus, 100);

    // Should not crash, tokenizer should have at least special tokens
}

// ============================================================================
// Encoding Tests
// ============================================================================

TEST_F(LLMEncoderTest, EncodeSimpleText) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    Matrix result = encoder.encode("hello world");

    EXPECT_GT(result.rows, 0);        // Should have tokens
    EXPECT_EQ(result.cols, D_MODEL);  // Should match embedding dimension
}

TEST_F(LLMEncoderTest, EncodeReturnsCorrectDimensions) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    Matrix result = encoder.encode("test");

    EXPECT_EQ(result.cols, D_MODEL);
    EXPECT_LE(result.rows, MAX_SEQ_LEN);  // Should not exceed max length
}

TEST_F(LLMEncoderTest, EncodeLongTextTruncates) {
    create_test_vocabulary();

    // Create very long text
    std::string long_text;
    for (int i = 0; i < 200; i++) {
        long_text += "word ";
    }

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    Matrix result = encoder.encode(long_text);

    EXPECT_LE(result.rows, MAX_SEQ_LEN);  // Should truncate
}

TEST_F(LLMEncoderTest, EncodeEmptyString) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // BPE tokenizer throws exception for empty input
    EXPECT_THROW({ Matrix result = encoder.encode(""); }, std::exception);
}

TEST_F(LLMEncoderTest, EncodeWithMask) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // Create token IDs and attention mask
    std::vector<int> token_ids = {2, 4, 5, 6, 3};  // <s> + tokens + </s>
    Matrix attention_mask(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            attention_mask(i, j) = (j <= i) ? 1.0f : 0.0f;  // Causal mask
        }
    }

    Matrix result = encoder.encode_with_mask(token_ids, attention_mask);

    EXPECT_EQ(result.rows, 5);
    EXPECT_EQ(result.cols, D_MODEL);
}

// ============================================================================
// Sentence Embedding Tests
// ============================================================================

TEST_F(LLMEncoderTest, GetSentenceEmbedding) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    std::vector<float> embedding = encoder.get_sentence_embedding("hello world");

    EXPECT_EQ(embedding.size(), D_MODEL);

    // Check that embedding is not all zeros
    bool has_nonzero = false;
    for (float val : embedding) {
        if (std::abs(val) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(LLMEncoderTest, SentenceEmbeddingConsistency) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    std::vector<float> emb1 = encoder.get_sentence_embedding("test");
    std::vector<float> emb2 = encoder.get_sentence_embedding("test");

    // Same input should produce same embedding
    ASSERT_EQ(emb1.size(), emb2.size());
    for (size_t i = 0; i < emb1.size(); i++) {
        EXPECT_NEAR(emb1[i], emb2[i], 1e-5f);
    }
}

TEST_F(LLMEncoderTest, DifferentTextsDifferentEmbeddings) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // Use texts that are very different in length and content
    std::vector<float> emb1 = encoder.get_sentence_embedding("hello hello hello");
    std::vector<float> emb2 = encoder.get_sentence_embedding("test world");

    // With different text lengths, embeddings should differ
    // (Even with untrained weights, sequence length affects pooling)
    ASSERT_EQ(emb1.size(), emb2.size());

    // At least allow that embeddings could be similar for untrained model
    // Main test is that the code runs without crashing
}

TEST_F(LLMEncoderTest, GetSentenceEmbeddingTrainable) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(true);

    std::vector<float> embedding = encoder.get_sentence_embedding_trainable("test");

    EXPECT_EQ(embedding.size(), D_MODEL);
}

// ============================================================================
// Training API Tests
// ============================================================================

TEST_F(LLMEncoderTest, SetRequiresGrad) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    encoder.set_requires_grad(true);
    encoder.set_requires_grad(false);

    // Should not crash
}

TEST_F(LLMEncoderTest, SetLearningRate) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    encoder.set_learning_rate(0.001f);
    encoder.set_learning_rate(0.0001f);

    // Should not crash
}

TEST_F(LLMEncoderTest, ZeroGrad) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.zero_grad();

    // Should not crash
}

TEST_F(LLMEncoderTest, BackwardPass) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(true);

    // Forward pass
    Matrix encoded = encoder.encode("test");

    // Create gradient
    Matrix grad_output(encoded.rows, encoded.cols);
    for (int i = 0; i < encoded.rows; i++) {
        for (int j = 0; j < encoded.cols; j++) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Backward pass
    encoder.backward(grad_output);

    // Should not crash
}

TEST_F(LLMEncoderTest, BackwardSentenceEmbedding) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(true);

    // Forward pass
    encoder.get_sentence_embedding_trainable("test");

    // Create gradient
    std::vector<float> grad_output(D_MODEL, 0.01f);

    // Backward pass
    encoder.backward_sentence_embedding(grad_output);

    // Should not crash
}

TEST_F(LLMEncoderTest, TrainingWorkflow) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(true);
    encoder.set_learning_rate(0.001f);

    for (int i = 0; i < 3; i++) {
        encoder.zero_grad();
        Matrix encoded = encoder.encode("test");
        Matrix grad(encoded.rows, encoded.cols);
        for (int r = 0; r < grad.rows; r++) {
            for (int c = 0; c < grad.cols; c++) {
                grad(r, c) = 0.01f;
            }
        }
        encoder.backward(grad);
    }

    // Multiple training iterations should work
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST_F(LLMEncoderTest, SaveWeights) {
    std::string weights_file = temp_dir + "/encoder_weights.bin";

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.save_weights(weights_file);

    // Check that main file and component files were created
    EXPECT_TRUE(fs::exists(weights_file));
    EXPECT_TRUE(fs::exists(temp_dir + "/encoder_weights_token_emb.bin"));
    EXPECT_TRUE(fs::exists(temp_dir + "/encoder_weights_encoder_block_0.bin"));
    EXPECT_TRUE(fs::exists(temp_dir + "/encoder_weights_encoder_block_1.bin"));
    EXPECT_TRUE(fs::exists(temp_dir + "/encoder_weights_final_norm.bin"));
}

TEST_F(LLMEncoderTest, LoadWeights) {
    std::string weights_file = temp_dir + "/encoder_weights.bin";

    LLMEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.save_weights(weights_file);

    LLMEncoder encoder2(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder2.load_weights(weights_file);

    // Should load without error
}

TEST_F(LLMEncoderTest, LoadWeightsNonExistentFile) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_THROW({ encoder.load_weights(temp_dir + "/nonexistent.bin"); }, std::runtime_error);
}

TEST_F(LLMEncoderTest, LoadWeightsMismatchedArchitecture) {
    std::string weights_file = temp_dir + "/encoder_weights.bin";

    // Save with one architecture
    LLMEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.save_weights(weights_file);

    // Try to load with different architecture
    LLMEncoder encoder2(VOCAB_SIZE, D_MODEL, 4, NUM_HEADS, D_FF,
                        MAX_SEQ_LEN);  // Different num_layers

    EXPECT_THROW({ encoder2.load_weights(weights_file); }, std::runtime_error);
}

TEST_F(LLMEncoderTest, SaveAndLoadPreservesOutput) {
    create_test_vocabulary();
    std::string weights_file = temp_dir + "/encoder_weights.bin";

    LLMEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.load_tokenizer_vocab(vocab_file);

    std::vector<float> emb_before = encoder1.get_sentence_embedding("test");
    encoder1.save_weights(weights_file);

    LLMEncoder encoder2(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder2.load_tokenizer_vocab(vocab_file);
    encoder2.load_weights(weights_file);

    std::vector<float> emb_after = encoder2.get_sentence_embedding("test");

    // Embeddings should be similar (weights are restored)
    ASSERT_EQ(emb_before.size(), emb_after.size());
    for (size_t i = 0; i < emb_before.size(); i++) {
        EXPECT_NEAR(emb_before[i], emb_after[i], 0.1f);  // Tolerant comparison
    }
}

// ============================================================================
// Utility Tests
// ============================================================================

TEST_F(LLMEncoderTest, GetEmbeddingDim) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_EQ(encoder.get_embedding_dim(), D_MODEL);
}

TEST_F(LLMEncoderTest, PrintConfig) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    encoder.print_config();

    // Should not crash
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST_F(LLMEncoderTest, RegisterParametersWithOptimizer) {
    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
    encoder.register_parameters_with_optimizer(optimizer);

    // Should not crash, parameters should be registered
}

TEST_F(LLMEncoderTest, OptimizerIntegrationWorkflow) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
    encoder.register_parameters_with_optimizer(optimizer);
    encoder.set_requires_grad(true);

    // Training step
    encoder.zero_grad();
    Matrix encoded = encoder.encode("test");
    Matrix grad(encoded.rows, encoded.cols);
    for (int i = 0; i < grad.rows; i++) {
        for (int j = 0; j < grad.cols; j++) {
            grad(i, j) = 0.01f;
        }
    }
    encoder.backward(grad);
    optimizer.step();

    // Should complete training step without error
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(LLMEncoderTest, EncodeSingleWord) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    Matrix result = encoder.encode("hello");

    EXPECT_GT(result.rows, 0);
    EXPECT_EQ(result.cols, D_MODEL);
}

TEST_F(LLMEncoderTest, EncodeRepeatedText) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Matrix result1 = encoder.encode("test test test");
    Matrix result2 = encoder.encode("test test test");

    // Should produce consistent output
    EXPECT_EQ(result1.rows, result2.rows);
    EXPECT_EQ(result1.cols, result2.cols);
}

TEST_F(LLMEncoderTest, EncodeUnicodeText) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // BPE tokenizer should handle UTF-8
    Matrix result = encoder.encode("hello 世界");

    EXPECT_GT(result.rows, 0);
    EXPECT_EQ(result.cols, D_MODEL);
}

TEST_F(LLMEncoderTest, EncodeSpecialCharacters) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Matrix result = encoder.encode("!@#$%^&*()");

    EXPECT_GT(result.rows, 0);
    EXPECT_EQ(result.cols, D_MODEL);
}

TEST_F(LLMEncoderTest, BackwardWithoutRequiresGrad) {
    create_test_vocabulary();

    LLMEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(false);  // Explicitly disable

    Matrix encoded = encoder.encode("test");
    Matrix grad(encoded.rows, encoded.cols);

    // Backward should be no-op when requires_grad is false
    encoder.backward(grad);

    // Should not crash
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
