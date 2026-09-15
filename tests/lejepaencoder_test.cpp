/**
 * @file lejepaencoder_test.cpp
 * @brief Tests for LeJEPAEncoder (TD-177 / LJ-2a) — construction, encode(), save/load, and
 *        plumbing methods, per docs/proposals/lejepa_world_model_gated_injection_plan.md's
 *        Component 1. `train_step()` is deliberately out of scope here (TD-178).
 */

#include "../src/LeJEPAEncoder.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"
#include "../src/encoder.hpp"

namespace fs = std::filesystem;

class LeJEPAEncoderTest : public ::testing::Test {
   protected:
    static constexpr int VOCAB_SIZE = 1000;
    static constexpr int D_MODEL = 32;  // small for faster tests
    static constexpr int NUM_LAYERS = 2;
    static constexpr int NUM_HEADS = 4;
    static constexpr int D_FF = 64;
    static constexpr int MAX_SEQ_LEN = 32;

    std::string temp_dir;
    std::string vocab_file;

    void SetUp() override {
        temp_dir = fs::temp_directory_path() / "lejepaencoder_test";
        fs::create_directories(temp_dir);
        vocab_file = temp_dir + "/vocab.txt";
    }

    void TearDown() override {
        if (fs::exists(temp_dir)) {
            fs::remove_all(temp_dir);
        }
    }

    // Same BPE vocabulary file format llmencoder_test.cpp uses, so both encoders can be built
    // against an identical vocabulary for the drop-in-shape comparison below.
    void create_test_vocabulary() {
        std::ofstream file(vocab_file);
        ASSERT_TRUE(file.is_open());

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

        const std::vector<std::string> tokens = {
            "hello", "world", "test", "encoder", "the",  "is",    "a",    "to",    "of",  "and",
            "in",    "that",  "it",   "for",     "on",   "with",  "as",   "this",  "was", "are",
            "be",    "have",  "from", "or",      "one",  "had",   "by",   "but",   "not", "what"};

        int id = 4;
        for (const auto& token : tokens) {
            file << token << "\t" << id++ << "\n";
        }

        file.close();
    }

    std::vector<std::string> create_test_corpus() {
        return {"hello world test", "encoder test case", "this is a test",
                "another test sentence"};
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(LeJEPAEncoderTest, ConstructWithDefaultParameters) {
    LeJEPAEncoder encoder(VOCAB_SIZE);

    EXPECT_EQ(encoder.get_embedding_dim(), 512);  // default d_model, matches LLMEncoder's own
}

TEST_F(LeJEPAEncoderTest, ConstructWithCustomParameters) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_EQ(encoder.get_embedding_dim(), D_MODEL);
    EXPECT_EQ(encoder.get_num_layers(), NUM_LAYERS);
}

TEST_F(LeJEPAEncoderTest, ConstructWithMinimalConfiguration) {
    LeJEPAEncoder encoder(100, 16, 1, 2, 32, 32);

    EXPECT_EQ(encoder.get_embedding_dim(), 16);
}

// ============================================================================
// Tokenizer Operations Tests
// ============================================================================

TEST_F(LeJEPAEncoderTest, LoadTokenizerVocab) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_NO_THROW(encoder.load_tokenizer_vocab(vocab_file));
}

TEST_F(LeJEPAEncoderTest, LoadTokenizerVocabNonExistentFile) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_THROW(encoder.load_tokenizer_vocab(temp_dir + "/nonexistent.txt"), std::exception);
}

TEST_F(LeJEPAEncoderTest, BuildTokenizerFromCorpus) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_NO_THROW(encoder.build_tokenizer(create_test_corpus(), 50));
}

// ============================================================================
// encode() Tests
// ============================================================================

TEST_F(LeJEPAEncoderTest, EncodeReturnsCorrectDimensions) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Matrix result = encoder.encode("hello world");

    EXPECT_GT(result.rows, 0);
    EXPECT_EQ(result.cols, D_MODEL);
    EXPECT_LE(result.rows, MAX_SEQ_LEN);
}

TEST_F(LeJEPAEncoderTest, EncodeLongTextTruncates) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    std::string long_text;
    for (int i = 0; i < 200; ++i) {
        long_text += "word ";
    }

    Matrix result = encoder.encode(long_text);

    EXPECT_LE(result.rows, MAX_SEQ_LEN);
}

TEST_F(LeJEPAEncoderTest, EncodeEmptyStringThrows) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // Same BPE-tokenizer behavior LLMEncoder::encode() inherits for empty input.
    EXPECT_THROW({ encoder.encode(""); }, std::exception);
}

TEST_F(LeJEPAEncoderTest, DifferentTextsProduceDifferentOutput) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Matrix out1 = encoder.encode("hello");
    Matrix out2 = encoder.encode("world test");

    // Different token counts alone already prove these aren't coincidentally identical, but
    // check content too when shapes happen to match.
    bool differ = (out1.rows != out2.rows);
    if (!differ) {
        for (int i = 0; i < out1.rows && !differ; ++i) {
            for (int j = 0; j < out1.cols && !differ; ++j) {
                if (std::abs(out1(i, j) - out2(i, j)) > 1e-6f) {
                    differ = true;
                }
            }
        }
    }
    EXPECT_TRUE(differ);
}

// Drop-in-shape requirement from TD-177's own Action Items: encode()'s output must be usable
// anywhere an LLMEncoder::encode() output is expected — [seq_len, d_model] for the same
// tokenizer/text, needed by TD-179's HippocampalMemory key reuse.
TEST_F(LeJEPAEncoderTest, EncodeShapeMatchesLLMEncoderForSameInput) {
    create_test_vocabulary();

    LeJEPAEncoder lejepa_encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    lejepa_encoder.load_tokenizer_vocab(vocab_file);

    LLMEncoder llm_encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    llm_encoder.load_tokenizer_vocab(vocab_file);

    Matrix lejepa_output = lejepa_encoder.encode("hello world test");
    Matrix llm_output = llm_encoder.encode("hello world test");

    // Same tokenizer/text -> same token count -> same shape, even though the two encoders'
    // weights (and therefore values) differ.
    EXPECT_EQ(lejepa_output.rows, llm_output.rows);
    EXPECT_EQ(lejepa_output.cols, llm_output.cols);
    EXPECT_EQ(lejepa_output.cols, D_MODEL);
}

// ============================================================================
// Plumbing Tests (set_requires_grad / set_learning_rate / register_parameters_with_optimizer)
// ============================================================================

TEST_F(LeJEPAEncoderTest, SetRequiresGradDoesNotCrash) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_NO_THROW(encoder.set_requires_grad(false));
    EXPECT_NO_THROW(encoder.set_requires_grad(true));
}

TEST_F(LeJEPAEncoderTest, EncodeWorksWithRequiresGradFalse) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(false);

    Matrix result = encoder.encode("hello world");

    EXPECT_EQ(result.cols, D_MODEL);
}

TEST_F(LeJEPAEncoderTest, SetLearningRateDoesNotCrash) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_NO_THROW(encoder.set_learning_rate(0.0001f));
}

TEST_F(LeJEPAEncoderTest, RegisterParametersWithOptimizer) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    Optimizer optimizer(OptimizerType::ADAMW, 0.001f);

    EXPECT_NO_THROW(encoder.register_parameters_with_optimizer(optimizer));
}

// ============================================================================
// get_encoder_block() Tests
// ============================================================================

TEST_F(LeJEPAEncoderTest, GetEncoderBlockValidLayer) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EncoderBlock* block0 = encoder.get_encoder_block(0);
    EncoderBlock* block1 = encoder.get_encoder_block(1);

    ASSERT_NE(block0, nullptr);
    ASSERT_NE(block1, nullptr);
    EXPECT_NE(block0, block1);
}

TEST_F(LeJEPAEncoderTest, GetEncoderBlockOutOfRangeThrows) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_THROW(encoder.get_encoder_block(-1), std::out_of_range);
    EXPECT_THROW(encoder.get_encoder_block(NUM_LAYERS), std::out_of_range);
}

// ============================================================================
// print_config() Test
// ============================================================================

TEST_F(LeJEPAEncoderTest, PrintConfigDoesNotCrash) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_NO_THROW(encoder.print_config());
}

// ============================================================================
// save() / load() Tests
// ============================================================================

TEST_F(LeJEPAEncoderTest, SaveCreatesDirectoryAndComponentFiles) {
    std::string save_dir = temp_dir + "/saved_encoder";
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    encoder.save(save_dir);

    EXPECT_TRUE(fs::exists(save_dir));
    EXPECT_TRUE(fs::exists(save_dir + "/config.bin"));
    EXPECT_TRUE(fs::exists(save_dir + "/token_embedding.bin"));
    EXPECT_TRUE(fs::exists(save_dir + "/encoder_block_0.bin"));
    EXPECT_TRUE(fs::exists(save_dir + "/encoder_block_1.bin"));
    EXPECT_TRUE(fs::exists(save_dir + "/final_norm.bin"));
}

TEST_F(LeJEPAEncoderTest, LoadRoundTripsWithoutError) {
    std::string save_dir = temp_dir + "/saved_encoder";

    LeJEPAEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.save(save_dir);

    LeJEPAEncoder encoder2(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    EXPECT_NO_THROW(encoder2.load(save_dir));
}

TEST_F(LeJEPAEncoderTest, LoadNonExistentDirectoryThrows) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_THROW(encoder.load(temp_dir + "/does_not_exist"), std::runtime_error);
}

TEST_F(LeJEPAEncoderTest, LoadMismatchedArchitectureThrows) {
    std::string save_dir = temp_dir + "/saved_encoder";

    LeJEPAEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.save(save_dir);

    LeJEPAEncoder encoder2(VOCAB_SIZE, D_MODEL, 4, NUM_HEADS, D_FF, MAX_SEQ_LEN);  // different
                                                                                    // num_layers

    EXPECT_THROW(encoder2.load(save_dir), std::runtime_error);
}

TEST_F(LeJEPAEncoderTest, SaveAndLoadPreservesEncodeOutput) {
    create_test_vocabulary();
    std::string save_dir = temp_dir + "/saved_encoder";

    LeJEPAEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.load_tokenizer_vocab(vocab_file);
    Matrix output_before = encoder1.encode("test");
    encoder1.save(save_dir);

    LeJEPAEncoder encoder2(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder2.load_tokenizer_vocab(vocab_file);
    encoder2.load(save_dir);
    Matrix output_after = encoder2.encode("test");

    ASSERT_EQ(output_before.rows, output_after.rows);
    ASSERT_EQ(output_before.cols, output_after.cols);
    for (int i = 0; i < output_before.rows; ++i) {
        for (int j = 0; j < output_before.cols; ++j) {
            EXPECT_NEAR(output_before(i, j), output_after(i, j), 1e-4f);
        }
    }
}
