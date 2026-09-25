/**
 * @file lejepaencoder_test.cpp
 * @brief Tests for LeJEPAEncoder — construction, encode(), save/load, and plumbing methods
 *        (TD-177 / LJ-2a), plus the self-supervised train_step() training loop (TD-178 / LJ-2b),
 *        per docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 1.
 */

#include "../src/LeJEPAEncoder.hpp"
#include <gtest/gtest.h>
#include <cmath>
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

// TD-183: sigreg_num_sketches trailing constructor parameter.
TEST_F(LeJEPAEncoderTest, DefaultSigregNumSketchesMatchesSIGRegDefault) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);

    EXPECT_EQ(encoder.get_sigreg_num_sketches(), 64);
}

TEST_F(LeJEPAEncoderTest, CustomSigregNumSketchesIsApplied) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN,
                          /*sigreg_num_sketches=*/16);

    EXPECT_EQ(encoder.get_sigreg_num_sketches(), 16);
}

TEST_F(LeJEPAEncoderTest, CustomSigregNumSketchesStillProducesFiniteTrainStepLosses) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN,
                          /*sigreg_num_sketches=*/8);
    create_test_vocabulary();
    encoder.load_tokenizer_vocab(vocab_file);

    auto [predictor_loss, sigreg_loss] = encoder.train_step("hello world test encoder");
    EXPECT_TRUE(std::isfinite(predictor_loss));
    EXPECT_TRUE(std::isfinite(sigreg_loss));
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

// ============================================================================
// train_step() Tests (TD-178 / LJ-2b)
// ============================================================================

TEST_F(LeJEPAEncoderTest, TrainStepReturnsFiniteNonNegativeLosses) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    auto [predictor_loss, sigreg_loss] = encoder.train_step("hello world this is a test");

    EXPECT_TRUE(std::isfinite(predictor_loss));
    EXPECT_TRUE(std::isfinite(sigreg_loss));
    EXPECT_GE(predictor_loss, 0.0f);
    EXPECT_GE(sigreg_loss, 0.0f);
}

TEST_F(LeJEPAEncoderTest, TrainStepHandlesShortText) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // bos + one content token + eos = 3 tokens — small enough to exercise span-length clamping
    // (span_len must never reach the full sequence; at least one context token must remain).
    EXPECT_NO_THROW(encoder.train_step("hello"));
}

TEST_F(LeJEPAEncoderTest, TrainStepUpdatesWeightsWithoutOptimizer) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_learning_rate(0.1f);

    Matrix output_before = encoder.encode("hello world test");
    encoder.train_step("hello world this is a test");
    Matrix output_after = encoder.encode("hello world test");

    bool changed = false;
    for (int i = 0; i < output_before.rows && !changed; ++i) {
        for (int j = 0; j < output_before.cols && !changed; ++j) {
            if (std::abs(output_after(i, j) - output_before(i, j)) > 1e-6f) {
                changed = true;
            }
        }
    }
    EXPECT_TRUE(changed);
}

TEST_F(LeJEPAEncoderTest, TrainStepIsNoOpForWeightsWhenFrozen) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_requires_grad(false);

    Matrix output_before = encoder.encode("hello world test");
    auto [predictor_loss, sigreg_loss] = encoder.train_step("hello world this is a test");
    Matrix output_after = encoder.encode("hello world test");

    // Losses are still computed (useful for eval-time diagnostics on a frozen encoder)...
    EXPECT_TRUE(std::isfinite(predictor_loss));
    EXPECT_TRUE(std::isfinite(sigreg_loss));

    // ...but no weight update happened, mirroring LLMEncoder::backward()'s own
    // no-op-when-!requires_grad convention.
    for (int i = 0; i < output_before.rows; ++i) {
        for (int j = 0; j < output_before.cols; ++j) {
            EXPECT_FLOAT_EQ(output_after(i, j), output_before(i, j));
        }
    }

    // No weight update also means no gradient norm to report.
    EXPECT_FLOAT_EQ(encoder.get_last_gradient_norm(), 0.0f);
}

// Regression test for a real, already-shipping bug found during the advanced-metrics work:
// LeJEPAEncoder::update_weights() zeroes every gradient buffer internally, and train_step() calls
// it TWICE before ever returning — so calling optimizer.get_gradient_norm() from OUTSIDE
// train_step() after it returns always reads 0.0f. get_last_gradient_norm() captures the norm
// from INSIDE train_step(), before each internal update_weights() call zeroes it.
TEST_F(LeJEPAEncoderTest, TrainStepSetsPositiveGradientNormWithRegisteredOptimizer) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    encoder.register_parameters_with_optimizer(optimizer);

    encoder.train_step("hello world this is a test");

    // The bug this test guards against: reading optimizer.get_gradient_norm() here (outside
    // train_step()) would read 0.0f, since both of train_step()'s internal update_weights() calls
    // already zeroed it by the time control returns.
    EXPECT_GT(encoder.get_last_gradient_norm(), 0.0f);
}

TEST_F(LeJEPAEncoderTest, TrainStepGradientNormIsZeroWithoutRegisteredOptimizer) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    encoder.set_learning_rate(0.1f);

    // No register_parameters_with_optimizer() call — plain-SGD fallback path, which has no single
    // combined-norm primitive across every sub-component (see get_last_gradient_norm()'s own doc
    // comment for why this scope is intentionally narrow).
    encoder.train_step("hello world this is a test");

    EXPECT_FLOAT_EQ(encoder.get_last_gradient_norm(), 0.0f);
}

TEST_F(LeJEPAEncoderTest, TrainStepMaskingRatioIsWithinValidRangeAndTracksNominal) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    // A long-enough sentence that the nominal 25% span ratio isn't distorted by the
    // at-least-one-context-token clamp short sequences hit.
    encoder.train_step("the quick brown fox jumps over the lazy dog again and again today");

    const float ratio = encoder.get_last_masking_ratio();
    EXPECT_GT(ratio, 0.0f);
    EXPECT_LE(ratio, 1.0f);
    EXPECT_NEAR(ratio, 0.25f, 0.15f) << "should roughly track the nominal 25% mask ratio for a "
                                        "sequence long enough to avoid the short-sequence clamp";
}

TEST_F(LeJEPAEncoderTest, TrainStepCosineSimilarityIsBounded) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    encoder.train_step("hello world this is a test");

    const float cos_sim = encoder.get_last_predictor_target_cosine_sim();
    EXPECT_TRUE(std::isfinite(cos_sim));
    EXPECT_GE(cos_sim, -1.0f);
    EXPECT_LE(cos_sim, 1.0f);
}

TEST_F(LeJEPAEncoderTest, TrainStepSigregVarianceStatsAreFiniteAndStddevNonNegative) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    encoder.train_step("hello world this is a test");

    EXPECT_TRUE(std::isfinite(encoder.get_last_sigreg_variance_mean()));
    EXPECT_TRUE(std::isfinite(encoder.get_last_sigreg_variance_stddev()));
    EXPECT_GE(encoder.get_last_sigreg_variance_stddev(), 0.0f);
}

// TD-178's own Action Items: both loss terms must trend downward on a small synthetic corpus.
// Registers a real Optimizer (ADAM) rather than using the plain-SGD fallback — this doubles as
// a regression guard for a real bug found and fixed during this item's own implementation
// (LeJEPAEncoder::update_weights() calling optimizer_->step() exactly once per train_step, not
// once per sub-component sharing the same registered Optimizer): had that bug been present, the
// shared Optimizer would have been over-stepped once per sub-component (5x for this fixture's
// 2-layer configuration: token_embedding, 2 encoder blocks, final_norm, predictor), a
// corruption severe enough that a healthy downward trend across dozens of iterations would be
// very unlikely to survive it.
TEST_F(LeJEPAEncoderTest, TrainStepLossesTrendDownwardOnSyntheticCorpus) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Optimizer optimizer(OptimizerType::ADAM, 0.02f);
    encoder.register_parameters_with_optimizer(optimizer);
    // Strengthens SIGReg's gradient contribution relative to the predictor's own for this test:
    // sigreg_loss is estimated from just this one call's own per-token embeddings (a handful of
    // rows), a far noisier "batch" than SIGReg's own dedicated tests use (thousands of rows) —
    // giving it more relative weight here makes its downward trend detectable against that
    // noise floor within a practical number of test iterations.
    encoder.set_sigreg_lambda(3.0f);

    // Longer sentences than a minimal smoke test would need — more tokens per call means a
    // larger, less noisy "batch" for SIGReg's own per-call loss estimate.
    const std::vector<std::string> corpus = {
        "hello world the is a to of and in that it for on with",
        "as this was are be have from or one had by but not what",
        "hello the world is a to of and in that it for on",
        "with as this was are be have from or one had by but"};

    const int total_iters = 120;
    std::vector<float> predictor_losses;
    std::vector<float> sigreg_losses;
    predictor_losses.reserve(total_iters);
    sigreg_losses.reserve(total_iters);

    for (int i = 0; i < total_iters; ++i) {
        auto [predictor_loss, sigreg_loss] = encoder.train_step(corpus[i % corpus.size()]);
        ASSERT_TRUE(std::isfinite(predictor_loss)) << "iteration " << i;
        ASSERT_TRUE(std::isfinite(sigreg_loss)) << "iteration " << i;
        predictor_losses.push_back(predictor_loss);
        sigreg_losses.push_back(sigreg_loss);
    }

    // Compare the first half's average against the second half's, not just a handful of samples
    // at each end — each call's own span placement is randomized (a fresh std::random_device
    // seed per train_step(), matching this session's own established convention for SIGReg's/
    // Predictor's non-deterministic weight init), so averaging over many more samples per group
    // is what keeps this comparison meaningful against that per-call noise rather than a couple
    // of unlucky samples at either boundary.
    const int half = total_iters / 2;
    auto average = [](const std::vector<float>& v, int start, int count) {
        float sum = 0.0f;
        for (int i = start; i < start + count; ++i) {
            sum += v[i];
        }
        return sum / static_cast<float>(count);
    };

    float predictor_first = average(predictor_losses, 0, half);
    float predictor_last = average(predictor_losses, half, half);
    float sigreg_first = average(sigreg_losses, 0, half);
    float sigreg_last = average(sigreg_losses, half, half);

    EXPECT_LT(predictor_last, predictor_first)
        << "predictor_loss should trend downward (first-half avg=" << predictor_first
        << ", second-half avg=" << predictor_last << ")";
    EXPECT_LT(sigreg_last, sigreg_first)
        << "sigreg_loss should trend downward (first-half avg=" << sigreg_first
        << ", second-half avg=" << sigreg_last << ")";
}

namespace {
float matrix_delta_l2(const Matrix& before, const Matrix& after) {
    float sum_sq = 0.0f;
    for (int i = 0; i < before.rows; ++i) {
        for (int j = 0; j < before.cols; ++j) {
            const float d = after(i, j) - before(i, j);
            sum_sq += d * d;
        }
    }
    return std::sqrt(sum_sq);
}
}  // namespace

// TD-189 regression test: LayerNorm::backward()/FeedForward::backward()/
// MultiHeadAttention::backward() overwrite (not accumulate) their own gradient members on every
// call. train_step() used to call backward() twice — once for the target view (predictor's
// target-side term + SIGReg), once for the context view (predictor's context-side term) — before
// a single update_weights(), silently discarding whichever gradient the FIRST call computed for
// every encoder_blocks/final_norm weight (only TokenEmbedding::backward() genuinely accumulates).
// In practice this meant sigreg_lambda had NO effect whatsoever on encoder_blocks/final_norm
// weights, regardless of its magnitude — only the predictor's own (context-view) gradient ever
// reached them. This test checks that cranking sigreg_lambda up visibly increases how much
// encoder_blocks[0]'s own FeedForward weight (W1) moves in a single train_step() call — under the
// bug, a huge sigreg_lambda moved that weight no more than sigreg_lambda=0 did, since SIGReg's
// gradient never survived to be applied there.
TEST_F(LeJEPAEncoderTest, SigregLambdaAffectsEncoderBlockWeightsNotJustTokenEmbedding) {
    create_test_vocabulary();

    const int trials = 8;
    float sum_delta_low = 0.0f;
    float sum_delta_high = 0.0f;

    for (int t = 0; t < trials; ++t) {
        LeJEPAEncoder encoder_low(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
        encoder_low.load_tokenizer_vocab(vocab_file);
        encoder_low.set_learning_rate(0.5f);
        encoder_low.set_sigreg_lambda(0.0f);
        Matrix w1_before_low = encoder_low.get_encoder_block(0)->get_feed_forward()->get_W1();
        encoder_low.train_step("hello world this is a test");
        Matrix w1_after_low = encoder_low.get_encoder_block(0)->get_feed_forward()->get_W1();
        sum_delta_low += matrix_delta_l2(w1_before_low, w1_after_low);

        LeJEPAEncoder encoder_high(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
        encoder_high.load_tokenizer_vocab(vocab_file);
        encoder_high.set_learning_rate(0.5f);
        encoder_high.set_sigreg_lambda(50.0f);
        Matrix w1_before_high = encoder_high.get_encoder_block(0)->get_feed_forward()->get_W1();
        encoder_high.train_step("hello world this is a test");
        Matrix w1_after_high = encoder_high.get_encoder_block(0)->get_feed_forward()->get_W1();
        sum_delta_high += matrix_delta_l2(w1_before_high, w1_after_high);
    }

    EXPECT_GT(sum_delta_high, 2.0f * sum_delta_low)
        << "sigreg_lambda should visibly scale how much encoder_blocks[0]'s own FeedForward "
           "weights move (sum over "
        << trials << " trials: lambda=0 -> " << sum_delta_low << ", lambda=50 -> " << sum_delta_high
        << ") — if it doesn't, the target-view/SIGReg gradient isn't reaching them";
}
