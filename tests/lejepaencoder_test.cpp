/**
 * @file lejepaencoder_test.cpp
 * @brief Tests for LeJEPAEncoder — construction, encode(), save/load, and plumbing methods
 *        (TD-177 / LJ-2a), plus the self-supervised train_step() training loop (TD-178 / LJ-2b),
 *        per docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 1.
 */

#include "../src/LeJEPAEncoder.hpp"
#include "../src/SIGReg.hpp"
#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>
#include <algorithm>
#include <cmath>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../src/BPETokenizer.hpp"
#include "../src/Logger.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"
#include "../src/encoder.hpp"

namespace fs = std::filesystem;

static float max_abs_diff(const Matrix& a, const Matrix& b) {
    float worst = 0.0f;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
    return worst;
}

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
    // LeJEPAEncoder::encode() wraps the tokens in BOS/EOS; LLMEncoder::encode() does so only when
    // asked since TD-364 (its default now matches EncoderDecoderModel's encoder input).
    Matrix llm_output = llm_encoder.encode("hello world test", /*add_special_tokens=*/true);

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

// TD-427: save() used to skip the predictor, so every resumed pass started with a random one.
TEST_F(LeJEPAEncoderTest, SaveAndLoadRestoreThePredictor) {
    std::string save_dir = temp_dir + "/saved_encoder";
    LeJEPAEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    LeJEPAEncoder encoder2(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    Matrix probe(2, D_MODEL);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < D_MODEL; ++j)
            probe(i, j) = 0.1f * std::cos(static_cast<float>(i * D_MODEL + j));

    const Matrix expected = encoder1.get_predictor()->forward(probe);
    ASSERT_GT(max_abs_diff(expected, encoder2.get_predictor()->forward(probe)), 0.0f);

    encoder1.save(save_dir);
    EXPECT_TRUE(fs::exists(save_dir + "/predictor.bin"));
    encoder2.load(save_dir);
    EXPECT_EQ(max_abs_diff(expected, encoder2.get_predictor()->forward(probe)), 0.0f);
}

// A checkpoint saved before TD-427 has no predictor.bin: it still loads, with a fresh predictor.
TEST_F(LeJEPAEncoderTest, LoadOfACheckpointWithoutAPredictorStillLoads) {
    std::string save_dir = temp_dir + "/saved_encoder";
    LeJEPAEncoder encoder1(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder1.save(save_dir);
    fs::remove(save_dir + "/predictor.bin");

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

namespace {
struct LogCapture {
    std::ostringstream out;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink =
        std::make_shared<spdlog::sinks::ostream_sink_mt>(out);
    LogCapture() {
        adai::Logger::get()->sinks().push_back(sink);
    }
    ~LogCapture() {
        auto& sinks = adai::Logger::get()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
    }
};
}  // namespace

// TD-429: the config's reads weren't checked, so a short config.bin left fields 0 and failed as a
// misleading "architecture mismatch". Short and over-long configs throw, naming the problem, and
// leave the encoder unchanged.
TEST_F(LeJEPAEncoderTest, ShortOrOverLongConfigThrowsAndLeavesTheEncoderUnchanged) {
    create_test_vocabulary();
    const std::string save_dir = temp_dir + "/config_check";
    LeJEPAEncoder saved(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    saved.save(save_dir);

    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    const Matrix before = encoder.encode("hello world");

    const std::string config = save_dir + "/config.bin";
    fs::resize_file(config, 5 * sizeof(int));
    try {
        encoder.load(save_dir);
        ADD_FAILURE() << "a short config loaded";
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(std::string(e.what()).find("architecture mismatch"), std::string::npos)
            << e.what();
    }
    {
        saved.save(save_dir);
        std::ofstream append(config, std::ios::binary | std::ios::app);
        append.put('x');
    }
    EXPECT_THROW(encoder.load(save_dir), std::runtime_error);
    EXPECT_EQ(max_abs_diff(encoder.encode("hello world"), before), 0.0f);
}

// TD-429: max_seq_length was read but never compared. No weight depends on it, so a different one
// loads with a warning and gives the same output; the same one loads silently.
TEST_F(LeJEPAEncoderTest, DifferentMaxSeqLengthLoadsWithAWarning) {
    create_test_vocabulary();
    const std::string save_dir = temp_dir + "/seq_len";
    LeJEPAEncoder saved(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, /*max_seq_length=*/16);
    saved.load_tokenizer_vocab(vocab_file);
    saved.save(save_dir);
    {
        LogCapture log;
        LeJEPAEncoder same(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, 16);
        same.load(save_dir);
        EXPECT_EQ(log.out.str().find("were saved with max_seq_length"), std::string::npos);
    }
    LogCapture log;
    LeJEPAEncoder longer(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, /*max_seq_length=*/32);
    longer.load_tokenizer_vocab(vocab_file);
    ASSERT_NO_THROW(longer.load(save_dir));
    EXPECT_NE(log.out.str().find("were saved with max_seq_length 16; loading them with "
                                 "max_seq_length 32"),
              std::string::npos)
        << log.out.str();
    EXPECT_EQ(max_abs_diff(saved.encode("hello world"), longer.encode("hello world")), 0.0f);
}

// TD-429: the constructor, save/load, the tokenizer methods and print_config() wrote to stdout.
TEST_F(LeJEPAEncoderTest, WritesNothingToStdout) {
    create_test_vocabulary();
    const std::string save_dir = temp_dir + "/quiet";
    LogCapture log;
    testing::internal::CaptureStdout();
    {
        LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
        encoder.load_tokenizer_vocab(vocab_file);
        encoder.print_config();
        encoder.save(save_dir);
        encoder.load(save_dir);
    }
    const std::string out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(out.find("LeJEPA"), std::string::npos) << out;
    const std::string logged = log.out.str();
    EXPECT_NE(logged.find("LeJEPAEncoder configuration: vocab 1000, d_model 32, 2 encoder layers"),
              std::string::npos)
        << logged;
    EXPECT_NE(logged.find("Saved LeJEPA encoder weights to " + save_dir), std::string::npos)
        << logged;
    EXPECT_NE(logged.find("Loaded LeJEPA encoder weights from " + save_dir), std::string::npos)
        << logged;
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
// it before returning — so calling optimizer.get_gradient_norm() from OUTSIDE
// train_step() after it returns always reads 0.0f. get_last_gradient_norm() captures the norm
// from INSIDE train_step(), before the internal update_weights() call zeroes it.
TEST_F(LeJEPAEncoderTest, TrainStepSetsPositiveGradientNormWithRegisteredOptimizer) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);

    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    encoder.register_parameters_with_optimizer(optimizer);

    encoder.train_step("hello world this is a test");

    // The bug this test guards against: reading optimizer.get_gradient_norm() here (outside
    // train_step()) would read 0.0f, since train_step()'s internal update_weights() call
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
// reached them. (TD-195 made those backward()s accumulate, so train_step() does this again
// correctly since TD-428.) This test checks that cranking sigreg_lambda up visibly increases how much
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

// TD-428: both views' gradients go into one update. Adam's first update moves every weight by at
// most the learning rate, and the old pair of updates moved them by up to twice that.
TEST_F(LeJEPAEncoderTest, TrainStepTakesOneOptimizerStep) {
    create_test_vocabulary();
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.load_tokenizer_vocab(vocab_file);
    const float lr = 0.01f;
    Optimizer optimizer(OptimizerType::ADAM, lr);
    encoder.register_parameters_with_optimizer(optimizer);

    const Matrix before = encoder.get_encoder_block(0)->get_feed_forward()->get_W1();
    encoder.train_step("hello world this is a test");
    const Matrix after = encoder.get_encoder_block(0)->get_feed_forward()->get_W1();

    float largest = 0.0f;
    for (int i = 0; i < before.rows; ++i) {
        for (int j = 0; j < before.cols; ++j) {
            largest = std::max(largest, std::abs(after(i, j) - before(i, j)));
        }
    }
    EXPECT_GT(largest, 0.5f * lr);
    EXPECT_LE(largest, 1.02f * lr) << "more than one Adam step";
}

// TD-428: the one update is the gradient of the whole objective, including the context view's path
// through the encoder (which needs its activations back in the cache). A plain SGD step gives each
// weight's gradient; compare it with finite differences of the loss on the same masked span.
TEST_F(LeJEPAEncoderTest, TrainStepUpdateMatchesFiniteDifferences) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.set_sigreg_lambda(0.0f);  // the returned loss is then the whole objective
    // Built from the text: with unknown tokens the masked context view would equal the target's.
    const std::string text = "the quick brown fox jumps over the lazy dog";
    encoder.build_tokenizer({text}, 64);

    // A fresh predictor outputs almost nothing, so the context view's gradient would be too small
    // to check. Some training gives it weight.
    Optimizer warmup(OptimizerType::ADAM, 0.01f);
    encoder.register_parameters_with_optimizer(warmup);
    for (int i = 0; i < 40; ++i) {
        encoder.train_step(text);
    }

    const float lr = 1.0f;
    Optimizer optimizer(OptimizerType::SGD, lr);
    encoder.register_parameters_with_optimizer(optimizer);
    FeedForward* ff = encoder.get_encoder_block(0)->get_feed_forward();
    Matrix w1 = ff->get_W1();
    const auto loss_now = [&] {
        encoder.seed_span_rng(5);
        encoder.set_requires_grad(false);
        const float loss = encoder.train_step(text).first;
        encoder.set_requires_grad(true);
        return loss;
    };

    Matrix numeric(w1.rows, w1.cols);
    const float eps = 2e-2f;
    for (int i = 0; i < w1.rows; i += 3) {
        for (int j = 0; j < w1.cols; j += 7) {
            const float saved = w1(i, j);
            w1(i, j) = saved + eps;
            ff->set_W1(w1);
            const float plus = loss_now();
            w1(i, j) = saved - eps;
            ff->set_W1(w1);
            const float minus = loss_now();
            w1(i, j) = saved;
            ff->set_W1(w1);
            numeric(i, j) = (plus - minus) / (2.0f * eps);
        }
    }

    encoder.seed_span_rng(5);
    encoder.train_step(text);
    const Matrix& after = ff->get_W1();

    double error_sq = 0.0, numeric_sq = 0.0;
    for (int i = 0; i < w1.rows; i += 3) {
        for (int j = 0; j < w1.cols; j += 7) {
            const double analytic = (w1(i, j) - after(i, j)) / lr;
            error_sq += (analytic - numeric(i, j)) * (analytic - numeric(i, j));
            numeric_sq += numeric(i, j) * numeric(i, j);
        }
    }
    ASSERT_GT(numeric_sq, 0.0);
    // Measured: 0.001, and about 0.8 without the context view's forward before its backward.
    EXPECT_LT(std::sqrt(error_sq / numeric_sq), 0.05);
}

// ============================================================================
// train_batch(): SIGReg across the batch's pooled embeddings (TD-503's second item)
// ============================================================================

namespace {

const std::vector<std::string> kBatchTexts = {
    "the cat sat on the mat",       "a dog ran far away home",    "birds sing in the morning",
    "rain falls on the old roof",   "she reads a long book",      "we walk to the river",
    "the sun is warm today",        "he plays a quiet song",      "fish swim in the lake",
    "my friend cooks dinner",       "the wind moves the trees",   "children laugh at the park",
    "a train leaves the station",   "the moon rises slowly",      "they paint the blue door",
    "snow covers the hills"};

// How far the texts' mean-pooled embeddings are from N(0, I): SIGReg averaged over many fresh
// direction draws from a seeded probe, so it compares two encoders fairly.
float pooled_gaussian_gap(LeJEPAEncoder& encoder, int d_model) {
    Matrix pooled(static_cast<int>(kBatchTexts.size()), d_model);
    for (size_t b = 0; b < kBatchTexts.size(); ++b) {
        Matrix e = encoder.encode(kBatchTexts[b]);
        for (int i = 0; i < e.rows; ++i)
            for (int j = 0; j < d_model; ++j)
                pooled(static_cast<int>(b), j) += e(i, j) / static_cast<float>(e.rows);
    }
    SIGReg probe(d_model, 16);
    probe.set_seed(7);
    float sum = 0.0f;
    for (int k = 0; k < 40; ++k) {
        probe.resample_directions();
        sum += probe.compute_loss(pooled);
    }
    return sum / 40.0f;
}

}  // namespace

TEST_F(LeJEPAEncoderTest, TrainBatchSkipsUnusableTextsAndReportsTheTrainedOnes) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.build_tokenizer(kBatchTexts, VOCAB_SIZE);
    auto result = encoder.train_batch({kBatchTexts[0], "", kBatchTexts[1], kBatchTexts[2]});
    EXPECT_EQ(result.trained, (std::vector<size_t>{0, 2, 3}));
    ASSERT_EQ(result.predictor_losses.size(), 3u);
    for (float loss : result.predictor_losses)
        EXPECT_TRUE(std::isfinite(loss));
    EXPECT_TRUE(std::isfinite(result.sigreg_loss));
}

// TD-429: invalid UTF-8 throws TokenizerEncodingError, a runtime_error rather than the
// invalid_argument the pass once caught, so one bad line in a book ended the whole pass. A batch
// skips that text and trains the rest.
TEST_F(LeJEPAEncoderTest, TrainBatchSkipsInvalidUtf8) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.build_tokenizer(kBatchTexts, VOCAB_SIZE);
    const std::string invalid = "caf\xc3 \xff\xfe bad bytes";
    BPETokenizer probe;
    probe.build_vocab(kBatchTexts, VOCAB_SIZE);
    ASSERT_THROW(probe.encode(invalid, true), TokenizerEncodingError);

    LeJEPAEncoder::BatchStepResult result;
    ASSERT_NO_THROW(result = encoder.train_batch({kBatchTexts[0], invalid, kBatchTexts[1]}));
    EXPECT_EQ(result.trained, (std::vector<size_t>{0, 2}));
    for (float loss : result.predictor_losses)
        EXPECT_TRUE(std::isfinite(loss));
}

TEST_F(LeJEPAEncoderTest, TrainBatchOfOneHasNoSigregLoss) {
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.build_tokenizer(kBatchTexts, VOCAB_SIZE);
    auto result = encoder.train_batch({kBatchTexts[0]});
    ASSERT_EQ(result.trained.size(), 1u);
    EXPECT_EQ(result.sigreg_loss, 0.0f);  // SIGReg needs at least 2 pooled embeddings
}

// The batch SIGReg term does what it's for: two encoders from the same weights train on the same
// batches, one with the SIGReg term (default lambda) and one without; the one with it keeps its
// pooled embeddings much closer to N(0, I).
TEST_F(LeJEPAEncoderTest, TrainBatchSigregPullsPooledEmbeddingsTowardGaussian) {
    const std::string save_dir = temp_dir + "/batch_start";
    LeJEPAEncoder with_sigreg(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    with_sigreg.build_tokenizer(kBatchTexts, VOCAB_SIZE);
    with_sigreg.save(save_dir);
    LeJEPAEncoder without(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    without.build_tokenizer(kBatchTexts, VOCAB_SIZE);
    without.load(save_dir);

    with_sigreg.set_sigreg_lambda(1.0f);  // the default
    without.set_sigreg_lambda(0.0f);
    with_sigreg.set_learning_rate(0.01f);
    without.set_learning_rate(0.01f);
    const float start = pooled_gaussian_gap(with_sigreg, D_MODEL);
    for (int step = 0; step < 30; ++step) {
        with_sigreg.train_batch(kBatchTexts);
        without.train_batch(kBatchTexts);
    }
    const float gap_with = pooled_gaussian_gap(with_sigreg, D_MODEL);
    const float gap_without = pooled_gaussian_gap(without, D_MODEL);
    // Without the SIGReg term the predictor pulls the pooled embeddings away from N(0, I); with
    // it they stay closer. Both encoders share their start, so compare their end values: a ratio
    // of drifts from the unseeded start failed about 1 run in 10 once TD-371 shrank the initial
    // weights. Measured gap_without - gap_with: lambda 1, mean 0.185, lowest 0.093 in 913 runs;
    // lambda 0.1, at most 0.047 in 100; lambda 0, at most 0.024 in 200.
    EXPECT_GT(gap_without - gap_with, 0.06f)
        << "start " << start << ", with " << gap_with << ", without " << gap_without;
}


// TD-548: <session_dir>/world_model is also the standalone `world_model` dataset pool, so the
// default checkpoint moved to <session_dir>/world_model_checkpoint, and "has a checkpoint" means
// its config file, not just the directory.
TEST_F(LeJEPAEncoderTest, CheckpointDirectoryResolution) {
    const std::string session = temp_dir + "/session";
    const std::string current = session + "/world_model_checkpoint";
    const std::string legacy = session + "/world_model";
    EXPECT_EQ(LeJEPAEncoder::default_checkpoint_dir(session), current);

    // Nothing saved yet: the new default (and no checkpoint there).
    EXPECT_EQ(LeJEPAEncoder::resolve_checkpoint_dir(session, ""), current);
    EXPECT_FALSE(LeJEPAEncoder::has_checkpoint(current));

    // A dataset pool in the old directory is not a checkpoint.
    fs::create_directories(legacy);
    std::ofstream(legacy + "/pending_files.txt") << "x\n";
    EXPECT_FALSE(LeJEPAEncoder::has_checkpoint(legacy));
    EXPECT_EQ(LeJEPAEncoder::resolve_checkpoint_dir(session, ""), current);

    // A checkpoint saved before TD-548 is still found...
    LeJEPAEncoder encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN);
    encoder.save(legacy);
    EXPECT_TRUE(LeJEPAEncoder::has_checkpoint(legacy));
    EXPECT_EQ(LeJEPAEncoder::resolve_checkpoint_dir(session, ""), legacy);
    // ...until one exists in the new place.
    encoder.save(current);
    EXPECT_EQ(LeJEPAEncoder::resolve_checkpoint_dir(session, ""), current);

    // An artifact path (MNS or WORLD_MODEL_ARTIFACT_PATH) always wins.
    EXPECT_EQ(LeJEPAEncoder::resolve_checkpoint_dir(session, "/elsewhere/wm"), "/elsewhere/wm");
}

// TD-416: the LeJEPA tokenizer was always ASCII; its mode is now set like the chatbot's, and a vocabulary
// that records its own mode wins. TD-564: a vocabulary without a TOKENIZER_MODE line loads as ASCII
// whatever the configured mode: the line was added with Unicode mode itself (June 21, 2026), so such a
// file was built from bytes. Unicode is now the default.
TEST(LeJEPAEncoderTokenizerMode, FollowsTheConfiguredModeUnlessTheVocabularyRecordsOne) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("adai_td416_vocab_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    BPETokenizer source(TokenizerMode::ASCII);
    source.build_vocab({"hello world", "how are you"}, 40);
    source.save_vocab((dir / "with_mode.txt").string());
    {
        std::ifstream in(dir / "with_mode.txt");
        std::ofstream out(dir / "no_mode.txt");
        for (std::string line; std::getline(in, line);)
            if (line.rfind("TOKENIZER_MODE", 0) != 0) out << line << '\n';
    }
    const int vocab = static_cast<int>(source.get_vocab_size());

    LeJEPAEncoder headerless(vocab, 64, 1, 2, 128, 32, 16);
    headerless.set_tokenizer_mode(TokenizerMode::UNICODE);
    headerless.load_tokenizer_vocab((dir / "no_mode.txt").string());
    EXPECT_EQ(headerless.get_tokenizer_mode(), TokenizerMode::ASCII);  // TD-564

    LeJEPAEncoder recorded(vocab, 64, 1, 2, 128, 32, 16);
    recorded.set_tokenizer_mode(TokenizerMode::UNICODE);
    recorded.load_tokenizer_vocab((dir / "with_mode.txt").string());
    EXPECT_EQ(recorded.get_tokenizer_mode(), TokenizerMode::ASCII);  // the file's own mode

    LeJEPAEncoder defaulted(vocab, 64, 1, 2, 128, 32, 16);
    EXPECT_EQ(defaulted.get_tokenizer_mode(), TokenizerMode::UNICODE);  // TD-564: the default
    fs::remove_all(dir);
}

// TD-589: prepared texts train exactly like train_batch() on the same batches, and are tokenized once for
// every epoch instead of once per epoch.
TEST_F(LeJEPAEncoderTest, PreparedTextsTrainLikeTrainBatchAndTokenizeOnce) {
    const std::vector<std::string> texts = {"hello world test", "", "encoder test case", "this is a test",
                                            "another test sentence"};
    const std::string dir = temp_dir + "/prepared";
    LeJEPAEncoder a(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    a.build_tokenizer(create_test_corpus(), 50);
    a.save(dir);
    LeJEPAEncoder b(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    b.build_tokenizer(create_test_corpus(), 50);
    b.load(dir);
    for (LeJEPAEncoder* e : {&a, &b}) {
        e->seed_span_rng(3);
        e->seed_sigreg(4);
    }
    Optimizer oa(OptimizerType::ADAM, 0.001f), ob(OptimizerType::ADAM, 0.001f);
    a.register_parameters_with_optimizer(oa);
    b.register_parameters_with_optimizer(ob);
    b.prepare_texts(texts);
    for (int epoch = 0; epoch < 2; ++epoch) {
        for (size_t first = 0; first < texts.size(); first += 3) {
            const size_t last = std::min(texts.size(), first + 3);
            const auto ra = a.train_batch(std::vector<std::string>(texts.begin() + first, texts.begin() + last));
            const auto rb = b.train_prepared(first, last);
            EXPECT_EQ(rb.trained, ra.trained);
            EXPECT_EQ(rb.predictor_losses, ra.predictor_losses);
            EXPECT_EQ(rb.sigreg_loss, ra.sigreg_loss);
        }
    }
    EXPECT_EQ(b.texts_tokenized(), texts.size()) << "once for both epochs";
    EXPECT_EQ(a.texts_tokenized(), 2 * texts.size());
}

#ifdef ADAI_ENABLE_GPU
// TD-588: training on the device against training on the host, on the oneAPI CPU device in a SYCL build
// (ADAI_SYCL_ALLOW_CPU_DEVICE=1); skipped without a device.
#include "../src/gpu/GPUUtils.hpp"

namespace {
bool lejepa_gpu_ready() {
    return adai::gpu::GPUManager::probe() && adai::gpu::GPUManager::initialize(0, 0.5f);
}
const std::vector<std::string> kGpuCorpus = {
    "the cat sat on the mat and looked at the door", "a dog ran far away from home in the rain",
    "birds sing in the morning when the sun comes up", "the river runs past the old mill by the hill",
    "she read the letter twice before she put it down", "tomorrow the market opens early for the fair"};
}  // namespace

// Same weights, vocabulary, span seed and SIGReg seed: two batches on the device give the host's losses,
// diagnostics and weights, with every optimizer group on the device.
TEST_F(LeJEPAEncoderTest, DeviceTrainingMatchesHostTraining) {
    if (!lejepa_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const std::string dir = temp_dir + "/gpu_parity";
    LeJEPAEncoder host(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    host.build_tokenizer(kGpuCorpus, 120);
    host.save(dir);
    LeJEPAEncoder device(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    device.build_tokenizer(kGpuCorpus, 120);
    device.load(dir);
    for (LeJEPAEncoder* e : {&host, &device}) {
        e->seed_span_rng(11);
        e->seed_sigreg(12);
    }
    Optimizer host_opt(OptimizerType::ADAM, 0.001f);
    host.register_parameters_with_optimizer(host_opt);
    Optimizer device_opt(OptimizerType::ADAM, 0.001f);
    device.register_parameters_with_optimizer(device_opt);
    EXPECT_EQ(device.gpu_begin_training(), 0u) << "every optimizer group has a device copy";
    ASSERT_TRUE(device.gpu_training());

    const std::vector<std::vector<std::string>> batches = {{kGpuCorpus[0], kGpuCorpus[1], kGpuCorpus[2], ""},
                                                           {kGpuCorpus[3], kGpuCorpus[4], kGpuCorpus[5]}};
    for (size_t i = 0; i < batches.size(); ++i) {
        SCOPED_TRACE("batch " + std::to_string(i));
        const auto h = host.train_batch(batches[i]);
        const auto d = device.train_batch(batches[i]);
        EXPECT_EQ(d.trained, h.trained) << "the same texts are skipped (the empty one)";
        ASSERT_EQ(d.predictor_losses.size(), h.predictor_losses.size());
        for (size_t b = 0; b < h.predictor_losses.size(); ++b)
            EXPECT_NEAR(d.predictor_losses[b], h.predictor_losses[b], 2e-3f * std::max(1.0f, h.predictor_losses[b]));
        EXPECT_NEAR(d.sigreg_loss, h.sigreg_loss, 2e-3f * std::max(1.0f, std::abs(h.sigreg_loss)));
        EXPECT_FLOAT_EQ(device.get_last_masking_ratio(), host.get_last_masking_ratio()) << "the same spans";
        EXPECT_NEAR(device.get_last_predictor_target_cosine_sim(), host.get_last_predictor_target_cosine_sim(), 2e-3f);
        EXPECT_NEAR(device.get_last_gradient_norm(), host.get_last_gradient_norm(),
                    5e-3f * std::max(1.0f, host.get_last_gradient_norm()));
        EXPECT_NEAR(device.get_last_sigreg_variance_mean(), host.get_last_sigreg_variance_mean(),
                    2e-3f * std::max(1.0f, host.get_last_sigreg_variance_mean()));
    }
    // The device weights are the live copy until synced; then both encoders encode the same.
    device.sync_host_from_device();
    const Matrix want = host.encode(kGpuCorpus[1]);
    const Matrix got = device.encode(kGpuCorpus[1]);
    EXPECT_LT(max_abs_diff(got, want), 5e-3f);
    // And gpu_encode() uses the trained device weights, not a stale upload.
    EXPECT_LT(max_abs_diff(Matrix::from_gpu(device.gpu_encode(kGpuCorpus[1])), want), 5e-3f);
}

// A checkpoint written after syncing restores the device-trained weights.
TEST_F(LeJEPAEncoderTest, DeviceTrainedWeightsSurviveSaveAfterSync) {
    if (!lejepa_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const std::string dir = temp_dir + "/gpu_checkpoint";
    LeJEPAEncoder device(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    device.build_tokenizer(kGpuCorpus, 120);
    Optimizer opt(OptimizerType::ADAM, 0.01f);
    device.register_parameters_with_optimizer(opt);
    const Matrix before = device.encode(kGpuCorpus[0]);
    device.gpu_begin_training();
    device.train_batch({kGpuCorpus[0], kGpuCorpus[1], kGpuCorpus[2]});
    EXPECT_EQ(max_abs_diff(device.encode(kGpuCorpus[0]), before), 0.0f) << "the host copy is stale until synced";
    device.sync_host_from_device();
    const Matrix trained = device.encode(kGpuCorpus[0]);
    EXPECT_GT(max_abs_diff(trained, before), 0.0f);
    device.save(dir);
    LeJEPAEncoder restored(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    restored.build_tokenizer(kGpuCorpus, 120);
    restored.load(dir);
    EXPECT_EQ(max_abs_diff(restored.encode(kGpuCorpus[0]), trained), 0.0f);
}

// TD-589: on the device, prepared texts (uploaded once) train like the host's train_batch().
TEST_F(LeJEPAEncoderTest, PreparedTextsOnTheDeviceMatchTheHost) {
    if (!lejepa_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const std::string dir = temp_dir + "/gpu_prepared";
    LeJEPAEncoder host(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    host.build_tokenizer(kGpuCorpus, 120);
    host.save(dir);
    LeJEPAEncoder device(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    device.build_tokenizer(kGpuCorpus, 120);
    device.load(dir);
    for (LeJEPAEncoder* e : {&host, &device}) {
        e->seed_span_rng(21);
        e->seed_sigreg(22);
    }
    Optimizer host_opt(OptimizerType::ADAM, 0.001f), device_opt(OptimizerType::ADAM, 0.001f);
    host.register_parameters_with_optimizer(host_opt);
    device.register_parameters_with_optimizer(device_opt);
    device.gpu_begin_training();
    device.prepare_texts(kGpuCorpus);
    for (int epoch = 0; epoch < 2; ++epoch) {
        for (size_t first = 0; first < kGpuCorpus.size(); first += 3) {
            SCOPED_TRACE("epoch " + std::to_string(epoch) + " batch at " + std::to_string(first));
            const auto h = host.train_batch(std::vector<std::string>(kGpuCorpus.begin() + first, kGpuCorpus.begin() + first + 3));
            const auto d = device.train_prepared(first, first + 3);
            EXPECT_EQ(d.trained, h.trained);
            ASSERT_EQ(d.predictor_losses.size(), h.predictor_losses.size());
            for (size_t b = 0; b < h.predictor_losses.size(); ++b)
                EXPECT_NEAR(d.predictor_losses[b], h.predictor_losses[b], 2e-3f * std::max(1.0f, h.predictor_losses[b]));
            EXPECT_NEAR(d.sigreg_loss, h.sigreg_loss, 2e-3f * std::max(1.0f, std::abs(h.sigreg_loss)));
        }
    }
    EXPECT_EQ(device.texts_tokenized(), kGpuCorpus.size());
}

// TD-589: the device diagnostics count one sample per encoder-block forward call, and the saturation agrees
// with the host hook on the same weights. (The entropy is each head's average, as in chatbot GPU training;
// the host hook takes the entropy of the head-averaged weights, a different number.)
TEST_F(LeJEPAEncoderTest, DeviceDiagnosticsAccumulateOnTheDevice) {
    if (!lejepa_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    LeJEPAEncoder device(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    device.build_tokenizer(kGpuCorpus, 120);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    device.register_parameters_with_optimizer(opt);
    // The host hook on the untrained weights, for the target views phase 1 encodes first.
    float host_sat = 0.0f;
    int host_calls = 0;
    for (int l = 0; l < NUM_LAYERS; ++l) {
        device.get_encoder_block(l)->get_feed_forward()->set_activation_hook([&](const Matrix& m) {
            int s = 0;
            for (const auto& row : m.data)
                for (float v : row)
                    s += std::abs(v) < 0.01f ? 1 : 0;
            host_sat += static_cast<float>(s) / static_cast<float>(m.rows * m.cols);
            ++host_calls;
        });
    }
    for (int i = 0; i < 2; ++i)
        device.encode(kGpuCorpus[static_cast<size_t>(i)]);
    device.gpu_begin_training();
    adai::gpu::GPUMatrix acc(1, 4);
    acc.zero();
    device.set_gpu_diagnostics(acc.device_ptr());
    // Two texts: phase 1 encodes both target views (2 x NUM_LAYERS block calls) before any step.
    device.gpu_encode(kGpuCorpus[0]);
    device.gpu_encode(kGpuCorpus[1]);
    float d[4];
    acc.download(d, 4);
    EXPECT_FLOAT_EQ(d[1], 2.0f * NUM_LAYERS) << "one saturation sample per feed-forward call";
    EXPECT_FLOAT_EQ(d[3], 2.0f * NUM_LAYERS) << "one entropy sample per attention call";
    EXPECT_NEAR(d[0] / d[1], host_sat / static_cast<float>(host_calls), 1e-3f);
    EXPECT_GT(d[2] / d[3], 0.0f);
    device.set_gpu_diagnostics(nullptr);
    device.gpu_encode(kGpuCorpus[2]);
    acc.download(d, 4);
    EXPECT_FLOAT_EQ(d[1], 2.0f * NUM_LAYERS) << "nullptr turns it off";
}

// TD-590: a device batch of prepared texts moves only its scalars: per text, the optimizer's one float per
// group, and one download of the batch's results. No IDs (uploaded once for the pass), weights, activations
// or gradients, and no host Matrix operation sent to the device.
TEST_F(LeJEPAEncoderTest, ADeviceBatchMovesOnlyItsScalars) {
    if (!lejepa_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    LeJEPAEncoder device(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    device.build_tokenizer(kGpuCorpus, 120);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    device.register_parameters_with_optimizer(opt);
    device.gpu_begin_training();
    device.prepare_texts(kGpuCorpus);
    device.train_prepared(0, 3);  // uploads the prepared IDs once
    const auto before = adai::gpu::transfer_stats();
    const auto result = device.train_prepared(3, 6);
    const auto d = adai::gpu::transfer_delta(adai::gpu::transfer_stats(), before);
    const auto batch = result.trained.size();
    ASSERT_EQ(batch, 3u);
    EXPECT_EQ(d.uploads, 0u) << "the IDs are already on the device";
    EXPECT_EQ(d.downloads, batch + 1) << "one norm download per optimizer step, one for the batch's results";
    // The weights alone are VOCAB_SIZE x D_MODEL floats for the embedding (128 KB here): the scalars are
    // a few hundred bytes.
    EXPECT_LT(d.download_bytes, 4096u) << d.download_bytes << " bytes";
    EXPECT_EQ(d.host_round_trips, 0u) << "no host Matrix operation went to the device";
}

TEST_F(LeJEPAEncoderTest, DeviceTrainingNeedsARegisteredOptimizer) {
    if (!lejepa_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    LeJEPAEncoder device(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_SEQ_LEN, 16);
    device.build_tokenizer(kGpuCorpus, 120);
    EXPECT_THROW(device.gpu_begin_training(), std::logic_error);
    EXPECT_THROW(device.gpu_train_batch({{1, 2, 3}}), std::logic_error);
}
#endif  // ADAI_ENABLE_GPU
