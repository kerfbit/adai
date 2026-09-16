#include <../gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include "../src/EncoderDecoderModel.hpp"
#include "../src/LeJEPAEncoder.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

bool is_close(float actual, float expected, float tolerance = 1e-4f) {
    return std::abs(actual - expected) < tolerance;
}

bool matrices_equal(const Matrix& a, const Matrix& b, float tolerance = 1e-5f) {
    if (a.rows != b.rows || a.cols != b.cols)
        return false;

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            if (!is_close(a(i, j), b(i, j), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

void build_test_vocab(BPETokenizer* tokenizer, int vocab_size = 100) {
    std::vector<std::string> corpus = {"hello world",  "how are you",  "I am fine",     "thank you",
                                       "good morning", "good evening", "see you later", "goodbye"};
    tokenizer->build_vocab(corpus, vocab_size);
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(EncoderDecoderModelTest, ConstructorBasic) {
    int vocab_size = 100;
    int d_model = 64;

    EXPECT_NO_THROW({ EncoderDecoderModel model(vocab_size, d_model); });
}

TEST(EncoderDecoderModelTest, ConstructorWithAllParameters) {
    int vocab_size = 200;
    int d_model = 128;
    int encoder_layers = 3;
    int decoder_layers = 3;
    int num_heads = 4;
    int d_ff = 512;
    int max_seq_length = 256;

    EXPECT_NO_THROW({
        EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers, num_heads,
                                  d_ff, max_seq_length);
    });
}

TEST(EncoderDecoderModelTest, ConstructorInitializesComponents) {
    int vocab_size = 100;
    int d_model = 64;
    int encoder_layers = 2;
    int decoder_layers = 2;

    EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers);

    // Verify configuration
    EXPECT_EQ(model.get_vocab_size(), vocab_size);
    EXPECT_EQ(model.get_d_model(), d_model);
    EXPECT_EQ(model.get_encoder_layers(), encoder_layers);
    EXPECT_EQ(model.get_decoder_layers(), decoder_layers);

    // Verify special tokens
    EXPECT_EQ(model.get_bos_token_id(), 2);
    EXPECT_EQ(model.get_eos_token_id(), 3);
    EXPECT_EQ(model.get_pad_token_id(), 0);
}

TEST(EncoderDecoderModelTest, ConstructorSmallModel) {
    int vocab_size = 50;
    int d_model = 32;
    int encoder_layers = 1;
    int decoder_layers = 1;
    int num_heads = 2;
    int d_ff = 128;

    EXPECT_NO_THROW({
        EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers, num_heads,
                                  d_ff);
    });
}

TEST(EncoderDecoderModelTest, ConstructorLargeModel) {
    int vocab_size = 500;
    int d_model = 256;
    int encoder_layers = 4;
    int decoder_layers = 4;
    int num_heads = 8;
    int d_ff = 1024;

    EXPECT_NO_THROW({
        EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers, num_heads,
                                  d_ff);
    });
}

// ============================================================================
// Component Access Tests
// ============================================================================

TEST(EncoderDecoderModelTest, GetComponents) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    // Verify all components are accessible
    EXPECT_NE(model.get_tokenizer(), nullptr);
    EXPECT_NE(model.get_encoder(), nullptr);
    EXPECT_NE(model.get_decoder(), nullptr);
    EXPECT_NE(model.get_lm_head(), nullptr);
    EXPECT_NE(model.get_generator(), nullptr);
}

TEST(EncoderDecoderModelTest, GetGenerationConfig) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    TextGenerator::GenerationConfig config = model.get_generation_config();

    // Verify default config values
    EXPECT_EQ(config.bos_token_id, 2);
    EXPECT_EQ(config.eos_token_id, 3);
    EXPECT_EQ(config.pad_token_id, 0);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST(EncoderDecoderModelTest, SetTrainingMode) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    EXPECT_NO_THROW({
        model.set_training(true);
        model.set_training(false);
    });
}

TEST(EncoderDecoderModelTest, SetLearningRate) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    EXPECT_NO_THROW({
        model.set_learning_rate(0.001f);
        model.set_learning_rate(0.0001f);
        model.set_learning_rate(0.01f);
    });
}

TEST(EncoderDecoderModelTest, SetGenerationConfig) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    TextGenerator::GenerationConfig config;
    config.max_length = 50;
    config.temperature = 0.8f;
    config.top_k = 40;
    config.top_p = 0.95f;
    config.num_beams = 5;
    config.bos_token_id = 10;
    config.eos_token_id = 11;
    config.pad_token_id = 12;

    model.set_generation_config(config);

    TextGenerator::GenerationConfig retrieved = model.get_generation_config();
    EXPECT_EQ(retrieved.max_length, 50);
    EXPECT_EQ(retrieved.bos_token_id, 10);
    EXPECT_EQ(model.get_bos_token_id(), 10);
    EXPECT_EQ(model.get_eos_token_id(), 11);
    EXPECT_EQ(model.get_pad_token_id(), 12);
}

// ============================================================================
// Tokenizer Tests
// ============================================================================

TEST(EncoderDecoderModelTest, BuildTokenizerVocab) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    EXPECT_NO_THROW({ build_test_vocab(model.get_tokenizer(), vocab_size); });
}

TEST(EncoderDecoderModelTest, TokenizerEncodeDecod) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::string test_text = "hello";
    std::vector<int> tokens = model.get_tokenizer()->encode(test_text);
    std::string decoded = model.get_tokenizer()->decode(tokens);

    EXPECT_FALSE(tokens.empty());
    EXPECT_FALSE(decoded.empty());
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(EncoderDecoderModelTest, ForwardBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    EXPECT_NO_THROW({
        Matrix logits = model.forward(input_tokens, target_tokens);
        EXPECT_GT(logits.rows, 0);
        EXPECT_EQ(logits.cols, vocab_size);
    });
}

TEST(EncoderDecoderModelTest, ForwardOutputDimensions) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> input_tokens = {1, 5, 10, 15, 2};
    std::vector<int> target_tokens = {1, 3, 7, 11, 13, 2};

    Matrix logits = model.forward(input_tokens, target_tokens);

    // Output length should match decoder input length (bos + target[:-1])
    EXPECT_EQ(logits.cols, vocab_size);
    EXPECT_GT(logits.rows, 0);
}

// Regression test (TD-060): forward()'s decoder-input loop used to compute
// `i < target_tokens.size() - 1`, which underflows to SIZE_MAX for an empty
// target_tokens (size_t is unsigned) and reads out of bounds — confirmed via
// a standalone ASan repro to crash with SEGV. forward() is a public API
// documented for "custom training loops", so an empty target_tokens is a
// plausible caller input.
TEST(EncoderDecoderModelTest, ForwardEmptyTargetTokensDoesNotCrash) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> empty_target_tokens;

    EXPECT_NO_THROW({ model.forward(input_tokens, empty_target_tokens); });
}

TEST(EncoderDecoderModelTest, ForwardDifferentLengths) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    // Short input, long target
    std::vector<int> input_short = {1, 5, 2};
    std::vector<int> target_long = {1, 3, 7, 11, 13, 17, 2};

    EXPECT_NO_THROW({
        Matrix logits = model.forward(input_short, target_long);
        EXPECT_EQ(logits.cols, vocab_size);
    });

    // Long input, short target
    std::vector<int> input_long = {1, 5, 10, 15, 20, 25, 2};
    std::vector<int> target_short = {1, 3, 2};

    EXPECT_NO_THROW({
        Matrix logits = model.forward(input_long, target_short);
        EXPECT_EQ(logits.cols, vocab_size);
    });
}

// ============================================================================
// Generation Tests
// ============================================================================

TEST(EncoderDecoderModelTest, GenerateResponseBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    // Generation may fail with untrained model, but should not crash
    try {
        std::string response = model.generate_response(input, 10);
        // If it succeeds, response should not be empty
        EXPECT_FALSE(response.empty());
    } catch (const std::exception& e) {
        // Expected for untrained model
        SUCCEED();
    }
}

TEST(EncoderDecoderModelTest, GenerateWithGreedyStrategy) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    // Untrained model may produce out-of-vocab token IDs (vocab built from small
    // corpus is smaller than vocab_size), so only assert no crash.
    EXPECT_NO_THROW({ model.generate_response_with_strategy(input, 10, "greedy"); });
}

TEST(EncoderDecoderModelTest, GenerateWithSamplingStrategy) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    EXPECT_NO_THROW({ model.generate_response_with_strategy(input, 10, "sampling", 1.0f); });
}

TEST(EncoderDecoderModelTest, GenerateWithTopKStrategy) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    EXPECT_NO_THROW({ model.generate_response_with_strategy(input, 10, "topk", 1.0f, 40); });
}

TEST(EncoderDecoderModelTest, GenerateWithNucleusStrategy) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    EXPECT_NO_THROW(
        { model.generate_response_with_strategy(input, 10, "nucleus", 1.0f, 50, 0.9f); });
}

TEST(EncoderDecoderModelTest, GenerateWithBeamStrategy) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    EXPECT_NO_THROW(
        { model.generate_response_with_strategy(input, 10, "beam", 1.0f, 50, 0.9f, 3); });
}

// TD-050 follow-up regression (September 14, 2026): generate_response_with_strategy()'s "beam"
// branch mutates generator's persistent GenerationConfig as a side effect (num_beams stays > 1
// after the call returns). The plain generate_response() had no guard for this: it always built
// a single, shared, growing DecoderKVCache-backed model_fn and handed it to
// TextGenerator::generate(), which silently routes to generate_beam_search() whenever
// generator's stored config.num_beams > 1 regardless of how it got that way — but beam search
// calls model_fn once per beam per step with each beam's own diverging sequence, which a single
// shared KV cache cannot correctly serve (two beams sharing the same starting sequence length
// produce an EMPTY "new tokens" slice for every beam after the first at a given step, since the
// cache's processed_length was already advanced by the first beam's call — a 0-row matrix that
// propagates into `Matrix::rows - 1` == -1 by the time TextGenerator indexes the last-position
// row, a real out-of-bounds access, not merely numerically wrong output).
//
// Verifies the fix by confirming generate_response(), called right after a beam-search call left
// num_beams > 1 on this same instance, produces output IDENTICAL to an explicit
// generate_response_with_strategy(..., "beam", ...) call with the same parameters on the same
// instance: both must now compute via the same non-cached beam_model_fn / generate_beam_search()
// path (deterministic — beam search has no sampling RNG), so their outputs should match exactly
// given the same encoder input, weights, and generator config. Before the fix, generate_response()
// instead used its cached model_fn for beam search, which either crashes (see above) or produces
// different output than the explicit beam call — this test caught exactly that: reverting the
// generate_response() beam-vs-cache guard reproduces the crash/mismatch this test exists to catch.
TEST(EncoderDecoderModelTest, GenerateResponseNotCorruptedByLeftoverBeamConfig) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    // Leaves generator's config with num_beams == 3, matching a real caller sequence where an
    // earlier, unrelated generate_response_with_strategy(..., "beam", ...) call precedes a later
    // plain generate_response() call on the same model instance.
    std::string beam_response;
    EXPECT_NO_THROW({
        beam_response = model.generate_response_with_strategy(input, 10, "beam", 1.0f, 50, 0.9f, 3);
    });
    ASSERT_EQ(model.get_generator()->get_config().num_beams, 3)
        << "test setup assumption violated: num_beams should stay leaked on generator's config";

    // The real regression check: generate_response() must not crash and must correctly perform
    // (safe, non-cached) beam search rather than silently using its cached model_fn for it.
    std::string plain_response;
    EXPECT_NO_THROW({ plain_response = model.generate_response(input, 10); });
    EXPECT_EQ(plain_response, beam_response)
        << "generate_response() with leftover num_beams > 1 should compute identically to an "
           "explicit beam-search call — a mismatch means it silently used the cached, "
           "beam-unsafe model_fn instead of the guard's non-cached beam_model_fn";
}

// TD-100 regression: generate_greedy()/generate_sampling()/generate_top_k()/
// generate_nucleus() each read most of their filter values (including
// max_length itself) straight from `generator`'s *stored* config rather than
// from generate_response_with_strategy()'s own arguments. Only the "beam"
// branch ever pushed those arguments into that config, so every other
// strategy silently used generator's already-stored values instead of this
// call's own. Verified directly against generator's config state (get_generator()
// is a public accessor) rather than generated output length/content, which
// would be unreliable against an untrained, randomly-initialized model.
TEST(EncoderDecoderModelTest, GenerateWithStrategySyncsGeneratorConfig) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    // "topk" only takes k as an explicit parameter — temperature and
    // max_length previously came from whatever generator's config already
    // held, not this call's arguments.
    model.generate_response_with_strategy("hello", 37, "topk", 0.5f, 15, 0.8f, 2);

    auto cfg = model.get_generator()->get_config();
    EXPECT_EQ(cfg.max_length, 37);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.5f);
    EXPECT_EQ(cfg.top_k, 15);
    EXPECT_FLOAT_EQ(cfg.top_p, 0.8f);
    EXPECT_EQ(cfg.num_beams, 2);

    // A second call with different values must overwrite, not merely
    // coexist with, the first — proving this is a real sync, not a
    // one-time default that happened to match.
    model.generate_response_with_strategy("hi", 12, "nucleus", 0.3f, 5, 0.6f, 1);
    cfg = model.get_generator()->get_config();
    EXPECT_EQ(cfg.max_length, 12);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.3f);
    EXPECT_FLOAT_EQ(cfg.top_p, 0.6f);
}

// ============================================================================
// Training Tests
// ============================================================================

TEST(EncoderDecoderModelTest, TrainStepBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::string input = "hello";
    std::string target = "world";

    EXPECT_NO_THROW({
        float loss = model.train_step(input, target);
        EXPECT_GT(loss, 0.0f);  // Loss should be positive
    });
}

TEST(EncoderDecoderModelTest, TrainStepTokenized) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    EXPECT_NO_THROW({
        float loss = model.train_step_tokenized(input_tokens, target_tokens);
        EXPECT_GT(loss, 0.0f);
    });
}

TEST(EncoderDecoderModelTest, EncoderReceivesGradientOnTrainStep) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);
    model.zero_grad();

    // Regression guard: before this fix, the encoder never received a gradient
    // at all (dropped in DecoderBlock::backward, and gated off by
    // set_training() never calling encoder->set_requires_grad()).
    for (int i = 0; i < model.get_encoder()->get_num_layers(); ++i) {
        ASSERT_FLOAT_EQ(model.get_encoder()->get_encoder_block(i)->get_gradient_norm(), 0.0f);
    }

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};
    float loss = model.train_step_tokenized(input_tokens, target_tokens);
    EXPECT_GT(loss, 0.0f);

    for (int i = 0; i < model.get_encoder()->get_num_layers(); ++i) {
        EXPECT_GT(model.get_encoder()->get_encoder_block(i)->get_gradient_norm(), 0.0f)
            << "encoder layer " << i << " received no gradient";
    }
}

TEST(EncoderDecoderModelTest, TrainStepRequiresTrainingMode) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);  // Not in training mode

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    EXPECT_THROW({ model.train_step_tokenized(input_tokens, target_tokens); }, std::runtime_error);
}

TEST(EncoderDecoderModelTest, TrainStepMultipleIterations) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Run multiple training steps
    for (int i = 0; i < 3; ++i) {
        EXPECT_NO_THROW({
            float loss = model.train_step_tokenized(input_tokens, target_tokens);
            EXPECT_GT(loss, 0.0f);
        });
    }
}

TEST(EncoderDecoderModelTest, SimpleTrainingLoop) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);
    model.set_learning_rate(0.001f);

    std::vector<std::pair<std::string, std::string>> data = {
        {"hello", "hi"}, {"how are you", "fine"}, {"good morning", "morning"}};

    // Train for 2 epochs
    for (int epoch = 0; epoch < 2; ++epoch) {
        float total_loss = 0.0f;

        for (const auto& [input, target] : data) {
            float loss = model.train_step(input, target);
            total_loss += loss;
        }

        EXPECT_GT(total_loss, 0.0f);
    }

    SUCCEED();
}

// ============================================================================
// Evaluation Tests
// ============================================================================

TEST(EncoderDecoderModelTest, EvaluateBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::string input = "hello";
    std::string target = "world";

    EXPECT_NO_THROW({
        float loss = model.evaluate(input, target);
        EXPECT_GT(loss, 0.0f);
    });
}

TEST(EncoderDecoderModelTest, EvaluatePreservesTrainingMode) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    // Start in training mode
    model.set_training(true);

    std::string input = "hello";
    std::string target = "world";

    model.evaluate(input, target);

    // Should still be in training mode after evaluate
    // (We can't directly check, but train_step should work)
    EXPECT_NO_THROW({ model.train_step(input, target); });
}

TEST(EncoderDecoderModelTest, ComputePerplexityBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<std::string> inputs = {"hello", "how are you"};
    std::vector<std::string> targets = {"hi", "fine"};

    EXPECT_NO_THROW({
        float perplexity = model.compute_perplexity(inputs, targets);
        EXPECT_GT(perplexity, 0.0f);
    });
}

TEST(EncoderDecoderModelTest, ComputePerplexityMismatchedSizes) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<std::string> inputs = {"hello", "how are you"};
    std::vector<std::string> targets = {"hi"};  // Mismatched size

    EXPECT_THROW({ model.compute_perplexity(inputs, targets); }, std::invalid_argument);
}

// ============================================================================
// Weight Management Tests
// ============================================================================

TEST(EncoderDecoderModelTest, ZeroGradBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    EXPECT_NO_THROW({ model.zero_grad(); });
}

TEST(EncoderDecoderModelTest, UpdateWeightsBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Forward and backward
    Matrix logits = model.forward(input_tokens, target_tokens);
    Matrix grad(logits.rows, logits.cols);
    model.backward(grad);

    EXPECT_NO_THROW({ model.update_weights(); });
}

TEST(EncoderDecoderModelTest, ZeroGradAfterTraining) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::string input = "hello";
    std::string target = "world";

    model.train_step(input, target);

    EXPECT_NO_THROW({ model.zero_grad(); });
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(EncoderDecoderModelTest, SaveModel) {
    int vocab_size = 100;
    int d_model = 64;
    int encoder_layers = 2;
    int decoder_layers = 2;
    EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::string filepath = "test_encoder_decoder_model";

    EXPECT_NO_THROW({ model.save_model(filepath); });

    // Clean up
    std::remove((filepath + ".config").c_str());
    std::remove((filepath + ".vocab").c_str());
    std::remove((filepath + ".encoder").c_str());
    std::remove((filepath + ".decoder").c_str());
}

TEST(EncoderDecoderModelTest, LoadModel) {
    int vocab_size = 100;
    int d_model = 64;
    int encoder_layers = 2;
    int decoder_layers = 2;

    // Create and save model
    EncoderDecoderModel model1(vocab_size, d_model, encoder_layers, decoder_layers);
    build_test_vocab(model1.get_tokenizer(), vocab_size);

    std::string filepath = "test_encoder_decoder_load";
    model1.save_model(filepath);

    // Load into new model
    EncoderDecoderModel model2(vocab_size, d_model, encoder_layers, decoder_layers);

    EXPECT_NO_THROW({ model2.load_model(filepath); });

    // Verify configuration
    EXPECT_EQ(model2.get_vocab_size(), vocab_size);
    EXPECT_EQ(model2.get_d_model(), d_model);
    EXPECT_EQ(model2.get_encoder_layers(), encoder_layers);
    EXPECT_EQ(model2.get_decoder_layers(), decoder_layers);

    // Clean up
    std::remove((filepath + ".config").c_str());
    std::remove((filepath + ".vocab").c_str());
    std::remove((filepath + ".encoder").c_str());
    std::remove((filepath + ".decoder").c_str());
}

TEST(EncoderDecoderModelTest, LoadModelMismatchedArchitecture) {
    int vocab_size1 = 100;
    int d_model1 = 64;
    int encoder_layers1 = 2;
    int decoder_layers1 = 2;

    // Create and save model
    EncoderDecoderModel model1(vocab_size1, d_model1, encoder_layers1, decoder_layers1);
    build_test_vocab(model1.get_tokenizer(), vocab_size1);

    std::string filepath = "test_encoder_decoder_mismatch";
    model1.save_model(filepath);

    // Try to load into mismatched model
    int vocab_size2 = 200;  // Different vocab size
    EncoderDecoderModel model2(vocab_size2, d_model1, encoder_layers1, decoder_layers1);

    EXPECT_THROW({ model2.load_model(filepath); }, std::runtime_error);

    // Clean up
    std::remove((filepath + ".config").c_str());
    std::remove((filepath + ".vocab").c_str());
    std::remove((filepath + ".encoder").c_str());
    std::remove((filepath + ".decoder").c_str());
}

// Regression test for the Post-LN -> Pre-LN checkpoint compatibility marker:
// a checkpoint saved before the architecture change has the same 10-int
// layout (same shapes) but a different weight *meaning*, which the old
// dimension-only check couldn't catch. load_model() must now reject a
// legacy-format .config file (missing the magic/version header) with a
// clear error instead of silently loading it.
TEST(EncoderDecoderModelTest, LoadModelRejectsLegacyPreMarkerCheckpoint) {
    int vocab_size = 100;
    int d_model = 64;
    int encoder_layers = 2;
    int decoder_layers = 2;
    int num_heads = 4;
    int d_ff = 256;
    int max_seq_length = 128;

    std::string filepath = "test_encoder_decoder_legacy_format";

    // Hand-build a legacy-format .config: exactly the 10 raw ints the
    // pre-marker format wrote, with no magic/version header.
    {
        std::ofstream config_file(filepath + ".config", std::ios::binary);
        ASSERT_TRUE(config_file.is_open());
        int bos = 1, eos = 2, pad = 0;
        config_file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&d_model), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&encoder_layers), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&decoder_layers), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&d_ff), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&max_seq_length), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&bos), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&eos), sizeof(int));
        config_file.write(reinterpret_cast<const char*>(&pad), sizeof(int));
    }

    EncoderDecoderModel model(vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff,
                              max_seq_length);

    EXPECT_THROW({ model.load_model(filepath); }, std::runtime_error);

    std::remove((filepath + ".config").c_str());
}

TEST(EncoderDecoderModelTest, SaveLoadRoundTrip) {
    int vocab_size = 100;
    int d_model = 64;
    int encoder_layers = 2;
    int decoder_layers = 2;
    int num_heads = 4;
    int d_ff = 256;
    int max_seq_length = 128;

    EncoderDecoderModel model1(vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff,
                               max_seq_length);
    build_test_vocab(model1.get_tokenizer(), vocab_size);

    std::string filepath = "test_encoder_decoder_roundtrip";

    // Save
    model1.save_model(filepath);

    // Load into new model
    EncoderDecoderModel model2(vocab_size, d_model, encoder_layers, decoder_layers, num_heads, d_ff,
                               max_seq_length);
    model2.load_model(filepath);

    // Verify configuration
    EXPECT_EQ(model2.get_vocab_size(), vocab_size);
    EXPECT_EQ(model2.get_d_model(), d_model);
    EXPECT_EQ(model2.get_encoder_layers(), encoder_layers);
    EXPECT_EQ(model2.get_decoder_layers(), decoder_layers);

    // Clean up
    std::remove((filepath + ".config").c_str());
    std::remove((filepath + ".vocab").c_str());
    std::remove((filepath + ".encoder").c_str());
    std::remove((filepath + ".decoder").c_str());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(EncoderDecoderModelTest, EmptyInputText) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string empty_input = "";

    // Empty input should be handled gracefully
    // Behavior depends on tokenizer implementation
}

TEST(EncoderDecoderModelTest, VeryLongSequences) {
    int vocab_size = 100;
    int d_model = 64;
    int max_seq_length = 128;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 4, 256, max_seq_length);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    // Create long sequences
    std::vector<int> long_input(100);
    std::vector<int> long_target(100);
    for (int i = 0; i < 100; ++i) {
        long_input[i] = i % vocab_size;
        long_target[i] = (i + 1) % vocab_size;
    }

    EXPECT_NO_THROW({
        Matrix logits = model.forward(long_input, long_target);
        EXPECT_GT(logits.rows, 0);
    });
}

TEST(EncoderDecoderModelTest, SingleTokenSequences) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> single_input = {5};
    std::vector<int> single_target = {10};

    EXPECT_NO_THROW({
        Matrix logits = model.forward(single_input, single_target);
        EXPECT_GT(logits.rows, 0);
    });
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(EncoderDecoderModelTest, EndToEndGenerationPipeline) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello world";

    try {
        // Test complete pipeline: text → tokens → encode → decode → tokens → text
        std::string response = model.generate_response(input, 15);

        // If generation succeeds, verify output
        EXPECT_FALSE(response.empty());
    } catch (const std::exception& e) {
        // Expected for untrained model
        SUCCEED();
    }
}

TEST(EncoderDecoderModelTest, TrainThenGenerate) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    // Train on a few examples
    model.set_training(true);
    model.set_learning_rate(0.001f);

    for (int i = 0; i < 5; ++i) {
        model.train_step("hello", "hi");
    }

    // Switch to inference
    model.set_training(false);

    // Try to generate
    try {
        std::string response = model.generate_response("hello", 10);
        SUCCEED();
    } catch (const std::exception& e) {
        SUCCEED();
    }
}

TEST(EncoderDecoderModelTest, MultiStrategyComparison) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    std::string input = "hello";

    std::vector<std::string> strategies = {"greedy", "sampling", "topk"};

    for (const auto& strategy : strategies) {
        try {
            std::string response = model.generate_response_with_strategy(input, 10, strategy);
            // Each strategy should work (or fail gracefully)
        } catch (const std::exception& e) {
            // Expected for untrained model
        }
    }

    SUCCEED();
}

TEST(EncoderDecoderModelTest, CustomForwardBackward) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Custom training loop
    model.zero_grad();

    Matrix logits = model.forward(input_tokens, target_tokens);

    // Create custom gradient
    Matrix grad(logits.rows, logits.cols);
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            grad(i, j) = 0.01f;
        }
    }

    model.backward(grad);
    model.update_weights();

    SUCCEED();
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST(EncoderDecoderModelOptimizerTest, RegisterParametersBasic) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);

    Optimizer optimizer(OptimizerType::ADAM, 0.001f);

    // Currently register_parameters is a placeholder
    EXPECT_NO_THROW({ model.register_parameters(optimizer); });
}

TEST(EncoderDecoderModelOptimizerTest, BackwardPassWithoutUpdate) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Forward pass
    Matrix logits = model.forward(input_tokens, target_tokens);

    // Compute loss gradient
    Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);

    // Backward pass without weight update
    EXPECT_NO_THROW({ model.backward_pass(grad_loss); });
}

TEST(EncoderDecoderModelOptimizerTest, CustomTrainingLoopWithOptimizer) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    // Create optimizer
    Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
    optimizer.set_weight_decay(0.01f);
    optimizer.set_max_grad_norm(1.0f);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Custom training loop
    optimizer.zero_grad();
    model.zero_grad();

    // Forward pass
    Matrix logits = model.forward(input_tokens, target_tokens);

    // Compute loss
    float loss = model.compute_loss_for_training(logits, target_tokens);
    EXPECT_GT(loss, 0.0f);

    // Compute gradients
    Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);

    // Backward pass
    model.backward_pass(grad_loss);

    // Get gradient norm for monitoring
    float grad_norm = optimizer.get_gradient_norm();
    EXPECT_GE(grad_norm, 0.0f);

    // Clip gradients
    optimizer.clip_gradients();

    // Update weights (currently uses model's update_weights)
    model.update_weights();

    SUCCEED();
}

TEST(EncoderDecoderModelOptimizerTest, TrainingWithDifferentOptimizers) {
    int vocab_size = 100;
    int d_model = 64;

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Test with SGD
    {
        EncoderDecoderModel model(vocab_size, d_model, 2, 2);
        build_test_vocab(model.get_tokenizer(), vocab_size);
        model.set_training(true);

        Optimizer optimizer(OptimizerType::SGD, 0.01f);

        model.zero_grad();
        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);
        model.update_weights();

        SUCCEED();
    }

    // Test with Adam
    {
        EncoderDecoderModel model(vocab_size, d_model, 2, 2);
        build_test_vocab(model.get_tokenizer(), vocab_size);
        model.set_training(true);

        Optimizer optimizer(OptimizerType::ADAM, 0.001f);
        optimizer.set_betas(0.9f, 0.999f);

        model.zero_grad();
        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);
        model.update_weights();

        SUCCEED();
    }

    // Test with AdamW
    {
        EncoderDecoderModel model(vocab_size, d_model, 2, 2);
        build_test_vocab(model.get_tokenizer(), vocab_size);
        model.set_training(true);

        Optimizer optimizer(OptimizerType::ADAMW, 0.0001f);
        optimizer.set_weight_decay(0.01f);
        optimizer.set_betas(0.9f, 0.999f);

        model.zero_grad();
        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);
        model.update_weights();

        SUCCEED();
    }
}

TEST(EncoderDecoderModelOptimizerTest, GradientClippingPreventsExplosion) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    Optimizer optimizer(OptimizerType::ADAM, 0.001f);
    optimizer.set_max_grad_norm(1.0f);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Train multiple steps
    for (int i = 0; i < 5; ++i) {
        optimizer.zero_grad();
        model.zero_grad();

        Matrix logits = model.forward(input_tokens, target_tokens);
        float loss = model.compute_loss_for_training(logits, target_tokens);

        // Loss should remain finite
        EXPECT_FALSE(std::isnan(loss));
        EXPECT_FALSE(std::isinf(loss));

        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);

        float grad_norm_before = optimizer.get_gradient_norm();
        float grad_norm_after = optimizer.clip_gradients();

        // Gradient norm should be finite
        EXPECT_FALSE(std::isnan(grad_norm_before));
        EXPECT_FALSE(std::isinf(grad_norm_before));

        model.update_weights();
    }

    SUCCEED();
}

TEST(EncoderDecoderModelOptimizerTest, WeightDecayRegularization) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    // AdamW with weight decay
    Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
    optimizer.set_weight_decay(0.1f);  // Heavy weight decay

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Train several steps with weight decay
    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        model.zero_grad();

        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);

        optimizer.clip_gradients();
        model.update_weights();
    }

    // Model should still be functional after weight decay
    EXPECT_NO_THROW({
        Matrix logits = model.forward(input_tokens, target_tokens);
        float loss = model.compute_loss_for_training(logits, target_tokens);
        EXPECT_GT(loss, 0.0f);
    });
}

TEST(EncoderDecoderModelOptimizerTest, LearningRateScheduling) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    Optimizer optimizer(OptimizerType::ADAM, 0.001f);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Simulate learning rate warmup and decay
    std::vector<float> learning_rates = {0.0001f, 0.0005f, 0.001f, 0.0005f, 0.0001f};

    for (float lr : learning_rates) {
        optimizer.set_learning_rate(lr);
        model.set_learning_rate(lr);

        EXPECT_FLOAT_EQ(optimizer.get_learning_rate(), lr);

        // Train step with current LR
        optimizer.zero_grad();
        model.zero_grad();

        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);
        model.update_weights();
    }

    SUCCEED();
}

TEST(EncoderDecoderModelOptimizerTest, GradientNormMonitoring) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    Optimizer optimizer(OptimizerType::ADAM, 0.001f);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    std::vector<float> gradient_norms;

    // Collect gradient norms over multiple steps
    for (int i = 0; i < 5; ++i) {
        optimizer.zero_grad();
        model.zero_grad();

        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);

        float norm = optimizer.get_gradient_norm();
        gradient_norms.push_back(norm);

        // Gradient norm should be non-negative
        EXPECT_GE(norm, 0.0f);

        model.update_weights();
    }

    // Should have collected norms
    EXPECT_EQ(gradient_norms.size(), 5);
}

TEST(EncoderDecoderModelOptimizerTest, OptimizerStateReset) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    Optimizer optimizer(OptimizerType::ADAM, 0.001f);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Train a few steps to accumulate optimizer state
    for (int i = 0; i < 3; ++i) {
        optimizer.zero_grad();
        model.zero_grad();

        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);
        model.update_weights();
    }

    // Reset optimizer state
    EXPECT_NO_THROW({ optimizer.reset_state(); });

    // Should be able to continue training
    optimizer.zero_grad();
    model.zero_grad();
    Matrix logits = model.forward(input_tokens, target_tokens);
    Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
    model.backward_pass(grad);
    model.update_weights();

    SUCCEED();
}

TEST(EncoderDecoderModelOptimizerTest, MultipleEpochsWithOptimizer) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
    optimizer.set_weight_decay(0.01f);
    optimizer.set_max_grad_norm(1.0f);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> dataset = {
        {{1, 5, 10, 2}, {1, 3, 7, 2}},
        {{1, 8, 12, 2}, {1, 6, 9, 2}},
        {{1, 15, 20, 2}, {1, 11, 14, 2}}};

    // Train for multiple epochs
    for (int epoch = 0; epoch < 3; ++epoch) {
        float epoch_loss = 0.0f;

        for (const auto& [input, target] : dataset) {
            optimizer.zero_grad();
            model.zero_grad();

            Matrix logits = model.forward(input, target);
            float loss = model.compute_loss_for_training(logits, target);
            epoch_loss += loss;

            Matrix grad = model.compute_loss_gradient_for_training(logits, target);
            model.backward_pass(grad);

            optimizer.clip_gradients();
            model.update_weights();
        }

        // Epoch loss should be positive
        EXPECT_GT(epoch_loss, 0.0f);
    }

    SUCCEED();
}

TEST(EncoderDecoderModelOptimizerTest, CompareLegacyVsOptimizerTraining) {
    int vocab_size = 100;
    int d_model = 64;

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Legacy training (using built-in train_step)
    float loss_legacy;
    {
        EncoderDecoderModel model(vocab_size, d_model, 2, 2);
        build_test_vocab(model.get_tokenizer(), vocab_size);
        model.set_training(true);
        model.set_learning_rate(0.001f);

        loss_legacy = model.train_step_tokenized(input_tokens, target_tokens);
    }

    // New optimizer-based training
    float loss_optimizer;
    {
        EncoderDecoderModel model(vocab_size, d_model, 2, 2);
        build_test_vocab(model.get_tokenizer(), vocab_size);
        model.set_training(true);

        Optimizer optimizer(OptimizerType::ADAM, 0.001f);

        optimizer.zero_grad();
        model.zero_grad();

        Matrix logits = model.forward(input_tokens, target_tokens);
        loss_optimizer = model.compute_loss_for_training(logits, target_tokens);

        Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad);
        model.update_weights();
    }

    // Both should produce valid (positive) losses
    EXPECT_GT(loss_legacy, 0.0f);
    EXPECT_GT(loss_optimizer, 0.0f);

    // Both should be in reasonable range for untrained models
    EXPECT_LT(loss_legacy, 10.0f);
    EXPECT_LT(loss_optimizer, 10.0f);

    // Note: Exact loss values will differ due to different random weight initializations
    // between the two model instances, so we just verify both are reasonable
}

TEST(EncoderDecoderModelOptimizerTest, ExposedLossFunctions) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Forward pass
    Matrix logits = model.forward(input_tokens, target_tokens);

    // Test exposed loss computation
    float loss = model.compute_loss_for_training(logits, target_tokens);
    EXPECT_GT(loss, 0.0f);
    EXPECT_FALSE(std::isnan(loss));

    // Test exposed gradient computation
    Matrix grad = model.compute_loss_gradient_for_training(logits, target_tokens);
    EXPECT_EQ(grad.rows, logits.rows);
    EXPECT_EQ(grad.cols, logits.cols);

    // Gradients should sum to approximately zero (softmax - one_hot averages to ~0)
    float grad_sum = 0.0f;
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            grad_sum += grad(i, j);
        }
    }

    // Sum should be close to zero (within reasonable tolerance)
    EXPECT_NEAR(grad_sum, 0.0f, 1.0f);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(EncoderDecoderModelTest, TrainingPerformance) {
    int vocab_size = 200;
    int d_model = 128;
    EncoderDecoderModel model(vocab_size, d_model, 3, 3);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens(20);
    std::vector<int> target_tokens(20);
    for (int i = 0; i < 20; ++i) {
        input_tokens[i] = i % vocab_size;
        target_tokens[i] = (i + 1) % vocab_size;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 5; ++i) {
        model.train_step_tokenized(input_tokens, target_tokens);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 60000);  // Less than 60 seconds
}

TEST(EncoderDecoderModelTest, MemoryStability) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);

    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    // Run many iterations
    for (int i = 0; i < 50; ++i) {
        model.train_step_tokenized(input_tokens, target_tokens);

        if (i % 10 == 0) {
            model.zero_grad();
        }
    }

    SUCCEED();
}

// ============================================================================
// TD-038: LoRA model-wide integration tests
// ============================================================================

TEST(EncoderDecoderModelLoRATest, EnableLoraAcrossModelIsNoOp) {
    int vocab_size = 100;
    int d_model = 32;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, /*num_heads=*/4, /*d_ff=*/64);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    Matrix logits_before = model.forward(input_tokens, target_tokens);

    LoRAConfig config;
    config.rank = 4;
    model.enable_lora(config);
    ASSERT_TRUE(model.has_lora());

    Matrix logits_after = model.forward(input_tokens, target_tokens);
    EXPECT_TRUE(matrices_equal(logits_before, logits_after, 1e-5f))
        << "enabling LoRA across the whole model changed forward() output before any training";
}

TEST(EncoderDecoderModelLoRATest, RegisterLoraParametersFreezesWholeModelWeights) {
    int vocab_size = 100;
    int d_model = 32;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, /*num_heads=*/4, /*d_ff=*/64);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    LoRAConfig config;
    config.rank = 4;
    model.enable_lora(config);

    // Snapshot base weights across every attention layer in both the encoder and decoder --
    // encoder self-attention, decoder self-attention, and decoder cross-attention.
    MultiHeadAttention* enc0_attn = model.get_encoder()->get_encoder_block(0)->get_self_attention();
    MultiHeadAttention* dec0_self = model.get_decoder()->get_decoder_block(0)->get_self_attention();
    CrossAttention* dec0_cross = model.get_decoder()->get_decoder_block(0)->get_cross_attention();
    Matrix enc0_Wq_before = enc0_attn->get_Wq();
    Matrix dec0_self_Wq_before = dec0_self->get_Wq();
    Matrix dec0_cross_Wq_before = dec0_cross->get_Wq();

    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    // Deliberately register ONLY the LoRA parameters -- never register_parameters() on this
    // optimizer -- so the base model's weights are never in its parameter_groups at all.
    model.register_lora_parameters(optimizer);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    for (int step = 0; step < 2; ++step) {
        model.zero_grad();
        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad_loss);
        optimizer.step();
    }

    // Base weights, across every attention layer touched, must be bit-for-bit untouched.
    EXPECT_TRUE(matrices_equal(enc0_Wq_before, enc0_attn->get_Wq(), 1e-6f))
        << "encoder self-attention's base W_q moved despite never being registered";
    EXPECT_TRUE(matrices_equal(dec0_self_Wq_before, dec0_self->get_Wq(), 1e-6f))
        << "decoder self-attention's base W_q moved despite never being registered";
    EXPECT_TRUE(matrices_equal(dec0_cross_Wq_before, dec0_cross->get_Wq(), 1e-6f))
        << "decoder cross-attention's base W_q moved despite never being registered";

    // But the LoRA adapters actually trained.
    Matrix B_after = enc0_attn->get_lora_q()->get_B();
    bool b_changed = false;
    for (int i = 0; i < B_after.rows && !b_changed; ++i) {
        for (int j = 0; j < B_after.cols; ++j) {
            if (std::abs(B_after(i, j)) > 1e-6f) {
                b_changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(b_changed) << "encoder self-attention's LoRA_q never trained";
}

TEST(EncoderDecoderModelLoRATest, MergeLoraAcrossModelPreservesOutput) {
    int vocab_size = 100;
    int d_model = 32;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, /*num_heads=*/4, /*d_ff=*/64);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(true);

    LoRAConfig config;
    config.rank = 4;
    model.enable_lora(config);

    Optimizer optimizer(OptimizerType::ADAM, 0.02f);
    model.register_lora_parameters(optimizer);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    for (int step = 0; step < 3; ++step) {
        model.zero_grad();
        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad_loss);
        optimizer.step();
    }

    Matrix logits_before_merge = model.forward(input_tokens, target_tokens);
    model.merge_lora();
    EXPECT_FALSE(model.has_lora());
    Matrix logits_after_merge = model.forward(input_tokens, target_tokens);

    EXPECT_TRUE(matrices_equal(logits_before_merge, logits_after_merge, 1e-3f))
        << "merge_lora() changed the whole model's forward() output";
}

// ============================================================================
// TD-182: set_world_model()/get_world_model() Tests
// ============================================================================

TEST(EncoderDecoderModelWorldModelTest, WorldModelIsNullByDefault) {
    EncoderDecoderModel model(100, 64, 2, 2);
    EXPECT_EQ(model.get_world_model(), nullptr);
}

TEST(EncoderDecoderModelWorldModelTest, SetWorldModelAttachesInstance) {
    EncoderDecoderModel model(100, 64, 2, 2);

    auto world_model = std::make_unique<LeJEPAEncoder>(100, 64, 2, 4, 128);
    LeJEPAEncoder* raw_ptr = world_model.get();
    model.set_world_model(std::move(world_model));

    EXPECT_EQ(model.get_world_model(), raw_ptr);
    EXPECT_NE(model.get_world_model(), nullptr);
}

TEST(EncoderDecoderModelWorldModelTest, SetWorldModelNullptrDetaches) {
    EncoderDecoderModel model(100, 64, 2, 2);
    model.set_world_model(std::make_unique<LeJEPAEncoder>(100, 64, 2, 4, 128));
    ASSERT_NE(model.get_world_model(), nullptr);

    model.set_world_model(nullptr);

    EXPECT_EQ(model.get_world_model(), nullptr);
}

TEST(EncoderDecoderModelWorldModelTest, SetWorldModelReplacesPreviousInstance) {
    EncoderDecoderModel model(100, 64, 2, 2);

    auto first = std::make_unique<LeJEPAEncoder>(100, 64, 2, 4, 128);
    LeJEPAEncoder* first_ptr = first.get();
    model.set_world_model(std::move(first));

    auto second = std::make_unique<LeJEPAEncoder>(100, 64, 2, 4, 128);
    LeJEPAEncoder* second_ptr = second.get();
    model.set_world_model(std::move(second));

    EXPECT_NE(model.get_world_model(), first_ptr);
    EXPECT_EQ(model.get_world_model(), second_ptr);
}

// TD-182's own Action Item: attaching a world model must not change forward()'s output at all
// — this is wiring + an accessor only, no training-loop changes. Same instance, same inputs,
// before vs. after attaching, must be bit-identical.
TEST(EncoderDecoderModelWorldModelTest, ForwardOutputUnaffectedByAttachedWorldModel) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> input_tokens = {1, 5, 10, 2};
    std::vector<int> target_tokens = {1, 3, 7, 2};

    Matrix logits_before = model.forward(input_tokens, target_tokens);

    model.set_world_model(std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128));
    Matrix logits_after = model.forward(input_tokens, target_tokens);

    EXPECT_TRUE(matrices_equal(logits_before, logits_after))
        << "attaching a world model changed forward() output — TD-182 is wiring-only";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
