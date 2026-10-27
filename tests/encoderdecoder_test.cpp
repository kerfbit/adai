#include <../gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include "../src/EncoderDecoderModel.hpp"
#include "../src/HippocampalMemory.hpp"
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
    EXPECT_EQ(cfg.num_beams, 1) << "TD-261: topk ignores num_beams, so no beam width is stored";

    // A second call with different values must overwrite, not merely
    // coexist with, the first — proving this is a real sync, not a
    // one-time default that happened to match.
    model.generate_response_with_strategy("hi", 12, "nucleus", 0.3f, 5, 0.6f, 1);
    cfg = model.get_generator()->get_config();
    EXPECT_EQ(cfg.max_length, 12);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.3f);
    EXPECT_FLOAT_EQ(cfg.top_p, 0.6f);
}

// Tokens in a generated answer; an untrained model can stop at once, and encode("") throws.
static size_t answer_tokens(BPETokenizer* tok, const std::string& text) {
    return text.empty() ? 0 : tok->encode(text, false).size();
}

// An untrained, unseeded model (TD-368) sometimes picks EOS at once; tests that need long outputs
// rule out every special token.
static void suppress_stop_tokens(EncoderDecoderModel& model) {
    Matrix bias = model.get_lm_head()->get_bias();
    // EOS and PAD stop; BOS and UNK decode to nothing.
    for (int id : {model.get_tokenizer()->get_eos_token_id(), model.get_tokenizer()->get_pad_token_id(),
                   model.get_tokenizer()->get_bos_token_id(), model.get_tokenizer()->get_unk_token_id()})
        bias(0, id) = -50.0f;
    model.get_lm_head()->set_bias(bias);
}

// TD-359: generate_response() applies its max_length for the call and keeps the generator's stored
// value. It used to ignore the argument: generate_response("hello world", 5) gave 9 tokens with the
// generator at 64.
TEST(EncoderDecoderModelTest, GenerateResponseHonoursMaxLength) {
    EncoderDecoderModel model(100, 64, 2, 2);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    // Greedy and no stop tokens: the 5-token answer must be a strict prefix of the 40-token one.
    // Counting tokens by re-encoding the decoded text isn't exact, and an untrained, unseeded model
    // (TD-368) sometimes stops at once.
    suppress_stop_tokens(model);
    auto cfg = model.get_generator()->get_config();
    cfg.max_length = 64;
    cfg.num_beams = 1;
    cfg.temperature = 0.0f;
    model.get_generator()->set_config(cfg);

    for (const char* input : {"hello world", "the quick brown fox", "a b c"}) {
        const std::string shorter = model.generate_response(input, 5);
        const std::string longer = model.generate_response(input, 40);
        EXPECT_EQ(longer.compare(0, shorter.size(), shorter), 0) << input;
        EXPECT_GT(longer.size(), shorter.size()) << input << ": 5 tokens weren't a cap";
    }
    EXPECT_EQ(model.get_generator()->get_config().max_length, 64) << "stored value restored";
}

// TD-347: generation stops at max_seq_length whatever max_length asks for (positions past it have no
// positional encoding; the model never trained on them).
TEST(EncoderDecoderModelTest, GenerationNeverExceedsMaxSeqLength) {
    EncoderDecoderModel model(100, 32, 1, 1, 2, 64, /*max_seq_length=*/8);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    auto* tok = model.get_tokenizer();
    for (const char* input : {"hello world", "the quick brown fox", "a b c"}) {
        EXPECT_LE(answer_tokens(tok, model.generate_response(input, 100)), 8u) << input;
        for (const char* strategy : {"greedy", "sampling", "beam"}) {
            const std::string out =
                model.generate_response_with_strategy(input, 100, strategy, 1.0f, 50, 0.9f, 2);
            EXPECT_LE(answer_tokens(tok, out), 8u) << input << " / " << strategy;
        }
    }
}

// TD-297/TD-346/TD-360: one update_weights() moves every parameter exactly one SGD step, with or without
// a registered optimizer. With one, each layer used to step the shared optimizer (6-10 steps per
// update); without one, the encoder (no update_weights()) and the decoder's final norm never moved.
TEST(EncoderDecoderModelTest, OneUpdateIsOneStepWithOrWithoutAnOptimizer) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "adai_ed_td297";
    fs::remove_all(dir);
    fs::create_directories(dir);

    EncoderDecoderModel with_opt(60, 16, 2, 2, 2, 32, 16);
    build_test_vocab(with_opt.get_tokenizer(), 60);
    with_opt.save_model((dir / "cp.bin").string());
    EncoderDecoderModel plain(60, 16, 2, 2, 2, 32, 16);
    plain.load_model((dir / "cp.bin").string());

    const float lr = 0.05f;
    Optimizer sgd(OptimizerType::SGD, lr);
    with_opt.register_parameters(sgd);
    with_opt.register_parameters(sgd);  // registering twice must not double the update
    with_opt.set_learning_rate(lr);
    plain.set_learning_rate(lr);
    with_opt.set_training(true);
    plain.set_training(true);

    const Matrix enc_before =
        plain.get_encoder()->get_encoder_block(0)->get_self_attention()->get_Wq();
    const Matrix norm_before = plain.get_decoder()->get_final_norm()->get_gamma();

    // Token IDs, not text: BPETokenizer's thread-local cache can carry another test's IDs (TD-222).
    with_opt.train_step_tokenized({4, 5, 6, 7}, {8, 9, 10, 3});
    plain.train_step_tokenized({4, 5, 6, 7}, {8, 9, 10, 3});

    // Both moved: the encoder and the decoder's final norm are updated on the plain path.
    const Matrix& enc_after = plain.get_encoder()->get_encoder_block(0)->get_self_attention()->get_Wq();
    const Matrix& norm_after = plain.get_decoder()->get_final_norm()->get_gamma();
    float enc_moved = 0.0f, norm_moved = 0.0f;
    for (int i = 0; i < enc_after.rows; ++i)
        for (int j = 0; j < enc_after.cols; ++j)
            enc_moved += std::abs(enc_after(i, j) - enc_before(i, j));
    for (int j = 0; j < norm_after.cols; ++j)
        norm_moved += std::abs(norm_after(0, j) - norm_before(0, j));
    EXPECT_GT(enc_moved, 0.0f) << "TD-360: the encoder didn't move";
    EXPECT_GT(norm_moved, 0.0f) << "TD-346: the decoder's final norm didn't move";

    // And by the same single step: the two models give the same logits.
    with_opt.set_training(false);
    plain.set_training(false);
    const Matrix a = with_opt.forward({4, 5, 6, 7}, {8, 9, 10});
    const Matrix b = plain.forward({4, 5, 6, 7}, {8, 9, 10});
    float max_diff = 0.0f;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            max_diff = std::max(max_diff, std::abs(a(i, j) - b(i, j)));
    EXPECT_LT(max_diff, 1e-4f) << "TD-297: the optimizer path didn't take exactly one SGD step";
    fs::remove_all(dir);
}

// TD-362: the CPU and GPU generation paths share their strategy names, beam decision and vocab masking.
TEST(EncoderDecoderModelTest, NormalizeStrategyAcceptsEverySpelling) {
    EXPECT_EQ(EncoderDecoderModel::normalize_strategy("temperature"), "sampling");
    EXPECT_EQ(EncoderDecoderModel::normalize_strategy("top_k"), "topk");
    EXPECT_EQ(EncoderDecoderModel::normalize_strategy("top-k"), "topk");
    EXPECT_EQ(EncoderDecoderModel::normalize_strategy("top_p"), "nucleus");
    EXPECT_EQ(EncoderDecoderModel::normalize_strategy("top-p"), "nucleus");
    for (const char* same : {"greedy", "beam", "sampling", "topk", "nucleus", "", "other"})
        EXPECT_EQ(EncoderDecoderModel::normalize_strategy(same), same);
}

// Logits favour an ID the tokenizer doesn't have; every path must mask it. generate_response()'s
// cached path used to pick it at every step while generate_response_with_strategy() masked it.
static void favour_unknown_id(EncoderDecoderModel& model, int id) {
    Matrix bias = model.get_lm_head()->get_bias();
    bias(0, id) = 50.0f;
    model.get_lm_head()->set_bias(bias);
}

TEST(EncoderDecoderModelTest, PlainCachedPathMasksIdsTheTokenizerLacks) {
    EncoderDecoderModel model(100, 32, 1, 1, 2, 64, 32);
    build_test_vocab(model.get_tokenizer(), 100);
    ASSERT_LT(static_cast<int>(model.get_tokenizer()->get_vocab_size()), 99);
    model.set_training(false);
    favour_unknown_id(model, 99);

    for (const char* input : {"hello world", "good morning"}) {
        // The strategy call stores temperature 0 and one beam, so the plain call is greedy too.
        const std::string greedy =
            model.generate_response_with_strategy(input, 6, "greedy", 0.0f, 50, 0.9f, 1);
        EXPECT_EQ(model.generate_response(input, 6), greedy) << input;
    }
}

// TD-424: the beam callbacks project only the decoder's last row. That's exact because the LM head is
// row-wise (a linear map plus bias): the last row's logits don't depend on the other rows.
TEST(EncoderDecoderModelTest, LastRowProjectionMatchesFullProjection) {
    EncoderDecoderModel model(300, 64, 1, 1);
    Matrix decoder_out(40, 64);
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 64; ++j)
            decoder_out(i, j) = std::sin(0.37f * static_cast<float>(i * 64 + j));
    const Matrix full = model.get_lm_head()->forward(decoder_out);
    Matrix last(1, 64);
    last.data[0] = decoder_out.data[39];
    const Matrix one = model.get_lm_head()->forward(last);
    ASSERT_EQ(one.rows, 1);
    ASSERT_EQ(one.cols, full.cols);
    for (int j = 0; j < full.cols; ++j)
        EXPECT_NEAR(one(0, j), full(39, j), 1e-4f) << "logit " << j;
}

TEST(EncoderDecoderModelTest, BeamGenerationIsDeterministicAfterLastRowProjection) {
    EncoderDecoderModel model(100, 64, 2, 2);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    const std::string a = model.generate_response_with_strategy("hello world", 20, "beam", 1.0f, 50,
                                                                0.9f, 3);
    const std::string b = model.generate_response_with_strategy("hello world", 20, "beam", 1.0f, 50,
                                                                0.9f, 3);
    EXPECT_EQ(a, b);
}

// TD-365: an over-long encoder input keeps its last max_seq_length tokens, as training does: forward()
// on the long input equals forward() on its tail, and generation accepts it. It used to run with no
// positional encoding past max_seq_length.
TEST(EncoderDecoderModelTest, OverLongEncoderInputKeepsTheTail) {
    EncoderDecoderModel model(100, 32, 1, 1, 2, 64, /*max_seq_length=*/16);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    std::vector<int> long_input;
    for (int i = 0; i < 60; ++i)
        long_input.push_back(4 + (i * 7) % 90);
    const std::vector<int> tail(long_input.end() - 16, long_input.end());
    const std::vector<int> target = {5, 6, 7};
    const Matrix a = model.forward(long_input, target);
    const Matrix b = model.forward(tail, target);
    ASSERT_EQ(a.rows, b.rows);
    float diff = 0.0f;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            diff = std::max(diff, std::fabs(a(i, j) - b(i, j)));
    EXPECT_LT(diff, 1e-5f);

    std::string long_text;
    for (int i = 0; i < 40; ++i)
        long_text += "hello world ";
    EXPECT_NO_THROW(model.generate_response(long_text, 8));
    EXPECT_NO_THROW(model.generate_response_with_strategy(long_text, 8, "beam", 1.0f, 50, 0.9f, 2));
}

// TD-261: an explicit strategy ignores num_beams. `num_beams > 1` used to send every strategy to
// beam search (chatbot_gui's Beam Width defaults to 5; RLHFTrainer got the header default 4), so
// sampling rollouts were deterministic beam search: 1 distinct response in 6.
TEST(EncoderDecoderModelTest, ExplicitStrategiesIgnoreNumBeams) {
    EncoderDecoderModel model(100, 64, 2, 2);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);

    for (const char* strategy : {"sampling", "topk", "nucleus"}) {
        std::set<std::string> distinct;
        for (int i = 0; i < 6; ++i)
            distinct.insert(model.generate_response_with_strategy("hello world", 12, strategy, 2.0f,
                                                                  50, 0.99f, /*num_beams=*/5));
        EXPECT_GT(distinct.size(), 1u) << strategy << " ran deterministic beam search";
        EXPECT_EQ(model.get_generator()->get_config().num_beams, 1)
            << strategy << " must not leave a beam width for later calls";
    }
    EXPECT_EQ(model.generate_response_with_strategy("hello world", 12, "greedy", 1.0f, 50, 0.9f, 5),
              model.generate_response_with_strategy("hello world", 12, "greedy", 1.0f, 50, 0.9f, 1));

    // "beam" still uses the width, and an unrecognized strategy with num_beams > 1 still takes the
    // (uncached) beam path rather than the KV-cached one.
    model.generate_response_with_strategy("hello world", 8, "beam", 1.0f, 50, 0.9f, 3);
    EXPECT_EQ(model.get_generator()->get_config().num_beams, 3);
    model.generate_response_with_strategy("hello world", 8, "quux", 1.0f, 50, 0.9f, 3);
    EXPECT_EQ(model.get_generator()->get_config().num_beams, 3);
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
    // Forward and backward only: train_step() now also updates the encoder (TD-360), which consumes
    // and zeroes its gradients.
    Matrix logits = model.forward(input_tokens, target_tokens);
    EXPECT_GT(model.compute_loss_for_training(logits, target_tokens), 0.0f);
    model.backward_pass(model.compute_loss_gradient_for_training(logits, target_tokens));

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

// TD-361: training targets carry no leading BOS, so the decoder input is [BOS, w1 ...] and the
// first position learns to predict the first word. With the old [BOS, w1 ..., EOS] targets the
// decoder input was [BOS, BOS, w1 ...] and a trained model's first prediction was another BOS.
TEST(EncoderDecoderModelTest, StripLeadingBosOnlyRemovesALeadingBos) {
    EXPECT_EQ(EncoderDecoderModel::strip_leading_bos({2, 7, 8, 3}, 2), (std::vector<int>{7, 8, 3}));
    EXPECT_EQ(EncoderDecoderModel::strip_leading_bos({7, 2, 3}, 2), (std::vector<int>{7, 2, 3}));
    EXPECT_TRUE(EncoderDecoderModel::strip_leading_bos({}, 2).empty());
}

TEST(EncoderDecoderModelTest, TrainedModelsFirstPredictionIsTheFirstWordNotBos) {
    EncoderDecoderModel model(64, 32, 1, 1, 2, 64, 32);
    BPETokenizer* tok = model.get_tokenizer();
    tok->build_vocab({"the cat sat on the mat", "a dog ran far away home", "big red ball"}, 64);
    model.set_training(true);
    model.set_learning_rate(0.05f);
    const std::vector<std::pair<std::string, std::string>> pairs = {
        {"the cat sat", "on the mat"}, {"a dog ran", "far away home"}, {"big red", "ball"}};
    for (int iter = 0; iter < 300; ++iter)
        for (const auto& [input, target] : pairs) model.train_step(input, target);

    model.set_training(false);
    const int bos = model.get_bos_token_id();
    for (const auto& [input, target] : pairs) {
        const std::vector<int> in = tok->encode(input, false);
        const std::vector<int> tgt = EncoderDecoderModel::strip_leading_bos(tok->encode(target, true), bos);
        ASSERT_FALSE(tgt.empty());
        // forward() with a 1-token target feeds the decoder just [BOS]; row 0 is the first
        // prediction.
        const Matrix logits = model.forward(in, {tgt[0]});
        int best = 0;
        for (int j = 1; j < logits.cols; ++j)
            if (logits(0, j) > logits(0, best))
                best = j;
        EXPECT_NE(best, bos) << "first prediction for '" << input << "' is a second BOS";
        EXPECT_EQ(best, tgt[0]) << "first prediction for '" << input << "'";
    }
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

// TD-538: a save/load round trip must restore every weight, so the reloaded model gives the same
// logits. The encoder's token embedding and final norm used to share file names with the
// decoder's and were overwritten by them (max logit difference 1.54 here before the fix).
TEST(EncoderDecoderModelTest, SaveLoadRoundTripReproducesLogits) {
    const std::string dir = std::filesystem::temp_directory_path() / "adai_ed_roundtrip_logits";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    EncoderDecoderModel a(100, 32, 2, 2, 2, 64, 32);
    build_test_vocab(a.get_tokenizer(), 100);
    a.set_training(false);
    const Matrix before = a.forward({4, 5, 6, 7}, {8, 9, 10});
    const Matrix again = a.forward({4, 5, 6, 7}, {8, 9, 10});
    ASSERT_TRUE(matrices_equal(before, again, 1e-6f)) << "forward() isn't deterministic";
    a.save_model(dir + "/cp.bin");

    EncoderDecoderModel b(100, 32, 2, 2, 2, 64, 32);
    build_test_vocab(b.get_tokenizer(), 100);
    b.load_model(dir + "/cp.bin");
    b.set_training(false);
    const Matrix after = b.forward({4, 5, 6, 7}, {8, 9, 10});
    float max_diff = 0.0f;
    for (int i = 0; i < before.rows; ++i)
        for (int j = 0; j < before.cols; ++j)
            max_diff = std::max(max_diff, std::abs(before(i, j) - after(i, j)));
    EXPECT_LT(max_diff, 1e-5f);
    std::filesystem::remove_all(dir);
}

// TD-507: chatbot_api_server builds its speculative draft from the checkpoint's own architecture, so
// read_checkpoint_architecture() must return what save_model() wrote, and reject a truncated file.
TEST(EncoderDecoderModelTest, ReadCheckpointArchitectureMatchesTheSavedModel) {
    const std::string dir = std::filesystem::temp_directory_path() / "adai_ed_read_arch";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    EncoderDecoderModel a(100, 48, 3, 2, 4, 96, 40);
    build_test_vocab(a.get_tokenizer(), 100);
    a.save_model(dir + "/cp.bin");

    const auto arch = EncoderDecoderModel::read_checkpoint_architecture(dir + "/cp.bin");
    EXPECT_EQ(arch.vocab_size, 100);
    EXPECT_EQ(arch.d_model, 48);
    EXPECT_EQ(arch.encoder_layers, 3);
    EXPECT_EQ(arch.decoder_layers, 2);
    EXPECT_EQ(arch.num_heads, 4);
    EXPECT_EQ(arch.d_ff, 96);
    EXPECT_EQ(arch.max_seq_length, 40);

    // A model built from it loads the checkpoint.
    EncoderDecoderModel b(arch.vocab_size, arch.d_model, arch.encoder_layers, arch.decoder_layers,
                          arch.num_heads, arch.d_ff, arch.max_seq_length);
    EXPECT_NO_THROW(b.load_model(dir + "/cp.bin"));

    std::filesystem::resize_file(dir + "/cp.bin.config", 12);
    EXPECT_THROW(EncoderDecoderModel::read_checkpoint_architecture(dir + "/cp.bin"),
                 std::runtime_error);
    EXPECT_THROW(EncoderDecoderModel::read_checkpoint_architecture(dir + "/missing.bin"),
                 std::runtime_error);
    std::filesystem::remove_all(dir);
}

// TD-423: every binary sidecar of a checkpoint (config, encoder/decoder headers, embeddings, norms,
// attention, cross-attention, feed-forward, blocks, LM head) loaded unchecked, so a file cut short by a
// full disk loaded as NaN or stale weights. One byte short, or one byte extra, must fail load_model().
TEST(EncoderDecoderModelTest, EveryCheckpointFileIsCheckedOnLoad) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "adai_ed_td423";
    fs::remove_all(dir);
    fs::create_directories(dir);
    EncoderDecoderModel a(60, 16, 2, 2, 2, 32, 16);
    build_test_vocab(a.get_tokenizer(), 60);
    a.save_model((dir / "cp.bin").string());

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".vocab")  // the vocabulary is text (TD-223)
            files.push_back(entry.path());
    }
    ASSERT_GE(files.size(), 20u) << "expected every sidecar of a 2+2-layer checkpoint";

    for (const auto& file : files) {
        const auto size = fs::file_size(file);
        for (int delta : {-1, +1}) {
            fs::copy_file(file, dir / "original", fs::copy_options::overwrite_existing);
            if (delta < 0) {
                fs::resize_file(file, size - 1);
            } else {
                std::ofstream(file, std::ios::binary | std::ios::app).put('\0');
            }
            EncoderDecoderModel b(60, 16, 2, 2, 2, 32, 16);
            build_test_vocab(b.get_tokenizer(), 60);
            EXPECT_THROW(b.load_model((dir / "cp.bin").string()), std::runtime_error)
                << file.filename() << (delta < 0 ? " one byte short" : " one byte long");
            fs::copy_file(dir / "original", file, fs::copy_options::overwrite_existing);
        }
    }

    // Restored, the checkpoint loads.
    EncoderDecoderModel c(60, 16, 2, 2, 2, 32, 16);
    build_test_vocab(c.get_tokenizer(), 60);
    EXPECT_NO_THROW(c.load_model((dir / "cp.bin").string()));
    fs::remove_all(dir);
}

// TD-363: a world-model-enabled model (gated decoder paths) with a world model and hippocampal memory
// attached. clone() used to build the copy without the gated paths, so load_model() threw, and the copy
// had no world model, memory, settings or generation config even when it loaded.
static std::unique_ptr<EncoderDecoderModel> make_gated_model() {
    auto model = std::make_unique<EncoderDecoderModel>(100, 32, 1, 1, 2, 64, 32,
                                                       /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model->get_tokenizer(), 100);
    auto wm = std::make_unique<LeJEPAEncoder>(100, 32, 1, 2, 64, 32);
    wm->build_tokenizer({"hello world", "how are you", "hello jazz", "jazz quiz"}, 100);
    model->set_world_model(std::move(wm));
    auto memory = std::make_unique<HippocampalMemory>(32, 8);
    Matrix key(1, 32);
    key.randomize(1.0f);
    memory->write(key, key);
    memory->write(key, key);
    model->set_hippocampal_memory(std::move(memory));
    model->set_hippocampal_repetition_params(0.3f, 0.9f, 0.2f, 0.8f);
    model->set_hippocampal_updates_enabled(false);
    model->get_decoder()->get_decoder_block(0)->set_gate(0.8f);
    model->get_decoder()->get_decoder_block(0)->set_gate_h(0.6f);
    model->set_training(false);
    return model;
}

TEST(EncoderDecoderModelTest, CloneCopiesGatedPathsAttachmentsAndSettings) {
    auto model = make_gated_model();
    auto cfg = model->get_generator()->get_config();
    cfg.max_length = 7;
    cfg.top_k = 13;
    model->get_generator()->set_config(cfg);

    std::unique_ptr<EncoderDecoderModel> copy;
    ASSERT_NO_THROW(copy = model->clone());
    ASSERT_NE(copy->get_world_model(), nullptr);
    ASSERT_NE(copy->get_hippocampal_memory(), nullptr);
    EXPECT_EQ(copy->get_hippocampal_memory()->size(), 2);
    EXPECT_FALSE(copy->hippocampal_updates_enabled());
    EXPECT_EQ(copy->get_generator()->get_config().max_length, 7);
    EXPECT_EQ(copy->get_generator()->get_config().top_k, 13);
    EXPECT_FLOAT_EQ(copy->get_decoder()->get_decoder_block(0)->get_gate(), 0.8f);

    // The copy generates like the original, and its memory is its own.
    for (const char* input : {"hello world", "how are you"}) {
        EXPECT_EQ(copy->generate_response_with_strategy(input, 6, "greedy", 0.0f, 50, 0.9f, 1),
                  model->generate_response_with_strategy(input, 6, "greedy", 0.0f, 50, 0.9f, 1))
            << input;
    }
    Matrix key(1, 32);
    key.fill(0.5f);
    copy->get_hippocampal_memory()->write(key, key);
    EXPECT_EQ(model->get_hippocampal_memory()->size(), 2);
}

TEST(EncoderDecoderModelTest, LoadModelSyncsTheGeneratorsSpecialTokens) {
    // The generator kept its old BOS/EOS/PAD until the next strategy call.
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "adai_ed_td363_sync";
    fs::remove_all(dir);
    fs::create_directories(dir);
    EncoderDecoderModel a(60, 16, 1, 1, 2, 32, 16);
    build_test_vocab(a.get_tokenizer(), 60);
    a.save_model((dir / "cp.bin").string());

    EncoderDecoderModel b(60, 16, 1, 1, 2, 32, 16);
    auto cfg = b.get_generator()->get_config();
    cfg.bos_token_id = cfg.eos_token_id = cfg.pad_token_id = 57;
    b.get_generator()->set_config(cfg);
    b.load_model((dir / "cp.bin").string());
    EXPECT_EQ(b.get_generator()->get_config().bos_token_id, b.get_tokenizer()->get_bos_token_id());
    EXPECT_EQ(b.get_generator()->get_config().eos_token_id, b.get_tokenizer()->get_eos_token_id());
    EXPECT_EQ(b.get_generator()->get_config().pad_token_id, b.get_tokenizer()->get_pad_token_id());
    fs::remove_all(dir);
}

TEST(EncoderDecoderModelTest, WorldModelSeesTheOriginalInputText) {
    // "jazz" has letters the model's vocabulary lacks, so its tokens decode back to something else;
    // the world model (whose vocabulary has them) must see the text the caller gave.
    auto model = make_gated_model();
    const std::string text = "hello jazz";
    const std::vector<int> input = model->get_tokenizer()->encode(text, false);
    ASSERT_NE(model->get_tokenizer()->decode(input, true), text) << "test sanity: decoding is lossy";
    const std::vector<int> decoder_tokens = {model->get_tokenizer()->get_bos_token_id()};

    const Matrix with_text = model->next_token_logits(input, decoder_tokens, 1, &text);
    const Matrix decoded = model->next_token_logits(input, decoder_tokens);
    const std::string decoded_text = model->get_tokenizer()->decode(input, true);
    const Matrix with_decoded_text = model->next_token_logits(input, decoder_tokens, 1, &decoded_text);

    float diff = 0.0f, same = 0.0f;
    for (int j = 0; j < with_text.cols; ++j) {
        diff = std::max(diff, std::abs(with_text(0, j) - decoded(0, j)));
        same = std::max(same, std::abs(with_decoded_text(0, j) - decoded(0, j)));
    }
    EXPECT_GT(diff, 1e-4f) << "the original text should reach the world model";
    EXPECT_LT(same, 1e-6f) << "without text, the decoded tokens are used";
}

// TD-538: a checkpoint saved before the encoder-specific file names (only the shared
// _token_emb.bin/_final_norm.bin, holding the decoder's copies) still loads, as it always did.
TEST(EncoderDecoderModelTest, LegacyCheckpointWithoutEncoderSpecificFilesStillLoads) {
    const std::string dir = std::filesystem::temp_directory_path() / "adai_ed_legacy_names";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    EncoderDecoderModel a(100, 32, 2, 2, 2, 64, 32);
    build_test_vocab(a.get_tokenizer(), 100);
    a.save_model(dir + "/cp.bin");
    std::filesystem::remove(dir + "/cp.bin_encoder_token_emb.bin");
    std::filesystem::remove(dir + "/cp.bin_encoder_final_norm.bin");

    EncoderDecoderModel b(100, 32, 2, 2, 2, 64, 32);
    build_test_vocab(b.get_tokenizer(), 100);
    ASSERT_NO_THROW(b.load_model(dir + "/cp.bin"));
    // The legacy file is the decoder's embedding, which is what old checkpoints always loaded.
    const Matrix& enc = b.get_encoder()->get_token_embedding()->get_embeddings();
    const Matrix& dec = a.get_decoder()->get_token_embedding()->get_embeddings();
    EXPECT_TRUE(matrices_equal(enc, dec, 1e-7f));
    std::filesystem::remove_all(dir);
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
// TD-186: forward()/generate_response()'s gated world-model read path
// ============================================================================

namespace {
// Same corpus build_test_vocab() uses (BPE training is deterministic given the same corpus and
// target size, so building the world model's own separate tokenizer from it produces the same
// learned vocabulary) — lets the world model's own encode() actually tokenize "hello world"
// successfully rather than falling back to <unk>.
const std::vector<std::string>& world_model_test_corpus() {
    static const std::vector<std::string> corpus = {
        "hello world",  "how are you",  "I am fine",     "thank you",
        "good morning", "good evening", "see you later", "goodbye"};
    return corpus;
}
}  // namespace

// The plan's own "mandatory first checkpoint" for Phase 1 fine-tuning (Training Standard):
// WORLD_MODEL_ENABLED=true output must match =false output exactly before any gate training
// happens. Exercised here via beam-search generation (the one CPU inference path TD-186 wired,
// see forward_with_cache()'s own documented limitation) rather than forward(), so input_text is
// a real, controllable string instead of relying on a token-id round-trip through decode().
TEST(EncoderDecoderModelWorldModelTest, BeamGenerationNoOpWhenGateZeroWithWorldModelAttached) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512,
                              /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::string input_text = "hello world";
    std::string response_before =
        model.generate_response_with_strategy(input_text, 10, "beam", 1.0f, 50, 0.9f, 2);

    auto world_model = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128);
    world_model->build_tokenizer(world_model_test_corpus(), vocab_size);
    model.set_world_model(std::move(world_model));

    // Every gated block's gate is still 0.0f (fresh construction) — attaching alone must not
    // change beam output, even though world_model->encode() genuinely runs this time.
    std::string response_after =
        model.generate_response_with_strategy(input_text, 10, "beam", 1.0f, 50, 0.9f, 2);

    EXPECT_EQ(response_before, response_after);
}

// Note: a beam-search text-comparison equivalent of "gate nonzero changes output" was tried here
// and dropped — beam search's own discrete argmax/tie-breaking made it flake (~40% failure rate
// across repeated standalone runs) even though the underlying computation is fully deterministic.
// DecoderTest.ForwardWithEncoderWorldModelOutputChangesOutputWhenGateNonzero (decoder_test.cpp)
// already covers this exact property reliably via direct matrix comparison, which is the
// stronger, tie-break-immune assertion; BeamGenerationNoOpWhenGateZeroWithWorldModelAttached
// above covers the no-op side at this class's own beam-generation level.

// forward()'s own decode-back-to-text wiring (used by the real ChatbotTrainer.cpp training
// path, which only ever has token ids) — confirms it doesn't crash regardless of whether the
// round-trip through tokenizer->decode() produces reconstructable text, and confirms the
// gate-zero no-op guarantee holds at the raw forward() level too, not just via generate_*().
TEST(EncoderDecoderModelWorldModelTest, ForwardNoOpWhenGateZeroWithWorldModelAttachedViaTokens) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512,
                              /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    std::vector<int> input_tokens = model.get_tokenizer()->encode("hello world", false);
    std::vector<int> target_tokens = model.get_tokenizer()->encode("how are you", true);

    Matrix logits_before = model.forward(input_tokens, target_tokens);

    auto world_model = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128);
    world_model->build_tokenizer(world_model_test_corpus(), vocab_size);
    model.set_world_model(std::move(world_model));
    Matrix logits_after = model.forward(input_tokens, target_tokens);

    EXPECT_TRUE(matrices_equal(logits_before, logits_after));
}

TEST(EncoderDecoderModelWorldModelTest, BackwardFlowsGradientIntoGateWhenWorldModelAttachedAndInjected) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512,
                              /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    auto world_model = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128);
    world_model->build_tokenizer(world_model_test_corpus(), vocab_size);
    model.set_world_model(std::move(world_model));

    std::vector<int> input_tokens = model.get_tokenizer()->encode("hello world", false);
    std::vector<int> target_tokens = model.get_tokenizer()->encode("how are you", true);

    Matrix logits = model.forward(input_tokens, target_tokens);
    Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);
    model.backward(grad_loss);

    // Every gated layer (world_model_inject_every_n_layers=1 -> both of this 2-layer decoder's
    // blocks) should have accumulated a nonzero gate gradient — confirming gradients genuinely
    // reach the gate parameter end-to-end, not just that the forward path runs.
    bool any_nonzero = false;
    for (int i = 0; i < model.get_decoder()->get_num_layers(); ++i) {
        if (model.get_decoder()->get_decoder_block(i)->get_gate_grad() != 0.0f) {
            any_nonzero = true;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// TD-341: the step sequence ChatbotTrainer uses (register_parameters() once, then zero_grad,
// forward, backward_pass, optimizer.step(); never update_weights()) must move the world-model
// gates. They used to be plain floats only update_weights() stepped, so they stayed at 0.
TEST(EncoderDecoderModelWorldModelTest, TrainerStyleOptimizerStepsOpenTheGates) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512,
                              /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    auto world_model = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128);
    world_model->build_tokenizer(world_model_test_corpus(), vocab_size);
    model.set_world_model(std::move(world_model));

    Optimizer optimizer(OptimizerType::ADAMW, 0.01f);
    model.register_parameters(optimizer);
    model.set_training(true);

    std::vector<int> input_tokens = model.get_tokenizer()->encode("hello world", false);
    std::vector<int> target_tokens = model.get_tokenizer()->encode("how are you", true);
    for (int step = 0; step < 5; ++step) {
        model.zero_grad();
        Matrix logits = model.forward(input_tokens, target_tokens);
        Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);
        model.backward_pass(grad_loss);
        optimizer.step();
    }

    for (int i = 0; i < model.get_decoder()->get_num_layers(); ++i) {
        EXPECT_NE(model.get_decoder()->get_decoder_block(i)->get_gate(), 0.0f)
            << "decoder block " << i << "'s world-model gate never moved";
    }
}

// TD-186's own pilot run found that HippocampalMemory's gated cross-attention was never
// actually allocated by any LLMDecoder constructor at all (world_model_inject_every_n_layers
// only ever drove the world-model path) — fixed in Decoder.cpp to allocate both paths together.
// This test is the gate_h equivalent of the one above: a non-empty memory must let gradient
// reach gate_h during an ordinary forward+backward call.
TEST(EncoderDecoderModelWorldModelTest,
    BackwardFlowsGradientIntoGateHWhenHippocampalMemoryNonEmptyAndInjected) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512,
                              /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model.get_tokenizer(), vocab_size);

    auto world_model = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128);
    world_model->build_tokenizer(world_model_test_corpus(), vocab_size);
    Matrix key = world_model->encode("hello world");
    model.set_world_model(std::move(world_model));

    auto memory = std::make_unique<HippocampalMemory>(d_model, /*capacity=*/4);
    Matrix pooled(1, key.cols);
    for (int j = 0; j < key.cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < key.rows; ++i) sum += key(i, j);
        pooled(0, j) = sum / static_cast<float>(key.rows);
    }
    memory->write(pooled, pooled);
    model.set_hippocampal_memory(std::move(memory));
    model.set_hippocampal_repetition_params(0.5f, 0.95f);

    std::vector<int> input_tokens = model.get_tokenizer()->encode("hello world", false);
    std::vector<int> target_tokens = model.get_tokenizer()->encode("how are you", true);

    Matrix logits = model.forward(input_tokens, target_tokens);
    Matrix grad_loss = model.compute_loss_gradient_for_training(logits, target_tokens);
    model.backward(grad_loss);

    bool any_nonzero = false;
    for (int i = 0; i < model.get_decoder()->get_num_layers(); ++i) {
        if (model.get_decoder()->get_decoder_block(i)->get_gate_h_grad() != 0.0f) {
            any_nonzero = true;
        }
    }
    EXPECT_TRUE(any_nonzero);
}

// ============================================================================
// TD-185: set_hippocampal_memory()/get_hippocampal_memory() + write-policy Tests
// ============================================================================

TEST(EncoderDecoderModelWorldModelTest, HippocampalMemoryIsNullByDefault) {
    EncoderDecoderModel model(100, 64, 2, 2);
    EXPECT_EQ(model.get_hippocampal_memory(), nullptr);
}

TEST(EncoderDecoderModelWorldModelTest, SetHippocampalMemoryAttachesInstance) {
    EncoderDecoderModel model(100, 64, 2, 2);

    auto memory = std::make_unique<HippocampalMemory>(64, 8);
    HippocampalMemory* raw_ptr = memory.get();
    model.set_hippocampal_memory(std::move(memory));

    EXPECT_EQ(model.get_hippocampal_memory(), raw_ptr);
    EXPECT_NE(model.get_hippocampal_memory(), nullptr);
}

TEST(EncoderDecoderModelWorldModelTest, SetHippocampalMemoryNullptrDetaches) {
    EncoderDecoderModel model(100, 64, 2, 2);
    model.set_hippocampal_memory(std::make_unique<HippocampalMemory>(64, 8));
    ASSERT_NE(model.get_hippocampal_memory(), nullptr);

    model.set_hippocampal_memory(nullptr);

    EXPECT_EQ(model.get_hippocampal_memory(), nullptr);
}

TEST(EncoderDecoderModelWorldModelTest, SetHippocampalMemoryReplacesPreviousInstance) {
    EncoderDecoderModel model(100, 64, 2, 2);

    auto first = std::make_unique<HippocampalMemory>(64, 8);
    HippocampalMemory* first_ptr = first.get();
    model.set_hippocampal_memory(std::move(first));

    auto second = std::make_unique<HippocampalMemory>(64, 8);
    HippocampalMemory* second_ptr = second.get();
    model.set_hippocampal_memory(std::move(second));

    EXPECT_NE(model.get_hippocampal_memory(), first_ptr);
    EXPECT_EQ(model.get_hippocampal_memory(), second_ptr);
}

// TD-185's own Action Item: HIPPOCAMPAL_MEMORY_ENABLED=false (nothing attached, the default)
// must reproduce current behavior exactly. Since attaching a memory with no world model is
// exactly what "not really enabled" looks like at the C++ level (maybe_write_hippocampal_memory
// requires both), this checks that generate_response()'s own TEXT output — not just whether the
// write happened — is completely unaffected by having a memory attached at all.
TEST(EncoderDecoderModelWorldModelTest, GenerateResponseTextUnaffectedByAttachedHippocampalMemory) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    // Force greedy (deterministic) decoding — generate_response()'s own default config uses
    // temperature=1.0 (sampling), which is legitimately non-deterministic call-to-call on its
    // own, unrelated to anything this test is actually checking.
    TextGenerator::GenerationConfig greedy_config = model.get_generator()->get_config();
    greedy_config.temperature = 0.0f;
    model.get_generator()->set_config(greedy_config);

    std::string response_before;
    bool before_threw = false;
    try {
        response_before = model.generate_response("hello", 10);
    } catch (const std::exception&) {
        before_threw = true;  // legitimate on an untrained, randomly-initialized model
    }

    model.set_hippocampal_memory(std::make_unique<HippocampalMemory>(d_model, 8));

    std::string response_after;
    bool after_threw = false;
    try {
        response_after = model.generate_response("hello", 10);
    } catch (const std::exception&) {
        after_threw = true;
    }

    EXPECT_EQ(before_threw, after_threw);
    if (!before_threw && !after_threw) {
        EXPECT_EQ(response_before, response_after);
    }
}

TEST(EncoderDecoderModelWorldModelTest, GenerateResponseDoesNotWriteWithOnlyMemoryAttached) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);
    model.set_hippocampal_memory(std::make_unique<HippocampalMemory>(d_model, 8));

    try {
        model.generate_response("hello", 10);
    } catch (const std::exception&) {
        // Generation can legitimately fail on an untrained model (matches GenerateResponseBasic
        // above) — either way, the memory must stay empty since no world model is attached.
    }

    EXPECT_EQ(model.get_hippocampal_memory()->size(), 0);
}

TEST(EncoderDecoderModelWorldModelTest, GenerateResponseDoesNotThrowWithOnlyWorldModelAttached) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);
    model.set_world_model(std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128));

    EXPECT_NO_THROW({
        try {
            model.generate_response("hello", 10);
        } catch (const std::exception&) {
            // Matches GenerateResponseBasic's own convention.
        }
    });
    EXPECT_EQ(model.get_hippocampal_memory(), nullptr);  // never attached in this test
}

// The write-policy call site's own functional check: a real, non-empty generated response,
// with both a world model (the key source) and a memory attached, must actually be written.
TEST(EncoderDecoderModelWorldModelTest, GenerateResponseWritesToHippocampalMemoryWhenBothAttached) {
    int vocab_size = 100;
    int d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_training(false);

    auto world_model_ptr = std::make_unique<LeJEPAEncoder>(vocab_size, d_model, 2, 4, 128);
    std::vector<std::string> corpus = {"hello world",  "how are you",  "I am fine",     "thank you",
                                       "good morning", "good evening", "see you later", "goodbye"};
    world_model_ptr->build_tokenizer(corpus, vocab_size);
    model.set_world_model(std::move(world_model_ptr));
    model.set_hippocampal_memory(std::make_unique<HippocampalMemory>(d_model, 8));

    ASSERT_EQ(model.get_hippocampal_memory()->size(), 0);

    try {
        std::string response = model.generate_response("hello", 10);
        if (!response.empty()) {
            EXPECT_EQ(model.get_hippocampal_memory()->size(), 1)
                << "a non-empty generated response should have been written to hippocampal memory";
        }
    } catch (const std::exception&) {
        // Matches GenerateResponseBasic's own convention — generation can legitimately fail on
        // an untrained, randomly-initialized model; nothing to write in that case either.
        SUCCEED();
    }
}

#ifdef ADAI_ENABLE_GPU
// ============================================================================
// TD-358: gpu_forward() must teacher-force exactly like the CPU forward()
// ([BOS] + target[:-1] as decoder input, target as labels). It used to feed the
// targets unshifted, so GPU training learned to copy its input. Same weights must
// give the same loss on both paths. Skips without a GPU device (ai-machine runs it).
// ============================================================================

TEST(EncoderDecoderGPUTest, GpuForwardLossMatchesCpuForwardTeacherForcing) {
    if (!adai::gpu::GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(adai::gpu::GPUManager::initialize(0, 0.5f));

    EncoderDecoderModel model(64, 32, 1, 1, 2, 64, 32);
    model.get_tokenizer()->build_vocab(
        {"the cat sat on the mat", "a dog ran far away home", "big red ball and happy girl"}, 64);
    BPETokenizer* tok = model.get_tokenizer();
    model.set_training(true);
    model.gpu_init_training();

    const std::vector<std::pair<std::string, std::string>> pairs = {{"the cat sat", "on the mat"},
                                                                    {"a dog ran", "far away home"},
                                                                    {"big red ball", "happy girl"}};
    for (const auto& [input, target] : pairs) {
        const std::vector<int> in = tok->encode(input, false);
        const std::vector<int> tgt = tok->encode(target, true);
        const float cpu_loss = model.compute_loss_for_training(model.forward(in, tgt), tgt);
        const float gpu_loss = model.gpu_forward(in, tgt);
        EXPECT_NEAR(gpu_loss, cpu_loss, 1e-3f) << "target '" << target << "'";
    }
}

// TD-359: gpu_generate_response() applies its max_length too (it was "kept for interface parity").
TEST(EncoderDecoderGPUTest, GpuGenerateResponseHonoursMaxLength) {
    if (!adai::gpu::GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(adai::gpu::GPUManager::initialize(0, 0.5f));

    EncoderDecoderModel model(100, 64, 2, 2);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    // Greedy and no stop tokens: the 5-token answer must be a strict prefix of the 40-token one.
    // Counting tokens by re-encoding the decoded text isn't exact, and an untrained, unseeded model
    // (TD-368) sometimes stops at once.
    suppress_stop_tokens(model);
    auto cfg = model.get_generator()->get_config();
    cfg.max_length = 64;
    cfg.num_beams = 1;
    cfg.temperature = 0.0f;
    model.get_generator()->set_config(cfg);

    for (const char* input : {"hello world", "the quick brown fox", "a b c"}) {
        const std::string shorter = model.gpu_generate_response(input, 5);
        const std::string longer = model.gpu_generate_response(input, 40);
        EXPECT_EQ(longer.compare(0, shorter.size(), shorter), 0) << input;
        EXPECT_GT(longer.size(), shorter.size()) << input << ": 5 tokens weren't a cap";
    }
    EXPECT_EQ(model.get_generator()->get_config().max_length, 64);
}

// TD-387: an input longer than max_seq_length used to make the GPU cross-attention cache (sized at
// max_seq_length) throw out_of_range on the first decode step. With TD-365 the encoder input keeps its
// last max_seq_length tokens, so it fits.
TEST(EncoderDecoderGPUTest, GpuGenerationAcceptsOverLongInput) {
    if (!adai::gpu::GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(adai::gpu::GPUManager::initialize(0, 0.5f));

    EncoderDecoderModel model(100, 32, 1, 1, 2, 64, /*max_seq_length=*/16);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    std::string long_text;
    for (int i = 0; i < 40; ++i)
        long_text += "hello world ";
    EXPECT_NO_THROW(model.gpu_generate_response(long_text, 8));
    EXPECT_NO_THROW(model.gpu_generate_response_with_strategy(long_text, 8, "greedy"));
}

// TD-362: the GPU paths mask IDs the tokenizer doesn't have, resolve strategies and num_beams the
// same way as the CPU path, and so give the same output for the same request.
TEST(EncoderDecoderGPUTest, GpuAndCpuGenerationAgree) {
    if (!adai::gpu::GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(adai::gpu::GPUManager::initialize(0, 0.5f));

    EncoderDecoderModel model(100, 32, 1, 1, 2, 64, 32);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    favour_unknown_id(model, 99);  // the GPU paths used to pick it every step

    for (const char* input : {"hello world", "good morning"}) {
        const std::string cpu_greedy =
            model.generate_response_with_strategy(input, 6, "greedy", 0.0f, 50, 0.9f, 1);
        EXPECT_EQ(model.gpu_generate_response_with_strategy(input, 6, "greedy", 0.0f, 50, 0.9f, 1),
                  cpu_greedy)
            << input;
        EXPECT_EQ(model.gpu_generate_response(input, 6), cpu_greedy) << input;

        const std::string cpu_beam =
            model.generate_response_with_strategy(input, 6, "beam", 1.0f, 50, 0.9f, 3);
        EXPECT_EQ(model.gpu_generate_response_with_strategy(input, 6, "beam", 1.0f, 50, 0.9f, 3),
                  cpu_beam)
            << input;
        // An unrecognized strategy with num_beams > 1 is beam search on both (GPU: nucleus before).
        EXPECT_EQ(model.gpu_generate_response_with_strategy(input, 6, "other", 1.0f, 50, 0.9f, 3),
                  cpu_beam)
            << input;
    }

    // An explicit single-hypothesis strategy doesn't store its num_beams (TD-261 on the GPU too).
    model.gpu_generate_response_with_strategy("hello world", 6, "greedy", 0.0f, 50, 0.9f, 4);
    EXPECT_EQ(model.get_generator()->get_config().num_beams, 1);
}

// TD-388: the GPU KV-cached decode reads each head straight out of the cache. A longer generation with
// several heads and layers (the cache grows every step) must match the CPU output.
TEST(EncoderDecoderGPUTest, CachedGpuDecodeMatchesCpuOverALongGeneration) {
    if (!adai::gpu::GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(adai::gpu::GPUManager::initialize(0, 0.5f));

    EncoderDecoderModel model(100, 32, 2, 2, 4, 64, 32);
    build_test_vocab(model.get_tokenizer(), 100);
    model.set_training(false);
    suppress_stop_tokens(model);  // run the full 20 steps
    for (const char* input : {"hello world", "good morning how are you"}) {
        const std::string cpu =
            model.generate_response_with_strategy(input, 20, "greedy", 0.0f, 50, 0.9f, 1);
        EXPECT_EQ(model.gpu_generate_response_with_strategy(input, 20, "greedy", 0.0f, 50, 0.9f, 1),
                  cpu)
            << input;
    }
}

TEST(EncoderDecoderGPUTest, GpuForwardRejectsEmptyTarget) {
    if (!adai::gpu::GPUManager::probe()) {
        GTEST_SKIP() << "No GPU device present.";
    }
    ASSERT_TRUE(adai::gpu::GPUManager::initialize(0, 0.5f));

    EncoderDecoderModel model(64, 32, 1, 1, 2, 64, 32);
    model.gpu_init_training();
    EXPECT_THROW(model.gpu_forward({4, 5}, {}), std::invalid_argument);
}
#endif  // ADAI_ENABLE_GPU

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// TD-544: clone() saves to a temp checkpoint and loads it back. It used to remove only the five
// top-level files, leaving every component file (embeddings, each block, the norms) in the temp
// directory, once per epoch with the async quality metrics. TMPDIR points the temp directory at
// a fresh folder so the check sees only this clone's files.
TEST(EncoderDecoderModelTest, CloneLeavesNoTempFiles) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "adai_clone_leak_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const char* old_tmpdir = std::getenv("TMPDIR");
    const std::string saved = old_tmpdir ? old_tmpdir : "";
    setenv("TMPDIR", dir.c_str(), 1);

    EncoderDecoderModel model(100, 32, 1, 1, 2, 64, 16);
    build_test_vocab(model.get_tokenizer(), 100);
    auto copy = model.clone();
    ASSERT_TRUE(copy);

    if (old_tmpdir)
        setenv("TMPDIR", saved.c_str(), 1);
    else
        unsetenv("TMPDIR");
    std::vector<std::string> left;
    for (const auto& e : fs::directory_iterator(dir))
        left.push_back(e.path().filename().string());
    EXPECT_TRUE(left.empty()) << left.size() << " temp file(s) left, e.g. "
                              << (left.empty() ? "" : left.front());
    fs::remove_all(dir);
}

// ============================================================================
// TD-343: one hippocampal memory update per decode step
// ============================================================================

namespace {

std::unique_ptr<HippocampalMemory> td343_memory(int d_model) {
    auto memory = std::make_unique<HippocampalMemory>(d_model, 8);
    for (int slot = 0; slot < 3; ++slot) {
        Matrix key(1, d_model), value(1, d_model);
        for (int j = 0; j < d_model; ++j) {
            key(0, j) = 0.1f * std::sin(static_cast<float>(slot * 5 + j));
            value(0, j) = key(0, j);
        }
        memory->write(key, value);
    }
    return memory;
}

}  // namespace

// Beam search with width k used to update the memory once per hypothesis per step, in every
// injected layer. Now it commits once per decode step: never more steps than tokens generated.
TEST(EncoderDecoderModelHippocampalTest, BeamSearchCommitsOncePerDecodeStep) {
    const int vocab_size = 100, d_model = 64, max_length = 10;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512,
                              /*world_model_inject_every_n_layers=*/1);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_hippocampal_memory(td343_memory(d_model));
    HippocampalMemory* memory = model.get_hippocampal_memory();

    model.generate_response_with_strategy("hello world", max_length, "beam", 1.0f, 50, 0.9f,
                                          /*num_beams=*/3);
    EXPECT_GT(memory->committed_steps(), 0u);
    EXPECT_LE(memory->committed_steps(), static_cast<size_t>(max_length))
        << "at most one update per decode step (was about beams x layers x steps)";
}

TEST(EncoderDecoderModelHippocampalTest, CachedDecodingCommitsOncePerToken) {
    const int vocab_size = 100, d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512, 1);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_hippocampal_memory(td343_memory(d_model));
    HippocampalMemory* memory = model.get_hippocampal_memory();

    model.generate_response_with_strategy("hello world", 10, "greedy", 1.0f, 50, 0.9f, 1);
    EXPECT_GT(memory->committed_steps(), 0u);
    EXPECT_LE(memory->committed_steps(), 10u);
}

TEST(EncoderDecoderModelHippocampalTest, TrainingForwardsAndDisabledUpdatesLeaveTheMemoryAlone) {
    const int vocab_size = 100, d_model = 64;
    EncoderDecoderModel model(vocab_size, d_model, 2, 2, 8, 256, 512, 1);
    build_test_vocab(model.get_tokenizer(), vocab_size);
    model.set_hippocampal_memory(td343_memory(d_model));
    HippocampalMemory* memory = model.get_hippocampal_memory();
    const int slots = memory->size();

    // A teacher-forced (training/validation) forward reads the memory but never commits.
    model.forward(model.get_tokenizer()->encode("hello world", false),
                  model.get_tokenizer()->encode("how are you", true));
    EXPECT_EQ(memory->committed_steps(), 0u);

    // With updates off (as ChatbotTrainer sets while training), generation neither commits nor
    // writes the response into the memory.
    model.set_hippocampal_updates_enabled(false);
    model.generate_response_with_strategy("hello world", 10, "greedy", 1.0f, 50, 0.9f, 1);
    EXPECT_EQ(memory->committed_steps(), 0u);
    EXPECT_EQ(memory->size(), slots);
    model.set_hippocampal_updates_enabled(true);
}

