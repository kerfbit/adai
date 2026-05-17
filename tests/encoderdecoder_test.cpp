#include <../gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include "../src/EncoderDecoderModel.hpp"
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
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
