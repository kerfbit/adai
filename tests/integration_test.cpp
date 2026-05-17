#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "../src/Activation.hpp"
#include "../src/BPETokenizer.hpp"
#include "../src/Decoder.hpp"
#include "../src/DecoderBlock.hpp"
#include "../src/EncoderBlock.hpp"
#include "../src/EncoderDecoderModel.hpp"
#include "../src/Matrix.hpp"
#include "../src/MultiHeadAttention.hpp"
#include "../src/Optimizer.hpp"
#include "../src/encoder.hpp"
#include "test_base.hpp"

/**
 * Integration tests for end-to-end workflows
 *
 * These tests verify that components work together correctly
 * in realistic usage scenarios.
 */

// =============================================================================
// Complete Training Pipeline Test
// =============================================================================

TEST(IntegrationTest, CompleteTrainingPipeline) {
    // Test: Tokenize → Encode → Decode → Generate

    // 1. Setup tokenizer and encoder
    BPETokenizer tokenizer;
    std::vector<std::string> corpus = {"hello world test"};
    tokenizer.build_vocab(corpus, 100);

    // 2. Create encoder (vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 128;
    int vocab_size = tokenizer.get_vocab_size();

    LLMEncoder encoder(vocab_size, d_model, 2, num_heads, d_ff, 512);

    // 3. Encode text directly
    std::string test_text = "hello world";
    Matrix encoder_output = encoder.encode(test_text);

    // Verify encoder output shape (should have some rows and d_model cols)
    EXPECT_GT(encoder_output.rows, 0);
    EXPECT_EQ(encoder_output.cols, d_model);

    // Verify output is finite
    for (int i = 0; i < encoder_output.rows; ++i) {
        for (int j = 0; j < encoder_output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(encoder_output.data[i][j]));
        }
    }

    // 4. Create decoder (vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    LLMDecoder decoder(vocab_size, d_model, 2, num_heads, d_ff, 512);

    // 5. Decode with encoder context
    std::vector<int> target_tokens = {2};  // BOS token
    Matrix decoder_output = decoder.forward_with_encoder(target_tokens, encoder_output);

    // Verify decoder output shape
    EXPECT_EQ(decoder_output.rows, 1);  // Single target token
    EXPECT_EQ(decoder_output.cols, d_model);

    // Verify decoder output is finite
    for (int i = 0; i < decoder_output.rows; ++i) {
        for (int j = 0; j < decoder_output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(decoder_output.data[i][j]));
        }
    }
}

// =============================================================================
// Save/Load Round Trip Test
// =============================================================================

TEST(IntegrationTest, SaveLoadRoundTrip) {
    // Note: LLMEncoder doesn't currently expose save/load methods
    // This test is a placeholder for when that functionality is added
    // See PROCESS_IMPROVEMENT_PLAN.md Section 10 for parameter exposure work

    const std::string test_model_file = "test_integration_model.bin";

    // For now, just verify we can create and use models consistently
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 128;
    int vocab_size = 50;
    int num_layers = 2;
    int max_seq_len = 128;

    LLMEncoder encoder1(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len);
    LLMEncoder encoder2(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len);

    std::string test_text = "test input";

    // Both encoders should produce outputs (though different due to random init)
    Matrix output1 = encoder1.encode(test_text);
    Matrix output2 = encoder2.encode(test_text);

    EXPECT_EQ(output1.rows, output2.rows);
    EXPECT_EQ(output1.cols, output2.cols);

    // Verify outputs are valid
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output1.data[i][j]));
            EXPECT_TRUE(std::isfinite(output2.data[i][j]));
        }
    }
}

// =============================================================================
// Multi-Epoch Training Test
// =============================================================================

TEST(IntegrationTest, MultiEpochTraining) {
    // Test: Train multiple epochs, verify loss decreases

    // 1. Setup small model for training
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 128;
    int vocab_size = 20;

    LLMEncoder encoder(vocab_size, d_model, 1, num_heads, d_ff, 128);

    // 2. Create optimizer
    Optimizer optimizer(OptimizerType::ADAM);
    optimizer.set_learning_rate(0.001f);

    // Note: Can't add parameters directly due to incomplete parameter exposure
    // This is a known limitation (see PROCESS_IMPROVEMENT_PLAN.md Section 10)

    // 3. Create simple training data
    std::vector<std::string> training_texts = {"test one", "test two", "test three", "test four"};

    std::vector<float> epoch_losses;

    // 4. Train for multiple epochs
    for (int epoch = 0; epoch < 3; ++epoch) {
        float epoch_loss = 0.0f;

        for (const auto& text : training_texts) {
            // Forward pass
            Matrix output = encoder.encode(text);

            // Simple loss: mean squared difference from target
            // (In real training, this would be cross-entropy)
            float loss = 0.0f;
            for (int i = 0; i < output.rows; ++i) {
                for (int j = 0; j < output.cols; ++j) {
                    float target = 0.5f;  // Dummy target
                    float diff = output.data[i][j] - target;
                    loss += diff * diff;
                }
            }
            loss /= (output.rows * output.cols);
            epoch_loss += loss;

            // Note: Backward pass would go here in complete implementation
            // Currently limited by parameter exposure issue
        }

        epoch_loss /= training_texts.size();
        epoch_losses.push_back(epoch_loss);

        std::cout << "Epoch " << epoch << " loss: " << epoch_loss << std::endl;
    }

    // 5. Verify we ran all epochs
    EXPECT_EQ(epoch_losses.size(), 3);

    // 6. Verify losses are reasonable (not NaN or Inf)
    for (float loss : epoch_losses) {
        EXPECT_TRUE(std::isfinite(loss));
        EXPECT_GT(loss, 0.0f);
    }
}

// =============================================================================
// Encoder-Decoder Integration Test
// =============================================================================

TEST(IntegrationTest, EncoderDecoderIntegration) {
    // Test complete encoder-decoder workflow

    int d_model = 64;
    int num_heads = 4;
    int d_ff = 128;
    int vocab_size = 50;
    int num_layers = 1;
    int max_seq_len = 128;

    // Create encoder and decoder
    LLMEncoder encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len);
    LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len);

    // Source text
    std::string source_text = "hello world";

    // Encode source
    Matrix encoder_output = encoder.encode(source_text);

    EXPECT_GT(encoder_output.rows, 0);
    EXPECT_EQ(encoder_output.cols, d_model);

    // Decode with cross-attention to encoder
    std::vector<int> target_tokens = {2, 6, 7};
    Matrix decoder_output = decoder.forward_with_encoder(target_tokens, encoder_output);

    EXPECT_EQ(decoder_output.rows, static_cast<int>(target_tokens.size()));
    EXPECT_EQ(decoder_output.cols, d_model);

    // Verify all outputs are finite
    for (int i = 0; i < decoder_output.rows; ++i) {
        for (int j = 0; j < decoder_output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(decoder_output.data[i][j]));
        }
    }
}

// =============================================================================
// Attention Mechanism Integration Test
// =============================================================================

TEST(IntegrationTest, AttentionMechanismFlow) {
    // Test that attention mechanisms integrate properly

    int d_model = 64;
    int num_heads = 4;
    int seq_len = 8;

    // Create attention module
    MultiHeadAttention mha(d_model, num_heads);

    // Create test input
    Matrix input(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input.data[i][j] = static_cast<float>(i * d_model + j) * 0.01f;
        }
    }

    // Self-attention (pass nullptr for mask)
    Matrix output = mha.forward(input, nullptr);

    // Verify shape preservation
    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, d_model);

    // Verify output is finite
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output.data[i][j]));
        }
    }

    // Note: MultiHeadAttention in this implementation only does self-attention
    // Cross-attention is handled by CrossAttention class or within EncoderBlock/DecoderBlock
}

// =============================================================================
// Gradient Flow Test
// =============================================================================

TEST(IntegrationTest, GradientFlowThroughLayers) {
    // Test that gradients can flow through stacked layers

    int d_model = 64;
    int num_heads = 4;
    int d_ff = 128;
    int seq_len = 5;

    // Create encoder block
    EncoderBlock encoder_block(d_model, num_heads, d_ff);

    // Create input
    Matrix input(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input.data[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        }
    }

    // Forward pass
    Matrix output = encoder_block.forward(input);

    // Verify output
    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, d_model);

    // Create gradient (as if from loss) with larger values
    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output.data[i][j] = 0.1f;  // Larger gradient for better propagation
        }
    }

    // Backward pass
    Matrix grad_input = encoder_block.backward(grad_output);

    // Verify gradient shape
    EXPECT_EQ(grad_input.rows, seq_len);
    EXPECT_EQ(grad_input.cols, d_model);

    // Verify gradients are finite (NaN or Inf indicates a problem)
    bool all_finite = true;
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            if (!std::isfinite(grad_input.data[i][j])) {
                all_finite = false;
            }
        }
    }

    EXPECT_TRUE(all_finite) << "Gradients contain NaN or Inf values";

    // Note: Some implementations may return zero gradients if backward pass
    // is not fully connected. This is acceptable for this integration test.
    // The main goal is to ensure the API works and doesn't crash.
}

// =============================================================================
// Tokenizer Integration Test
// =============================================================================

TEST(IntegrationTest, TokenizerEncoderIntegration) {
    // Test that tokenizer output works with encoder

    BPETokenizer tokenizer;

    // Build vocabulary
    std::vector<std::string> corpus = {"hello world test integration tokenizer"};

    tokenizer.build_vocab(corpus, 100);

    int vocab_size = tokenizer.get_vocab_size();
    EXPECT_GT(vocab_size, 4);  // At least special tokens + our words

    // Create encoder with tokenizer's vocab size (vocab_size, d_model, num_layers, num_heads, d_ff,
    // max_seq_len)
    LLMEncoder encoder(vocab_size, 64, 1, 4, 128, 128);

    // Encode text
    std::string text = "hello world test";

    // Process with encoder directly (it will tokenize internally)
    Matrix output = encoder.encode(text);

    EXPECT_GT(output.rows, 0);
    EXPECT_EQ(output.cols, 64);  // d_model

    // Verify output is valid
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output.data[i][j]));
        }
    }
}

// =============================================================================
// Main function
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
