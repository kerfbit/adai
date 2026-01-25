#include "../src/Decoder.hpp"
#include <../gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "../src/Matrix.hpp"

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

float compute_gradient_norm(const Matrix& grad) {
    float sum = 0.0f;
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            sum += grad(i, j) * grad(i, j);
        }
    }
    return std::sqrt(sum);
}

Matrix create_test_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

bool is_causal_mask(const Matrix& mask) {
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            if (j > i && mask(i, j) != 0.0f)
                return false;
            if (j <= i && mask(i, j) != 1.0f)
                return false;
        }
    }
    return true;
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(DecoderTest, ConstructorBasic) {
    // Test basic construction with default parameters
    int vocab_size = 100;
    int d_model = 64;

    EXPECT_NO_THROW({ LLMDecoder decoder(vocab_size, d_model); });
}

TEST(DecoderTest, ConstructorWithAllParameters) {
    // Test construction with all parameters specified
    int vocab_size = 500;
    int d_model = 128;
    int num_layers = 3;
    int num_heads = 4;
    int d_ff = 512;
    int max_seq_length = 256;

    EXPECT_NO_THROW(
        { LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length); });
}

TEST(DecoderTest, ConstructorInitializesComponents) {
    // Test that constructor properly initializes all components
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;
    int num_heads = 4;
    int d_ff = 256;
    int max_seq_length = 128;

    LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length);

    // Verify configuration getters
    EXPECT_EQ(decoder.get_vocab_size(), vocab_size);
    EXPECT_EQ(decoder.get_d_model(), d_model);
    EXPECT_EQ(decoder.get_num_layers(), num_layers);
    EXPECT_EQ(decoder.get_max_seq_length(), max_seq_length);
}

TEST(DecoderTest, ConstructorWithSmallModel) {
    // Test with minimal model dimensions
    int vocab_size = 50;
    int d_model = 32;
    int num_layers = 1;
    int num_heads = 2;
    int d_ff = 128;
    int max_seq_length = 32;

    EXPECT_NO_THROW(
        { LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length); });
}

TEST(DecoderTest, ConstructorWithLargeModel) {
    // Test with larger model dimensions
    int vocab_size = 1000;
    int d_model = 256;
    int num_layers = 4;
    int num_heads = 8;
    int d_ff = 1024;
    int max_seq_length = 512;

    EXPECT_NO_THROW(
        { LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length); });
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST(DecoderTest, GetTokenEmbedding) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Verify token embedding is accessible
    TokenEmbedding* embedding = decoder.get_token_embedding();
    EXPECT_NE(embedding, nullptr);
}

TEST(DecoderTest, GetDecoderBlock) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 3;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    // Verify decoder blocks are accessible
    for (int i = 0; i < num_layers; ++i) {
        DecoderBlock* block = decoder.get_decoder_block(i);
        EXPECT_NE(block, nullptr);
    }
}

TEST(DecoderTest, GetDecoderBlockOutOfRange) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 3;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    // Test out of range access
    EXPECT_THROW(decoder.get_decoder_block(-1), std::out_of_range);
    EXPECT_THROW(decoder.get_decoder_block(num_layers), std::out_of_range);
}

TEST(DecoderTest, GetLastOutputEmpty) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Before any forward pass, last output should be empty
    Matrix last_output = decoder.get_last_output();
    EXPECT_EQ(last_output.rows, 0);
    EXPECT_EQ(last_output.cols, 0);
}

// ============================================================================
// Forward Pass Tests (Decoder-Only Mode)
// ============================================================================

TEST(DecoderTest, ForwardBasic) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    // Create simple token sequence
    std::vector<int> token_ids = {1, 5, 10};

    EXPECT_NO_THROW({
        Matrix output = decoder.forward(token_ids);
        EXPECT_EQ(output.rows, token_ids.size());
        EXPECT_EQ(output.cols, d_model);
    });
}

TEST(DecoderTest, ForwardOutputDimensions) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    // Test various sequence lengths
    std::vector<int> lengths = {1, 3, 5, 10};

    for (int len : lengths) {
        std::vector<int> token_ids(len);
        for (int i = 0; i < len; ++i) {
            token_ids[i] = i % vocab_size;
        }

        Matrix output = decoder.forward(token_ids);
        EXPECT_EQ(output.rows, len);
        EXPECT_EQ(output.cols, d_model);
    }
}

TEST(DecoderTest, ForwardSingleToken) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {42};

    Matrix output = decoder.forward(token_ids);
    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderTest, ForwardLongSequence) {
    int vocab_size = 100;
    int d_model = 64;
    int max_seq_length = 128;
    LLMDecoder decoder(vocab_size, d_model, 2, 4, 256, max_seq_length);

    // Create sequence close to max length
    std::vector<int> token_ids(50);
    for (int i = 0; i < 50; ++i) {
        token_ids[i] = i % vocab_size;
    }

    Matrix output = decoder.forward(token_ids);
    EXPECT_EQ(output.rows, 50);
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderTest, ForwardMultipleCalls) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Multiple forward passes should work
    std::vector<int> tokens1 = {1, 2, 3};
    std::vector<int> tokens2 = {4, 5};
    std::vector<int> tokens3 = {6, 7, 8, 9};

    EXPECT_NO_THROW({
        decoder.forward(tokens1);
        decoder.forward(tokens2);
        decoder.forward(tokens3);
    });
}

TEST(DecoderTest, ForwardDeterministic) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 5, 10, 15};

    // Same input should produce same output (deterministic)
    Matrix output1 = decoder.forward(token_ids);
    Matrix output2 = decoder.forward(token_ids);

    EXPECT_TRUE(matrices_equal(output1, output2, 1e-5f));
}

// ============================================================================
// Forward Pass Tests (Encoder-Decoder Mode)
// ============================================================================

TEST(DecoderTest, ForwardWithEncoderBasic) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix encoder_output(5, d_model);  // Encoder sequence length = 5

    // Fill with dummy values
    for (int i = 0; i < encoder_output.rows; ++i) {
        for (int j = 0; j < encoder_output.cols; ++j) {
            encoder_output(i, j) = (i + j) * 0.1f;
        }
    }

    EXPECT_NO_THROW({
        Matrix output = decoder.forward_with_encoder(token_ids, encoder_output);
        EXPECT_EQ(output.rows, token_ids.size());
        EXPECT_EQ(output.cols, d_model);
    });
}

TEST(DecoderTest, ForwardWithEncoderDimensions) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3, 4};
    Matrix encoder_output(8, d_model);

    Matrix output = decoder.forward_with_encoder(token_ids, encoder_output);
    EXPECT_EQ(output.rows, token_ids.size());
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderTest, ForwardWithEncoderVsDecoderOnly) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix encoder_output(5, d_model);

    // Fill encoder output
    for (int i = 0; i < encoder_output.rows; ++i) {
        for (int j = 0; j < encoder_output.cols; ++j) {
            encoder_output(i, j) = (i + j) * 0.1f;
        }
    }

    Matrix output_decoder_only = decoder.forward(token_ids);
    Matrix output_with_encoder = decoder.forward_with_encoder(token_ids, encoder_output);

    // Outputs should be different (encoder provides additional context)
    EXPECT_EQ(output_decoder_only.rows, output_with_encoder.rows);
    EXPECT_EQ(output_decoder_only.cols, output_with_encoder.cols);
    // Not checking equality since cross-attention changes output
}

TEST(DecoderTest, ForwardWithEmptyEncoder) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix empty_encoder(0, 0);  // Empty encoder output

    // Should handle empty encoder gracefully (like decoder-only mode)
    EXPECT_NO_THROW({
        Matrix output = decoder.forward_with_encoder(token_ids, empty_encoder);
        EXPECT_EQ(output.rows, token_ids.size());
        EXPECT_EQ(output.cols, d_model);
    });
}

// ============================================================================
// Forward Pass with Custom Mask Tests
// ============================================================================

TEST(DecoderTest, ForwardWithMaskBasic) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3, 4};
    Matrix causal_mask = create_test_causal_mask(token_ids.size());

    EXPECT_NO_THROW({
        Matrix output = decoder.forward_with_mask(token_ids, causal_mask);
        EXPECT_EQ(output.rows, token_ids.size());
        EXPECT_EQ(output.cols, d_model);
    });
}

TEST(DecoderTest, ForwardWithMaskAndEncoder) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix causal_mask = create_test_causal_mask(token_ids.size());
    Matrix encoder_output(5, d_model);

    EXPECT_NO_THROW({
        Matrix output = decoder.forward_with_mask(token_ids, causal_mask, &encoder_output);
        EXPECT_EQ(output.rows, token_ids.size());
        EXPECT_EQ(output.cols, d_model);
    });
}

TEST(DecoderTest, ForwardWithMaskNullEncoder) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix causal_mask = create_test_causal_mask(token_ids.size());

    // Null encoder pointer
    EXPECT_NO_THROW({
        Matrix output = decoder.forward_with_mask(token_ids, causal_mask, nullptr);
        EXPECT_EQ(output.rows, token_ids.size());
        EXPECT_EQ(output.cols, d_model);
    });
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST(DecoderTest, BackwardBasic) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Forward pass first
    std::vector<int> token_ids = {1, 2, 3};
    Matrix output = decoder.forward(token_ids);

    // Create gradient
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Backward should not throw
    EXPECT_NO_THROW({ decoder.backward(grad_output); });
}

TEST(DecoderTest, BackwardAfterForwardWithEncoder) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix encoder_output(5, d_model);

    Matrix output = decoder.forward_with_encoder(token_ids, encoder_output);

    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output(i, j) = 0.05f;
        }
    }

    EXPECT_NO_THROW({ decoder.backward(grad_output); });
}

TEST(DecoderTest, BackwardMultipleTimes) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3};

    for (int iter = 0; iter < 3; ++iter) {
        Matrix output = decoder.forward(token_ids);
        Matrix grad_output(output.rows, output.cols);

        for (int i = 0; i < grad_output.rows; ++i) {
            for (int j = 0; j < grad_output.cols; ++j) {
                grad_output(i, j) = 0.1f * (iter + 1);
            }
        }

        EXPECT_NO_THROW({ decoder.backward(grad_output); });
    }
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST(DecoderTest, SetTrainingMode) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Set training mode
    EXPECT_NO_THROW({
        decoder.set_training(true);
        decoder.set_training(false);
    });
}

TEST(DecoderTest, SetLearningRate) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    EXPECT_NO_THROW({
        decoder.set_learning_rate(0.001f);
        decoder.set_learning_rate(0.0001f);
        decoder.set_learning_rate(0.01f);
    });
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(DecoderTest, UpdateWeightsBasic) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Forward + backward
    std::vector<int> token_ids = {1, 2, 3};
    Matrix output = decoder.forward(token_ids);

    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }
    decoder.backward(grad_output);

    // Update weights
    float lr = 0.001f;
    EXPECT_NO_THROW({ decoder.update_weights(lr); });
}

TEST(DecoderTest, UpdateWeightsMultipleTimes) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3};
    float lr = 0.001f;

    for (int iter = 0; iter < 5; ++iter) {
        Matrix output = decoder.forward(token_ids);
        Matrix grad_output(output.rows, output.cols);

        for (int i = 0; i < grad_output.rows; ++i) {
            for (int j = 0; j < grad_output.cols; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        decoder.backward(grad_output);

        EXPECT_NO_THROW({ decoder.update_weights(lr); });
    }
}

TEST(DecoderTest, ZeroGradBasic) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    EXPECT_NO_THROW({ decoder.zero_grad(); });
}

TEST(DecoderTest, ZeroGradAfterBackward) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix output = decoder.forward(token_ids);

    Matrix grad_output(output.rows, output.cols);
    decoder.backward(grad_output);

    EXPECT_NO_THROW({ decoder.zero_grad(); });
}

// ============================================================================
// Training Loop Tests
// ============================================================================

TEST(DecoderTest, SimpleTrainingLoop) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    decoder.set_training(true);
    float lr = 0.001f;

    std::vector<int> token_ids = {1, 5, 10, 15};

    // Run a few training iterations
    for (int epoch = 0; epoch < 3; ++epoch) {
        decoder.zero_grad();

        Matrix output = decoder.forward(token_ids);

        // Create dummy gradient
        Matrix grad_output(output.rows, output.cols);
        for (int i = 0; i < grad_output.rows; ++i) {
            for (int j = 0; j < grad_output.cols; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        decoder.backward(grad_output);
        decoder.update_weights(lr);
    }

    // Should complete without errors
    SUCCEED();
}

TEST(DecoderTest, TrainingWithEncoderOutput) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    decoder.set_training(true);
    float lr = 0.001f;

    std::vector<int> token_ids = {1, 2, 3};
    Matrix encoder_output(5, d_model);

    for (int epoch = 0; epoch < 3; ++epoch) {
        decoder.zero_grad();

        Matrix output = decoder.forward_with_encoder(token_ids, encoder_output);

        Matrix grad_output(output.rows, output.cols);
        for (int i = 0; i < grad_output.rows; ++i) {
            for (int j = 0; j < grad_output.cols; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        decoder.backward(grad_output);
        decoder.update_weights(lr);
    }

    SUCCEED();
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(DecoderTest, SaveWeights) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    std::string filepath = "test_decoder_weights.bin";

    // Save should work (even if incomplete)
    EXPECT_NO_THROW({ decoder.save_weights(filepath); });

    // Clean up
    std::remove(filepath.c_str());
}

TEST(DecoderTest, LoadWeights) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;

    // Create and save decoder
    LLMDecoder decoder1(vocab_size, d_model, num_layers);
    std::string filepath = "test_decoder_weights.bin";
    decoder1.save_weights(filepath);

    // Load into new decoder
    LLMDecoder decoder2(vocab_size, d_model, num_layers);
    EXPECT_NO_THROW({ decoder2.load_weights(filepath); });

    // Clean up
    std::remove(filepath.c_str());
}

TEST(DecoderTest, LoadWeightsMismatchedArchitecture) {
    int vocab_size1 = 100;
    int d_model1 = 64;
    int num_layers1 = 2;

    int vocab_size2 = 200;  // Different vocab size
    int d_model2 = 64;
    int num_layers2 = 2;

    // Create and save decoder
    LLMDecoder decoder1(vocab_size1, d_model1, num_layers1);
    std::string filepath = "test_decoder_mismatch.bin";
    decoder1.save_weights(filepath);

    // Try to load into mismatched decoder
    LLMDecoder decoder2(vocab_size2, d_model2, num_layers2);
    EXPECT_THROW({ decoder2.load_weights(filepath); }, std::runtime_error);

    // Clean up
    std::remove(filepath.c_str());
}

TEST(DecoderTest, SaveLoadRoundTrip) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;
    int num_heads = 4;
    int d_ff = 256;
    int max_seq_length = 128;

    LLMDecoder decoder1(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length);
    std::string filepath = "test_decoder_roundtrip.bin";

    // Save
    decoder1.save_weights(filepath);

    // Load into new decoder with same architecture
    LLMDecoder decoder2(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length);
    decoder2.load_weights(filepath);

    // Verify configuration
    EXPECT_EQ(decoder2.get_vocab_size(), vocab_size);
    EXPECT_EQ(decoder2.get_d_model(), d_model);
    EXPECT_EQ(decoder2.get_num_layers(), num_layers);
    EXPECT_EQ(decoder2.get_max_seq_length(), max_seq_length);

    // Clean up
    std::remove(filepath.c_str());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(DecoderTest, EmptyTokenSequence) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    std::vector<int> empty_tokens;

    // Empty sequence should be handled gracefully or throw
    // Behavior depends on implementation
    // For now, just ensure it doesn't crash unpredictably
}

TEST(DecoderTest, VeryLongSequence) {
    int vocab_size = 100;
    int d_model = 64;
    int max_seq_length = 256;
    LLMDecoder decoder(vocab_size, d_model, 2, 4, 256, max_seq_length);

    // Create sequence at max length
    std::vector<int> long_tokens(max_seq_length);
    for (int i = 0; i < max_seq_length; ++i) {
        long_tokens[i] = i % vocab_size;
    }

    EXPECT_NO_THROW({
        Matrix output = decoder.forward(long_tokens);
        EXPECT_EQ(output.rows, max_seq_length);
    });
}

TEST(DecoderTest, RepeatedTokens) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // All same token
    std::vector<int> repeated_tokens = {5, 5, 5, 5, 5};

    EXPECT_NO_THROW({
        Matrix output = decoder.forward(repeated_tokens);
        EXPECT_EQ(output.rows, repeated_tokens.size());
    });
}

TEST(DecoderTest, SequentialTokens) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model);

    // Sequential tokens
    std::vector<int> sequential;
    for (int i = 0; i < 20; ++i) {
        sequential.push_back(i);
    }

    EXPECT_NO_THROW({
        Matrix output = decoder.forward(sequential);
        EXPECT_EQ(output.rows, sequential.size());
    });
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(DecoderTest, MultipleLayersConsistency) {
    int vocab_size = 100;
    int d_model = 64;

    // Test different number of layers produce different outputs
    LLMDecoder decoder1(vocab_size, d_model, 1);
    LLMDecoder decoder2(vocab_size, d_model, 3);

    std::vector<int> token_ids = {1, 2, 3, 4};

    Matrix output1 = decoder1.forward(token_ids);
    Matrix output2 = decoder2.forward(token_ids);

    // Same dimensions
    EXPECT_EQ(output1.rows, output2.rows);
    EXPECT_EQ(output1.cols, output2.cols);

    // But likely different values (different number of layers)
    // Not checking actual values since they're random
}

TEST(DecoderTest, DifferentModelDimensions) {
    int vocab_size = 100;

    LLMDecoder decoder_small(vocab_size, 32);
    LLMDecoder decoder_large(vocab_size, 128);

    std::vector<int> token_ids = {1, 2, 3};

    Matrix output_small = decoder_small.forward(token_ids);
    Matrix output_large = decoder_large.forward(token_ids);

    EXPECT_EQ(output_small.rows, token_ids.size());
    EXPECT_EQ(output_small.cols, 32);

    EXPECT_EQ(output_large.rows, token_ids.size());
    EXPECT_EQ(output_large.cols, 128);
}

TEST(DecoderTest, GetLastOutputAfterForward) {
    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 2;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix output = decoder.forward(token_ids);

    Matrix last_output = decoder.get_last_output();

    // Last output should match the forward output
    EXPECT_EQ(last_output.rows, output.rows);
    EXPECT_EQ(last_output.cols, output.cols);
}

TEST(DecoderTest, AutoregressiveGenerationPattern) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    // Simulate autoregressive generation
    std::vector<int> generated = {1};  // Start token

    for (int step = 0; step < 5; ++step) {
        Matrix output = decoder.forward(generated);

        // In real scenario, would pick next token from output
        // Here just add a dummy token
        generated.push_back((step + 2) % vocab_size);
    }

    EXPECT_EQ(generated.size(), 6);  // 1 start + 5 generated
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(DecoderTest, ForwardPassPerformance) {
    int vocab_size = 1000;
    int d_model = 128;
    int num_layers = 4;
    LLMDecoder decoder(vocab_size, d_model, num_layers);

    std::vector<int> token_ids(50);
    for (int i = 0; i < 50; ++i) {
        token_ids[i] = i % vocab_size;
    }

    // Should complete reasonably quickly
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; ++i) {
        decoder.forward(token_ids);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Just ensure it completes (not checking specific time)
    EXPECT_LT(duration.count(), 60000);  // Less than 60 seconds for 10 iterations
}

TEST(DecoderTest, MemoryStability) {
    int vocab_size = 100;
    int d_model = 64;
    LLMDecoder decoder(vocab_size, d_model, 2);

    std::vector<int> token_ids = {1, 2, 3, 4, 5};

    // Run many iterations to check for memory leaks/corruption
    for (int i = 0; i < 100; ++i) {
        Matrix output = decoder.forward(token_ids);

        if (i % 10 == 0) {
            decoder.zero_grad();
        }
    }

    // Should complete without memory issues
    SUCCEED();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
