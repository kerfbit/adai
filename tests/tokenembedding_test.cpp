#include "../src/TokenEmbedding.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(TokenEmbeddingConstructorTest, BasicConstruction) {
    TokenEmbedding embeddings(100, 50);

    EXPECT_EQ(embeddings.get_vocab_size(), 100);
    EXPECT_EQ(embeddings.get_embedding_dim(), 50);
    EXPECT_FLOAT_EQ(embeddings.learning_rate, 0.001f);  // Default learning rate

    // Check embeddings matrix dimensions
    const Matrix& emb_matrix = embeddings.get_embeddings();
    EXPECT_EQ(emb_matrix.rows, 100);
    EXPECT_EQ(emb_matrix.cols, 50);
}

TEST(TokenEmbeddingConstructorTest, SmallVocabulary) {
    TokenEmbedding embeddings(10, 5);

    EXPECT_EQ(embeddings.get_vocab_size(), 10);
    EXPECT_EQ(embeddings.get_embedding_dim(), 5);
}

TEST(TokenEmbeddingConstructorTest, LargeVocabulary) {
    TokenEmbedding embeddings(50000, 512);

    EXPECT_EQ(embeddings.get_vocab_size(), 50000);
    EXPECT_EQ(embeddings.get_embedding_dim(), 512);
}

TEST(TokenEmbeddingConstructorTest, XavierInitialization) {
    TokenEmbedding embeddings(100, 64);
    const Matrix& emb_matrix = embeddings.get_embeddings();

    // Calculate mean and variance
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;

    for (int i = 0; i < emb_matrix.rows; ++i) {
        for (int j = 0; j < emb_matrix.cols; ++j) {
            float val = emb_matrix(i, j);
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }

    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);

    // Xavier initialization should have mean close to 0
    EXPECT_NEAR(mean, 0.0, 0.1);

    // Variance should be approximately 1/d_model
    double expected_variance = 1.0 / 64.0;
    EXPECT_NEAR(variance, expected_variance, 0.05);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(TokenEmbeddingForwardTest, SingleToken) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);

    std::vector<int> token_ids = {5};
    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, 10);

    // All values should be 1.0
    for (int j = 0; j < 10; ++j) {
        EXPECT_FLOAT_EQ(output(0, j), 1.0f);
    }
}

TEST(TokenEmbeddingForwardTest, MultipleTokens) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(2.5f);

    std::vector<int> token_ids = {1, 5, 10, 20};
    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 4);
    EXPECT_EQ(output.cols, 10);

    // All values should be 2.5
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), 2.5f);
        }
    }
}

TEST(TokenEmbeddingForwardTest, RepeatedTokens) {
    TokenEmbedding embeddings(100, 10);

    // Set specific values for token 5
    const Matrix& emb_matrix = embeddings.get_embeddings();
    for (int j = 0; j < 10; ++j) {
        const_cast<Matrix&>(emb_matrix)(5, j) = static_cast<float>(j);
    }

    std::vector<int> token_ids = {5, 5, 5};
    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 3);
    EXPECT_EQ(output.cols, 10);

    // All three rows should be identical
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), static_cast<float>(j));
        }
    }
}

TEST(TokenEmbeddingForwardTest, DifferentTokensHaveDifferentEmbeddings) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {0, 1};
    Matrix output = embeddings.forward(token_ids);

    // Check that embeddings for different tokens are different
    bool different = false;
    for (int j = 0; j < 10; ++j) {
        if (std::abs(output(0, j) - output(1, j)) > 1e-6f) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST(TokenEmbeddingForwardTest, InvalidTokenNegative) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {-1};
    EXPECT_THROW(embeddings.forward(token_ids), std::out_of_range);
}

TEST(TokenEmbeddingForwardTest, InvalidTokenTooLarge) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {100};  // vocab_size is 100, so max valid is 99
    EXPECT_THROW(embeddings.forward(token_ids), std::out_of_range);
}

TEST(TokenEmbeddingForwardTest, EmptyInput) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {};
    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 0);
    EXPECT_EQ(output.cols, 10);
}

TEST(TokenEmbeddingForwardTest, LongSequence) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids;
    for (int i = 0; i < 1000; ++i) {
        token_ids.push_back(i % 100);
    }

    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 1000);
    EXPECT_EQ(output.cols, 10);
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST(TokenEmbeddingBackwardTest, SingleTokenGradient) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);

    // Set gradient to all 1.0
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.backward(token_ids, grad_output);

    // Only token 5 should have gradient
    // We can't directly access embedding_grad, so we check via get_gradient_norm
    EXPECT_GT(embeddings.get_gradient_norm(), 0.0f);
}

TEST(TokenEmbeddingBackwardTest, MultipleTokensGradient) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    std::vector<int> token_ids = {1, 2, 3};
    Matrix grad_output(3, 10);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            grad_output(i, j) = static_cast<float>(i + 1);
        }
    }

    embeddings.backward(token_ids, grad_output);

    float grad_norm = embeddings.get_gradient_norm();
    EXPECT_GT(grad_norm, 0.0f);
}

TEST(TokenEmbeddingBackwardTest, RepeatedTokenGradientAccumulation) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    // Token 5 appears 3 times, so gradient should accumulate
    std::vector<int> token_ids = {5, 5, 5};
    Matrix grad_output(3, 10);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    embeddings.backward(token_ids, grad_output);

    // Gradient should be accumulated (3x)
    float grad_norm = embeddings.get_gradient_norm();
    EXPECT_GT(grad_norm, 0.0f);
}

TEST(TokenEmbeddingBackwardTest, DimensionMismatch) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {1, 2, 3};
    Matrix grad_output(2, 10);  // Wrong number of rows

    EXPECT_THROW(embeddings.backward(token_ids, grad_output), std::invalid_argument);
}

TEST(TokenEmbeddingBackwardTest, WrongEmbeddingDimension) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {1, 2};
    Matrix grad_output(2, 5);  // Wrong embedding dimension

    EXPECT_THROW(embeddings.backward(token_ids, grad_output), std::invalid_argument);
}

TEST(TokenEmbeddingBackwardTest, InvalidToken) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {150};  // Out of range
    Matrix grad_output(1, 10);

    EXPECT_THROW(embeddings.backward(token_ids, grad_output), std::out_of_range);
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(TokenEmbeddingUpdateTest, BasicUpdate) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.1f;
    embeddings.zero_grad();

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);

    // Set gradient to all 1.0
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();

    // After update, embedding should be 1.0 - 0.1 * 1.0 = 0.9
    std::vector<float> embedding = embeddings.get_token_embedding(5);
    for (float val : embedding) {
        EXPECT_NEAR(val, 0.9f, 1e-5f);
    }

    // Gradient should be zeroed after update
    EXPECT_FLOAT_EQ(embeddings.get_gradient_norm(), 0.0f);
}

TEST(TokenEmbeddingUpdateTest, MultipleUpdates) {
    TokenEmbedding embeddings(10, 5);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.1f;

    for (int step = 0; step < 5; ++step) {
        embeddings.zero_grad();

        std::vector<int> token_ids = {3};
        Matrix grad_output(1, 5);
        for (int j = 0; j < 5; ++j) {
            grad_output(0, j) = 0.5f;
        }

        embeddings.backward(token_ids, grad_output);
        embeddings.update_weights();
    }

    // After 5 updates: 1.0 - 5 * (0.1 * 0.5) = 0.75
    std::vector<float> embedding = embeddings.get_token_embedding(3);
    for (float val : embedding) {
        EXPECT_NEAR(val, 0.75f, 1e-5f);
    }
}

TEST(TokenEmbeddingUpdateTest, DifferentLearningRates) {
    TokenEmbedding embeddings1(10, 5);
    TokenEmbedding embeddings2(10, 5);

    embeddings1.initialize_constant(1.0f);
    embeddings2.initialize_constant(1.0f);

    embeddings1.learning_rate = 0.01f;
    embeddings2.learning_rate = 0.1f;

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 5);
    for (int j = 0; j < 5; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings1.backward(token_ids, grad_output);
    embeddings2.backward(token_ids, grad_output);

    embeddings1.update_weights();
    embeddings2.update_weights();

    // embeddings1: 1.0 - 0.01 * 1.0 = 0.99
    // embeddings2: 1.0 - 0.1 * 1.0 = 0.9
    std::vector<float> emb1 = embeddings1.get_token_embedding(5);
    std::vector<float> emb2 = embeddings2.get_token_embedding(5);

    EXPECT_NEAR(emb1[0], 0.99f, 1e-5f);
    EXPECT_NEAR(emb2[0], 0.9f, 1e-5f);
}

// ============================================================================
// Zero Gradient Tests
// ============================================================================

TEST(TokenEmbeddingZeroGradTest, ZeroAfterBackward) {
    TokenEmbedding embeddings(100, 10);

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.backward(token_ids, grad_output);
    EXPECT_GT(embeddings.get_gradient_norm(), 0.0f);

    embeddings.zero_grad();
    EXPECT_FLOAT_EQ(embeddings.get_gradient_norm(), 0.0f);
}

TEST(TokenEmbeddingZeroGradTest, MultipleZeroCalls) {
    TokenEmbedding embeddings(100, 10);

    embeddings.zero_grad();
    EXPECT_FLOAT_EQ(embeddings.get_gradient_norm(), 0.0f);

    embeddings.zero_grad();
    EXPECT_FLOAT_EQ(embeddings.get_gradient_norm(), 0.0f);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST(TokenEmbeddingAccessorTest, GetTokenEmbedding) {
    TokenEmbedding embeddings(100, 10);

    std::vector<float> embedding = embeddings.get_token_embedding(5);
    EXPECT_EQ(embedding.size(), 10);
}

TEST(TokenEmbeddingAccessorTest, GetTokenEmbeddingInvalid) {
    TokenEmbedding embeddings(100, 10);

    EXPECT_THROW(embeddings.get_token_embedding(-1), std::out_of_range);
    EXPECT_THROW(embeddings.get_token_embedding(100), std::out_of_range);
}

TEST(TokenEmbeddingAccessorTest, GetEmbeddings) {
    TokenEmbedding embeddings(100, 10);

    const Matrix& emb_matrix = embeddings.get_embeddings();
    EXPECT_EQ(emb_matrix.rows, 100);
    EXPECT_EQ(emb_matrix.cols, 10);
}

TEST(TokenEmbeddingAccessorTest, GetVocabSize) {
    TokenEmbedding embeddings(50000, 512);
    EXPECT_EQ(embeddings.get_vocab_size(), 50000);
}

TEST(TokenEmbeddingAccessorTest, GetEmbeddingDim) {
    TokenEmbedding embeddings(100, 768);
    EXPECT_EQ(embeddings.get_embedding_dim(), 768);
}

// ============================================================================
// Initialization Tests
// ============================================================================

TEST(TokenEmbeddingInitializationTest, InitializeConstant) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(5.5f);

    const Matrix& emb_matrix = embeddings.get_embeddings();
    for (int i = 0; i < emb_matrix.rows; ++i) {
        for (int j = 0; j < emb_matrix.cols; ++j) {
            EXPECT_FLOAT_EQ(emb_matrix(i, j), 5.5f);
        }
    }
}

TEST(TokenEmbeddingInitializationTest, InitializeZero) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(0.0f);

    const Matrix& emb_matrix = embeddings.get_embeddings();
    for (int i = 0; i < emb_matrix.rows; ++i) {
        for (int j = 0; j < emb_matrix.cols; ++j) {
            EXPECT_FLOAT_EQ(emb_matrix(i, j), 0.0f);
        }
    }
}

TEST(TokenEmbeddingInitializationTest, Reinitialize) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);

    const Matrix& emb_matrix = embeddings.get_embeddings();
    float first_value = emb_matrix(0, 0);
    EXPECT_FLOAT_EQ(first_value, 1.0f);

    embeddings.reinitialize();

    // After reinitialize, values should be different (random)
    float new_value = emb_matrix(0, 0);
    EXPECT_NE(first_value, new_value);
}

// ============================================================================
// Gradient Monitoring Tests
// ============================================================================

TEST(TokenEmbeddingGradientTest, GradientNormZeroInitially) {
    TokenEmbedding embeddings(100, 10);
    EXPECT_FLOAT_EQ(embeddings.get_gradient_norm(), 0.0f);
}

TEST(TokenEmbeddingGradientTest, GradientNormAfterBackward) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);

    // Set gradient with known values
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.backward(token_ids, grad_output);

    float expected_norm = std::sqrt(10.0f);  // sqrt(sum of 10 ones squared)
    EXPECT_NEAR(embeddings.get_gradient_norm(), expected_norm, 1e-4f);
}

TEST(TokenEmbeddingGradientTest, ClipGradients) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);

    // Set large gradients
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 100.0f;
    }

    embeddings.backward(token_ids, grad_output);

    float norm_before = embeddings.get_gradient_norm();
    EXPECT_GT(norm_before, 5.0f);

    embeddings.clip_gradients(5.0f);

    float norm_after = embeddings.get_gradient_norm();
    EXPECT_NEAR(norm_after, 5.0f, 1e-3f);
}

TEST(TokenEmbeddingGradientTest, ClipGradientsNoClipNeeded) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);

    // Set small gradients
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 0.1f;
    }

    embeddings.backward(token_ids, grad_output);

    float norm_before = embeddings.get_gradient_norm();
    embeddings.clip_gradients(10.0f);
    float norm_after = embeddings.get_gradient_norm();

    // Should be unchanged since norm < max_norm
    EXPECT_NEAR(norm_before, norm_after, 1e-6f);
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST(TokenEmbeddingPersistenceTest, SaveAndLoad) {
    TokenEmbedding embeddings1(50, 10);
    embeddings1.initialize_constant(3.14f);

    std::string filename = "test_embeddings.bin";
    embeddings1.save_embeddings(filename);

    TokenEmbedding embeddings2(50, 10);
    embeddings2.load_pretrained(filename);

    // Check that loaded embeddings match saved ones
    const Matrix& emb1 = embeddings1.get_embeddings();
    const Matrix& emb2 = embeddings2.get_embeddings();

    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ(emb1(i, j), emb2(i, j));
        }
    }

    // Clean up
    std::remove(filename.c_str());
}

TEST(TokenEmbeddingPersistenceTest, LoadDimensionMismatch) {
    TokenEmbedding embeddings1(50, 10);
    embeddings1.save_embeddings("test_embeddings_mismatch.bin");

    TokenEmbedding embeddings2(100, 10);  // Different vocab size
    EXPECT_THROW(embeddings2.load_pretrained("test_embeddings_mismatch.bin"), std::runtime_error);

    TokenEmbedding embeddings3(50, 20);  // Different d_model
    EXPECT_THROW(embeddings3.load_pretrained("test_embeddings_mismatch.bin"), std::runtime_error);

    // Clean up
    std::remove("test_embeddings_mismatch.bin");
}

TEST(TokenEmbeddingPersistenceTest, LoadNonexistentFile) {
    TokenEmbedding embeddings(100, 10);
    EXPECT_THROW(embeddings.load_pretrained("nonexistent_file.bin"), std::runtime_error);
}

TEST(TokenEmbeddingPersistenceTest, SaveLoadComplexValues) {
    TokenEmbedding embeddings1(10, 5);

    // Set specific pattern
    const Matrix& emb_matrix = embeddings1.get_embeddings();
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 5; ++j) {
            const_cast<Matrix&>(emb_matrix)(i, j) = i * 10.0f + j * 0.1f;
        }
    }

    std::string filename = "test_complex_embeddings.bin";
    embeddings1.save_embeddings(filename);

    TokenEmbedding embeddings2(10, 5);
    embeddings2.load_pretrained(filename);

    // Verify pattern is preserved
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 5; ++j) {
            float expected = i * 10.0f + j * 0.1f;
            EXPECT_FLOAT_EQ(embeddings2.get_embeddings()(i, j), expected);
        }
    }

    // Clean up
    std::remove(filename.c_str());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(TokenEmbeddingIntegrationTest, ForwardBackwardUpdate) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.1f;

    std::vector<int> token_ids = {5, 10, 15};

    // Forward pass
    Matrix output = embeddings.forward(token_ids);
    EXPECT_EQ(output.rows, 3);
    EXPECT_EQ(output.cols, 10);

    // Create gradient
    Matrix grad_output(3, 10);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            grad_output(i, j) = 0.5f;
        }
    }

    // Backward pass
    embeddings.backward(token_ids, grad_output);

    // Update
    embeddings.update_weights();

    // Check updated embeddings: 1.0 - 0.1 * 0.5 = 0.95
    for (int token : token_ids) {
        std::vector<float> emb = embeddings.get_token_embedding(token);
        for (float val : emb) {
            EXPECT_NEAR(val, 0.95f, 1e-5f);
        }
    }
}

TEST(TokenEmbeddingIntegrationTest, MultipleEpochs) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.01f;

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    // Train for 10 epochs
    for (int epoch = 0; epoch < 10; ++epoch) {
        embeddings.forward(token_ids);
        embeddings.backward(token_ids, grad_output);
        embeddings.update_weights();
    }

    // After 10 updates: 1.0 - 10 * (0.01 * 1.0) = 0.9
    std::vector<float> emb = embeddings.get_token_embedding(5);
    for (float val : emb) {
        EXPECT_NEAR(val, 0.9f, 1e-5f);
    }
}

TEST(TokenEmbeddingIntegrationTest, BatchProcessing) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.1f;

    // Process multiple batches
    std::vector<std::vector<int>> batches = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    for (const auto& batch : batches) {
        Matrix output = embeddings.forward(batch);

        Matrix grad_output(batch.size(), 10);
        for (size_t i = 0; i < batch.size(); ++i) {
            for (int j = 0; j < 10; ++j) {
                grad_output(i, j) = 0.1f;
            }
        }

        embeddings.backward(batch, grad_output);
        embeddings.update_weights();
    }

    // Check that tokens were updated
    for (int token = 1; token <= 9; ++token) {
        std::vector<float> emb = embeddings.get_token_embedding(token);
        for (float val : emb) {
            // 1.0 - 0.1 * 0.1 = 0.99
            EXPECT_NEAR(val, 0.99f, 1e-5f);
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TokenEmbeddingEdgeCaseTest, SingleDimension) {
    TokenEmbedding embeddings(100, 1);

    EXPECT_EQ(embeddings.get_vocab_size(), 100);
    EXPECT_EQ(embeddings.get_embedding_dim(), 1);

    std::vector<int> token_ids = {5};
    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, 1);
}

TEST(TokenEmbeddingEdgeCaseTest, VerySmallVocab) {
    TokenEmbedding embeddings(1, 10);

    std::vector<int> token_ids = {0};
    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, 10);
}

TEST(TokenEmbeddingEdgeCaseTest, AllTokensInVocab) {
    int vocab_size = 10;
    TokenEmbedding embeddings(vocab_size, 5);
    embeddings.initialize_constant(1.0f);

    std::vector<int> token_ids;
    for (int i = 0; i < vocab_size; ++i) {
        token_ids.push_back(i);
    }

    Matrix output = embeddings.forward(token_ids);

    EXPECT_EQ(output.rows, vocab_size);
    EXPECT_EQ(output.cols, 5);
}

TEST(TokenEmbeddingEdgeCaseTest, VeryLargeGradient) {
    TokenEmbedding embeddings(100, 10);
    embeddings.zero_grad();

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);

    // Set very large gradients
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1e10f;
    }

    embeddings.backward(token_ids, grad_output);

    float grad_norm = embeddings.get_gradient_norm();
    EXPECT_GT(grad_norm, 1e9f);

    // Clipping should bring it down
    embeddings.clip_gradients(1.0f);
    EXPECT_NEAR(embeddings.get_gradient_norm(), 1.0f, 1e-3f);
}

// ============================================================================
// Configuration and Debugging Tests
// ============================================================================

TEST(TokenEmbeddingDebugTest, PrintConfig) {
    TokenEmbedding embeddings(10000, 512);

    // This should not crash
    testing::internal::CaptureStdout();
    embeddings.print_config();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("10000") != std::string::npos);
    EXPECT_TRUE(output.find("512") != std::string::npos);
}

TEST(TokenEmbeddingDebugTest, PrintConfigCustomName) {
    TokenEmbedding embeddings(100, 64);

    testing::internal::CaptureStdout();
    embeddings.print_config("MyEmbedding");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("MyEmbedding") != std::string::npos);
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST(TokenEmbeddingOptimizerTest, SetOptimizerBasic) {
    TokenEmbedding embeddings(100, 64);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    // Should not throw
    EXPECT_NO_THROW(embeddings.set_optimizer(&opt));
}

TEST(TokenEmbeddingOptimizerTest, SetOptimizerNullptr) {
    TokenEmbedding embeddings(100, 64);

    // Should handle nullptr gracefully
    EXPECT_NO_THROW(embeddings.set_optimizer(nullptr));
}

TEST(TokenEmbeddingOptimizerTest, UpdateWithOptimizer) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);

    Optimizer opt(OptimizerType::ADAM, 0.001f);
    opt.set_betas(0.9f, 0.999f);
    embeddings.set_optimizer(&opt);

    // Forward pass
    std::vector<int> token_ids = {5, 10, 15};
    Matrix output = embeddings.forward(token_ids);

    // Create gradient (all ones)
    Matrix grad_output(3, 10);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    embeddings.backward(token_ids, grad_output);

    // Get embedding before update
    const Matrix& emb_before = embeddings.get_embeddings();
    float val_before = emb_before(5, 0);

    // Update using optimizer
    embeddings.update_weights();

    // Embedding should have changed
    const Matrix& emb_after = embeddings.get_embeddings();
    float val_after = emb_after(5, 0);

    EXPECT_NE(val_before, val_after);
}

TEST(TokenEmbeddingOptimizerTest, UpdateWithoutOptimizer) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.01f;

    // Don't set optimizer - should use simple gradient descent

    // Forward pass
    std::vector<int> token_ids = {5};
    Matrix output = embeddings.forward(token_ids);

    // Create gradient
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 2.0f;
    }

    embeddings.backward(token_ids, grad_output);

    const Matrix& emb_before = embeddings.get_embeddings();
    float val_before = emb_before(5, 0);

    embeddings.update_weights();

    const Matrix& emb_after = embeddings.get_embeddings();
    float val_after = emb_after(5, 0);

    // Should be: 1.0 - 0.01 * 2.0 = 0.98
    EXPECT_FLOAT_EQ(val_after, 0.98f);
}

TEST(TokenEmbeddingOptimizerTest, OptimizerVsSimpleGradientDescent) {
    // Create two identical embeddings
    TokenEmbedding emb_with_opt(100, 10);
    TokenEmbedding emb_without_opt(100, 10);

    emb_with_opt.initialize_constant(1.0f);
    emb_without_opt.initialize_constant(1.0f);

    // Set optimizer for first one with Adam (uses momentum and adaptive learning rates)
    Optimizer opt(OptimizerType::ADAM, 0.1f);  // Higher LR to see difference
    opt.set_betas(0.9f, 0.999f);
    emb_with_opt.set_optimizer(&opt);

    emb_without_opt.learning_rate = 0.1f;

    // Run multiple training steps with varying gradients
    std::vector<int> token_ids = {5, 10};

    for (int iter = 0; iter < 5; ++iter) {
        Matrix grad_output(2, 10);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 10; ++j) {
                // Varying gradient magnitude
                grad_output(i, j) = (iter % 2 == 0) ? 1.0f : 0.5f;
            }
        }

        emb_with_opt.forward(token_ids);
        emb_with_opt.backward(token_ids, grad_output);
        emb_with_opt.update_weights();

        emb_without_opt.forward(token_ids);
        emb_without_opt.backward(token_ids, grad_output);
        emb_without_opt.update_weights();
    }

    // Results should be different (Adam uses momentum and adapts per-parameter)
    const Matrix& emb1 = emb_with_opt.get_embeddings();
    const Matrix& emb2 = emb_without_opt.get_embeddings();

    // Check that at least one value is significantly different
    float diff = std::abs(emb1(5, 0) - emb2(5, 0));
    EXPECT_GT(diff, 0.001f);  // Should have noticeable difference
}

TEST(TokenEmbeddingOptimizerTest, MultipleUpdatesWithOptimizer) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);

    Optimizer opt(OptimizerType::ADAM, 0.001f);
    opt.set_betas(0.9f, 0.999f);
    embeddings.set_optimizer(&opt);

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    // Multiple updates
    for (int i = 0; i < 5; ++i) {
        embeddings.forward(token_ids);
        embeddings.backward(token_ids, grad_output);
        embeddings.update_weights();
    }

    // Embedding should have changed
    const Matrix& emb = embeddings.get_embeddings();
    EXPECT_NE(emb(5, 0), 1.0f);
}

TEST(TokenEmbeddingOptimizerTest, SwitchOptimizer) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);

    // Start with Adam
    Optimizer opt1(OptimizerType::ADAM, 0.001f);
    embeddings.set_optimizer(&opt1);

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();

    // Switch to different optimizer
    Optimizer opt2(OptimizerType::SGD, 0.01f);
    embeddings.set_optimizer(&opt2);

    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);

    // Should not throw
    EXPECT_NO_THROW(embeddings.update_weights());
}

TEST(TokenEmbeddingOptimizerTest, OptimizerWithDifferentLearningRates) {
    TokenEmbedding emb1(100, 10);
    TokenEmbedding emb2(100, 10);

    emb1.initialize_constant(1.0f);
    emb2.initialize_constant(1.0f);

    Optimizer opt1(OptimizerType::SGD, 0.001f);
    Optimizer opt2(OptimizerType::SGD, 0.1f);

    emb1.set_optimizer(&opt1);
    emb2.set_optimizer(&opt2);

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    emb1.forward(token_ids);
    emb1.backward(token_ids, grad_output);
    emb1.update_weights();

    emb2.forward(token_ids);
    emb2.backward(token_ids, grad_output);
    emb2.update_weights();

    const Matrix& m1 = emb1.get_embeddings();
    const Matrix& m2 = emb2.get_embeddings();

    // Higher learning rate should cause bigger change
    float change1 = std::abs(m1(5, 0) - 1.0f);
    float change2 = std::abs(m2(5, 0) - 1.0f);

    EXPECT_LT(change1, change2);
}

TEST(TokenEmbeddingOptimizerTest, RegisterParametersExplicit) {
    TokenEmbedding embeddings(100, 10);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    embeddings.set_optimizer(&opt);

    // register_parameters() should have been called by set_optimizer()
    // Verify by doing update
    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);

    // Should not throw if parameters are registered
    EXPECT_NO_THROW(embeddings.update_weights());
}

TEST(TokenEmbeddingOptimizerTest, ParametersChangeWithOptimizer) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(2.0f);

    Optimizer opt(OptimizerType::ADAM, 0.01f);
    opt.set_betas(0.9f, 0.999f);
    embeddings.set_optimizer(&opt);

    std::vector<int> token_ids = {5, 10, 15};
    Matrix grad_output(3, 10);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            grad_output(i, j) = 0.5f;
        }
    }

    const Matrix& emb_before = embeddings.get_embeddings();
    std::vector<float> vals_before;
    for (int token : token_ids) {
        vals_before.push_back(emb_before(token, 0));
    }

    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();

    const Matrix& emb_after = embeddings.get_embeddings();

    // All touched tokens should have changed
    for (size_t i = 0; i < token_ids.size(); ++i) {
        EXPECT_NE(emb_after(token_ids[i], 0), vals_before[i]);
    }

    // Untouched tokens should not have changed
    EXPECT_EQ(emb_after(50, 0), 2.0f);
}

TEST(TokenEmbeddingOptimizerTest, LearningRateScheduling) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);

    Optimizer opt(OptimizerType::SGD, 0.1f);
    embeddings.set_optimizer(&opt);

    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    // First update with LR = 0.1
    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();

    const Matrix& emb1 = embeddings.get_embeddings();
    float val1 = emb1(5, 0);

    // Change learning rate
    opt.set_learning_rate(0.01f);
    embeddings.initialize_constant(1.0f);

    // Second update with LR = 0.01
    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();

    const Matrix& emb2 = embeddings.get_embeddings();
    float val2 = emb2(5, 0);

    // Changes should be different due to different learning rates
    float change1 = std::abs(val1 - 1.0f);
    float change2 = std::abs(val2 - 1.0f);

    EXPECT_GT(change1, change2);
}

TEST(TokenEmbeddingOptimizerTest, BackwardCompatibilityNoOptimizer) {
    TokenEmbedding embeddings(100, 10);
    embeddings.initialize_constant(1.0f);
    embeddings.learning_rate = 0.01f;

    // Old-style usage without optimizer
    std::vector<int> token_ids = {5};
    Matrix grad_output(1, 10);
    for (int j = 0; j < 10; ++j) {
        grad_output(0, j) = 1.0f;
    }

    embeddings.forward(token_ids);
    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();

    const Matrix& emb = embeddings.get_embeddings();

    // Should use simple gradient descent: 1.0 - 0.01 * 1.0 = 0.99
    EXPECT_FLOAT_EQ(emb(5, 0), 0.99f);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
