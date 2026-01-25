#include "../src/FeedForward.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "../src/Activation.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(FeedForwardConstructorTest, BasicConstruction) {
    FeedForward ff(512, 2048);

    EXPECT_EQ(ff.get_d_model(), 512);
    EXPECT_EQ(ff.get_d_ff(), 2048);
    EXPECT_FLOAT_EQ(ff.learning_rate, 0.001f);
}

TEST(FeedForwardConstructorTest, SmallModel) {
    FeedForward ff(128, 512);

    EXPECT_EQ(ff.get_d_model(), 128);
    EXPECT_EQ(ff.get_d_ff(), 512);
}

TEST(FeedForwardConstructorTest, LargeModel) {
    FeedForward ff(1024, 4096);

    EXPECT_EQ(ff.get_d_model(), 1024);
    EXPECT_EQ(ff.get_d_ff(), 4096);
}

TEST(FeedForwardConstructorTest, StandardExpansionRatio) {
    // Test typical 4x expansion ratio
    FeedForward ff(256, 1024);

    EXPECT_EQ(ff.get_d_ff(), 4 * ff.get_d_model());
}

TEST(FeedForwardConstructorTest, NonStandardExpansionRatio) {
    // Test 2x expansion ratio
    FeedForward ff(512, 1024);

    EXPECT_EQ(ff.get_d_ff(), 2 * ff.get_d_model());
}

TEST(FeedForwardConstructorTest, LearningRateModification) {
    FeedForward ff(512, 2048);
    ff.learning_rate = 0.0001f;

    EXPECT_FLOAT_EQ(ff.learning_rate, 0.0001f);
}

TEST(FeedForwardConstructorTest, MinimalDimensions) {
    FeedForward ff(2, 4);

    EXPECT_EQ(ff.get_d_model(), 2);
    EXPECT_EQ(ff.get_d_ff(), 4);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(FeedForwardForwardTest, BasicForward) {
    FeedForward ff(64, 256);

    Matrix input(10, 64);  // seq_len=10, d_model=64
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.01f * (i + j);
        }
    }

    Matrix output = ff.forward(input);

    EXPECT_EQ(output.rows, 10);
    EXPECT_EQ(output.cols, 64);
}

TEST(FeedForwardForwardTest, SingleToken) {
    FeedForward ff(128, 512);

    Matrix input(1, 128);
    for (int j = 0; j < 128; ++j) {
        input(0, j) = 0.1f;
    }

    Matrix output = ff.forward(input);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, 128);
}

TEST(FeedForwardForwardTest, LongSequence) {
    FeedForward ff(256, 1024);

    Matrix input(100, 256);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 256; ++j) {
            input(i, j) = 0.001f * i;
        }
    }

    Matrix output = ff.forward(input);

    EXPECT_EQ(output.rows, 100);
    EXPECT_EQ(output.cols, 256);
}

TEST(FeedForwardForwardTest, OutputNonZero) {
    FeedForward ff(32, 128);

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix output = ff.forward(input);

    // Check that output has non-zero values
    bool has_nonzero = false;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            if (std::abs(output(i, j)) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(FeedForwardForwardTest, DifferentInputsProduceDifferentOutputs) {
    FeedForward ff(64, 256);

    Matrix input1(5, 64);
    Matrix input2(5, 64);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input1(i, j) = 0.1f * i;
            input2(i, j) = 0.2f * i;
        }
    }

    Matrix output1 = ff.forward(input1);
    Matrix output2 = ff.forward(input2);

    // Outputs should be different
    bool outputs_differ = false;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            if (std::abs(output1(i, j) - output2(i, j)) > 1e-5f) {
                outputs_differ = true;
                break;
            }
        }
    }
    EXPECT_TRUE(outputs_differ);
}

TEST(FeedForwardForwardTest, Deterministic) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.01f * (i + j);
        }
    }

    Matrix output1 = ff.forward(input);
    Matrix output2 = ff.forward(input);

    // Multiple forwards with same input should produce identical output
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_FLOAT_EQ(output1(i, j), output2(i, j));
        }
    }
}

TEST(FeedForwardForwardTest, ZeroInput) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.0f;
        }
    }

    Matrix output = ff.forward(input);

    // Output should not be all zeros (due to biases)
    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, 64);
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST(FeedForwardBackwardTest, BasicBackward) {
    FeedForward ff(64, 256);

    Matrix input(10, 64);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.01f * (i + j);
        }
    }

    Matrix output = ff.forward(input);

    Matrix grad_output(10, 64);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.001f;
        }
    }

    Matrix grad_input = ff.backward(grad_output);

    EXPECT_EQ(grad_input.rows, 10);
    EXPECT_EQ(grad_input.cols, 64);
}

TEST(FeedForwardBackwardTest, GradientDimensions) {
    FeedForward ff(128, 512);

    Matrix input(20, 128);
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.1f;
        }
    }

    ff.forward(input);

    Matrix grad_output(20, 128);
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 128; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_input = ff.backward(grad_output);

    EXPECT_EQ(grad_input.rows, input.rows);
    EXPECT_EQ(grad_input.cols, input.cols);
}

TEST(FeedForwardBackwardTest, GradientsNonZero) {
    FeedForward ff(32, 128);

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.5f;
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    Matrix grad_input = ff.backward(grad_output);

    // Check that gradients have non-zero values
    bool has_nonzero = false;
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            if (std::abs(grad_input(i, j)) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(FeedForwardBackwardTest, GradientAccumulation) {
    FeedForward ff(64, 256);

    Matrix input1(5, 64);
    Matrix input2(5, 64);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input1(i, j) = 0.1f * i;
            input2(i, j) = 0.2f * i;
        }
    }

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ff.zero_grad();

    ff.forward(input1);
    ff.backward(grad_output);
    float norm1 = ff.get_gradient_norm();

    ff.forward(input2);
    ff.backward(grad_output);
    float norm2 = ff.get_gradient_norm();

    // Gradients should accumulate
    EXPECT_GT(norm2, norm1);
}

// ============================================================================
// Gradient Management Tests
// ============================================================================

TEST(FeedForwardGradientTest, ZeroGrad) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ff.backward(grad_output);

    float norm_before = ff.get_gradient_norm();
    EXPECT_GT(norm_before, 0.0f);

    ff.zero_grad();

    float norm_after = ff.get_gradient_norm();
    EXPECT_FLOAT_EQ(norm_after, 0.0f);
}

TEST(FeedForwardGradientTest, GetGradientNorm) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    ff.backward(grad_output);

    float norm = ff.get_gradient_norm();
    EXPECT_GT(norm, 0.0f);
    EXPECT_LT(norm, 1e6f);  // Should be reasonable
}

TEST(FeedForwardGradientTest, ClipGradients) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 10.0f;  // Large input
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 10.0f;  // Large gradient
        }
    }

    ff.backward(grad_output);

    float max_norm = 5.0f;
    ff.clip_gradients(max_norm);

    float norm_after = ff.get_gradient_norm();
    EXPECT_LE(norm_after, max_norm + 1e-4f);  // Allow small numerical error
}

TEST(FeedForwardGradientTest, ClipGradientsNoEffect) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.01f;
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ff.backward(grad_output);

    float norm_before = ff.get_gradient_norm();

    ff.clip_gradients(1000.0f);  // Very high threshold

    float norm_after = ff.get_gradient_norm();

    EXPECT_FLOAT_EQ(norm_before, norm_after);
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(FeedForwardUpdateTest, BasicUpdate) {
    FeedForward ff(32, 128);
    ff.learning_rate = 0.01f;

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output1 = ff.forward(input);

    Matrix grad_output(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ff.backward(grad_output);
    ff.update_weights();

    Matrix output2 = ff.forward(input);

    // Outputs should differ after weight update
    bool outputs_differ = false;
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            if (std::abs(output1(i, j) - output2(i, j)) > 1e-6f) {
                outputs_differ = true;
                break;
            }
        }
    }
    EXPECT_TRUE(outputs_differ);
}

TEST(FeedForwardUpdateTest, UpdateZerosGradients) {
    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ff.backward(grad_output);

    float norm_before = ff.get_gradient_norm();
    EXPECT_GT(norm_before, 0.0f);

    ff.update_weights();

    float norm_after = ff.get_gradient_norm();
    EXPECT_FLOAT_EQ(norm_after, 0.0f);
}

TEST(FeedForwardUpdateTest, MultipleUpdates) {
    FeedForward ff(32, 128);
    ff.learning_rate = 0.1f;

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix grad_output(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    // Perform multiple training steps
    for (int step = 0; step < 10; ++step) {
        ff.forward(input);
        ff.backward(grad_output);
        ff.update_weights();
    }

    // Network should still function
    Matrix final_output = ff.forward(input);
    EXPECT_EQ(final_output.rows, 5);
    EXPECT_EQ(final_output.cols, 32);
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(FeedForwardPersistenceTest, SaveAndLoad) {
    const std::string filename = "test_feedforward.bin";

    FeedForward ff1(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output1 = ff1.forward(input);

    ff1.save_weights(filename);

    FeedForward ff2(64, 256);
    ff2.load_weights(filename);

    Matrix output2 = ff2.forward(input);

    // Outputs should be identical
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_FLOAT_EQ(output1(i, j), output2(i, j));
        }
    }

    std::remove(filename.c_str());
}

TEST(FeedForwardPersistenceTest, LoadAfterTraining) {
    const std::string filename = "test_feedforward_trained.bin";

    FeedForward ff1(64, 256);
    ff1.learning_rate = 0.01f;

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Train for a few steps
    for (int i = 0; i < 5; ++i) {
        ff1.forward(input);
        ff1.backward(grad_output);
        ff1.update_weights();
    }

    Matrix output1 = ff1.forward(input);

    ff1.save_weights(filename);

    FeedForward ff2(64, 256);
    ff2.load_weights(filename);

    Matrix output2 = ff2.forward(input);

    // Outputs should match
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_FLOAT_EQ(output1(i, j), output2(i, j));
        }
    }

    std::remove(filename.c_str());
}

TEST(FeedForwardPersistenceTest, DimensionMismatch) {
    const std::string filename = "test_feedforward_mismatch.bin";

    FeedForward ff1(64, 256);
    ff1.save_weights(filename);

    FeedForward ff2(128, 512);  // Different dimensions

    EXPECT_THROW(ff2.load_weights(filename), std::runtime_error);

    std::remove(filename.c_str());
}

TEST(FeedForwardPersistenceTest, FileNotFound) {
    FeedForward ff(64, 256);

    EXPECT_THROW(ff.load_weights("nonexistent_file.bin"), std::runtime_error);
}

TEST(FeedForwardPersistenceTest, MultipleSaveLoad) {
    const std::string filename = "test_feedforward_multiple.bin";

    FeedForward ff(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    // Save and load multiple times
    for (int iter = 0; iter < 3; ++iter) {
        ff.save_weights(filename);

        FeedForward ff_temp(64, 256);
        ff_temp.load_weights(filename);

        Matrix output1 = ff.forward(input);
        Matrix output2 = ff_temp.forward(input);

        for (int i = 0; i < output1.rows; ++i) {
            for (int j = 0; j < output1.cols; ++j) {
                EXPECT_FLOAT_EQ(output1(i, j), output2(i, j));
            }
        }
    }

    std::remove(filename.c_str());
}

// ============================================================================
// Variable Sequence Length Tests
// ============================================================================

TEST(FeedForwardSequenceLengthTest, DifferentSequenceLengths) {
    FeedForward ff(64, 256);

    std::vector<int> seq_lengths = {1, 5, 10, 50, 100};

    for (int seq_len : seq_lengths) {
        Matrix input(seq_len, 64);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 64; ++j) {
                input(i, j) = 0.1f;
            }
        }

        Matrix output = ff.forward(input);

        EXPECT_EQ(output.rows, seq_len);
        EXPECT_EQ(output.cols, 64);
    }
}

TEST(FeedForwardSequenceLengthTest, BackwardDifferentLengths) {
    FeedForward ff(64, 256);

    std::vector<int> seq_lengths = {1, 10, 50};

    for (int seq_len : seq_lengths) {
        Matrix input(seq_len, 64);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 64; ++j) {
                input(i, j) = 0.1f;
            }
        }

        ff.forward(input);

        Matrix grad_output(seq_len, 64);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 64; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        Matrix grad_input = ff.backward(grad_output);

        EXPECT_EQ(grad_input.rows, seq_len);
        EXPECT_EQ(grad_input.cols, 64);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(FeedForwardIntegrationTest, SimpleTrainingLoop) {
    FeedForward ff(32, 128);
    ff.learning_rate = 0.01f;

    Matrix input(5, 32);
    Matrix target_output(5, 32);

    // Create simple training data
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * i;
            target_output(i, j) = 0.2f * i;
        }
    }

    // Train for several iterations
    for (int iter = 0; iter < 20; ++iter) {
        Matrix output = ff.forward(input);

        // Compute gradient (simple MSE)
        Matrix grad_output(5, 32);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 32; ++j) {
                grad_output(i, j) = 2.0f * (output(i, j) - target_output(i, j)) / (5 * 32);
            }
        }

        ff.backward(grad_output);
        ff.update_weights();
    }

    // Network should still function after training
    Matrix final_output = ff.forward(input);
    EXPECT_EQ(final_output.rows, 5);
    EXPECT_EQ(final_output.cols, 32);
}

TEST(FeedForwardIntegrationTest, ResidualConnection) {
    FeedForward ff(64, 256);

    Matrix input(10, 64);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix ff_output = ff.forward(input);

    // Simulate residual connection: output = input + ff_output
    Matrix residual_output(10, 64);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            residual_output(i, j) = input(i, j) + ff_output(i, j);
        }
    }

    EXPECT_EQ(residual_output.rows, 10);
    EXPECT_EQ(residual_output.cols, 64);

    // Backward through residual
    Matrix grad_residual(10, 64);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_residual(i, j) = 0.01f;
        }
    }

    Matrix grad_input = ff.backward(grad_residual);

    EXPECT_EQ(grad_input.rows, 10);
    EXPECT_EQ(grad_input.cols, 64);
}

TEST(FeedForwardIntegrationTest, MultipleInstances) {
    FeedForward ff1(64, 256);
    FeedForward ff2(64, 256);
    FeedForward ff3(64, 256);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output1 = ff1.forward(input);
    Matrix output2 = ff2.forward(input);
    Matrix output3 = ff3.forward(input);

    // Different instances should produce different outputs (different random init)
    bool all_same = true;
    for (int i = 0; i < 5 && all_same; ++i) {
        for (int j = 0; j < 64 && all_same; ++j) {
            if (std::abs(output1(i, j) - output2(i, j)) > 1e-5f ||
                std::abs(output2(i, j) - output3(i, j)) > 1e-5f) {
                all_same = false;
            }
        }
    }
    EXPECT_FALSE(all_same);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(FeedForwardEdgeCaseTest, VerySmallLearningRate) {
    FeedForward ff(32, 128);
    ff.learning_rate = 1e-10f;

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output1 = ff.forward(input);

    Matrix grad_output(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    ff.backward(grad_output);
    ff.update_weights();

    Matrix output2 = ff.forward(input);

    // Outputs should be nearly identical with tiny learning rate
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_NEAR(output1(i, j), output2(i, j), 1e-6f);
        }
    }
}

TEST(FeedForwardEdgeCaseTest, LargeLearningRate) {
    FeedForward ff(32, 128);
    ff.learning_rate = 100.0f;

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.01f;
        }
    }

    ff.forward(input);

    Matrix grad_output(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ff.backward(grad_output);
    ff.update_weights();

    // Network should still function (though may not train well)
    Matrix output = ff.forward(input);
    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, 32);
}

TEST(FeedForwardEdgeCaseTest, NegativeInputs) {
    FeedForward ff(32, 128);

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = -0.5f;
        }
    }

    Matrix output = ff.forward(input);

    // GELU should handle negative inputs
    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, 32);

    // Check that output contains finite values
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

TEST(FeedForwardEdgeCaseTest, LargeInputValues) {
    FeedForward ff(32, 128);

    Matrix input(5, 32);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 100.0f;
        }
    }

    Matrix output = ff.forward(input);

    // Check for numerical stability
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST(FeedForwardOptimizerTest, SetOptimizerBasic) {
    FeedForward ff(64, 256);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    ff.set_optimizer(&opt);

    // Verify optimizer was set by checking it works
    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }
    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    ff.forward(input);
    ff.backward(grad_output);
    EXPECT_NO_THROW(ff.update_weights());
}

TEST(FeedForwardOptimizerTest, SetOptimizerNullptr) {
    FeedForward ff(64, 256);

    // Should not crash when setting nullptr
    ff.set_optimizer(nullptr);

    // Should still work with default learning rate
    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }
    Matrix output = ff.forward(input);

    EXPECT_EQ(output.rows, 3);
    EXPECT_EQ(output.cols, 64);
}

TEST(FeedForwardOptimizerTest, UpdateWithOptimizer) {
    FeedForward ff(64, 256);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    ff.set_optimizer(&opt);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Forward and backward pass
    Matrix out_before = ff.forward(input);
    ff.backward(grad_output);
    ff.update_weights();

    // Forward again - output should have changed
    Matrix out_after = ff.forward(input);

    bool weights_changed = false;
    for (int i = 0; i < out_before.rows; ++i) {
        for (int j = 0; j < out_before.cols; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                weights_changed = true;
                break;
            }
        }
        if (weights_changed)
            break;
    }

    EXPECT_TRUE(weights_changed);
}

TEST(FeedForwardOptimizerTest, UpdateWithoutOptimizer) {
    FeedForward ff(64, 256);
    ff.learning_rate = 0.001f;

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Forward and backward pass
    Matrix out_before = ff.forward(input);
    ff.backward(grad_output);
    ff.update_weights();

    // Forward again - output should have changed
    Matrix out_after = ff.forward(input);

    bool weights_changed = false;
    for (int i = 0; i < out_before.rows; ++i) {
        for (int j = 0; j < out_before.cols; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                weights_changed = true;
                break;
            }
        }
        if (weights_changed)
            break;
    }

    EXPECT_TRUE(weights_changed);
}

TEST(FeedForwardOptimizerTest, OptimizerVsSimpleGradientDescent) {
    // Create FeedForward instance
    FeedForward ff(32, 128);

    // Set optimizer for ff with SGD (no momentum)
    Optimizer opt(OptimizerType::SGD, 0.01f);
    ff.set_optimizer(&opt);

    Matrix input(2, 32);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 32);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Forward and backward
    ff.forward(input);
    ff.backward(grad_output);

    // Update weights - SGD should work
    EXPECT_NO_THROW(ff.update_weights());

    // Should produce valid output
    Matrix output = ff.forward(input);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

TEST(FeedForwardOptimizerTest, MultipleUpdatesWithOptimizer) {
    FeedForward ff(64, 256);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    ff.set_optimizer(&opt);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    for (int i = 0; i < 5; ++i) {
        ff.forward(input);
        ff.backward(grad_output);
        ff.update_weights();
    }

    // Should complete without errors
    SUCCEED();
}

TEST(FeedForwardOptimizerTest, SwitchOptimizer) {
    FeedForward ff(64, 256);

    // Start with Adam
    Optimizer opt1(OptimizerType::ADAM, 0.001f);
    ff.set_optimizer(&opt1);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    ff.forward(input);
    ff.backward(grad_output);
    ff.update_weights();

    // Switch to SGD
    Optimizer opt2(OptimizerType::SGD, 0.01f);
    ff.set_optimizer(&opt2);

    ff.forward(input);
    ff.backward(grad_output);
    ff.update_weights();

    SUCCEED();
}

TEST(FeedForwardOptimizerTest, OptimizerWithDifferentLearningRates) {
    FeedForward ff1(64, 256);
    FeedForward ff2(64, 256);

    // Set different learning rates
    Optimizer opt1(OptimizerType::ADAM, 0.001f);
    Optimizer opt2(OptimizerType::ADAM, 0.01f);

    ff1.set_optimizer(&opt1);
    ff2.set_optimizer(&opt2);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix out1_before = ff1.forward(input);
    ff1.backward(grad_output);
    ff1.update_weights();
    Matrix out1_after = ff1.forward(input);

    Matrix out2_before = ff2.forward(input);
    ff2.backward(grad_output);
    ff2.update_weights();
    Matrix out2_after = ff2.forward(input);

    // Calculate update magnitudes
    float update_mag_1 = 0.0f;
    float update_mag_2 = 0.0f;

    for (int i = 0; i < out1_before.rows; ++i) {
        for (int j = 0; j < out1_before.cols; ++j) {
            update_mag_1 += std::abs(out1_after(i, j) - out1_before(i, j));
            update_mag_2 += std::abs(out2_after(i, j) - out2_before(i, j));
        }
    }

    // Higher learning rate should produce larger updates
    EXPECT_GT(update_mag_2, update_mag_1);
}

TEST(FeedForwardOptimizerTest, RegisterParametersExplicit) {
    FeedForward ff(64, 256);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    ff.set_optimizer(&opt);

    // register_parameters() should have been called by set_optimizer()
    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    ff.forward(input);
    ff.backward(grad_output);

    // Should not throw if parameters are registered
    EXPECT_NO_THROW(ff.update_weights());
}

TEST(FeedForwardOptimizerTest, ParametersChangeWithOptimizer) {
    FeedForward ff(64, 256);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    ff.set_optimizer(&opt);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.5f;
        }
    }

    Matrix out_before = ff.forward(input);

    // Train for a few steps
    for (int i = 0; i < 3; ++i) {
        ff.forward(input);
        ff.backward(grad_output);
        ff.update_weights();
    }

    Matrix out_after = ff.forward(input);

    // Output should have changed
    bool changed = false;
    for (int i = 0; i < out_before.rows && !changed; ++i) {
        for (int j = 0; j < out_before.cols && !changed; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                changed = true;
            }
        }
    }

    EXPECT_TRUE(changed);
}

TEST(FeedForwardOptimizerTest, LearningRateScheduling) {
    FeedForward ff(64, 256);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    ff.set_optimizer(&opt);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // First update with lr=0.001
    Matrix out1 = ff.forward(input);
    ff.backward(grad_output);
    ff.update_weights();
    Matrix out1_after = ff.forward(input);

    // Change learning rate to much higher
    opt.set_learning_rate(0.1f);  // 100x higher

    // Second update with lr=0.1
    Matrix out2 = ff.forward(input);
    ff.backward(grad_output);
    ff.update_weights();
    Matrix out2_after = ff.forward(input);

    // Calculate update magnitudes
    float update_mag_1 = 0.0f;
    float update_mag_2 = 0.0f;

    for (int i = 0; i < out1.rows; ++i) {
        for (int j = 0; j < out1.cols; ++j) {
            update_mag_1 += std::abs(out1_after(i, j) - out1(i, j));
            update_mag_2 += std::abs(out2_after(i, j) - out2(i, j));
        }
    }

    // Second update should be significantly larger (100x higher learning rate)
    EXPECT_GT(update_mag_2, update_mag_1 * 10.0f);
}

TEST(FeedForwardOptimizerTest, BackwardCompatibilityNoOptimizer) {
    FeedForward ff(64, 256);
    ff.learning_rate = 0.01f;

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Should work without optimizer (backward compatibility)
    for (int i = 0; i < 3; ++i) {
        Matrix output = ff.forward(input);

        EXPECT_EQ(output.rows, 3);
        EXPECT_EQ(output.cols, 64);

        ff.backward(grad_output);
        ff.update_weights();
    }

    SUCCEED();
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
