#include "../src/PositionalEncoding.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// ============================================================================
// Test Suite 1: Initialization Tests
// ============================================================================

TEST(PositionalEncodingTest, BasicInitialization) {
    PositionalEncoding pe(100, 64);

    EXPECT_EQ(pe.get_max_len(), 100);
    EXPECT_EQ(pe.get_d_model(), 64);

    const Matrix& encoding = pe.get_encoding();
    EXPECT_EQ(encoding.rows, 100);
    EXPECT_EQ(encoding.cols, 64);
}

TEST(PositionalEncodingTest, SmallConfiguration) {
    PositionalEncoding pe(10, 8);

    EXPECT_EQ(pe.get_max_len(), 10);
    EXPECT_EQ(pe.get_d_model(), 8);
}

TEST(PositionalEncodingTest, LargeConfiguration) {
    PositionalEncoding pe(2048, 1024);

    EXPECT_EQ(pe.get_max_len(), 2048);
    EXPECT_EQ(pe.get_d_model(), 1024);

    // Memory should be allocated correctly
    const Matrix& encoding = pe.get_encoding();
    EXPECT_EQ(encoding.rows, 2048);
    EXPECT_EQ(encoding.cols, 1024);
}

TEST(PositionalEncodingTest, BERTConfiguration) {
    PositionalEncoding pe(512, 768);

    EXPECT_EQ(pe.get_max_len(), 512);
    EXPECT_EQ(pe.get_d_model(), 768);
}

// ============================================================================
// Test Suite 2: Encoding Value Tests
// ============================================================================

TEST(PositionalEncodingTest, EncodingValuesInRange) {
    PositionalEncoding pe(100, 128);

    const Matrix& encoding = pe.get_encoding();

    // All values should be in [-1, 1] (sine and cosine range)
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 128; ++j) {
            EXPECT_GE(encoding(i, j), -1.0f) << "Value at (" << i << "," << j << ") below -1";
            EXPECT_LE(encoding(i, j), 1.0f) << "Value at (" << i << "," << j << ") above 1";
        }
    }
}

TEST(PositionalEncodingTest, Position0Values) {
    PositionalEncoding pe(10, 64);

    std::vector<float> pos_0 = pe.get_position_encoding(0);

    // At position 0, sine of 0 is 0 (even dimensions)
    for (int i = 0; i < 64; i += 2) {
        EXPECT_NEAR(pos_0[i], 0.0f, 1e-5f) << "Even dim " << i << " should be 0 at position 0";
    }

    // At position 0, cosine of 0 is 1 (odd dimensions)
    for (int i = 1; i < 64; i += 2) {
        EXPECT_NEAR(pos_0[i], 1.0f, 1e-5f) << "Odd dim " << i << " should be 1 at position 0";
    }
}

TEST(PositionalEncodingTest, SinusoidalPattern) {
    PositionalEncoding pe(100, 64);

    // Check that adjacent dimensions use sine and cosine
    std::vector<float> pos_5 = pe.get_position_encoding(5);

    // For dimension 0 and 1, they should have the same angle but different functions
    // sin²(x) + cos²(x) = 1
    for (int i = 0; i < 64; i += 2) {
        float sin_val = pos_5[i];
        float cos_val = pos_5[i + 1];
        float sum_squares = sin_val * sin_val + cos_val * cos_val;
        EXPECT_NEAR(sum_squares, 1.0f, 1e-4f) << "Pythagorean identity failed at dim " << i;
    }
}

TEST(PositionalEncodingTest, UniquePositions) {
    PositionalEncoding pe(50, 64);

    // Each position should have a unique encoding
    std::vector<float> pos_10 = pe.get_position_encoding(10);
    std::vector<float> pos_20 = pe.get_position_encoding(20);

    bool different = false;
    for (int i = 0; i < 64; ++i) {
        if (std::abs(pos_10[i] - pos_20[i]) > 1e-5f) {
            different = true;
            break;
        }
    }

    EXPECT_TRUE(different) << "Different positions should have different encodings";
}

// ============================================================================
// Test Suite 3: Forward Pass Tests
// ============================================================================

TEST(PositionalEncodingTest, ForwardPassBasic) {
    PositionalEncoding pe(10, 8);

    Matrix input(5, 8);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 8; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix output = pe.forward(input);

    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, 8);

    // Output should be input + encoding
    for (int i = 0; i < 5; ++i) {
        std::vector<float> pos_enc = pe.get_position_encoding(i);
        for (int j = 0; j < 8; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), 1.0f + pos_enc[j]);
        }
    }
}

TEST(PositionalEncodingTest, ForwardPassPreservesInput) {
    PositionalEncoding pe(10, 8);

    Matrix input(5, 8);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 8; ++j) {
            input(i, j) = static_cast<float>(i * 8 + j);
        }
    }

    Matrix input_copy = input;
    Matrix output = pe.forward(input);

    // Original input should be unchanged
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 8; ++j) {
            EXPECT_FLOAT_EQ(input(i, j), input_copy(i, j));
        }
    }
}

TEST(PositionalEncodingTest, ForwardPassCorrectAddition) {
    PositionalEncoding pe(20, 16);

    Matrix input(10, 16);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 16; ++j) {
            input(i, j) = 0.5f * (i + j);
        }
    }

    Matrix output = pe.forward(input);
    const Matrix& encoding = pe.get_encoding();

    // Verify element-wise addition
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 16; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), input(i, j) + encoding(i, j));
        }
    }
}

TEST(PositionalEncodingTest, ForwardPassZeroInput) {
    PositionalEncoding pe(10, 8);

    Matrix input(5, 8);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 8; ++j) {
            input(i, j) = 0.0f;
        }
    }

    Matrix output = pe.forward(input);

    // Output should equal encoding when input is zero
    for (int i = 0; i < 5; ++i) {
        std::vector<float> pos_enc = pe.get_position_encoding(i);
        for (int j = 0; j < 8; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), pos_enc[j]);
        }
    }
}

// ============================================================================
// Test Suite 4: Relative Position Tests
// ============================================================================

TEST(PositionalEncodingTest, RelativePositionSimilarity) {
    PositionalEncoding pe(100, 128);

    // Test that encodings change continuously with position
    // Higher dimensions (lower frequency) should have more similar patterns
    std::vector<float> pos_5 = pe.get_position_encoding(5);
    std::vector<float> pos_6 = pe.get_position_encoding(6);
    std::vector<float> pos_15 = pe.get_position_encoding(15);
    std::vector<float> pos_16 = pe.get_position_encoding(16);

    // Compute differences between adjacent positions
    std::vector<float> diff_5_6(128);
    std::vector<float> diff_15_16(128);

    for (int i = 0; i < 128; ++i) {
        diff_5_6[i] = pos_6[i] - pos_5[i];
        diff_15_16[i] = pos_16[i] - pos_15[i];
    }

    // For higher dimensions (lower frequency), adjacent position differences
    // should be more similar across different positions
    int similar_dims = 0;
    for (int i = 64; i < 128; ++i) {  // Only check higher dimensions
        if (std::abs(diff_5_6[i] - diff_15_16[i]) < 0.1f) {
            similar_dims++;
        }
    }

    // At least half of high-frequency dimensions should have similar patterns
    EXPECT_GT(similar_dims, 32) << "High-frequency dimensions should have consistent patterns";
}

TEST(PositionalEncodingTest, DotProductSimilarity) {
    PositionalEncoding pe(100, 64);

    // Positions with same relative distance should have similar dot products
    std::vector<float> pos_10 = pe.get_position_encoding(10);
    std::vector<float> pos_20 = pe.get_position_encoding(20);
    std::vector<float> pos_30 = pe.get_position_encoding(30);
    std::vector<float> pos_40 = pe.get_position_encoding(40);

    // Dot product helper
    auto dot_product = [](const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    };

    float sim_10_20 = dot_product(pos_10, pos_20);
    float sim_20_30 = dot_product(pos_20, pos_30);
    float sim_30_40 = dot_product(pos_30, pos_40);

    // Similar relative distances should have similar similarities
    EXPECT_NEAR(sim_10_20, sim_20_30, 5.0f);
    EXPECT_NEAR(sim_20_30, sim_30_40, 5.0f);
}

// ============================================================================
// Test Suite 5: Wavelength Tests
// ============================================================================

TEST(PositionalEncodingTest, WavelengthProgression) {
    PositionalEncoding pe(100, 64);

    // Lower dimensions should change faster than higher dimensions
    std::vector<float> pos_0 = pe.get_position_encoding(0);
    std::vector<float> pos_1 = pe.get_position_encoding(1);

    // Difference in first dimension (high frequency)
    float diff_dim_0 = std::abs(pos_1[0] - pos_0[0]);

    // Difference in last dimension (low frequency)
    float diff_dim_63 = std::abs(pos_1[63] - pos_0[63]);

    // Lower dimensions should change more between adjacent positions
    EXPECT_GT(diff_dim_0, diff_dim_63);
}

TEST(PositionalEncodingTest, HighFrequencyDimensions) {
    PositionalEncoding pe(100, 64);

    // First few dimensions should oscillate rapidly
    int sign_changes = 0;
    std::vector<float> prev_pos = pe.get_position_encoding(0);

    for (int pos = 1; pos < 20; ++pos) {
        std::vector<float> curr_pos = pe.get_position_encoding(pos);

        // Check if sign changed in first dimension
        if ((prev_pos[0] >= 0 && curr_pos[0] < 0) || (prev_pos[0] < 0 && curr_pos[0] >= 0)) {
            sign_changes++;
        }

        prev_pos = curr_pos;
    }

    // Should have multiple sign changes in high-frequency dimension
    EXPECT_GT(sign_changes, 2);
}

// ============================================================================
// Test Suite 6: Boundary and Edge Cases
// ============================================================================

TEST(PositionalEncodingTest, SinglePosition) {
    PositionalEncoding pe(1, 8);

    EXPECT_EQ(pe.get_max_len(), 1);

    std::vector<float> pos_0 = pe.get_position_encoding(0);
    EXPECT_EQ(pos_0.size(), 8);
}

TEST(PositionalEncodingTest, SingleDimension) {
    PositionalEncoding pe(10, 2);

    EXPECT_EQ(pe.get_d_model(), 2);

    std::vector<float> pos_5 = pe.get_position_encoding(5);
    EXPECT_EQ(pos_5.size(), 2);

    // First should be sine, second should be cosine
    float sin_val = pos_5[0];
    float cos_val = pos_5[1];
    EXPECT_NEAR(sin_val * sin_val + cos_val * cos_val, 1.0f, 1e-5f);
}

TEST(PositionalEncodingTest, MaxLengthBoundary) {
    PositionalEncoding pe(10, 8);

    // Should work for max_len - 1
    std::vector<float> pos_9 = pe.get_position_encoding(9);
    EXPECT_EQ(pos_9.size(), 8);

    // Should throw for max_len and beyond
    EXPECT_THROW(pe.get_position_encoding(10), std::out_of_range);
    EXPECT_THROW(pe.get_position_encoding(100), std::out_of_range);
}

TEST(PositionalEncodingTest, SequenceLongerThanMaxLen) {
    PositionalEncoding pe(10, 8);

    Matrix input(15, 8);  // Longer than max_len
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 8; ++j) {
            input(i, j) = 1.0f;
        }
    }

    // Should not crash, but only encode first max_len positions
    Matrix output = pe.forward(input);

    EXPECT_EQ(output.rows, 15);
    EXPECT_EQ(output.cols, 8);

    // First 10 positions should have encoding added
    for (int i = 0; i < 10; ++i) {
        std::vector<float> pos_enc = pe.get_position_encoding(i);
        for (int j = 0; j < 8; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), 1.0f + pos_enc[j]);
        }
    }

    // Positions beyond max_len should be unchanged
    for (int i = 10; i < 15; ++i) {
        for (int j = 0; j < 8; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), 1.0f);
        }
    }
}

TEST(PositionalEncodingTest, EmptySequence) {
    PositionalEncoding pe(10, 8);

    Matrix input(0, 8);  // Empty sequence
    Matrix output = pe.forward(input);

    EXPECT_EQ(output.rows, 0);
    EXPECT_EQ(output.cols, 8);
}

// ============================================================================
// Test Suite 7: Mathematical Properties
// ============================================================================

TEST(PositionalEncodingTest, PythagoreanIdentity) {
    PositionalEncoding pe(50, 64);

    // For all positions and dimension pairs, sin²(x) + cos²(x) = 1
    for (int pos = 0; pos < 50; ++pos) {
        std::vector<float> encoding = pe.get_position_encoding(pos);

        for (int i = 0; i < 64; i += 2) {
            float sin_val = encoding[i];
            float cos_val = encoding[i + 1];
            float sum_squares = sin_val * sin_val + cos_val * cos_val;

            EXPECT_NEAR(sum_squares, 1.0f, 1e-4f) << "Failed at pos=" << pos << ", dim=" << i;
        }
    }
}

TEST(PositionalEncodingTest, OddEvenPattern) {
    PositionalEncoding pe(20, 32);

    const Matrix& encoding = pe.get_encoding();

    // For same angle, sin and cos should satisfy: sin²(x) + cos²(x) = 1
    for (int pos = 0; pos < 20; ++pos) {
        for (int i = 0; i < 32; i += 2) {
            float even_val = encoding(pos, i);     // sine
            float odd_val = encoding(pos, i + 1);  // cosine

            float sum = even_val * even_val + odd_val * odd_val;
            EXPECT_NEAR(sum, 1.0f, 1e-4f);
        }
    }
}

TEST(PositionalEncodingTest, FormulaVerification) {
    PositionalEncoding pe(10, 8);

    // Manually compute expected values for position 3
    int pos = 3;
    int d_model = 8;
    std::vector<float> expected(8);

    for (int i = 0; i < 8; ++i) {
        float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / static_cast<float>(d_model));
        expected[i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
    }

    std::vector<float> actual = pe.get_position_encoding(pos);

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "Mismatch at dimension " << i;
    }
}

// ============================================================================
// Test Suite 8: Integration Tests
// ============================================================================

TEST(PositionalEncodingTest, TransformerIntegration) {
    // Simulate a small transformer embedding + positional encoding
    int vocab_size = 100;
    int d_model = 64;
    int max_len = 50;
    int seq_len = 10;

    PositionalEncoding pe(max_len, d_model);

    // Simulate token embeddings (random values)
    Matrix embeddings(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embeddings(i, j) = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Add positional encodings
    Matrix positioned = pe.forward(embeddings);

    EXPECT_EQ(positioned.rows, seq_len);
    EXPECT_EQ(positioned.cols, d_model);

    // Verify addition occurred
    for (int i = 0; i < seq_len; ++i) {
        std::vector<float> pos_enc = pe.get_position_encoding(i);
        for (int j = 0; j < d_model; ++j) {
            EXPECT_FLOAT_EQ(positioned(i, j), embeddings(i, j) + pos_enc[j]);
        }
    }
}

TEST(PositionalEncodingTest, MultipleForwardPasses) {
    PositionalEncoding pe(20, 16);

    Matrix input1(5, 16);
    Matrix input2(10, 16);
    Matrix input3(3, 16);

    // Fill with different values
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 16; ++j) {
            input1(i, j) = 1.0f;
        }
    }
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 16; ++j) {
            input2(i, j) = 2.0f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 16; ++j) {
            input3(i, j) = 3.0f;
        }
    }

    Matrix output1 = pe.forward(input1);
    Matrix output2 = pe.forward(input2);
    Matrix output3 = pe.forward(input3);

    // Verify each independently
    EXPECT_EQ(output1.rows, 5);
    EXPECT_EQ(output2.rows, 10);
    EXPECT_EQ(output3.rows, 3);

    // Verify correctness of first output
    for (int i = 0; i < 5; ++i) {
        std::vector<float> pos_enc = pe.get_position_encoding(i);
        for (int j = 0; j < 16; ++j) {
            EXPECT_FLOAT_EQ(output1(i, j), 1.0f + pos_enc[j]);
        }
    }
}

TEST(PositionalEncodingTest, BERTStyleUsage) {
    // BERT-base configuration
    PositionalEncoding pe(512, 768);

    // Typical sentence length
    Matrix embeddings(20, 768);
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 768; ++j) {
            embeddings(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = pe.forward(embeddings);

    EXPECT_EQ(output.rows, 20);
    EXPECT_EQ(output.cols, 768);

    // Verify encoding added correctly
    const Matrix& encoding = pe.get_encoding();
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 768; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), embeddings(i, j) + encoding(i, j));
        }
    }
}

// ============================================================================
// Test Suite 9: Utility Function Tests
// ============================================================================

TEST(PositionalEncodingTest, GetPositionEncodingVector) {
    PositionalEncoding pe(10, 8);

    std::vector<float> pos_5 = pe.get_position_encoding(5);

    EXPECT_EQ(pos_5.size(), 8);

    // Should match matrix values
    const Matrix& encoding = pe.get_encoding();
    for (int j = 0; j < 8; ++j) {
        EXPECT_FLOAT_EQ(pos_5[j], encoding(5, j));
    }
}

TEST(PositionalEncodingTest, GetEncodingMatrix) {
    PositionalEncoding pe(10, 8);

    const Matrix& encoding1 = pe.get_encoding();
    const Matrix& encoding2 = pe.get_encoding();

    // Should return same reference
    EXPECT_EQ(&encoding1, &encoding2);

    EXPECT_EQ(encoding1.rows, 10);
    EXPECT_EQ(encoding1.cols, 8);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
