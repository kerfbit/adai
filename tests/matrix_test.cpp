#include "../src/Matrix.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <stdexcept>

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(MatrixConstructorTest, DefaultConstructor) {
    Matrix m;
    EXPECT_EQ(m.rows, 0);
    EXPECT_EQ(m.cols, 0);
    EXPECT_TRUE(m.data.empty());
}

TEST(MatrixConstructorTest, DimensionalConstructor) {
    Matrix m(3, 4);
    EXPECT_EQ(m.rows, 3);
    EXPECT_EQ(m.cols, 4);
    EXPECT_EQ(m.data.size(), 3);
    EXPECT_EQ(m.data[0].size(), 4);

    // Verify zero initialization
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_FLOAT_EQ(m(i, j), 0.0f);
        }
    }
}

TEST(MatrixConstructorTest, DataConstructor) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};

    Matrix m(data);
    EXPECT_EQ(m.rows, 2);
    EXPECT_EQ(m.cols, 3);
    EXPECT_FLOAT_EQ(m(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(m(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(m(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(m(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(m(1, 2), 6.0f);
}

TEST(MatrixConstructorTest, DataConstructorInconsistentRows) {
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f}  // Different size
    };

    EXPECT_THROW(Matrix m(data), std::invalid_argument);
}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST(MatrixAccessTest, ValidAccess) {
    Matrix m(3, 3);
    m(1, 2) = 7.5f;
    EXPECT_FLOAT_EQ(m(1, 2), 7.5f);
}

TEST(MatrixAccessTest, InvalidRowAccess) {
    Matrix m(3, 3);
    EXPECT_THROW(m(5, 1), std::out_of_range);
}

TEST(MatrixAccessTest, InvalidColAccess) {
    Matrix m(3, 3);
    EXPECT_THROW(m(1, 5), std::out_of_range);
}

TEST(MatrixAccessTest, ConstAccess) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    const Matrix m(data);
    EXPECT_FLOAT_EQ(m(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(m(1, 1), 4.0f);
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST(MatrixMultiplicationTest, BasicMultiplication) {
    std::vector<std::vector<float>> data_a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::vector<float>> data_b = {{5.0f, 6.0f}, {7.0f, 8.0f}};

    Matrix a(data_a);
    Matrix b(data_b);
    Matrix c = a * b;

    EXPECT_EQ(c.rows, 2);
    EXPECT_EQ(c.cols, 2);
    EXPECT_FLOAT_EQ(c(0, 0), 19.0f);  // 1*5 + 2*7
    EXPECT_FLOAT_EQ(c(0, 1), 22.0f);  // 1*6 + 2*8
    EXPECT_FLOAT_EQ(c(1, 0), 43.0f);  // 3*5 + 4*7
    EXPECT_FLOAT_EQ(c(1, 1), 50.0f);  // 3*6 + 4*8
}

TEST(MatrixMultiplicationTest, NonSquareMultiplication) {
    Matrix a(2, 3);  // 2x3
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;
    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    Matrix b(3, 2);  // 3x2
    b(0, 0) = 7;
    b(0, 1) = 8;
    b(1, 0) = 9;
    b(1, 1) = 10;
    b(2, 0) = 11;
    b(2, 1) = 12;

    Matrix c = a * b;

    EXPECT_EQ(c.rows, 2);
    EXPECT_EQ(c.cols, 2);
    EXPECT_FLOAT_EQ(c(0, 0), 58.0f);   // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(c(0, 1), 64.0f);   // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ(c(1, 0), 139.0f);  // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ(c(1, 1), 154.0f);  // 4*8 + 5*10 + 6*12
}

TEST(MatrixMultiplicationTest, DimensionMismatch) {
    Matrix a(2, 3);
    Matrix b(4, 2);  // Incompatible dimensions

    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(MatrixMultiplicationTest, IdentityMultiplication) {
    std::vector<std::vector<float>> identity = {
        {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
    std::vector<std::vector<float>> test_data = {
        {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};

    Matrix I(identity);
    Matrix A(test_data);
    Matrix result = A * I;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(result(i, j), A(i, j));
        }
    }
}

// ============================================================================
// Addition Tests
// ============================================================================

TEST(MatrixAdditionTest, BasicAddition) {
    std::vector<std::vector<float>> data_a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::vector<float>> data_b = {{5.0f, 6.0f}, {7.0f, 8.0f}};

    Matrix a(data_a);
    Matrix b(data_b);
    Matrix c = a + b;

    EXPECT_FLOAT_EQ(c(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 8.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 12.0f);
}

TEST(MatrixAdditionTest, DimensionMismatch) {
    Matrix a(2, 3);
    Matrix b(3, 2);

    EXPECT_THROW(a + b, std::invalid_argument);
}

// ============================================================================
// Subtraction Tests
// ============================================================================

TEST(MatrixSubtractionTest, BasicSubtraction) {
    std::vector<std::vector<float>> data_a = {{10.0f, 8.0f}, {6.0f, 4.0f}};
    std::vector<std::vector<float>> data_b = {{1.0f, 2.0f}, {3.0f, 4.0f}};

    Matrix a(data_a);
    Matrix b(data_b);
    Matrix c = a - b;

    EXPECT_FLOAT_EQ(c(0, 0), 9.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 6.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 0.0f);
}

TEST(MatrixSubtractionTest, DimensionMismatch) {
    Matrix a(2, 3);
    Matrix b(3, 2);

    EXPECT_THROW(a - b, std::invalid_argument);
}

// ============================================================================
// Transpose Tests
// ============================================================================

TEST(MatrixTransposeTest, SquareMatrix) {
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};

    Matrix m(data);
    Matrix t = m.transpose();

    EXPECT_EQ(t.rows, 3);
    EXPECT_EQ(t.cols, 3);
    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(t(0, 2), 7.0f);
    EXPECT_FLOAT_EQ(t(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(t(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(t(1, 2), 8.0f);
}

TEST(MatrixTransposeTest, RectangularMatrix) {
    Matrix m(2, 3);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(0, 2) = 3;
    m(1, 0) = 4;
    m(1, 1) = 5;
    m(1, 2) = 6;

    Matrix t = m.transpose();

    EXPECT_EQ(t.rows, 3);
    EXPECT_EQ(t.cols, 2);
    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(t(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(t(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(t(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(t(2, 1), 6.0f);
}

TEST(MatrixTransposeTest, DoubleTranspose) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix t = m.transpose().transpose();

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(t(i, j), m(i, j));
        }
    }
}

// ============================================================================
// Hadamard Product Tests
// ============================================================================

TEST(MatrixHadamardTest, BasicHadamard) {
    std::vector<std::vector<float>> data_a = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::vector<float>> data_b = {{2.0f, 3.0f}, {4.0f, 5.0f}};

    Matrix a(data_a);
    Matrix b(data_b);
    Matrix c = a.hadamard(b);

    EXPECT_FLOAT_EQ(c(0, 0), 2.0f);   // 1*2
    EXPECT_FLOAT_EQ(c(0, 1), 6.0f);   // 2*3
    EXPECT_FLOAT_EQ(c(1, 0), 12.0f);  // 3*4
    EXPECT_FLOAT_EQ(c(1, 1), 20.0f);  // 4*5
}

TEST(MatrixHadamardTest, DimensionMismatch) {
    Matrix a(2, 3);
    Matrix b(3, 2);

    EXPECT_THROW(a.hadamard(b), std::invalid_argument);
}

// ============================================================================
// Scale Tests
// ============================================================================

TEST(MatrixScaleTest, BasicScale) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix scaled = m.scale(2.0f);

    EXPECT_FLOAT_EQ(scaled(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(scaled(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(scaled(1, 0), 6.0f);
    EXPECT_FLOAT_EQ(scaled(1, 1), 8.0f);
}

TEST(MatrixScaleTest, ScaleByZero) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix scaled = m.scale(0.0f);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(scaled(i, j), 0.0f);
        }
    }
}

TEST(MatrixScaleTest, ScaleByNegative) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix scaled = m.scale(-1.0f);

    EXPECT_FLOAT_EQ(scaled(0, 0), -1.0f);
    EXPECT_FLOAT_EQ(scaled(0, 1), -2.0f);
    EXPECT_FLOAT_EQ(scaled(1, 0), -3.0f);
    EXPECT_FLOAT_EQ(scaled(1, 1), -4.0f);
}

// ============================================================================
// Fill Tests
// ============================================================================

TEST(MatrixFillTest, FillWithZero) {
    Matrix m(3, 4);
    m(0, 0) = 5.0f;
    m(1, 1) = 10.0f;
    m.fill(0.0f);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_FLOAT_EQ(m(i, j), 0.0f);
        }
    }
}

TEST(MatrixFillTest, FillWithConstant) {
    Matrix m(2, 3);
    m.fill(7.5f);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(m(i, j), 7.5f);
        }
    }
}

// ============================================================================
// Sum and Mean Tests
// ============================================================================

TEST(MatrixStatisticsTest, Sum) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);

    EXPECT_FLOAT_EQ(m.sum(), 10.0f);
}

TEST(MatrixStatisticsTest, Mean) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);

    EXPECT_FLOAT_EQ(m.mean(), 2.5f);
}

TEST(MatrixStatisticsTest, EmptyMatrixSum) {
    Matrix m;
    EXPECT_FLOAT_EQ(m.sum(), 0.0f);
}

TEST(MatrixStatisticsTest, EmptyMatrixMean) {
    Matrix m;
    EXPECT_TRUE(std::isnan(m.mean()) || m.mean() == 0.0f);
}

// ============================================================================
// Apply Gradients Tests
// ============================================================================

TEST(MatrixGradientsTest, BasicGradientApplication) {
    Matrix weights(2, 2);
    weights.fill(1.0f);

    Matrix gradients(2, 2);
    gradients.fill(0.1f);

    weights.apply_gradients(gradients, 1.0f);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(weights(i, j), 0.9f);
        }
    }
}

TEST(MatrixGradientsTest, WithLearningRate) {
    Matrix weights(2, 2);
    weights.fill(1.0f);

    Matrix gradients(2, 2);
    gradients.fill(0.1f);

    weights.apply_gradients(gradients, 0.5f);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(weights(i, j), 0.95f);  // 1.0 - 0.5*0.1
        }
    }
}

TEST(MatrixGradientsTest, DimensionMismatch) {
    Matrix weights(2, 3);
    Matrix gradients(3, 2);

    EXPECT_THROW(weights.apply_gradients(gradients, 0.01f), std::invalid_argument);
}

// ============================================================================
// Randomize Tests
// ============================================================================

TEST(MatrixRandomizeTest, NonZeroAfterRandomize) {
    Matrix m(10, 10);
    m.randomize(0.1f);

    bool has_nonzero = false;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            if (m(i, j) != 0.0f) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero)
            break;
    }

    EXPECT_TRUE(has_nonzero);
}

TEST(MatrixRandomizeTest, ValuesWithinExpectedRange) {
    Matrix m(100, 100);
    m.randomize(0.1f);

    // Most values should be within 3 standard deviations (99.7%)
    int within_range = 0;
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            if (std::abs(m(i, j)) <= 0.3f) {  // 3 * scale
                within_range++;
            }
        }
    }

    // At least 95% should be within range
    EXPECT_GT(within_range, 9500);
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST(MatrixReshapeTest, ValidReshape) {
    Matrix m(2, 6);
    for (int i = 0; i < 12; i++) {
        m.data[i / 6][i % 6] = static_cast<float>(i);
    }

    Matrix reshaped = m.reshape(3, 4);

    EXPECT_EQ(reshaped.rows, 3);
    EXPECT_EQ(reshaped.cols, 4);

    // Check elements are preserved in row-major order
    for (int i = 0; i < 12; i++) {
        int orig_row = i / 6;
        int orig_col = i % 6;
        int new_row = i / 4;
        int new_col = i % 4;
        EXPECT_FLOAT_EQ(reshaped(new_row, new_col), m(orig_row, orig_col));
    }
}

TEST(MatrixReshapeTest, InvalidReshape) {
    Matrix m(2, 3);  // 6 elements

    EXPECT_THROW(m.reshape(2, 4), std::invalid_argument);  // 8 elements
}

TEST(MatrixReshapeTest, ReshapeToSameDimensions) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    Matrix reshaped = m.reshape(2, 2);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(reshaped(i, j), m(i, j));
        }
    }
}

// ============================================================================
// Row/Column Access Tests
// ============================================================================

TEST(MatrixRowColTest, GetRow) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix m(data);

    std::vector<float> row0 = m.get_row(0);
    EXPECT_EQ(row0.size(), 3);
    EXPECT_FLOAT_EQ(row0[0], 1.0f);
    EXPECT_FLOAT_EQ(row0[1], 2.0f);
    EXPECT_FLOAT_EQ(row0[2], 3.0f);

    std::vector<float> row1 = m.get_row(1);
    EXPECT_FLOAT_EQ(row1[0], 4.0f);
    EXPECT_FLOAT_EQ(row1[1], 5.0f);
    EXPECT_FLOAT_EQ(row1[2], 6.0f);
}

TEST(MatrixRowColTest, GetColumn) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Matrix m(data);

    std::vector<float> col0 = m.get_col(0);
    EXPECT_EQ(col0.size(), 2);
    EXPECT_FLOAT_EQ(col0[0], 1.0f);
    EXPECT_FLOAT_EQ(col0[1], 4.0f);

    std::vector<float> col2 = m.get_col(2);
    EXPECT_FLOAT_EQ(col2[0], 3.0f);
    EXPECT_FLOAT_EQ(col2[1], 6.0f);
}

TEST(MatrixRowColTest, SetRow) {
    Matrix m(2, 3);
    std::vector<float> new_row = {7.0f, 8.0f, 9.0f};

    m.set_row(1, new_row);

    EXPECT_FLOAT_EQ(m(1, 0), 7.0f);
    EXPECT_FLOAT_EQ(m(1, 1), 8.0f);
    EXPECT_FLOAT_EQ(m(1, 2), 9.0f);
}

TEST(MatrixRowColTest, SetColumn) {
    Matrix m(2, 3);
    std::vector<float> new_col = {10.0f, 11.0f};

    m.set_col(1, new_col);

    EXPECT_FLOAT_EQ(m(0, 1), 10.0f);
    EXPECT_FLOAT_EQ(m(1, 1), 11.0f);
}

TEST(MatrixRowColTest, InvalidRowAccess) {
    Matrix m(2, 3);
    EXPECT_THROW(m.get_row(5), std::out_of_range);
    EXPECT_THROW(m.set_row(5, {1.0f, 2.0f, 3.0f}), std::out_of_range);
}

TEST(MatrixRowColTest, InvalidColumnAccess) {
    Matrix m(2, 3);
    EXPECT_THROW(m.get_col(5), std::out_of_range);
    EXPECT_THROW(m.set_col(5, {1.0f, 2.0f}), std::out_of_range);
}

TEST(MatrixRowColTest, SetRowWrongSize) {
    Matrix m(2, 3);
    std::vector<float> wrong_size = {1.0f, 2.0f};  // Size 2, needs 3

    EXPECT_THROW(m.set_row(0, wrong_size), std::invalid_argument);
}

TEST(MatrixRowColTest, SetColumnWrongSize) {
    Matrix m(2, 3);
    std::vector<float> wrong_size = {1.0f};  // Size 1, needs 2

    EXPECT_THROW(m.set_col(0, wrong_size), std::invalid_argument);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST(MatrixValidationTest, ValidMatrix) {
    Matrix m(3, 4);
    EXPECT_TRUE(m.is_valid());
}

TEST(MatrixValidationTest, EmptyMatrix) {
    Matrix m;
    EXPECT_FALSE(m.is_valid());  // Empty matrix is not valid (has zero dimensions)
}

TEST(MatrixValidationTest, ValidDataConstructor) {
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    Matrix m(data);
    EXPECT_TRUE(m.is_valid());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(MatrixIntegrationTest, LinearLayerForward) {
    // Simulate a simple linear layer: Y = X * W
    Matrix X(2, 3);  // 2 samples, 3 input features
    X(0, 0) = 1;
    X(0, 1) = 2;
    X(0, 2) = 3;
    X(1, 0) = 4;
    X(1, 1) = 5;
    X(1, 2) = 6;

    Matrix W(3, 2);  // 3 input features, 2 output features
    W.randomize(0.1f);

    Matrix Y = X * W;

    EXPECT_EQ(Y.rows, 2);
    EXPECT_EQ(Y.cols, 2);
}

TEST(MatrixIntegrationTest, GradientDescentIteration) {
    // Simulate one gradient descent iteration
    Matrix weights(3, 3);
    weights.randomize(0.1f);

    Matrix original = weights;

    Matrix gradients(3, 3);
    gradients.fill(0.01f);

    float lr = 0.1f;
    weights.apply_gradients(gradients, lr);

    // Weights should have changed
    bool weights_changed = false;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::abs(weights(i, j) - original(i, j)) > 1e-6f) {
                weights_changed = true;
                break;
            }
        }
        if (weights_changed)
            break;
    }

    EXPECT_TRUE(weights_changed);
}

TEST(MatrixIntegrationTest, BatchProcessing) {
    // Simulate processing a batch through a layer
    int batch_size = 32;
    int input_dim = 128;
    int output_dim = 64;

    Matrix batch(batch_size, input_dim);
    batch.randomize(0.1f);

    Matrix weights(input_dim, output_dim);
    weights.randomize(0.1f);

    Matrix output = batch * weights;

    EXPECT_EQ(output.rows, batch_size);
    EXPECT_EQ(output.cols, output_dim);
}

TEST(MatrixIntegrationTest, AttentionScoreComputation) {
    // Simulate attention score computation: scores = Q * K^T
    int seq_len = 10;
    int d_k = 64;

    Matrix Q(seq_len, d_k);
    Matrix K(seq_len, d_k);
    Q.randomize(0.1f);
    K.randomize(0.1f);

    Matrix scores = Q * K.transpose();

    EXPECT_EQ(scores.rows, seq_len);
    EXPECT_EQ(scores.cols, seq_len);

    // Scale scores
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    Matrix scaled_scores = scores.scale(scale);

    EXPECT_EQ(scaled_scores.rows, seq_len);
    EXPECT_EQ(scaled_scores.cols, seq_len);
}

TEST(MatrixIntegrationTest, ResidualConnection) {
    // Simulate residual connection: output = input + layer(input)
    Matrix input(10, 64);
    input.randomize(0.1f);

    Matrix weights(64, 64);
    weights.randomize(0.1f);

    Matrix layer_output = input * weights;
    Matrix residual_output = input + layer_output;

    EXPECT_EQ(residual_output.rows, 10);
    EXPECT_EQ(residual_output.cols, 64);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(MatrixEdgeCaseTest, SingleElement) {
    Matrix m(1, 1);
    m(0, 0) = 5.0f;

    EXPECT_FLOAT_EQ(m(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(m.sum(), 5.0f);
    EXPECT_FLOAT_EQ(m.mean(), 5.0f);
}

TEST(MatrixEdgeCaseTest, SingleRow) {
    Matrix m(1, 5);
    for (int i = 0; i < 5; i++) {
        m(0, i) = static_cast<float>(i);
    }

    Matrix t = m.transpose();
    EXPECT_EQ(t.rows, 5);
    EXPECT_EQ(t.cols, 1);
}

TEST(MatrixEdgeCaseTest, SingleColumn) {
    Matrix m(5, 1);
    for (int i = 0; i < 5; i++) {
        m(i, 0) = static_cast<float>(i);
    }

    Matrix t = m.transpose();
    EXPECT_EQ(t.rows, 1);
    EXPECT_EQ(t.cols, 5);
}

TEST(MatrixEdgeCaseTest, LargeMatrix) {
    Matrix m(1000, 1000);
    m.randomize(0.01f);

    EXPECT_TRUE(m.is_valid());
    EXPECT_EQ(m.rows, 1000);
    EXPECT_EQ(m.cols, 1000);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
