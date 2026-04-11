// matrix_simd_test.cpp — TD-007: SIMD-specific tests for Matrix operations
//
// Verifies that all SIMD and BLAS code paths produce numerically correct
// results, correctly handle dimensions not divisible by the SIMD width,
// and that runtime CPU feature detection works.
//
// Tests are independent of which code path is compiled in: they exercise
// the public Matrix API and compare results against reference values.

#include "../src/Matrix.hpp"
#include "../src/MatrixSIMD.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

// ============================================================================
// Helper: arithmetic reference (scalar, no SIMD, no OpenMP shortcuts)
// ============================================================================

static float ref_sum(const Matrix& m) {
    float t = 0.0f;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            t += m.data[i][j];
    return t;
}

static Matrix ref_mul(const Matrix& a, const Matrix& b) {
    Matrix r(a.rows, b.cols);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < b.cols; ++j) {
            float s = 0.0f;
            for (int k = 0; k < a.cols; ++k)
                s += a.data[i][k] * b.data[k][j];
            r.data[i][j] = s;
        }
    return r;
}

// Fill a matrix with a predictable pattern: m[i][j] = base + i*cols + j
static Matrix make_seq(int rows, int cols, float base = 1.0f, float scale = 0.1f) {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m.data[i][j] = base + static_cast<float>(i * cols + j) * scale;
    return m;
}

// Tolerance for "near-equal" floating-point comparisons
static constexpr float kTol = 1e-4f;

// ============================================================================
// TD-007: Runtime CPU feature detection
// ============================================================================

TEST(MatrixSIMDCPUDetect, HasAVX2ReturnsBool) {
    // Just verify the function is callable and returns a sensible boolean.
    bool result = adai::simd::has_avx2();
    // On x86 platforms either true or false is valid; on others always false.
    EXPECT_TRUE(result == true || result == false);
}

TEST(MatrixSIMDCPUDetect, HasFMAReturnsBool) {
    bool result = adai::simd::has_fma();
    EXPECT_TRUE(result == true || result == false);
}

#if defined(ADAI_SIMD_AVX2)
TEST(MatrixSIMDCPUDetect, CompileTimeAVX2MatchesRuntime) {
    // When ADAI_SIMD_AVX2 is defined at compile time the CPU should actually
    // support AVX2 at runtime (otherwise we'd have an illegal instruction fault).
    EXPECT_TRUE(adai::simd::has_avx2());
}
#endif

// ============================================================================
// TD-007: SIMD addition — all column widths (including non-multiples of 8)
// ============================================================================

class MatrixSIMDAddTest : public ::testing::TestWithParam<int> {};

TEST_P(MatrixSIMDAddTest, AddMatchesReference) {
    int cols = GetParam();
    Matrix a = make_seq(4, cols, 1.0f, 0.5f);
    Matrix b = make_seq(4, cols, 0.5f, 0.3f);

    Matrix result = a + b;

    for (int i = 0; i < result.rows; ++i)
        for (int j = 0; j < result.cols; ++j)
            EXPECT_NEAR(result.data[i][j],
                        a.data[i][j] + b.data[i][j], kTol)
                << "at (" << i << ", " << j << ") with cols=" << cols;
}

// Widths that stress the SIMD remainder path: below 8, at 8, above 8,
// non-multiples of 8, large round number.
INSTANTIATE_TEST_SUITE_P(ColumnWidths, MatrixSIMDAddTest,
    ::testing::Values(1, 3, 7, 8, 9, 15, 16, 17, 32, 35, 64, 100, 128, 256));

// ============================================================================
// TD-007: SIMD subtraction
// ============================================================================

class MatrixSIMDSubTest : public ::testing::TestWithParam<int> {};

TEST_P(MatrixSIMDSubTest, SubMatchesReference) {
    int cols = GetParam();
    Matrix a = make_seq(4, cols, 2.0f, 0.4f);
    Matrix b = make_seq(4, cols, 0.5f, 0.2f);

    Matrix result = a - b;

    for (int i = 0; i < result.rows; ++i)
        for (int j = 0; j < result.cols; ++j)
            EXPECT_NEAR(result.data[i][j],
                        a.data[i][j] - b.data[i][j], kTol)
                << "at (" << i << ", " << j << ") with cols=" << cols;
}

INSTANTIATE_TEST_SUITE_P(ColumnWidths, MatrixSIMDSubTest,
    ::testing::Values(1, 3, 7, 8, 9, 15, 16, 17, 32, 33, 64, 100, 256));

// ============================================================================
// TD-007: SIMD Hadamard (element-wise multiply)
// ============================================================================

class MatrixSIMDHadamardTest : public ::testing::TestWithParam<int> {};

TEST_P(MatrixSIMDHadamardTest, HadamardMatchesReference) {
    int cols = GetParam();
    Matrix a = make_seq(3, cols, 1.0f, 0.1f);
    Matrix b = make_seq(3, cols, 2.0f, 0.05f);

    Matrix result = a.hadamard(b);

    for (int i = 0; i < result.rows; ++i)
        for (int j = 0; j < result.cols; ++j)
            EXPECT_NEAR(result.data[i][j],
                        a.data[i][j] * b.data[i][j], kTol)
                << "at (" << i << ", " << j << ") with cols=" << cols;
}

INSTANTIATE_TEST_SUITE_P(ColumnWidths, MatrixSIMDHadamardTest,
    ::testing::Values(1, 5, 8, 13, 16, 17, 32, 64, 128, 200));

// ============================================================================
// TD-007: SIMD scale (scalar multiply)
// ============================================================================

class MatrixSIMDScaleTest : public ::testing::TestWithParam<int> {};

TEST_P(MatrixSIMDScaleTest, ScaleMatchesReference) {
    int cols = GetParam();
    float factor = 2.5f;
    Matrix a = make_seq(4, cols, 1.0f, 0.3f);

    Matrix result = a.scale(factor);

    for (int i = 0; i < result.rows; ++i)
        for (int j = 0; j < result.cols; ++j)
            EXPECT_NEAR(result.data[i][j], a.data[i][j] * factor, kTol)
                << "at (" << i << ", " << j << ") with cols=" << cols;
}

INSTANTIATE_TEST_SUITE_P(ColumnWidths, MatrixSIMDScaleTest,
    ::testing::Values(1, 7, 8, 9, 16, 17, 33, 64, 128, 255));

// ============================================================================
// TD-007: SIMD apply_gradients (W = W - lr * g)
// ============================================================================

class MatrixSIMDGradientsTest : public ::testing::TestWithParam<int> {};

TEST_P(MatrixSIMDGradientsTest, ApplyGradientsMatchesReference) {
    int cols = GetParam();
    float lr = 0.01f;

    Matrix weights = make_seq(4, cols, 1.0f, 0.2f);
    Matrix grads   = make_seq(4, cols, 0.1f, 0.05f);

    // Compute expected: weights_orig - lr * grads
    std::vector<std::vector<float>> expected(4, std::vector<float>(cols));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < cols; ++j)
            expected[i][j] = weights.data[i][j] - lr * grads.data[i][j];

    weights.apply_gradients(grads, lr);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < cols; ++j)
            EXPECT_NEAR(weights.data[i][j], expected[i][j], kTol)
                << "at (" << i << ", " << j << ") with cols=" << cols;
}

INSTANTIATE_TEST_SUITE_P(ColumnWidths, MatrixSIMDGradientsTest,
    ::testing::Values(1, 7, 8, 9, 16, 17, 33, 64, 100, 256));

// ============================================================================
// TD-007: SIMD sum (horizontal reduction)
// ============================================================================

class MatrixSIMDSumTest : public ::testing::TestWithParam<std::pair<int,int>> {};

TEST_P(MatrixSIMDSumTest, SumMatchesReference) {
    auto [rows, cols] = GetParam();
    Matrix m = make_seq(rows, cols, 0.0f, 1.0f);

    float simd_sum     = m.sum();
    float expected_sum = ref_sum(m);

    // Use relative tolerance for large matrices where absolute error accumulates
    float tolerance = std::max(kTol, std::abs(expected_sum) * 1e-5f);
    EXPECT_NEAR(simd_sum, expected_sum, tolerance)
        << "rows=" << rows << " cols=" << cols;
}

INSTANTIATE_TEST_SUITE_P(Shapes, MatrixSIMDSumTest,
    ::testing::Values(
        std::make_pair(1,  1),
        std::make_pair(1,  7),
        std::make_pair(1,  8),
        std::make_pair(1,  9),
        std::make_pair(4,  16),
        std::make_pair(4,  17),
        std::make_pair(8,  32),
        std::make_pair(16, 64),
        std::make_pair(32, 128),
        std::make_pair(10, 100)
    ));

// ============================================================================
// TD-007: SIMD matrix multiply — ikj loop, various shapes
// ============================================================================

class MatrixSIMDMulTest : public ::testing::TestWithParam<std::tuple<int,int,int>> {};

TEST_P(MatrixSIMDMulTest, MulMatchesReference) {
    auto [M, K, N] = GetParam();
    // Use small values so accumulated sums stay well within float32 precision
    Matrix a = make_seq(M, K, 0.1f, 0.001f);
    Matrix b = make_seq(K, N, 0.1f, 0.001f);

    Matrix result   = a * b;
    Matrix expected = ref_mul(a, b);

    ASSERT_EQ(result.rows, M);
    ASSERT_EQ(result.cols, N);

    // Relative tolerance: errors due to different FP accumulation orders (ijk vs ikj)
    // scale with the magnitude of each expected value + a minimum absolute floor.
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float tol = std::max(kTol, std::abs(expected.data[i][j]) * 1e-4f);
            EXPECT_NEAR(result.data[i][j], expected.data[i][j], tol)
                << "at (" << i << ", " << j << ") M=" << M << " K=" << K << " N=" << N;
        }
}

// Cover: square, rectangular, non-multiples of 4/8, and small sizes
INSTANTIATE_TEST_SUITE_P(Shapes, MatrixSIMDMulTest,
    ::testing::Values(
        std::make_tuple(1,  1,  1),
        std::make_tuple(1,  8,  1),
        std::make_tuple(2,  3,  4),
        std::make_tuple(4,  4,  4),
        std::make_tuple(8,  8,  8),
        std::make_tuple(9,  9,  9),   // non-multiple of 8
        std::make_tuple(16, 16, 16),
        std::make_tuple(17, 17, 17),  // non-multiple of 8
        std::make_tuple(32, 64, 32),
        std::make_tuple(64, 32, 64),
        std::make_tuple(7,  13, 11),  // odd shapes
        std::make_tuple(64, 64, 64)
    ));

// ============================================================================
// TD-007: BLAS path — large matrix multiply (≥ 256 in all dims)
//         This test always passes; it only reaches the BLAS code path when
//         ADAI_ENABLE_BLAS is defined AND cblas_sgemm is available.
// ============================================================================

TEST(MatrixSIMDBLAS, LargeMatMulMatchesReference) {
    // Use 256x256 to trigger the BLAS threshold
    constexpr int SZ = 256;
    Matrix a(SZ, SZ);
    Matrix b(SZ, SZ);
    for (int i = 0; i < SZ; ++i)
        for (int j = 0; j < SZ; ++j) {
            a.data[i][j] = static_cast<float>((i + j) % 13) * 0.001f;
            b.data[i][j] = static_cast<float>((i * j + 1) % 7) * 0.001f;
        }

    Matrix result = a * b;
    Matrix ref    = ref_mul(a, b);

    // Relative tolerance per element
    for (int i = 0; i < SZ; ++i)
        for (int j = 0; j < SZ; ++j) {
            float tol = std::max(kTol, std::abs(ref.data[i][j]) * 1e-4f);
            EXPECT_NEAR(result.data[i][j], ref.data[i][j], tol)
                << "at (" << i << ", " << j << ")";
        }
}

// ============================================================================
// TD-007: Edge cases — 1×1 matrix, zero matrix, single row/column
// ============================================================================

TEST(MatrixSIMDEdgeCases, SingleElementAdd) {
    Matrix a(std::vector<std::vector<float>>{{3.0f}});
    Matrix b(std::vector<std::vector<float>>{{4.0f}});
    Matrix r = a + b;
    EXPECT_FLOAT_EQ(r.data[0][0], 7.0f);
}

TEST(MatrixSIMDEdgeCases, SingleElementMul) {
    Matrix a(std::vector<std::vector<float>>{{3.0f}});
    Matrix b(std::vector<std::vector<float>>{{4.0f}});
    Matrix r = a * b;
    EXPECT_FLOAT_EQ(r.data[0][0], 12.0f);
}

TEST(MatrixSIMDEdgeCases, SingleElementSum) {
    Matrix m(std::vector<std::vector<float>>{{5.5f}});
    EXPECT_FLOAT_EQ(m.sum(), 5.5f);
}

TEST(MatrixSIMDEdgeCases, ZeroMatrixMul) {
    Matrix a(4, 4);  // all zeros
    Matrix b = make_seq(4, 4);
    Matrix r = a * b;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_FLOAT_EQ(r.data[i][j], 0.0f);
}

TEST(MatrixSIMDEdgeCases, SingleRowMatrix) {
    Matrix a(1, 15);
    Matrix b(1, 15);
    for (int j = 0; j < 15; ++j) {
        a.data[0][j] = static_cast<float>(j + 1);
        b.data[0][j] = static_cast<float>(j + 1);
    }
    Matrix r = a + b;
    for (int j = 0; j < 15; ++j)
        EXPECT_FLOAT_EQ(r.data[0][j], 2.0f * static_cast<float>(j + 1));
}

TEST(MatrixSIMDEdgeCases, GradientUpdatePreservesOtherRows) {
    Matrix w = make_seq(4, 9, 1.0f, 0.1f);
    Matrix g = make_seq(4, 9, 0.0f, 0.0f);  // zero gradient
    Matrix w_copy = w;
    w.apply_gradients(g, 0.1f);
    // Zero gradient → weights unchanged
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 9; ++j)
            EXPECT_FLOAT_EQ(w.data[i][j], w_copy.data[i][j]);
}

// ============================================================================
// TD-007: Numerical stability — sum of alternating +1/-1 should cancel
// ============================================================================

TEST(MatrixSIMDNumerical, AlternatingSumNearZero) {
    constexpr int N = 128;
    Matrix m(1, N);
    for (int j = 0; j < N; ++j)
        m.data[0][j] = (j % 2 == 0) ? 1.0f : -1.0f;
    EXPECT_NEAR(m.sum(), 0.0f, 1e-5f);
}

TEST(MatrixSIMDNumerical, MeanAfterScaleConsistent) {
    Matrix m = make_seq(8, 32, 0.0f, 1.0f);
    float factor = 3.0f;
    Matrix scaled = m.scale(factor);
    EXPECT_NEAR(scaled.mean(), m.mean() * factor, kTol);
}

TEST(MatrixSIMDNumerical, SubtractionSelfIsZero) {
    Matrix m = make_seq(5, 17, 1.0f, 0.7f);
    Matrix r = m - m;
    EXPECT_NEAR(r.sum(), 0.0f, 1e-5f);
    for (int i = 0; i < r.rows; ++i)
        for (int j = 0; j < r.cols; ++j)
            EXPECT_FLOAT_EQ(r.data[i][j], 0.0f);
}
