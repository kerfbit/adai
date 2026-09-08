/**
 * @file quantization_test.cpp
 * @brief Unit tests for Quantization.hpp (Quantizer, QuantizedMatrix)
 *
 * Previously untested (no test file existed for this component). Added
 * alongside the TD-075 fix for ASYMMETRIC_INT8's int8_t storage/dequantize
 * sign-extension bug.
 */
#include <gtest/gtest.h>
#include <cmath>
#include "../src/Quantization.hpp"

namespace {
float max_abs_error(const std::vector<float>& original, const std::vector<float>& dequantized) {
    float max_err = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
        max_err = std::max(max_err, std::abs(original[i] - dequantized[i]));
    }
    return max_err;
}
}  // namespace

// ============================================================================
// SYMMETRIC_INT8 — baseline sanity (already correct before TD-075)
// ============================================================================

TEST(QuantizerTest, SymmetricInt8RoundTripIsAccurate) {
    Quantizer q(QuantizationMode::SYMMETRIC_INT8, CalibrationMethod::MIN_MAX);
    std::vector<float> data = {-20.0f, -5.0f, 0.0f, 5.0f, 20.0f};

    auto params = q.calibrate(data);
    auto quantized = q.quantize(data, params);
    auto dequantized = q.dequantize(quantized, params);

    EXPECT_LT(max_abs_error(data, dequantized), 0.2f);
}

// ============================================================================
// TD-075 regression: ASYMMETRIC_INT8's quantized range [0,255] doesn't fit
// in the signed int8_t storage used by quantize()/dequantize(). Values that
// quantize above 127 used to come back sign-extended (and often negative)
// instead of the true value, once dequantize() implicitly widened the
// stored int8_t back to int.
// ============================================================================

TEST(QuantizerTest, AsymmetricInt8RoundTripAboveMidpointIsAccurate) {
    Quantizer q(QuantizationMode::ASYMMETRIC_INT8, CalibrationMethod::MIN_MAX);
    // Post-ReLU-style activations: all non-negative, spanning the full
    // range — exactly the case asymmetric quantization exists for, and the
    // case that exposed the bug (values quantizing above 127).
    std::vector<float> data = {0.0f, 5.0f, 10.0f, 15.0f, 20.0f};

    auto params = q.calibrate(data);
    auto quantized = q.quantize(data, params);
    auto dequantized = q.dequantize(quantized, params);

    ASSERT_EQ(dequantized.size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        // Before the fix, values quantizing above 127 (here: 15.0 and 20.0)
        // dequantized to negative numbers with error > 20.0.
        EXPECT_NEAR(data[i], dequantized[i], 0.2f) << "at index " << i;
    }
}

TEST(QuantizerTest, AsymmetricInt8SingleValueQuantizedAbove127DoesNotGoNegative) {
    Quantizer q(QuantizationMode::ASYMMETRIC_INT8, CalibrationMethod::MIN_MAX);
    std::vector<float> data = {0.0f, 20.0f};  // calibration data fixing the scale
    auto params = q.calibrate(data);

    // 20.0f is the max of the calibration data, so it quantizes to qmax=255 —
    // squarely in the region that doesn't fit in a signed int8_t.
    int true_q = q.quantize_value(20.0f, params);
    EXPECT_EQ(true_q, 255);

    auto quantized = q.quantize({20.0f}, params);
    auto dequantized = q.dequantize(quantized, params);
    EXPECT_NEAR(dequantized[0], 20.0f, 0.1f);
}

TEST(QuantizerTest, QuantizedMatrixRoundTripAsymmetricInt8) {
    Matrix m(2, 2);
    m(0, 0) = 0.0f;
    m(0, 1) = 8.0f;
    m(1, 0) = 15.0f;
    m(1, 1) = 20.0f;

    Quantizer q(QuantizationMode::ASYMMETRIC_INT8, CalibrationMethod::MIN_MAX);
    QuantizedMatrix qm;
    qm.quantize_from(m, q);
    Matrix dequantized = qm.dequantize(q);

    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            EXPECT_NEAR(m(r, c), dequantized(r, c), 0.2f) << "at (" << r << "," << c << ")";
        }
    }
}

// ============================================================================
// Other modes — sanity that the fix didn't disturb already-correct paths
// ============================================================================

TEST(QuantizerTest, AsymmetricInt4RoundTripIsAccurate) {
    Quantizer q(QuantizationMode::ASYMMETRIC_INT4, CalibrationMethod::MIN_MAX);
    std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f};

    auto params = q.calibrate(data);
    auto quantized = q.quantize(data, params);
    auto dequantized = q.dequantize(quantized, params);

    EXPECT_LT(max_abs_error(data, dequantized), 0.5f);
}

TEST(QuantizerTest, SymmetricInt4RoundTripIsAccurate) {
    Quantizer q(QuantizationMode::SYMMETRIC_INT4, CalibrationMethod::MIN_MAX);
    std::vector<float> data = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};

    auto params = q.calibrate(data);
    auto quantized = q.quantize(data, params);
    auto dequantized = q.dequantize(quantized, params);

    EXPECT_LT(max_abs_error(data, dequantized), 0.5f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
