/**
 * @file dropout_test.cpp
 * @brief TD-344: the transformer blocks stored a dropout rate (0.1) but never applied it. Covers
 * the Dropout component, the blocks' backward passes with dropout on (finite differences, the masks
 * replayed from a seed), and that a model drops only in training mode after set_dropout(). The GPU
 * path is checked against the CPU in gpu_optimizer_test.cpp.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "../src/DecoderBlock.hpp"
#include "../src/Dropout.hpp"
#include "../src/DropoutMask.hpp"
#include "../src/EncoderBlock.hpp"
#include "../src/EncoderDecoderModel.hpp"
#include "../src/Matrix.hpp"

namespace {

Matrix random_matrix(int rows, int cols, std::mt19937& rng, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(rows, cols);
    for (auto& row : m.data)
        for (auto& x : row)
            x = d(rng);
    return m;
}

Matrix filled(int rows, int cols, float value) {
    Matrix m(rows, cols);
    for (auto& row : m.data)
        for (auto& x : row)
            x = value;
    return m;
}

Matrix causal(int n) {
    Matrix m(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m(i, j) = (j <= i) ? 1.0f : 0.0f;
    return m;
}

float dot(const Matrix& a, const Matrix& b) {
    double s = 0.0;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            s += static_cast<double>(a(i, j)) * b(i, j);
    return static_cast<float>(s);
}

}  // namespace

TEST(DropoutTest, OffChangesNothing) {
    Dropout off;
    off.set_rate(0.5f);  // a rate, but not active
    Matrix x = filled(4, 8, 2.0f);
    off.forward(x);
    EXPECT_FLOAT_EQ(x(3, 7), 2.0f);
    Dropout zero;
    zero.set_rate(0.0f);
    zero.set_active(true);  // active, but rate 0
    zero.forward(x);
    EXPECT_FLOAT_EQ(x(0, 0), 2.0f);
    Matrix g = filled(4, 8, 1.0f);
    zero.backward(g);
    EXPECT_FLOAT_EQ(g(1, 1), 1.0f);
}

TEST(DropoutTest, DropsAboutTheRateAndScalesTheRest) {
    Dropout d;
    d.set_rate(0.3f);
    d.set_active(true);
    d.seed(7);
    Matrix x = filled(100, 100, 1.0f);
    d.forward(x);
    int dropped = 0;
    double sum = 0.0;
    for (const auto& row : x.data) {
        for (float v : row) {
            if (v == 0.0f) {
                ++dropped;
            } else {
                EXPECT_NEAR(v, 1.0f / 0.7f, 1e-6f);
            }
            sum += v;
        }
    }
    EXPECT_NEAR(dropped / 10000.0, 0.3, 0.02);
    EXPECT_NEAR(sum / 10000.0, 1.0, 0.03) << "inverted dropout keeps the expected value";
}

TEST(DropoutTest, BackwardUsesTheForwardMask) {
    Dropout d;
    d.set_rate(0.5f);
    d.set_active(true);
    Matrix x = filled(6, 6, 1.0f);
    d.forward(x);
    Matrix g = filled(6, 6, 3.0f);
    d.backward(g);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_FLOAT_EQ(g(i, j), 3.0f * x(i, j)) << i << "," << j;
    Matrix wrong = filled(5, 6, 1.0f);
    EXPECT_THROW(d.backward(wrong), std::invalid_argument);
}

TEST(DropoutTest, SeedReplaysTheMaskAndCallsDiffer) {
    Dropout a, b;
    for (Dropout* d : {&a, &b}) {
        d->set_rate(0.5f);
        d->set_active(true);
        d->seed(99);
    }
    Matrix xa = filled(8, 8, 1.0f), xb = filled(8, 8, 1.0f);
    a.forward(xa);
    b.forward(xb);
    EXPECT_EQ(xa.data, xb.data);
    Matrix again = filled(8, 8, 1.0f);
    a.forward(again);  // the next call draws a new seed
    EXPECT_NE(again.data, xa.data);
}

TEST(DropoutTest, RejectsRatesOutsideZeroToOne) {
    Dropout d;
    EXPECT_THROW(d.set_rate(-0.1f), std::invalid_argument);
    EXPECT_THROW(d.set_rate(1.0f), std::invalid_argument);
    EXPECT_THROW(d.set_rate(std::nanf("")), std::invalid_argument);
    EXPECT_NO_THROW(d.set_rate(0.99f));
}

TEST(DropoutTest, MaskValueIsAPureFunctionOfSeedAndIndex) {
    EXPECT_EQ(adai::dropout_mask_value(5, 17, 0.5f), adai::dropout_mask_value(5, 17, 0.5f));
    int kept = 0;
    for (std::uint64_t i = 0; i < 20000; ++i)
        kept += adai::dropout_mask_value(123, i, 0.8f) > 0.0f ? 1 : 0;
    EXPECT_NEAR(kept / 20000.0, 0.8, 0.01);
    EXPECT_FLOAT_EQ(adai::dropout_mask_value(1, 1, 1.0f), 1.0f) << "keep_prob 1 keeps everything";
}

// The blocks' backward passes with dropout on, against finite differences. Each forward replays the
// same masks (seed_dropout()), so the loss is a fixed function of the input.
TEST(BlockDropoutGradientTest, EncoderBlockInputGradientMatchesFiniteDifferences) {
    std::mt19937 rng(11);
    EncoderBlock block(8, 2, 16, 0.1f);
    block.set_dropout(0.3f, true);
    Matrix input = random_matrix(4, 8, rng, 0.5f), dout = random_matrix(4, 8, rng, 0.5f);
    block.seed_dropout(42);
    const Matrix out = block.forward(input);
    const Matrix grad = block.backward(dout);

    int checked = 0;
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; j += 3, ++checked) {
            const float eps = 1e-2f;
            Matrix plus = input, minus = input;
            plus(i, j) += eps;
            minus(i, j) -= eps;
            block.seed_dropout(42);
            const float lp = dot(block.forward(plus), dout);
            block.seed_dropout(42);
            const float lm = dot(block.forward(minus), dout);
            const float fd = (lp - lm) / (2 * eps);  // float differences: ~1% at |grad| ~ 10-50
            EXPECT_NEAR(grad(i, j), fd, 2e-2f * std::max(1.0f, std::abs(fd))) << i << "," << j;
        }
    }
    EXPECT_GT(checked, 0);
    // And the dropout really was on: a forward without it differs.
    block.set_dropout(0.3f, false);
    EXPECT_NE(block.forward(input).data, out.data);
}

TEST(BlockDropoutGradientTest, DecoderBlockInputGradientMatchesFiniteDifferences) {
    std::mt19937 rng(12);
    DecoderBlock block(8, 2, 16, 0.1f);
    block.set_dropout(0.3f, true);
    Matrix input = random_matrix(3, 8, rng, 0.5f), enc = random_matrix(4, 8, rng, 0.5f);
    Matrix dout = random_matrix(3, 8, rng, 0.5f);
    const Matrix mask = causal(3);
    block.seed_dropout(7);
    const Matrix out = block.forward(input, enc, mask);
    const Matrix grad = block.backward(dout);

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; j += 3) {
            const float eps = 1e-2f;
            Matrix plus = input, minus = input;
            plus(i, j) += eps;
            minus(i, j) -= eps;
            block.seed_dropout(7);
            const float lp = dot(block.forward(plus, enc, mask), dout);
            block.seed_dropout(7);
            const float lm = dot(block.forward(minus, enc, mask), dout);
            const float fd = (lp - lm) / (2 * eps);  // float differences: ~1% at |grad| ~ 10-50
            EXPECT_NEAR(grad(i, j), fd, 2e-2f * std::max(1.0f, std::abs(fd))) << i << "," << j;
        }
    }
    block.set_dropout(0.3f, false);
    EXPECT_NE(block.forward(input, enc, mask).data, out.data);
}

// A model drops nothing until set_dropout(), and only in training mode after it.
TEST(ModelDropoutTest, OnlyAfterSetDropoutAndOnlyInTrainingMode) {
    EncoderDecoderModel model(40, 16, 1, 1, 2, 32, 16);
    const std::vector<int> in = {4, 5, 6, 7}, tgt = {8, 9, 10};
    model.set_training(true);
    EXPECT_EQ(model.forward(in, tgt).data, model.forward(in, tgt).data)
        << "no dropout without set_dropout(), even in training mode";

    model.set_dropout(0.5f);
    EXPECT_FLOAT_EQ(model.get_dropout(), 0.5f);
    EXPECT_NE(model.forward(in, tgt).data, model.forward(in, tgt).data) << "training: dropped";

    model.set_training(false);
    const Matrix eval = model.forward(in, tgt);
    EXPECT_EQ(model.forward(in, tgt).data, eval.data) << "evaluation: never dropped";

    model.set_training(true);
    model.set_dropout(0.0f);
    EXPECT_EQ(model.forward(in, tgt).data, eval.data) << "rate 0 is the evaluation output";

    EXPECT_THROW(model.set_dropout(1.0f), std::invalid_argument);
    EXPECT_THROW(model.set_dropout(-0.5f), std::invalid_argument);
}
