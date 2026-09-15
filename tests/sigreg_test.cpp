/**
 * @file sigreg_test.cpp
 * @brief Tests for SIGReg (TD-175 / LJ-1a) — Sketched Isotropic Gaussian Regularization, from
 *        docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 3.
 *
 * Three things this item's own Action Items ask for: loss near zero on a synthetic isotropic
 * Gaussian batch, loss large on a degenerate (collapsed) batch, and a finite-difference gradient
 * check against backward()'s analytic gradient.
 */

#include "../src/SIGReg.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include "../src/Matrix.hpp"

namespace {

// A real batch of i.i.d. standard normal samples — genuinely isotropic Gaussian, not a
// hand-constructed approximation, so this test exercises the same statistical assumption
// compute_loss() is designed to detect.
Matrix make_isotropic_gaussian_batch(int batch, int d_model, unsigned seed) {
    Matrix m(batch, d_model);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            m.data[i][j] = dist(gen);
        }
    }
    return m;
}

// A collapsed batch: every row identical to the same fixed, nonzero vector — the canonical
// representation-collapse failure mode SIGReg exists to penalize (every embedding maps to the
// same point, so no 1D projection of the batch can look like a genuine spread-out N(0,1)).
Matrix make_collapsed_batch(int batch, int d_model) {
    Matrix m(batch, d_model);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            // Deliberately not all-zero (a degenerate-at-zero batch would still trivially satisfy
            // "no spread," but varying the fixed value across dimensions makes this a more
            // general collapse case, and keeps every row's projection nonzero for effectively
            // every random direction, avoiding a coincidental accidental zero-crossing).
            m.data[i][j] = 1.5f + 0.1f * static_cast<float>(j);
        }
    }
    return m;
}

}  // namespace

TEST(SIGRegTest, ConstructorRejectsNonPositiveDimensions) {
    EXPECT_THROW(SIGReg(0, 64), std::invalid_argument);
    EXPECT_THROW(SIGReg(-4, 64), std::invalid_argument);
    EXPECT_THROW(SIGReg(16, 0), std::invalid_argument);
    EXPECT_THROW(SIGReg(16, -1), std::invalid_argument);
}

TEST(SIGRegTest, AccessorsReturnConstructorArguments) {
    SIGReg sigreg(32, 48);
    EXPECT_EQ(sigreg.get_d_model(), 32);
    EXPECT_EQ(sigreg.get_num_sketches(), 48);
}

TEST(SIGRegTest, ComputeLossRejectsMismatchedDModel) {
    SIGReg sigreg(16, 32);
    Matrix wrong_shape(10, 8);
    EXPECT_THROW(sigreg.compute_loss(wrong_shape), std::invalid_argument);
    EXPECT_THROW(sigreg.backward(wrong_shape), std::invalid_argument);
}

TEST(SIGRegTest, EmptyBatchIsHandledWithoutCrashing) {
    SIGReg sigreg(16, 32);
    Matrix empty(0, 16);
    EXPECT_FLOAT_EQ(sigreg.compute_loss(empty), 0.0f);
    Matrix grad = sigreg.backward(empty);
    EXPECT_EQ(grad.rows, 0);
    EXPECT_EQ(grad.cols, 16);
}

TEST(SIGRegTest, LossNearZeroForIsotropicGaussianBatch) {
    const int d_model = 16;
    const int batch = 4000;  // large enough that Monte-Carlo noise in the empirical
                              // characteristic function stays small relative to the
                              // degenerate-batch loss checked below
    SIGReg sigreg(d_model, /*num_sketches=*/64);

    Matrix gaussian_batch = make_isotropic_gaussian_batch(batch, d_model, /*seed=*/42);
    float loss = sigreg.compute_loss(gaussian_batch);

    EXPECT_GE(loss, 0.0f);
    EXPECT_LT(loss, 0.02f) << "loss on a genuinely isotropic Gaussian batch should be small";
}

TEST(SIGRegTest, LossLargeForDegenerateBatch) {
    const int d_model = 16;
    SIGReg sigreg(d_model, /*num_sketches=*/64);

    Matrix gaussian_batch = make_isotropic_gaussian_batch(2000, d_model, /*seed=*/7);
    Matrix collapsed_batch = make_collapsed_batch(50, d_model);

    float gaussian_loss = sigreg.compute_loss(gaussian_batch);
    float collapsed_loss = sigreg.compute_loss(collapsed_batch);

    // The whole point of the regularizer: a collapsed batch must score unambiguously worse than
    // a genuinely isotropic one, by a wide margin, not just "somewhat higher."
    EXPECT_GT(collapsed_loss, 10.0f * gaussian_loss);
    EXPECT_GT(collapsed_loss, 0.1f);
}

TEST(SIGRegTest, GradientMatchesFiniteDifference) {
    const int d_model = 5;
    const int batch = 6;
    SIGReg sigreg(d_model, /*num_sketches=*/16);

    // Fixed, deterministic, moderately-sized values — not drawn from randomize() — so the
    // central-difference estimate below is numerically well-behaved (the loss is smooth in the
    // embeddings everywhere, but large values combined with the largest frequency (t=2.0) could
    // make a coarse finite-difference step noisier than necessary).
    Matrix embeddings(batch, d_model);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embeddings.data[i][j] = dist(gen);
        }
    }

    Matrix analytic_grad = sigreg.backward(embeddings);
    ASSERT_EQ(analytic_grad.rows, batch);
    ASSERT_EQ(analytic_grad.cols, d_model);

    const float epsilon = 1e-3f;
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < d_model; ++j) {
            const float orig = embeddings.data[i][j];

            embeddings.data[i][j] = orig + epsilon;
            const float loss_plus = sigreg.compute_loss(embeddings);

            embeddings.data[i][j] = orig - epsilon;
            const float loss_minus = sigreg.compute_loss(embeddings);

            embeddings.data[i][j] = orig;

            const float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            EXPECT_NEAR(analytic_grad.data[i][j], numerical_grad, 5e-3f)
                << "mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(SIGRegTest, DifferentInstancesUseIndependentDirectionsButAgreeOnStatisticalBehavior) {
    // Two independently-constructed SIGReg instances get different random directions (no shared
    // seed in the interface), but both must still correctly separate a Gaussian batch from a
    // collapsed one — the qualitative behavior shouldn't depend on which particular random
    // directions happened to be sampled.
    const int d_model = 12;
    SIGReg sigreg_a(d_model, 64);
    SIGReg sigreg_b(d_model, 64);

    Matrix gaussian_batch = make_isotropic_gaussian_batch(3000, d_model, /*seed=*/99);
    Matrix collapsed_batch = make_collapsed_batch(50, d_model);

    EXPECT_LT(sigreg_a.compute_loss(gaussian_batch), sigreg_a.compute_loss(collapsed_batch));
    EXPECT_LT(sigreg_b.compute_loss(gaussian_batch), sigreg_b.compute_loss(collapsed_batch));
}
