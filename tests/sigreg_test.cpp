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
#include <vector>
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

    // TD-503: unbiased, so zero in expectation and possibly slightly negative.
    EXPECT_LT(std::abs(loss), 0.02f) << "loss on a genuinely isotropic Gaussian batch should be small";
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

// ============================================================================
// compute_variance_stats() — advanced LeJEPA metrics diagnostic, distinct from compute_loss()'s
// characteristic-function test: per-direction projected variance (mean + stddev across
// directions), computed on the same projections compute_loss() forms internally.
// ============================================================================

TEST(SIGRegTest, ComputeVarianceStatsRejectsMismatchedDModel) {
    SIGReg sigreg(16, 32);
    Matrix wrong_shape(10, 8);
    EXPECT_THROW(sigreg.compute_variance_stats(wrong_shape), std::invalid_argument);
}

TEST(SIGRegTest, ComputeVarianceStatsEmptyBatchReturnsZero) {
    SIGReg sigreg(16, 32);
    Matrix empty(0, 16);
    auto stats = sigreg.compute_variance_stats(empty);
    EXPECT_FLOAT_EQ(stats.mean, 0.0f);
    EXPECT_FLOAT_EQ(stats.stddev, 0.0f);
}

TEST(SIGRegTest, ComputeVarianceStatsIsotropicBatchHasMeanNearOneAndLowStddev) {
    const int d_model = 16;
    const int batch = 4000;  // same rationale as LossNearZeroForIsotropicGaussianBatch — large
                              // enough that Monte-Carlo noise stays small
    SIGReg sigreg(d_model, /*num_sketches=*/64);

    Matrix gaussian_batch = make_isotropic_gaussian_batch(batch, d_model, /*seed=*/123);
    auto stats = sigreg.compute_variance_stats(gaussian_batch);

    EXPECT_NEAR(stats.mean, 1.0f, 0.1f)
        << "every direction's projected variance should be ~1.0 for a genuine N(0,I) batch";
    EXPECT_LT(stats.stddev, 0.15f)
        << "variance should be roughly uniform across directions for an isotropic batch";
}

TEST(SIGRegTest, ComputeVarianceStatsCollapsedBatchHasNearZeroMean) {
    const int d_model = 16;
    SIGReg sigreg(d_model, /*num_sketches=*/64);

    Matrix collapsed_batch = make_collapsed_batch(50, d_model);
    auto stats = sigreg.compute_variance_stats(collapsed_batch);

    // Every row identical -> zero variance along every projection direction.
    EXPECT_NEAR(stats.mean, 0.0f, 1e-4f);
    EXPECT_NEAR(stats.stddev, 0.0f, 1e-4f);
}

TEST(SIGRegTest, ComputeVarianceStatsAnisotropicBatchHasLargerStddevThanIsotropic) {
    const int d_model = 16;
    const int batch = 4000;
    SIGReg sigreg(d_model, /*num_sketches=*/64);

    // High variance along dimension 0 only, ~zero elsewhere — the aggregate compute_loss() scalar
    // can't distinguish this from "globally slightly off," but compute_variance_stats()'s stddev
    // should clearly separate it from the genuinely isotropic case above.
    Matrix anisotropic_batch(batch, d_model);
    std::mt19937 gen(99);
    std::normal_distribution<float> wide_dist(0.0f, 10.0f);
    for (int i = 0; i < batch; ++i) {
        anisotropic_batch.data[i][0] = wide_dist(gen);
        for (int j = 1; j < d_model; ++j) {
            anisotropic_batch.data[i][j] = 0.0f;
        }
    }

    auto isotropic_stats =
        sigreg.compute_variance_stats(make_isotropic_gaussian_batch(batch, d_model, 123));
    auto anisotropic_stats = sigreg.compute_variance_stats(anisotropic_batch);

    EXPECT_GT(anisotropic_stats.stddev, 3.0f * isotropic_stats.stddev);
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

// ============================================================================
// TD-502: directions redrawn per step; TD-503: unbiased estimator
// ============================================================================

namespace {

// x projected onto the span of `directions`' columns (Gram-Schmidt basis): the batch keeps its
// Gaussian spread inside that span and is collapsed in every other dimension.
Matrix project_onto_span(const Matrix& x, const Matrix& directions) {
    const int d = directions.rows, k = directions.cols;
    std::vector<std::vector<double>> basis;
    for (int c = 0; c < k; ++c) {
        std::vector<double> v(d);
        for (int i = 0; i < d; ++i)
            v[i] = directions.data[i][c];
        for (const auto& b : basis) {
            double dot = 0.0;
            for (int i = 0; i < d; ++i)
                dot += v[i] * b[i];
            for (int i = 0; i < d; ++i)
                v[i] -= dot * b[i];
        }
        double norm = 0.0;
        for (double e : v)
            norm += e * e;
        norm = std::sqrt(norm);
        if (norm < 1e-9)
            continue;
        for (double& e : v)
            e /= norm;
        basis.push_back(std::move(v));
    }
    Matrix out(x.rows, d);
    for (int r = 0; r < x.rows; ++r)
        for (const auto& b : basis) {
            double dot = 0.0;
            for (int i = 0; i < d; ++i)
                dot += x.data[r][i] * b[i];
            for (int i = 0; i < d; ++i)
                out.data[r][i] += static_cast<float>(dot * b[i]);
        }
    return out;
}

}  // namespace

// TD-502: one draw constrains only its own span, so a batch collapsed onto it scores the same as
// the uncollapsed batch. Redrawing the directions (as LeJEPAEncoder::train_step() now does every
// step) exposes the collapse. With directions fixed for the whole process, as before, it never was.
TEST(SIGRegTest, RedrawnDirectionsPenalizeCollapseOutsideOneDrawsSpan) {
    const int d_model = 64, sketches = 8;
    SIGReg sigreg(d_model, sketches);
    sigreg.set_seed(1);
    Matrix gaussian = make_isotropic_gaussian_batch(512, d_model, /*seed=*/3);
    Matrix collapsed = project_onto_span(gaussian, sigreg.get_directions());

    EXPECT_NEAR(sigreg.compute_loss(collapsed), sigreg.compute_loss(gaussian), 1e-4f)
        << "within one draw the collapse is invisible";

    float gaussian_sum = 0.0f, collapsed_sum = 0.0f;
    for (int step = 0; step < 20; ++step) {
        sigreg.resample_directions();
        gaussian_sum += sigreg.compute_loss(gaussian);
        collapsed_sum += sigreg.compute_loss(collapsed);
    }
    EXPECT_GT(collapsed_sum / 20.0f, 0.05f);
    EXPECT_GT(collapsed_sum, 10.0f * std::abs(gaussian_sum));
}

TEST(SIGRegTest, SetSeedMakesTheDirectionsReproducible) {
    SIGReg a(16, 8), b(16, 8);
    a.set_seed(42);
    b.set_seed(42);
    Matrix batch = make_isotropic_gaussian_batch(64, 16, /*seed=*/5);
    EXPECT_EQ(a.compute_loss(batch), b.compute_loss(batch));
    a.resample_directions();
    b.resample_directions();
    EXPECT_EQ(a.compute_loss(batch), b.compute_loss(batch));
}

// TD-503: the plain squared gap had an expected floor of (1 - |phi|^2)/n, so a true Gaussian of 8
// rows (one short text's tokens) scored about 0.079 and 128 rows about 0.006: the loss tracked
// sequence length. The unbiased estimator is about 0 at every n, while collapse still scores high.
TEST(SIGRegTest, UnbiasedLossDoesNotDependOnSequenceLength) {
    const int d_model = 16;
    SIGReg sigreg(d_model, 16);
    sigreg.set_seed(9);
    for (int n : {8, 128}) {
        double sum = 0.0;
        const int trials = 300;
        for (int t = 0; t < trials; ++t) {
            sigreg.resample_directions();
            sum += sigreg.compute_loss(
                make_isotropic_gaussian_batch(n, d_model, static_cast<unsigned>(1000 * n + t)));
        }
        EXPECT_LT(std::abs(sum / trials), 0.01) << "n=" << n;
    }
    EXPECT_GT(sigreg.compute_loss(make_collapsed_batch(8, d_model)), 0.2f);
}

TEST(SIGRegTest, FewerThanTwoRowsScoreZeroWithZeroGradient) {
    SIGReg sigreg(8, 4);
    Matrix one = make_isotropic_gaussian_batch(1, 8, /*seed=*/1);
    EXPECT_EQ(sigreg.compute_loss(one), 0.0f);
    Matrix grad = sigreg.backward(one);
    ASSERT_EQ(grad.rows, 1);
    for (int j = 0; j < 8; ++j)
        EXPECT_EQ(grad.data[0][j], 0.0f);
}


#ifdef ADAI_ENABLE_GPU
// TD-586: SIGReg on the device against the host SIGReg, on the oneAPI CPU device in a SYCL build
// (ADAI_SYCL_ALLOW_CPU_DEVICE=1); skipped without a device.
#include <random>
#include "../src/gpu/MatrixGPU.hpp"

namespace {

bool sigreg_gpu_ready() {
    return adai::gpu::GPUManager::probe() && adai::gpu::GPUManager::initialize(0, 0.5f);
}

Matrix normal_matrix(int rows, int cols, unsigned seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, scale);
    Matrix m(rows, cols);
    for (auto& row : m.data)
        for (auto& v : row)
            v = d(rng);
    return m;
}

float max_abs_diff(const Matrix& a, const Matrix& b) {
    float worst = 0.0f;
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
    return worst;
}

float max_abs(const Matrix& a) {
    float worst = 0.0f;
    for (const auto& row : a.data)
        for (float v : row)
            worst = std::max(worst, std::abs(v));
    return worst;
}

}  // namespace

// The direction kernel draws what the host draws for the same seed and draw number, up to float rounding:
// icpx's default fast floating-point model approximates log/cos on the device (seen: 5e-5 on entries of
// about 0.1), so the backends agree closely but not bit for bit.
TEST(SIGRegGpuTest, DeviceDirectionsMatchTheHost) {
    if (!sigreg_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d_model = 96, sketches = 40;
    SIGReg sigreg(d_model, sketches);
    sigreg.set_seed(17);  // draw number 1
    adai::gpu::GPUMatrix device(d_model, sketches);
    adai::gpu::sigreg_directions_gpu(device.device_ptr(), d_model, sketches, 17, 1);
    EXPECT_LT(max_abs_diff(Matrix::from_gpu(device), sigreg.get_directions()), 2e-4f);
}

// Loss, variance stats and gradient from gpu_compute() match compute_loss(), compute_variance_stats() and
// backward(), from a single draw, at batch sizes around the estimator's minimum and the work-group size.
TEST(SIGRegGpuTest, GpuComputeMatchesTheHost) {
    if (!sigreg_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d_model = 64, sketches = 64;
    SIGReg sigreg(d_model, sketches);
    sigreg.set_seed(5);
    adai::gpu::GPUMemory<float> out(3);
    for (int batch : {1, 2, 32, 300}) {
        SCOPED_TRACE("batch=" + std::to_string(batch));
        // Not a standard normal, so the loss and gradient are far from 0.
        const Matrix x = normal_matrix(batch, d_model, 100 + batch, 2.5f);
        adai::gpu::GPUMatrix grad(1, 1);
        sigreg.gpu_compute(x.to_gpu(), out.get(), &grad);
        float got[3];
        out.copy_to_host(got, 3);
        const float want_loss = sigreg.compute_loss(x);
        const auto want_stats = sigreg.compute_variance_stats(x);
        const Matrix want_grad = sigreg.backward(x);
        EXPECT_NEAR(got[0], want_loss, 1e-4f * std::max(1.0f, std::abs(want_loss)));
        EXPECT_NEAR(got[1], want_stats.mean, 1e-3f * std::max(1.0f, want_stats.mean));
        EXPECT_NEAR(got[2], want_stats.stddev, 1e-3f * std::max(1.0f, want_stats.stddev));
        ASSERT_EQ(grad.rows, batch);
        ASSERT_EQ(grad.cols, d_model);
        // The directions differ in the last digits (above), and batch 2 doubles that through the unbiased factor.
        EXPECT_LT(max_abs_diff(Matrix::from_gpu(grad), want_grad), 5e-3f * max_abs(want_grad) + 1e-7f);
        if (batch == 1) {
            EXPECT_EQ(got[0], 0.0f) << "one row: no estimate, as on the host";
            EXPECT_EQ(max_abs(Matrix::from_gpu(grad)), 0.0f);
        }
    }
}

// A resample moves the device to the new draw too, without the host building it.
TEST(SIGRegGpuTest, ResampleMovesTheDeviceToTheNextDraw) {
    if (!sigreg_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    const int d_model = 32, sketches = 16;
    SIGReg sigreg(d_model, sketches);
    sigreg.set_seed(9);
    const Matrix x = normal_matrix(20, d_model, 7, 1.5f);
    adai::gpu::GPUMemory<float> out(3);
    float first[3], second[3];
    sigreg.gpu_compute(x.to_gpu(), out.get());
    out.copy_to_host(first, 3);
    sigreg.resample_directions();
    sigreg.gpu_compute(x.to_gpu(), out.get());
    out.copy_to_host(second, 3);
    EXPECT_NE(first[0], second[0]) << "a new draw scores differently";
    EXPECT_NEAR(second[0], sigreg.compute_loss(x), 1e-4f * std::max(1.0f, std::abs(second[0])));
}

TEST(SIGRegGpuTest, BadInputThrows) {
    if (!sigreg_gpu_ready())
        GTEST_SKIP() << "No GPU device present.";
    SIGReg sigreg(16, 8);
    adai::gpu::GPUMemory<float> out(3);
    EXPECT_THROW(sigreg.gpu_compute(normal_matrix(4, 15, 1, 1.0f).to_gpu(), out.get()), std::invalid_argument);
}
#endif  // ADAI_ENABLE_GPU
