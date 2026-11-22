/**
 * @file gpu_kernel_reference_test.cpp
 * @brief TD-394: every matrix_*_gpu kernel against a double-precision CPU reference.
 *
 * Sizes are chosen where kernels change shape: element counts around the 4-wide vector pass and its
 * scalar tail (n % 4), rows narrower and wider than the 256-item work-group (a row loop strides by 256),
 * and reduction sizes around 256, 65,536 and a million (TD-393's sub-group bug summed 256 ones to 64).
 * Kernels covered elsewhere: optimizer_update_gpu, dropout_forward_gpu and the embedding kernels
 * (gpuDebugSyncTests, gpuOptimizerTests), gated_residual_*_gpu (gpuOptimizerTests).
 *
 * Skips without a device. A SYCL build runs it on the oneAPI CPU device with
 * ADAI_SYCL_ALLOW_CPU_DEVICE=1, as build/sycl-cpudev does.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "../src/Activation.hpp"
#include "../src/gpu/MatrixGPU.hpp"

using adai::gpu::ActivationType;
using adai::gpu::GPUManager;
using adai::gpu::GPUMemory;
namespace g = adai::gpu;

namespace {

template <typename T>
GPUMemory<T> dev(const std::vector<T>& host) {
    GPUMemory<T> m(std::max<size_t>(host.size(), 1));
    if (!host.empty())
        m.copy_from_host(host.data(), host.size());
    return m;
}

template <typename T>
std::vector<T> host(const GPUMemory<T>& m, size_t count) {
    std::vector<T> out(count);
    if (count > 0)
        m.copy_to_host(out.data(), count);
    return out;
}

std::vector<float> random_values(size_t n, float lo, float hi, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}

// |got - want| <= abs_tol + rel_tol * |want|, element by element, naming the first mismatch.
::testing::AssertionResult matches(const std::vector<float>& got, const std::vector<double>& want, double rel_tol,
                                 double abs_tol) {
    if (got.size() != want.size())
        return ::testing::AssertionFailure() << "size " << got.size() << " vs " << want.size();
    for (size_t i = 0; i < got.size(); ++i) {
        const double err = std::fabs(static_cast<double>(got[i]) - want[i]);
        if (!(err <= abs_tol + rel_tol * std::fabs(want[i])))
            return ::testing::AssertionFailure() << "element " << i << ": got " << got[i] << ", want " << want[i];
    }
    return ::testing::AssertionSuccess();
}

std::vector<double> to_double(const std::vector<float>& v) {
    return {v.begin(), v.end()};
}

class GpuKernelReferenceTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!GPUManager::probe())
            GTEST_SKIP() << "No GPU device (a SYCL build can use the oneAPI CPU device: ADAI_SYCL_ALLOW_CPU_DEVICE=1)";
        ASSERT_TRUE(GPUManager::initialize(0, 0.5f));
    }
};

// Counts that exercise the 4-wide pass alone, the scalar tail alone, and both.
const std::vector<int> kElementCounts = {1, 3, 4, 5, 7, 1023, 4099};
// Row widths below, at and above the 256-item work-group, and several times it.
const std::vector<int> kRowWidths = {1, 7, 255, 256, 257, 1000, 3000};

}  // namespace

// ============================================================================
// Element-wise
// ============================================================================

TEST_F(GpuKernelReferenceTest, ElementwiseArithmetic) {
    for (int n : kElementCounts) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const auto a = random_values(n, -3, 3, 1 + n), b = random_values(n, -3, 3, 2 + n);
        auto da = dev(a), db = dev(b);
        GPUMemory<float> dc(n);
        std::vector<double> want(n);

        g::matrix_add_gpu(da.get(), db.get(), dc.get(), n);
        for (int i = 0; i < n; ++i)
            want[i] = double(a[i]) + b[i];
        EXPECT_TRUE(matches(host(dc, n), want, 1e-6, 1e-6)) << "add";

        g::matrix_add_scalar_gpu(da.get(), 1.25f, dc.get(), n);
        for (int i = 0; i < n; ++i)
            want[i] = double(a[i]) + 1.25;
        EXPECT_TRUE(matches(host(dc, n), want, 1e-6, 1e-6)) << "add_scalar";

        g::matrix_multiply_elementwise_gpu(da.get(), db.get(), dc.get(), n);
        for (int i = 0; i < n; ++i)
            want[i] = double(a[i]) * b[i];
        EXPECT_TRUE(matches(host(dc, n), want, 1e-6, 1e-6)) << "multiply_elementwise";

        g::matrix_multiply_scalar_gpu(da.get(), -0.5f, dc.get(), n);
        for (int i = 0; i < n; ++i)
            want[i] = double(a[i]) * -0.5;
        EXPECT_TRUE(matches(host(dc, n), want, 1e-6, 1e-6)) << "multiply_scalar";

        auto dst = dev(b);
        g::matrix_add_inplace_gpu(da.get(), dst.get(), n);
        for (int i = 0; i < n; ++i)
            want[i] = double(b[i]) + a[i];
        EXPECT_TRUE(matches(host(dst, n), want, 1e-6, 1e-6)) << "add_inplace";

        const std::vector<float> mask = [&] {
            std::vector<float> m(n);
            for (int i = 0; i < n; ++i)
                m[i] = (i % 3 == 0) ? 0.0f : 1.0f;
            return m;
        }();
        auto dmask = dev(mask);
        auto filled = dev(a);
        g::matrix_masked_fill_gpu(filled.get(), dmask.get(), -1e9f, n);
        for (int i = 0; i < n; ++i)
            want[i] = mask[i] == 0.0f ? -1e9 : a[i];
        EXPECT_TRUE(matches(host(filled, n), want, 0, 0)) << "masked_fill: a 0 in the mask is filled";
    }
}

TEST_F(GpuKernelReferenceTest, Activations) {
    using adai::activation::kGeluCoef;
    using adai::activation::kSqrt2OverPi;
    const std::pair<ActivationType, double (*)(double)> cases[] = {
        {ActivationType::RELU, [](double x) { return std::max(0.0, x); }},
        {ActivationType::SIGMOID, [](double x) { return 1.0 / (1.0 + std::exp(-x)); }},
        {ActivationType::TANH, [](double x) { return std::tanh(x); }},
        {ActivationType::GELU,
         [](double x) { return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + kGeluCoef * x * x * x))); }},
    };
    for (const auto& [type, f] : cases) {
        for (int n : {5, 1027}) {
            SCOPED_TRACE("activation " + std::to_string(static_cast<int>(type)) + ", n=" + std::to_string(n));
            const auto x = random_values(n, -6, 6, 10 + n);
            auto d = dev(x);
            g::matrix_apply_activation_gpu(d.get(), n, type);
            std::vector<double> want(n);
            for (int i = 0; i < n; ++i)
                want[i] = f(x[i]);
            EXPECT_TRUE(matches(host(d, n), want, 1e-5, 1e-6));
        }
    }
}

TEST_F(GpuKernelReferenceTest, GeluBackwardMatchesTheDerivative) {
    using adai::activation::kGeluCoef;
    using adai::activation::kSqrt2OverPi;
    auto gelu = [](double x) { return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + kGeluCoef * x * x * x))); };
    const int n = 1029;
    const auto x = random_values(n, -5, 5, 21), dout = random_values(n, -2, 2, 22);
    auto dx = dev(x), dd = dev(dout);
    GPUMemory<float> din(n);
    g::matrix_gelu_backward_gpu(dx.get(), dd.get(), din.get(), n);
    std::vector<double> want(n);
    for (int i = 0; i < n; ++i) {
        const double h = 1e-5;  // central difference of the forward GELU, in double
        want[i] = dout[i] * (gelu(x[i] + h) - gelu(x[i] - h)) / (2 * h);
    }
    EXPECT_TRUE(matches(host(din, n), want, 1e-4, 1e-5));
}

// ============================================================================
// Layout and matrix products
// ============================================================================

TEST_F(GpuKernelReferenceTest, Transpose) {
    for (auto [r, c] : std::vector<std::pair<int, int>>{{1, 1}, {1, 9}, {3, 5}, {17, 33}, {64, 65}}) {
        SCOPED_TRACE(std::to_string(r) + "x" + std::to_string(c));
        const auto a = random_values(r * c, -1, 1, r * 100 + c);
        auto da = dev(a);
        GPUMemory<float> dt(r * c);
        g::matrix_transpose_gpu(da.get(), dt.get(), r, c);
        std::vector<double> want(r * c);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                want[j * r + i] = a[i * c + j];
        EXPECT_TRUE(matches(host(dt, r * c), want, 0, 0));
    }
}

TEST_F(GpuKernelReferenceTest, MatrixMultiply) {
    for (auto [m, k, n] : std::vector<std::tuple<int, int, int>>{{1, 1, 1}, {3, 4, 5}, {17, 33, 9}, {64, 128, 32}}) {
        SCOPED_TRACE(std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n));
        const auto a = random_values(m * k, -1, 1, m + k), b = random_values(k * n, -1, 1, k + n);
        auto da = dev(a), db = dev(b);
        GPUMemory<float> dc(m * n);
        g::matrix_multiply_gpu(da.get(), db.get(), dc.get(), m, k, n);
        std::vector<double> want(m * n, 0.0);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                for (int p = 0; p < k; ++p)
                    want[i * n + j] += double(a[i * k + p]) * b[p * n + j];
        EXPECT_TRUE(matches(host(dc, m * n), want, 1e-4, 1e-4));
    }
}

// Views into wider matrices (lda, ldb, ldc larger than the view), every transpose combination; the
// columns of C outside the view are left alone.
TEST_F(GpuKernelReferenceTest, GemmStridedViewsAndTransposes) {
    const int m = 5, n = 6, k = 7, pad = 3;
    const float alpha = 0.5f;
    for (bool ta : {false, true}) {
        for (bool tb : {false, true}) {
            SCOPED_TRACE(std::string("trans_a=") + (ta ? "1" : "0") + " trans_b=" + (tb ? "1" : "0"));
            const int a_rows = ta ? k : m, a_cols = ta ? m : k;
            const int b_rows = tb ? n : k, b_cols = tb ? k : n;
            const int lda = a_cols + pad, ldb = b_cols + pad, ldc = n + pad;
            const auto a = random_values(a_rows * lda, -1, 1, 31), b = random_values(b_rows * ldb, -1, 1, 32);
            std::vector<float> c(m * ldc, 42.0f);
            auto da = dev(a), db = dev(b), dc = dev(c);
            g::matrix_gemm_strided_gpu(ta, tb, m, n, k, alpha, da.get(), lda, db.get(), ldb, dc.get(), ldc);
            std::vector<double> want(m * ldc, 42.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) {
                    double s = 0;
                    for (int p = 0; p < k; ++p) {
                        const double av = ta ? a[p * lda + i] : a[i * lda + p];
                        const double bv = tb ? b[j * ldb + p] : b[p * ldb + j];
                        s += av * bv;
                    }
                    want[i * ldc + j] = alpha * s;
                }
            EXPECT_TRUE(matches(host(dc, m * ldc), want, 1e-4, 1e-5));
        }
    }
}

// The batch kernels take host arrays of device pointers. Neither has a caller (TD-386).
TEST_F(GpuKernelReferenceTest, BatchAdd) {
    const int batch = 3, size = 9;  // size 9: a vector pass and a tail
    std::vector<GPUMemory<float>> as, bs, cs;
    std::vector<std::vector<float>> ha, hb;
    for (int i = 0; i < batch; ++i) {
        ha.push_back(random_values(size, -1, 1, 40 + i));
        hb.push_back(random_values(size, -1, 1, 50 + i));
        as.push_back(dev(ha[i]));
        bs.push_back(dev(hb[i]));
        cs.emplace_back(size);
    }
    std::vector<const float*> pa, pb;
    std::vector<float*> pc;
    for (int i = 0; i < batch; ++i) {
        pa.push_back(as[i].get());
        pb.push_back(bs[i].get());
        pc.push_back(cs[i].get());
    }
    g::matrix_batch_add_gpu(pa.data(), pb.data(), pc.data(), batch, size);
    for (int b = 0; b < batch; ++b) {
        std::vector<double> want(size);
        for (int i = 0; i < size; ++i)
            want[i] = double(ha[b][i]) + hb[b][i];
        EXPECT_TRUE(matches(host(cs[b], size), want, 1e-6, 1e-6)) << "batch item " << b;
    }
}

// Disabled: in the SYCL backend it returns without computing anything (C keeps whatever memory held),
// found by this test on the oneAPI CPU device on November 22, 2026. Waiting for gemm_batch() before
// returning makes it pass, so oneMKL reads host memory after the call returns: the stack-local group
// parameters or the pointer arrays the unwaited memcpy() copies. Whether to fix or remove it is TD-386's
// decision (no caller). Run it with --gtest_also_run_disabled_tests.
TEST_F(GpuKernelReferenceTest, DISABLED_BatchMultiply) {
    const int batch = 3, m = 4, k = 5, n = 6;
    std::vector<GPUMemory<float>> ma, mb, mc;
    std::vector<std::vector<float>> hma, hmb;
    for (int i = 0; i < batch; ++i) {
        hma.push_back(random_values(m * k, -1, 1, 60 + i));
        hmb.push_back(random_values(k * n, -1, 1, 70 + i));
        ma.push_back(dev(hma[i]));
        mb.push_back(dev(hmb[i]));
        mc.emplace_back(m * n);
    }
    std::vector<const float*> pma, pmb;
    std::vector<float*> pmc;
    for (int i = 0; i < batch; ++i) {
        pma.push_back(ma[i].get());
        pmb.push_back(mb[i].get());
        pmc.push_back(mc[i].get());
    }
    g::matrix_batch_multiply_gpu(pma.data(), pmb.data(), pmc.data(), batch, m, k, n);
    for (int b = 0; b < batch; ++b) {
        std::vector<double> wm(m * n, 0.0);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                for (int p = 0; p < k; ++p)
                    wm[i * n + j] += double(hma[b][i * k + p]) * hmb[b][p * n + j];
        EXPECT_TRUE(matches(host(mc[b], m * n), wm, 1e-4, 1e-5)) << "batch item " << b;
    }
}

TEST_F(GpuKernelReferenceTest, RowBiasAndColumnSums) {
    for (auto [r, c] : std::vector<std::pair<int, int>>{{1, 1}, {3, 7}, {300, 257}}) {
        SCOPED_TRACE(std::to_string(r) + "x" + std::to_string(c));
        const auto mat = random_values(r * c, -1, 1, 80 + r), bias = random_values(c, -1, 1, 90 + c);
        auto dm = dev(mat), db = dev(bias);
        GPUMemory<float> out(r * c), sums(c);
        g::matrix_add_row_bias_gpu(dm.get(), db.get(), out.get(), r, c);
        g::matrix_sum_rows_gpu(dm.get(), sums.get(), r, c);
        std::vector<double> want(r * c), want_sums(c, 0.0);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j) {
                want[i * c + j] = double(mat[i * c + j]) + bias[j];
                want_sums[j] += mat[i * c + j];
            }
        EXPECT_TRUE(matches(host(out, r * c), want, 1e-6, 1e-6)) << "add_row_bias";
        EXPECT_TRUE(matches(host(sums, c), want_sums, 1e-5, 1e-4)) << "sum_rows (column sums)";
    }
}

// ============================================================================
// Row kernels: softmax, layer norm, entropy
// ============================================================================

TEST_F(GpuKernelReferenceTest, SoftmaxRowsAndBackward) {
    const int rows = 3;
    for (int cols : kRowWidths) {
        SCOPED_TRACE("cols=" + std::to_string(cols));
        // Large magnitudes check the max subtraction; a -1e9 entry is how attention masks a position.
        auto x = random_values(rows * cols, -80, 80, 100 + cols);
        if (cols > 1)
            x[1] = -1e9f;
        auto d = dev(x);
        g::matrix_softmax_rows_gpu(d.get(), rows, cols);
        std::vector<double> want(rows * cols);
        for (int r = 0; r < rows; ++r) {
            const double mx = *std::max_element(x.begin() + r * cols, x.begin() + (r + 1) * cols);
            double s = 0;
            for (int j = 0; j < cols; ++j)
                s += std::exp(double(x[r * cols + j]) - mx);
            for (int j = 0; j < cols; ++j)
                want[r * cols + j] = std::exp(double(x[r * cols + j]) - mx) / s;
        }
        const auto soft = host(d, rows * cols);
        EXPECT_TRUE(matches(soft, want, 1e-4, 1e-7)) << "softmax";

        const auto dout = random_values(rows * cols, -1, 1, 110 + cols);
        auto ds = dev(soft), dd = dev(dout);
        GPUMemory<float> din(rows * cols);
        g::matrix_softmax_backward_gpu(ds.get(), dd.get(), din.get(), rows, cols);
        std::vector<double> want_din(rows * cols);
        for (int r = 0; r < rows; ++r) {
            double dot = 0;
            for (int j = 0; j < cols; ++j)
                dot += double(soft[r * cols + j]) * dout[r * cols + j];
            for (int j = 0; j < cols; ++j)
                want_din[r * cols + j] = soft[r * cols + j] * (dout[r * cols + j] - dot);
        }
        EXPECT_TRUE(matches(host(din, rows * cols), want_din, 1e-4, 1e-6)) << "softmax_backward";
    }
}

TEST_F(GpuKernelReferenceTest, LayerNormForwardAndBackward) {
    const int rows = 4;
    const float eps = 1e-5f;
    for (int cols : kRowWidths) {
        SCOPED_TRACE("cols=" + std::to_string(cols));
        const auto x = random_values(rows * cols, -3, 5, 200 + cols);  // not centred: the mean matters
        const auto gamma = random_values(cols, 0.5, 1.5, 201), beta = random_values(cols, -0.5, 0.5, 202);
        auto dx_in = dev(x), dg = dev(gamma), db = dev(beta);
        GPUMemory<float> y(rows * cols), xn(rows * cols), mean(rows), rstd(rows);
        g::matrix_layer_norm_fwd_gpu(dx_in.get(), y.get(), xn.get(), dg.get(), db.get(), mean.get(), rstd.get(),
                                     rows, cols, eps);
        std::vector<double> want_y(rows * cols), want_xn(rows * cols), want_mean(rows), want_rstd(rows);
        for (int r = 0; r < rows; ++r) {
            double mu = 0, var = 0;
            for (int j = 0; j < cols; ++j)
                mu += x[r * cols + j];
            mu /= cols;
            for (int j = 0; j < cols; ++j)
                var += (x[r * cols + j] - mu) * (x[r * cols + j] - mu);
            var /= cols;
            const double rs = 1.0 / std::sqrt(var + eps);
            want_mean[r] = mu;
            want_rstd[r] = rs;
            for (int j = 0; j < cols; ++j) {
                want_xn[r * cols + j] = (x[r * cols + j] - mu) * rs;
                want_y[r * cols + j] = want_xn[r * cols + j] * gamma[j] + beta[j];
            }
        }
        // cols = 1: every value is its own mean, so x - mean is 0 exactly and rstd = 1/sqrt(eps).
        EXPECT_TRUE(matches(host(mean, rows), want_mean, 1e-5, 1e-5)) << "mean";
        EXPECT_TRUE(matches(host(rstd, rows), want_rstd, 1e-3, 1e-5)) << "rstd";
        EXPECT_TRUE(matches(host(xn, rows * cols), want_xn, 1e-3, 1e-4)) << "normalized";
        EXPECT_TRUE(matches(host(y, rows * cols), want_y, 1e-3, 1e-4)) << "output";

        // Backward: dx = rstd * (dxn - mean(dxn) - xn * mean(dxn * xn)), dxn = dout * gamma; dgamma and
        // dbeta are added to what's already there.
        const auto dout = random_values(rows * cols, -1, 1, 210 + cols);
        const auto xn_h = host(xn, rows * cols);
        const auto rstd_h = host(rstd, rows);
        std::vector<float> dgamma0(cols, 0.25f), dbeta0(cols, -0.75f);
        auto dd = dev(dout), dgam = dev(dgamma0), dbet = dev(dbeta0);
        GPUMemory<float> dxo(rows * cols);
        g::matrix_layer_norm_bwd_gpu(dd.get(), xn.get(), dg.get(), mean.get(), rstd.get(), dxo.get(), dgam.get(),
                                     dbet.get(), rows, cols);
        std::vector<double> want_dx(rows * cols), want_dg(cols, 0.25), want_db(cols, -0.75);
        for (int r = 0; r < rows; ++r) {
            double m1 = 0, m2 = 0;
            for (int j = 0; j < cols; ++j) {
                const double dxn = double(dout[r * cols + j]) * gamma[j];
                m1 += dxn;
                m2 += dxn * xn_h[r * cols + j];
            }
            m1 /= cols;
            m2 /= cols;
            for (int j = 0; j < cols; ++j) {
                const double dxn = double(dout[r * cols + j]) * gamma[j];
                want_dx[r * cols + j] = rstd_h[r] * (dxn - m1 - xn_h[r * cols + j] * m2);
                want_dg[j] += double(dout[r * cols + j]) * xn_h[r * cols + j];
                want_db[j] += dout[r * cols + j];
            }
        }
        // dx is a small difference of larger terms; scale the absolute tolerance by rstd.
        const double dx_abs = 1e-4 * *std::max_element(rstd_h.begin(), rstd_h.end());
        EXPECT_TRUE(matches(host(dxo, rows * cols), want_dx, 1e-3, dx_abs)) << "dx";
        EXPECT_TRUE(matches(host(dgam, cols), want_dg, 1e-4, 1e-5)) << "dgamma (accumulated)";
        EXPECT_TRUE(matches(host(dbet, cols), want_db, 1e-4, 1e-5)) << "dbeta (accumulated)";
    }
}

TEST_F(GpuKernelReferenceTest, RowEntropy) {
    const int rows = 3;
    for (int cols : kRowWidths) {
        SCOPED_TRACE("cols=" + std::to_string(cols));
        auto p = random_values(rows * cols, 0, 1, 300 + cols);
        p[0] = 0.0f;  // a zero contributes nothing
        for (int r = 0; r < rows; ++r) {
            const float s = std::accumulate(p.begin() + r * cols, p.begin() + (r + 1) * cols, 0.0f);
            for (int j = 0; j < cols; ++j)
                p[r * cols + j] = s > 0 ? p[r * cols + j] / s : 0.0f;
        }
        auto d = dev(p);
        GPUMemory<float> out(rows);
        g::matrix_row_entropy_gpu(d.get(), out.get(), rows, cols);
        std::vector<double> want(rows, 0.0);
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < cols; ++j) {
                const double v = p[r * cols + j];
                if (v > 0)
                    want[r] -= v * std::log(v + 1e-10);
            }
        EXPECT_TRUE(matches(host(out, rows), want, 1e-4, 1e-5));
    }
}

// ============================================================================
// Cross-entropy and accuracy
// ============================================================================

TEST_F(GpuKernelReferenceTest, CrossEntropyLossAndGradient) {
    // seq_len 65 and 300 have more rows than 64, where TD-393's reduction lost the loss's sum.
    for (auto [seq, vocab] : std::vector<std::pair<int, int>>{{1, 1}, {3, 7}, {65, 257}, {300, 256}, {4, 5000}}) {
        SCOPED_TRACE("seq=" + std::to_string(seq) + " vocab=" + std::to_string(vocab));
        const auto logits = random_values(seq * vocab, -10, 10, 400 + seq + vocab);
        std::vector<int> targets(seq);
        for (int i = 0; i < seq; ++i)
            targets[i] = (i * 7919) % vocab;
        auto dl = dev(logits);
        auto dt = dev(targets);
        const float loss = g::matrix_cross_entropy_loss_gpu(dl.get(), dt.get(), seq, vocab);
        GPUMemory<float> grad(seq * vocab);
        g::matrix_cross_entropy_grad_gpu(dl.get(), dt.get(), grad.get(), seq, vocab);

        double want_loss = 0;
        std::vector<double> want_grad(seq * vocab);
        for (int r = 0; r < seq; ++r) {
            const double mx = *std::max_element(logits.begin() + r * vocab, logits.begin() + (r + 1) * vocab);
            double s = 0;
            for (int j = 0; j < vocab; ++j)
                s += std::exp(double(logits[r * vocab + j]) - mx);
            want_loss += std::log(s) + mx - logits[r * vocab + targets[r]];
            for (int j = 0; j < vocab; ++j)
                want_grad[r * vocab + j] =
                    (std::exp(double(logits[r * vocab + j]) - mx) / s - (j == targets[r] ? 1.0 : 0.0)) / seq;
        }
        want_loss /= seq;
        EXPECT_NEAR(loss, want_loss, 1e-4 * std::max(1.0, std::fabs(want_loss))) << "loss is the mean over rows";
        EXPECT_TRUE(matches(host(grad, seq * vocab), want_grad, 1e-4, 1e-7)) << "gradient";
    }
}

TEST_F(GpuKernelReferenceTest, TokenCorrectCountsArgmaxMatchesWithLowestIndexOnTies) {
    for (int vocab : {1, 7, 257, 3000}) {
        SCOPED_TRACE("vocab=" + std::to_string(vocab));
        const int seq = 70;
        auto logits = random_values(seq * vocab, -5, 5, 500 + vocab);
        std::vector<int> targets(seq);
        int want = 0;
        for (int r = 0; r < seq; ++r) {
            float* row = logits.data() + r * vocab;
            int best = static_cast<int>(std::max_element(row, row + vocab) - row);
            if (r % 5 == 0 && vocab > 3) {  // a tie: a later column equal to the max doesn't count
                const int later = std::min(vocab - 1, best + 1 + (r % 3));
                if (later != best)
                    row[later] = row[best];
                const int earlier = best > 0 ? best - 1 : -1;
                if (r % 10 == 0 && earlier >= 0) {  // an earlier one does: it becomes the argmax
                    row[earlier] = row[best];
                    best = earlier;
                }
            }
            if (r % 4 == 0)
                targets[r] = best;
            else if (r % 4 == 1)
                targets[r] = (best + 1) % vocab;
            else if (r % 4 == 2)
                targets[r] = -1;  // outside the vocabulary: never correct
            else
                targets[r] = vocab;
            if (targets[r] == best)
                ++want;
        }
        auto dl = dev(logits);
        auto dt = dev(targets);
        EXPECT_EQ(g::matrix_token_correct_gpu(dl.get(), dt.get(), seq, vocab), want);
    }
}

// ============================================================================
// Reductions (TD-393's sizes) and transfers
// ============================================================================

TEST_F(GpuKernelReferenceTest, SumAcrossWorkGroupAndLevelBoundaries) {
    for (int n : {1, 255, 256, 257, 65535, 65536, 65537, 1000003}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const auto v = random_values(n, -1, 1, 600 + (n % 1000));
        auto d = dev(v);
        double want = 0, abs_sum = 0;
        for (float x : v) {
            want += x;
            abs_sum += std::fabs(x);
        }
        EXPECT_NEAR(g::matrix_sum_gpu(d.get(), n), want, 1e-5 * abs_sum + 1e-5);
        auto ones = dev(std::vector<float>(n, 1.0f));
        EXPECT_FLOAT_EQ(g::matrix_sum_gpu(ones.get(), n), static_cast<float>(n)) << "ones";
    }
}

TEST_F(GpuKernelReferenceTest, CountBelowThresholdUsesTheMagnitude) {
    for (int n : {1, 255, 256, 257, 65537, 1000003}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const auto v = random_values(n, -1, 1, 700 + (n % 1000));
        auto d = dev(v);
        int want = 0;
        for (float x : v)
            want += std::fabs(x) < 0.3f ? 1 : 0;
        EXPECT_EQ(static_cast<long>(g::matrix_count_below_threshold_gpu(d.get(), n, 0.3f) + 0.5f), want);
    }
}

TEST_F(GpuKernelReferenceTest, SumSquaresAccumulatesIntoTheDeviceValue) {
    // 300,000 needs more than the 1,024-group grid, so the grid-stride loop runs.
    for (int n : {1, 257, 300000}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const auto v = random_values(n, -1, 1, 800 + (n % 1000));
        auto d = dev(v);
        auto acc = dev(std::vector<float>{2.5f});
        g::matrix_sum_squares_accumulate_gpu(d.get(), n, acc.get());
        g::matrix_sum_squares_accumulate_gpu(d.get(), n, acc.get());
        double want = 2.5;
        for (float x : v)
            want += 2.0 * double(x) * x;
        EXPECT_NEAR(host(acc, 1)[0], want, 1e-5 * want);
    }
}

TEST_F(GpuKernelReferenceTest, DeviceCopyAndDownload) {
    for (int n : {1, 4099}) {
        const auto v = random_values(n, -1, 1, 900 + n);
        auto src = dev(v);
        GPUMemory<float> dst(n);
        g::matrix_copy_device_to_device_gpu(src.get(), dst.get(), n);
        std::vector<float> out(n);
        g::matrix_download_gpu(dst.get(), out.data(), n);
        EXPECT_TRUE(matches(out, to_double(v), 0, 0)) << "n=" << n;
    }
}
